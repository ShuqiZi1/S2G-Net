"""
Main file for training GraphGPS models with neighbor sampling
"""
import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from src.models.utils import collect_outputs, get_checkpoint_path, seed_everything, define_loss_fn
from src.dataloader.pyg_reader_extended import GraphDataset
from src.models.graphgps import define_graphgps_encoder
from src.evaluation.metrics import get_metrics, get_per_node_result
from src.config.args import add_configs, init_graphgps_args
from src.utils import write_json, write_pkl, record_results
from torch_geometric.data import NeighborSampler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from src.evaluation.tracking.runtime import RuntimeTracker, measure_inference_speed
from src.evaluation.tracking.params import analyze_model_complexity
import json, time

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=True,
    mode='min'
)


def collate_graphgps(samples):
    assert len(samples) == 1, "GraphGPS validation/test loader should only have batch size = 1"
    return samples[0]

class GraphGPSModel(pl.LightningModule):
    """
    GraphGPS model with neighbor sampling
    """
    def __init__(self, config, dataset, train_loader, subgraph_loader, eval_split='test'):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.batch_loader = train_loader
        self.subgraph_loader = subgraph_loader
        self.learning_rate = config['lr']
        
        self.clip_grad = config.get('clip_grad', 1.0)  
        if self.clip_grad <= 0: 
            self.clip_grad = None

        GraphGPSEncoder = define_graphgps_encoder()
        self.net = GraphGPSEncoder(config)

        self.loss = define_loss_fn(config)

        self.collect_outputs = lambda x: collect_outputs(x, config.get('multi_gpu', False))
        self.compute_metrics = lambda x: get_metrics(x['truth'], x['pred'], config.get('verbose', False), config.get('classification', False))
        self.per_node_metrics = lambda x: get_per_node_result(x['truth'], x['pred'], 
                                                             self.dataset.idx_test, config.get('classification', False))

        self.eval_split = eval_split
        self.eval_mask = self.dataset.data.val_mask if eval_split == 'val' else self.dataset.data.test_mask
        
        self.val_preds = []
        self.val_truths = []
        self.test_preds = []
        self.test_truths = []

    def on_train_start(self):
        """
        Set seeds and move data to device at start of training
        """
        seed_everything(self.config['seed'])
        
        if self.device.type == 'cuda':
            if hasattr(self.dataset.data, 'x') and self.dataset.data.x.device.type != 'cuda':
                self.dataset.data.x = self.dataset.data.x.to(self.device)
            if hasattr(self.dataset.data, 'flat') and self.dataset.data.flat is not None and self.dataset.data.flat.device.type != 'cuda':
                self.dataset.data.flat = self.dataset.data.flat.to(self.device)
            if hasattr(self.dataset.data, 'x_hist') and self.dataset.data.x_hist is not None and self.dataset.data.x_hist.device.type != 'cuda':
                self.dataset.data.x_hist = self.dataset.data.x_hist.to(self.device)
            if hasattr(self.dataset.data, 'ts_seq') and self.dataset.data.ts_seq is not None \
                    and self.dataset.data.ts_seq.device.type != 'cuda':
                self.dataset.data.ts_seq = self.dataset.data.ts_seq.to(self.device)
            if hasattr(self.dataset.data, 'y') and self.dataset.data.y.device.type != 'cuda':
                self.dataset.data.y = self.dataset.data.y.to(self.device)
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None and self.dataset.data.edge_attr.device.type != 'cuda':
                self.dataset.data.edge_attr = self.dataset.data.edge_attr.to(self.device)

        
    def on_validation_epoch_start(self):
        """Reset prediction lists before validation epoch"""
        self.val_preds = []
        self.val_truths = []
        
    def on_test_epoch_start(self):
        """Reset prediction lists before test epoch"""
        self.test_preds = []
        self.test_truths = []

    def forward(self, x, edge_index, edge_attr=None, batch=None, flat=None, x_hist=None):
        """Forward pass of the model"""
        if batch.dim() == 2:
            batch = batch.view(-1) 
        out = self.net(x, edge_index, edge_attr, batch, flat, x_hist) 
        return out

    def training_step(self, batch, batch_idx):
        """
        Training step with neighbor sampling
        """
        batch_size, n_id, adjs = batch
        
        target_indices = n_id[:batch_size]
        
        if self.device.type == 'cuda' and target_indices.device.type == 'cpu':
            target_indices = target_indices.to(self.device)
        
        x = self.dataset.data.x[n_id].to(self.device)
        
        flat = None
        if hasattr(self.dataset.data, 'flat') and self.dataset.data.flat is not None:
            flat = self.dataset.data.flat[n_id].to(self.device)
        
        x_hist = None
        if hasattr(self.dataset.data, 'x_hist') and self.dataset.data.x_hist is not None:
             x_hist = self.dataset.data.x_hist[n_id].to(self.device)
        elif hasattr(self.dataset.data, 'ts_seq') and self.dataset.data.ts_seq is not None:
            x_hist = self.dataset.data.ts_seq[n_id].to(self.device)   # [B, T, D]
            x = x_hist[:, -1, :]
        
        target_y = self.dataset.data.y[target_indices].to(self.device)
        
        edge_index, e_id, size = adjs[0]
        edge_index = edge_index.to(self.device)
        
        edge_attr = None
        if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
            if e_id.device != self.dataset.data.edge_attr.device:
                e_id = e_id.to(self.dataset.data.edge_attr.device)
            edge_attr = self.dataset.data.edge_attr[e_id].to(self.device)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        if flat is not None and torch.isnan(flat).any():
            flat = torch.nan_to_num(flat, nan=0.0)
        
        batch_sub = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        out = self(x, edge_index, edge_attr, batch=batch_sub, flat=flat, x_hist=x_hist)
        
        out = out[:batch_size]
        
        if torch.isnan(out).any() or torch.isnan(target_y).any():
            out = torch.nan_to_num(out, nan=0.0)
            target_y = torch.nan_to_num(target_y, nan=0.0)
        
        train_loss = self.loss(out.squeeze(), target_y)
        
        if torch.isnan(train_loss):
            train_loss = torch.tensor(999999.0, device=self.device, requires_grad=True)
        
        self.log('train_loss', train_loss, prog_bar=True)
        
        log_dict = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        seq_full, flat, edge_index, N, edge_attr = batch
        y = self.dataset.data.y
        batch_vec = torch.zeros(int(N), dtype=torch.long, device=self.device)
        
        x_hist = seq_full.to(self.device)            # [N, T, D]
        x = x_hist[:, -1, :]
        
        edge_index = edge_index.to(self.device)
        if edge_attr is not None: edge_attr = edge_attr.to(self.device)
        if flat is not None:      flat = flat.to(self.device)
        truth = y.to(self.device)
        
        with torch.no_grad():
            out = self(x, edge_index, edge_attr, batch=batch_vec, flat=flat, x_hist=x_hist)
        
        val_indices = torch.where(self.eval_mask)[0]

        if len(val_indices) > 0:
            val_truth = truth[val_indices]
            val_pred = out[val_indices]
            
            self.val_truths.append(val_truth.detach().cpu())
            self.val_preds.append(val_pred.detach().cpu())
            
            val_loss = self.loss(val_pred.squeeze(), val_truth)
        else:
            val_loss = torch.tensor(999999.0, device=self.device)
        
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        seq_full, flat, edge_index, N, edge_attr = batch
        y = self.dataset.data.y
        batch_vec = torch.zeros(int(N), dtype=torch.long, device=self.device)
        
        x = self.dataset.data.x.to(self.device)      # [N, 672]
        x_hist = seq_full.to(self.device)            # [N, T, D], D=14
        
        edge_index = edge_index.to(self.device)
        if edge_attr is not None: edge_attr = edge_attr.to(self.device)
        if flat is not None:      flat = flat.to(self.device)
        truth = y.to(self.device)
        
        with torch.no_grad():
            out = self(x, edge_index, edge_attr, batch=batch_vec, flat=flat, x_hist=x_hist)
            
        test_indices = torch.where(self.eval_mask)[0]
        
        if len(test_indices) > 0:
            test_truth = truth[test_indices]
            test_pred = out[test_indices]
            
            self.test_truths.append(test_truth.detach().cpu())
            self.test_preds.append(test_pred.detach().cpu())
            
            test_loss = self.loss(test_pred.squeeze(), test_truth)
        else:
            test_loss = torch.tensor(999999.0, device=self.device)
        
        return {'test_loss': test_loss}

    def validation_epoch_end(self, outputs):
        """
        Process validation results
        """
        try:
            all_preds = torch.cat(self.val_preds, dim=0)
            all_truths = torch.cat(self.val_truths, dim=0)
            
            val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            
            collect_dict = {
                'truth': all_truths,
                'pred': all_preds,
                'val_loss': val_loss
            }
            
            log_dict = self.compute_metrics(collect_dict)
            log_dict['val_loss'] = float(val_loss)
            
            self.log('val_loss', log_dict['val_loss'], prog_bar=True)
            
            results = {'log': log_dict}
            results = {**results, **log_dict}
            
            return results
        except Exception as e:
            try:
                val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
                self.log('val_loss', float(val_loss), prog_bar=True)
                return {'val_loss': float(val_loss)}
            except:
                val_loss = 999999.0
                self.log('val_loss', val_loss, prog_bar=True)
                return {'val_loss': val_loss}

    def test_epoch_end(self, outputs):
        """
        Process test results
        """
        try:
            all_preds = torch.cat(self.test_preds, dim=0)
            all_truths = torch.cat(self.test_truths, dim=0)
            
            all_truths   = torch.expm1(all_truths)
            all_preds    = torch.expm1(all_preds)
        
            test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            
            collect_dict = {
                'truth': all_truths,
                'pred': all_preds,
                'test_loss': test_loss
            }
            
            log_dict = self.compute_metrics(collect_dict)
            log_dict = {'test_' + m: log_dict[m] for m in log_dict}
            log_dict['test_loss'] = float(test_loss)
            
            for metric_name, metric_value in log_dict.items():
                self.log(metric_name, metric_value)
            
            per_node_results = self.per_node_metrics(collect_dict)
            
            results = {'log': log_dict, 'per_node': per_node_results}
            results = {**results, **log_dict}
            
            return results
        except Exception as e:
            try:
                test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
                self.log('test_loss', float(test_loss))
                
                try:
                    if len(self.test_preds) > 0 and len(self.test_truths) > 0:
                        collect_dict = {
                            'truth': self.test_truths[-1],
                            'pred': self.test_preds[-1]
                        }
                        per_node_results = self.per_node_metrics(collect_dict)
                    else:
                        per_node_results = {}
                except Exception:
                    per_node_results = {}
                
                return {'test_loss': float(test_loss), 'per_node': per_node_results}
            except:
                return {'test_loss': 999999.0, 'per_node': {}}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """Apply gradient clipping before optimizer step"""
        if hasattr(self, 'clip_grad') and self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, 
                               weight_decay=self.config.get('l2', 0.0))
        
        if self.config.get('sch', 'plateau') == 'cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config['epochs'])
        else:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        """Return training dataloader"""
        return self.batch_loader

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(self.dataset, batch_size=1, num_workers=0, collate_fn=collate_graphgps)

    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(self.dataset, batch_size=1, num_workers=0, collate_fn=collate_graphgps)

    @staticmethod
    def load_model(log_dir, **hparams):
        """
        Load model from checkpoint
        """
        assert os.path.exists(log_dir)
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            config = yaml.load(fp, Loader=yaml.Loader)
            config.update(hparams)

        dataset, train_loader, subgraph_loader = get_data(config)

        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        args = {'hyperparameters': dict(config), 'dataset': dataset,
                'train_loader': train_loader, 'subgraph_loader': subgraph_loader}
        model = GraphGPSModel.load_from_checkpoint(checkpoint_path=str(model_path), **args)

        return model, config, dataset, train_loader, subgraph_loader


def get_data(config, us=None, vs=None):
    
    dataset = GraphDataset(config)

    if dataset.data.edge_index.size(0) > 2:
        dataset.data.edge_index = dataset.data.edge_index[:2]

    if hasattr(dataset.data, 'ts_seq') and dataset.data.ts_seq is not None:
        config['mamba_indim'] = dataset.data.ts_seq.size(-1)  # D
    elif hasattr(dataset.data, 'x_hist') and dataset.data.x_hist is not None:
        config['mamba_indim'] = dataset.data.x_hist.size(-1)
    else:
        config['mamba_indim'] = dataset.x_dim  # fallback (flattened)

    config['flat_dim'] = dataset.flat_dim if config.get('use_flat', True) else 0
    config['class_weights'] = dataset.class_weights

    batch_size = config.get('neighbor_batch_size', 512)
    num_workers = config.get('num_workers', 2)
    pin = True 
    
    sample_sizes = [config['ns_size1']] if config.get('gps_layers', 1) == 1 else [config['ns_size1'] + config['ns_size2'], config['ns_size1']]

    train_loader = NeighborSampler(
        dataset.data.edge_index,
        node_idx=dataset.data.train_mask,
        sizes=sample_sizes,
        batch_size=batch_size,        
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=False)
 
    subgraph_loader = NeighborSampler(
        dataset.data.edge_index, node_idx=None, sizes=[-1],
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=False)

    config['gps_node_dim'] = int(getattr(dataset, 'x_dim', 0) or 0)
    config['flat_dim']     = int(getattr(dataset, 'flat_dim', 0) or 0)
    config['edge_dim']     = int(getattr(dataset, 'edge_dim', 1) or 1)
    assert config['gps_node_dim'] > 0, "gps_node_dim must be > 0"

    if hasattr(dataset.data, 'ts_seq'):
        config['mamba_indim'] = dataset.data.ts_seq.size(-1)
    elif hasattr(dataset.data, 'x_hist'):
        config['mamba_indim'] = dataset.data.x_hist.size(-1)
    else:
        raise ValueError("Dataset must contain ts_seq or x_hist for time-series input.")
    
    config['gps_node_dim'] = config['mamba_indim']

    return dataset, train_loader, subgraph_loader


def main(config):
    """
    Main function for training GraphGPS models.
    """
    dataset, train_loader, subgraph_loader = get_data(config)
      
    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    model = GraphGPSModel(config, dataset, train_loader, subgraph_loader, eval_split='val')
    
    chkpt = None if config.get('load') is None else get_checkpoint_path(config['load'])

    gpus_to_use = config['gpus'] if torch.cuda.is_available() else None

    rt = RuntimeTracker()
    rt.start()

    trainer = pl.Trainer(
        amp_backend='native', 
        gradient_clip_val=1.0,
        gpus=gpus_to_use,
        logger=logger,
        callbacks=[early_stop_callback],
        max_epochs=config['epochs'],
        distributed_backend='dp' if gpus_to_use and config.get('multi_gpu', False) else None,
        precision=16 if config.get('use_amp', False) and gpus_to_use else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        resume_from_checkpoint=chkpt,
        auto_lr_find=config.get('auto_lr', False),
        auto_scale_batch_size=config.get('auto_bsz', False)
    )
    
    trainer.fit(model)

    rt.stop()
    
    train_hours = rt.get_training_hours()
    peak_vram = rt.get_peak_vram_gb()
    epoch_rate = 0.0
    if train_hours > 0 and config.get('epochs'):
        epoch_rate = config['epochs'] / (train_hours * 3600)
    
    model_info = analyze_model_complexity(model)
    total_params = model_info['total_params']
    model_size_mb = model_info['model_size_mb']
    
    try:
        device = next(model.parameters()).device
        test_loader = loaderDict['test'] if 'loaderDict' in locals() else None
        inf_ms = measure_inference_speed(model, test_loader, device) if test_loader else 0.0
    except Exception:
        inf_ms = 0.0
    
    runtime_stats = {
        "train_hours": train_hours,
        "epoch_per_sec": epoch_rate,
        "inference_ms_per_patient": inf_ms,
        "peak_vram_GB": peak_vram,
        "total_params": total_params,
        "model_size_mb": model_size_mb
    }
    
    rt_path = Path(config['log_path']) / "runtime.json"
    rt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rt_path, "w") as f:
        json.dump(runtime_stats, f, indent=2)
    print(f"📝 Saved runtime: {rt_path}")
    
    for phase in ['test', 'valid']:
        res_dir = Path(config['log_path']) / 'default'
        if config['version'] is not None:
            res_dir = res_dir / config['version']
        else:
            res_dir = res_dir / ('results_' + str(config['seed']))
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        
        if phase == 'valid':
            model.eval_split = 'val'
            model.eval_mask = dataset.data.val_mask
        else:
            model.eval_split = 'test'
            model.eval_mask = dataset.data.test_mask
                
        ret = trainer.test(model)
        
        if isinstance(ret, list):
            ret = ret[0]

        if not ret:
            ret = {f"{phase}_loss": 999999.0, "per_node": {}}
        
        per_node = {}
        if 'per_node' in ret:
            per_node = ret.pop('per_node')
            write_pkl(per_node, res_dir / f'{phase}_per_node.pkl')
        else:
            try:
                collect_dict = {
                    'truth': model.val_truths[-1] if phase == 'valid' else model.test_truths[-1],
                    'pred': model.val_preds[-1] if phase == 'valid' else model.test_preds[-1]
                }
                per_node = model.per_node_metrics(collect_dict)
                write_pkl(per_node, res_dir / f'{phase}_per_node.pkl')
            except Exception:
                pass
        
        test_results = ret
        write_json(test_results, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)
        
        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, test_results)


def main_test(hparams, path_results=None):
    """
    Main function to load and evaluate a trained model.
    """
    assert (hparams['load'] is not None) and (hparams['phase'] is not None)
    phase = hparams['phase']
    log_dir = hparams['load']

    model, config, dataset, train_loader, subgraph_loader = GraphGPSModel.load_model(
        log_dir, 
        data_dir=hparams.get('data_dir'), 
        graph_dir=hparams.get('graph_dir'),
        multi_gpu=hparams['multi_gpu'], 
        num_workers=hparams['num_workers']
    )

    gpus_to_use = hparams['gpus'] if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        gpus=gpus_to_use,
        logger=None,
        max_epochs=hparams['epochs'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    
    if phase == 'valid':
        model.eval_split = 'val'
        model.eval_mask = dataset.data.val_mask

    test_results = trainer.test(model)
    
    if isinstance(test_results, list):
        test_results = test_results[0]
    
    if not test_results:
        test_results = {f"{phase}_loss": 999999.0, "per_node": {}}
    
    per_node = {}
    if 'per_node' in test_results:
        per_node = test_results.pop('per_node')
    else:
        try:
            collect_dict = {
                'truth': model.val_truths[-1] if phase == 'valid' else model.test_truths[-1],
                'pred': model.val_preds[-1] if phase == 'valid' else model.test_preds[-1]
            }
            per_node = model.per_node_metrics(collect_dict)
        except Exception:
            pass
    
    results_path = Path(log_dir) / f'{phase}_results.json'
    write_json(test_results, results_path, sort_keys=True, verbose=True)
    write_pkl(per_node, Path(log_dir) / f'{phase}_per_node.pkl')

    if path_results is None:
        path_results = Path(log_dir).parent / 'results.csv'
    tmp = {'version': hparams['version']}
    tmp = {**tmp, **config}
    record_results(path_results, tmp, test_results)


if __name__ == '__main__':
    parser = init_graphgps_args()
    config = parser.parse_args()
    if not hasattr(config, 'g_version') or config.g_version is None or config.g_version == 'default':
        if not config.dynamic_g and not config.random_g:
            config.g_version = 'gps_k3_1_fa_gdcl_sym'
    config.model = 'graphgps'
    config.flatten = True

    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['test']:
        main_test(config)
    else:
        main(config)