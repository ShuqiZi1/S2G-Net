"""
Main file for training GNNs with node sampling
"""
import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from src.models.utils import collect_outputs
from src.dataloader.pyg_reader import GraphDataset
from src.models.pyg_ns import NsGNN
from src.models.utils import get_checkpoint_path, seed_everything
from src.evaluation.metrics import get_loss_function, get_metrics, get_per_node_result
from src.config.args import add_configs, init_lstmgnn_args
from src.utils import write_json, write_pkl
from torch_geometric.data import NeighborSampler
from src.utils import record_results
from pytorch_lightning.callbacks import EarlyStopping
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
class NsGnnModel(pl.LightningModule):
    """
    Basic GNN model with neighbor sampling
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

        self.net = NsGNN(config)

        self.loss = get_loss_function(config['task'], config['class_weights'])

        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda x: get_metrics(x['truth'], x['pred'], config['verbose'], config['classification'])
        self.per_node_metrics = lambda x: get_per_node_result(x['truth'], x['pred'], self.dataset.idx_test, config['classification'])

        self.eval_split = eval_split
        self.eval_mask = self.dataset.data.val_mask if eval_split == 'val' else self.dataset.data.test_mask
        
        self.val_preds = []
        self.val_truths = []
        self.test_preds = []
        self.test_truths = []

    def on_train_start(self):
        seed_everything(self.config['seed'])
        
        if self.device.type == 'cuda':
            if hasattr(self.dataset.data, 'x') and self.dataset.data.x.device.type != 'cuda':
                self.dataset.data.x = self.dataset.data.x.to(self.device)
            if hasattr(self.dataset.data, 'flat') and self.dataset.data.flat.device.type != 'cuda':
                self.dataset.data.flat = self.dataset.data.flat.to(self.device)
            if hasattr(self.dataset.data, 'y') and self.dataset.data.y.device.type != 'cuda':
                self.dataset.data.y = self.dataset.data.y.to(self.device)
        
    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_truths = []
        
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_truths = []

    def forward(self, x, flat, adjs, edge_weight):
        out = self.net(x, flat, adjs, edge_weight)
        return out

    def training_step(self, batch, batch_idx):
        batch_size, n_id, adjs = batch
        bsz_nids = n_id[:batch_size]
        
        if self.device.type == 'cuda' and bsz_nids.device.type == 'cpu':
            bsz_nids = bsz_nids.to(self.device)
        
        x = self.dataset.data.x[n_id].to(self.device)
        flat = self.dataset.data.flat[bsz_nids].to(self.device)
        bsz_y = self.dataset.data.y[bsz_nids].to(self.device)
        
        edge_weight = None
        if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
            edge_weight = self.dataset.data.edge_attr.to(self.device)
        
        if torch.isnan(x).any() or torch.isnan(flat).any():
            x = torch.nan_to_num(x, nan=0.0)
            flat = torch.nan_to_num(flat, nan=0.0)
        
        adjs_on_device = []
        adjs = [adjs] if isinstance(adjs, tuple) else adjs
        for adj in adjs:
            edge_index, e_id, size = adj
            edge_index = edge_index.to(self.device)
            e_id = e_id.to(self.device)
            adjs_on_device.append((edge_index, e_id, size))
        
        out = self(x, flat, adjs_on_device, edge_weight)
        
        if torch.isnan(out).any() or torch.isnan(bsz_y).any():
            out = torch.nan_to_num(out, nan=0.0)
            bsz_y = torch.nan_to_num(bsz_y, nan=0.0)
        
        train_loss = self.loss(out.squeeze(), bsz_y)
        
        if torch.isnan(train_loss):
            train_loss = torch.tensor(999999.0, device=self.device, requires_grad=True)
            
        log_dict = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        """
        Validation step that stores predictions
        """
        try:
            x = self.dataset.data.x.to(self.device)
            flat = self.dataset.data.flat.to(self.device)
            
            truth = self.dataset.data.y
            
            edge_weight = None
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                edge_weight = self.dataset.data.edge_attr.to(self.device)
            
            with torch.no_grad():
                inference_result = self.net.inference(x, flat, self.subgraph_loader, self.device, edge_weight)
            
            if isinstance(inference_result, tuple):
                out = inference_result[0]
            else:
                out = inference_result
            
            val_mask = self.dataset.data.val_mask.to(self.device)
            if val_mask.sum() == 0:
                self.log('val_loss', 999999.0, prog_bar=True)
                return {'val_loss': torch.tensor(999999.0, device=self.device)}
    
            truth     = self.dataset.data.y.to(self.device)   # move once
            val_pred  = out[val_mask]
            val_truth = truth[val_mask]                       # safe index

            val_loss = self.loss(val_pred.squeeze(), val_truth)

            self.val_preds.append(val_pred.detach().cpu())
            self.val_truths.append(val_truth.detach().cpu())
    
            return {'val_loss': val_loss}
        except Exception as e:
            # log the root cause instead of hiding it
            self.print(f"[validation_step] Exception: {e}")
            raise

    def test_step(self, batch, batch_idx):
        """
        Test step that stores predictions
        """
        try:
            x = self.dataset.data.x.to(self.device)
            flat = self.dataset.data.flat.to(self.device)
            
            truth = self.dataset.data.y
            
            edge_weight = None
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                edge_weight = self.dataset.data.edge_attr.to(self.device)
            
            with torch.no_grad():
                inference_result = self.net.inference(x, flat, self.subgraph_loader, self.device, edge_weight)
            
            if isinstance(inference_result, tuple):
                out = inference_result[0]
            else:
                out = inference_result
            
            eval_mask = self.eval_mask.to(self.device)
            if eval_mask.sum() == 0:
                self.log('test_loss', 999999.0)
                return {'test_loss': torch.tensor(999999.0, device=self.device)}
    
            truth      = self.dataset.data.y.to(self.device)
            test_pred  = out[eval_mask]
            test_truth = truth[eval_mask]
    
            test_loss = self.loss(test_pred.squeeze(), test_truth)
    
            self.test_preds.append(test_pred.detach().cpu())
            self.test_truths.append(test_truth.detach().cpu())
    
            return {'test_loss': test_loss}
        except Exception as e:
            return {'test_loss': torch.tensor(999999.0, device=self.device)}

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
                except Exception as inner_e:
                    per_node_results = {}
                
                return {'test_loss': float(test_loss), 'per_node': per_node_results}
            except:
                return {'test_loss': 999999.0, 'per_node': {}}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if hasattr(self, 'clip_grad') and self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.config['l2'])
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
        return self.batch_loader

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0)

    @staticmethod
    def load_model(log_dir, **hparams):
        """
        :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
        :param config: list of named arguments, used to update the model hyperparameters
        """
        assert os.path.exists(log_dir)
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            config = yaml.load(fp, Loader=yaml.Loader)
            config.update(hparams)

        dataset, train_loader, subgraph_loader = get_data(config)

        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        args = {'hyperparameters': dict(config), 'dataset': dataset, \
            'train_loader': train_loader, 'subgraph_loader': subgraph_loader}
        model = NsGnnModel.load_from_checkpoint(checkpoint_path=str(model_path), **args)

        return model, config, dataset, train_loader, subgraph_loader


def get_data(config, us=None, vs=None):
    """
    produce dataloaders for training and validating
    """
    dataset = GraphDataset(config, us, vs)
    config['gnn_indim'] = dataset.x_dim
    config['num_flat_feats'] = dataset.flat_dim
    config['class_weights'] = dataset.class_weights
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    if config['gnn_name'] != 'mpnn':
        sample_sizes = [config['ns_size1'] + config['ns_size2'], config['ns_size1']]
    else:
        sample_sizes = [config['ns_size1']]
    train_loader = NeighborSampler(dataset.data.edge_index, node_idx=dataset.data.train_mask,
                               sizes=sample_sizes, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    subgraph_loader = NeighborSampler(dataset.data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    return dataset, train_loader, subgraph_loader


def main(config):
    """
    Main function for training GNNs with node sampling.
    After training, results on validation & test sets are recorded in the specified log_path.
    """
    dataset, train_loader, subgraph_loader = get_data(config)

    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    model = NsGnnModel(config, dataset, train_loader, subgraph_loader)
    chkpt = None if config['load'] is None else get_checkpoint_path(config['load'])

    gpus_to_use = config['gpus'] if torch.cuda.is_available() else None

    rt = RuntimeTracker()
    rt.start()

    trainer = pl.Trainer(
        gpus=gpus_to_use,
        logger=logger,
        max_epochs=config['epochs'],
        callbacks=[early_stop_callback],
        distributed_backend='dp' if gpus_to_use else None,
        precision=16 if config['use_amp'] and gpus_to_use else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        resume_from_checkpoint=chkpt,
        auto_lr_find=config['auto_lr'],
        auto_scale_batch_size=config['auto_bsz']
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
            except Exception as e:
                pass
        
        test_results = ret
        
        write_json(test_results, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)
        
        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, test_results)


def main_test(hparams, path_results=None):
    """
    main function to load and evaluate a trained model. 
    """
    assert (hparams['load'] is not None) and (hparams['phase'] is not None)
    phase = hparams['phase']
    log_dir = hparams['load']

    model, config, dataset, train_loader, subgraph_loader = NsGnnModel.load_model(log_dir, \
        data_dir=hparams.get('data_dir'), graph_dir=hparams.get('graph_dir'), \
        multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])

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
        except Exception as e:
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
    parser = init_lstmgnn_args()
    config = parser.parse_args()
    config.model = 'gnn'
    config.flatten = True

    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['test']:
        main_test(config)
    else:
        main(config)