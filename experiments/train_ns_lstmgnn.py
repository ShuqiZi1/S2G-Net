"""
Main file for training LSTM-GNNs
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
from src.models.pyg_lstmgnn import NsLstmGNN
from src.models.utils import get_checkpoint_path, seed_everything
from src.evaluation.metrics import get_loss_function, get_metrics, get_per_node_result
from src.config.args import add_configs, init_lstmgnn_args
from src.utils import write_json, write_pkl
from torch_geometric.data import NeighborSampler
from src.dataloader.ts_reader import LstmDataset, collate_fn
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

class Model(pl.LightningModule):
    """
    LSTM-GNN model with node sampling
    """
    def __init__(self, config, dataset, train_loader, subgraph_loader, eval_split='test', \
            chkpt=None, get_emb=False, get_logits=False, get_attn=False):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.batch_loader = train_loader
        self.subgraph_loader = subgraph_loader
        self.learning_rate = config['lr']
        self.get_emb = get_emb
        self.get_logits = get_logits
        self.get_attn = get_attn
        
        self.clip_grad = config.get('clip_grad', 1.0)  
        if self.clip_grad <= 0: 
            self.clip_grad = None

        self.net = NsLstmGNN(self.config)
        
        if chkpt is not None:
            self.net.load_state_dict(chkpt['state_dict'], strict=False)
                
        self.loss = get_loss_function(config['task'], config['class_weights'])

        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda truth, pred : get_metrics(truth, pred, config['verbose'], config['classification'])
        self.per_node_metrics = lambda truth, pred : get_per_node_result(truth, pred, self.dataset.idx_test, config['classification'])

        self.eval_split = eval_split
        self.eval_mask = self.dataset.data.val_mask if eval_split == 'val' else self.dataset.data.test_mask

        entire_set = LstmDataset(config)
        collate = lambda x: collate_fn(x, config['task'])
        self.ts_loader = DataLoader(entire_set, collate_fn=collate, \
                batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
        self.lg_alpha = config['lg_alpha']
        
        self.val_preds = []
        self.val_preds_lstm = []
        self.val_truths = []
        self.test_preds = []
        self.test_preds_lstm = []
        self.test_truths = []
        
    def on_fit_start(self):
        if self.device.type == 'cuda':
            if hasattr(self.dataset.data, 'x') and self.dataset.data.x.device.type != 'cuda':
                self.dataset.data.x = self.dataset.data.x.to(self.device)
            if hasattr(self.dataset.data, 'flat') and self.dataset.data.flat.device.type != 'cuda':
                self.dataset.data.flat = self.dataset.data.flat.to(self.device)
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None and self.dataset.data.edge_attr.device.type != 'cuda':
                self.dataset.data.edge_attr = self.dataset.data.edge_attr.to(self.device)
            if hasattr(self.dataset.data, 'y') and self.dataset.data.y.device.type != 'cuda':
                self.dataset.data.y = self.dataset.data.y.to(self.device)

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_preds_lstm = []
        self.val_truths = []
    
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_preds_lstm = []
        self.test_truths = []

    def on_train_start(self):
        seed_everything(self.config['seed'])
    
    def forward(self, x, flat, adjs, batch_size, edge_weight):
        out, out_lstm = self.net(x, flat, adjs, batch_size, edge_weight)
        return out, out_lstm

    def add_losses(self, out, out_lstm, bsz_y):
        if torch.isnan(out).any() or torch.isnan(out_lstm).any() or torch.isnan(bsz_y).any():
            out = torch.nan_to_num(out, nan=0.0)
            out_lstm = torch.nan_to_num(out_lstm, nan=0.0)
            bsz_y = torch.nan_to_num(bsz_y, nan=0.0)
            
        train_loss = self.loss(out.squeeze(), bsz_y)
        train_loss_lstm = self.loss(out_lstm.squeeze(), bsz_y)
        tot_loss = train_loss + train_loss_lstm * self.lg_alpha
        
        if torch.isnan(train_loss) or torch.isnan(train_loss_lstm) or torch.isnan(tot_loss):
            if torch.isnan(train_loss):
                train_loss = torch.tensor(999999.0, device=self.device, requires_grad=True)
            if torch.isnan(train_loss_lstm):
                train_loss_lstm = torch.tensor(999999.0, device=self.device, requires_grad=True)
            if torch.isnan(tot_loss):
                tot_loss = torch.tensor(999999.0, device=self.device, requires_grad=True)
        
        return train_loss, train_loss_lstm, tot_loss

    def training_step(self, batch, batch_idx):
        batch_size, n_id, adjs = batch
        
        if self.device.type == 'cuda' and n_id.device.type == 'cpu':
            n_id = n_id.to(self.device)
        
        in_x = self.dataset.data.x[n_id]
        in_flat = self.dataset.data.flat[n_id[:batch_size]]
        
        edge_weight = None
        if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
            edge_weight = self.dataset.data.edge_attr
        
        bsz_y = self.dataset.data.y[n_id[:batch_size]]
        
        if torch.isnan(in_x).any() or torch.isnan(in_flat).any():
            in_x = torch.nan_to_num(in_x, nan=0.0)
            in_flat = torch.nan_to_num(in_flat, nan=0.0)
        
        out, out_lstm = self(in_x, in_flat, adjs, batch_size, edge_weight)
        train_loss, train_loss_lstm, tot_loss = self.add_losses(out, out_lstm, bsz_y)
        
        log_dict = {'train_loss': train_loss, 'train_loss_lstm': train_loss_lstm, 'train_tot_loss': tot_loss}
        
        for key, value in log_dict.items():
            self.log(key, value, prog_bar=True if key == 'train_tot_loss' else False)
            
        return {'loss': tot_loss, 'log': log_dict}

    def validation_step(self, batch, batch_idx):
        """
        Validation step with error handling
        """
        try:
            x = self.dataset.data.x
            flat = self.dataset.data.flat
            
            edge_weight = None
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                edge_weight = self.dataset.data.edge_attr
            
            with torch.no_grad():
                inference_result = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device)
            
            if isinstance(inference_result, tuple):
                out, out_lstm = inference_result
            else:
                out, out_lstm = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device)
            
            val_indices = torch.where(self.dataset.data.val_mask)[0]
            
            if len(val_indices) > 0:
                truth = self.dataset.data.y[self.dataset.data.val_mask]
                masked_out = out[self.dataset.data.val_mask]
                masked_out_lstm = out_lstm[self.dataset.data.val_mask]
                
                val_loss, val_loss_lstm, val_tot_loss = self.add_losses(masked_out, masked_out_lstm, truth)
                
                self.val_preds.append(masked_out.detach().cpu())
                self.val_preds_lstm.append(masked_out_lstm.detach().cpu())
                self.val_truths.append(truth.detach().cpu())
            else:
                val_loss = torch.tensor(999999.0, device=self.device)
                val_loss_lstm = torch.tensor(999999.0, device=self.device)
                val_tot_loss = torch.tensor(999999.0, device=self.device)
            
            return {
                'val_loss': val_loss,
                'val_loss_lstm': val_loss_lstm,
                'val_tot_loss': val_tot_loss
            }
        except Exception as e:
            return {
                'val_loss': torch.tensor(999999.0, device=self.device),
                'val_loss_lstm': torch.tensor(999999.0, device=self.device),
                'val_tot_loss': torch.tensor(999999.0, device=self.device)
            }

    def test_step(self, batch, batch_idx):
        if self.get_emb or self.get_logits:
            try:
                x = self.dataset.data.x
                flat = self.dataset.data.flat
                
                edge_weight = None
                if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                    edge_weight = self.dataset.data.edge_attr
                
                if self.get_attn:
                    edge_index = self.dataset.data.edge_index
                    with torch.no_grad():
                        out, out_lstm, edge_index_w_self_loops, all_edge_attn = self.net.inference_w_attn(
                            x, flat, edge_weight, edge_index, self.ts_loader, self.subgraph_loader, self.device)
                    return {'hid': out, 'edge_index': edge_index_w_self_loops, 
                            'edge_attn_1': all_edge_attn[0], 'edge_attn_2': all_edge_attn[1], 'per_node': {}}
                else:
                    with torch.no_grad():
                        if self.get_logits:
                            out, out_lstm = self.net.inference(x, flat, edge_weight, 
                                                          self.ts_loader, self.subgraph_loader, self.device)
                        else:
                            out, out_lstm = self.net.inference(x, flat, edge_weight, 
                                                          self.ts_loader, self.subgraph_loader, self.device, get_emb=True)
                    return {'hid': out, 'per_node': {}}
            except Exception as e:
                return {'hid': None, 'per_node': {}}
        else:
            try:
                x = self.dataset.data.x
                flat = self.dataset.data.flat
                
                edge_weight = None
                if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                    edge_weight = self.dataset.data.edge_attr
                
                with torch.no_grad():
                    inference_result = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device)
                
                if isinstance(inference_result, tuple):
                    out, out_lstm = inference_result
                else:
                    out, out_lstm = self.net.inference(x, flat, edge_weight, self.ts_loader, self.subgraph_loader, self.device)
                
                test_indices = torch.where(self.eval_mask)[0]
                
                if len(test_indices) > 0:
                    truth = self.dataset.data.y[self.eval_mask]
                    masked_out = out[self.eval_mask]
                    masked_out_lstm = out_lstm[self.eval_mask]
                    
                    test_loss, test_loss_lstm, test_tot_loss = self.add_losses(masked_out, masked_out_lstm, truth)
                    
                    self.test_preds.append(masked_out.detach().cpu())
                    self.test_preds_lstm.append(masked_out_lstm.detach().cpu())
                    self.test_truths.append(truth.detach().cpu())
                else:
                    test_loss = torch.tensor(999999.0, device=self.device)
                    test_loss_lstm = torch.tensor(999999.0, device=self.device)
                    test_tot_loss = torch.tensor(999999.0, device=self.device)
                
                return {
                    'test_loss': test_loss,
                    'test_loss_lstm': test_loss_lstm,
                    'test_tot_loss': test_tot_loss
                }
            except Exception as e:
                return {
                    'test_loss': torch.tensor(999999.0, device=self.device),
                    'test_loss_lstm': torch.tensor(999999.0, device=self.device),
                    'test_tot_loss': torch.tensor(999999.0, device=self.device)
                }

    def validation_epoch_end(self, outputs):
        try:
            if len(self.val_preds) == 0 or len(self.val_preds_lstm) == 0 or len(self.val_truths) == 0:
                self.log('val_loss', 999999.0, prog_bar=True)
                self.log('val_loss_lstm', 999999.0)
                self.log('val_tot_loss', 999999.0)
                return {'val_loss': 999999.0, 'per_node': {}}
            
            all_preds = torch.cat(self.val_preds, dim=0)
            all_preds_lstm = torch.cat(self.val_preds_lstm, dim=0)
            all_truths = torch.cat(self.val_truths, dim=0)
                
            try:
                val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
                val_loss_lstm = torch.stack([x['val_loss_lstm'] for x in outputs]).mean()
                val_tot_loss = torch.stack([x['val_tot_loss'] for x in outputs]).mean()
            except:
                val_loss = torch.tensor(999999.0, device=self.device)
                val_loss_lstm = torch.tensor(999999.0, device=self.device)
                val_tot_loss = torch.tensor(999999.0, device=self.device)
            
            try:
                metrics_gnn = self.compute_metrics(all_truths, all_preds)
                metrics_lstm = self.compute_metrics(all_truths, all_preds_lstm)
                
                metrics_lstm = {n + '_lstm': metrics_lstm[n] for n in metrics_lstm}
            except Exception as e:
                metrics_gnn = {}
                metrics_lstm = {}
            
            all_metrics = {
                **metrics_gnn, 
                **metrics_lstm,
                'val_loss_gnn': float(val_loss),
                'val_loss_lstm': float(val_loss_lstm),
                'val_tot_loss': float(val_tot_loss)
            }
            
            for key, value in all_metrics.items():
                self.log(key, value, prog_bar=True if key == 'val_tot_loss' else False)
            
            self.log('val_loss', float(val_tot_loss), prog_bar=True)
            
            try:
                per_node_gnn = self.per_node_metrics(all_truths, all_preds)
                per_node_lstm = self.per_node_metrics(all_truths, all_preds_lstm)
                per_node_lstm = {n + '_lstm': per_node_lstm[n] for n in per_node_lstm}
                per_node = {**per_node_gnn, **per_node_lstm}
            except Exception as e:
                per_node = {}
            
            return {**all_metrics, 'per_node': per_node}
        except Exception as e:
            return {'val_loss': 999999.0, 'per_node': {}}

    def test_epoch_end(self, outputs):
        if self.get_emb or self.get_logits or self.get_attn:
            try:
                collected_outputs = self.collect_outputs(outputs)
                if 'per_node' not in collected_outputs:
                    collected_outputs['per_node'] = {}
                return collected_outputs
            except Exception as e:
                return {'hid': None, 'per_node': {}}
        else:
            try:
                if len(self.test_preds) == 0 or len(self.test_preds_lstm) == 0 or len(self.test_truths) == 0:
                    return {'test_loss': 999999.0, 'per_node': {}}
                
                all_preds = torch.cat(self.test_preds, dim=0)
                all_preds_lstm = torch.cat(self.test_preds_lstm, dim=0)
                all_truths = torch.cat(self.test_truths, dim=0)

                all_truths   = torch.expm1(all_truths)
                all_preds    = torch.expm1(all_preds)
                all_preds_lstm = torch.expm1(all_preds_lstm)
            
                try:
                    test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
                    test_loss_lstm = torch.stack([x['test_loss_lstm'] for x in outputs]).mean()
                    test_tot_loss = torch.stack([x['test_tot_loss'] for x in outputs]).mean()
                except:
                    test_loss = torch.tensor(999999.0, device=self.device)
                    test_loss_lstm = torch.tensor(999999.0, device=self.device)
                    test_tot_loss = torch.tensor(999999.0, device=self.device)
                
                try:
                    metrics_gnn = self.compute_metrics(all_truths, all_preds)
                    metrics_lstm = self.compute_metrics(all_truths, all_preds_lstm)
                    
                    metrics_lstm = {n + '_lstm': metrics_lstm[n] for n in metrics_lstm}
                except Exception as e:
                    metrics_gnn = {}
                    metrics_lstm = {}
                
                test_metrics = {}
                for k, v in {**metrics_gnn, **metrics_lstm}.items():
                    test_metrics['test_' + k] = v
                
                losses = {
                    'test_loss_gnn': float(test_loss),
                    'test_loss_lstm': float(test_loss_lstm),
                    'test_tot_loss': float(test_tot_loss)
                }
                
                all_metrics = {**test_metrics, **losses}
                
                for key, value in all_metrics.items():
                    self.log(key, value, prog_bar=True if key == 'test_tot_loss' else False)
                
                try:
                    per_node_gnn = self.per_node_metrics(all_truths, all_preds)
                    per_node_lstm = self.per_node_metrics(all_truths, all_preds_lstm)
                    per_node_lstm = {n + '_lstm': per_node_lstm[n] for n in per_node_lstm}
                    per_node = {**per_node_gnn, **per_node_lstm}
                except Exception as e:
                    per_node = {}
                
                results = {'per_node': per_node, **all_metrics}
                
                return results
            except Exception as e:
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
                "monitor": "r2", 
                "frequency": 1
            }
        }
        
    def train_dataloader(self):
        return self.batch_loader

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)

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
        model = Model.load_from_checkpoint(checkpoint_path=str(model_path), **args)

        return model, config, dataset, train_loader, subgraph_loader


def get_data(config, us=None, vs=None):
    """
    produce dataloaders for training and validating
    """
    dataset = GraphDataset(config, us, vs)
    
    config['lstm_indim'] = dataset.x_dim
    config['num_flat_feats'] = dataset.flat_dim
    config['class_weights'] = dataset.class_weights
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    sample_sizes = [config['ns_size1']] if config['gnn_name'] == 'mpnn' else [config['ns_size1'] + config['ns_size2'], config['ns_size1']]

    train_loader = NeighborSampler(dataset.data.edge_index, node_idx=dataset.data.train_mask,
                               sizes=sample_sizes, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    subgraph_loader = NeighborSampler(dataset.data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    return dataset, train_loader, subgraph_loader


def main(config):
    """
    Main function for training LSTM-GNN.
    After training, results on validation & test sets are recorded in the specified log_path.
    """
    dataset, train_loader, subgraph_loader = get_data(config)

    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    chkpt = None
    if config['load'] is not None:
        chkpt_path = get_checkpoint_path(config['load'])
        chkpt = torch.load(chkpt_path, map_location=torch.device('cpu'))

    model = Model(config, dataset, train_loader, subgraph_loader, chkpt=chkpt)
    
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
        
        test_results = ret
        
        write_json(test_results, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)
        
        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, test_results)


if __name__ == '__main__':
    parser = init_lstmgnn_args()
    parser.add_argument('--fp_emb', action='store_true', help='forward pass to get embeddings')
    parser.add_argument('--fp_logits', action='store_true', help='forward pass to get logits')
    parser.add_argument('--fp_attn', action='store_true', help='forward pass to get attention weights (for GAT)')
    config = parser.parse_args()
    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['fp_emb'] or config['fp_logits'] or config['fp_attn']:
        main_forward_pass(config)
    
    if config['test']:
        main_test(config)
    
    else:
        main(config)