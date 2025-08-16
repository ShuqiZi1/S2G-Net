"""
Training script for MambaGraphGPS models with enhanced features
"""
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import json
from pytorch_lightning import loggers
from torch_geometric.loader import DataLoader
from src.models.utils import collect_outputs, seed_everything
from src.dataloader.pyg_reader_extended import GraphDataset
from src.models.pyg_mamba_gps import MambaGraphGPS
from src.models.utils import get_checkpoint_path
from src.evaluation.metrics import get_metrics, get_per_node_result
from src.config.args import add_configs, init_mamba_gps_args
from src.utils import write_json, write_pkl, record_results
from torch_geometric.data import NeighborSampler
from src.dataloader.ts_reader import create_mamba_dataloader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from src.evaluation.explain.run_explain import run_interpretability_experiments
from src.evaluation.ablation.ablate import run_ablation_experiment, run_all_ablations
from src.evaluation.tracking.runtime import RuntimeTracker, measure_inference_speed
from src.evaluation.tracking.params import analyze_model_complexity
from src.evaluation.ablation.ablate import EdgeDropWrapper
import json, time

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=True,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
ckpt = ModelCheckpoint(monitor='val_r2', mode='max', save_top_k=1)

class MambaGPSModel(pl.LightningModule):
    def __init__(self, config, dataset, train_loader, subgraph_loader, eval_split='test',
                 chkpt=None, get_emb=False, get_logits=False, get_attn=False):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.batch_loader = train_loader
        self.subgraph_loader = subgraph_loader
        self.learning_rate = config.get('lr', 5e-4)
        self.get_emb = get_emb
        self.get_logits = get_logits
        self.get_attn = get_attn
        
        self.clip_grad = config.get('clip_grad', 1.0)  
        if self.clip_grad <= 0: 
            self.clip_grad = None

        self.net = MambaGraphGPS(self.config)

        drop_p = self.config.get('drop_edges', 0.0)
        if drop_p > 0.0:
            self.net = EdgeDropWrapper(
                self.net,
                drop_rate=drop_p,
                seed=self.config.get('seed', 42)  
            )

        if chkpt is not None:
            incompat = self.load_state_dict(chkpt["state_dict"], strict=True)
    
        def loss_fn(pred, y):
            huber = F.huber_loss(pred.clamp_min(1e-6),
                                  y, delta=0.5, reduction='none')
            weight = 1.0 + 2.5 * (y > 2.7).float()
            return (huber * weight).mean()
        self.loss = loss_fn

        self.lg_alpha = float(config.get('lg_alpha', 0.5))

        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda truth, pred: get_metrics(truth, pred, config['verbose'], config['classification'])
        self.per_node_metrics = lambda truth, pred: get_per_node_result(truth, pred, self.dataset.idx_test, config['classification'])

        self.eval_split = eval_split
        self.eval_mask = self.dataset.data.val_mask if eval_split == 'val' else self.dataset.data.test_mask

        self.ts_loader = create_mamba_dataloader(config)
        
        self.val_preds = []
        self.val_preds_ts = []
        self.val_truths = []
        self.test_preds = []
        self.test_preds_ts = []
        self.test_truths = []
        
        self.lambda_history = {
            'epoch': [],
            'train_loss': [],
            'train_r2': [],
            'val_loss': [],
            'val_r2': [],
            'lambda_ts': [],
            'lambda_gps': []
        }
        self.save_dir = Path(config['results_dir']) 
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._train_loss_accumulator = []
        self._train_preds_accumulator = []
        self._train_truths_accumulator = []
        
    def on_fit_start(self):
        if self.device.type == 'cuda':
            if hasattr(self.dataset.data, 'x') and self.dataset.data.x.device.type != 'cuda':
                self.dataset.data.x = self.dataset.data.x.to(self.device)
            if hasattr(self.dataset.data, 'flat') and self.dataset.data.flat.device.type != 'cuda':
                self.dataset.data.flat = self.dataset.data.flat.to(self.device)
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                self.dataset.data.edge_attr = self.dataset.data.edge_attr.to(self.device)
            if hasattr(self.dataset.data, 'y') and self.dataset.data.y.device.type != 'cuda':
                self.dataset.data.y = self.dataset.data.y.to(self.device)

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_preds_ts = []
        self.val_truths = []
    
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_preds_ts = []
        self.test_truths = []

    def on_train_start(self):
        seed_everything(self.config['seed'])
    
    def forward(self, x, flat, adjs, batch_size, edge_weight):
        out, out_ts = self.net(x, flat, adjs, batch_size, edge_weight)
        return out, out_ts

    def add_losses(self, out, out_ts, bsz_y):
        if out.dim() > 1 and out.size(1) == 1:
            out = out.squeeze(1)
        if out_ts.dim() > 1 and out_ts.size(1) == 1:
            out_ts = out_ts.squeeze(1)
        if bsz_y.dim() > 1 and bsz_y.size(1) == 1:
            bsz_y = bsz_y.squeeze(1)
        
        gps_loss = self.loss(out, bsz_y)
        ts_loss = self.loss(out_ts, bsz_y)
        
        tot_loss = (1 - self.lg_alpha) * gps_loss + self.lg_alpha * ts_loss
        
        return gps_loss, ts_loss, tot_loss
    
    def training_step(self, batch, batch_idx):
        batch_size, n_id, adjs = batch
        
        if self.device.type == 'cuda' and n_id.device.type == 'cpu':
            n_id = n_id.to(self.device)
        
        in_x = self.dataset.data.x[n_id]
        in_flat = self.dataset.data.flat[n_id]
        
        edge_weight = None
        if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
            edge_weight = self.dataset.data.edge_attr
        
        bsz_y = self.dataset.data.y[n_id[:batch_size]]
        
        out, out_ts = self(in_x, in_flat, adjs, batch_size, edge_weight)
        
        train_loss_gps, train_loss_ts, tot_loss = self.add_losses(out, out_ts, bsz_y)
        
        self.log('train_loss', train_loss_gps)
        self.log('train_loss_ts', train_loss_ts)
        self.log('train_tot_loss', tot_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        self._train_loss_accumulator.append(tot_loss.item())
        
        self._train_preds_accumulator.append(out.detach().cpu())
        self._train_truths_accumulator.append(bsz_y.detach().cpu())
        
        if hasattr(self.net, 'graphgps_encoder'):
            lambda_ts = self.net.graphgps_encoder.lambda_ts.item()
            lambda_gps = self.net.graphgps_encoder.lambda_gps.item()
            self.log('lambda_ts', lambda_ts)
            self.log('lambda_gps', lambda_gps)
            
        return {'loss': tot_loss}

    def validation_step(self, batch, batch_idx):
        x = self.dataset.data.x
        flat = self.dataset.data.flat
        
        edge_weight = None
        if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
            edge_weight = self.dataset.data.edge_attr
        
        with torch.no_grad():
            out, out_ts = self.net.inference(x, flat, edge_weight, self.ts_loader, 
                                           self.subgraph_loader, self.device)
        
        val_indices = torch.where(self.dataset.data.val_mask)[0]
        
        if len(val_indices) > 0:
            truth = self.dataset.data.y[self.dataset.data.val_mask]
            masked_out = out[self.dataset.data.val_mask]
            masked_out_ts = out_ts[self.dataset.data.val_mask]
            
            val_loss, val_loss_ts, val_tot_loss = self.add_losses(masked_out, masked_out_ts, truth)
            
            self.val_preds.append(masked_out.detach().cpu())
            self.val_preds_ts.append(masked_out_ts.detach().cpu())
            self.val_truths.append(truth.detach().cpu())
            
            self.log('val_loss', val_tot_loss, prog_bar=True, on_step=False, on_epoch=True)
            
            return {
                'val_loss': val_loss,
                'val_loss_ts': val_loss_ts,
                'val_tot_loss': val_tot_loss
            }
        else:
            default_loss = torch.tensor(999999.0, device=self.device)
            self.log('val_loss', default_loss, prog_bar=True, on_step=False, on_epoch=True)
            return {'val_loss': default_loss}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.clip_grad is not None and self.clip_grad > 0:
            gps_params = list(self.net.graphgps_encoder.parameters())
            mamba_params = list(self.net.mamba_encoder.parameters())
            
            torch.nn.utils.clip_grad_norm_(gps_params, self.clip_grad)
            torch.nn.utils.clip_grad_norm_(mamba_params, self.clip_grad) 
            
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)

    def test_step(self, batch, batch_idx):
        if self.get_emb or self.get_logits or self.get_attn:
            x = self.dataset.data.x
            flat = self.dataset.data.flat
            
            edge_weight = None
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                edge_weight = self.dataset.data.edge_attr
            
            if self.get_attn:
                edge_index = self.dataset.data.edge_index
                with torch.no_grad():
                    out, out_ts, edge_index_w_self_loops, all_edge_attn = self.net.inference_w_attn(
                        x, flat, edge_weight, edge_index, self.ts_loader, self.subgraph_loader, self.device)
                return {'hid': out, 'edge_index': edge_index_w_self_loops, 
                        'edge_attn_1': all_edge_attn[0], 'edge_attn_2': all_edge_attn[-1], 'per_node': {}}
            else:
                with torch.no_grad():
                    out, out_ts = self.net.inference(x, flat, edge_weight, 
                                                  self.ts_loader, self.subgraph_loader, self.device,
                                                  get_emb=self.get_emb)
                return {'hid': out, 'per_node': {}}
        else:
            x = self.dataset.data.x
            flat = self.dataset.data.flat
            
            edge_weight = None
            if hasattr(self.dataset.data, 'edge_attr') and self.dataset.data.edge_attr is not None:
                edge_weight = self.dataset.data.edge_attr
            
            with torch.no_grad():
                out, out_ts = self.net.inference(x, flat, edge_weight, self.ts_loader, 
                                               self.subgraph_loader, self.device)
            
            test_indices = torch.where(self.eval_mask)[0]
            
            if len(test_indices) > 0:
                truth = self.dataset.data.y[self.eval_mask]
                masked_out = out[self.eval_mask]
                masked_out_ts = out_ts[self.eval_mask]
                
                test_loss, test_loss_ts, test_tot_loss = self.add_losses(masked_out, masked_out_ts, truth)
                
                self.test_preds.append(masked_out.detach().cpu())
                self.test_preds_ts.append(masked_out_ts.detach().cpu())
                self.test_truths.append(truth.detach().cpu())
                
                return {
                    'test_loss': test_loss,
                    'test_loss_ts': test_loss_ts,
                    'test_tot_loss': test_tot_loss
                }
            else:
                return {'test_loss': torch.tensor(999999.0, device=self.device)}

    def validation_epoch_end(self, outputs):
        if len(self.val_preds) == 0:
            return {'val_loss': 999999.0}
            
        all_preds = torch.cat(self.val_preds, dim=0)
        all_preds_ts = torch.cat(self.val_preds_ts, dim=0)
        all_truths = torch.cat(self.val_truths, dim=0)
            
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_loss_ts = torch.stack([x['val_loss_ts'] for x in outputs if 'val_loss_ts' in x]).mean()
        val_tot_loss = torch.stack([x['val_tot_loss'] for x in outputs if 'val_tot_loss' in x]).mean()
        
        metrics_gps = self.compute_metrics(all_truths, all_preds)
        metrics_ts = self.compute_metrics(all_truths, all_preds_ts)
        
        metrics_ts = {n + '_ts': metrics_ts[n] for n in metrics_ts}
        
        all_metrics = {
            **metrics_gps, 
            **metrics_ts,
            'val_loss_gps': float(val_loss),
            'val_loss_ts': float(val_loss_ts),
            'val_tot_loss': float(val_tot_loss)
        }
        
        for key, value in all_metrics.items():
            self.log(key, value)
        
        self.log('val_loss', float(val_tot_loss), prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_r2', metrics_gps['r2'], prog_bar=True, on_step=False, on_epoch=True)
        
        per_node_gps = self.per_node_metrics(all_truths, all_preds)
        per_node_ts = self.per_node_metrics(all_truths, all_preds_ts)
        per_node_ts = {n + '_ts': per_node_ts[n] for n in per_node_ts}
        per_node = {**per_node_gps, **per_node_ts}
        
        train_r2 = 0.0
        train_loss_mean = None
        if hasattr(self, '_train_preds_accumulator') and self._train_preds_accumulator and self._train_truths_accumulator:
            train_preds = torch.cat(self._train_preds_accumulator, dim=0)
            train_truths = torch.cat(self._train_truths_accumulator, dim=0)

            
            train_metrics = self.compute_metrics(train_truths, train_preds)
            train_r2 = train_metrics['r2']
        
        if hasattr(self, '_train_loss_accumulator') and self._train_loss_accumulator:
            train_loss_mean = np.mean(self._train_loss_accumulator)
        
        current_epoch = self.current_epoch
        self.lambda_history["epoch"].append(current_epoch)
        
        self.lambda_history["train_loss"].append(train_loss_mean)
        self.lambda_history["train_r2"].append(train_r2)
        
        self.lambda_history["val_loss"].append(float(val_tot_loss))
        self.lambda_history["val_r2"].append(metrics_gps['r2'])
        
        if hasattr(self.net, 'graphgps_encoder'):
            lambda_ts = self.net.graphgps_encoder.lambda_ts.item()
            lambda_gps = self.net.graphgps_encoder.lambda_gps.item()
        else:
            lambda_ts = 1.0
            lambda_gps = 1.0
        
        self.lambda_history["lambda_ts"].append(lambda_ts)
        self.lambda_history["lambda_gps"].append(lambda_gps)
        
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.lambda_history, f, indent=2)
        
        if hasattr(self, '_train_loss_accumulator'):
            self._train_loss_accumulator = []
        if hasattr(self, '_train_preds_accumulator'):
            self._train_preds_accumulator = []
        if hasattr(self, '_train_truths_accumulator'):
            self._train_truths_accumulator = []
        
        return {**all_metrics, 'per_node': per_node}
    
    def test_epoch_end(self, outputs):
        if self.get_emb or self.get_logits or self.get_attn:
            collected_outputs = self.collect_outputs(outputs)
            if 'per_node' not in collected_outputs:
                collected_outputs['per_node'] = {}
            return collected_outputs
        
        if len(self.test_preds) == 0:
            return {'test_loss': 999999.0, 'per_node': {}}
            
        all_preds = torch.cat(self.test_preds, dim=0)
        all_preds_ts = torch.cat(self.test_preds_ts, dim=0)
        all_truths = torch.cat(self.test_truths, dim=0)
        
        all_truths   = torch.expm1(all_truths.float())
        all_preds    = torch.expm1(all_preds.float())
        all_preds_ts = torch.expm1(all_preds_ts.float())
            
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_loss_ts = torch.stack([x['test_loss_ts'] for x in outputs if 'test_loss_ts' in x]).mean()
        test_tot_loss = torch.stack([x['test_tot_loss'] for x in outputs if 'test_tot_loss' in x]).mean()
        
        metrics_gps = self.compute_metrics(all_truths, all_preds)
        metrics_ts = self.compute_metrics(all_truths, all_preds_ts)
        
        metrics_ts = {n + '_ts': metrics_ts[n] for n in metrics_ts}
        
        test_metrics = {}
        for k, v in {**metrics_gps, **metrics_ts}.items():
            test_metrics['test_' + k] = v
        
        losses = {
            'test_loss_gps': float(test_loss),
            'test_loss_ts': float(test_loss_ts),
            'test_tot_loss': float(test_tot_loss)
        }
        
        all_metrics = {**test_metrics, **losses}
        
        for key, value in all_metrics.items():
            self.log(key, value)
        
        per_node_gps = self.per_node_metrics(all_truths, all_preds)
        per_node_ts = self.per_node_metrics(all_truths, all_preds_ts)
        per_node_ts = {n + '_ts': per_node_ts[n] for n in per_node_ts}
        per_node = {**per_node_gps, **per_node_ts}
        
        results = {'per_node': per_node, **all_metrics}
        
        return results

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.config['l2'])
        
        if self.config.get('sch', 'plateau') == 'cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config['epochs'])
            return {
                "optimizer": opt,
                "lr_scheduler": sch
            }
        else:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, mode='max')
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "val_r2",
                    "frequency": 1
                }
            }
        
    def train_dataloader(self):
        return self.batch_loader

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)


def get_data(config, us=None, vs=None):
    dataset = GraphDataset(config)
    
    if dataset.data.edge_index.size(0) > 2:
        dataset.data.edge_index = dataset.data.edge_index[:2]

    config['mamba_indim'] = dataset.x_dim
    config['flat_dim'] = dataset.flat_dim
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
        
    return dataset, train_loader, subgraph_loader


def main_forward_pass(config):
    dataset, train_loader, subgraph_loader = get_data(config)
    
    model = MambaGPSModel(
        config, dataset, train_loader, subgraph_loader,
        get_emb=config['fp_emb'], get_logits=config['fp_logits'], get_attn=config['fp_attn']
    )
    
    trainer = pl.Trainer(
        gpus=config['gpus'] if torch.cuda.is_available() else None,
        precision="16-mixed",
        accumulate_grad_batches=config.get("accumulate_grad_batches", 4),
        max_epochs=config.get("max_epochs", 100),
        logger=False,
        callbacks=[early_stop, lr_monitor],
    )
    
    results = trainer.test(model)
    
    if isinstance(results, list):
        results = results[0]
    
    out_dir = Path(config['log_path']) / 'embeddings'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if 'hid' in results and results['hid'] is not None:
        out_path = out_dir / f"{config['version']}_{config['task']}"
        
        if config['fp_emb']:
            torch.save(results['hid'], f"{out_path}_node_emb.pt")
        
        if config['fp_logits']:
            torch.save(results['hid'], f"{out_path}_logits.pt")
        
        if config['fp_attn'] and 'edge_attn_1' in results:
            torch.save(
                (results['edge_index'], results['edge_attn_1'], results['edge_attn_2']),
                f"{out_path}_attn.pt"
            )


def main_test(config):
    model_dir = config['load']
    model, config, dataset, train_loader, subgraph_loader = MambaGPSModel.load_model(model_dir)
    
    trainer = pl.Trainer(
        gpus=config['gpus'] if torch.cuda.is_available() else None,
        logger=False
    )
    
    for phase in ['test', 'valid']:
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
        
        res_dir = Path(config['log_path']) / 'default'
        if config['version'] is not None:
            res_dir = res_dir / config['version']
        else:
            res_dir = res_dir / ('results_' + str(config['seed']))
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        
        per_node = ret.pop('per_node', {})
        write_pkl(per_node, res_dir / f'{phase}_per_node.pkl')
        
        write_json(ret, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)
        
        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, ret)


def main(config):
    dataset, train_loader, subgraph_loader = get_data(config)

    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    chkpt = None
    if config['load'] is not None:
        chkpt_path = get_checkpoint_path(config['load'])
        chkpt = torch.load(chkpt_path, map_location=torch.device('cpu'))

    model = MambaGPSModel(config, dataset, train_loader, subgraph_loader, chkpt=chkpt)
    
    gpus_to_use = config['gpus'] if torch.cuda.is_available() else None

    rt = RuntimeTracker()
    rt.start()

    trainer = pl.Trainer(
        gpus=gpus_to_use,
        logger=logger,
        max_epochs=config['epochs'],
        callbacks=[early_stop, lr_monitor, ckpt],
        distributed_backend='dp' if gpus_to_use and gpus_to_use > 1 else None,
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
        else:
            model.eval_split = 'test'
            model.eval_mask = dataset.data.test_mask
        
        ret = trainer.test(model)
        
        if isinstance(ret, list):
            ret = ret[0]

        if not ret:
            ret = {f"{phase}_loss": 999999.0, "per_node": {}}
        
        per_node = ret.pop('per_node', {})
        write_pkl(per_node, res_dir / f'{phase}_per_node.pkl')
        
        write_json(ret, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)
        
        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, ret)

def auto_disable_edge_types(cfg: dict, ckpt_path: str):
    """
    Disable edge-type one-hots if the checkpoint was trained with edge_dim = 1.
    The function modifies cfg *in-place*.
    """
    if not ckpt_path:
        return

    import torch
    try:
        sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
        w = sd.get('net.graphgps_encoder.edge_encoder.0.weight', None)
        if w is None:
            return

        ckpt_edge_dim = w.shape[1]
        planned_edge_dim = (1 if not cfg.get('with_edge_types', True)
                            else 1 + cfg.get('n_edge_types', 3))

        if ckpt_edge_dim != planned_edge_dim and ckpt_edge_dim == 1:
            cfg['with_edge_types'] = False

    except Exception as e:
        pass


if __name__ == '__main__':
    parser = init_mamba_gps_args()
    parser.add_argument('--fp_emb', action='store_true', help='forward pass to get embeddings')
    parser.add_argument('--fp_logits', action='store_true', help='forward pass to get logits')
    parser.add_argument('--fp_attn', action='store_true', help='forward pass to get attention weights')
    args = parser.parse_args()
    if not hasattr(args, 'g_version') or args.g_version is None or args.g_version == 'default':
        if not args.dynamic_g and not args.random_g:
            args.g_version = 'gps_k3_1_fa_gdcl_sym'
    config = add_configs(args)
    auto_disable_edge_types(config, config.get('load', None))
    
    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if args.mode in ["explain", "ablate"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dataset, train_loader, subgraph_loader = get_data(config)
        ts_loader = create_mamba_dataloader(config)
    
        ckpt_path = args.load
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path = get_checkpoint_path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = MambaGPSModel(config, dataset, train_loader, subgraph_loader, chkpt=ckpt)
        model = model.to(device).eval()                     
        core_net = getattr(model, "net", model)
        
        if args.mode == "explain":
            run_interpretability_experiments(
                core_net, dataset, train_loader, subgraph_loader, ts_loader,
                config, device=device            
            )
        else:
            if args.abl:
                run_ablation_experiment(
                    core_net, train_loader, subgraph_loader, ts_loader,
                    dataset, config, args.abl,
                    device=device        
                )
            else:
                run_all_ablations(
                    core_net, train_loader, subgraph_loader, ts_loader,
                    dataset, config,
                    device=device     
                )
        sys.exit(0)

    if config["fp_emb"] or config["fp_logits"] or config["fp_attn"]:
        main_forward_pass(config)
    elif config["test"]:
        main_test(config)
    else:
        main(config)
