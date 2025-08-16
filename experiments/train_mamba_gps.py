"""
Training script for MambaGraphGPS models
"""
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from src.models.utils import collect_outputs, seed_everything
from src.dataloader.pyg_reader import GraphDataset
from src.models.pyg_mamba_gps import MambaGraphGPS
from src.models.utils import get_checkpoint_path
from src.evaluation.metrics import get_metrics, get_per_node_result
from src.config.args import add_configs, init_mamba_gps_args
from src.utils import write_json, write_pkl, record_results
from torch_geometric.data import NeighborSampler
from src.dataloader.ts_reader import create_mamba_dataloader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

early_stop = EarlyStopping(monitor='val_r2', min_delta=0.0001, mode='max', patience=4, verbose=True)
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
        
        if chkpt is not None:
            self.net.load_state_dict(chkpt['state_dict'], strict=False)
                
        self.loss = torch.nn.MSELoss()

        self.lg_alpha = float(config.get('lg_alpha', 0.3))

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
        
        bsz_y_log = torch.log1p(bsz_y)
        gps_loss = self.loss(torch.log1p(out.clamp(min=1e-6)), bsz_y_log)
        ts_loss = self.loss(torch.log1p(out_ts.clamp(min=1e-6)), bsz_y_log)
        
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
        
        train_loss, train_loss_ts, tot_loss = self.add_losses(out, out_ts, bsz_y)
        
        self.log('train_loss', train_loss)
        self.log('train_loss_ts', train_loss_ts)
        self.log('train_tot_loss', tot_loss, prog_bar=True)
        
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
            
            self.log('val_loss', val_tot_loss, prog_bar=True)
            
            return {
                'val_loss': val_loss,
                'val_loss_ts': val_loss_ts,
                'val_tot_loss': val_tot_loss
            }
        else:
            default_loss = torch.tensor(999999.0, device=self.device)
            self.log('val_loss', default_loss, prog_bar=True)
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
        
        self.log('val_loss', float(val_tot_loss), prog_bar=True)
        self.log('val_r2', metrics_gps['r2'], prog_bar=True)
        
        per_node_gps = self.per_node_metrics(all_truths, all_preds)
        per_node_ts = self.per_node_metrics(all_truths, all_preds_ts)
        per_node_ts = {n + '_ts': per_node_ts[n] for n in per_node_ts}
        per_node = {**per_node_gps, **per_node_ts}
        
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
    dataset = GraphDataset(config, us, vs)
    
    config['mamba_indim'] = dataset.x_dim
    config['flat_dim'] = dataset.flat_dim
    config['class_weights'] = dataset.class_weights

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    sample_sizes = [config['ns_size1']] if config.get('gps_layers', 1) == 1 else [config['ns_size1'] + config['ns_size2'], config['ns_size1']]

    train_loader = NeighborSampler(dataset.data.edge_index, node_idx=dataset.data.train_mask,
                               sizes=sample_sizes, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    
    subgraph_loader = NeighborSampler(dataset.data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    
    return dataset, train_loader, subgraph_loader


def main_forward_pass(config):
    dataset, train_loader, subgraph_loader = get_data(config)
    
    model = MambaGPSModel(
        config, dataset, train_loader, subgraph_loader,
        get_emb=config['fp_emb'], get_logits=config['fp_logits'], get_attn=config['fp_attn']
    )
    
    trainer = pl.Trainer(
        gpus=config['gpus'] if torch.cuda.is_available() else None,
        logger=False,
        max_epochs=0,
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


if __name__ == '__main__':
    parser = init_mamba_gps_args()
    parser.add_argument('--fp_emb', action='store_true', help='forward pass to get embeddings')
    parser.add_argument('--fp_logits', action='store_true', help='forward pass to get logits')
    parser.add_argument('--fp_attn', action='store_true', help='forward pass to get attention weights')
    config = parser.parse_args()
    config = add_configs(config)
    
    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['fp_emb'] or config['fp_logits'] or config['fp_attn']:
        main_forward_pass(config)
    elif config['test']:
        main_test(config)
    else:
        main(config)