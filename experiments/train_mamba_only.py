import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from src.models.mamba import define_mamba_encoder
from src.dataloader.ts_reader import MambaDataset, create_mamba_dataloader
from src.evaluation.metrics import get_loss_function, get_metrics, get_per_node_result
from src.config.args import init_mamba_gps_args,init_mamba_args, add_configs
from src.utils import write_json, write_pkl, record_results
import torch.nn as nn
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

class MambaOnlyModel(pl.LightningModule):
    """
    LightningModule for training and testing the Mamba time-series model
    """
    def __init__(self, config, eval_split='test'):
        super().__init__()
        self.config = config
        self.net = define_mamba_encoder()(config)
        
        flat_dim = config['num_flat_feats']
        flat_out_dim = 128
        self.flat_fc = nn.Linear(flat_dim, flat_out_dim)
        self.flat_drop = nn.Dropout(0.1)
        
        self.head = nn.Linear(config['mamba_d_model'] + flat_out_dim, 1)
        
        self.clip_grad = config.get('clip_grad', 1.0)  
        if self.clip_grad <= 0: 
            self.clip_grad = None
        
        self.loss_fn = get_loss_function(config) if 'loss_fn' in config else torch.nn.MSELoss()
        
        def compute_metrics_fn(y=None, y_hat=None, dict_input=None):
            if dict_input is not None:
                return get_metrics(dict_input['truth'], dict_input['pred'], 
                                  config.get('verbose', False), config.get('classification', False))
            else:
                return get_metrics(y, y_hat, config.get('verbose', False), config.get('classification', False))
                
        self.compute_metrics = compute_metrics_fn
        
        if 'idx_test' in config:
            self.idx_test = config['idx_test']
        else:
            self.idx_test = None
            
        self.per_node_metrics = lambda x: get_per_node_result(x['truth'], x['pred'], 
                                                             self.idx_test, config.get('classification', False))
        
        self.eval_split = eval_split
        
        self.val_preds = []
        self.val_truths = []
        self.test_preds = []
        self.test_truths = []

    def forward(self, x, flat, mask):
        """
        Forward pass.
        """
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        if flat is not None and torch.isnan(flat).any():
            flat = torch.nan_to_num(flat, nan=0.0)
            
        seq_rep, _ = self.net(x, mask)
        flat_rep = self.flat_drop(self.flat_fc(flat))
        y_hat = self.head(torch.cat([seq_rep, flat_rep], dim=1))
        return y_hat.squeeze(-1)

    def on_train_start(self):
        """Set seeds at start of training"""
        if 'seed' in self.config:
            pl.seed_everything(self.config['seed'])

    def training_step(self, batch, batch_idx):
        (seq, flat, mask), y, ids = batch
        y_hat = self.forward(seq, flat, mask)
        y_hat = torch.clamp(y_hat, min=1e-6)
        
        if torch.isnan(y_hat).any() or torch.isnan(y).any():
            y_hat = torch.nan_to_num(y_hat, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
            
        loss = self.loss_fn(y_hat, y)
        
        if torch.isnan(loss):
            loss = torch.tensor(999999.0, device=self.device, requires_grad=True)
            
        self.log('train_loss', loss, prog_bar=True)

        if batch_idx == 0 and self.config.get('verbose', False):
            hat = self.forward(seq, flat, mask)
            self.zero_grad(set_to_none=True)
            hat.sum().backward()
            for n, p in self.net.named_parameters():
                if p.grad is not None:
                    break
        return loss

    def validation_step(self, batch, batch_idx):
        (seq, flat, mask), y, ids = batch
        y_hat = self.forward(seq, flat, mask)
        y_hat = torch.clamp(y_hat, min=1e-6)
        
        if torch.isnan(y_hat).any() or torch.isnan(y).any():
            y_hat = torch.nan_to_num(y_hat, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
            
        loss = self.loss_fn(y_hat, y)
        
        if torch.isnan(loss):
            loss = torch.tensor(999999.0, device=self.device)
            
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_preds.append(y_hat.detach().cpu())
        self.val_truths.append(y.cpu())
        
        return {'val_loss': loss}
    
    def on_validation_epoch_start(self):
        """Reset prediction lists before validation epoch"""
        self.val_preds = []
        self.val_truths = []

    def test_step(self, batch, batch_idx):
        """
        Test step with enhanced results storage.
        """
        (seq, flat, mask), y, ids = batch
        y_hat = self.forward(seq, flat, mask)
        y_hat = torch.clamp(y_hat, min=1e-6)
        
        if torch.isnan(y_hat).any() or torch.isnan(y).any():
            y_hat = torch.nan_to_num(y_hat, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
            
        loss = self.loss_fn(y_hat, y)
        
        if torch.isnan(loss):
            loss = torch.tensor(999999.0, device=self.device)
        
        self.test_preds.append(y_hat.detach().cpu())
        self.test_truths.append(y.cpu())
        
        return {'test_loss': loss}
    
    def on_test_epoch_start(self):
        """Reset prediction lists before test epoch"""
        self.test_preds = []
        self.test_truths = []

    def validation_epoch_end(self, outputs):
        """
        Process validation results at the end of the epoch
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
            
            log_dict = self.compute_metrics(dict_input=collect_dict)
            
            val_metrics = {}
            for metric_name, metric_value in log_dict.items():
                val_name = 'val_' + metric_name if not metric_name.startswith('val_') else metric_name
                val_metrics[val_name] = metric_value
            
            val_metrics['val_loss'] = float(val_loss)
            
            for metric_name, metric_value in val_metrics.items():
                self.log(metric_name, metric_value)
            
            results = {'log': val_metrics}
            results = {**results, **val_metrics}
            
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
        Process test results at the end of the epoch
        """
        try:
            all_preds = torch.cat(self.test_preds, dim=0)
            all_truths = torch.cat(self.test_truths, dim=0)

            all_truths   = torch.expm1(all_truths.float()).cpu()
            all_preds    = torch.expm1(all_preds.float()).cpu()
            
            test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            
            collect_dict = {
                'truth': all_truths,
                'pred': all_preds,
                'test_loss': test_loss
            }
            
            log_dict = self.compute_metrics(dict_input=collect_dict)
            log_dict = {'test_' + m: log_dict[m] for m in log_dict}
            log_dict['test_loss'] = float(test_loss)
            
            for metric_name, metric_value in log_dict.items():
                self.log(metric_name, metric_value)
            
            if hasattr(self, 'idx_test') and self.idx_test is not None:
                per_node_results = self.per_node_metrics(collect_dict)
            else:
                per_node_results = {}
            
            results = {'log': log_dict, 'per_node': per_node_results}
            results = {**results, **log_dict}
            
            return results
        except Exception as e:
            try:
                test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
                self.log('test_loss', float(test_loss))
                return {'test_loss': float(test_loss), 'per_node': {}}
            except:
                return {'test_loss': 999999.0, 'per_node': {}}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """Apply gradient clipping before optimizer step"""
        if hasattr(self, 'clip_grad') and self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.get('lr', 1e-3),
                                      weight_decay=self.config.get('weight_decay', 1e-2))
        if self.config.get('sch', 'cosine') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.trainer.max_epochs)
            return [optimizer], [scheduler]
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
    
    @staticmethod
    def load_model(log_dir, **hparams):
        """
        Load model from checkpoint
        """
        assert os.path.exists(log_dir)
        try:
            with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
                config = yaml.load(fp, Loader=yaml.Loader)
                config.update(hparams)
        except Exception as e:
            raise

        try:
            model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
            model = MambaOnlyModel.load_from_checkpoint(checkpoint_path=str(model_path), config=config)
        except Exception as e:
            raise

        return model, config

def main(config):

    ts_dataset = MambaDataset(config)
    config['num_flat_feats'] = ts_dataset.flat_dim

    ts_loader = create_mamba_dataloader(config, split='train')
    val_loader = create_mamba_dataloader(config, split='val')
    test_loader = create_mamba_dataloader(config, split='test')

    model = MambaOnlyModel(config)
        
    gpus_to_use = config.get('gpus', 0) if torch.cuda.is_available() else None

    rt = RuntimeTracker()
    rt.start()

    trainer = pl.Trainer(
        gpus=gpus_to_use,
        max_epochs=config.get('epochs', 50),
        callbacks=[early_stop_callback],
        distributed_backend='dp' if gpus_to_use and config.get('multi_gpu', False) else None,
        precision=16 if config.get('use_amp', False) and gpus_to_use else 32,
        default_root_dir=config.get('log_path', './logs'),
        deterministic=True,
        auto_lr_find=config.get('auto_lr', False),
        auto_scale_batch_size=config.get('auto_bsz', False)
    )
    
    trainer.fit(model, train_dataloaders=ts_loader, val_dataloaders=val_loader)
    
    valid_metrics = trainer.validate(model, dataloaders=val_loader)
    test_metrics = trainer.test(model, dataloaders=test_loader)
    
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
        if test_loader:
            inf_ms = measure_inference_speed(lambda batch: model(*batch), test_loader, device)
        else:
            inf_ms = 0.0
    except Exception as e:
        print("⚠️ Inference speed test failed:", e)
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
        res_dir = Path(config.get('log_path', './logs')) / 'default'
        if config.get('version') is not None:
            res_dir = res_dir / config['version']
        else:
            res_dir = res_dir / ('results_' + str(config.get('seed', 42)))
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        
        if phase == 'valid':
            model.eval_split = 'val'
        else:
            model.eval_split = 'test'
                
        if phase == 'valid':
            ret = trainer.validate(model, dataloaders=val_loader)
        else:
            ret = trainer.test(model, dataloaders=test_loader)
        
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

        path_results = Path(config.get('log_path', './logs')) / f'all_{phase}_results.csv'
        record_results(path_results, config, test_results)
        
def main_test(hparams, path_results=None):
    """
    Main function to load and evaluate a trained model.
    """
    assert hparams.get('load') is not None, "Must provide a checkpoint path using --load"
    phase = hparams.get('phase', 'test')
    log_dir = hparams['load']

    try:
        model, config = MambaOnlyModel.load_model(
            log_dir, 
            data_dir=hparams.get('data_dir')
        )

        ts_dataset = MambaDataset(config)
        
        if phase == 'valid':
            loader = create_mamba_dataloader(config, split='val')
        else:
            loader = create_mamba_dataloader(config, split='test')

        trainer = pl.Trainer(
            gpus=config.get('gpus', 0) if torch.cuda.is_available() else None,
            logger=None,
            max_epochs=1
        )
        
        model.eval_split = 'val' if phase == 'valid' else 'test'

        if phase == 'valid':
            test_results = trainer.validate(model, dataloaders=loader)
        else:
            test_results = trainer.test(model, dataloaders=loader)
        
        if isinstance(test_results, list):
            test_results = test_results[0]
        
        if not test_results:
            test_results = {f"{phase}_loss": 999999.0, "per_node": {}}
        
        per_node = {}
        if 'per_node' in test_results:
            per_node = test_results.pop('per_node')
        
        res_dir = Path(config.get('results_dir', './results'))
        res_dir.mkdir(parents=True, exist_ok=True)
        
        write_pkl(per_node, res_dir / f'{phase}_per_node.pkl')
        write_json(test_results, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)

        if path_results is None:
            path_results = Path(config.get('log_path', './logs')) / f'all_{phase}_results.csv'
        tmp = {'version': hparams.get('version')}
        tmp = {**tmp, **config}
        record_results(path_results, tmp, test_results)

        return test_results
    except Exception as e:
        return {f"{phase}_loss": 999999.0, "per_node": {}}


if __name__ == '__main__':
    parser = init_mamba_gps_args()
    config = parser.parse_args()
    config.model = 'mamba'
    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: {config[key]}')
    
    if config['test']:
        main_test(config)
    else:
        main(config)


