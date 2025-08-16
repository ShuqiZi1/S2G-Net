"""
Main file for training Dynamic LSTM-GNNs
"""
import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from src.models.utils import collect_outputs
from src.dataloader.ts_reader import LstmDataset, collate_fn
from src.models.dgnn import DynamicLstmGnn
from src.models.utils import get_checkpoint_path, seed_everything
from src.evaluation.metrics import get_loss_function, get_metrics, get_per_node_result
from src.config.args import init_lstmgnn_args, add_configs
from src.utils import write_json, write_pkl
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

class DynamicGraphModel(pl.LightningModule):
    """
    Dynamic Graph Model for LSTM-GNN training
    """
    def __init__(self, config, collate, train_set=None, val_set=None, test_set=None):
        super().__init__()
        self.config = config
        self.trainset = train_set
        self.validset = val_set
        self.testset = test_set
        self.learning_rate = self.config['lr']
        self.collate = collate
        self.task = config['task']
        self.is_cls = config['classification']
        
        self.clip_grad = config.get('clip_grad', 1.0)  
        if self.clip_grad <= 0: 
            self.clip_grad = None
        
        self.net = DynamicLstmGnn(config)

        self.loss = get_loss_function(self.task, config['class_weights'])
        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda truth, pred : get_metrics(truth, pred, config['verbose'], config['classification'])
        self.per_node_metrics = lambda truth, pred : get_per_node_result(truth, pred, self.testset.idx_test, config['classification'])
        self.lg_alpha = config['lg_alpha']
        
        self.val_preds = []
        self.val_preds_lstm = []
        self.val_truths = []
        self.test_preds = []
        self.test_preds_lstm = []
        self.test_truths = []

    def on_train_start(self):
        seed_everything(self.config['seed'])
    
    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_preds_lstm = []
        self.val_truths = []
    
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_preds_lstm = []
        self.test_truths = []
    
    def forward(self, seq, flat):
        out = self.net(seq, flat=flat)
        return out
    
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
        try:
            inputs, truth, _ = batch
            seq, flat = inputs
            
            if torch.isnan(seq).any() or (flat is not None and torch.isnan(flat).any()):
                seq = torch.nan_to_num(seq, nan=0.0)
                if flat is not None:
                    flat = torch.nan_to_num(flat, nan=0.0)
            
            pred, pred_lstm = self(seq, flat)
            train_loss, train_loss_lstm, tot_loss = self.add_losses(pred, pred_lstm, truth)
            
            self.log('train_loss', train_loss, prog_bar=True)
            self.log('train_loss_lstm', train_loss_lstm)
            self.log('train_tot_loss', tot_loss)
            
            return {'loss': tot_loss}
        except Exception as e:
            default_loss = torch.tensor(999999.0, device=self.device, requires_grad=True)
            self.log('train_loss', default_loss, prog_bar=True)
            return {'loss': default_loss}

    def validation_step(self, batch, batch_idx):
        try:
            inputs, truth, _ = batch
            seq, flat = inputs
            
            if torch.isnan(seq).any() or (flat is not None and torch.isnan(flat).any()):
                seq = torch.nan_to_num(seq, nan=0.0)
                if flat is not None:
                    flat = torch.nan_to_num(flat, nan=0.0)
            
            out, out_lstm = self(seq, flat)
            loss, loss_lstm, tot_loss = self.add_losses(out, out_lstm, truth)
            
            self.val_preds.append(out.detach().cpu())
            self.val_preds_lstm.append(out_lstm.detach().cpu())
            self.val_truths.append(truth.detach().cpu())
            
            self.log('val_loss_step', loss, prog_bar=False)
            self.log('val_loss_lstm_step', loss_lstm, prog_bar=False)
            self.log('val_tot_loss_step', tot_loss, prog_bar=False)
            
            return {'val_loss': loss, 'val_loss_lstm': loss_lstm, 'val_tot_loss': tot_loss}
        except Exception as e:
            default_loss = torch.tensor(999999.0, device=self.device)
            return {
                'val_loss': default_loss,
                'val_loss_lstm': default_loss,
                'val_tot_loss': default_loss
            }

    def test_step(self, batch, batch_idx):
        try:
            inputs, truth, _ = batch
            seq, flat = inputs
            
            if torch.isnan(seq).any() or (flat is not None and torch.isnan(flat).any()):
                seq = torch.nan_to_num(seq, nan=0.0)
                if flat is not None:
                    flat = torch.nan_to_num(flat, nan=0.0)
            
            out, out_lstm = self(seq, flat)
            loss, loss_lstm, tot_loss = self.add_losses(out, out_lstm, truth)
            
            self.test_preds.append(out.detach().cpu())
            self.test_preds_lstm.append(out_lstm.detach().cpu())
            self.test_truths.append(truth.detach().cpu())
            
            self.log('test_loss_step', loss, prog_bar=False)
            self.log('test_loss_lstm_step', loss_lstm, prog_bar=False)
            self.log('test_tot_loss_step', tot_loss, prog_bar=False)
            
            return {'test_loss': loss, 'test_loss_lstm': loss_lstm, 'test_tot_loss': tot_loss}
        except Exception as e:
            default_loss = torch.tensor(999999.0, device=self.device)
            return {
                'test_loss': default_loss,
                'test_loss_lstm': default_loss,
                'test_tot_loss': default_loss
            }

    def validation_epoch_end(self, outputs):
        try:
            if len(self.val_preds) == 0 or len(self.val_preds_lstm) == 0 or len(self.val_truths) == 0:
                self.log('val_loss', 999999.0, prog_bar=True)
                return {'val_loss': 999999.0, 'per_node': {}}
            
            try:
                avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
                avg_loss_lstm = torch.stack([x['val_loss_lstm'] for x in outputs]).mean()
                avg_tot_loss = torch.stack([x['val_tot_loss'] for x in outputs]).mean()
            except Exception as e:
                avg_loss = torch.tensor(999999.0, device=self.device)
                avg_loss_lstm = torch.tensor(999999.0, device=self.device)
                avg_tot_loss = torch.tensor(999999.0, device=self.device)
            
            self.log('val_loss', avg_loss, prog_bar=True)
            self.log('val_loss_lstm', avg_loss_lstm, prog_bar=False)
            self.log('val_tot_loss', avg_tot_loss, prog_bar=False)
            
            all_preds = torch.cat(self.val_preds, dim=0)
            all_preds_lstm = torch.cat(self.val_preds_lstm, dim=0)
            all_truths = torch.cat(self.val_truths, dim=0)

            self.all_val_preds = all_preds
            self.all_val_preds_lstm = all_preds_lstm
            self.all_val_truths = all_truths
            
            try:
                log_dict_1 = self.compute_metrics(all_truths, all_preds)
                log_dict_2 = self.compute_metrics(all_truths, all_preds_lstm)
                log_dict_2 = {n+ '_lstm': log_dict_2[n] for n in log_dict_2}
            except Exception as e:
                log_dict_1 = {}
                log_dict_2 = {}
            
            for name, value in log_dict_1.items():
                self.log(f'val_{name}', value, prog_bar=False)
                
            for name, value in log_dict_2.items():
                self.log(f'val_{name}', value, prog_bar=False)
            
            per_node = {}
            try:
                per_node_1 = self.per_node_metrics(all_truths, all_preds)
                per_node_2 = self.per_node_metrics(all_truths, all_preds_lstm)
                per_node_2 = {n + '_lstm': per_node_2[n] for n in per_node_2}
                per_node = {**per_node_1, **per_node_2}
            except Exception as e:
                pass
            
            self.val_per_node = per_node
            
            results = {
                'val_loss': avg_loss,
                'metrics': {**log_dict_1, **log_dict_2},
                'progress_bar': {'val_loss': avg_loss},
                'per_node': per_node
            }
            
            return results
        except Exception as e:
            return {'val_loss': 999999.0, 'per_node': {}}

    def test_epoch_end(self, outputs):
        try:
            if len(self.test_preds) == 0 or len(self.test_preds_lstm) == 0 or len(self.test_truths) == 0:
                return {'test_loss': 999999.0, 'per_node': {}}
            
            try:
                avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
                avg_loss_lstm = torch.stack([x['test_loss_lstm'] for x in outputs]).mean()
                avg_tot_loss = torch.stack([x['test_tot_loss'] for x in outputs]).mean()
            except Exception as e:
                avg_loss = torch.tensor(999999.0, device=self.device)
                avg_loss_lstm = torch.tensor(999999.0, device=self.device)
                avg_tot_loss = torch.tensor(999999.0, device=self.device)
            
            self.log('test_loss', avg_loss, prog_bar=True)
            self.log('test_loss_lstm', avg_loss_lstm, prog_bar=False)
            self.log('test_tot_loss', avg_tot_loss, prog_bar=False)
            
            all_preds = torch.cat(self.test_preds, dim=0)
            all_preds_lstm = torch.cat(self.test_preds_lstm, dim=0)
            all_truths = torch.cat(self.test_truths, dim=0)

            all_truths   = torch.expm1(all_truths)
            all_preds    = torch.expm1(all_preds)
            all_preds_lstm = torch.expm1(all_preds_lstm)
            
            try:
                log_dict_1 = self.compute_metrics(all_truths, all_preds)
                log_dict_2 = self.compute_metrics(all_truths, all_preds_lstm)
                log_dict_2 = {n+ '_lstm': log_dict_2[n] for n in log_dict_2}
            except Exception as e:
                log_dict_1 = {}
                log_dict_2 = {}
            
            test_log_dict_1 = {f'test_{k}': v for k, v in log_dict_1.items()}
            test_log_dict_2 = {f'test_{k}': v for k, v in log_dict_2.items()}
            
            for name, value in test_log_dict_1.items():
                self.log(name, value, prog_bar=False)
                
            for name, value in test_log_dict_2.items():
                self.log(name, value, prog_bar=False)
            
            per_node = {}
            try:
                per_node_1 = self.per_node_metrics(all_truths, all_preds)
                per_node_2 = self.per_node_metrics(all_truths, all_preds_lstm)
                per_node_2 = {n + '_lstm': per_node_2[n] for n in per_node_2}
                per_node = {**per_node_1, **per_node_2}
            except Exception as e:
                pass
            
            log_dict = {**test_log_dict_1, **test_log_dict_2, 'test_loss': float(avg_loss),
                      'test_loss_lstm': float(avg_loss_lstm), 'test_tot_loss': float(avg_tot_loss)}
            
            return {'log': log_dict, 'per_node': per_node, **log_dict}
        except Exception as e:
            return {'test_loss': 999999.0, 'per_node': {}}
            
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if hasattr(self, 'clip_grad') and self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.config['l2'])
        if self.config['sch'] == 'cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "val_r2", 
                    "interval": "epoch"
                }
            }
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
        return DataLoader(self.trainset, collate_fn=self.collate, batch_size=self.config['batch_size'], 
                         num_workers=self.config['num_workers'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, collate_fn=self.collate, batch_size=self.config['batch_size'], 
                         num_workers=self.config['num_workers'], shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, collate_fn=self.collate, batch_size=self.config['batch_size'], 
                         num_workers=self.config['num_workers'], shuffle=False)

    @staticmethod
    def load_model(log_dir, **hconfig):
        """
        :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
        :param config: list of named arguments, used to update the model hyperparameters
        """
        assert os.path.exists(log_dir)
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            config = yaml.load(fp, Loader=yaml.Loader)
            config.update(hconfig)

        loaderDict, collate = get_data(config)

        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        args = {'hyperparameters': dict(config), 'collate': collate,
            'train_set': loaderDict['train'], 'val_set': loaderDict['val'], 'test_set': loaderDict['test']}
        model = DynamicGraphModel.load_from_checkpoint(checkpoint_path=str(model_path), **args)

        return model, config, loaderDict, collate


def get_data(config):
    """
    produce dataloaders for training and validating
    """
    loaderDict = {split: LstmDataset(config, split) for split in ['train', 'val', 'test']}
    config['lstm_indim'] = loaderDict['train'].ts_dim
    config['num_flat_feats'] = loaderDict['train'].flat_dim if config['add_flat'] else 0
    config['class_weights'] = loaderDict['train'].class_weights if config['class_weights'] else False
    collate = lambda x: collate_fn(x, config['task'])

    return loaderDict, collate


def main(config):
    """
    Main function for training Dynamic LSTM-GNNs.
    After training, results on validation & test sets are recorded in the specified log_path.
    """
    loaderDict, collate = get_data(config)

    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    if config['debug']:
        model = DynamicGraphModel(config, collate, loaderDict['val'], loaderDict['test'], loaderDict['test'])
    else:    
        model = DynamicGraphModel(config, collate, loaderDict['train'], loaderDict['val'], loaderDict['test'])
    chkpt = None if config['load'] is None else get_checkpoint_path(config['load'])

    rt = RuntimeTracker()
    rt.start()

    trainer = pl.Trainer(
        gpus=config['gpus'],
        logger=logger,
        max_epochs=config['epochs'],
        callbacks=[early_stop_callback],
        distributed_backend='dp',
        precision=16 if config['use_amp'] else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        resume_from_checkpoint=chkpt,
        auto_lr_find=config['auto_lr'],
        auto_scale_batch_size=config['auto_bsz'],
        check_val_every_n_epoch=1,
        val_check_interval=1.0
    )
    torch.cuda.reset_peak_memory_stats() 
    trainer.fit(model)
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
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
        if phase == 'valid':
            ret = trainer.test(test_dataloaders=model.val_dataloader())
        else:
            ret = trainer.test()
        if isinstance(ret, list):
            ret = ret[0]

        per_node = {}
        if 'per_node' in ret:
            per_node = ret.pop('per_node')
        test_results = ret
        res_dir = Path(config['log_path']) / 'default' 
        if config['version'] is not None:
            res_dir = res_dir / config['version']
        else:
            res_dir = res_dir / ('results_' + str(config['seed']))
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        write_json(test_results, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)
        write_pkl(per_node, res_dir / f'{phase}_per_node.pkl')

        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, test_results)


def main_test(hparams, path_results=None):
    """
    main function to load and evaluate a trained model. 
    """
    assert (hparams['load'] is not None) and (hparams['phase'] is not None)
    phase = hparams['phase']
    log_dir = hparams['load']

    model, config, loaderDict, collate = DynamicGraphModel.load_model(log_dir, \
        data_dir=hparams['data_dir'], 
        multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])
    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=None,
        max_epochs=hparams['epochs'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    test_dataloader = DataLoader(loaderDict[phase], collate_fn=collate, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    test_results = trainer.test(model, test_dataloaders=test_dataloader)
    if isinstance(test_results, list):
        test_results = test_results[0]
    per_node = test_results.pop('per_node')
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
    config.model = 'lstmgnn'
    config.dynamic_g = True
    config = add_configs(config)
    
    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['test']:
        main_test(config)
    else:
        main(config)