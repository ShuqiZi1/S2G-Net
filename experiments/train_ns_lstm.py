"""
Main file for training LSTMs and RNNs
"""
import os
from pathlib import Path
from tqdm import tqdm
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from src.models.utils import collect_outputs
from src.dataloader.pyg_reader import GraphDataset
from src.models.lstm import Net as LSTMNet
from src.models.rnn  import Net as RNNNet
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
import json

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=True,
    mode='min'
)

class Model(pl.LightningModule):
    """
    Node sampling LSTM model
    """
    def __init__(self, config, dataset, train_loader, subgraph_loader, eval_split='test', get_lstm_out=False, get_logits=False):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.batch_loader = train_loader
        self.subgraph_loader = subgraph_loader
        self.learning_rate = config['lr']
        self.get_lstm_out = get_lstm_out
        self.get_logits = get_logits
        
        self.clip_grad = config.get('clip_grad', 1.0)  
        if self.clip_grad <= 0: 
            self.clip_grad = None
        
        NetCls = RNNNet if config.get('model', 'lstm') == 'rnn' else LSTMNet
        self.net = NetCls(config)
        
        self.loss = get_loss_function(config['task'], config['class_weights'])

        self.collect_outputs = lambda x: collect_outputs(x, config['multi_gpu'])
        self.compute_metrics = lambda x: get_metrics(x['truth'], x['pred'], config['verbose'], config['classification'])
        self.per_node_metrics = lambda x: get_per_node_result(x['truth'], x['pred'], self.dataset.idx_test, config['classification'])

        self.eval_split = eval_split
        self.eval_mask = self.dataset.data.val_mask if eval_split == 'val' else self.dataset.data.test_mask

        entire_set = LstmDataset(config)
        collate = lambda x: collate_fn(x, config['task'])
        self.ts_loader = DataLoader(entire_set, collate_fn=collate, \
                batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
        
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
    
    def forward(self, x, flat):
        out = self.net(x, flat)
        return out

    def training_step(self, batch, batch_idx):
        batch_size, n_id, adjs = batch
        bsz_nids = n_id[:batch_size]
        
        if self.device.type == 'cuda' and bsz_nids.device.type == 'cpu':
            bsz_nids = bsz_nids.to(self.device)
        
        x = self.dataset.data.x[bsz_nids].to(self.device)
        flat = self.dataset.data.flat[bsz_nids].to(self.device)
        
        if torch.isnan(x).any() or torch.isnan(flat).any():
            x = torch.nan_to_num(x, nan=0.0)
            flat = torch.nan_to_num(flat, nan=0.0)
        
        out = self(x, flat=flat)
        bsz_y = self.dataset.data.y[bsz_nids].to(self.device)

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
        lstm_outs = []
        truth = []
        
        for inputs, labels, ids in tqdm(self.ts_loader):
            seq, flat = inputs
            out = self.net.forward(seq.to(self.device), flat=flat.to(self.device))
            lstm_outs.append(out)
            truth.append(labels)
        
        all_truth = torch.cat(truth, dim=0)
        all_out = torch.cat(lstm_outs, dim=0)
        
        min_dim = min(all_truth.shape[0], all_out.shape[0], self.dataset.data.val_mask.shape[0])
        mask = self.dataset.data.val_mask[:min_dim]
        truth_masked = all_truth[:min_dim]
        out_masked = all_out[:min_dim]
        
        val_indices = torch.where(mask)[0]
        
        if len(val_indices) > 0:
            val_truth = truth_masked[val_indices].to(self.device)
            val_pred = out_masked[val_indices].to(self.device)
            
            self.val_truths.append(val_truth.detach().cpu())
            self.val_preds.append(val_pred.detach().cpu())
            
            val_loss = self.loss(val_pred.squeeze(), val_truth)
        else:
            val_loss = torch.tensor(999999.0, device=self.device)
        
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        """
        Test step that stores predictions
        """
        if self.get_lstm_out or self.get_logits:
            lstm_outs = []
            for inputs, labels, ids in tqdm(self.ts_loader):
                seq, flat = inputs
                if self.get_lstm_out:
                    hid = self.net.forward_to_lstm(seq.to(self.device), flat=flat.to(self.device))
                    lstm_outs.append(hid)
                else:
                    out = self.net.forward(seq.to(self.device), flat=flat.to(self.device))
                    lstm_outs.append(out)
            out = torch.cat(lstm_outs, dim=0)
            return {'hid': out}
        
        lstm_outs = []
        truth = []
        
        for inputs, labels, ids in tqdm(self.ts_loader):
            seq, flat = inputs
            out = self.net.forward(seq.to(self.device), flat=flat.to(self.device))
            lstm_outs.append(out)
            truth.append(labels)
        
        all_truth = torch.cat(truth, dim=0)
        all_out = torch.cat(lstm_outs, dim=0)
        
        min_dim = min(all_truth.shape[0], all_out.shape[0], self.eval_mask.shape[0])
        mask = self.eval_mask[:min_dim]
        truth_masked = all_truth[:min_dim]
        out_masked = all_out[:min_dim]
        
        test_indices = torch.where(mask)[0]
        
        if len(test_indices) > 0:
            test_truth = truth_masked[test_indices].to(self.device)
            test_pred = out_masked[test_indices].to(self.device)
            
            self.test_truths.append(test_truth.detach().cpu())
            self.test_preds.append(test_pred.detach().cpu())
            
            test_loss = self.loss(test_pred.squeeze(), test_truth)
        else:
            test_loss = torch.tensor(999999.0, device=self.device)
        
        return {'test_loss': test_loss}

    def validation_epoch_end(self, outputs):
        """
        Processes all validation outputs
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
            
            for metric_name, metric_value in log_dict.items():
                self.log(metric_name, metric_value, prog_bar=True)
            
            results = {'log': log_dict}
            results = {**results, **log_dict}
            
            return results
        except Exception as e:
            val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            self.log('val_loss', float(val_loss), prog_bar=True)
            return {'val_loss': float(val_loss)}

    def test_epoch_end(self, outputs):
        """
        Processes all test outputs
        """
        if self.get_lstm_out or self.get_logits:
            return outputs[0] if isinstance(outputs, list) else outputs
        
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
            test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            self.log('test_loss', float(test_loss))
            return {'test_loss': float(test_loss)}

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
        if 'add_diag' not in config:
            config['add_diag'] = False

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

    train_loader = NeighborSampler(dataset.data.edge_index, node_idx=dataset.data.train_mask,
                               sizes=[25, 10], batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    subgraph_loader = NeighborSampler(dataset.data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)

    return dataset, train_loader, subgraph_loader


def main(config):
    """
    Main function for training LSTMs.
    After training, results on validation & test sets are recorded in the specified log_path.
    """
    dataset, train_loader, subgraph_loader = get_data(config)
    
    rt = RuntimeTracker()
    rt.start()

    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], version=config['version'])
    logger.log_hyperparams(params=config)

    model = Model(config, dataset, train_loader, subgraph_loader)
    chkpt = None if config['load'] is None else get_checkpoint_path(config['load'])

    gpus_to_use = config['gpus'] if torch.cuda.is_available() else None

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
    hrs = rt.get_training_hours()
    eps = trainer.current_epoch / (hrs * 3600) if hrs else 0

    try:
        inf_ms = measure_inference_speed(
            model.net,
            model.ts_loader,
            model.device,
            num_samples=100
        )
    except Exception as e:
        inf_ms = 0.0

    minfo = analyze_model_complexity(model.net)
    runtime_stats = {
        "train_hours": round(hrs, 4),
        "epoch_per_sec": round(eps, 4),
        "inference_ms_per_patient": round(inf_ms, 4),
        "peak_vram_GB": round(rt.get_peak_vram_gb(), 2),
        "total_params": int(minfo["total_params"]),
        "model_size_mb": minfo["model_size_mb"]
    }
    
    rt_path = Path(config['log_path']) / "runtime.json"
    rt_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(Path(config['log_path']) / "runtime.json", "w") as f:
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
            ret = {f"{phase}_loss": 999999.0}
        
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

    model, config, dataset, train_loader, subgraph_loader = Model.load_model(log_dir, multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])
    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=None,
        max_epochs=hparams['epochs'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    
    if phase == 'valid':
        trainer.eval_split = 'val'
        trainer.eval_mask = dataset.data.val_mask

    test_results = trainer.test(model)
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

def main_forward_pass(hparams):
    """
    Main function to load a trained model and execute a forward pass to get logits for analysis.
    """
    log_dir = hparams['load']
    model, config, dataset, train_loader, subgraph_loader = Model.load_model(log_dir, \
        data_dir=hparams['data_dir'], multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])
    if hparams['fp_lstm']:
        model.get_lstm_out = True
    else:
        model.get_logits = True

    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=None,
        max_epochs=hparams['epochs'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    import numpy as np
    test_results = trainer.test(model)
    if isinstance(test_results, list):
        test_results = test_results[0]
    hid = test_results['hid']
    
    if hparams['fp_lstm']:
        out_path = Path(log_dir) / 'lstm_last_hid.npy'
    else:
        out_path = Path(log_dir) / 'lstm_logits.npy'
    with open(out_path, 'wb') as f:
        np.save(f, hid)


if __name__ == '__main__':
    parser = init_lstmgnn_args()
    parser.add_argument('--fp_lstm', action='store_true', help='forward pass to get lstm embeddings')
    parser.add_argument('--fp_logits', action='store_true', help='forward pass to get logits')
    config = parser.parse_args()
    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: ', config[key])
    
    if config['fp_lstm'] or config['fp_logits']:
        main_forward_pass(config)
    elif config['test']:
        main_test(config)
    else:
        main(config)
