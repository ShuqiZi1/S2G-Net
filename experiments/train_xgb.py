"""
Training script for XGBoost baseline model
Compatible with S2G-Net pipeline structure
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import json
import time
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

from src.models.xgb import XGBoostModel
from src.dataloader.pyg_reader import GraphDataset
from src.evaluation.metrics import get_metrics, get_per_node_result
from src.config.args import add_configs, init_arguments
from src.utils import write_json, write_pkl, record_results
from src.models.utils import seed_everything
from src.evaluation.tracking.runtime import RuntimeTracker
from src.evaluation.tracking.params import analyze_model_complexity


class XGBoostTrainer:
    """
    Trainer class for XGBoost models following S2G-Net pipeline structure
    """
    
    def __init__(self, config, dataset):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            dataset: GraphDataset instance containing data
        """
        self.config = config
        self.dataset = dataset
        self.model = XGBoostModel(config)
        
        # Set up paths
        self.log_path = Path(config['log_path'])
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = (
            Path(self.log_path) / 'default' / (self.config['version'] or f"results_{self.config['seed']}")
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.best_val_metric = float('-inf') if not config['classification'] else float('-inf')
        self.training_history = {
            'train_metrics': [],
            'val_metrics': [],
            'epoch_times': []
        }
        
        self.early_stopping_rounds = config.get('xgb_es_rounds', 50)
        
    def prepare_data(self):
        """
        Prepare training, validation, and test data from GraphDataset
        
        Returns:
            Dictionary containing prepared data splits
        """
        data = self.dataset.data
        
        # Extract masks
        train_mask = data.train_mask.cpu().numpy()
        val_mask = data.val_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
        
        # Extract features
        if data.x.dim() == 3:  # Time series data [N, T, D]
            seq = data.x.cpu().numpy()
        elif data.x.dim() == 2:  # Already flattened [N, D]
            seq = data.x.cpu().numpy()
        else:
            raise ValueError(f"Unexpected data.x dimension: {data.x.dim()}")
        
        # Extract flat features if available
        flat = data.flat.cpu().numpy() if hasattr(data, 'flat') and data.flat is not None else None
        
        # Extract labels
        labels = data.y.cpu().numpy()
        
        # Prepare features for XGBoost
        X_all = self.model.prepare_features(seq, flat)
        y_all = labels
        
        # Split data
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        
        X_val = X_all[val_mask]
        y_val = y_all[val_mask]
        
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]
        
        print(f"Data prepared:")
        print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }
    
    from src.evaluation.metrics import get_metrics

    def compute_metrics(self, y_true, y_pred, phase='train'):
        # to numpy
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
    
        # For LOS, align with LSTM: evaluate on original scale
        if self.config['task'] == 'los':
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
    
        # Reuse project-wide metrics for apples-to-apples comparison
        m = get_metrics(y_true, y_pred, self.config.get('verbose', False), self.config['classification'])
    
        # prefix with phase and cast to float
        return {f"{phase}_{k}": (float(v) if hasattr(v, 'item') else float(v)) for k, v in m.items()}

    def train(self):
        """
        Train the XGBoost model
        
        Returns:
            Dictionary containing training results
        """
        print("\n" + "="*80)
        print("Training XGBoost Model")
        print("="*80)
        
        # Prepare data
        data_dict = self.prepare_data()
        
        # Start runtime tracking
        rt = RuntimeTracker()
        rt.start()
        
        start_time = time.time()
        
        # Train model
        print("\nTraining model...")
        self.model.fit(
            data_dict['X_train'], 
            data_dict['y_train'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val']
        )
        
        train_time = time.time() - start_time
        rt.stop()
        
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate on all splits
        results = {}
        
        for phase in ['train', 'val', 'test']:
            X = data_dict[f'X_{phase}']
            y = data_dict[f'y_{phase}']
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Compute metrics
            metrics = self.compute_metrics(y, y_pred, phase)
            results.update(metrics)
            
            # Print metrics
            print(f"\n{phase.upper()} Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        
        # Save model
        model_path = self.results_dir / 'xgboost_model.pkl'
        self.model.save(str(model_path))
        print(f"\nModel saved to: {model_path}")
        
        # Get feature importance
        feature_importance = self.model.get_feature_importance()
        feature_importance = {k: float(v) for k, v in feature_importance.items()}
        importance_path = self.results_dir / 'feature_importance.json'
        write_json(feature_importance, importance_path, sort_keys=True)
        print(f"Feature importance saved to: {importance_path}")
        
        # Save runtime statistics
        runtime_stats = {
            "train_hours": train_time / 3600,
            "train_seconds": train_time,
            "peak_vram_GB": rt.get_peak_vram_gb(),
            "n_estimators": self.model.xgb_params['n_estimators'],
            "n_features": data_dict['X_train'].shape[1]
        }
        
        rt_path = self.results_dir / "runtime.json"
        write_json(runtime_stats, rt_path, sort_keys=True)
        print(f"Runtime stats saved to: {rt_path}")
        
        # Per-node results
        for phase in ['val', 'test']:
            X = data_dict[f'X_{phase}']
            y = data_dict[f'y_{phase}']
            y_pred = self.model.predict(X)
            
            # Convert back to original scale for LOS
            if self.config['task'] == 'los':
                y = np.expm1(y)
                y_pred = np.expm1(y_pred)
            
            per_node = get_per_node_result(
                torch.from_numpy(y),
                torch.from_numpy(y_pred),
                self.dataset.idx_test if phase == 'test' else self.dataset.idx_val,
                self.config['classification']
            )
            
            per_node_path = self.results_dir / f'{phase}_per_node.pkl'
            write_pkl(per_node, per_node_path)
            print(f"{phase.upper()} per-node results saved to: {per_node_path}")
        
        # Save overall results
        results_path = self.results_dir / 'results.json'
        write_json(results, results_path, sort_keys=True, verbose=True)
        print(f"Results saved to: {results_path}")
        
        # Record results to CSV
        for phase in ['test', 'val']:
            phase_results = {k: v for k, v in results.items() if k.startswith(f'{phase}_')}
            csv_path = self.log_path / f'all_{phase}_results.csv'
            record_results(csv_path, self.config, phase_results)
        
        return results
    
    def test(self, model_path=None):
        """
        Test a trained model
        
        Args:
            model_path: Path to saved model (optional, loads from results_dir if None)
            
        Returns:
            Dictionary containing test results
        """
        # Load model if path provided
        if model_path:
            self.model = XGBoostModel.load(model_path)
        elif not self.model.fitted:
            # Try to load from default location
            default_path = self.results_dir / 'xgboost_model.pkl'
            if default_path.exists():
                self.model = XGBoostModel.load(str(default_path))
            else:
                raise ValueError("No trained model found. Please train or provide model path.")
        
        # Prepare data
        data_dict = self.prepare_data()
        
        # Evaluate on test set
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        y_pred = self.model.predict(X_test)
        
        # Compute metrics
        test_metrics = self.compute_metrics(y_test, y_pred, 'test')
        
        print("\nTest Results:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save test results
        test_results_path = self.results_dir / 'test_results.json'
        write_json(test_metrics, test_results_path, sort_keys=True, verbose=True)
        
        return test_metrics


def get_data(config):
    """
    Load dataset
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GraphDataset instance
    """
    dataset = GraphDataset(config)
    
    # Set feature dimensions in config
    config['xgb_indim'] = dataset.x_dim
    config['flat_dim'] = dataset.flat_dim
    config['class_weights'] = dataset.class_weights
    
    return dataset


def main(config):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
    """
    # Seed everything for reproducibility
    seed_everything(config['seed'])
    
    # Load data
    print("Loading dataset...")
    dataset = get_data(config)
    
    # Initialize trainer
    trainer = XGBoostTrainer(config, dataset)
    
    # Train model
    results = trainer.train()
    
    return results


def main_test(config):
    """
    Main testing function
    
    Args:
        config: Configuration dictionary
    """
    # Seed everything
    seed_everything(config['seed'])
    
    # Load data
    print("Loading dataset...")
    dataset = get_data(config)
    
    # Initialize trainer
    trainer = XGBoostTrainer(config, dataset)
    
    # Test model
    model_path = config.get('load', None)
    results = trainer.test(model_path)
    
    return results


if __name__ == '__main__':
    # Initialize argument parser
    parser = init_arguments()
    
    # Add XGBoost-specific arguments
    parser.add_argument('--xgb_lr', type=float, default=0.05,
                        help='Learning rate for XGBoost (vs. lr for deep nets)')
    parser.add_argument('--xgb_max_depth', type=int, default=3, 
                        help='Maximum tree depth for XGBoost')
    parser.add_argument('--xgb_n_estimators', type=int, default=100,
                        help='Number of boosting rounds')
    parser.add_argument('--xgb_subsample', type=float, default=0.7,
                        help='Subsample ratio of training instances')
    parser.add_argument('--xgb_colsample', type=float, default=0.7,
                        help='Subsample ratio of columns when constructing each tree')
    parser.add_argument('--xgb_ts_agg', type=str, default='last',
                        choices=['mean', 'last', 'all', 'statistical'],
                        help='Time series aggregation method')
    parser.add_argument('--xgb_es_rounds', type=int, default=40,
                        help='Early stopping rounds for XGBoost (only used if eval_set is provided)')
    
    # Parse arguments
    args = parser.parse_args()
    args.model = 'xgboost'
    
    # Add configurations
    config = add_configs(args)
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 80)
    for key in sorted(config):
        print(f'{key}: {config[key]}')
    print("-" * 80)
    
    # Run training or testing
    if config['test']:
        main_test(config)
    else:
        main(config)
