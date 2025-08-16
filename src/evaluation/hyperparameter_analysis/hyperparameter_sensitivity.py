"""
Publishable-level hyperparameter sensitivity analysis for Mamba-GPS ICU LOS prediction
This script performs systematic hyperparameter optimization and sensitivity analysis
for the Mamba-GPS model on ICU length-of-stay prediction task.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import optuna
import gc, inspect
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from src.config.args import init_mamba_gps_args, add_configs
from experiments.train_mamba_gps_enhgraph import MambaGPSModel, get_data

from pytorch_lightning.callbacks import Callback


def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
class SafePruningCallback(Callback):
    """Compatible pruning callback for old Lightning versions."""
    def __init__(self, trial, monitor: str):
        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()
        self._trial.report(current_score, step=trainer.current_epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()

class HyperparameterSensitivityAnalyzer:
    """
    Comprehensive hyperparameter sensitivity analysis for Mamba-GPS model
    """
    
    def __init__(self, base_config, output_dir="sensitivity_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define hyperparameter search spaces based on your model architecture
        self.search_spaces = {
            # Mamba time-series encoder parameters
            'mamba_d_model': [64, 128, 256],
            'mamba_layers': [2, 3, 4],
            'mamba_d_state': [16, 32, 64],
            'mamba_dropout': [0.1, 0.2],
            'mamba_pooling': ['mean', 'last'],
            
            # GraphGPS encoder parameters
            'gps_layers': [2, 3, 4],
            'gps_dropout': [0.1, 0.2],
            
            # Fusion and training parameters
            'lg_alpha': [0.3, 0.5, 0.7],
            'lr': [1e-5, 1e-4, 5e-4],
            'batch_size': [32, 64, 128],
            'clip_grad': [0.0, 2.0, 5.0],
            
            # Neighborhood sampling (keeping both parameters together)
            'sampling_config': [(15, 10), (25, 15)]
        }
        
        self.results = []
        
    def create_config_from_params(self, params):
        config = self.base_config.copy()
        for key, value in params.items():
            if key == 'sampling_config':
                config['ns_size1'], config['ns_size2'] = value
            else:
                config[key] = value
                
        config['gps_node_dim'] = config['mamba_d_model']
        config['gps_hidden_dim'] = config['mamba_d_model']
        config['hidden_dim']     = config['mamba_d_model']
        config['mamba_last_ts_dim'] = config['mamba_d_model'] 
        return config
    
    def objective_function(self, trial):
        """
        Optuna objective function for hyperparameter optimization
        Returns validation R² score
        """
        try:
            # Sample hyperparameters
            params = {}
            for param_name, values in self.search_spaces.items():
                if param_name == 'lr':
                    # Use log scale for learning rate
                    params[param_name] = trial.suggest_float(param_name, 1e-5, 1e-3, log=True)
                elif isinstance(values[0], (int, float)):
                    if isinstance(values[0], int):
                        params[param_name] = trial.suggest_categorical(param_name, values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, values)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, values)
            
            # Create config
            config = self.create_config_from_params(params)
            
            # Get data with current config
            dataset, train_loader, subgraph_loader = get_data(config)
            
            # Create model
            model = MambaGPSModel(config, dataset, train_loader, subgraph_loader)
            
            # Setup trainer with pruning callback
            trainer = pl.Trainer(
                max_epochs=config.get('max_epochs', 20),  # Reduced for sensitivity analysis
                gpus=1 if torch.cuda.is_available() else 0,
                logger=False,
                checkpoint_callback=False,
                callbacks=[
                    SafePruningCallback(trial, monitor="val_r2"),
                    pl.callbacks.EarlyStopping(monitor='val_r2', patience=6, mode='max')
                ],
                deterministic=True
            )
            
            # Train model
            trainer.fit(model)
            
            metric_keys = [
                'val_loss', 'val_r2',
                'kappa', 'mad', 'mse', 'mape', 'msle'
            ]
            
            metrics = {}
            for k in metric_keys:
                v = trainer.callback_metrics.get(k)
                if isinstance(v, torch.Tensor):
                    v = v.item()
                metrics[k] = float(v) if v is not None else None 
            
            val_r2 = metrics.get('val_r2')
            if val_r2 is None:   
                raise optuna.exceptions.TrialPruned()
            
            
            result = params.copy()
            result.update(metrics)  
            result['trial_number'] = trial.number
            self.results.append(result)
           
            return val_r2
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            return 0.0

        finally:
            try:
                if 'trainer' in locals() and hasattr(trainer, "teardown"):
                    trainer.teardown()
            except Exception as ex:
                print(f"Teardown failed: {ex}")
            for var in ['trainer', 'model', 'dataset', 'train_loader', 'subgraph_loader']:
                if var in locals():
                    try:
                        del locals()[var]
                    except Exception:
                        pass
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        
    def run_sensitivity_analysis(self, n_trials=200, study_name="mamba_gps_sensitivity"):
        """
        Run comprehensive hyperparameter sensitivity analysis
        """
        print(f"Starting hyperparameter sensitivity analysis with {n_trials} trials...")
        
        # Create Optuna study
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            study_name=study_name,
            storage=f"sqlite:///{self.output_dir}/optuna_study.db",
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective_function, n_trials=n_trials)
        
        # Save results
        self.save_results(study)
        
        self.export_all_analysis_data(study)
        
        return study
    
    def save_results(self, study):
        """Save optimization results to files"""
        # Save trial results
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(self.output_dir / "trial_results.csv", index=False)
        
        # Save best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        with open(self.output_dir / "best_parameters.json", 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': float(best_value),
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        print(f"Best validation R²: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
    
    def export_all_analysis_data(self, study):
        print("Exporting analysis data as CSV (no plots)...")
    
        self._save_parameter_importance_data(study)
        self._save_individual_sensitivity_data()
        self._save_parameter_interaction_data()
        self._save_optimization_history(study)
        self._save_top_trials_parallel_data(study)
    
        print("All analysis CSVs saved to:", self.output_dir)
    
    def _save_parameter_importance_data(self, study):
        """Save parameter importance scores to CSV instead of plotting"""
        importance = optuna.importance.get_param_importances(study)
        df = pd.DataFrame(list(importance.items()), columns=['parameter', 'importance'])
        df.to_csv(self.output_dir / "parameter_importance.csv", index=False)

    
    def _save_individual_sensitivity_data(self):
        df = pd.DataFrame(self.results)
    
        numeric_params = ['mamba_d_model', 'mamba_layers', 'mamba_d_state', 'mamba_dropout',
                           'gps_layers', 'gps_dropout', 'lg_alpha', 'lr', 'clip_grad']
    
        records = []
        for param in numeric_params:
            if param in df.columns:
                grouped = df.groupby(param)['val_r2'].agg(['mean', 'std']).reset_index()
                grouped.insert(0, 'parameter', param)
                records.append(grouped)
    
        df_combined = pd.concat(records, axis=0)
        df_combined.to_csv(self.output_dir / "individual_sensitivity_summary.csv", index=False)

    
    def _save_parameter_interaction_data(self):
        df = pd.DataFrame(self.results)
        key_params = ['mamba_d_model', 'mamba_layers', 'gps_layers', 'lg_alpha']
        interaction_records = []
    
        for i, p1 in enumerate(key_params):
            for j, p2 in enumerate(key_params):
                if i < j and p1 in df.columns and p2 in df.columns:
                    interaction_df = df.groupby([p1, p2])['val_r2'].mean().reset_index()
                    interaction_df['param1'] = p1
                    interaction_df['param2'] = p2
                    interaction_records.append(interaction_df)
    
        if interaction_records:
            df_interactions = pd.concat(interaction_records, axis=0)
            df_interactions.to_csv(self.output_dir / "parameter_interactions.csv", index=False)

        
    def _save_optimization_history(self, study):
        records = []
        for t in study.trials:
            if t.value is not None:
                records.append({
                    'trial_number': t.number,
                    'val_r2': t.value
                })
    
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / "optimization_history.csv", index=False)

    
    def _save_top_trials_parallel_data(self, study, top_n=20):
        trials = [t for t in study.trials if t.value is not None]
        sorted_trials = sorted(trials, key=lambda x: x.value, reverse=True)[:top_n]
    
        rows = []
        for t in sorted_trials:
            row = t.params.copy()
            row['val_r2'] = t.value
            row['trial_number'] = t.number
            rows.append(row)
    
        df_top = pd.DataFrame(rows)
        df_top.to_csv(self.output_dir / "top_trials.csv", index=False)

    
    def generate_summary_report(self, study):
        """Generate a summary report of findings"""
        df = pd.DataFrame(self.results)
        
        report = f"""
# Hyperparameter Sensitivity Analysis Report
## Mamba-GPS ICU Length-of-Stay Prediction

### Experimental Setup
- Total trials: {len(study.trials)}
- Best validation R²: {study.best_value:.4f}
- Search space: {len(self.search_spaces)} hyperparameters

### Key Findings

#### Best Hyperparameters:
"""
        
        for param, value in study.best_params.items():
            report += f"- {param}: {value}\n"
        
        report += f"""
#### Parameter Sensitivity Rankings:
"""
        
        # Calculate parameter importance (simplified)
        if len(df) > 10:
            importance_scores = {}
            for param in self.search_spaces.keys():
                if param in df.columns and param != 'sampling_config':
                    try:
                        if df[param].dtype in ['object', 'category']:
                            # For categorical variables
                            variance = df.groupby(param)['val_r2'].var().mean()
                        else:
                            # For numeric variables
                            correlation = abs(df[param].corr(df['val_r2']))
                            variance = correlation
                        importance_scores[param] = variance
                    except:
                        importance_scores[param] = 0.0
            
            # Sort by importance
            sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (param, score) in enumerate(sorted_importance[:10]):
                report += f"{i+1}. {param}: {score:.4f}\n"
        
        report += f"""
### Recommendations:
1. Focus on optimizing the top 3-5 most sensitive parameters
2. Parameters with low sensitivity can be fixed to reduce search space
3. Consider parameter interactions for fine-tuning

### Files Generated:
- trial_results.csv: Complete trial results
- best_parameters.json: Best hyperparameters found
- *.png: Visualization plots
"""
        
        with open(self.output_dir / "sensitivity_report.md", 'w') as f:
            f.write(report)
        
        print("Summary report generated: sensitivity_report.md")


def main():
    import argparse
    
    cli_parser = argparse.ArgumentParser(
        description="Hyperparameter Sensitivity Analysis for Mamba-GPS")
    cli_parser.add_argument('--n_trials',   type=int,   default=100)
    cli_parser.add_argument('--output_dir', type=str, default="results/hyperparameter", help="Output directory")
    cli_parser.add_argument('--study_name', type=str,   default="mamba_gps_sensitivity")
    
    cli_parser.add_argument('--g_version',  type=str,   default='default',
                            help="Graph version name; use 'default' for built-in setting")
    cli_parser.add_argument('--dynamic_g',  action='store_true',
                            help="Use dynamic k-NN graph construction")
    cli_parser.add_argument('--random_g',   action='store_true',
                            help="Use random graph baseline")

    cli_args = cli_parser.parse_args() 
    if not hasattr(cli_args, 'g_version') or cli_args.g_version is None or cli_args.g_version == 'default':
        if not cli_args.dynamic_g and not cli_args.random_g:
            cli_args.g_version = 'gps_k3_1_fa_gdcl_sym'
            print(f"Setting default g_version to: {cli_args.g_version}")

    if cli_args.g_version in (None, 'default'):
        cli_args.g_version = 'gps_k3_1_fa_gdcl_sym'

    base_cmd = [
        '--model', 'mamba-gps',
        '--ts_mask', '--add_flat', '--class_weights',
        '--add_diag', '--task', 'los', 
        '--with_edge_types',
        '--g_version', cli_args.g_version
    ]
    if cli_args.dynamic_g:
        base_cmd.append('--dynamic_g')
    if cli_args.random_g:
        base_cmd.append('--random_g')

    base_args   = init_mamba_gps_args().parse_args(base_cmd)
    base_config = add_configs(base_args)
    
    # Initialize analyzer
    analyzer = HyperparameterSensitivityAnalyzer(base_config, cli_args.output_dir)
    
    # Run sensitivity analysis
    study = analyzer.run_sensitivity_analysis(
        n_trials=cli_args.n_trials,
        study_name=cli_args.study_name
    )


    # Generate summary report
    analyzer.generate_summary_report(study)
    
    print(f"\nHyperparameter sensitivity analysis completed!")
    print(f"Results saved to: {cli_args.output_dir}")
    print(f"Best validation R²: {study.best_value:.4f}")


if __name__ == "__main__":
    main()