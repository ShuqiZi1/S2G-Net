"""
XGBoost model implementation for healthcare prediction tasks
Compatible with the S2G-Net pipeline structure
"""
import numpy as np
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any


class XGBoostModel:
    """
    XGBoost model wrapper that follows the S2G-Net model interface.
    Handles both time series and flat features for healthcare prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost model with configuration
        
        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        self.config = config
        self.task = config.get('task', 'los')
        self.classification = config.get('classification', False)
        
        # XGBoost hyperparameters
        self.xgb_params = {
            'max_depth': config.get('xgb_max_depth', 6),
            'learning_rate': config.get('xgb_lr', config.get('lr', 0.05)),
            'n_estimators': config.get('xgb_n_estimators', 100),
            'objective': self._get_objective(),
            'eval_metric': self._get_eval_metric(),
            'subsample': config.get('xgb_subsample', 0.8),
            'colsample_bytree': config.get('xgb_colsample', 0.8),
            'reg_alpha': config.get('l2', 0.0),  # L1 regularization
            'reg_lambda': config.get('l2', 0.001),  # L2 regularization
            'random_state': config.get('seed', 2020),
            'n_jobs': config.get('num_workers', -1),
            'verbosity': 1 if config.get('verbose', False) else 0,
        }
        
        # Add GPU support if available
        if config.get('gpus', 0) and torch.cuda.is_available():
            self.xgb_params['tree_method'] = 'gpu_hist'
            self.xgb_params['gpu_id'] = 0
        
        # Initialize model
        if self.classification:
            self.model = xgb.XGBClassifier(**self.xgb_params)
        else:
            self.model = xgb.XGBRegressor(**self.xgb_params)
        
        # Feature preprocessing
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Feature aggregation settings
        self.ts_aggregation = config.get('xgb_ts_agg', 'last')  # 'mean', 'last', 'all', 'statistical'
        self.add_flat = config.get('add_flat', True)
        self.add_diag = config.get('add_diag', True)
        
        self.early_stopping_rounds = int(config.get('xgb_es_rounds', 0) or 0)
        
    def _get_objective(self) -> str:
        """Get XGBoost objective function based on task"""
        if self.classification:
            return 'binary:logistic'
        else:
            return 'reg:squarederror'
    
    def _get_eval_metric(self) -> str:
        """Get evaluation metric based on task"""
        if self.classification:
            return 'auc'
        else:
            return 'rmse'
    
    def _aggregate_timeseries(self, seq: np.ndarray) -> np.ndarray:
        """
        Aggregate time series data into fixed-size feature vector
        
        Args:
            seq: Time series data of shape [N, T, D] or [N, D]
            
        Returns:
            Aggregated features of shape [N, D_agg]
        """
        if seq.ndim == 2:
            return seq
        
        N, T, D = seq.shape
        
        if self.ts_aggregation == 'mean':
            # Mean pooling across time
            return seq.mean(axis=1)
        
        elif self.ts_aggregation == 'last':
            # Use last time step
            return seq[:, -1, :]
        
        elif self.ts_aggregation == 'statistical':
            # Compute statistical features: mean, std, min, max, last
            features = []
            features.append(seq.mean(axis=1))  # Mean
            features.append(seq.std(axis=1))   # Std
            features.append(seq.min(axis=1))   # Min
            features.append(seq.max(axis=1))   # Max
            features.append(seq[:, -1, :])     # Last
            return np.concatenate(features, axis=1)
        
        elif self.ts_aggregation == 'all':
            # Flatten all time steps
            return seq.reshape(N, -1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.ts_aggregation}")
    
    def prepare_features(
        self, 
        seq: np.ndarray, 
        flat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Prepare input features for XGBoost by aggregating time series and concatenating flat features
        
        Args:
            seq: Time series data [N, T, D] or [N, D]
            flat: Flat features [N, F] (optional)
            
        Returns:
            Combined feature matrix [N, D_total]
        """
        # Convert torch tensors to numpy if needed
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        if isinstance(flat, torch.Tensor):
            flat = flat.cpu().numpy()
        
        # Aggregate time series
        ts_features = self._aggregate_timeseries(seq)
        
        # Combine with flat features
        if self.add_flat and flat is not None:
            features = np.concatenate([ts_features, flat], axis=1)
        else:
            features = ts_features
        
        return features
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        eval_set: Optional[list] = None
    ):
        """
        Train the XGBoost model with optional early stopping (callback-based for compatibility)
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
    
        # Prepare evaluation set
        if eval_set is None and X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        elif eval_set is not None:
            eval_set = [(self.scaler.transform(X), y) for X, y in eval_set]
    
        # Build kwargs for fit (keep it minimal and version-safe)
        fit_kwargs = {"verbose": self.config.get('verbose', False)}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
    
        # Try callback-based early stopping (works across many XGBoost versions)
        callbacks = []
        if eval_set and self.early_stopping_rounds > 0:
            try:
                # Avoid extra args for maximum compatibility
                callbacks = [xgb.callback.EarlyStopping(rounds=self.early_stopping_rounds)]
                fit_kwargs["callbacks"] = callbacks
            except Exception:
                # If callbacks are unavailable, silently fall back to no early stopping
                pass
    
        # Fit with best-effort compatibility
        try:
            self.model.fit(X_train_scaled, y_train, **fit_kwargs)
        except TypeError:
            # Some very old versions may not accept "callbacks"; retry without it
            fit_kwargs.pop("callbacks", None)
            self.model.fit(X_train_scaled, y_train, **fit_kwargs)
    
        self.fitted = True
        return self

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
    
        X_scaled = self.scaler.transform(X)
    
        # Prefer new API iteration_range; fallback to best_ntree_limit if needed
        best_iter = getattr(self.model, 'best_iteration', None)
        use_best = best_iter is not None
    
        try:
            if use_best:
                iter_range = (0, int(best_iter) + 1)
                if self.classification:
                    return self.model.predict_proba(X_scaled, iteration_range=iter_range)
                else:
                    return self.model.predict(X_scaled, iteration_range=iter_range)
        except TypeError:
            # xgboost<1.6: use best_ntree_limit
            if use_best:
                best_ntree_limit = getattr(self.model, 'best_ntree_limit', None)
                if best_ntree_limit is not None:
                    if self.classification:
                        return self.model.predict_proba(X_scaled, ntree_limit=best_ntree_limit)
                    else:
                        return self.model.predict(X_scaled, ntree_limit=best_ntree_limit)
    
        # No early stopping info or old API not available
        if self.classification:
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict(X_scaled)
    
    def predict_from_raw(
        self, 
        seq: np.ndarray, 
        flat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions from raw time series and flat features
        
        Args:
            seq: Time series data [N, T, D]
            flat: Flat features [N, F] (optional)
            
        Returns:
            Predictions [N]
        """
        X = self.prepare_features(seq, flat)
        return self.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def save(self, path: str):
        """Save model to file"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        """Load model from file"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model_wrapper = cls(data['config'])
        model_wrapper.model = data['model']
        model_wrapper.scaler = data['scaler']
        model_wrapper.fitted = True
        
        return model_wrapper
