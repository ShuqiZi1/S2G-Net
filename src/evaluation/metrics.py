import numpy as np
import torch
import torch.nn as nn
from torchmetrics import classification, regression
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred), torch.log(actual))


def get_loss_function(task, class_weights):
    if task == 'ihm':
        if class_weights is False:
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.MSELoss()


def define_metrics(task):
    m = {}
    if task == 'ihm':
        # these take labels:
        m['acc'] = classification.Accuracy()
    else:
        m['mae'] = regression.MAE()
        m['rmse'] = regression.RMSE()
        m['rmlse'] = regression.RMSLE()
    return m


class CustomBins:
    inf = 1e18
    # 10 classes
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return None


def get_percentile_bin(x, edges):
    """Assign bin based on percentile edges"""
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2

def mean_absolute_percentage_error(y_true, y_pred):
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    denom = np.maximum(1e-8, np.abs(y_true_log))
    return np.mean(np.abs((y_true_log - y_pred_log) / denom)) * 100

def mean_squared_logarithmic_error(y_true, y_pred):
    """Standard MSLE = mean( (log(1 + y_true) - log(1 + y_pred))^2 )"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred)))


def get_per_node_result(y_true, pred_prob, ids, is_cls):
    if is_cls:
        return per_node_class_result(y_true, pred_prob, ids)
    else:
        return per_node_reg_result(y_true, pred_prob, ids)


def per_node_class_result(y_true, pred_prob, ids):
    """
    Collect per node results for classification tasks
    Returns boolean indicators of correct predictions
    """
    if isinstance(ids, range):
        ids = np.array(ids)
    labels = pred_prob.argmax(axis=1)
    
    per_node = {}
    per_node['ids'] = ids.astype(int)
    per_node['true'] = y_true.astype(int)
    per_node['pred'] = labels.astype(int)
    per_node['correct_cls'] = (labels == y_true).astype(int)
    per_node['correct_0'] = per_node['correct_cls'] & (y_true == 0)
    per_node['correct_1'] = per_node['correct_cls'] & (y_true == 1)
    return per_node


def per_node_reg_result(y_true, pred, ids):
    """
    Collect per node results for regression tasks
    Returns various error metrics for each node
    """
    if isinstance(ids, range):
        ids = np.array(ids)
    per_node = {}
    per_node['ids'] = ids.astype(int)
    per_node['y_true'] = y_true
    per_node['pred'] = pred
    per_node['err'] = (y_true - pred)
    per_node['abs_err'] = np.abs((y_true - pred))  # absolute error
    per_node['per_err'] = np.abs((y_true - pred) / np.maximum(4/24, y_true))
    per_node['sq_err'] = np.square((y_true - pred))  # square error
    per_node['sq_log_err'] = np.square(np.log(y_true/pred)) # square log error
    return per_node


def print_metrics_regression(y_true, predictions, verbose):
    """
    Calculate regression metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        predictions = predictions.detach().cpu().numpy()
    
    # Ensure inputs are arrays, not scalars
    if np.isscalar(y_true) or (hasattr(y_true, 'ndim') and y_true.ndim == 0):
        y_true = np.array([y_true])
    
    if np.isscalar(predictions) or (hasattr(predictions, 'ndim') and predictions.ndim == 0):
        predictions = np.array([predictions])
    
    # Make sure inputs are 1D arrays
    y_true = np.atleast_1d(y_true).flatten()
    predictions = np.atleast_1d(predictions).flatten()
    
    # Remove NaN values if any
    if np.isnan(y_true).any() or np.isnan(predictions).any():
        valid_indices = ~(np.isnan(y_true) | np.isnan(predictions))
        y_true = y_true[valid_indices]
        predictions = predictions[valid_indices]
        
    # Handle empty arrays
    if len(y_true) == 0 or len(predictions) == 0:
        return {
            'kappa': 0.0,
            'mad': 0.0,
            'mse': 0.0,
            'mape': 0.0,
            'msle': 0.0,
            'r2': 0.0
        }
    
    # Ensure array lengths match
    assert len(y_true) == len(predictions), f"Length mismatch: y_true({len(y_true)}) vs predictions({len(predictions)})"
    
    # Calculate bin-based metrics
    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    
    # Ensure valid label range
    valid_labels = list(range(CustomBins.nbins))
    cf = confusion_matrix(y_true_bins, prediction_bins, labels=valid_labels)
    
    results = {}
    
    # Calculate Kappa coefficient
    try:
        # Calculate observed agreement rate (po)
        po = np.sum(np.diag(cf)) / np.sum(cf) if np.sum(cf) > 0 else 0
        
        # Calculate expected agreement rate (pe)
        row_sums = np.sum(cf, axis=1)
        col_sums = np.sum(cf, axis=0)
        total = np.sum(cf)
        
        if total > 0:
            pe = sum([(row_sums[i] * col_sums[i]) / (total**2) for i in range(len(valid_labels))])
            
            # Calculate weighted kappa
            if abs(pe - 1.0) < 1e-10:
                results['kappa'] = 1.0
            else:
                weights = np.zeros((len(valid_labels), len(valid_labels)))
                for i in range(len(valid_labels)):
                    for j in range(len(valid_labels)):
                        # Linear weights
                        weights[i, j] = 1.0 - abs(i - j) / (len(valid_labels) - 1) if len(valid_labels) > 1 else 1.0
                
                po_w = sum([cf[i, j] * weights[i, j] for i in range(len(valid_labels)) for j in range(len(valid_labels))]) / total
                pe_w = sum([row_sums[i] * col_sums[j] * weights[i, j] for i in range(len(valid_labels)) for j in range(len(valid_labels))]) / (total**2)
                
                if abs(pe_w - 1.0) < 1e-10:
                    results['kappa'] = 1.0
                else:
                    kappa = (po_w - pe_w) / (1 - pe_w)
                    # Ensure valid range
                    results['kappa'] = max(-1.0, min(1.0, kappa))
        else:
            results['kappa'] = 0.0
    except Exception:
        results['kappa'] = 0.0
    
    # Standard regression metrics
    results['mad'] = metrics.mean_absolute_error(y_true, predictions)
    results['mse'] = metrics.mean_squared_error(y_true, predictions)
    
    # Safely calculate MAPE
    try:
        results['mape'] = mean_absolute_percentage_error(y_true, predictions)
    except Exception:
        results['mape'] = 0.0
    
    # Safely calculate MSLE
    try:
        # Ensure positive values
        valid_indices = (y_true > 0) & (predictions > 0)
        if np.any(valid_indices):
            results['msle'] = mean_squared_logarithmic_error(
                y_true[valid_indices], predictions[valid_indices])
        else:
            results['msle'] = 0.0
    except Exception:
        results['msle'] = 0.0
    
    # Calculate R²
    try:
        if len(y_true) > 1:
            y_mean = np.mean(y_true)
            
            # Calculate total sum of squares
            ss_total = np.sum((y_true - y_mean) ** 2)
            
            # Calculate residual sum of squares
            ss_residual = np.sum((y_true - predictions) ** 2)
            
            # Avoid division by zero
            if ss_total > 1e-6:
                r2 = 1 - (ss_residual / ss_total)
                results['r2'] = max(-1.0, min(1.0, r2))
            else:
                if ss_residual < 1e-6:
                    results['r2'] = 1.0
                else:
                    results['r2'] = 0.0
        else:
            results['r2'] = 0.0
    except Exception:
        results['r2'] = 0.0
    
    # Ensure all results are float values
    results = {key: float(results[key]) for key in results}
    
    if verbose:
        for key in results:
            print("{}: {:.4f}".format(key, results[key]))
    
    return results


def get_metrics(y_true, pred_prob, verbose, is_cls):
    """
    Process metrics for all samples together.
    
    Parameters:
    y_true: List or tensor of true values from all batches
    pred_prob: List or tensor of predicted values from all batches
    verbose: Whether to print verbose output
    is_cls: Whether this is a classification task
    
    Returns:
    Dictionary of metrics
    """
    # Concatenate if they're lists
    if isinstance(y_true, list):
        if isinstance(y_true[0], torch.Tensor):
            y_true = torch.cat(y_true)
        else:
            y_true = np.concatenate(y_true)
    
    if isinstance(pred_prob, list):
        if isinstance(pred_prob[0], torch.Tensor):
            pred_prob = torch.cat(pred_prob)
        else:
            pred_prob = np.concatenate(pred_prob)
    
    if is_cls:
        return compute_binary_metrics(y_true, pred_prob, verbose)
    else:
        return print_metrics_regression(y_true, pred_prob, verbose)


def compute_binary_metrics(y, pred_prob, verbose):
    pred_prob = np.array(pred_prob)
    labels = pred_prob.argmax(axis=1)
    cf = confusion_matrix(y, labels, labels=range(2))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)
    
    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    if len(set(y)) != 1:
        auroc = metrics.roc_auc_score(y, pred_prob[:, 1])
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y, pred_prob[:, 1])
        auprc = metrics.auc(recalls, precisions)
        minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
        f1macro = metrics.f1_score(y, labels, average='macro')
    else:
        auroc = np.nan
        auprc = np.nan
        minpse = np.nan
        f1macro = np.nan

    results = {"acc": acc, "prec0": prec0, "prec1": prec1, "rec0": rec0, "rec1": rec1,
               "auroc": auroc, "auprc": auprc, "minpse": minpse, 
               "f1macro": f1macro}
    results = {key: float(results[key]) for key in results}
    if verbose:
        for key in results:
            print("{}: {:.4f}".format(key, results[key]))

    return results
