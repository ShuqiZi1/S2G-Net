"""
Command line arguments module
"""
import argparse
import torch
import os
import re
from pathlib import Path
from src.config.hyperparameters.best_parameters import lstmgnn, dynamic, ns_gnn_default, mamba_gps_default, mamba_default, transformer_default
from src.utils import load_json


def parse_graph_version_suffix(g_version: str) -> str:
    """
    Parse graph version string to create a descriptive suffix for directory names
    
    Args:
        g_version: Graph version string (e.g., "gps_k5_3_tf_gdc_")
        
    Returns:
        suffix: Descriptive suffix for directory name
    """
    if not g_version:
        return ""
    
    # Remove trailing underscore if present
    g_version = g_version.rstrip('_')
    
    # Try to parse structured format
    if 'gps_k' in g_version:
        match = re.match(r'gps_k(\d+)_(\d+)(?:_(.+))?', g_version)
        if match:
            k_diag = match.group(1)
            k_bert = match.group(2)
            extra = match.group(3) or ""
            
            suffix_parts = [f"_k{k_diag}b{k_bert}"]
            
            # Parse extra components
            if extra:
                # Method
                if 'tf' in extra or 'tfidf' in extra:
                    suffix_parts.append("tf")
                elif 'fa' in extra or 'faiss' in extra:
                    suffix_parts.append("fa")
                elif 'pn' in extra or 'penalize' in extra:
                    suffix_parts.append("pn")
                
                # Rewiring
                if 'mst' in extra:
                    suffix_parts.append("mst")
                elif 'gdcl' in extra or 'gdc_light' in extra:
                    suffix_parts.append("gdcl")
                elif 'gdc' in extra:
                    suffix_parts.append("gdc")
                
                # Score transform
                if 'log' in extra:
                    suffix_parts.append("log")
                elif 'raw' in extra:
                    suffix_parts.append("raw")
            
            return "_".join(suffix_parts)
    
    # For custom versions, just clean and return
    cleaned = g_version.replace('/', '_').replace(' ', '_')
    return f"_{cleaned}"


def extract_graph_params_from_version(g_version: str) -> dict:
    """
    Extract graph construction parameters from version string
    
    Args:
        g_version: Graph version string
        
    Returns:
        params: Dictionary of extracted parameters
    """
    params = {}
    
    if not g_version:
        return params
    
    # Remove trailing underscore
    g_version = g_version.rstrip('_')
    
    if 'gps_k' in g_version:
        match = re.match(r'gps_k(\d+)_(\d+)(?:_(.+))?', g_version)
        if match:
            params['k_diag'] = int(match.group(1))
            params['k_bert'] = int(match.group(2))
            extra = match.group(3) or ""
            
            # Parse method
            if 'tf' in extra or 'tfidf' in extra:
                params['method'] = 'tfidf'
            elif 'fa' in extra or 'faiss' in extra:
                params['method'] = 'faiss'
            elif 'pn' in extra or 'penalize' in extra:
                params['method'] = 'penalize'
            
            # Parse rewiring
            if 'mst' in extra:
                params['rewiring'] = 'mst'
            elif 'gdcl' in extra or 'gdc_light' in extra:
                params['rewiring'] = 'gdc_light'
            elif 'gdc' in extra:
                params['rewiring'] = 'gdc'
            else:
                params['rewiring'] = 'none'
            
            # Parse score transform
            if 'log' in extra:
                params['score_transform'] = 'log1p'
            elif 'raw' in extra:
                params['score_transform'] = 'none'
            else:
                params['score_transform'] = 'zscore'
    
    return params


def add_best_params(config):
    """Read best set of hyperparams"""
    best = None
    if config['model'] == 'gnn':
        best = ns_gnn_default
    elif config['model'] == 'mamba':
        best = mamba_default
    elif config['model'] == 'mamba-gps':
        best = mamba_gps_default
    elif config['dynamic_g']:
        best = dynamic
    elif config['model'] == 'lstmgnn':
        best = lstmgnn
    elif config['model'] == 'transformer':
        best = transformer_default
    elif config['model'] == 'xgboost':
        return
    
    if best is None:
        raise ValueError(f"Invalid configuration: {config}")

    if config['task'] not in best:
        raise KeyError(f"Invalid task '{config['task']}' in best parameters.")

    # For GNN models, we check the gnn_name
    if 'gnn' in config['model'] and 'gnn_name' in config and config['gnn_name'] not in best[config['task']]:
        raise KeyError(f"Invalid GNN name '{config['gnn_name']}' in best parameters.")
    
    # Get the appropriate parameter set
    best = best[config['task']]
    if 'gnn' in config['model'] and 'gnn_name' in config and config['gnn_name'] in best:
        best = best[config['gnn_name']]

    # Apply best parameters to hyperparameters
    for key, value in best.items():
        config[key] = value
    print('*** using best values for these params', [p for p in best])


def init_arguments():
    """Define general hyperparams"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train',
        choices=['train', 'explain', 'ablate'],
        help='train=train explain=generate explanation ablate=ablate')
    parser.add_argument('--abl', default=None,
        help='--mode ablate specifies a single ablation: static_only / no_static / fixed_lambda / drop_edges / 24h_window')
    
    # General
    parser.add_argument('--config_file', default='paths.json', type=str,
        help='Config file path - command line arguments will override those in the file.')
    parser.add_argument('--read_best', action='store_true')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--task', type=str, choices=['ihm', 'los'], default='ihm')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpus', type=int, default=-1, help='number of available GPUs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
    parser.add_argument('--test', action='store_true', help='enable to skip training and evaluate trained model')
    parser.add_argument('--phase', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--with_edge_types',  dest='with_edge_types',
                        action='store_true',  help='Add one-hot edge-type features')
    parser.add_argument('--no_edge_types',    dest='with_edge_types',
                        action='store_false', help='Ignore edge types (score only)')
    parser.set_defaults(with_edge_types=True)

    # Paths
    parser.add_argument('--version', type=str, help='version tag')
    parser.add_argument('--graph_dir', type=str, help='path of dir storing graph edge data')
    parser.add_argument('--data_dir', type=str, help='path of dir storing raw node data')
    parser.add_argument('--log_path', type=str, help='path to store model')
    parser.add_argument('--load', type=str, help='path to load model from')

    # Data
    parser.add_argument('--ts_mask', action='store_true', help='consider time series mask')
    parser.add_argument('--add_flat', action='store_true', help='concatenate data with flat features.')
    parser.add_argument('--add_diag', action='store_true', help='concatenate data with diag features.')
    parser.add_argument('--flat_first', action='store_true', help='concatenate inputs with flat features.')
    parser.add_argument('--random_g', action='store_true', help='use random graph')
    parser.add_argument('--sample_layers', type=int, help='no. of layers for neighbourhood sampling')

    # Model
    parser.add_argument('--flat_nhid', type=int, default=64)
    parser.add_argument('--model', type=str, choices=['lstm', 'rnn', 'transformer', 'lstmgnn', 'gnn', 'mamba', 'graphgps', 'mamba-gps', 'xgboost'], default='lstmgnn')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fc_dim', type=int, default=32)
    parser.add_argument('--main_dropout', type=float, default=0.45)
    parser.add_argument('--main_act_fn', type=str, default='relu')
    parser.add_argument('--batch_norm_loc', type=str,
        choices=['gnn', 'cat', 'fc', 'transformer'], help='apply batch norm before the specified component.')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--l2', default=5e-4, type=float, help='1e-4')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--sch', type=str, choices=['cosine', 'plateau'], default='plateau')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--clip_grad', type=float, default=0, help='clipping gradient')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--auto_lr', action='store_true')
    parser.add_argument('--auto_bsz', action='store_true')
    return parser


def init_lstm_args():
    """Define LSTM-related hyperparams"""
    parser = init_arguments()
    parser.add_argument('--lstm_indim', type=int)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_nhid', type=int, default=64)
    parser.add_argument('--lstm_pooling', type=str, choices=['all', 'last', 'mean', 'max'], default='last')
    parser.add_argument('--lstm_dropout', type=float, default=0.2)
    parser.add_argument('--bilstm', action='store_true')
    return parser


def init_gnn_args(parser):
    """Define GNN-related hyperparams"""
    parser.add_argument('--dynamic_g', action='store_true', help='dynamic graph')
    parser.add_argument('--edge_weight', action='store_true', help='use edge weight')
    parser.add_argument('--g_version', type=str, default='default')
    parser.add_argument('--ns_size1', type=int, default=25)
    parser.add_argument('--ns_size2', type=int, default=10)
    parser.add_argument('--gnn_name', type=str, choices=['mpnn', 'sgc', 'gcn', 'gat', 'sage'], default='gat')
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--inductive', action='store_true', help='inductive = train / val /test graphs are different')
    parser.add_argument('--self_loop', action='store_true', help='add self loops')
    parser.add_argument('--diag_to_gnn', action='store_true', help='give diag vector to gnn')
    
    # Shared
    parser.add_argument('--gnn_indim', type=int)
    parser.add_argument('--gnn_outdim', type=int, default=64)
    parser.add_argument('--dg_k', type=int, default=3, help='dynamic graph knn')

    # GNN specific parameters
    parser.add_argument('--sgc_layers', type=int, default=1)
    parser.add_argument('--sgc_k', type=int, default=2)
    parser.add_argument('--no_sgc_bias', action='store_true')
    parser.add_argument('--sgc_norm', type=str)
    parser.add_argument('--gcn_nhid', type=int, default=64)
    parser.add_argument('--gcn_layers', type=int, default=1)
    parser.add_argument('--gcn_activation', type=str, default='relu')
    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--gat_nhid', type=int, default=64)
    parser.add_argument('--gat_layers', type=int, default=1)
    parser.add_argument('--gat_n_heads', type=int, default=8)
    parser.add_argument('--gat_n_out_heads', type=int, default=8)
    parser.add_argument('--gat_activation', type=str, default='elu')
    parser.add_argument('--gat_featdrop', type=float, default=0.2)
    parser.add_argument('--gat_attndrop', type=float, default=0.2)
    parser.add_argument('--gat_negslope', type=float, default=0.2)
    parser.add_argument('--gat_residual', action='store_true')
    parser.add_argument('--sage_nhid', type=int, default=64)
    parser.add_argument('--mpnn_nhid', type=int, default=64)
    parser.add_argument('--mpnn_step_mp', type=int, default=3)
    parser.add_argument('--mpnn_step_s2s', type=int, default=6)
    parser.add_argument('--mpnn_layer_s2s', type=int, default=3)
    return parser


def init_lstmgnn_args():
    """Define hyperparams for models with LSTM & GNN components"""
    parser = init_lstm_args()
    parser = init_gnn_args(parser)
    parser.add_argument('--lg_alpha', type=float, default=0.5)
    return parser


def init_mamba_args():
    """Define Mamba-related hyperparameters"""
    parser = init_arguments()
    parser.add_argument('--mamba_indim', type=int)
    parser.add_argument('--mamba_d_model', type=int, default=128)
    parser.add_argument('--mamba_layers', type=int, default=2)
    parser.add_argument('--mamba_dropout', type=float, default=0.1)
    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_pooling', type=str, choices=['all', 'last', 'mean', 'max'], default='last')
    parser.set_defaults(lr=1e-4)
    parser.set_defaults(batch_size=256)
    return parser


def init_transformergnn_args():
    """Define hyperparams for models with Transformer & GNN components"""
    parser = init_transformer_args()
    parser = init_gnn_args(parser)
    parser.add_argument('--lg_alpha', type=float, default=0.5)
    return parser
    
def init_transformer_args():
    """Define Transformer-specific hyperparams."""
    parser = init_arguments()
    parser.add_argument('--trans_indim', type=int, default=256,
                        help='Input feature dimension of transformer (e.g., 26 for time series variables + masks)')
    parser.add_argument('--trans_hidden_dim', type=int, default=128,
                        help='Transformer hidden dimension (d_model)')
    parser.add_argument('--trans_num_heads', type=int, default=4,
                        help='Number of transformer attention heads')
    parser.add_argument('--trans_layers', type=int, default=3,
                        help='Number of transformer encoder layers')
    parser.add_argument('--trans_ffn_dim', type=int, default=256,
                        help='Transformer FFN hidden dimension')
    parser.add_argument('--trans_dropout', type=float, default=0.1,
                        help='Dropout rate in transformer')
    parser.add_argument('--trans_pooling', type=str, choices=['mean', 'last', 'max', 'all'], default='mean',
                        help='Pooling type for transformer output')
    parser.add_argument('--max_seq_len', type=int, default=336,
                        help='Maximum sequence length (e.g., 14*24=336 for 14 days hourly)')
    
    
    parser.add_argument('--batchnorm', type=str, default='default', choices=['default', 'mybatchnorm', 'low_momentum', 'none'])
    parser.add_argument('--final_act_fn', type=str, default='hardtanh', help='Final activation function')


    return parser


def init_graphgps_args(parser=None):
    """Define GraphGPS-related hyperparameters"""
    if parser is None:
        parser = init_arguments()
    parser = init_gnn_args(parser)
    
    parser.add_argument('--gps_hidden_dim', type=int, default=128)
    parser.add_argument('--gps_layers', type=int, default=2)
    parser.add_argument('--gps_dropout', type=float, default=0.1)
    parser.add_argument('--gps_node_dim', type=int)
    parser.add_argument('--gps_out_dim', type=int)
    parser.add_argument('--gps_act_fn', type=str, default='gelu')
    parser.add_argument('--edge_dim', type=int, default=1)
    parser.set_defaults(model='graphgps')
    return parser


def init_mamba_gps_args():
    """Define hyperparameters for models with Mamba & GraphGPS components"""
    parser = init_mamba_args()
    parser = init_graphgps_args(parser)
    parser.add_argument('--drop_edges', type=float, default=0.0,
                    help='Edge dropout rate during *both* training and inference (0~1)')
    parser.add_argument('--lg_alpha', type=float, default=0.5)
    return parser


def get_lstm_out_dim(config):
    """Calculate output dimension of LSTM"""
    if 'lstm_nhid' not in config:
        return 0, 0
    lstm_last_ts_dim = config['lstm_nhid']
    if config['lstm_pooling'] == 'all':
        lstm_out_dim = config['lstm_nhid'] * 24
    else:
        lstm_out_dim = config['lstm_nhid']
    return lstm_out_dim, lstm_last_ts_dim


def get_mamba_out_dim(config):
    """Calculate output dimension of Mamba"""
    if 'mamba_d_model' not in config:
        return 0, 0
    mamba_last_ts_dim = config['mamba_d_model']
    if config['mamba_pooling'] == 'all':
        mamba_out_dim = config['mamba_d_model'] * 24
    else:
        mamba_out_dim = config['mamba_d_model']
    return mamba_out_dim, mamba_last_ts_dim


def get_version_name(config):
    """Return string for model version name with detailed graph construction info"""
    if config['read_best']:
        config['version'] = None
        config['verbose'] = True
        return config

    # Flat feature string
    if config['add_flat']:
        fv = 'flat' + str(config.get('flat_nhid', '')) + '_'
    else:
        fv = ''        

    # Model-specific name components
    model_name = ''
    if 'lstm' in config['model']:
        lstm_nm = 'LSTM'
        if config.get('bilstm', False):
            lstm_nm = 'bi' + lstm_nm
        lstm_nm += str(config.get('lstm_nhid', ''))
        model_name += lstm_nm
    
    if 'gnn' in config['model']:
        gnn_name = config.get('gnn_name', '')
        gnn_nhid = config.get(f'{gnn_name}_nhid', '')
        gnn_outdim = config.get('gnn_outdim', '')
        model_name += f"{gnn_name}{gnn_nhid}out{gnn_outdim}"
    
    if 'mamba' in config['model']:
        mamba_nm = f"Mamba{config.get('mamba_d_model', '')}"
        
        if config['model'] == 'mamba-gps':
            gps_dim = config.get('gps_hidden_dim', '')
            gps_layers = config.get('gps_layers', '')
            mamba_nm += f"_GPS{gps_dim}L{gps_layers}"
        
        model_name += mamba_nm
    
    if config['model'] == 'graphgps':
        gps_dim = config.get('gps_hidden_dim', '')
        gps_layers = config.get('gps_layers', '')
        model_name += f"GPS{gps_dim}L{gps_layers}"
            
    if config['version'] is None:
        # Model description
        version = f"e{config['epochs']}{model_name}"
        # Data
        version += fv
        
        # Graph construction details for GNN-based models
        if ('gnn' in config['model'] or config['model'] in ['graphgps', 'mamba-gps']):
            if config.get('dynamic_g', False):
                # Dynamic graph configuration
                version += "_dynG"
                if config.get('dg_k', 0) != 3:
                    version += str(config.get('dg_k'))
            elif config.get('g_version'):
                # Static graph with specific construction version
                g_version = config['g_version']
                
                # Parse graph construction parameters from g_version
                # Expected format: "gps_k{k_diag}_{k_bert}_" or custom name
                if 'gps_k' in g_version:
                    # Extract k values and method info
                    match = re.match(r'gps_k(\d+)_(\d+)(?:_(.+))?_?', g_version)
                    if match:
                        k_diag = match.group(1)
                        k_bert = match.group(2)
                        suffix = match.group(3) or ""
                        
                        # Build descriptive version string
                        version += f"_Gk{k_diag}b{k_bert}"
                        
                        # Add method if specified in suffix
                        if 'tf' in suffix or 'tfidf' in suffix:
                            version += "tf"
                        elif 'fa' in suffix or 'faiss' in suffix:
                            version += "fa"
                        elif 'pn' in suffix or 'penalize' in suffix:
                            version += "pn"
                        
                        # Add rewiring info if present
                        if 'mst' in suffix:
                            version += "mst"
                        elif 'gdc' in suffix:
                            version += "gdc"
                else:
                    # Custom graph version name
                    version += f"_G{g_version.replace('_', '')}"
                
                # Add additional graph construction parameters if available
                if config.get('edge_weight', False):
                    version += "w"  # weighted edges
                if config.get('self_loop', False):
                    version += "sl"  # self loops
        
        # Training info
        version += 'lr' + str(config['lr'])
        if config['class_weights']:
            version += 'cw_'
        if config['sch'] == 'cosine':
            version += 'cos'
        version += 'l2' + str(config['l2'])
        
        # Sampling configuration for GNNs
        if config.get('ns_sizes') and config.get('ns_sizes') != '25_10':
            version += 'ns' + config['ns_sizes'].replace('_', ':')
        
        # Additional tags
        if config.get('tag') is not None:
            version += 'tag_' + config['tag']
            
        config['version'] = version

    return config


def add_configs(config):
    """Add in additional hyperparameters"""
    config = vars(config)
    
    # Basic hyperparameters setup
    config['verbose'] = config.get('verbose', False)
    config.setdefault('g_version', 'default')
    config.setdefault('dynamic_g', False)
    ns1 = int(config.get('ns_size1', 25))
    ns2 = int(config.get('ns_size2', 10))
    config['ns_sizes'] = f"{ns1 + ns2}_{ns1}"
    config['flat_after'] = config['add_flat'] and (not config['flat_first'])
    config['read_lstm_emb'] = False

    # Validation checks
    if config['add_diag'] and not config['add_flat']:
        raise ValueError("add_diag requires add_flat to be True")

    if (('gnn' in config['model'] or config['model'] in ['mamba-gps', 'graphgps']) and
            not config['dynamic_g'] and not config['random_g'] and not config['g_version']):
        raise ValueError("For GNN models, either dynamic_g, random_g, or g_version must be specified")

    # Task configuration
    if config['task'] == 'ihm':
        config['classification'] = True
        config['out_dim'] = 2
        config['num_cls'] = 2
        config['final_act_fn'] = None
    else:
        config['classification'] = False
        config['out_dim'] = 1
        config['num_cls'] = 1
        config['final_act_fn'] = 'hardtanh'

    config['lstm_attn_type'] = None

    # Model-specific configurations
    if config['model'] == 'mamba':
        # Configure Mamba output dimensions
        config['mamba_outdim'], config['mamba_last_ts_dim'] = get_mamba_out_dim(config)
    elif config['model'] == 'graphgps':
        # GraphGPS specific configurations
        config['lr'] = config.get('lr', 1e-4)
        config['epochs'] = config.get('epochs', 30)
        config['batch_size'] = config.get('batch_size', 128)
        config['gps_node_dim'] = config.get('gps_node_dim', 64)
        config['gps_out_dim'] = config.get('gps_out_dim', config.get('out_dim', 1))
        config['add_last_ts'] = False
    elif config['model'] == 'mamba-gps':
        config['mamba_outdim'], config['mamba_last_ts_dim'] = get_mamba_out_dim(config)
        config['gps_node_dim'] = config['mamba_outdim']
        config['add_last_ts'] = True
        config['gps_out_dim'] = config.get('out_dim', 1)
    elif ('lstm' in config['model']) or ('rnn' in config['model']):
        config['lstm_outdim'], config['lstm_last_ts_dim'] = get_lstm_out_dim(config)
        if config['model'] == 'lstmgnn':
            config['gnn_indim'] = config['lstm_outdim']
            config['add_last_ts'] = True
        else:
            config['add_last_ts'] = False
    else:
        config['add_last_ts'] = False
    
    # GNN specific configurations
    if 'gnn' in config['model'] or config['model'] in ['graphgps', 'mamba-gps']:
        if config.get('gnn_name') == 'gat':
            config['gat_heads'] = ([config['gat_n_heads']] * config['gat_layers']) + [config['gat_n_out_heads']]
        
        # Output dimension setup
        if not config['flat_after']:
            if config['model'] == 'graphgps':
                config['gps_out_dim'] = config['out_dim']
            elif config['model'] != 'mamba-gps':
                config['gnn_outdim'] = config['out_dim']

    # Special case for MPNN
    if (config['model'] == 'lstmgnn') and (not config['dynamic_g']) and config['gnn_name'] == 'mpnn':
        config['ns_sizes'] = str(config['ns_size1'])
    elif config['model'] == 'lstm':
        config['sampling_layers'] = 1

    # GPU configuration
    if config['cpu']:
        num_gpus = 0
        config['gpus'] = None
    else:
        if config['gpus'] is not None:
            num_gpus = torch.cuda.device_count() if config['gpus'] == -1 else config['gpus']
            if num_gpus > 0:
                config['batch_size'] *= num_gpus
                config['num_workers'] *= num_gpus
        else:
            num_gpus = 0
    config['num_gpus'] = num_gpus
    config['multi_gpu'] = num_gpus > 1

    # Load additional configurations
    if 'config_file' in config:
        read_params_from_file(config)

    if config['read_best']:
        add_best_params(config)
    
    # Extract graph parameters from g_version if available
    if config.get('g_version'):
        graph_params = extract_graph_params_from_version(config['g_version'])
        for key, value in graph_params.items():
            config[f'graph_{key}'] = value
        if graph_params:
            print(f"Extracted graph parameters from '{config['g_version']}':")
            for key, value in graph_params.items():
                print(f"  graph_{key}: {value}")

    # Define directory name for each model type with detailed graph version
    if config['model'] == 'gnn':
        dir_name = config['gnn_name'] + '_' + config['task']
        if config.get('g_version'):
            graph_suffix = parse_graph_version_suffix(config['g_version'])
            dir_name += graph_suffix
    elif config['model'] == 'lstmgnn':
        dir_name = 'lstm' + config['gnn_name'] + '_alpha' + str(config['lg_alpha'])
        if config.get('g_version'):
            graph_suffix = parse_graph_version_suffix(config['g_version'])
            dir_name += graph_suffix
    elif config['model'] == 'mamba-gps':
        dir_name = 'mamba-gps_alpha' + str(config['lg_alpha'])
        if config.get('g_version'):
            graph_suffix = parse_graph_version_suffix(config['g_version'])
            dir_name += graph_suffix
    elif config['model'] == 'mamba':
        dir_name = f"mamba_{config['mamba_d_model']}_{config['mamba_layers']}"
    elif config['model'] == 'graphgps':
        dir_name = f"graphgps_{config['gps_hidden_dim']}_{config['gps_layers']}"
        if config.get('g_version'):
            graph_suffix = parse_graph_version_suffix(config['g_version'])
            dir_name += graph_suffix
    else:
        dir_name = config['model']

    # Define inputs string
    inputs = ''
    if config['ts_mask']:
        inputs += 'tm_'
    if config['add_flat']:
        inputs += 'flat'
        if config['flat_first']:
            inputs += 'f'
        if config['add_diag']:
            inputs += 'd'
    if inputs == '':
        inputs = 'seq'
    
    # Handle special path cases
    if config['read_best'] and 'repeat_path' in config:
        config['log_path'] = config['repeat_path']
    else:
        # Configure path based on model type
        if config['model'] == 'mamba':
            config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'mamba' / dir_name)
        elif config['model'] == 'mamba-gps':
            if config['dynamic_g']:
                config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'mamba_gps_dynamic' / dir_name)
            else:
                graph_v = 'graphV' + str(config['g_version'])
                dir_name += '_' + graph_v
                if config['load'] is not None:
                    config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'mamba_gps_pt' / dir_name)
                else:    
                    config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'mamba_gps' / dir_name)
        elif config['model'] == 'graphgps':
            if config['dynamic_g']:
                config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'graphgps_dynamic' / dir_name)
            else:
                graph_v = 'graphV' + str(config['g_version'])
                dir_name += '_' + graph_v
                if config['load'] is not None:
                    config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'graphgps_pt' / dir_name)
                else:    
                    config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'graphgps' / dir_name)
        elif 'gnn' in config['model']:
            graph_v = 'graphV' + str(config['g_version'])
            if config['dynamic_g']:
                config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / (config['model'] + '_led') / dir_name)
            else:
                dir_name += '_' + graph_v
                if config['load'] is not None:
                    config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / (config['model'] + '_pt') / dir_name)
                else:    
                    config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / config['model'] / dir_name)
        elif config['model'] == 'xgboost':
            config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'xgboost')
        else:
            config['log_path'] = str(Path(config['log_path']).resolve() / config['task'] / inputs / 'lstm_baselines')

    # Generate and set version name
    get_version_name(config)
    
    # Create a dedicated results directory for consistent saving
    config['results_dir'] = os.path.join(config['log_path'], 
                                         config['version'] if config['version'] else f"results_{config['seed']}")
    os.makedirs(config['results_dir'], exist_ok=True)
    
    print('log path =', config['log_path'])
    print('results dir =', config['results_dir'])
    print('version name =', config['version'])

    return config


def read_params_from_file(arg_dict, overwrite=False):
    """Read params defined in config_file"""
    if '/' not in arg_dict['config_file']:
        config_path = Path.cwd() / arg_dict['config_file']
    else:
        config_path = Path(arg_dict['config_file'])
    
    # Handle missing hyperparameters file gracefully
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        arg_dict.pop('config_file', None)
        return
    
    try:
        data = load_json(config_path)
        arg_dict.pop('config_file')

        if not overwrite:
            for key, value in data.items():
                if isinstance(value, list) and (key in arg_dict):
                    for v in value:
                        arg_dict[key].append(v)
                elif (key not in arg_dict) or (arg_dict[key] is None):
                    arg_dict[key] = value
        else:
            for key, value in data.items():
                arg_dict[key] = value
    except Exception as e:
        print(f"Error loading hyperparameters file: {e}")
        arg_dict.pop('config_file', None)
