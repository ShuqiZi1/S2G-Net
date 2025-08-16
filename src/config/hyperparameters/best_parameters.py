lstmgnn = {
    'los':{
        'mpnn':{
            'batch_size': 32,
            'lr': 0.000493634,
            'l2': 1.71462e-05,
            'main_dropout': 0.485793,
            'lg_alpha': 1.543381,
            'ns_size1': 20,
            'gnn_outdim': 512,
            'mpnn_nhid': 64,
            'mpnn_step_mp': 4
        },
        'gat':{
            'batch_size': 128,
            'lr': 0.000422616,
            'l2': 0.000333634,
            'main_dropout': 0.224398,
            'lg_alpha': 1.42422,
            'ns_size1': 30,
            'ns_size2': 10,
            'gnn_outdim': 256,
            'gat_nhid': 128,
            'gat_n_heads': 12,
            'gat_n_out_heads': 10,
            'gat_featdrop': 0.453771,
            'gat_attndrop': 0.616016
        },
        'sage':{
            'batch_size': 256,
            'lr': 0.000400711,
            'l2': 5.15579e-05,
            'main_dropout': 0.352443,
            'lg_alpha': 1.35791,
            'ns_size1': 10,
            'ns_size2': 30,
            'gnn_outdim': 64,
            'sage_nhid': 256
        }
    },
    'ihm':{
        'mpnn':{
            'batch_size': 32,
            'lr': 0.000693634,
            'l2': 1.71462e-05,
            'main_dropout': 0.485793,
            'lg_alpha': 0.543381,
            'ns_size1': 20,
            'gnn_outdim': 512,
            'mpnn_nhid': 64,
            'mpnn_step_mp': 4
        },
        'gat':{
            'batch_size': 128,
            'lr': 0.000722616,
            'l2': 0.000333634,
            'main_dropout': 0.224398,
            'lg_alpha': 1.32422,
            'ns_size1': 30,
            'ns_size2': 10,
            'gnn_outdim': 256,
            'gat_nhid': 128,
            'gat_n_heads': 12,
            'gat_n_out_heads': 10,
            'gat_featdrop': 0.453771,
            'gat_attndrop': 0.616016
        },
        'sage': {
            'batch_size': 64,
            'lr': 0.000627118,
            'l2': 2.21619e-05,
            'main_dropout': 0.252103,
            'lg_alpha': 1.63131,
            'ns_size1': 10,
            'ns_size2': 20,
            'gnn_outdim': 64,
            'sage_nhid': 64
        }
    }
}


dynamic = {
    'los':{
        'mpnn': {
            'batch_size': 32,
            'lr': 0.00077886,
            'l2': 9.61934e-05,
            'main_dropout': 0.466177,
            'gnn_outdim': 128,
            'mpnn_nhid': 512,
            'mpnn_step_mp': 3
        },
        'gcn': {
            'batch_size': 32,
            'lr': 0.000972068,
            'l2': 2.91297e-05,
            'main_dropout': 0.415498,
            'gnn_outdim': 64,
            'gcn_nhid': 128,
            'gcn_dropout': 0.214027
        },
        'gat': {
            'batch_size': 64,
            'lr': 0.000913728,
            'l2': 0.00010372,
            'main_dropout': 0.452033,
            'gnn_outdim': 128,
            'gat_nhid': 256,
            'gat_n_heads': 12,
            'gat_n_out_heads': 10,
            'gat_featdrop': 0.499333,
            'gat_attndrop': 0.340979
        }
    },
    'ihm': {
        'mpnn': {
            'batch_size': 128,
            'lr': 0.000598424,
            'l2': 2.12073e-05,
            'main_dropout': 0.211259,
            'gnn_outdim': 128,
            'mpnn_nhid': 256,
            'mpnn_step_mp': 2
        },
        'gcn': {
            'batch_size': 32,
            'lr': 0.000576053,
            'l2': 0.000174181,
            'main_dropout': 0.107752,
            'gnn_outdim': 256,
            'gcn_nhid': 64,
            'gcn_dropout': 0.476601
        },
        'gat': {
            'batch_size': 64,
            'lr': 0.000586093,
            'l2': 0.000243593,
            'main_dropout': 0.0106298,
            'gnn_outdim': 64,
            'gat_nhid': 64,
            'gat_n_heads': 6,
            'gat_n_out_heads': 8,
            'gat_featdrop': 0.292106,
            'gat_attndrop': 0.64364
        }
    }
}

lstm_no_diag = {
    'los':{
        'batch_size': 64,
        'lr': 0.000660119,
        'l2': 0.000938585,
        'main_dropout': 0.168568
    },
    'ihm': {
        'batch_size': 64,
        'lr': 0.000812688,
        'l2': 2.34691e-05,
        'main_dropout': 0.430714
    }
}

lstm_diag = {
    'los':{
        'batch_size': 32,
        'lr': 0.000960172,
        'l2': 3.3784e-05,
        'main_dropout': 0.134232
    },
    'ihm': {
        'batch_size': 256,
        'lr': 0.000610885,
        'l2': 9.8266e-05,
        'main_dropout': 0.0389121
    }
}

# NS GNN on g_version 2d


ns_gnn_default = {
    'los':{
        'gat': {
            'batch_size': 64,
            'lr': 0.001,
            'l2': 1e-4,
            'main_dropout': 0.1,
        },
        'sage':{
            'batch_size': 64,
            'lr': 0.001,
            'l2': 1e-4,
            'main_dropout': 0.1,
        },
        'mpnn':{
            'batch_size': 64,
            'lr': 1e-5,
            'l2': 1e-5,
            'main_dropout': 0.1,
        },
    },
    'ihm':{
        'gat':{ 
            'batch_size': 128,
            'lr': 0.000571719,
            'l2': 2.19849e-05,
            'main_dropout':  0.473152
        },
        'sage':{ 
            'batch_size': 128,
            'lr': 0.000501282,
            'l2': 6.62311e-05,
            'main_dropout': 0.235923
        }
    }
}

# Lstm embeddings
ns_gnn_4 = {
    'los':{
        'gat': { # from 2d
            'batch_size': 64,
            'lr': 0.000558217,
            'l2': 2.1417e-05,
            'main_dropout': 0.254591
        },
        'sage':{ # from 2d
            'batch_size': 64,
            'lr': 0.000662324,
            'l2': 5.9185e-05,
            'main_dropout':  0.411038,
        }
    },
    'ihm':{
        'gat':{ #  val loss 0.5002
            'batch_size': 128,
            'lr': 0.000811698,
            'l2': 0.000357263,
            'main_dropout':   0.380046,
            'lg_alpha':  1.95513,
            'ns_size1': 15,
            'ns_size2': 15,
            'gnn_outdim': 256,
            'gat_nhid': 128,
            'gat_n_heads': 8,
            'gat_n_out_heads': 10,
            'gat_featdrop': 0.300233,
            'gat_attndrop': 0.663338
        },
        'sage':{ # val loss 0.4997
            'batch_size': 128,
            'lr': 0.000501282,
            'l2': 6.62311e-05,
            'main_dropout': 0.235923,
            'lg_alph': 2.24069,
            'ns_size1': 20,
            'ns_size2': 20,
            'gnn_outdim': 64,
            'sage_nhid':129
        }
    }
}

mamba_default = {
    'ihm': {
        # Mamba model parameters
        'mamba_d_model': 128,         # hidden size
        'mamba_layers': 2,            # number of layers
        'mamba_dropout': 0.1,         # dropout rate
        'mamba_d_state': 16,           # state size
        'mamba_d_conv': 4,             # convolution kernel size
        'mamba_expand': 2,             # expansion factor
        'mamba_pooling': 'last',       # pooling method (last timestep)
        
        # Training parameters
        'batch_size': 256,             
        'lr': 1e-4,                    
        'l2': 1e-6,                   
        'epochs': 20,                  
        'clip_grad': 5.0,              
        'main_dropout': 0.1,           # general dropout
        'sch': 'plateau',              # learning rate scheduler
        'main_act_fn': 'leaky_relu',   # activation function
        'lg_alpha': 0.2,               # loss balancing alpha
    },
    'los': {
        'mamba_d_model': 256,
        'mamba_layers': 4,
        'mamba_dropout': 0.2,
        'mamba_d_state': 128,
        'mamba_d_conv': 8,
        'mamba_expand': 2,
        'mamba_pooling': 'mean',    
        
        'batch_size': 256,
        'lr': 5e-5,
        'l2': 1e-6,
        'epochs': 40,
        'clip_grad': 5.0,
        'main_dropout': 0.3,  
        'sch': 'plateau',              
        'main_act_fn': 'leaky_relu',   
        'lg_alpha': 0.5,  
    }
}

mamba_gps_default = {
    'ihm': {
        # Mamba parameters - increased dimensions for better representation
        'mamba_d_model': 128,        
        'mamba_layers': 2,            
        'mamba_dropout': 0.1,        
        'mamba_d_state': 16,         
        'mamba_d_conv': 4,           
        'mamba_expand': 2,           
        'mamba_pooling': 'last',     
        
        # GraphGPS parameters - improved settings
        'gps_hidden_dim': 256,       
        'gps_layers': 2,             
        'gps_dropout': 0.1,          
        'gps_act_fn': 'leaky_relu',  
        
        # Training parameters
        'dg_k': 3,                   
        'lr': 1e-4,                  
        'l2': 1e-6,                  
        'main_dropout': 0.1,        
        'lg_alpha': 0.2,         
        'clip_grad': 5.0,            
        'batch_size': 256,          
        'epochs': 20,                
        'sch': 'plateau',          
        
        # Activation functions
        'main_act_fn': 'leaky_relu', 
    },
    'los': {
        'mamba_d_model': 128,
        'mamba_layers': 4,
        'mamba_dropout': 0.1,
        'mamba_d_state': 64,
        'mamba_pooling': 'last',
        
        'gps_layers': 3,
        'gps_dropout': 0.1,
        
        'ns_size1': 15,   
        'ns_size2': 10,      
        'batch_size': 64,
        'lr': 3.97e-4,
        'l2': 1e-6,
        'main_dropout': 0.1,
        'main_act_fn': 'leaky_relu',
        'lg_alpha': 0.7,
        'clip_grad': 5.0,
        'sch': 'plateau',

    }
}

transformer_default = {
    'los': {
        'trans_indim': 256,
        'trans_hidden_dim': 256,
        'trans_num_heads': 4,
        'trans_layers': 4,
        'trans_ffn_dim': 256,
        'trans_dropout': 0.3,
        'trans_pooling': 'last',
        'max_seq_len': 336,
        'batchnorm': 'default',
        'final_act_fn': 'hardtanh',
        'batch_size': 128,
        'lr': 1e-5,
        'l2': 1e-4,
        'main_dropout': 0.2,
        'epochs': 30,
        'clip_grad': 2.0,
        'sch': 'plateau',
        'main_act_fn': 'gelu',
    },
    'ihm': {
        'trans_indim': 256,
        'trans_hidden_dim': 256,
        'trans_num_heads': 4,
        'trans_layers': 3,
        'trans_ffn_dim': 256,
        'trans_dropout': 0.1,
        'trans_pooling': 'mean',
        'max_seq_len': 336,
        'batchnorm': 'default',
        'final_act_fn': 'hardtanh',
        'batch_size': 128,
        'lr': 3e-4,
        'l2': 5e-5,
        'main_dropout': 0.15,
        'epochs': 30,
        'clip_grad': 2.0,
        'sch': 'plateau',
        'main_act_fn': 'gelu',
    }
}

