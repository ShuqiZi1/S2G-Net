"""
Defining Dynamic Graph Transformer models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from src.models.utils import init_weights, get_act_fn
from torch.nn import MultiheadAttention


def define_dynamic_graph_transformer(name='dynamic_graph_transformer'):
    """Returns the Dynamic Graph Transformer class based on name"""
    return DynamicGraphTransformer

class DynamicGraphTransformerLayer(nn.Module):
    """
    Single Dynamic Graph Transformer Layer
    """
    def __init__(self, config):
        super().__init__()
        
        # Parameters
        self.dim = config.get('transformer_hidden_dim', 128)
        self.num_heads = config.get('transformer_num_heads', 8)
        self.out_channels = self.dim // self.num_heads  # 16
        self.edge_dim = self.dim
        self.dropout = config.get('transformer_dropout', 0.1)
        self.use_edge_attr = config.get('use_edge_attr', False)


        # Multi-head self-attention
        self.self_attn = MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Graph-specific transformer conv (provides message passing along edges)
        self.graph_conv = TransformerConv(
            in_channels=self.dim,
            out_channels=self.dim,
            heads=self.num_heads,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
            concat=False,
            beta=True  # Enable dynamic attention
        )
        self.graph_proj = nn.Linear(self.dim, self.dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, config.get('transformer_ffn_dim', 2048)),
            get_act_fn(config.get('transformer_act_fn', 'gelu')),
            nn.Dropout(self.dropout),
            nn.Linear(config.get('transformer_ffn_dim', 2048), self.dim),
            nn.Dropout(self.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.norm3 = nn.LayerNorm(self.dim)
        
        # Additional projection for dynamic graph updates
        self.dynamic_update = nn.Linear(self.dim * 2, self.dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        init_weights(self.modules())
        
    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, return_attention=False):
        """
        x: node features [num_nodes, dim]
        edge_index: graph connectivity [2, num_edges]
        edge_weight: edge weights [num_edges]
        edge_attr: edge attributes [num_edges, edge_dim]
        """
        # Save original input for residual connection
        residual = x
        
        # Layer norm before self-attention
        x = self.norm1(x)
        
        # Convert graph data to dense format for global self-attention
        batch_size = 1  # Assuming single graph
        num_nodes = x.size(0)
        x_dense = x.unsqueeze(0)  # [1, num_nodes, dim]
        
        # Create adjacency mask from edge_index
        adj_mask = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device, dtype=torch.bool)
        if edge_index is not None:
            adj_mask[0, edge_index[0], edge_index[1]] = True
        
        # Instead of using the complex mask logic, we'll use a simpler approach
        # by just using the built-in attention without a mask
        attn_out, attn_weights = self.self_attn(
            query=x_dense,
            key=x_dense,
            value=x_dense,
            need_weights=return_attention
        )
        
        # Apply the graph structure mask to the attention weights if needed
        if return_attention and edge_index is not None:
            # Create a sparse mask (only connections in the graph)
            sparse_mask = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=torch.bool)
            sparse_mask[edge_index[0], edge_index[1]] = True
            
            # Apply the mask to attention weights (for visualization purposes)
            masked_attn_weights = attn_weights.clone()
            masked_attn_weights = masked_attn_weights.masked_fill(~sparse_mask.unsqueeze(0), 0.0)
            attn_weights = masked_attn_weights
        
        attn_out = attn_out.squeeze(0)  # [num_nodes, dim]
        
        # Add residual connection
        x = residual + attn_out
        
        # Layer norm before graph conv
        x_norm = self.norm2(x)
        
        # Apply graph-specific transformer conv with edge information
        graph_conv_output = self.graph_conv(
            x=x_norm, 
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention_weights=return_attention
        )
        
        # Handle the tuple return value from graph_conv
        if return_attention:
            if isinstance(graph_conv_output, tuple):
                graph_out, graph_attn_weights = graph_conv_output
            else:
                graph_out = graph_conv_output
                graph_attn_weights = None
        else:
            # If we're not returning attention weights but still got a tuple
            if isinstance(graph_conv_output, tuple):
                graph_out = graph_conv_output[0]  # Just take the first element (the node features)
            else:
                graph_out = graph_conv_output
        
        # Ensure dimensions match for residual connection
        if graph_out.size(-1) != x.size(-1):
            graph_out = self.graph_proj(graph_out)
        
        # Add residual connection
        x = x + graph_out
        
        # Layer norm before FFN
        x_norm = self.norm3(x)
        
        # Apply feed-forward network
        ffn_out = self.ffn(x_norm)
        
        # Add residual connection
        x = x + ffn_out
        
        # Calculate dynamic graph updates based on node features
        # This helps model temporal changes in the graph
        dynamic_features = torch.cat([x, residual], dim=-1)
        dynamic_update = torch.sigmoid(self.dynamic_update(dynamic_features))
        
        # Apply dynamic update
        x = x * dynamic_update + residual * (1 - dynamic_update)
        
        if return_attention:
            return x, attn_weights, graph_attn_weights
        return x
        
class DynamicGraphTransformer(nn.Module):
    """
    Dynamic Graph Transformer encoder
    """
    def __init__(self, config):
        super().__init__()
        
        # Parameters
        self.dim = config.get('transformer_hidden_dim', 512)
        self.input_dim = config.get('transformer_indim', 128)  
        self.num_layers = config.get('transformer_num_layers', 4)
        self.dropout = config.get('transformer_dropout', 0.1)
        
        # Input projection with correct dimensions
        self.input_proj = nn.Linear(self.input_dim, self.dim)
        
        # Positional encoding (optional)
        self.use_pos_encoding = config.get('use_pos_encoding', True)
        if self.use_pos_encoding:
            self.pos_encoder = nn.Linear(1, self.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DynamicGraphTransformerLayer(config) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.dim, config.get('out_dim', 512))
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._initialize_weights()
        
        self.use_checkpoint = True
        
    def _initialize_weights(self):
        init_weights(self.modules())

    
    def forward(self, x, flat, adjs, edge_weight, last, edge_attr=None):
        """
        x: node features from RWKV [num_nodes, input_dim]
        flat: flat_id for each node [num_nodes]
        adjs: list of adjacency structures for each layer
        edge_weight: edge weights
        last: pooled RWKV outputs [batch_size, input_dim]
        edge_attr: edge attributes (optional)
        """
         
        # Project input to transformer dimension
        x = self.input_proj(x)
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            # Create positional indices
            pos = torch.arange(0, x.size(0), device=x.device).unsqueeze(-1).float()
            pos_encoding = self.pos_encoder(pos)
            x = x + pos_encoding
            
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Process through transformer layers with dynamic graph updates
        attn_weights_list = []
        
        # Process each layer with the corresponding adjacency structure
        for i, (layer, adj) in enumerate(zip(self.layers, adjs)):
            edge_index, e_id, size = adj
            
            # Get correct edge attributes for this layer if provided
            curr_edge_attr = None
            if edge_attr is not None:
                curr_edge_attr = edge_attr[e_id]
                
            # Apply transformer layer
            if i == self.num_layers - 1:  # Last layer, return attention weights
                x, attn_weights, graph_attn = layer(
                    x, edge_index, edge_weight, curr_edge_attr, return_attention=True
                )
                attn_weights_list.append((attn_weights, graph_attn))
            else:
                x = layer(x, edge_index, edge_weight, curr_edge_attr)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Project to output dimension
        x = self.output_proj(x)
        
        # Select nodes corresponding to the main graph (not sampled neighbors)
        x = x[:flat.size(0)]
        
        return x
    
    def inference(self, x, flat, subgraph_loader, device, edge_weight=None, last=None, get_emb=False, edge_attr=None):
        """
        Inference method for large graphs using neighborhood sampling
        """
        x = self.input_proj(x.to(device))
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            pos = torch.arange(0, x.size(0), device=device).unsqueeze(-1).float()
            pos_encoding = self.pos_encoder(pos)
            x = x + pos_encoding
            
        x = self.dropout_layer(x)
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            xs = []
            
            # Process in batches using subgraph loader
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj
                
                # Get edge attributes for this batch if provided
                batch_edge_attr = None
                if edge_attr is not None:
                    # Select edge attributes corresponding to this batch's edges
                    # Implementation depends on your specific use of edge attributes
                    batch_edge_attr = edge_attr
                    
                # Get batch x
                x_batch = x[n_id]
                
                # Apply transformer layer
                x_batch = layer(x_batch, edge_index, edge_weight, batch_edge_attr)
                
                # Store processed nodes
                xs.append(x_batch[:batch_size])
                
            # Update node features for this layer
            x = torch.cat(xs, dim=0)
        
        # Final normalization
        x = self.norm(x)
        
        # Return embeddings if requested
        if get_emb:
            return x
            
        # Output projection
        x = self.output_proj(x)
        
        return x
        
    def inference_whole(self, x, flat, device, edge_weight, edge_index, last, get_attn=False, edge_attr=None):
        """
        Inference on the entire graph at once (for smaller graphs)
        """
        x = x.to(device)
        edge_index = edge_index.to(device)
        
        # Process node features
        x = self.input_proj(x)
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            pos = torch.arange(0, x.size(0), device=device).unsqueeze(-1).float()
            pos_encoding = self.pos_encoder(pos)
            x = x + pos_encoding
            
        x = self.dropout_layer(x)
        
        # Store attention weights if requested
        all_edge_attn = []
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1 and get_attn:  # Last layer, get attention weights
                x, attn_weights, graph_attn = layer(
                    x, edge_index, edge_weight, edge_attr, return_attention=True
                )
                all_edge_attn.append(graph_attn)
            else:
                x = layer(x, edge_index, edge_weight, edge_attr)
        
        # Final normalization
        x = self.norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        if get_attn:
            return x, edge_index, all_edge_attn
        else:
            return x