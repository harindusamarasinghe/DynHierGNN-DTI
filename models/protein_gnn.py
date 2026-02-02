"""
Protein GNN Encoder (processes multiple conformations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

class ProteinGCN(nn.Module):
    def __init__(self, in_channels=1300, hidden_dim=256, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        """
        Input: data can be a single graph or batch
        Output: protein embedding [batch_size, out_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        x = global_mean_pool(x, batch)
        return x
    
    def forward_multi_conf(self, conf_graphs_list):
        """
        Process multiple conformations
        Input: list of lists [[conf1, conf2, conf3], [conf1, conf2, conf3], ...]
        Output: [batch_size, num_conformations, out_dim]
        """
        batch_size = len(conf_graphs_list)
        num_conformations = len(conf_graphs_list[0])
        
        all_embeddings = []
        
        for i in range(batch_size):
            conf_embeddings = []
            for j in range(num_conformations):
                # Process single conformation
                graph = conf_graphs_list[i][j]
                emb = self.forward(graph)
                conf_embeddings.append(emb)
            
            # Stack: [num_conformations, out_dim]
            conf_embeddings = torch.stack(conf_embeddings, dim=0)
            all_embeddings.append(conf_embeddings)
        
        # Stack: [batch_size, num_conformations, out_dim]
        return torch.stack(all_embeddings, dim=0)
