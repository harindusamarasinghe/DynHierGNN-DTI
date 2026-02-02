"""
Drug GNN Encoder (will load pretrained weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class DrugGCN(nn.Module):
    def __init__(self, in_channels=9, hidden_dim=256, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        x = global_mean_pool(x, batch)
        x = F.normalize(x, p=2, dim=1)
        return x

def load_pretrained_drug_encoder(checkpoint_path='checkpoints/pretrained/drug_encoder_pretrained.pth'):
    """Load pretrained drug encoder"""
    model = DrugGCN()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"✓ Loaded pretrained drug encoder from {checkpoint_path}")
    return model
