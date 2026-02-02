#!/usr/bin/env python3
"""
POC Model V2: Multi-conformational protein model with DRUG-CONDITIONED temporal attention
FIXED: Cross-attention (drug queries protein conformations)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class DrugEncoder(nn.Module):
    """GNN encoder for drug molecules."""
    
    def __init__(self, node_dim=78, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        return global_mean_pool(x, batch)


class ProteinEncoder(nn.Module):
    """GNN encoder for protein conformations."""
    
    def __init__(self, node_dim=1280, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        return global_mean_pool(x, batch)


class TemporalAttention(nn.Module):
    """
    Drug-conditioned cross-attention across protein conformations.
    FIXED: Drug embedding serves as query to select relevant conformation.
    """
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)  # Drug → query
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)    # Protein → key
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)  # Protein → value
        self.scale = hidden_dim ** 0.5
    
    def forward(self, drug_emb, conf_embeddings):
        """
        Args:
            drug_emb: (batch_size, hidden_dim) - drug embedding (QUERY)
            conf_embeddings: (batch_size, num_conformations, hidden_dim) - protein conformations (KEY/VALUE)
        Returns:
            attended: (batch_size, hidden_dim) - attended protein embedding
            attn_weights: (batch_size, num_conformations) - attention weights
        """
        # Project drug as query
        query = self.query_proj(drug_emb)  # (B, H)
        query = query.unsqueeze(1)  # (B, 1, H)
        
        # Project conformations as keys and values
        keys = self.key_proj(conf_embeddings)  # (B, N, H)
        values = self.value_proj(conf_embeddings)  # (B, N, H)
        
        # Compute attention scores: drug asks "which conformation fits me best?"
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale  # (B, 1, N)
        scores = scores.squeeze(1)  # (B, N)
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=1)  # (B, N)
        
        # Weighted sum of conformations
        attn_weights_expanded = attn_weights.unsqueeze(2)  # (B, N, 1)
        attended = torch.sum(values * attn_weights_expanded, dim=1)  # (B, H)
        
        return attended, attn_weights


class POCModelV2(nn.Module):
    """
    Phase 1 POC: Multi-conformational DTI prediction model
    FIXED: Drug-conditioned cross-attention
    """
    
    def __init__(
        self,
        drug_node_dim=78,
        protein_node_dim=1280,
        hidden_dim=128,
        num_layers=3,
        pretrained_drug_path=None,
        freeze_drug=False,
    ):
        super().__init__()
        
        self.drug_encoder = DrugEncoder(drug_node_dim, hidden_dim, num_layers)
        self.protein_encoder = ProteinEncoder(protein_node_dim, hidden_dim, num_layers)
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Prediction head (binary classification)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)  # Binary output (logits)
        )
        
        # Load pretrained drug encoder
        if pretrained_drug_path:
            try:
                state_dict = torch.load(pretrained_drug_path, map_location='cpu')
                self.drug_encoder.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded pretrained drug encoder from {pretrained_drug_path}")
            except Exception as e:
                print(f"⚠️ Could not load pretrained drug encoder: {e}")
        
        if freeze_drug:
            for param in self.drug_encoder.parameters():
                param.requires_grad = False
            print("✓ Drug encoder frozen")
    
    def forward(self, drug_batch, conf_graphs_batch):
        """
        Args:
            drug_batch: Already batched drug graphs (from collate_fn)
            conf_graphs_batch: List of batched conformation graphs (from collate_fn)
        Returns:
            logits: (batch_size,) binary classification logits
            attn_weights: (batch_size, num_conformations) attention weights
        """
        device = next(self.parameters()).device
        
        # Encode drugs (already batched)
        drug_batch = drug_batch.to(device)
        drug_emb = self.drug_encoder(drug_batch)  # (batch_size, hidden_dim)
        
        # Encode protein conformations
        conf_embeddings = []
        for conf_batch in conf_graphs_batch:
            conf_batch = conf_batch.to(device)
            conf_emb = self.protein_encoder(conf_batch)  # (batch_size, hidden_dim)
            conf_embeddings.append(conf_emb)
        
        # Stack conformations: (batch_size, num_conformations, hidden_dim)
        conf_embeddings = torch.stack(conf_embeddings, dim=1)
        
        # Drug-conditioned cross-attention across conformations
        # Drug asks: "Which conformation is most compatible with me?"
        protein_emb, attn_weights = self.temporal_attention(drug_emb, conf_embeddings)
        
        # Concatenate drug and protein embeddings
        combined = torch.cat([drug_emb, protein_emb], dim=1)  # (batch_size, 2*hidden_dim)
        
        # Predict
        logits = self.predictor(combined).squeeze(-1)  # (batch_size,)
        
        return logits, attn_weights
