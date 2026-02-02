"""
Temporal Attention over Protein Conformations (CORE NOVELTY)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, drug_dim=128, protein_dim=128):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        
        # Attention projection (optional, can use direct dot product)
        self.query_proj = nn.Linear(drug_dim, protein_dim)
    
    def forward(self, drug_emb, protein_conf_embs):
        """
        Args:
            drug_emb: [batch_size, drug_dim]
            protein_conf_embs: [batch_size, num_conformations, protein_dim]
        
        Returns:
            attended_protein: [batch_size, protein_dim]
            attention_weights: [batch_size, num_conformations]
        """
        # Project drug as query
        query = self.query_proj(drug_emb)  # [batch_size, protein_dim]
        query = query.unsqueeze(1)  # [batch_size, 1, protein_dim]
        
        # Compute attention scores
        scores = torch.bmm(query, protein_conf_embs.transpose(1, 2))  # [batch_size, 1, num_conformations]
        scores = scores.squeeze(1)  # [batch_size, num_conformations]
        
        # Scale by sqrt(dim)
        scores = scores / (self.protein_dim ** 0.5)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, num_conformations]
        
        # Weighted sum
        attention_weights_expanded = attention_weights.unsqueeze(2)  # [batch_size, num_conformations, 1]
        attended_protein = torch.sum(protein_conf_embs * attention_weights_expanded, dim=1)  # [batch_size, protein_dim]
        
        return attended_protein, attention_weights
