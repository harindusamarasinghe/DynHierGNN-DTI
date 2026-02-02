#!/usr/bin/env python3
"""
Static POC Model: Baseline using only first conformation
FIXED: Accepts pre-batched data from collate_fn
"""

import torch
import torch.nn as nn
from models.poc_model_v2 import DrugEncoder, ProteinEncoder


class StaticPOCModel(nn.Module):
    """
    Static baseline: Uses only the first protein conformation
    FIXED: Accepts pre-batched data
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
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        # Load pretrained drug encoder
        if pretrained_drug_path:
            try:
                state_dict = torch.load(pretrained_drug_path, map_location='cpu')
                self.drug_encoder.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded pretrained drug encoder (static)")
            except Exception as e:
                print(f"⚠️  Could not load pretrained: {e}")
        
        if freeze_drug:
            for param in self.drug_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, drug_batch, conf_graphs_batch):
        """
        Args:
            drug_batch: Already batched drug graphs
            conf_graphs_batch: List of batched conformations (use only first)
        
        Returns:
            logits: (batch_size,) binary classification logits
        """
        device = next(self.parameters()).device
        
        # Encode drugs
        drug_batch = drug_batch.to(device)
        drug_emb = self.drug_encoder(drug_batch)
        
        # Encode ONLY first conformation (static baseline)
        first_conf_batch = conf_graphs_batch[0].to(device)
        protein_emb = self.protein_encoder(first_conf_batch)
        
        # Concatenate and predict
        combined = torch.cat([drug_emb, protein_emb], dim=1)
        logits = self.predictor(combined).squeeze(-1)
        
        return logits
