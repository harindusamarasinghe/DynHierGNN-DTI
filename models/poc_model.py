"""
POC Model: DrugGCN + ProteinGCN + TemporalAttention + MLP head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.drug_gnn import DrugGCN
from .protein_gnn import ProteinGCN
from models.temporal_attention import TemporalAttention


class POCModel(nn.Module):
    def __init__(self, pretrained_drug_path=None, freeze_drug=False):
        super().__init__()

        self.drug_encoder = DrugGCN()
        if pretrained_drug_path is not None:
            state = torch.load(pretrained_drug_path, map_location="cpu")
            self.drug_encoder.load_state_dict(state)
            print(f"✓ Loaded pretrained drug encoder from {pretrained_drug_path}")
            if freeze_drug:
                for p in self.drug_encoder.parameters():
                    p.requires_grad = False
                print("  Drug encoder frozen")

        self.protein_encoder = ProteinGCN()
        self.attention = TemporalAttention()

        # MLP head: 256 -> 128 -> 64 -> 1
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def encode_drug_batch(self, drug_graphs):
        """drug_graphs: list of PyG Data; returns [B, 128]"""
        batch = Batch.from_data_list(drug_graphs)
        return self.drug_encoder(batch)

    def encode_protein_batch(self, conf_graphs_batch):
        """
        conf_graphs_batch: list of [g1, g2, g3] per sample
        Returns: [B, 3, 128]
        """
        # For each conformation index 0,1,2, make a Batch
        num_conf = len(conf_graphs_batch[0])
        conf_embs = []
        for k in range(num_conf):
            graphs_k = [sample[k] for sample in conf_graphs_batch]
            batch_k = Batch.from_data_list(graphs_k)
            emb_k = self.protein_encoder(batch_k)  # [B, 128]
            conf_embs.append(emb_k)

        # Stack to [B, num_conf, 128]
        protein_conf_embs = torch.stack(conf_embs, dim=1)
        return protein_conf_embs

    def forward(self, drug_graphs, conf_graphs_batch):
        """
        drug_graphs: list[Data] length B
        conf_graphs_batch: list[list[Data]] length B x 3
        """
        drug_emb = self.encode_drug_batch(drug_graphs)        # [B,128]
        protein_conf_embs = self.encode_protein_batch(conf_graphs_batch)  # [B,3,128]

        attended_protein, attn_weights = self.attention(drug_emb, protein_conf_embs)

        z = torch.cat([drug_emb, attended_protein], dim=1)  # [B,256]
        y_pred = self.predictor(z).squeeze(1)               # [B]

        return y_pred, attn_weights
