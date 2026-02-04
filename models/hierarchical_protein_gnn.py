"""
Hierarchical Protein GNN - Multi-Scale Protein Structure Encoder

Inspired by:
- MGPPI (Frontiers 2024): Multiscale graph neural networks for PPI
- SSRGNet (arXiv 2025): Secondary structure relational GNN
- GGN-GO (OUP 2024): Geometric graph networks with multi-scale features
- HIGH-PPI (Nature Comms 2023): Hierarchical graph learning

Three-level encoding:
1. Residue-level: Amino acids with ESM-2 embeddings + spatial structure (GAT)
2. SSE-level: Secondary structure elements (helices, sheets, loops)
3. Domain-level: Functional domains (N-lobe, C-lobe, linkers)

Biology-aware design:
- Preserves secondary structure information (DSSP)
- Spatial clustering for domain detection
- Multi-scale edges (sequential + spatial contacts)
- ESM-2 integration for rich semantic features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
import numpy as np


class HierarchicalProteinGNN(nn.Module):
    """
    Hierarchical protein encoder for multi-conformational structures
    
    Architecture:
    Input: Hierarchical protein graph (residue → SSE → domain)
    Level 1: Residue-level GAT (1302-dim: 21 AA + 1280 ESM-2 + 1 B-factor)
    Level 2: SSE-level GAT (secondary structure elements)
    Level 3: Domain-level GCN (functional domains)
    Output: 256-dim protein embedding per conformation
    """
    
    def __init__(self,
                 residue_dim=1302,   # 21 (AA) + 1280 (ESM-2) + 1 (B-factor)
                 hidden_dim=128,
                 output_dim=256,
                 num_heads=4,
                 dropout=0.2):
        super().__init__()
        
        self.residue_dim = residue_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # ============================================================
        # Level 1: Residue-level encoding
        # ============================================================
        # Multi-head GAT for residue-residue interactions
        # Captures spatial contacts and sequential neighbors
        self.residue_gat1 = GATConv(
            residue_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=True
        )
        self.residue_gat2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False  # Average for final layer
        )
        
        self.residue_bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.residue_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # ============================================================
        # Level 2: SSE-level encoding
        # ============================================================
        # SSE features: pooled residue features + SS type + length
        sse_input_dim = hidden_dim + 3 + 1  # hidden + 3 (H/E/L one-hot) + 1 (length)
        
        self.sse_gat = GATConv(
            sse_input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False
        )
        self.sse_bn = nn.BatchNorm1d(hidden_dim)
        
        # ============================================================
        # Level 3: Domain-level encoding
        # ============================================================
        # Domain features: pooled SSE features
        self.domain_gcn = GCNConv(hidden_dim, output_dim)
        self.domain_bn = nn.BatchNorm1d(output_dim)
        
        # Readout: Multi-scale pooling
        # Combine domain-level and SSE-level for rich representation
        self.readout_transform = nn.Linear(output_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def pool_residues_to_sses(self, residue_features, sse_graph):
        """
        Pool residue features to SSE features
        
        Args:
            residue_features: [num_residues, hidden_dim]
            sse_graph: Data object with sse_groups attribute
        
        Returns:
            sse_features: [num_sses, sse_input_dim]
        """
        sse_features = []
        
        # SSE types for one-hot encoding
        ss_type_map = {'H': 0, 'E': 1, 'L': 2}
        
        num_residues = residue_features.shape[0]
        
        for idx, (ss_type, res_indices) in enumerate(sse_graph.sse_groups):
            # Pool residue features (mean)
            res_indices_tensor = torch.tensor(res_indices, dtype=torch.long, device=residue_features.device)
            pooled_residue_feat = residue_features[res_indices_tensor].mean(dim=0)  # [hidden_dim]
            
            # SS type one-hot
            ss_onehot = torch.zeros(3, device=residue_features.device)
            ss_onehot[ss_type_map.get(ss_type, 2)] = 1.0  # Default to Loop if unknown
            
            # SSE length (normalized)
            sse_length = torch.tensor([len(res_indices) / num_residues], 
                                     dtype=torch.float, 
                                     device=residue_features.device)
            
            # Concatenate: [hidden_dim + 3 + 1]
            sse_feat = torch.cat([pooled_residue_feat, ss_onehot, sse_length])
            sse_features.append(sse_feat)
        
        return torch.stack(sse_features)  # [num_sses, sse_input_dim]
    
    def pool_sses_to_domains(self, sse_features, domain_graph, sse_graph):
        """
        Pool SSE features to domain features
        
        Args:
            sse_features: [num_sses, hidden_dim]
            domain_graph: Data object with num_nodes
            sse_graph: Data object with sse_groups (for grouping SSEs)
        
        Returns:
            domain_features: [num_domains, hidden_dim]
        """
        num_domains = domain_graph.num_nodes
        num_sses = sse_features.shape[0]
        
        if num_domains == 0 or num_sses == 0:
            # Fallback: single domain
            return sse_features.mean(dim=0, keepdim=True)
        
        # Simple assignment: divide SSEs evenly across domains
        # (In practice, this comes from spatial clustering in builder)
        sses_per_domain = num_sses // num_domains
        
        domain_features = []
        for domain_idx in range(num_domains):
            start_sse = domain_idx * sses_per_domain
            end_sse = start_sse + sses_per_domain if domain_idx < num_domains - 1 else num_sses
            
            # Pool SSEs belonging to this domain
            domain_feat = sse_features[start_sse:end_sse].mean(dim=0)
            domain_features.append(domain_feat)
        
        return torch.stack(domain_features)  # [num_domains, hidden_dim]
    
    def forward(self, conformation_dict):
        """
        Forward pass: Residue → SSE → Domain
        
        Args:
            conformation_dict: Single conformation from hierarchical_proteins.pkl
                - level1_residues: Data (residue graph)
                - level2_secondary_structures: Data (SSE graph)
                - level3_domains: Data (domain graph)
        
        Returns:
            embedding: [output_dim] protein representation
        """
        # Extract levels
        res_graph = conformation_dict['level1_residues']
        sse_graph = conformation_dict['level2_secondary_structures']
        dom_graph = conformation_dict['level3_domains']
        
        # ============================================================
        # Level 1: Residue-level encoding
        # ============================================================
        x_res = res_graph.x.float()  # [num_residues, 1302]
        edge_index_res = res_graph.edge_index
        
        # GAT layer 1
        x_res = self.residue_gat1(x_res, edge_index_res)  # [num_residues, hidden_dim * num_heads]
        x_res = self.residue_bn1(x_res)
        x_res = self.relu(x_res)
        x_res = self.dropout(x_res)
        
        # GAT layer 2
        x_res = self.residue_gat2(x_res, edge_index_res)  # [num_residues, hidden_dim]
        x_res = self.residue_bn2(x_res)
        x_res = self.relu(x_res)
        x_res = self.dropout(x_res)
        
        # ============================================================
        # Level 2: SSE-level encoding
        # ============================================================
        # Pool residues → SSEs
        x_sse = self.pool_residues_to_sses(x_res, sse_graph)  # [num_sses, sse_input_dim]
        
        edge_index_sse = sse_graph.edge_index
        
        # GAT over SSE graph (if edges exist)
        if edge_index_sse.shape[1] > 0:
            x_sse = self.sse_gat(x_sse, edge_index_sse)  # [num_sses, hidden_dim]
            x_sse = self.sse_bn(x_sse)
            x_sse = self.relu(x_sse)
            x_sse = self.dropout(x_sse)
        else:
            # No SSE edges: use linear transform
            x_sse = self.sse_gat.lin_l(x_sse)  # [num_sses, hidden_dim]
            x_sse = self.sse_bn(x_sse)
            x_sse = self.relu(x_sse)
        
        # ============================================================
        # Level 3: Domain-level encoding
        # ============================================================
        # Pool SSEs → Domains
        x_dom = self.pool_sses_to_domains(x_sse, dom_graph, sse_graph)  # [num_domains, hidden_dim]
        
        edge_index_dom = dom_graph.edge_index
        
        # GCN over domain graph (if edges exist)
        if edge_index_dom.shape[1] > 0 and x_dom.shape[0] > 1:
            x_dom = self.domain_gcn(x_dom, edge_index_dom)  # [num_domains, output_dim]
            x_dom = self.domain_bn(x_dom)
            x_dom = self.relu(x_dom)
        else:
            # No domain edges or single domain: use linear transform
            x_dom = self.domain_gcn.lin(x_dom)  # [num_domains, output_dim]
            if x_dom.shape[0] > 1:  # Only apply BN if multiple domains
                x_dom = self.domain_bn(x_dom)
            x_dom = self.relu(x_dom)
        
        # ============================================================
        # Readout: Multi-scale global pooling
        # ============================================================
        # Pool at both SSE and domain levels for rich representation
        # SSE-level global features (captures secondary structure patterns)
        batch_sse = torch.zeros(x_sse.shape[0], dtype=torch.long, device=x_sse.device)
        sse_global = global_mean_pool(x_sse, batch_sse)  # [1, hidden_dim]
        
        # Project SSE global to output_dim
        sse_global_proj = nn.Linear(self.hidden_dim, self.output_dim, device=x_sse.device)(sse_global)
        
        # Domain-level global features (captures domain architecture)
        batch_dom = torch.zeros(x_dom.shape[0], dtype=torch.long, device=x_dom.device)
        dom_global_mean = global_mean_pool(x_dom, batch_dom)  # [1, output_dim]
        dom_global_sum = global_add_pool(x_dom, batch_dom)    # [1, output_dim]
        
        # Combine multi-scale features
        # SSE patterns + Domain mean + Domain sum
        x_global = torch.cat([sse_global_proj, dom_global_mean], dim=1)  # [1, output_dim * 2]
        
        # Final transform
        x_out = self.readout_transform(x_global)  # [1, output_dim]
        x_out = F.normalize(x_out, p=2, dim=1)    # L2 normalization
        
        return x_out.squeeze(0)  # [output_dim]


def test_hierarchical_protein_gnn():
    """Test the hierarchical protein GNN"""
    print("Testing HierarchicalProteinGNN...")
    
    import pickle
    from pathlib import Path
    
    # Try to load real hierarchical protein data
    hier_path = Path('data/processed/hierarchical_proteins.pkl')
    
    if hier_path.exists():
        print("✓ Found hierarchical proteins data")
        with open(hier_path, 'rb') as f:
            hier_proteins = pickle.load(f)
        
        # Get first protein, first conformation
        protein_id = list(hier_proteins.keys())[0]
        conformation = hier_proteins[protein_id][0]
        
        print(f"✓ Testing with protein: {protein_id}")
        print(f"  Residues: {conformation['level1_residues'].num_nodes}")
        print(f"  SSEs: {conformation['level2_secondary_structures'].num_nodes}")
        print(f"  Domains: {conformation['level3_domains'].num_nodes}")
        
        # Initialize model
        model = HierarchicalProteinGNN(
            residue_dim=1302,
            hidden_dim=128,
            output_dim=256,
            num_heads=4
        )
        
        # Forward pass
        embedding = model(conformation)
        
        print(f"✓ Output embedding shape: {embedding.shape}")
        print(f"✓ Expected shape: [256]")
        
        assert embedding.shape == (256,), f"Output shape mismatch: {embedding.shape}"
        print("✓ HierarchicalProteinGNN test passed!")
        
    else:
        print("⚠️  hierarchical_proteins.pkl not found")
        print("   Run: python scripts/build_hierarchical_proteins.py")
        print("   Creating dummy test instead...")
        
        # Create dummy hierarchical structure
        num_residues = 100
        num_sses = 10
        num_domains = 3
        
        # Level 1: Residues
        x_res = torch.randn(num_residues, 1302)
        edge_list_res = [[i, i+1] for i in range(num_residues-1)] + [[i+1, i] for i in range(num_residues-1)]
        edge_index_res = torch.tensor(edge_list_res, dtype=torch.long).t()
        res_graph = Data(x=x_res, edge_index=edge_index_res, num_nodes=num_residues)
        
        # Level 2: SSEs
        sse_groups = [((['H', 'E', 'L'][i % 3], list(range(i*10, (i+1)*10)))) for i in range(num_sses)]
        x_sse = torch.randn(num_sses, 132)  # Placeholder
        edge_list_sse = [[i, i+1] for i in range(num_sses-1)] + [[i+1, i] for i in range(num_sses-1)]
        edge_index_sse = torch.tensor(edge_list_sse, dtype=torch.long).t()
        sse_graph = Data(x=x_sse, edge_index=edge_index_sse, num_nodes=num_sses, sse_groups=sse_groups, sse_types=['H', 'E', 'L']*4)
        
        # Level 3: Domains
        x_dom = torch.randn(num_domains, 128)
        edge_list_dom = [[0, 1], [1, 0], [1, 2], [2, 1]]
        edge_index_dom = torch.tensor(edge_list_dom, dtype=torch.long).t()
        dom_graph = Data(x=x_dom, edge_index=edge_index_dom, num_nodes=num_domains)
        
        conformation = {
            'level1_residues': res_graph,
            'level2_secondary_structures': sse_graph,
            'level3_domains': dom_graph
        }
        
        # Initialize model
        model = HierarchicalProteinGNN(
            residue_dim=1302,
            hidden_dim=128,
            output_dim=256,
            num_heads=4
        )
        
        # Forward pass
        embedding = model(conformation)
        
        print(f"✓ Output embedding shape: {embedding.shape}")
        print(f"✓ Expected shape: [256]")
        
        assert embedding.shape == (256,), f"Output shape mismatch: {embedding.shape}"
        print("✓ HierarchicalProteinGNN test passed (dummy data)!")


if __name__ == '__main__':
    test_hierarchical_protein_gnn()
