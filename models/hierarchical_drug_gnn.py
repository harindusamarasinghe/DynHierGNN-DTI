"""
Hierarchical Drug GNN - Multi-Scale Molecular Graph Encoder

Inspired by:
- HiMol (Nature Communications 2023): Node-motif-graph hierarchical learning
- Group Graph (PMC 2024): Substructure-level molecular representation
- MGSSL (ICLR 2024): Multi-scale graph self-supervised learning

Three-level encoding:
1. Atom-level: Individual atoms with chemical features (GAT for attention)
2. Motif-level: Functional groups (rings, chains, key pharmacophores)
3. Scaffold-level: Global molecular structure (GCN for global pooling)

Chemistry-aware design:
- Motif detection using RDKit BRICS/functional groups
- Preserve chemical semantics (aromatic rings, carboxyl groups, etc.)
- Multi-head attention for atom-atom interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np


class HierarchicalDrugGNN(nn.Module):
    """
    Hierarchical molecular encoder with chemistry-aware motif detection
    
    Architecture:
    Input: Molecular graph (atoms, bonds)
    Level 1: Atom-level GAT (multi-head attention)
    Level 2: Motif pooling (functional groups)
    Level 3: Scaffold GCN (global structure)
    Output: 256-dim molecular embedding
    """
    
    def __init__(self, 
                 atom_dim=9,        # 9-dim one-hot (C, N, O, S, F, Cl, Br, I, others)
                 hidden_dim=128, 
                 output_dim=256,
                 num_heads=4,       # Multi-head attention
                 dropout=0.2):
        super().__init__()
        
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Level 1: Atom-level with multi-head GAT
        # GAT learns importance of neighboring atoms (chemical bonds)
        self.atom_gat1 = GATConv(
            atom_dim, 
            hidden_dim, 
            heads=num_heads, 
            dropout=dropout,
            add_self_loops=True
        )
        self.atom_gat2 = GATConv(
            hidden_dim * num_heads,  # Concatenated from multi-head
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False  # Average instead of concat for final layer
        )
        
        self.atom_bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.atom_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Level 2: Motif-level features
        # After pooling atoms → motifs, use GCN for motif-motif interactions
        self.motif_gcn = GCNConv(hidden_dim, hidden_dim)
        self.motif_bn = nn.BatchNorm1d(hidden_dim)
        
        # Level 3: Scaffold-level (global graph)
        # Final GCN for whole-molecule representation
        self.scaffold_gcn = GCNConv(hidden_dim, output_dim)
        self.scaffold_bn = nn.BatchNorm1d(output_dim)
        
        # Readout: Graph-level pooling
        # Using both mean and sum for richer representation
        self.readout_transform = nn.Linear(output_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def detect_motifs_chemistry_aware(self, mol, atom_features):
        """
        Chemistry-aware motif detection using RDKit
        Preserves chemical semantics (aromatic rings, functional groups)
        
        Returns:
        - motif_groups: List of atom indices for each motif
        - motif_edges: Edges between motifs (if atoms from different motifs are bonded)
        """
        if mol is None:
            # Fallback: treat entire molecule as one motif
            num_atoms = atom_features.shape[0]
            return [list(range(num_atoms))], []
        
        motif_groups = []
        
        # 1. Detect aromatic rings (benzene, pyridine, etc.)
        ring_info = mol.GetRingInfo()
        aromatic_rings = []
        for ring in ring_info.AtomRings():
            if len(ring) >= 5:  # 5+ member rings
                # Check if aromatic
                is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
                if is_aromatic:
                    aromatic_rings.append(list(ring))
        
        motif_groups.extend(aromatic_rings)
        
        # 2. Detect common functional groups using SMARTS patterns
        # Carboxyl, hydroxyl, amine, amide, ester, etc.
        functional_groups = {
            'carboxyl': 'C(=O)O',
            'amine': 'N',
            'hydroxyl': 'O',
            'carbonyl': 'C=O',
            'ester': 'C(=O)O',
            'amide': 'C(=O)N',
            'sulfone': 'S(=O)(=O)',
            'phosphate': 'P(=O)(O)(O)'
        }
        
        used_atoms = set()
        for ring in aromatic_rings:
            used_atoms.update(ring)
        
        for group_name, smarts in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    # Skip if atoms already in aromatic ring
                    if not any(atom_idx in used_atoms for atom_idx in match):
                        motif_groups.append(list(match))
                        used_atoms.update(match)
        
        # 3. Remaining atoms: group by simple chains
        num_atoms = mol.GetNumAtoms()
        remaining_atoms = [i for i in range(num_atoms) if i not in used_atoms]
        
        if remaining_atoms:
            # Simple heuristic: group atoms within 2 bonds
            visited = set()
            for atom_idx in remaining_atoms:
                if atom_idx not in visited:
                    chain = [atom_idx]
                    visited.add(atom_idx)
                    # Add neighbors within 2 bonds
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        if neighbor_idx in remaining_atoms and neighbor_idx not in visited:
                            chain.append(neighbor_idx)
                            visited.add(neighbor_idx)
                    if chain:
                        motif_groups.append(chain)
        
        # If no motifs detected, treat whole molecule as one motif
        if not motif_groups:
            motif_groups = [list(range(num_atoms))]
        
        # Build motif-motif edges
        # Two motifs are connected if any atoms between them are bonded
        motif_edges = []
        for i, motif1 in enumerate(motif_groups):
            for j, motif2 in enumerate(motif_groups):
                if i < j:  # Avoid duplicates
                    # Check if any bond between motif1 and motif2
                    connected = False
                    for atom1_idx in motif1:
                        for atom2_idx in motif2:
                            if mol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
                                connected = True
                                break
                        if connected:
                            break
                    if connected:
                        motif_edges.append([i, j])
                        motif_edges.append([j, i])  # Undirected
        
        return motif_groups, motif_edges
    
    def forward(self, drug_graph, mol=None):
        """
        Forward pass: Atom → Motif → Scaffold
        
        Args:
            drug_graph: torch_geometric.Data with x, edge_index, batch
            mol: RDKit Mol object (optional, for motif detection)
        
        Returns:
            embedding: [batch_size, output_dim] molecular representation
        """
        x = drug_graph.x.float()
        edge_index = drug_graph.edge_index
        batch = drug_graph.batch if hasattr(drug_graph, 'batch') else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # ============================================================
        # Level 1: Atom-level encoding with GAT
        # ============================================================
        # Multi-head attention: learn which atoms are important
        # (e.g., heteroatoms like N, O in binding sites)
        x_atom = self.atom_gat1(x, edge_index)  # [num_atoms, hidden_dim * num_heads]
        x_atom = self.atom_bn1(x_atom)
        x_atom = self.relu(x_atom)
        x_atom = self.dropout(x_atom)
        
        x_atom = self.atom_gat2(x_atom, edge_index)  # [num_atoms, hidden_dim]
        x_atom = self.atom_bn2(x_atom)
        x_atom = self.relu(x_atom)
        x_atom = self.dropout(x_atom)
        
        # ============================================================
        # Level 2: Motif-level encoding
        # ============================================================
        # Pool atoms → motifs (functional groups, rings)
        if mol is not None:
            motif_groups, motif_edges = self.detect_motifs_chemistry_aware(mol, x_atom)
        else:
            # Fallback: simple spatial clustering if RDKit mol not available
            # Use k-means or just treat all as one motif
            num_atoms = x_atom.shape[0]
            motif_groups = [list(range(num_atoms))]
            motif_edges = []
        
        # Aggregate atom features to motif features (mean pooling)
        motif_features = []
        for motif_group in motif_groups:
            motif_atom_indices = torch.tensor(motif_group, dtype=torch.long, device=x_atom.device)
            motif_feat = x_atom[motif_atom_indices].mean(dim=0)  # [hidden_dim]
            motif_features.append(motif_feat)
        
        x_motif = torch.stack(motif_features) if motif_features else x_atom.mean(dim=0, keepdim=True)  # [num_motifs, hidden_dim]
        
        # Build motif graph edges
        if motif_edges:
            motif_edge_index = torch.tensor(motif_edges, dtype=torch.long, device=x_motif.device).t()
        else:
            # Fully connected if no edges detected
            num_motifs = x_motif.shape[0]
            if num_motifs > 1:
                motif_edge_index = torch.combinations(torch.arange(num_motifs, device=x_motif.device), r=2).t()
                # Make undirected
                motif_edge_index = torch.cat([motif_edge_index, motif_edge_index.flip(0)], dim=1)
            else:
                motif_edge_index = torch.empty((2, 0), dtype=torch.long, device=x_motif.device)
        
        # GCN over motif graph
        if motif_edge_index.shape[1] > 0:
            x_motif = self.motif_gcn(x_motif, motif_edge_index)
            x_motif = self.motif_bn(x_motif)
            x_motif = self.relu(x_motif)
            x_motif = self.dropout(x_motif)
        
        # ============================================================
        # Level 3: Scaffold-level encoding
        # ============================================================
        # Map motif features back to atoms (for final GCN)
        # Each atom inherits its motif's feature
        atom_to_motif = {}
        for motif_idx, motif_group in enumerate(motif_groups):
            for atom_idx in motif_group:
                atom_to_motif[atom_idx] = motif_idx
        
        x_scaffold = []
        for atom_idx in range(x_atom.shape[0]):
            motif_idx = atom_to_motif.get(atom_idx, 0)
            x_scaffold.append(x_motif[motif_idx])
        x_scaffold = torch.stack(x_scaffold)  # [num_atoms, hidden_dim]
        
        # Final GCN over original atom graph (now with motif-enriched features)
        x_scaffold = self.scaffold_gcn(x_scaffold, edge_index)  # [num_atoms, output_dim]
        x_scaffold = self.scaffold_bn(x_scaffold)
        x_scaffold = self.relu(x_scaffold)
        
        # ============================================================
        # Readout: Global pooling
        # ============================================================
        # Combine mean and sum pooling for robust representation
        x_mean = global_mean_pool(x_scaffold, batch)  # [batch_size, output_dim]
        x_sum = global_add_pool(x_scaffold, batch)    # [batch_size, output_dim]
        x_global = torch.cat([x_mean, x_sum], dim=1)  # [batch_size, output_dim * 2]
        
        # Final transform
        x_out = self.readout_transform(x_global)  # [batch_size, output_dim]
        x_out = F.normalize(x_out, p=2, dim=1)    # L2 normalization for stability
        
        return x_out


def test_hierarchical_drug_gnn():
    """Test the hierarchical drug GNN"""
    print("Testing HierarchicalDrugGNN...")
    
    # Create dummy molecular graph
    num_atoms = 20
    x = torch.randn(num_atoms, 9)  # 9-dim atom features
    
    # Create edges (molecular bonds)
    edge_list = []
    for i in range(num_atoms - 1):
        edge_list.append([i, i+1])
        edge_list.append([i+1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    # Create batch (for graph batching)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    
    # Create Data object
    drug_graph = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Initialize model
    model = HierarchicalDrugGNN(
        atom_dim=9,
        hidden_dim=128,
        output_dim=256,
        num_heads=4
    )
    
    # Forward pass
    embedding = model(drug_graph, mol=None)
    
    print(f"✓ Input atoms: {num_atoms}")
    print(f"✓ Output embedding shape: {embedding.shape}")
    print(f"✓ Expected shape: [1, 256]")
    
    assert embedding.shape == (1, 256), f"Output shape mismatch: {embedding.shape}"
    print("✓ HierarchicalDrugGNN test passed!")


if __name__ == '__main__':
    test_hierarchical_drug_gnn()
