# scripts/build_hierarchical_drugs.py

"""
Hierarchical Drug Graph Builder - Chemistry-Informed

Inspired by:
- H2GNN-DTI: Two-level hierarchical heterogeneous graph learning
- HiMol: Hierarchical molecular graph with motif structures

Three levels:
1. Atom-level: Atomic graph (bonds as edges)
2. Motif-level: Functional groups / chemical motifs (pharmacophores)
3. Molecule-level: Global molecular scaffold

Chemistry rules from HiMol paper:
- Use BRICS decomposition for chemically meaningful motifs
- Preserve functional groups (carboxyl, amine, aromatic rings, etc.)
- Capture pharmacophore patterns
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import BRICS, Fragments, Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from collections import defaultdict

class ChemistryInformedDrugBuilder:
    """
    Build 3-level drug hierarchy with chemistry-aware decomposition
    Following H2GNN-DTI and HiMol design principles
    """
    
    def __init__(self):
        # Atom feature dimensions (match your current drug graphs)
        self.atom_feature_dim = 78  # Your existing atom features
        
    def get_atom_features(self, atom):
        """Extract atom features (same as your existing drug graph builder)"""
        # This should match your existing build_phase1_drug_graphs.py
        # I'll provide simplified version - replace with your exact features
        
        features = []
        
        # One-hot encoding for atom type (C, N, O, S, F, etc.)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Other']
        atom_type = atom.GetSymbol()
        features.extend([1 if atom_type == t else 0 for t in atom_types])
        
        # Degree (number of bonds)
        features.extend([1 if atom.GetDegree() == i else 0 for i in range(6)])
        
        # Hybridization
        hybrid_types = [Chem.rdchem.HybridizationType.SP,
                       Chem.rdchem.HybridizationType.SP2,
                       Chem.rdchem.HybridizationType.SP3]
        features.extend([1 if atom.GetHybridization() == h else 0 for h in hybrid_types])
        
        # Other properties
        features.append(atom.GetIsAromatic())
        features.append(atom.GetFormalCharge())
        features.append(atom.GetNumRadicalElectrons())
        features.append(atom.GetNumExplicitHs())
        features.append(atom.GetTotalNumHs())
        
        return features
    
    def build_level1_atom_graph(self, mol):
        """
        Level 1: Atom graph
        Same as your existing drug graphs - atoms as nodes, bonds as edges
        """
        num_atoms = mol.GetNumAtoms()
        
        # Node features: atom properties
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge index: bonds
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Undirected
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, num_nodes=num_atoms)
    
    def decompose_to_motifs_chemistry(self, mol):
        """
        Level 2: Motif decomposition using BRICS
        
        BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures):
        - Chemically meaningful fragmentation
        - Preserves functional groups
        - Used in HiMol paper
        """
        # BRICS decomposition
        try:
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            
            if not brics_bonds:
                # If no BRICS bonds, treat whole molecule as one motif
                all_atom_indices = list(range(mol.GetNumAtoms()))
                return {0: all_atom_indices}, ['whole_molecule']
            
            # Break at BRICS bonds to get motifs
            # For simplicity, group atoms by connectivity after removing BRICS bonds
            atom_to_motif = {}
            motif_id = 0
            visited = set()
            
            # Get bonds to remove
            bonds_to_remove = set()
            for bond_indices, bond_type in brics_bonds:
                bonds_to_remove.add(tuple(sorted(bond_indices)))
            
            # DFS to find connected components (motifs)
            motif_atoms = {}
            motif_types = []
            
            for start_atom in range(mol.GetNumAtoms()):
                if start_atom in visited:
                    continue
                
                # BFS to find connected atoms (excluding BRICS bonds)
                queue = [start_atom]
                current_motif = []
                
                while queue:
                    atom_idx = queue.pop(0)
                    if atom_idx in visited:
                        continue
                    
                    visited.add(atom_idx)
                    current_motif.append(atom_idx)
                    
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        bond_tuple = tuple(sorted([atom_idx, neighbor_idx]))
                        
                        # Only traverse if bond is not a BRICS bond
                        if bond_tuple not in bonds_to_remove and neighbor_idx not in visited:
                            queue.append(neighbor_idx)
                
                if current_motif:
                    motif_atoms[motif_id] = current_motif
                    
                    # Identify motif type
                    motif_type = self._identify_motif_type(mol, current_motif)
                    motif_types.append(motif_type)
                    
                    motif_id += 1
            
            return motif_atoms, motif_types
        
        except Exception as e:
            print(f"      BRICS failed: {e}, using fallback")
            # Fallback: simple functional group detection
            return self._fallback_motif_detection(mol)
    
    def _identify_motif_type(self, mol, atom_indices):
        """Identify what type of chemical motif this is"""
        atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in atom_indices]
        
        # Simple heuristics
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == 'C'):
            return 'aromatic_ring'
        elif 'O' in atom_symbols and 'C' in atom_symbols:
            if any(mol.GetAtomWithIdx(i).GetFormalCharge() < 0 for i in atom_indices):
                return 'carboxyl'
            return 'carbonyl'
        elif 'N' in atom_symbols:
            return 'amine'
        elif len(set(atom_symbols)) == 1 and 'C' in atom_symbols:
            return 'alkyl_chain'
        else:
            return 'other'
    
    def _fallback_motif_detection(self, mol):
        """Fallback: use RDKit fragment functions"""
        motif_atoms = {}
        motif_types = []
        motif_id = 0
        assigned = set()
        
        # Detect aromatic rings
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                motif_atoms[motif_id] = list(ring)
                motif_types.append('aromatic_ring')
                assigned.update(ring)
                motif_id += 1
        
        # Remaining atoms as backbone
        remaining = [i for i in range(mol.GetNumAtoms()) if i not in assigned]
        if remaining:
            motif_atoms[motif_id] = remaining
            motif_types.append('backbone')
        
        return motif_atoms, motif_types
    
    def build_level2_motif_graph(self, mol, atom_graph, motif_atoms, motif_types):
        """
        Level 2: Motif graph
        Aggregate atoms → motifs, connect motifs
        """
        num_motifs = len(motif_atoms)
        
        # Pool atom features to motif features (mean pooling)
        motif_features = []
        for motif_id in range(num_motifs):
            atom_indices = motif_atoms[motif_id]
            atom_feats = atom_graph.x[atom_indices]
            motif_feat = atom_feats.mean(dim=0)  # Mean pooling
            motif_features.append(motif_feat)
        
        motif_x = torch.stack(motif_features)
        
        # Build motif edges: connect motifs if they share a bond
        motif_edges = []
        for i in range(num_motifs):
            for j in range(i+1, num_motifs):
                atoms_i = set(motif_atoms[i])
                atoms_j = set(motif_atoms[j])
                
                # Check if any atom in motif i is bonded to any atom in motif j
                connected = False
                for atom_i in atoms_i:
                    for atom_j in atoms_j:
                        if mol.GetBondBetweenAtoms(int(atom_i), int(atom_j)):
                            connected = True
                            break
                    if connected:
                        break
                
                if connected:
                    motif_edges.append([i, j])
                    motif_edges.append([j, i])
        
        motif_edge_index = torch.tensor(motif_edges, dtype=torch.long).t().contiguous() if motif_edges else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=motif_x, edge_index=motif_edge_index, num_nodes=num_motifs), motif_types
    
    def build_level3_scaffold_graph(self, mol, motif_graph):
        """
        Level 3: Scaffold graph
        Extract Bemis-Murcko scaffold (core structure)
        """
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            has_scaffold = scaffold.GetNumAtoms() > 0
        except:
            has_scaffold = False
        
        # Pool motif features to scaffold-level (single node or molecule-level)
        scaffold_feat = motif_graph.x.mean(dim=0, keepdim=True)  # [1, feat_dim]
        
        # Scaffold graph: single node representing entire molecule
        scaffold_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=scaffold_feat, edge_index=scaffold_edge_index, num_nodes=1)
    
    def build_hierarchical_drug(self, smiles, drug_id):
        """Build complete 3-level hierarchy for one drug"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"    ✗ Invalid SMILES for drug {drug_id}")
            return None
        
        # Level 1: Atom graph
        atom_graph = self.build_level1_atom_graph(mol)
        
        # Level 2: Motif graph (chemistry-informed decomposition)
        motif_atoms, motif_types = self.decompose_to_motifs_chemistry(mol)
        motif_graph, motif_types = self.build_level2_motif_graph(mol, atom_graph, motif_atoms, motif_types)
        
        # Level 3: Scaffold graph
        scaffold_graph = self.build_level3_scaffold_graph(mol, motif_graph)
        
        return {
            'drug_id': drug_id,
            'smiles': smiles,
            'mol': mol,  # Store RDKit mol for reference
            'level1_atoms': atom_graph,
            'level2_motifs': motif_graph,
            'level3_scaffold': scaffold_graph,
            'motif_types': motif_types,
            'motif_to_atoms': motif_atoms,
        }

def main():
    print("="*70)
    print("HIERARCHICAL DRUG BUILDER - Chemistry-Informed")
    print("Based on H2GNN-DTI and HiMol architectures")
    print("="*70)
    
    # Load existing drug graphs (for reference)
    drug_graphs_path = Path('data/processed/validated_phase1_drug_graphs.pkl')
    with open(drug_graphs_path, 'rb') as f:
        drug_graphs = pickle.load(f)
    
    print(f"\n✓ Loaded {len(drug_graphs)} existing drug graphs")
    
    # Load drug SMILES
    import pandas as pd
    davis_df = pd.read_csv('data/poc/davis_subset_100.csv')
    drug_smiles_df = davis_df[['Drug_ID', 'Drug']].drop_duplicates()
    
    print(f"✓ Found {len(drug_smiles_df)} unique drugs with SMILES")
    
    # Build hierarchical drugs
    builder = ChemistryInformedDrugBuilder()
    hierarchical_drugs = {}
    
    print(f"\nBuilding hierarchical representations...")
    print("-"*70)
    
    for idx, row in drug_smiles_df.iterrows():
        drug_id = row['Drug_ID']
        smiles = row['Drug']
        
        # Only process drugs we have graphs for
        if drug_id not in drug_graphs:
            continue
        
        print(f"  Drug {drug_id}...")
        
        hier_drug = builder.build_hierarchical_drug(smiles, drug_id)
        
        if hier_drug:
            hierarchical_drugs[drug_id] = hier_drug
            print(f"    ✓ Atoms: {hier_drug['level1_atoms'].num_nodes}, "
                  f"Motifs: {hier_drug['level2_motifs'].num_nodes} {hier_drug['motif_types']}, "
                  f"Scaffold: {hier_drug['level3_scaffold'].num_nodes}")
    
    # Save
    output_path = Path('data/processed/hierarchical_drugs.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(hierarchical_drugs, f)
    
    print("-"*70)
    print(f"\n✅ SUCCESS: Built {len(hierarchical_drugs)} hierarchical drugs")
    print(f"✅ Saved to: {output_path}")
    
    # Statistics
    total_motifs = sum(d['level2_motifs'].num_nodes for d in hierarchical_drugs.values())
    avg_motifs = total_motifs / len(hierarchical_drugs)
    
    print(f"\n📊 Statistics:")
    print(f"   Average motifs per drug: {avg_motifs:.1f}")
    print(f"   Total motifs: {total_motifs}")
    
    # Motif type distribution
    motif_type_counts = defaultdict(int)
    for drug in hierarchical_drugs.values():
        for mtype in drug['motif_types']:
            motif_type_counts[mtype] += 1
    
    print(f"\n🧪 Motif Types Found:")
    for mtype, count in sorted(motif_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {mtype}: {count}")
    
    print("="*70)

if __name__ == "__main__":
    main()
