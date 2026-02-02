"""
Convert PDB conformations to PyTorch Geometric graphs
Usage: python scripts/build_protein_graphs.py
Output: data/processed/poc_protein_graphs.pkl
"""


import torch
import pickle
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.spatial.distance import cdist


class ProteinGraphBuilder:
    def __init__(self, conformation_dir='data/conformations/poc_consolidated', output_dir='data/processed'):
        self.conformation_dir = Path(conformation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PDBParser(QUIET=True)
        
        # Amino acid mapping (20 standard AAs)
        self.aa_to_idx = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
        }
    
    def pdb_to_graph(self, pdb_file, distance_threshold=8.0):
        """Convert PDB to residue graph"""
        structure = self.parser.get_structure('protein', pdb_file)
        
        # Extract residue info
        residues = []
        coords = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Skip hetero atoms
                        try:
                            # Get Cα coordinates
                            ca = residue['CA']
                            coord = ca.get_coord()
                            
                            # Get residue type
                            resname = residue.resname
                            
                            residues.append(resname)
                            coords.append(coord)
                        except:
                            continue
        
        if len(residues) == 0:
            return None
        
        coords = np.array(coords)
        
        # Build node features (one-hot amino acid type)
        node_features = []
        for res in residues:
            feat = np.zeros(20)
            if res in self.aa_to_idx:
                feat[self.aa_to_idx[res]] = 1
            node_features.append(feat)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Build edges (residues within distance_threshold)
        dist_matrix = cdist(coords, coords)
        edge_index = []
        edge_attr = []
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                if dist_matrix[i, j] <= distance_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Undirected
                    
                    # Edge feature: inverse distance
                    inv_dist = 1.0 / max(dist_matrix[i, j], 0.1)
                    edge_attr.append([inv_dist])
                    edge_attr.append([inv_dist])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def build_all_graphs(self):
        """Build graphs for all proteins (all conformations)"""
        print("=" * 70)
        print("BUILDING PROTEIN GRAPHS")
        print("=" * 70)
        
        protein_dirs = [d for d in self.conformation_dir.iterdir() 
                       if d.is_dir() and not d.name.endswith('_raw')]
        
        all_graphs = {}
        stats = {'3_conf': 0, '2_conf': 0, 'skipped': 0}
        
        for protein_dir in tqdm(protein_dirs, desc="Processing"):
            protein_id = protein_dir.name
            pdb_files = sorted(protein_dir.glob('conf_*.pdb'))
            
            # CHANGED: Accept 2 or 3 conformations
            if len(pdb_files) < 2:
                print(f"⚠️  Skipping {protein_id}: only {len(pdb_files)} conformation(s)")
                stats['skipped'] += 1
                continue
            
            # Convert each conformation to graph (take all available, not just first 3)
            conformation_graphs = []
            for pdb_file in pdb_files:  # CHANGED: Process all available
                graph = self.pdb_to_graph(pdb_file)
                if graph:
                    conformation_graphs.append(graph)
            
            # CHANGED: Accept 2 or 3 conformations
            if len(conformation_graphs) >= 2:
                all_graphs[protein_id] = conformation_graphs
                if len(conformation_graphs) == 3:
                    stats['3_conf'] += 1
                else:
                    stats['2_conf'] += 1
        
        # Save
        output_path = self.output_dir / 'poc_protein_graphs.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(all_graphs, f)
        
        print(f"\n✓ Processed {len(all_graphs)} proteins")
        print(f"  - {stats['3_conf']} proteins with 3 conformations")
        print(f"  - {stats['2_conf']} proteins with 2 conformations")
        if stats['skipped'] > 0:
            print(f"  - {stats['skipped']} proteins skipped (<2 conformations)")
        print(f"✓ Saved to: {output_path}")
        
        # Print statistics
        total_confs = sum([len(graphs) for graphs in all_graphs.values()])
        total_nodes = sum([sum([g.num_nodes for g in graphs]) for graphs in all_graphs.values()])
        total_edges = sum([sum([g.num_edges for g in graphs]) for graphs in all_graphs.values()])
        avg_nodes = total_nodes / total_confs
        avg_edges = total_edges / total_confs
        
        print(f"\nStatistics:")
        print(f"  Total conformations: {total_confs}")
        print(f"  Avg nodes/graph: {avg_nodes:.1f}")
        print(f"  Avg edges/graph: {avg_edges:.1f}")
        
        return all_graphs


if __name__ == "__main__":
    builder = ProteinGraphBuilder()
    builder.build_all_graphs()
