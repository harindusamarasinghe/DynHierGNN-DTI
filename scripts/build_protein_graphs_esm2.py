#!/usr/bin/env python3
"""
Build protein graphs with ESM-2 embeddings (1300-dim node features)
Input: data/conformations/full/POC_PROT_ID/conf_X.pdb
ESM-2: embeddings/all_proteins/POC_PROT_ID_esm2.npy
Output: data/processed/full_protein_graphs_esm2.pkl
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pandas as pd

class ProteinGraphBuilderESM2:
    def __init__(self, conformation_dir, esm2_dir, output_dir):
        self.conformation_dir = Path(conformation_dir)
        self.esm2_dir = Path(esm2_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = PDBParser(QUIET=True)
        
        # Amino acid to index mapping (20 standard AAs)
        self.aa_to_idx = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
        }
    
    def pdb_to_graph(self, pdb_file, esm_embeddings, distance_threshold=8.0):
        """Convert PDB + ESM-2 to protein graph"""
        
        # Parse PDB
        structure = self.parser.get_structure("protein", pdb_file)
        residues = []
        coords = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':  # Skip hetero atoms
                        continue
                    
                    try:
                        ca = residue['CA']
                        coord = ca.get_coord()
                        resname = residue.resname
                        
                        # Only standard 20 AAs
                        if resname in self.aa_to_idx:
                            residues.append(resname)
                            coords.append(coord)
                    except:
                        continue
        
        if len(residues) == 0:
            return None
        
        coords = np.array(coords)
        
        # One-hot amino acid features [N, 20]
        node_features_onehot = []
        for res in residues:
            feat = np.zeros(20)
            feat[self.aa_to_idx[res]] = 1
            node_features_onehot.append(feat)
        x_onehot = torch.tensor(node_features_onehot, dtype=torch.float)
        
        # ESM-2 features [N, 1280] - truncate/pad to match residues
        esm_features = []
        esm_idx = 0
        for i in range(len(residues)):
            if esm_idx < esm_embeddings.shape[0]:
                esm_features.append(esm_embeddings[esm_idx])
                esm_idx += 1
            else:
                # Pad with mean if ESM longer than residues
                esm_features.append(np.mean(esm_embeddings, axis=0))
        
        x_esm2 = torch.tensor(np.array(esm_features), dtype=torch.float)
        
        # Combined features [N, 1300]
        x = torch.cat([x_onehot, x_esm2], dim=1)
        
        # Build edges (residues within 8Å)
        dist_matrix = cdist(coords, coords)
        edge_index = []
        edge_attr = []
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if dist_matrix[i, j] < distance_threshold:
                    inv_dist = 1.0 / (dist_matrix[i, j] + 0.1)
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([inv_dist, inv_dist])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def build_all_graphs(self):
        """Build graphs for ALL proteins from multiple conformation directories"""
        print("="*70)
        print("BUILDING ESM-2 PROTEIN GRAPHS FOR ALL 100 PROTEINS (1300-dim features)")
        print("="*70)
        
        all_graphs = {}
        stats = {'3conf': 0, '2conf': 0, 'skipped': 0}
        
        # ADDED: Scan MULTIPLE conformation directories
        conf_dirs = [
            "data/conformations/poc_consolidated",  # POC batch1 (50 proteins)
            "data/conformations/full"              # Batch2 (50 proteins)  
        ]
        
        all_protein_dirs = []
        for conf_dir in conf_dirs:
            conf_path = Path(conf_dir)
            if conf_path.exists():
                protein_dirs = [d for d in conf_path.iterdir() 
                            if d.is_dir() and d.name.startswith("POC_PROT_")]
                all_protein_dirs.extend(protein_dirs)
                print(f"Found {len(protein_dirs)} proteins in {conf_dir}")
            else:
                print(f"⚠️ Directory not found: {conf_dir}")
        
        # Remove duplicates (if any overlap)
        protein_dirs = list(set(all_protein_dirs))
        print(f"\nTotal unique protein folders: {len(protein_dirs)}")
        
        for protein_dir in tqdm(protein_dirs, desc="Proteins"):
            protein_id = protein_dir.name.replace("POC_PROT_", "")
            pdbs = sorted(protein_dir.glob("conf_*.pdb"))
            
            if len(pdbs) < 2:
                print(f"Skipping {protein_id}: only {len(pdbs)} confs")
                stats['skipped'] += 1
                continue
            
            # Load ESM-2 embeddings (once per protein)
            esm_path = self.esm2_dir / f"{protein_id}_esm2.npy"
            if not esm_path.exists():
                print(f"Skipping {protein_id}: no ESM-2 embeddings")
                stats['skipped'] += 1
                continue
            
            esm_embeddings = np.load(esm_path)
            
            # Build graph for each conformation
            conformation_graphs = []
            for pdb_file in pdbs[:3]:  # Max 3 confs
                graph = self.pdb_to_graph(pdb_file, esm_embeddings)
                if graph is not None:
                    graph.protein_id = protein_id
                    graph.conf_idx = int(pdb_file.stem.split("_")[-1])
                    conformation_graphs.append(graph)
            
            if len(conformation_graphs) >= 2:
                all_graphs[protein_id] = conformation_graphs
                stats[f"{len(conformation_graphs)}conf"] += 1
        
        # Save OVERWRITING the previous file
        output_path = self.output_dir / "all_protein_graphs_esm2.pkl"  # CHANGED filename
        with open(output_path, 'wb') as f:
            pickle.dump(all_graphs, f)
        
        print(f"\n🎉 Saved {len(all_graphs)} proteins to {output_path}")
        print(f"Stats: {stats}")
        
        # Quick stats
        total_confs = sum(len(graphs) for graphs in all_graphs.values())
        print(f"Total conformations: {total_confs}")
        print(f"Avg nodes/graph: {np.mean([g.num_nodes for graphs in all_graphs.values() for g in graphs]):.0f}")
        print(f"Avg edges/graph: {np.mean([g.num_edges for graphs in all_graphs.values() for g in graphs]):.0f}")
        
        return all_graphs


if __name__ == "__main__":
    builder = ProteinGraphBuilderESM2(
        conformation_dir="data/conformations/full",
        esm2_dir="embeddings/all_proteins",
        output_dir="data/processed"
    )
    builder.build_all_graphs()
