"""
Build PyG graphs for POC drugs from SMILES.
Usage: python scripts/build_poc_drug_graphs.py
Output: data/processed/poc_drug_graphs.pkl
"""

import pickle
from pathlib import Path
import torch
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd

def smiles_to_graph(smiles: str) -> Data:
    """Convert SMILES to a simple atom-bond graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Node features: 9-dim one-hot by atomic number bucket
    atom_feats = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        feat = [0] * 9
        feat[min(z, 8)] = 1
        atom_feats.append(feat)

    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edges: undirected bonds
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING POC DRUG GRAPHS")
    print("=" * 70)
    
    # Collect all unique drugs from train/val/test CSVs
    all_drugs = {}
    
    for split in ['train', 'val', 'test']:
        csv_path = f'data/poc/poc_{split}.csv'
        df = pd.read_csv(csv_path)
        
        print(f"\n{split.upper()}: {len(df)} rows")
        
        for _, row in df.iterrows():
            drug_id = str(row['Drug_ID'])  # Convert to string
            smiles = row['Drug']
            
            if drug_id not in all_drugs:
                all_drugs[drug_id] = smiles
    
    print(f"\n✓ Total unique drugs: {len(all_drugs)}")
    
    # Build graphs
    drug_graphs = {}
    failed = []
    
    for drug_id, smiles in all_drugs.items():
        try:
            graph = smiles_to_graph(smiles)
            drug_graphs[drug_id] = graph
            print(f"✓ {drug_id}")
        except Exception as e:
            print(f"⚠️  Failed for {drug_id}: {e}")
            failed.append(drug_id)
    
    print(f"\n✓ Successfully built: {len(drug_graphs)} graphs")
    if failed:
        print(f"⚠️  Failed: {len(failed)} drugs")
    
    # Save
    out_path = out_dir / "poc_drug_graphs.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(drug_graphs, f)
    
    print(f"\n✓ Saved to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
