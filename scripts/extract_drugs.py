"""
Extract unique drug SMILES from Davis dataset
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

def extract_drugs():
    # Load Davis data
    df = pd.read_csv('data/raw/davis_full.csv')
    
    # Extract unique drugs
    # Davis uses Drug_ID and Drug (SMILES)
    drugs = df[['Drug_ID', 'Drug']].drop_duplicates()
    
    print(f"Found {len(drugs)} unique drugs")
    
    # Validate SMILES and compute properties
    valid_drugs = []
    for idx, row in drugs.iterrows():
        drug_id = row['Drug_ID']
        smiles = row['Drug']
        
        # Try to parse with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Compute properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            n_atoms = mol.GetNumAtoms()
            
            valid_drugs.append({
                'Drug_ID': drug_id,
                'SMILES': smiles,
                'MW': mw,
                'LogP': logp,
                'TPSA': tpsa,
                'NumAtoms': n_atoms
            })
        else:
            print(f"Warning: Invalid SMILES for {drug_id}")
    
    drugs_df = pd.DataFrame(valid_drugs)
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    drugs_df.to_csv('data/processed/drugs.csv', index=False)
    
    # Statistics
    print(f"\n✓ Valid drugs: {len(drugs_df)}")
    print(f"\nMolecular properties:")
    print(f"  MW: {drugs_df['MW'].mean():.1f} ± {drugs_df['MW'].std():.1f}")
    print(f"  LogP: {drugs_df['LogP'].mean():.2f} ± {drugs_df['LogP'].std():.2f}")
    print(f"  TPSA: {drugs_df['TPSA'].mean():.1f} ± {drugs_df['TPSA'].std():.1f}")
    print(f"  Atoms: {drugs_df['NumAtoms'].mean():.1f} ± {drugs_df['NumAtoms'].std():.1f}")
    
    print(f"\n✓ Drugs saved: data/processed/drugs.csv")
    
    return drugs_df

if __name__ == "__main__":
    extract_drugs()
