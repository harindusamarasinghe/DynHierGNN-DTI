#!/usr/bin/env python3
"""
Select 20 NEW diverse drugs for batch2 (different from POC)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from pathlib import Path
import random

# Load Davis dataset
davis = pd.read_csv('data/raw/davis_full.csv')

# Load POC drugs (already used)
poc_drugs = pd.read_csv('data/poc/drug_smiles.csv')
poc_drug_ids = set(poc_drugs['Drug_ID'].unique())

print("="*70)
print("BATCH 2 DRUG SELECTION")
print("="*70)
print(f"POC drugs: {len(poc_drug_ids)}")

# Get all unique drugs from Davis
all_drugs = davis['Drug_ID'].unique()
print(f"Total drugs in Davis: {len(all_drugs)}")

# Filter: NEW drugs only
new_candidate_drugs = [d for d in all_drugs if d not in poc_drug_ids]
print(f"Candidate new drugs: {len(new_candidate_drugs)}")

# Get SMILES for candidates (column is 'Drug' not 'Drug_SMILES')
drug_smiles_dict = {}
for _, row in davis.iterrows():
    if row['Drug_ID'] in new_candidate_drugs:
        drug_smiles_dict[row['Drug_ID']] = row['Drug']  # FIXED: 'Drug' column

# Calculate molecular fingerprints
fps = {}
valid_drugs = []

print("\nCalculating molecular fingerprints...")
for drug_id in new_candidate_drugs:
    if drug_id not in drug_smiles_dict:
        continue
        
    smile = drug_smiles_dict[drug_id]
    mol = Chem.MolFromSmiles(smile)
    
    if mol is not None:
        try:
            # Morgan fingerprint for similarity
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps[drug_id] = fp
            valid_drugs.append(drug_id)
        except:
            continue

print(f"Valid drugs for selection: {len(valid_drugs)}")

# Greedy selection for maximum diversity
print("\nSelecting 20 diverse drugs...")
selected_drugs = []
selected_fps = []

# Start with random drug
random.seed(456)  # Different from POC seed
first_drug = random.choice(valid_drugs)
selected_drugs.append(first_drug)
selected_fps.append(fps[first_drug])

# Greedily add most dissimilar drugs
while len(selected_drugs) < 20 and len(valid_drugs) > len(selected_drugs):
    max_min_distance = -1
    best_drug = None
    
    for drug in valid_drugs:
        if drug in selected_drugs:
            continue
        
        # Calculate similarities to all selected drugs
        drug_fp = fps[drug]
        similarities = [DataStructs.TanimotoSimilarity(drug_fp, sfp) 
                       for sfp in selected_fps]
        
        # Min similarity = max distance to nearest selected drug
        min_sim = min(similarities)
        
        # Want drug with max distance (most dissimilar)
        if min_sim > max_min_distance:
            max_min_distance = min_sim
            best_drug = drug
    
    if best_drug:
        selected_drugs.append(best_drug)
        selected_fps.append(fps[best_drug])

# Calculate properties for selected drugs
print("\nCalculating drug properties...")
drug_data = []
for drug_id in selected_drugs:
    smile = drug_smiles_dict[drug_id]
    mol = Chem.MolFromSmiles(smile)
    
    if mol is not None:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        atom_count = mol.GetNumAtoms()
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        drug_data.append({
            'Drug_ID': drug_id,
            'SMILES': smile,
            'MW': mw,
            'LogP': logp,
            'TPSA': tpsa,
            'Atom_Count': atom_count,
            'HBD': hbd,
            'HBA': hba
        })

# Save
Path('data/poc').mkdir(parents=True, exist_ok=True)
batch2_drugs = pd.DataFrame(drug_data)
batch2_drugs.to_csv('data/poc/batch2_drug_smiles.csv', index=False)

print("\n" + "="*70)
print("BATCH 2 DRUG SELECTION COMPLETE")
print("="*70)
print(f"Total selected: {len(batch2_drugs)}")
print(f"\nMolecular properties:")
print(f"  MW range:   {batch2_drugs['MW'].min():.1f} - {batch2_drugs['MW'].max():.1f} Da")
print(f"  LogP range: {batch2_drugs['LogP'].min():.2f} - {batch2_drugs['LogP'].max():.2f}")
print(f"  TPSA range: {batch2_drugs['TPSA'].min():.1f} - {batch2_drugs['TPSA'].max():.1f} Ų")
print(f"\nSample drugs:")
print(batch2_drugs[['Drug_ID', 'MW', 'LogP']].head(10))
print(f"\nSaved to: data/poc/batch2_drug_smiles.csv")
print("\n✓ Ready for graph construction!")