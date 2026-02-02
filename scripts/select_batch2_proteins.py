#!/usr/bin/env python3
"""
Select 50 NEW proteins for batch2 (different from POC)
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Load Davis dataset
davis = pd.read_csv('data/raw/davis_full.csv')

# Load POC proteins (already used)
poc_proteins = pd.read_csv('data/poc/protein_mapping.csv')
print("POC protein columns:", poc_proteins.columns.tolist())

# Use the correct column name
poc_protein_ids = set(poc_proteins['Protein_ID'].unique())

print(f"POC proteins: {len(poc_protein_ids)}")

# Get unique proteins from Davis
all_proteins = davis['Target_ID'].unique()
print(f"Total proteins in Davis: {len(all_proteins)}")

# Filter: NEW proteins + size constraints (180-650 aa)
new_candidates = []
for prot_id in all_proteins:
    # Skip if already in POC
    if prot_id in poc_protein_ids:
        continue
    
    # Get sequence
    prot_seq = davis[davis['Target_ID'] == prot_id]['Target'].iloc[0]
    seq_len = len(prot_seq)
    
    # Size constraints (same as POC)
    if 180 <= seq_len <= 650:
        new_candidates.append({
            'Protein_ID': prot_id,        # Changed from Target_ID
            'Sequence': prot_seq,
            'Length': seq_len             # Changed from Length_aa
        })

print(f"New candidate proteins (180-650 aa): {len(new_candidates)}")

# Stratified selection by length (same as POC)
new_candidates_df = pd.DataFrame(new_candidates)

# Bins: Short (180-350), Medium (350-500), Long (500-650)
new_candidates_df['Length_Bin'] = pd.cut(
    new_candidates_df['Length'],          # Changed from Length_aa
    bins=[180, 400, 550, 650],
    labels=['Short', 'Medium', 'Long']
)

# Select balanced: 15 short, 25 medium, 10 long
random.seed(456)  # Different seed than POC (was 42)
selected = []

for bin_name, target_count in [('Short', 15), ('Medium', 25), ('Long', 10)]:
    bin_proteins = new_candidates_df[new_candidates_df['Length_Bin'] == bin_name]
    
    if len(bin_proteins) >= target_count:
        selected_bin = bin_proteins.sample(n=target_count, random_state=456)
    else:
        selected_bin = bin_proteins
        print(f"Warning: Only {len(bin_proteins)} proteins in {bin_name} bin")
    
    selected.append(selected_bin)

batch2_proteins = pd.concat(selected, ignore_index=True)

# Keep same column format as POC (Protein_ID, Sequence, Length)
# No need for POC_PROT_ID column since POC doesn't have it

# Save mapping
Path('data/poc').mkdir(parents=True, exist_ok=True)
batch2_proteins[['Protein_ID', 'Sequence', 'Length']].to_csv(
    'data/poc/batch2_protein_mapping.csv', 
    index=False
)

print("\n" + "="*70)
print("BATCH 2 PROTEIN SELECTION COMPLETE")
print("="*70)
print(f"Total selected: {len(batch2_proteins)}")
print(f"\nLength distribution:")
print(batch2_proteins['Length_Bin'].value_counts().sort_index())
print(f"\nLength range: {batch2_proteins['Length'].min()} - {batch2_proteins['Length'].max()} aa")
print(f"\nSaved to: data/poc/batch2_protein_mapping.csv")
print("\nSample proteins:")
print(batch2_proteins[['Protein_ID', 'Length']].head(10))
