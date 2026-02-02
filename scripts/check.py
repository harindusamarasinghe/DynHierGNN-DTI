#!/usr/bin/env python3
"""
Diagnose why no positive samples are loading
"""

import pickle
import pandas as pd

# Load data
train_csv = "data/splits/phase1/phase1_train_esm2.csv"
train_df = pd.read_csv(train_csv)

with open("data/processed/validated_phase1_drug_graphs.pkl", "rb") as f:
    drug_graphs = pickle.load(f)

with open("data/processed/all_protein_graphs_esm2.pkl", "rb") as f:
    protein_graphs = pickle.load(f)

print("=" * 70)
print("DIAGNOSTIC: Why No Positive Samples?")
print("=" * 70)

print(f"\n1. RAW CSV:")
print(f"   Total rows: {len(train_df)}")
print(f"   Y=1 (active): {(train_df['Y']==1).sum()}")
print(f"   Y=0 (inactive): {(train_df['Y']==0).sum()}")
print(f"   Unique drugs: {train_df['Drug_ID'].nunique()}")
print(f"   Unique proteins: {train_df['Protein_ID'].nunique()}")

print(f"\n2. AVAILABLE GRAPHS:")
print(f"   Drug graphs: {len(drug_graphs)}")
print(f"   Protein graphs: {len(protein_graphs)}")

# Check coverage
csv_drugs = set(train_df['Drug_ID'].unique())
csv_proteins = set(train_df['Protein_ID'].unique())
graph_drugs = set(drug_graphs.keys())
graph_proteins = set(protein_graphs.keys())

print(f"\n3. COVERAGE:")
print(f"   Drug overlap: {len(csv_drugs & graph_drugs)}/{len(csv_drugs)}")
print(f"   Protein overlap: {len(csv_proteins & graph_proteins)}/{len(csv_proteins)}")

missing_drugs = csv_drugs - graph_drugs
missing_proteins = csv_proteins - graph_proteins

if missing_drugs:
    print(f"   Missing drugs: {list(missing_drugs)[:5]}")
if missing_proteins:
    print(f"   Missing proteins: {list(missing_proteins)[:5]}")

# Filter like the dataset does
valid_df = train_df[
    train_df['Drug_ID'].isin(graph_drugs) & 
    train_df['Protein_ID'].isin(graph_proteins)
]

print(f"\n4. AFTER FILTERING (what POCDTIDatasetV2 sees):")
print(f"   Valid rows: {len(valid_df)}")
print(f"   Y=1 (active): {(valid_df['Y']==1).sum()}")
print(f"   Y=0 (inactive): {(valid_df['Y']==0).sum()}")

if len(valid_df) > 0:
    print(f"\n5. BREAKDOWN BY CLASS:")
    active_df = valid_df[valid_df['Y']==1]
    inactive_df = valid_df[valid_df['Y']==0]
    
    print(f"   Active samples:")
    print(f"     Drugs: {active_df['Drug_ID'].nunique()}")
    print(f"     Proteins: {active_df['Protein_ID'].nunique()}")
    
    print(f"   Inactive samples:")
    print(f"     Drugs: {inactive_df['Drug_ID'].nunique()}")
    print(f"     Proteins: {inactive_df['Protein_ID'].nunique()}")
    
    # Check if active drugs were filtered out
    active_drugs = set(train_df[train_df['Y']==1]['Drug_ID'].unique())
    active_drugs_with_graphs = active_drugs & graph_drugs
    
    print(f"\n6. ACTIVE DRUGS:")
    print(f"   Total active drugs in CSV: {len(active_drugs)}")
    print(f"   Active drugs with graphs: {len(active_drugs_with_graphs)}")
    print(f"   Active drugs filtered out: {len(active_drugs - graph_drugs)}")
    
    if len(active_drugs - graph_drugs) > 0:
        print(f"   Missing active drug IDs: {list(active_drugs - graph_drugs)[:10]}")

print("\n" + "=" * 70)
