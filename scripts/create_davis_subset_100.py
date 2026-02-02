#!/usr/bin/env python3
"""
Extract interactions for your 100 pre-selected proteins from full Davis dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("EXTRACTING 100 PROTEINS FROM FULL DAVIS DATASET")
    print("=" * 70)
    
    # 1. Load your 100 proteins
    protein_map1 = pd.read_csv("data/poc/protein_mapping.csv")
    protein_map2 = pd.read_csv("data/poc/batch2_protein_mapping.csv")
    
    your_proteins = set(protein_map1['Protein_ID']) | set(protein_map2['Protein_ID'])
    print(f"\n✅ Your 100 proteins:")
    print(f"   Batch 1: {len(protein_map1)}")
    print(f"   Batch 2: {len(protein_map2)}")
    print(f"   Total: {len(your_proteins)}")
    print(f"   Sample: {sorted(list(your_proteins))[:5]}")
    
    # 2. Load full Davis dataset
    print(f"\n🔍 Looking for full Davis dataset...")
    
    possible_paths = [
        "data/raw/davis_full.csv",
        "data/davis.tab",
        "data/raw/davis.tab",
    ]
    
    davis_full = None
    for path in possible_paths:
        if Path(path).exists():
            print(f"   Found: {path}")
            try:
                if path.endswith('.tab'):
                    davis_full = pd.read_csv(path, sep='\t')
                else:
                    davis_full = pd.read_csv(path)
                print(f"   ✅ Loaded {len(davis_full)} interactions")
                break
            except Exception as e:
                print(f"   ⚠️ Error loading: {e}")
    
    if davis_full is None:
        print("\n❌ Could not find full Davis dataset!")
        print("\nYou have 2 options:")
        print("\nOPTION 1: Download full Davis")
        print("----------")
        print("from tdc.multi_pred import DTI")
        print("data = DTI(name='DAVIS')")
        print("df = data.get_data()")
        print("df.to_csv('data/raw/davis_full.csv', index=False)")
        print("\nOPTION 2: Use only poc_davis.csv (50 proteins)")
        print("----------")
        print("This limits Phase 1 to 50 proteins instead of 100")
        return
    
    print(f"\n✅ Full Davis dataset:")
    print(f"   Shape: {davis_full.shape}")
    print(f"   Columns: {davis_full.columns.tolist()}")
    
    # 3. Identify columns
    # Try to find the right column names
    protein_col = None
    drug_col = None
    y_col = None
    smiles_col = None
    seq_col = None
    
    for col in davis_full.columns:
        col_lower = col.lower()
        if 'target' in col_lower and 'id' in col_lower:
            protein_col = col
        elif 'drug' in col_lower and 'id' in col_lower:
            drug_col = col
        elif col in ['Y', 'y', 'affinity', 'Affinity']:
            y_col = col
        elif 'drug' in col_lower and ('smiles' in col_lower or col == 'Drug'):
            smiles_col = col
        elif 'target' in col_lower and ('seq' in col_lower or col == 'Target'):
            seq_col = col
    
    # Fallback guesses
    if not protein_col and 'Target_ID' in davis_full.columns:
        protein_col = 'Target_ID'
    if not drug_col and 'Drug_ID' in davis_full.columns:
        drug_col = 'Drug_ID'
    if not y_col and 'Y' in davis_full.columns:
        y_col = 'Y'
    
    print(f"\n✅ Column mapping:")
    print(f"   Protein ID: {protein_col}")
    print(f"   Drug ID: {drug_col}")
    print(f"   Drug SMILES: {smiles_col}")
    print(f"   Affinity: {y_col}")
    
    if not all([protein_col, drug_col, y_col]):
        print(f"\n❌ Could not identify all required columns!")
        print(f"   Available: {davis_full.columns.tolist()}")
        print(f"\n   First row:")
        print(davis_full.head(1).T)
        return
    
    # 4. Check if protein IDs need cleaning
    sample_protein = str(davis_full[protein_col].iloc[0])
    print(f"\n   Sample protein ID from Davis: {sample_protein}")
    print(f"   Sample from your mapping: {list(your_proteins)[0]}")
    
    # Check if we need to add/remove prefixes
    if 'POC_PROT_' in sample_protein and 'POC_PROT_' not in list(your_proteins)[0]:
        # Davis has prefix, your proteins don't
        your_proteins_clean = {f"POC_PROT_{p}" for p in your_proteins}
    elif 'POC_PROT_' not in sample_protein and 'POC_PROT_' in list(your_proteins)[0]:
        # Your proteins have prefix, Davis doesn't
        your_proteins_clean = {p.replace('POC_PROT_', '') for p in your_proteins}
    else:
        # No mismatch
        your_proteins_clean = your_proteins
    
    print(f"   Using cleaned IDs: {list(your_proteins_clean)[:3]}")
    
    # 5. Extract interactions for your 100 proteins
    df_subset = davis_full[davis_full[protein_col].isin(your_proteins_clean)].copy()
    
    print(f"\n✅ Extracted subset:")
    print(f"   Interactions: {len(df_subset)}")
    print(f"   Unique proteins: {df_subset[protein_col].nunique()}")
    print(f"   Unique drugs: {df_subset[drug_col].nunique()}")
    
    if len(df_subset) == 0:
        print("\n❌ NO MATCHES! Debug:")
        print(f"   Davis proteins (sample): {davis_full[protein_col].unique()[:5].tolist()}")
        print(f"   Your proteins (sample): {list(your_proteins_clean)[:5]}")
        return
    
    # Check which proteins are missing
    found_proteins = set(df_subset[protein_col].unique())
    missing_proteins = your_proteins_clean - found_proteins
    
    if missing_proteins:
        print(f"\n⚠️ {len(missing_proteins)} proteins have NO interactions in Davis:")
        for p in sorted(list(missing_proteins))[:10]:
            print(f"     - {p}")
        if len(missing_proteins) > 10:
            print(f"     ... and {len(missing_proteins) - 10} more")
    
    # 6. Standardize column names
    df_subset = df_subset.rename(columns={
        protein_col: 'Target_ID',
        drug_col: 'Drug_ID',
        y_col: 'Y'
    })
    
    if smiles_col and smiles_col != 'Drug':
        df_subset = df_subset.rename(columns={smiles_col: 'Drug'})
    
    # 7. Add Drug SMILES if missing
    if 'Drug' not in df_subset.columns:
        drug_smiles = pd.read_csv('data/poc/drug_smiles.csv')
        df_subset = df_subset.merge(
            drug_smiles[['Drug_ID', 'SMILES']],
            on='Drug_ID',
            how='left'
        )
        df_subset['Drug'] = df_subset['SMILES']
    
    # 8. Final format
    final_df = pd.DataFrame({
        'Drug_ID': df_subset['Drug_ID'],
        'Drug': df_subset['Drug'],
        'Target_ID': df_subset['Target_ID'],
        'Y': df_subset['Y']
    })
    
    final_df = final_df.dropna()
    
    print(f"\n✅ Final dataset (100 proteins from Davis):")
    print(f"   Interactions: {len(final_df)}")
    print(f"   Proteins: {final_df['Target_ID'].nunique()}")
    print(f"   Drugs: {final_df['Drug_ID'].nunique()}")
    print(f"   Y range: {final_df['Y'].min():.2f} - {final_df['Y'].max():.2f}")
    print(f"   Y mean: {final_df['Y'].mean():.2f} ± {final_df['Y'].std():.2f}")
    
    # 9. Save
    output_path = "data/poc/davis_subset_100.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n SAVED: {output_path}")
    print(f"\n{'='*70}")
    print("NEXT STEP: Create Phase 1 splits from this file")
    print(f"{'='*70}")
    print("python scripts/create_phase1_splits.py")


if __name__ == "__main__":
    main()
