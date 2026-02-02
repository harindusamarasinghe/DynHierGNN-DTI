"""
Create new POC CSVs based on proteins that have conformations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    print("=" * 70)
    print("CREATING NEW POC CSVs FROM CONFORMATIONS")
    print("=" * 70)
    
    # 1. Get proteins with conformations
    conf_dir = Path("data/conformations/poc_consolidated")
    protein_folders = [f for f in conf_dir.iterdir() if f.is_dir()]
    
    available_proteins = set()
    for folder in protein_folders:
        conf_files = list(folder.glob("conf_*.pdb"))
        if len(conf_files) >= 2:
            # Keep FULL name: POC_PROT_AURKA (don't remove prefix)
            protein_id = folder.name
            available_proteins.add(protein_id)
    
    print(f"\n1. Proteins with ≥2 conformations: {len(available_proteins)}")
    print(f"   Sample: {sorted(list(available_proteins))[:5]}")
    
    # 2. Load original POC dataset
    poc_davis = pd.read_csv('data/poc/poc_davis.csv')
    print(f"\n2. Original POC dataset: {len(poc_davis)} rows")
    print(f"   Columns: {poc_davis.columns.tolist()}")
    print(f"   Sample Y values: {poc_davis['Y'].head().tolist()}")
    
    # 3. Standardize protein IDs in CSV
    if 'Target_ID' in poc_davis.columns:
        protein_col = 'Target_ID'
    elif 'Target' in poc_davis.columns:
        protein_col = 'Target'
    else:
        print("ERROR: Cannot find Target_ID or Target column")
        return
    
    # ADD POC_PROT_ prefix if not present
    sample = str(poc_davis[protein_col].iloc[0])
    if 'POC_PROT_' not in sample:
        poc_davis['Target_ID'] = 'POC_PROT_' + poc_davis[protein_col].astype(str)
    else:
        poc_davis['Target_ID'] = poc_davis[protein_col]
    
    print(f"\n   After standardization:")
    print(f"   Sample Target_IDs: {poc_davis['Target_ID'].head().tolist()}")
    
    # 4. Filter for proteins with conformations
    df_filtered = poc_davis[poc_davis['Target_ID'].isin(available_proteins)].copy()
    
    print(f"\n3. After filtering for available proteins:")
    print(f"   Rows: {len(df_filtered)}")
    print(f"   Unique proteins: {df_filtered['Target_ID'].nunique()}")
    print(f"   Unique drugs: {df_filtered['Drug_ID'].nunique()}")
    
    if len(df_filtered) == 0:
        print("\n ERROR: NO MATCHES FOUND!")
        print("\nDebug info:")
        print(f"Sample protein IDs in CSV: {poc_davis['Target_ID'].head().tolist()}")
        print(f"Sample protein IDs in conformations: {sorted(list(available_proteins))[:5]}")
        return
    
    # 5. Convert Y to binary (threshold at median)
    print(f"\n4. Converting Y to binary:")
    print(f"   Original Y range: {df_filtered['Y'].min():.2f} - {df_filtered['Y'].max():.2f}")
    
    threshold = df_filtered['Y'].median()
    print(f"   Using threshold: {threshold:.2f} (median)")
    
    df_filtered['Y_original'] = df_filtered['Y']  # Keep original
    df_filtered['Y'] = (df_filtered['Y'] >= threshold).astype(int)  # Binary
    
    print(f"   After conversion:")
    print(f"     Active (Y=1): {(df_filtered['Y']==1).sum()}")
    print(f"     Inactive (Y=0): {(df_filtered['Y']==0).sum()}")
    
    # 6. Ensure all required columns exist
    if 'Drug' not in df_filtered.columns:
        # Merge with drug_smiles.csv
        drug_smiles = pd.read_csv('data/poc/drug_smiles.csv')
        
        smiles_col = 'SMILES' if 'SMILES' in drug_smiles.columns else 'Drug'
        drug_id_col = 'Drug_ID' if 'Drug_ID' in drug_smiles.columns else list(drug_smiles.columns)[0]
        
        df_filtered = df_filtered.merge(
            drug_smiles[[drug_id_col, smiles_col]],
            left_on='Drug_ID',
            right_on=drug_id_col,
            how='left'
        )
        df_filtered = df_filtered.rename(columns={smiles_col: 'Drug'})
    
    # 7. Create final format
    final_df = pd.DataFrame({
        'Drug_ID': df_filtered['Drug_ID'].astype(str),
        'Drug': df_filtered['Drug'],
        'Target_ID': df_filtered['Target_ID'],
        'Target': df_filtered['Target_ID'],
        'Y': df_filtered['Y'],
        'Y_original': df_filtered['Y_original']
    })
    
    final_df = final_df.dropna(subset=['Drug'])
    
    print(f"\n5. Final dataset:")
    print(f"   Pairs: {len(final_df)}")
    print(f"   Drugs: {final_df['Drug_ID'].nunique()}")
    print(f"   Proteins: {final_df['Target_ID'].nunique()}")
    print(f"   Class distribution:")
    print(f"     Active (Y=1): {(final_df['Y']==1).sum()}")
    print(f"     Inactive (Y=0): {(final_df['Y']==0).sum()}")
    
    # 8. Create splits (70/15/15)
    train_df, temp_df = train_test_split(
        final_df, test_size=0.3, random_state=42, 
        stratify=final_df['Y']  # Stratify by Y (binary)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, 
        stratify=temp_df['Y']
    )
    
    print(f"\n6. Splits:")
    print(f"   Train: {len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%)")
    print(f"     Active: {(train_df['Y']==1).sum()}, Inactive: {(train_df['Y']==0).sum()}")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(final_df)*100:.1f}%)")
    print(f"     Active: {(val_df['Y']==1).sum()}, Inactive: {(val_df['Y']==0).sum()}")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)")
    print(f"     Active: {(test_df['Y']==1).sum()}, Inactive: {(test_df['Y']==0).sum()}")
    
    # 9. Save
    train_df.to_csv('data/poc/poc_train.csv', index=False)
    val_df.to_csv('data/poc/poc_val.csv', index=False)
    test_df.to_csv('data/poc/poc_test.csv', index=False)
    
    print(f"\n✓ Saved new CSVs to data/poc/")
    print("=" * 70)
    print("DONE - Ready for training!")
    print("=" * 70)

if __name__ == "__main__":
    main()
