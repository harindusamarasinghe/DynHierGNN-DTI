#!/usr/bin/env python3
"""
Create train/val/test splits for Phase 1: 93 proteins with ESM-2 graphs
Addresses: Random split (interaction-level), class imbalance tracking
"""

import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("PHASE 1: CREATING ESM-2 SPLITS (93 PROTEINS)")
    print("=" * 70)
    
    # 1. Load available proteins (93 with ESM-2 graphs)
    with open("data/processed/all_protein_graphs_esm2.pkl", "rb") as f:
        protein_graphs = pickle.load(f)
    
    available_proteins = set(protein_graphs.keys())
    print(f"\n✅ Loaded {len(available_proteins)} proteins with ESM-2 graphs")
    print(f"   Sample: {sorted(list(available_proteins))[:5]}")
    
    # 2. Load full interaction data
    df = pd.read_csv("data/poc/davis_subset_100.csv")
    print(f"\n✅ Loaded davis_subset_100.csv:")
    print(f"   Total interactions: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # 3. Auto-detect column names
    protein_col = None
    drug_col = None
    y_col = None
    smiles_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'target' in col_lower or 'protein' in col_lower:
            if 'id' in col_lower or col in ['Target_ID', 'Protein_ID']:
                protein_col = col
        if 'drug' in col_lower:
            if 'id' in col_lower:
                drug_col = col
            elif 'smiles' in col_lower or col == 'Drug':
                smiles_col = col
        if col in ['Y', 'y', 'affinity', 'Affinity', 'Label']:
            y_col = col
    
    print(f"\n✅ Detected columns:")
    print(f"   Protein: {protein_col}")
    print(f"   Drug ID: {drug_col}")
    print(f"   Drug SMILES: {smiles_col}")
    print(f"   Label/Y: {y_col}")
    
    if not all([protein_col, drug_col, y_col]):
        print("\n❌ ERROR: Could not detect all required columns!")
        print(f"   First 3 rows:")
        print(df.head(3))
        return
    
    # 4. Standardize column names
    df = df.rename(columns={
        protein_col: 'Protein_ID',
        drug_col: 'Drug_ID',
        y_col: 'Y'
    })
    
    if smiles_col:
        df = df.rename(columns={smiles_col: 'Drug'})
    
    # Clean Protein_ID (remove POC_PROT_ prefix if present)
    if 'POC_PROT_' in str(df['Protein_ID'].iloc[0]):
        df['Protein_ID'] = df['Protein_ID'].str.replace('POC_PROT_', '')
    
    print(f"\n✅ Standardized format:")
    print(f"   Unique proteins: {df['Protein_ID'].nunique()}")
    print(f"   Unique drugs: {df['Drug_ID'].nunique()}")
    print(f"   Sample Protein_IDs: {df['Protein_ID'].head(3).tolist()}")
    
    # 5. Filter to only proteins with ESM-2 graphs
    df_filtered = df[df['Protein_ID'].isin(available_proteins)].copy()
    
    print(f"\n✅ Filtered to ESM-2 proteins:")
    print(f"   Interactions: {len(df_filtered)}")
    print(f"   Proteins: {df_filtered['Protein_ID'].nunique()}")
    print(f"   Drugs: {df_filtered['Drug_ID'].nunique()}")
    
    if len(df_filtered) == 0:
        print("\n❌ NO MATCHES!")
        print(f"   CSV proteins: {df['Protein_ID'].unique()[:5].tolist()}")
        print(f"   ESM-2 proteins: {sorted(list(available_proteins))[:5]}")
        return
    
    # 6. Add Drug SMILES if missing
    if 'Drug' not in df_filtered.columns or df_filtered['Drug'].isna().any():
        try:
            drug_smiles = pd.read_csv('data/poc/drug_smiles.csv')
            df_filtered = df_filtered.merge(
                drug_smiles[['Drug_ID', 'SMILES']],
                on='Drug_ID',
                how='left'
            )
            df_filtered['Drug'] = df_filtered['SMILES']
        except FileNotFoundError:
            print("⚠️  Warning: drug_smiles.csv not found, using existing Drug column")
    
    # 7. Convert to binary labels (IC50 threshold at 1000 nM)
    # CORRECT LABELING:
    # Active (Y=1): IC50 < 1000 nM (strong binders, MINORITY class)
    # Inactive (Y=0): IC50 >= 1000 nM (weak/non-binders, MAJORITY class)

    IC50_THRESHOLD = 1000.0  # nM (1 µM - biologically meaningful cutoff)

    df_filtered['IC50_nM'] = df_filtered['Y'].copy()  # Keep original IC50
    df_filtered['Y'] = (df_filtered['IC50_nM'] < IC50_THRESHOLD).astype(int)  # NOTE: < not >=

    print(f"\n Binary classification (IC50 < {IC50_THRESHOLD} nM = Active):")
    print(f"   Active (Y=1):   IC50 < {IC50_THRESHOLD} nM → {(df_filtered['Y']==1).sum()} ({(df_filtered['Y']==1).sum()/len(df_filtered)*100:.1f}%)")
    print(f"   Inactive (Y=0): IC50 ≥ {IC50_THRESHOLD} nM → {(df_filtered['Y']==0).sum()} ({(df_filtered['Y']==0).sum()/len(df_filtered)*100:.1f}%)")

    # Sanity checks
    active_count = (df_filtered['Y']==1).sum()
    inactive_count = (df_filtered['Y']==0).sum()
    active_pct = active_count / len(df_filtered) * 100

    print(f"\n🔍 Sanity Checks:")
    print(f"   Active % = {active_pct:.1f}% (Expected: 10-25% for kinase datasets)")
    print(f"   Censored samples (IC50=10000): {(df_filtered['IC50_nM']==10000).sum()} ({(df_filtered['IC50_nM']==10000).sum()/len(df_filtered)*100:.1f}%)")

    if active_pct > 30:
        print(f"     WARNING: Active % too high ({active_pct:.1f}%)")
        print(f"   Consider lowering threshold (e.g., 500 nM)")
    elif active_pct < 5:
        print(f"     WARNING: Active % too low ({active_pct:.1f}%)")
        print(f"   Consider raising threshold (e.g., 2000 nM)")
    else:
        print(f"   ✓ Active % looks reasonable for DTI classification")

    # Distribution of active samples by IC50 range
    print(f"\n IC50 Distribution (Active samples only):")
    active_ic50 = df_filtered[df_filtered['Y']==1]['IC50_nM']
    if len(active_ic50) > 0:
        print(f"   Min:    {active_ic50.min():.2f} nM")
        print(f"   25%:    {active_ic50.quantile(0.25):.2f} nM")
        print(f"   Median: {active_ic50.median():.2f} nM")
        print(f"   75%:    {active_ic50.quantile(0.75):.2f} nM")
        print(f"   Max:    {active_ic50.max():.2f} nM")


        # 8. Create final format
    final_df = pd.DataFrame({
        'Drug_ID': df_filtered['Drug_ID'],
        'Drug': df_filtered['Drug'],
        'Protein_ID': df_filtered['Protein_ID'],
        'Y': df_filtered['Y'],  # Binary label: 1=active (IC50<1000), 0=inactive
        'IC50_nM': df_filtered['IC50_nM']  # Keep original for reference/regression
    })

    final_df = final_df.dropna(subset=['Drug'])

    print(f"\n Final dataset ready:")
    print(f"   Total pairs: {len(final_df)}")
    print(f"   Active (binders): {(final_df['Y']==1).sum()}")
    print(f"   Inactive (non-binders): {(final_df['Y']==0).sum()}")
    print(f"   Missing SMILES: {final_df['Drug'].isna().sum()}")

    
    # 9. Create stratified splits (70/15/15) - INTERACTION-LEVEL SPLIT
    print(f"\n{'='*70}")
    print("CREATING SPLITS (Interaction-level, Random)")
    print(f"{'='*70}")
    print("  NOTE: This is a RANDOM split (not protein-based)")
    print("   - Splits INTERACTIONS randomly, not proteins")
    print("   - Same proteins appear in train/val/test (expected)")
    print("   - Good for Phase 1 POC: Tests multi-conf vs static")
    print("   - Phase 2 will use protein-based splits (unseen proteins)")
    
    train_df, temp_df = train_test_split(
        final_df, 
        test_size=0.3, 
        random_state=42,
        stratify=final_df['Y']  # Preserve class distribution
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42,
        stratify=temp_df['Y']
    )
    
    # 10. Print comprehensive statistics
    print(f"\n{'='*70}")
    print("SPLIT SUMMARY")
    print(f"{'='*70}")
    
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        active = (split_df['Y']==1).sum()
        inactive = (split_df['Y']==0).sum()
        print(f"{split_name:6s}: {len(split_df):4d} interactions ({len(split_df)/len(final_df)*100:5.1f}%)")
        print(f"        Active={active:4d} ({active/len(split_df)*100:5.1f}%), "
              f"Inactive={inactive:4d} ({inactive/len(split_df)*100:5.1f}%)")
        print(f"        Proteins={split_df['Protein_ID'].nunique():3d}, "
              f"Drugs={split_df['Drug_ID'].nunique():3d}")
    
    # 11. Check protein overlap (expected for random split)
    train_proteins = set(train_df['Protein_ID'].unique())
    val_proteins = set(val_df['Protein_ID'].unique())
    test_proteins = set(test_df['Protein_ID'].unique())
    
    print(f"\n{'='*70}")
    print("PROTEIN OVERLAP (Expected for random split)")
    print(f"{'='*70}")
    print(f"Train proteins:     {len(train_proteins)}")
    print(f"Val proteins:       {len(val_proteins)}")
    print(f"Test proteins:      {len(test_proteins)}")
    print(f"Train-Val overlap:  {len(train_proteins & val_proteins)} (normal for random split)")
    print(f"Train-Test overlap: {len(train_proteins & test_proteins)} (normal for random split)")
    print(f"Val-Test overlap:   {len(val_proteins & test_proteins)} (normal for random split)")
    
    # 12. Calculate class imbalance ratio
    pos_count = (final_df['Y']==1).sum()
    neg_count = (final_df['Y']==0).sum()
    imbalance_ratio = neg_count / pos_count
    
    print(f"\n{'='*70}")
    print("CLASS IMBALANCE HANDLING")
    print(f"{'='*70}")
    print(f"Positive (active):   {pos_count} samples")
    print(f"Negative (inactive): {neg_count} samples")
    print(f"Imbalance ratio:     {imbalance_ratio:.3f}")
    print(f"Recommended pos_weight: {imbalance_ratio:.3f}")
    print("✓ Training script will use BCEWithLogitsLoss(pos_weight=...)")
    
    # 13. Save splits
    output_dir = Path("data/splits/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "phase1_train_esm2.csv", index=False)
    val_df.to_csv(output_dir / "phase1_val_esm2.csv", index=False)
    test_df.to_csv(output_dir / "phase1_test_esm2.csv", index=False)
    final_df.to_csv(output_dir / "phase1_full_dataset.csv", index=False)
    
    print(f"\n{'='*70}")
    print("SAVED FILES")
    print(f"{'='*70}")
    print(f"✓ {output_dir / 'phase1_train_esm2.csv'}")
    print(f"✓ {output_dir / 'phase1_val_esm2.csv'}")
    print(f"✓ {output_dir / 'phase1_test_esm2.csv'}")
    print(f"✓ {output_dir / 'phase1_full_dataset.csv'}")
    
    # 14. Save detailed summary
    summary = f"""Phase 1 Dataset Summary
=======================
Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
----------------
Total Interactions: {len(final_df)}
Proteins (ESM-2):   {final_df['Protein_ID'].nunique()}
Drugs:              {final_df['Drug_ID'].nunique()}

BINARY LABELS (threshold={IC50_THRESHOLD})
-----------------------------------------
Active (Y=1):   {pos_count} ({pos_count/len(final_df)*100:.1f}%)
Inactive (Y=0): {neg_count} ({neg_count/len(final_df)*100:.1f}%)
Imbalance ratio: {imbalance_ratio:.3f}

SPLIT STRATEGY
--------------
Type: Random (interaction-level split)
Ratio: 70% train / 15% val / 15% test
Stratification: Yes (preserves class distribution)

Train: {len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%)
  Active={sum(train_df['Y']==1)}, Inactive={sum(train_df['Y']==0)}
  Proteins={train_df['Protein_ID'].nunique()}, Drugs={train_df['Drug_ID'].nunique()}

Val:   {len(val_df)} ({len(val_df)/len(final_df)*100:.1f}%)
  Active={sum(val_df['Y']==1)}, Inactive={sum(val_df['Y']==0)}
  Proteins={val_df['Protein_ID'].nunique()}, Drugs={val_df['Drug_ID'].nunique()}

Test:  {len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)
  Active={sum(test_df['Y']==1)}, Inactive={sum(test_df['Y']==0)}
  Proteins={test_df['Protein_ID'].nunique()}, Drugs={test_df['Drug_ID'].nunique()}

PROTEIN OVERLAP
---------------
This is EXPECTED for random splits (splits interactions, not proteins)
Train-Val overlap:  {len(train_proteins & val_proteins)} proteins
Train-Test overlap: {len(train_proteins & test_proteins)} proteins

Note: Phase 2 will use protein-based splits for generalization testing

CLASS IMBALANCE HANDLING
-------------------------
Recommended: BCEWithLogitsLoss(pos_weight={imbalance_ratio:.3f})
This downweights the majority class (active samples)

FILES
-----
- phase1_train_esm2.csv: Training set
- phase1_val_esm2.csv: Validation set  
- phase1_test_esm2.csv: Test set
- phase1_full_dataset.csv: Complete dataset

NEXT STEP
---------
Run training: python scripts/train_phase1.py
"""
    
    with open(output_dir / "phase1_summary.txt", "w") as f:
        f.write(summary)
    
    print(f"✓ {output_dir / 'phase1_summary.txt'}")
    
    print(f"\n{'='*70}")
    print("🎯 READY FOR TRAINING")
    print(f"{'='*70}")
    print("Update your training script with:")
    print('train_csv = "data/splits/phase1/phase1_train_esm2.csv"')
    print('val_csv   = "data/splits/phase1/phase1_val_esm2.csv"')
    print('test_csv  = "data/splits/phase1/phase1_test_esm2.csv"')
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
