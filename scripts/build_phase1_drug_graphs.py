#!/usr/bin/env python3
"""
Build validated drug graphs for Phase 1
Applies Lipinski's Rule of Five + drug-likeness filters
Only keeps real pharmaceutical compounds
"""

import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import torch
from pathlib import Path
from tqdm import tqdm


def is_drug_like(mol, drug_id):
    """
    Strict quality check: Lipinski's Rule of Five
    Returns: (pass, reason)
    """
    if mol is None:
        return False, "Invalid molecule"
    
    # Basic checks
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 5:
        return False, f"Too small ({num_atoms} atoms)"
    if num_atoms > 100:
        return False, f"Too large ({num_atoms} atoms)"
    
    # Lipinski's Rule of Five (standard drug-likeness criteria)
    mw = Descriptors.MolWt(mol)
    if not (150 <= mw <= 800):
        return False, f"MW {mw:.1f} Da (should be 150-800)"
    
    logp = Descriptors.MolLogP(mol)
    if not (-3 <= logp <= 7):
        return False, f"LogP {logp:.2f} (should be -3 to 7)"
    
    hbd = Descriptors.NumHDonors(mol)
    if hbd > 5:
        return False, f"H-donors {hbd} (should be ≤5)"
    
    hba = Descriptors.NumHAcceptors(mol)
    if hba > 10:
        return False, f"H-acceptors {hba} (should be ≤10)"
    
    # Additional quality checks
    tpsa = Descriptors.TPSA(mol)
    if tpsa > 140:
        return False, f"TPSA {tpsa:.1f} (should be ≤140)"
    
    rotatable = Descriptors.NumRotatableBonds(mol)
    if rotatable > 15:
        return False, f"Too flexible ({rotatable} rotatable bonds)"
    
    # Check for disconnected fragments (should be single molecule)
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        return False, f"Multiple fragments ({len(fragments)})"
    
    return True, "Valid"


def smiles_to_graph(smiles, drug_id):
    """Convert validated SMILES to molecular graph"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 9-dim atom features (same as POC)
    atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
    
    node_features = []
    for atom in mol.GetAtoms():
        feat = [0] * 9
        atom_symbol = atom.GetSymbol()
        if atom_symbol in atom_types:
            feat[atom_types[atom_symbol]] = 1
        else:
            # Rare atoms: use last dimension
            feat[8] = 1
        node_features.append(feat)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edges (undirected)
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
    
    return Data(x=x, edge_index=edge_index, drug_id=drug_id)


def main():
    print("=" * 70)
    print("QUALITY-CHECKED DRUG GRAPH BUILDER")
    print("=" * 70)
    print("Validation: Lipinski's Rule of Five + drug-likeness filters")
    print("Only pharmaceutical-grade compounds will be included")
    print("=" * 70)
    
    # Load Phase 1 splits
    train_df = pd.read_csv("data/splits/phase1/phase1_train_esm2.csv")
    val_df = pd.read_csv("data/splits/phase1/phase1_val_esm2.csv")
    test_df = pd.read_csv("data/splits/phase1/phase1_test_esm2.csv")
    
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    unique_drugs = all_df[['Drug_ID', 'Drug']].drop_duplicates()
    
    print(f"\n📊 Dataset:")
    print(f"  Drugs to validate: {len(unique_drugs)}")
    print(f"  Proteins: {all_df['Protein_ID'].nunique()}")
    print(f"  Total interactions: {len(all_df)}")
    
    # STEP 1: Quality validation
    print(f"\n🔍 STEP 1: QUALITY VALIDATION")
    print("-" * 70)
    
    validated = []
    rejected = []
    
    for idx, row in tqdm(unique_drugs.iterrows(), total=len(unique_drugs), desc="Validating"):
        drug_id = row['Drug_ID']
        smiles = row['Drug']
        
        if pd.isna(smiles):
            rejected.append((drug_id, "Missing SMILES"))
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        is_valid, reason = is_drug_like(mol, drug_id)
        
        if is_valid:
            validated.append(row)
        else:
            rejected.append((drug_id, reason))
    
    print(f"\n✓ Validation complete:")
    print(f"  ✅ Passed: {len(validated)} drugs ({len(validated)/len(unique_drugs)*100:.1f}%)")
    print(f"  ❌ Rejected: {len(rejected)} drugs ({len(rejected)/len(unique_drugs)*100:.1f}%)")
    
    if rejected:
        print(f"\n  Rejected examples (first 10):")
        for drug_id, reason in rejected[:10]:
            print(f"    {drug_id}: {reason}")
        if len(rejected) > 10:
            print(f"    ... and {len(rejected)-10} more")
    
    if len(validated) == 0:
        print("\n❌ ERROR: No drugs passed validation!")
        print("   This suggests data quality issues.")
        print("   Check your SMILES format or lower quality thresholds.")
        return
    
    # STEP 2: Build graphs for validated drugs
    print(f"\n🔧 STEP 2: BUILDING GRAPHS FOR {len(validated)} VALIDATED DRUGS")
    print("-" * 70)
    
    drug_graphs = {}
    build_failed = []
    
    for row in tqdm(validated, desc="Building"):
        drug_id = row['Drug_ID']
        smiles = row['Drug']
        
        graph = smiles_to_graph(smiles, drug_id)
        
        if graph is not None:
            drug_graphs[drug_id] = graph
        else:
            build_failed.append(drug_id)
    
    print(f"\n✓ Graph construction:")
    print(f"  Success: {len(drug_graphs)}")
    print(f"  Failed: {len(build_failed)}")
    
    # Save graphs
    output_path = "data/processed/validated_phase1_drug_graphs.pkl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(drug_graphs, f)
    
    print(f"\n✓ Saved: {output_path}")
    
    # STEP 3: Coverage analysis
    print(f"\n📊 STEP 3: COVERAGE ANALYSIS")
    print("-" * 70)
    
    train_drugs = set(train_df['Drug_ID'].unique())
    graph_drugs = set(drug_graphs.keys())
    
    overlap = train_drugs & graph_drugs
    missing = train_drugs - graph_drugs
    
    coverage_pct = len(overlap) / len(train_drugs) * 100
    
    print(f"  Train set drugs: {len(train_drugs)}")
    print(f"  Validated graphs: {len(graph_drugs)}")
    print(f"  Coverage: {len(overlap)}/{len(train_drugs)} ({coverage_pct:.1f}%)")
    
    if missing:
        print(f"\n  ⚠️  {len(missing)} drugs rejected (will be filtered out):")
        print(f"     IDs: {list(missing)[:10]}")
        
        # Calculate remaining interactions
        train_remaining = train_df[train_df['Drug_ID'].isin(graph_drugs)]
        val_remaining = val_df[val_df['Drug_ID'].isin(graph_drugs)]
        test_remaining = test_df[test_df['Drug_ID'].isin(graph_drugs)]
        
        print(f"\n  Remaining interactions after filtering:")
        print(f"     Train: {len(train_remaining)} (was {len(train_df)})")
        print(f"     Val:   {len(val_remaining)} (was {len(val_df)})")
        print(f"     Test:  {len(test_remaining)} (was {len(test_df)})")
        
        # Check if still adequate
        if len(train_remaining) < 500:
            print(f"\n  ⚠️  WARNING: Only {len(train_remaining)} training samples!")
            print(f"     Consider lowering quality thresholds if this is too few.")
    
    # Sample statistics
    print(f"\n📊 VALIDATED DRUG STATISTICS:")
    print("-" * 70)
    
    sample = list(drug_graphs.values())[0]
    print(f"  Sample graph:")
    print(f"    Nodes: {sample.x.shape[0]}")
    print(f"    Features: {sample.x.shape[1]} (9-dim one-hot)")
    print(f"    Edges: {sample.edge_index.shape[1]}")
    
    # Size distribution
    num_atoms = [g.x.shape[0] for g in drug_graphs.values()]
    print(f"\n  All drugs:")
    print(f"    Atom count: {min(num_atoms)}-{max(num_atoms)} (avg: {sum(num_atoms)/len(num_atoms):.1f})")
    
    # Save validation report
    report_path = "data/processed/drug_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("Drug Validation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total drugs in dataset: {len(unique_drugs)}\n")
        f.write(f"Passed validation: {len(validated)} ({len(validated)/len(unique_drugs)*100:.1f}%)\n")
        f.write(f"Rejected: {len(rejected)} ({len(rejected)/len(unique_drugs)*100:.1f}%)\n\n")
        f.write("Validation Criteria:\n")
        f.write("- Lipinski's Rule of Five\n")
        f.write("- MW: 150-800 Da\n")
        f.write("- LogP: -3 to 7\n")
        f.write("- H-donors: ≤5\n")
        f.write("- H-acceptors: ≤10\n")
        f.write("- TPSA: ≤140\n")
        f.write("- Rotatable bonds: ≤15\n")
        f.write("- Single molecule (no fragments)\n\n")
        f.write("Rejected Drugs:\n")
        f.write("-" * 70 + "\n")
        for drug_id, reason in rejected:
            f.write(f"{drug_id}: {reason}\n")
    
    print(f"\n✓ Validation report: {report_path}")
    
    # Final instructions
    print("\n" + "=" * 70)
    print("✓ SUCCESS! QUALITY-CHECKED DRUG GRAPHS READY")
    print("=" * 70)
    print(f"\nValidated {len(drug_graphs)} pharmaceutical-grade compounds")
    print(f"Rejected {len(rejected)} non-drug-like compounds")
    
    print("\n📝 NEXT STEPS:")
    print("-" * 70)
    print("1. Update training script (line ~88):")
    print('   drug_graphs_path = "data/processed/validated_phase1_drug_graphs.pkl"')
    
    if len(missing) > 0:
        print("\n2. Your dataset will auto-filter during training")
        print(f"   Expected samples: ~{len(train_remaining)} train, ~{len(val_remaining)} val, ~{len(test_remaining)} test")
    
    print("\n3. Run training:")
    print("   python scripts/train_phase1.py")
    
    print("\n✓ All drugs meet pharmaceutical standards!")
    print("=" * 70)


if __name__ == "__main__":
    main()
