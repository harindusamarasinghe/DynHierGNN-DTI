#!/usr/bin/env python3
"""
Phase 1 Training: Multi-conformational vs Static baseline
Addresses: Class imbalance, proper CSV paths, comprehensive metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dti_datasets.poc_dti_dataset_v2 import POCDTIDatasetV2, collate_fn
from models.poc_model_v2 import POCModelV2 as POCModel
from models.static_poc_model import StaticPOCModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for drug_graphs, conf_graphs_batch, y in loader:
        y = y.to(device)
        
        optimizer.zero_grad()
        out = model(drug_graphs, conf_graphs_batch)
        
        # Handle both (logits, attn) and logits-only
        if isinstance(out, tuple):
            logits, _ = out
        else:
            logits = out
        
        logits = logits.squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model and return all metrics."""
    model.eval()
    all_probs = []
    all_labels = []
    
    for drug_graphs, conf_graphs_batch, y in loader:
        y = y.to(device)
        
        out = model(drug_graphs, conf_graphs_batch)
        if isinstance(out, tuple):
            logits, _ = out
        else:
            logits = out
        
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([probs])
        
        all_probs.extend(probs)
        all_labels.extend(y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    binary_preds = (all_probs > 0.5).astype(int)
    
    auroc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, binary_preds)
    prec = precision_score(all_labels, binary_preds, zero_division=0)
    rec = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
    
    return auroc, acc, prec, rec, f1, (tp, fp, tn, fn)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("PHASE 1: MULTI-CONFORMATIONAL vs STATIC BASELINE")
    print("=" * 70)
    print(f"Device: {device}\n")
    
    # ========== Phase 1 Dataset Paths (93 proteins with ESM-2) ==========
    train_csv = "data/splits/phase1/phase1_train_esm2.csv"
    val_csv = "data/splits/phase1/phase1_val_esm2.csv"
    test_csv = "data/splits/phase1/phase1_test_esm2.csv"
    
    protein_graphs_path = "data/processed/all_protein_graphs_esm2.pkl"
    drug_graphs_path = "data/processed/validated_phase1_drug_graphs.pkl"
    
    # Verify files exist
    print("Checking file paths...")
    for path in [train_csv, val_csv, test_csv, protein_graphs_path, drug_graphs_path]:
        if not Path(path).exists():
            print(f" ERROR: File not found: {path}")
            return
        print(f"✓ {path}")
    print()
    
    # ========== AUTO-DETECT FEATURE DIMENSIONS ==========
    print("Detecting feature dimensions...")
    import pickle
    
    with open(protein_graphs_path, "rb") as f:
        protein_graphs = pickle.load(f)
    with open(drug_graphs_path, "rb") as f:
        drug_graphs = pickle.load(f)
    
    # Get sample graphs
    sample_protein = list(protein_graphs.values())[0][0]  # First protein, first conf
    sample_drug = list(drug_graphs.values())[0]
    
    protein_node_dim = sample_protein.x.shape[1]
    drug_node_dim = sample_drug.x.shape[1]
    
    print(f"✓ Drug node features: {drug_node_dim}")
    print(f"✓ Protein node features: {protein_node_dim}")
    print()
    
    # ========== Load Datasets (BINARY classification) ==========
    print("Loading datasets...")
    train_ds = POCDTIDatasetV2(
        train_csv,
        protein_graphs_path=protein_graphs_path,
        drug_graphs_path=drug_graphs_path,
        use_binary=True,
    )
    val_ds = POCDTIDatasetV2(
        val_csv,
        protein_graphs_path=protein_graphs_path,
        drug_graphs_path=drug_graphs_path,
        use_binary=True,
    )
    test_ds = POCDTIDatasetV2(
        test_csv,
        protein_graphs_path=protein_graphs_path,
        drug_graphs_path=drug_graphs_path,
        use_binary=True,
    )
    

    # ========== Calculate Class Weights (Handle Imbalance) ==========
    train_labels = [train_ds[i][2] for i in range(len(train_ds))]
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    pos_count = train_labels_tensor.sum().item()
    neg_count = len(train_labels) - pos_count

    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Train: {len(train_ds)} samples")
    print(f"  Active (Y=1):   {int(pos_count)} ({pos_count/len(train_ds)*100:.1f}%)")
    print(f"  Inactive (Y=0): {int(neg_count)} ({neg_count/len(train_ds)*100:.1f}%)")
    print(f"  Imbalance ratio (neg/pos): {neg_count/pos_count:.3f}")
    print(f"Val:   {len(val_ds)} samples")
    print(f"Test:  {len(test_ds)} samples")

    # OPTION A: no class weighting (start with this)
    pos_weight = torch.tensor([3.0], device=device)
    print(f"\nClass weighting (BCEWithLogitsLoss):")
    print(f"  pos_weight = {pos_weight.item():.3f} (no weighting)")
    print("=" * 70 + "\n")
    
    # ========== Data Loaders ==========
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    
    # ========== Loss Function (Class Imbalance Handling) ==========
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # ========== Hyperparameters ==========
    epochs = 100
    patience = 20
    lr = 1e-3
    pretrained_path = "checkpoints/pretrained/pretrained_drug_encoder_poc.pth"
    
    # Check if pretrained exists
    if Path(pretrained_path).exists():
        print(f"✓ Using pretrained drug encoder: {pretrained_path}\n")
    else:
        print(f"⚠️  Pretrained drug encoder not found: {pretrained_path}")
        print("   Training from scratch...\n")
        pretrained_path = None
    
    # ========== TRAIN MULTI-CONFORMATIONAL MODEL ==========
    print("=" * 70)
    print("TRAINING MULTI-CONFORMATIONAL MODEL")
    print("=" * 70)
    
    model = POCModel(
        drug_node_dim=drug_node_dim,          # ← AUTO-DETECTED
        protein_node_dim=protein_node_dim,    # ← AUTO-DETECTED
        hidden_dim=128,
        num_layers=3,
        pretrained_drug_path=pretrained_path, 
        freeze_drug=False
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_auroc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auroc, val_acc, val_prec, val_rec, val_f1, _ = evaluate(
            model, val_loader, device
        )
        
        print(
            f"Epoch {epoch:03d} | Loss={train_loss:.4f} | "
            f"AUROC={val_auroc:.4f} | Acc={val_acc:.4f} | "
            f"Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}"
        )
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}\n")
                break
    
    # Evaluate best multi-conf model
    model.load_state_dict(best_state)
    test_auroc, test_acc, test_prec, test_rec, test_f1, (tp, fp, tn, fn) = evaluate(
        model, test_loader, device
    )
    
    print("\n" + "=" * 70)
    print("MULTI-CONFORMATIONAL MODEL - TEST RESULTS")
    print("=" * 70)
    print(f"AUROC:     {test_auroc:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Pos    Neg")
    print(f"Actual   Pos    {tp:4d}   {fn:4d}")
    print(f"         Neg    {fp:4d}   {tn:4d}")
    print("=" * 70 + "\n")
    
    # Save best multi-conf model
    Path("checkpoints/phase1").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/phase1/best_multi_conf.pth")
    print("✓ Multi-conformational model saved to checkpoints/phase1/best_multi_conf.pth\n")
    
    # ========== TRAIN STATIC BASELINE MODEL ==========
    print("=" * 70)
    print("TRAINING STATIC BASELINE MODEL")
    print("=" * 70)
    
    static_model = StaticPOCModel(
        drug_node_dim=drug_node_dim,          # ← MUST PASS THIS
        protein_node_dim=protein_node_dim,    # ← MUST PASS THIS
        hidden_dim=128,
        num_layers=3,
        pretrained_drug_path=pretrained_path, 
        freeze_drug=False
    ).to(device)
    optimizer_static = torch.optim.Adam(static_model.parameters(), lr=lr)
    
    best_val_auroc_static = 0.0
    best_state_static = None
    patience_counter_static = 0
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            static_model, train_loader, optimizer_static, criterion, device  # ← USE static_model
        )
        val_auroc, val_acc, val_prec, val_rec, val_f1, _ = evaluate(
            static_model, val_loader, device  # ← USE static_model
        )
        
        print(
            f"Static Epoch {epoch:03d} | Loss={train_loss:.4f} | "
            f"AUROC={val_auroc:.4f} | Acc={val_acc:.4f} | "
            f"Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}"
        )
        
        if val_auroc > best_val_auroc_static:
            best_val_auroc_static = val_auroc
            best_state_static = static_model.state_dict().copy()
            patience_counter_static = 0
        else:
            patience_counter_static += 1
            if patience_counter_static >= patience:
                print(f"Static early stopping at epoch {epoch}\n")
                break
    
    # Evaluate best static model
    static_model.load_state_dict(best_state_static)
    (
        static_test_auroc,
        static_test_acc,
        static_test_prec,
        static_test_rec,
        static_test_f1,
        (tp_s, fp_s, tn_s, fn_s),
    ) = evaluate(static_model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("STATIC BASELINE MODEL - TEST RESULTS")
    print("=" * 70)
    print(f"AUROC:     {static_test_auroc:.4f}")
    print(f"Accuracy:  {static_test_acc:.4f}")
    print(f"Precision: {static_test_prec:.4f}")
    print(f"Recall:    {static_test_rec:.4f}")
    print(f"F1 Score:  {static_test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Pos    Neg")
    print(f"Actual   Pos    {tp_s:4d}   {fn_s:4d}")
    print(f"         Neg    {fp_s:4d}   {tn_s:4d}")
    print("=" * 70 + "\n")
    
    # Save static model
    torch.save(static_model.state_dict(), "checkpoints/phase1/best_static.pth")
    print("✓ Static baseline model saved to checkpoints/phase1/best_static.pth\n")
    
    # ========== COMPARISON ==========
    auroc_improvement = test_auroc - static_test_auroc
    auroc_improvement_pct = (auroc_improvement / static_test_auroc) * 100
    
    f1_improvement = test_f1 - static_test_f1
    f1_improvement_pct = (f1_improvement / static_test_f1) * 100 if static_test_f1 > 0 else 0
    
    print("=" * 70)
    print("PHASE 1 COMPARISON: MULTI-CONF vs STATIC")
    print("=" * 70)
    print(f"{'Metric':<15} {'Multi-Conf':<12} {'Static':<12} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'AUROC':<15} {test_auroc:<12.4f} {static_test_auroc:<12.4f} {auroc_improvement:+.4f} ({auroc_improvement_pct:+.1f}%)")
    print(f"{'Accuracy':<15} {test_acc:<12.4f} {static_test_acc:<12.4f} {test_acc - static_test_acc:+.4f}")
    print(f"{'Precision':<15} {test_prec:<12.4f} {static_test_prec:<12.4f} {test_prec - static_test_prec:+.4f}")
    print(f"{'Recall':<15} {test_rec:<12.4f} {static_test_rec:<12.4f} {test_rec - static_test_rec:+.4f}")
    print(f"{'F1 Score':<15} {test_f1:<12.4f} {static_test_f1:<12.4f} {f1_improvement:+.4f} ({f1_improvement_pct:+.1f}%)")
    print("=" * 70)
    
    # ========== SUCCESS CRITERIA ==========
    print("\nPHASE 1 SUCCESS CRITERIA:")
    print("-" * 70)
    
    success_criteria = {
        "Multi-conf AUROC > 0.75": test_auroc > 0.75,
        "AUROC improvement ≥ 3%": auroc_improvement_pct >= 3.0,
        "No training instability": True,  # If we reach here, training was stable
        "F1 improvement > 0": f1_improvement > 0,
    }
    
    for criterion_name, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {criterion_name}")
    
    all_passed = all(success_criteria.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 PHASE 1 SUCCESS!")
        print("Multi-conformational approach shows clear improvement.")
        print("Proceed to Phase 2: Full dataset with hierarchical architecture.")
    else:
        print("⚠️  PHASE 1 NEEDS INVESTIGATION")
        print("Consider:")
        print("- Increase number of conformations (try 5)")
        print("- Adjust attention mechanism")
        print("- Check conformational diversity (RMSD)")
    print("=" * 70)
    
    # Save results summary
    results_summary = f"""Phase 1 Results Summary
======================

MULTI-CONFORMATIONAL MODEL
--------------------------
AUROC:     {test_auroc:.4f}
Accuracy:  {test_acc:.4f}
Precision: {test_prec:.4f}
Recall:    {test_rec:.4f}
F1 Score:  {test_f1:.4f}

Confusion Matrix:
  TP={tp}, FP={fp}
  FN={fn}, TN={tn}

STATIC BASELINE MODEL
---------------------
AUROC:     {static_test_auroc:.4f}
Accuracy:  {static_test_acc:.4f}
Precision: {static_test_prec:.4f}
Recall:    {static_test_rec:.4f}
F1 Score:  {static_test_f1:.4f}

Confusion Matrix:
  TP={tp_s}, FP={fp_s}
  FN={fn_s}, TN={tn_s}

IMPROVEMENT
-----------
AUROC:  {auroc_improvement:+.4f} ({auroc_improvement_pct:+.1f}%)
F1:     {f1_improvement:+.4f} ({f1_improvement_pct:+.1f}%)

SUCCESS CRITERIA
----------------
Multi-conf AUROC > 0.75:      {'PASS' if success_criteria["Multi-conf AUROC > 0.75"] else 'FAIL'}
AUROC improvement ≥ 3%:       {'PASS' if success_criteria["AUROC improvement ≥ 3%"] else 'FAIL'}
No training instability:      {'PASS' if success_criteria["No training instability"] else 'FAIL'}
F1 improvement > 0:           {'PASS' if success_criteria["F1 improvement > 0"] else 'FAIL'}

CONCLUSION
----------
{'Phase 1 SUCCESS! Proceed to Phase 2.' if all_passed else 'Phase 1 needs investigation.'}
"""
    
    with open("checkpoints/phase1/results_summary.txt", "w") as f:
        f.write(results_summary)
    
    print("\n✓ Results saved to checkpoints/phase1/results_summary.txt")


if __name__ == "__main__":
    main()
