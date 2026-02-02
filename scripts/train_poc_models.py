"""
Train multi-conformational POC model and static baseline for BINARY DTI prediction.
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
    print(f"Using device: {device}")

    # Paths to splits
    train_csv = "data/poc/poc_train.csv"
    val_csv = "data/poc/poc_val.csv"
    test_csv = "data/poc/poc_test.csv"

    # Load datasets (BINARY classification)
    train_ds = POCDTIDatasetV2(
        train_csv,
        protein_graphs_path="data/processed/poc_protein_graphs.pkl",
        drug_graphs_path="data/processed/poc_drug_graphs.pkl",
        use_binary=True,
    )
    val_ds = POCDTIDatasetV2(
        val_csv,
        protein_graphs_path="data/processed/poc_protein_graphs.pkl",
        drug_graphs_path="data/processed/poc_drug_graphs.pkl",
        use_binary=True,
    )
    test_ds = POCDTIDatasetV2(
        test_csv,
        protein_graphs_path="data/processed/poc_protein_graphs.pkl",
        drug_graphs_path="data/processed/poc_drug_graphs.pkl",
        use_binary=True,
    )

    # Calculate class weights
    train_labels = [train_ds[i][2] for i in range(len(train_ds))]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=device)

    print("\n=== Dataset Statistics ===")
    print(f"Train: {len(train_ds)} samples ({pos_count} active, {neg_count} inactive)")
    print(f"Val:   {len(val_ds)} samples")
    print(f"Test:  {len(test_ds)} samples")
    print(f"Positive class weight: {pos_weight.item():.2f}\n")

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    pretrained_path = "checkpoints/pretrained/pretrained_drug_encoder_poc.pth"

    # Binary classification loss with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Shared training hyperparameters
    epochs = 100
    patience = 20

    # ========== Multi-conformational model ==========
    print("=" * 60)
    print("Training Multi-Conformational Model")
    print("=" * 60)

    model = POCModel(pretrained_drug_path=pretrained_path, freeze_drug=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluate best multi-conf model
    model.load_state_dict(best_state)
    test_auroc, test_acc, test_prec, test_rec, test_f1, (tp, fp, tn, fn) = evaluate(
        model, test_loader, device
    )

    print("\n" + "=" * 60)
    print("Multi-Conformational Model - TEST RESULTS")
    print("=" * 60)
    print(f"AUROC:     {test_auroc:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP={tp}, FP={fp}")
    print(f"  FN={fn}, TN={tn}")
    print("=" * 60 + "\n")

    # Save best multi-conf model
    Path("checkpoints/poc").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/poc/best_multi_conf_poc.pth")
    print("✓ Multi-conformational model saved to checkpoints/poc/best_multi_conf_poc.pth\n")

    # ========== Static baseline model ==========
    print("=" * 60)
    print("Training Static Baseline Model")
    print("=" * 60)

    static_model = StaticPOCModel(
        pretrained_drug_path=pretrained_path, freeze_drug=False
    ).to(device)
    optimizer = torch.optim.Adam(static_model.parameters(), lr=1e-3)

    best_val_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            static_model, train_loader, optimizer, criterion, device
        )
        val_auroc, val_acc, val_prec, val_rec, val_f1, _ = evaluate(
            static_model, val_loader, device
        )

        print(
            f"Static Epoch {epoch:03d} | Loss={train_loss:.4f} | "
            f"AUROC={val_auroc:.4f} | Acc={val_acc:.4f} | "
            f"Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = static_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Static early stopping at epoch {epoch}")
                break

    # Evaluate best static model
    static_model.load_state_dict(best_state)
    (
        static_test_auroc,
        static_test_acc,
        static_test_prec,
        static_test_rec,
        static_test_f1,
        (tp_s, fp_s, tn_s, fn_s),
    ) = evaluate(static_model, test_loader, device)

    print("\n" + "=" * 60)
    print("Static Baseline Model - TEST RESULTS")
    print("=" * 60)
    print(f"AUROC:     {static_test_auroc:.4f}")
    print(f"Accuracy:  {static_test_acc:.4f}")
    print(f"Precision: {static_test_prec:.4f}")
    print(f"Recall:    {static_test_rec:.4f}")
    print(f"F1 Score:  {static_test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP={tp_s}, FP={fp_s}")
    print(f"  FN={fn_s}, TN={tn_s}")
    print("=" * 60)

    # Save static model
    torch.save(static_model.state_dict(), "checkpoints/poc/best_static_poc.pth")
    print("✓ Static baseline model saved to checkpoints/poc/best_static_poc.pth")


if __name__ == "__main__":
    main()
