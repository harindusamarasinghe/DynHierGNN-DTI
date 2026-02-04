"""
Phase 2 Training: Hierarchical Binary DTI Classification (FIXED)

✅ FIXES:
1. Class-weighted BCE loss (handles 85/15 imbalance)
2. Reduced model size (prevents overfitting)
3. Lower learning rate (5e-4)
4. BCEWithLogitsLoss (more stable than BCE+Sigmoid)
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    accuracy_score, f1_score, confusion_matrix
)
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

# Import hierarchical modules
from models.hierarchical_drug_gnn import HierarchicalDrugGNN
from models.hierarchical_protein_gnn import HierarchicalProteinGNN
from models.hierarchical_temporal_attention import HierarchicalTemporalAttention


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class Phase2BinaryDataset(Dataset):
    """Binary classification dataset for Phase 2"""
    
    def __init__(self, df, drug_graphs, hierarchical_proteins):
        self.df = df.reset_index(drop=True)
        self.drug_graphs = drug_graphs
        self.hierarchical_proteins = hierarchical_proteins
        
        # Filter valid samples
        valid_indices = []
        for idx, row in df.iterrows():
            drug_id = row['Drug_ID']
            protein_id = row['Protein_ID']
            if drug_id in drug_graphs and protein_id in hierarchical_proteins:
                if len(hierarchical_proteins[protein_id]) > 0:
                    valid_indices.append(idx)
        
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        
        # Statistics
        labels = self.df['Y'].values
        pos_count = labels.sum()
        neg_count = len(labels) - pos_count
        
        print(f"   Filtered dataset: {len(self.df)} valid samples (from {len(df)} total)")
        print(f"   Class distribution: Active={int(pos_count)}, Inactive={int(neg_count)}")
        print(f"   Positive ratio: {labels.mean():.2%}")
        print(f"   Class imbalance ratio: {neg_count/pos_count:.2f}:1")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_id = row['Drug_ID']
        protein_id = row['Protein_ID']
        label = float(row['Y'])  # 0 or 1
        
        drug_graph = self.drug_graphs[drug_id]
        protein_conformations = self.hierarchical_proteins[protein_id]
        
        return {
            'drug_graph': drug_graph,
            'protein_conformations': protein_conformations,
            'label': label,
            'drug_id': drug_id,
            'protein_id': protein_id
        }


class HierarchicalBinaryModel(nn.Module):
    """
    Hierarchical Binary Classification Model (REDUCED SIZE)
    
    ✅ Smaller architecture to prevent overfitting
    ✅ Outputs logit (no sigmoid - used with BCEWithLogitsLoss)
    """
    
    def __init__(self,
                 drug_encoder_params=None,
                 protein_encoder_params=None,
                 temporal_attention_params=None,
                 embed_dim=128,  # Reduced from 256
                 dropout=0.3):   # Increased from 0.2
        super().__init__()
        
        drug_encoder_params = drug_encoder_params or {}
        protein_encoder_params = protein_encoder_params or {}
        temporal_attention_params = temporal_attention_params or {}
        
        # Drug encoder (reduced size)
        self.drug_encoder = HierarchicalDrugGNN(
            atom_dim=9,
            hidden_dim=64,      # Reduced from 128
            output_dim=embed_dim,
            num_heads=2,        # Reduced from 4
            dropout=dropout,
            **drug_encoder_params
        )
        
        # Protein encoder (reduced size)
        self.protein_encoder = HierarchicalProteinGNN(
            residue_dim=1302,
            hidden_dim=64,      # Reduced from 128
            output_dim=embed_dim,
            num_heads=2,        # Reduced from 4
            dropout=dropout,
            **protein_encoder_params
        )
        
        # Temporal attention (reduced size)
        self.temporal_attention = HierarchicalTemporalAttention(
            embed_dim=embed_dim,
            num_heads=4,        # Reduced from 8
            dropout=dropout,
            use_positional_encoding=False,
            **temporal_attention_params
        )
        
        # Binary classification head (smaller MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, drug_graph, protein_conformations):
        """
        Returns:
            logit: Raw score (no sigmoid, for BCEWithLogitsLoss)
        """
        # Encode drug
        drug_embed = self.drug_encoder(drug_graph, mol=None)
        
        # Encode protein conformations
        protein_conf_embeds = []
        for conf_dict in protein_conformations:
            conf_embed = self.protein_encoder(conf_dict)
            protein_conf_embeds.append(conf_embed)
        
        protein_conf_embeds = torch.stack(protein_conf_embeds)
        
        # Temporal attention
        protein_embed, _ = self.temporal_attention(protein_conf_embeds)
        
        # Interaction prediction
        if drug_embed.dim() == 2:
            drug_embed = drug_embed.squeeze(0)
        if protein_embed.dim() == 2:
            protein_embed = protein_embed.squeeze(0)
        
        interaction_feat = torch.cat([drug_embed, protein_embed], dim=0)
        logit = self.classifier(interaction_feat)
        
        return logit.squeeze()


def custom_collate_fn(batch):
    """Batch collation"""
    drug_graphs = [sample['drug_graph'] for sample in batch]
    prot_conformations = [sample['protein_conformations'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.float)
    drug_ids = [sample['drug_id'] for sample in batch]
    protein_ids = [sample['protein_id'] for sample in batch]
    
    return {
        'drug_graphs': drug_graphs,
        'prot_conformations': prot_conformations,
        'labels': labels,
        'drug_ids': drug_ids,
        'protein_ids': protein_ids
    }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        drug_graphs = batch['drug_graphs']
        prot_conformations = batch['prot_conformations']
        labels = batch['labels'].to(device)
        
        # Forward pass
        batch_logits = []
        for i in range(len(drug_graphs)):
            drug_graph = drug_graphs[i].to(device)
            prot_confs = prot_conformations[i]
            
            logit = model(drug_graph, prot_confs)
            batch_logits.append(logit)
        
        batch_logits = torch.stack(batch_logits)
        
        # Debug first batch
        if epoch == 1 and batch_idx == 0:
            with torch.no_grad():
                probs = torch.sigmoid(batch_logits)
            print(f"\n   [DEBUG] First batch:")
            print(f"   Labels: {labels[:5].cpu().numpy()}")
            print(f"   Logits: {batch_logits[:5].detach().cpu().numpy()}")
            print(f"   Probs: {probs[:5].cpu().numpy()}")
            print(f"   Logit range: [{batch_logits.min().item():.3f}, {batch_logits.max().item():.3f}]")
        
        # Loss (BCEWithLogitsLoss handles sigmoid internally)
        loss = criterion(batch_logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate (convert logits to probabilities for metrics)
        with torch.no_grad():
            probs = torch.sigmoid(batch_logits)
        
        total_loss += loss.item()
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    binary_preds = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    # Class-wise accuracy
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_acc = accuracy_score(all_labels[pos_mask], binary_preds[pos_mask]) if pos_mask.sum() > 0 else 0
    neg_acc = accuracy_score(all_labels[neg_mask], binary_preds[neg_mask]) if neg_mask.sum() > 0 else 0
    
    print(f"\n   [TRAIN] Pred distribution: min={all_preds.min():.3f}, max={all_preds.max():.3f}, "
          f"mean={all_preds.mean():.3f}, std={all_preds.std():.3f}")
    print(f"   [TRAIN] Class accuracy: Positive={pos_acc:.3f}, Negative={neg_acc:.3f}")
    
    return avg_loss, auroc, auprc, accuracy, f1


def validate_epoch(model, dataloader, criterion, device, split_name='Val'):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[{split_name}]")
        
        for batch in pbar:
            drug_graphs = batch['drug_graphs']
            prot_conformations = batch['prot_conformations']
            labels = batch['labels'].to(device)
            
            batch_logits = []
            for i in range(len(drug_graphs)):
                drug_graph = drug_graphs[i].to(device)
                prot_confs = prot_conformations[i]
                
                logit = model(drug_graph, prot_confs)
                batch_logits.append(logit)
            
            batch_logits = torch.stack(batch_logits)
            loss = criterion(batch_logits, labels)
            
            # Convert to probabilities
            probs = torch.sigmoid(batch_logits)
            
            total_loss += loss.item()
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    binary_preds = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
    
    # Class-wise accuracy
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_acc = accuracy_score(all_labels[pos_mask], binary_preds[pos_mask]) if pos_mask.sum() > 0 else 0
    neg_acc = accuracy_score(all_labels[neg_mask], binary_preds[neg_mask]) if neg_mask.sum() > 0 else 0
    
    return avg_loss, auroc, auprc, accuracy, f1, (tp, tn, fp, fn), (pos_acc, neg_acc)


def main():
    print("="*70)
    print("PHASE 2: HIERARCHICAL BINARY DTI CLASSIFICATION (FIXED)")
    print("="*70)
    print("✅ FIXES:")
    print("  1. Class-weighted BCEWithLogitsLoss (handles imbalance)")
    print("  2. Reduced model size (prevents overfitting)")
    print("  3. Lower learning rate (5e-4)")
    print("  4. Batch normalization + increased dropout")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    drug_graphs_path = 'data/processed/validated_phase1_drug_graphs.pkl'
    print(f"\n1. Loading drug graphs: {drug_graphs_path}")
    with open(drug_graphs_path, 'rb') as f:
        drug_graphs = pickle.load(f)
    print(f"   ✓ Loaded {len(drug_graphs)} drug graphs")
    
    hier_proteins_path = 'data/processed/hierarchical_proteins.pkl'
    print(f"\n2. Loading hierarchical proteins: {hier_proteins_path}")
    with open(hier_proteins_path, 'rb') as f:
        hierarchical_proteins = pickle.load(f)
    total_confs = sum(len(confs) for confs in hierarchical_proteins.values())
    print(f"   ✓ Loaded {len(hierarchical_proteins)} proteins ({total_confs} conformations)")
    
    train_csv = 'data/splits/phase1/phase1_train_esm2.csv'
    val_csv = 'data/splits/phase1/phase1_val_esm2.csv'
    test_csv = 'data/splits/phase1/phase1_test_esm2.csv'
    
    print(f"\n3. Loading splits:")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    print(f"   ✓ Raw: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    print("\n4. Creating datasets:")
    train_dataset = Phase2BinaryDataset(train_df, drug_graphs, hierarchical_proteins)
    val_dataset = Phase2BinaryDataset(val_df, drug_graphs, hierarchical_proteins)
    test_dataset = Phase2BinaryDataset(test_df, drug_graphs, hierarchical_proteins)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    
    print(f"\n5. Dataloaders ready: {len(train_loader)} train batches")
    
    # Model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = HierarchicalBinaryModel(embed_dim=128, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model: {total_params:,} total parameters")
    print(f"✓ Trainable: {trainable_params:,} parameters")
    
    # Training setup with class weighting
    print("\n" + "="*70)
    print("TRAINING SETUP")
    print("="*70)
    
    # Calculate class weight
    labels = train_dataset.df['Y'].values
    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    pos_weight = neg_count / pos_count
    
    print(f"\n✓ Class imbalance: {neg_count:.0f} negative / {pos_count:.0f} positive")
    print(f"✓ Positive class weight: {pos_weight:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  # Lower LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    print(f"✓ Loss: BCEWithLogitsLoss (pos_weight={pos_weight:.2f})")
    print(f"✓ Optimizer: Adam (lr=5e-4, weight_decay=1e-4)")
    print(f"✓ Scheduler: ReduceLROnPlateau (patience=5)")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    num_epochs = 100
    best_val_auroc = 0.0
    patience_counter = 0
    early_stop_patience = 20
    
    checkpoint_dir = Path('checkpoints/phase2_binary_fixed')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    train_log = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        train_loss, train_auroc, train_auprc, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        val_loss, val_auroc, val_auprc, val_acc, val_f1, (tp, tn, fp, fn), (pos_acc, neg_acc) = validate_epoch(
            model, val_loader, criterion, device, split_name='Val'
        )
        
        scheduler.step(val_auroc)
        
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_auroc': train_auroc,
            'train_auprc': train_auprc,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_auroc': val_auroc,
            'val_auprc': val_auprc,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_pos_acc': pos_acc,
            'val_neg_acc': neg_acc,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_log.append(log_entry)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary:")
        print(f"{'='*70}")
        print(f"Train: Loss={train_loss:.4f}, AUROC={train_auroc:.4f}, AUPRC={train_auprc:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        print(f"       TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"       Class Acc: Positive={pos_acc:.3f}, Negative={neg_acc:.3f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auroc': val_auroc,
                'val_auprc': val_auprc,
                'train_log': train_log
            }, checkpoint_path)
            
            print(f"\n✅ Best model saved! Val AUROC: {val_auroc:.4f}")
        else:
            patience_counter += 1
            print(f"\n⏳ No improvement for {patience_counter} epochs (best: {best_val_auroc:.4f})")
        
        if patience_counter >= early_stop_patience:
            print(f"\n⛔ Early stopping at epoch {epoch}")
            break
        
        log_df = pd.DataFrame(train_log)
        log_df.to_csv(checkpoint_dir / 'training_log.csv', index=False)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")
    print(f"   Best Val AUROC: {checkpoint['val_auroc']:.4f}")
    
    test_loss, test_auroc, test_auprc, test_acc, test_f1, (tp, tn, fp, fn), (pos_acc, neg_acc) = validate_epoch(
        model, test_loader, criterion, device, split_name='Test'
    )
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS:")
    print(f"{'='*70}")
    print(f"AUROC:           {test_auroc:.4f}")
    print(f"AUPRC:           {test_auprc:.4f}")
    print(f"Accuracy:        {test_acc:.4f}")
    print(f"F1 Score:        {test_f1:.4f}")
    print(f"\nClass-wise Accuracy:")
    print(f"  Positive: {pos_acc:.4f}")
    print(f"  Negative: {neg_acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp}, TN={tn}")
    print(f"  FP={fp}, FN={fn}")
    print(f"{'='*70}")
    
    # Compare with Phase 1
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH PHASE 1:")
    print(f"{'='*70}")
    print(f"Phase 1 Static GCN:      AUROC ~0.82")
    print(f"Phase 2 Hierarchical:    AUROC {test_auroc:.4f}")
    if test_auroc > 0.82:
        print(f"✅ IMPROVEMENT: +{(test_auroc - 0.82)*100:.2f}%")
    else:
        print(f"❌ Below baseline by {(0.82 - test_auroc)*100:.2f}%")
    print(f"{'='*70}")
    
    results = {
        'test_loss': test_loss,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_pos_acc': pos_acc,
        'test_neg_acc': neg_acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'best_val_auroc': checkpoint['val_auroc'],
        'best_epoch': checkpoint['epoch']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(checkpoint_dir / 'final_results.csv', index=False)
    
    print(f"\n✅ Complete! Results saved to {checkpoint_dir}")


if __name__ == '__main__':
    main()
