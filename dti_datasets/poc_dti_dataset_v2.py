#!/usr/bin/env python3
"""
POC DTI Dataset V2: Handles variable conformations (2-3 per protein)
FIXED: Handles integer Drug_IDs properly
"""


import pickle
import pandas as pd
import torch
from torch_geometric.data import Batch



def collate_fn(batch):
    """
    Custom collate with padding for variable conformations.
    Handles proteins with 2 or 3 conformations.
    """
    drug_graphs, conf_graphs_list, labels = zip(*batch)
    
    # Batch drug graphs
    drug_batch = Batch.from_data_list(drug_graphs)
    
    # Find max conformations in this batch
    max_confs = max(len(confs) for confs in conf_graphs_list)
    
    # Pad conformations to max_confs
    conf_graphs_batch = []
    for conf_idx in range(max_confs):
        conf_at_idx = []
        for sample_confs in conf_graphs_list:
            if conf_idx < len(sample_confs):
                # Use actual conformation
                conf_at_idx.append(sample_confs[conf_idx])
            else:
                # Pad by repeating last conformation
                conf_at_idx.append(sample_confs[-1])
        
        # Batch this conformation across all samples
        conf_graphs_batch.append(Batch.from_data_list(conf_at_idx))
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return drug_batch, conf_graphs_batch, labels



class POCDTIDatasetV2:
    """Dataset for Phase 1: Multi-conformational protein graphs"""
    
    def __init__(
        self,
        csv_path,
        protein_graphs_path,
        drug_graphs_path,
        use_binary=True,
    ):
        self.use_binary = use_binary
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Auto-detect columns
        if 'Protein_ID' in self.df.columns:
            self.protein_col = 'Protein_ID'
        elif 'Target_ID' in self.df.columns:
            self.protein_col = 'Target_ID'
        else:
            raise ValueError(f"No protein column! Available: {self.df.columns.tolist()}")
        
        if 'Drug_ID' not in self.df.columns:
            raise ValueError(f"No Drug_ID column! Available: {self.df.columns.tolist()}")
        self.drug_col = 'Drug_ID'
        
        if 'Y' in self.df.columns:
            self.label_col = 'Y'
        elif 'Label' in self.df.columns:
            self.label_col = 'Label'
        else:
            raise ValueError(f"No label column! Available: {self.df.columns.tolist()}")
        
        # Load graphs
        with open(protein_graphs_path, "rb") as f:
            self.protein_graphs = pickle.load(f)
        
        with open(drug_graphs_path, "rb") as f:
            self.drug_graphs = pickle.load(f)
        
        # DEBUG: Check key types
        sample_protein_key = list(self.protein_graphs.keys())[0]
        sample_drug_key = list(self.drug_graphs.keys())[0]
        
        print(f"\n🔍 POCDTIDatasetV2 Debug Info:")
        print(f"  CSV path: {csv_path}")
        print(f"  Total CSV rows: {len(self.df)}")
        print(f"  Sample CSV protein IDs: {self.df[self.protein_col].head(3).tolist()}")
        print(f"  Sample CSV drug IDs: {self.df[self.drug_col].head(3).tolist()}")
        print(f"  Graph protein key type: {type(sample_protein_key)} (sample: {sample_protein_key})")
        print(f"  Graph drug key type: {type(sample_drug_key)} (sample: {sample_drug_key})")
        
        # Filter valid samples (FIXED VERSION - handles integer Drug_IDs)
        valid_indices = []
        missing_proteins = set()
        missing_drugs = set()
        
        for idx in range(len(self.df)):
            protein_id_raw = str(self.df.iloc[idx][self.protein_col]).strip()
            drug_id_raw = self.df.iloc[idx][self.drug_col]  # Keep original type
            
            # Clean protein ID (always string)
            protein_id = protein_id_raw
            if 'POC_PROT_' in protein_id:
                protein_id = protein_id.replace('POC_PROT_', '')
            elif 'POCPROT' in protein_id:
                protein_id = protein_id.replace('POCPROT', '')
            
            # Handle Drug ID type matching
            # Try both as-is and as integer
            drug_id_candidates = [drug_id_raw]
            
            # If drug key in graph is integer, convert
            if isinstance(sample_drug_key, int):
                try:
                    drug_id_candidates.append(int(drug_id_raw))
                except:
                    pass
            # If drug key in graph is string, convert
            elif isinstance(sample_drug_key, str):
                drug_id_candidates.append(str(drug_id_raw))
            
            # Check if both exist
            protein_found = protein_id in self.protein_graphs
            drug_found = any(did in self.drug_graphs for did in drug_id_candidates)
            
            if protein_found and drug_found:
                valid_indices.append(idx)
            else:
                if not protein_found:
                    missing_proteins.add(protein_id_raw)
                if not drug_found:
                    missing_drugs.add(str(drug_id_raw))
        
        # Report filtering results
        print(f"\n  Filtering results:")
        print(f"    Valid samples: {len(valid_indices)}/{len(self.df)} ({len(valid_indices)/len(self.df)*100:.1f}%)")
        
        if missing_proteins:
            print(f"    Missing proteins: {len(missing_proteins)}")
            print(f"      Examples: {list(missing_proteins)[:5]}")
        
        if missing_drugs:
            print(f"    Missing drugs: {len(missing_drugs)}")
            print(f"      Examples: {list(missing_drugs)[:5]}")
        
        # Apply filter
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # Store drug key type for __getitem__
        self.drug_key_is_int = isinstance(sample_drug_key, int)
        
        # Report class distribution
        if len(self.df) > 0:
            y_values = self.df[self.label_col]
            pos_count = (y_values == 1).sum()
            neg_count = (y_values == 0).sum()
            print(f"\n  Final dataset:")
            print(f"    Samples: {len(self.df)}")
            print(f"    Y=1 (active): {pos_count} ({pos_count/len(self.df)*100:.1f}%)")
            print(f"    Y=0 (inactive): {neg_count} ({neg_count/len(self.df)*100:.1f}%)")
        else:
            print(f"\n  ⚠️  WARNING: No valid samples after filtering!")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        protein_id = str(row[self.protein_col]).strip()
        drug_id_raw = row[self.drug_col]
        
        # Clean protein ID
        if 'POC_PROT_' in protein_id:
            protein_id = protein_id.replace('POC_PROT_', '')
        elif 'POCPROT' in protein_id:
            protein_id = protein_id.replace('POCPROT', '')
        
        # Convert drug_id to match graph key type
        if self.drug_key_is_int:
            drug_id = int(drug_id_raw) if not isinstance(drug_id_raw, int) else drug_id_raw
        else:
            drug_id = str(drug_id_raw)
        
        drug_graph = self.drug_graphs[drug_id]
        protein_conf_graphs = self.protein_graphs[protein_id]  # List of 2-3 conformations
        label = float(row[self.label_col])
        
        return drug_graph, protein_conf_graphs, label
