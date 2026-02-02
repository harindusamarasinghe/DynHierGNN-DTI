"""
POC DTI Dataset:
Returns (drug_graph, [prot_conf1, prot_conf2, prot_conf3], affinity)
"""

from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset
import pandas as pd


class POC_DTIDataset(Dataset):

    def __init__(self, csv_path: str, drug_graphs_path: str = "data/processed/poc_drug_graphs.pkl", protein_graphs_path: str = "data/processed/poc_protein_graphs.pkl"):
        super().__init__()
        self.df = pd.read_csv(csv_path)

        self.drug_col = "Drug_ID"
        self.prot_col = "Target_ID"
        self.y_col = "Y"

        # Load graphs
        with open(drug_graphs_path, "rb") as f:
            self.drug_graphs = pickle.load(f)
        with open(protein_graphs_path, "rb") as f:
            self.protein_graphs = pickle.load(f)

        # Extract short protein ID from long folder name
        def extract_short_id(full_name: str) -> str:
            if full_name.startswith("POC_PROT_"):
                parts = full_name[len("POC_PROT_"):].split("_")
                return parts[0]
            return full_name

        # Build reverse mapping: short_id → full_folder_name
        self.prot_id_to_folder = {}
        for full_name in self.protein_graphs.keys():
            short_id = extract_short_id(full_name)
            self.prot_id_to_folder[short_id] = full_name

        # Filter CSV
        mask = (
            self.df[self.drug_col].isin(self.drug_graphs.keys()) &
            self.df[self.prot_col].isin(self.prot_id_to_folder.keys())
        )
        self.df = self.df[mask].reset_index(drop=True)

        print(f"Loaded {len(self.df)} interactions from {csv_path}")
        print(f"Available drug graphs: {len(self.drug_graphs)}")
        print(f"Available protein graphs: {len(self.protein_graphs)}")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_id = row[self.drug_col]
        prot_id = row[self.prot_col]
        y = row[self.y_col]

        drug_graph = self.drug_graphs[drug_id]
        
        # Convert short protein ID to full folder name
        prot_folder = self.prot_id_to_folder[prot_id]
        conf_graphs = self.protein_graphs[prot_folder]  # list of 3 Data objects

        return drug_graph, conf_graphs, float(y)
