"""
Script 1: Optimized ChEMBL Downloader & Filter
- Downloads ChEMBL 36
- Filters by Lipinski's Rule of Five
- Fast Sampling for MacBook Air
"""

import pandas as pd
import numpy as np
import requests
import gzip
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

class ChemblDownloader:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_file = self.output_dir / 'chembl_raw.txt.gz'
        self.filtered_file = self.output_dir / 'chembl_500k.csv'

    def download_data(self):
        if self.raw_file.exists():
            print(f"✓ Raw file already exists at {self.raw_file}")
            return
        
        url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_chemreps.txt.gz"
        print("="*70)
        print("STEP 1: Downloading ChEMBL Database")
        print("="*70)
        print(f"Downloading from {url}...")
        
        response = requests.get(url, stream=True)
        with open(self.raw_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Downloaded to {self.raw_file}")

    def is_drug_like(self, smiles):
        """Lipinski's Rule of Five Filter"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return False
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Constraints
            return (150 <= mw <= 500) and (-2 <= logp <= 5) and (hbd <= 5) and (hba <= 10)
        except:
            return False

    def run(self, n_samples=500000):
        self.download_data()
        
        print("\n" + "="*70)
        print("STEP 2: Loading & Filtering (Lipinski's Rule)")
        print("="*70)
        
        # Read the raw file
        df = pd.read_csv(self.raw_file, sep='\t', compression='gzip')
        # ChEMBL columns are usually: chembl_id, canonical_smiles, standard_inchi, standard_inchi_key
        # We only need the SMILES
        df = df[['canonical_smiles']].rename(columns={'canonical_smiles': 'SMILES'})
        
        print(f"Applying filters to {len(df):,} compounds...")
        # This part takes time (~20 mins)
        mask = df['SMILES'].apply(self.is_drug_like)
        df_filtered = df[mask].copy()
        
        print(f"✓ Filtered: {len(df):,} → {len(df_filtered):,} drug-like compounds")

        print("\n" + "="*70)
        print(f"STEP 3: Fast Sampling (Target: {n_samples:,} compounds)")
        print("="*70)
        
        if len(df_filtered) > n_samples:
            print(f"Performing fast random sampling...")
            df_final = df_filtered.sample(n=n_samples, random_state=42)
        else:
            df_final = df_filtered

        # Save the result
        df_final.to_csv(self.filtered_file, index=False)
        print(f"✓ Success! Saved {len(df_final):,} compounds to {self.filtered_file}")
        print("="*70)

if __name__ == "__main__":
    downloader = ChemblDownloader()
    downloader.run(n_samples=500000)