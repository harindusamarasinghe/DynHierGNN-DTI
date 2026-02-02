"""
Build Triplet Dataset for Contrastive Learning (ChEMBL → Triplets)

Usage:
    python scripts/build_triplets.py

Assumptions:
    - You already have: data/raw/chembl_500k.csv
      with at least a "SMILES" column.

Outputs:
    - data/pretraining/smiles_mapping.csv   (index ↔ SMILES)
    - data/pretraining/faiss_index.bin      (FAISS index on fingerprints)
    - data/pretraining/triplets.npy         (anchor, positive, negative indices)
"""

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import faiss
from pathlib import Path
from tqdm import tqdm


class TripletBuilder:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / 'pretraining'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameters
        self.max_compounds = 200_000   # target number of compounds after subsampling
        self.n_bits = 2048             # fingerprint size
        self.n_positives = 5           # top-k similar compounds as positives
        self.n_negatives = 20          # random negatives sampled per anchor
        self.neg_per_pos = 3           # negatives per positive
        self.batch_size = 10_000       # batch size for FAISS search

    # -------------------------------------------------------------------------
    def load_data(self):
        """Load ChEMBL data."""
        print("=" * 70)
        print("STEP 1: Loading ChEMBL Data")
        print("=" * 70)

        input_path = self.data_dir / 'raw' / 'chembl_500k.csv'
        if not input_path.exists():
            raise FileNotFoundError(
                f"{input_path} not found.\n"
                f"Make sure you ran download_chembl.py first."
            )

        df = pd.read_csv(input_path)
        if 'SMILES' not in df.columns:
            raise ValueError("Input file must contain a 'SMILES' column.")

        print(f"✓ Loaded {len(df):,} compounds from {input_path}")

        # Subsample for final pretraining scale
        if len(df) > self.max_compounds:
            df = df.sample(n=self.max_compounds, random_state=42).reset_index(drop=True)
            print(f"Subsampled to {len(df):,} compounds for contrastive pretraining")

        return df

    # -------------------------------------------------------------------------
    def compute_fingerprints(self, df: pd.DataFrame):
        """Compute Morgan fingerprints (radius 2, n_bits length)."""
        print("\n" + "=" * 70)
        print("STEP 2: Computing Morgan Fingerprints")
        print("=" * 70)

        fps = []
        valid_smiles = []

        for smiles in tqdm(df['SMILES'], desc="Computing"):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=2,
                    nBits=self.n_bits
                )
                arr = np.zeros((self.n_bits,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                valid_smiles.append(smiles)

        X = np.array(fps, dtype='float32')
        print(f"✓ Computed fingerprints: {X.shape[0]:,} × {X.shape[1]}")

        # Save SMILES mapping
        smiles_df = pd.DataFrame({'index': range(len(valid_smiles)), 'SMILES': valid_smiles})
        smiles_path = self.output_dir / 'smiles_mapping.csv'
        smiles_df.to_csv(smiles_path, index=False)
        print(f"✓ Saved SMILES mapping: {smiles_path}")

        # Optional: save fingerprints for reuse
        fps_path = self.output_dir / 'chembl_fps.npy'
        np.save(fps_path, X)
        print(f"✓ Saved fingerprints: {fps_path}")

        return X, valid_smiles

    # -------------------------------------------------------------------------
    def build_faiss_index(self, X: np.ndarray):
        """Build FAISS similarity index (cosine via normalized inner product)."""
        print("\n" + "=" * 70)
        print("STEP 3: Building FAISS Similarity Index")
        print("=" * 70)

        # Normalize for cosine similarity
        faiss.normalize_L2(X)

        # Build index
        index = faiss.IndexFlatIP(X.shape[1])  # Inner product = cosine similarity after L2-normalization
        index.add(X)

        print(f"✓ Indexed {index.ntotal:,} vectors")

        # Save index
        index_path = self.output_dir / 'faiss_index.bin'
        faiss.write_index(index, str(index_path))
        print(f"✓ Saved FAISS index: {index_path}")

        return index

    # -------------------------------------------------------------------------
    def create_triplets(self, index, X: np.ndarray):
        """Create (anchor, positive, negative) triplets with batching."""
        print("\n" + "=" * 70)
        print("STEP 4: Creating Triplet Dataset")
        print("=" * 70)
        print("Strategy:")
        print(f"  - For each anchor: find {self.n_positives} most similar positives")
        print(f"  - Sample {self.n_negatives} random negatives")
        print(f"  - For each positive: {self.neg_per_pos} negatives")
        print(f"  → ≈ {self.n_positives * self.neg_per_pos} triplets per anchor")

        triplets = []
        n_compounds = len(X)

        print("\nGenerating triplets in batches...")
        for start in tqdm(range(0, n_compounds, self.batch_size), desc="Progress"):
            end = min(start + self.batch_size, n_compounds)

            # Search neighbors for this batch
            D, I = index.search(X[start:end], self.n_positives + 1)

            for local_i in range(end - start):
                i = start + local_i
                anchor_idx = i

                # Positive indices (skip self at index 0)
                pos_indices = I[local_i, 1:self.n_positives + 1]

                # Negative indices (sampled from entire set)
                neg_indices = np.random.choice(n_compounds, size=self.n_negatives, replace=False)

                for p_idx in pos_indices:
                    selected_negs = np.random.choice(
                        neg_indices,
                        self.neg_per_pos,
                        replace=False
                    )
                    for n_idx in selected_negs:
                        triplets.append([anchor_idx, p_idx, n_idx])

        triplets = np.array(triplets, dtype=np.int32)
        print(f"\n✓ Created {len(triplets):,} triplets")
        print(f"  Shape: {triplets.shape}")
        return triplets

    # -------------------------------------------------------------------------
    def save_triplets(self, triplets: np.ndarray):
        """Save triplet dataset to .npy."""
        output_path = self.output_dir / 'triplets.npy'
        np.save(output_path, triplets)
        size_mb = output_path.stat().st_size / 1e6
        print(f"\n✓ Saved triplets: {output_path}")
        print(f"  Size: {size_mb:.1f} MB")

    # -------------------------------------------------------------------------
    def run(self):
        """Run full pipeline: load → fingerprints → FAISS → triplets → save."""
        # Load data
        df = self.load_data()

        # Compute fingerprints
        X, smiles = self.compute_fingerprints(df)

        # Build FAISS index
        index = self.build_faiss_index(X)

        # Create triplets
        triplets = self.create_triplets(index, X)

        # Save triplets
        self.save_triplets(triplets)

        print("\n" + "=" * 70)
        print("✅ TRIPLET DATASET CREATION COMPLETE")
        print("=" * 70)
        print("Files created:")
        print(f"  - {self.output_dir / 'triplets.npy'} ({len(triplets):,} triplets)")
        print(f"  - {self.output_dir / 'smiles_mapping.csv'} ({len(smiles):,} compounds)")
        print(f"  - {self.output_dir / 'faiss_index.bin'}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    builder = TripletBuilder()
    builder.run()
