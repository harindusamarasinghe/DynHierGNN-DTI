"""
Organize ColabFold outputs into structured format
Usage: python scripts/organize_conformations.py

Expected raw files (examples):
    POC_PROT_JNK1_Length_427_Original_JNK1_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
    POC_PROT_JNK1_Length_427_Original_JNK1_scores_rank_001_alphafold2_ptm_model_1_seed_000.json
    POC_PROT_BRK_Length_451_Original_BRK_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
    ...

The script will create:
    data/conformations/poc/POC_PROT_JNK1/
        conf_1.pdb
        conf_2.pdb
        conf_3.pdb
        *.json
    data/conformations/poc/POC_PROT_BRK/
        ...
"""

import shutil
from pathlib import Path
from collections import defaultdict
import re


class ConformationOrganizer:
    def __init__(self, base_dir: str = "data/conformations/poc"):
        self.base_dir = Path(base_dir)

        # UPDATE THESE TO MATCH YOUR REAL FOLDER NAMES
        # e.g. if your unzipped folders are batch_results1, batch_results2, ...
        self.raw_dirs = [
            self.base_dir / "batch_results1",
            self.base_dir / "batch_results2",
            self.base_dir / "batch_results3",
            self.base_dir / "batch_results4",
        ]

        self.output_dir = self.base_dir

        # Regex for your filenames:
        #   POC_PROT_JNK1_Length_427_Original_JNK1_unrelaxed_...
        # We take everything up to "_Length_" as the protein_id
        self.protein_regex = re.compile(r"^(.*?_Length_\d+)_")

    def extract_protein_id(self, filename: str) -> str | None:
        """Extract protein ID (e.g. 'POC_PROT_JNK1_Length_427') from filename."""
        m = self.protein_regex.match(filename)
        if m:
            return m.group(1)
        return None


    def organize(self):
        print("=" * 70)
        print("ORGANIZING COLABFOLD OUTPUTS")
        print("=" * 70)

        protein_files: dict[str, list[Path]] = defaultdict(list)

        # Collect files
        for raw_dir in self.raw_dirs:
            if not raw_dir.exists():
                print(f"⚠️  Skipping {raw_dir} (not found)")
                continue

            print(f"\nProcessing {raw_dir} ...")
            for file in raw_dir.iterdir():
                if not file.is_file():
                    continue

                protein_id = self.extract_protein_id(file.name)
                if protein_id is None:
                    continue
                protein_files[protein_id].append(file)

        print(f"\n✓ Found {len(protein_files)} proteins")

        # Organize each protein
        for protein_id, files in protein_files.items():
            protein_dir = self.output_dir / protein_id
            protein_dir.mkdir(exist_ok=True)

            pdb_files = [f for f in files if f.suffix == ".pdb"]
            json_files = [f for f in files if f.suffix == ".json"]

            pdb_files_sorted = sorted(pdb_files, key=lambda x: x.name)

            # Take up to first 3 PDBs as conf_1, conf_2, conf_3
            for i, pdb_file in enumerate(pdb_files_sorted[:3], start=1):
                dest = protein_dir / f"conf_{i}.pdb"
                shutil.copy2(pdb_file, dest)

            # Copy all JSON files (scores, etc.)
            for json_file in json_files:
                dest = protein_dir / json_file.name
                shutil.copy2(json_file, dest)

            print(f"  ✓ {protein_id}: {len(pdb_files_sorted[:3])} conformations")

        print("\n" + "=" * 70)
        print("✅ ORGANIZATION COMPLETE")
        print("=" * 70)
        print(f"Output base: {self.output_dir}")
        print(f"Total protein folders: {len(protein_files)}")


if __name__ == "__main__":
    organizer = ConformationOrganizer()
    organizer.organize()
