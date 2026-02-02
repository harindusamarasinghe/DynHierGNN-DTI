"""
Quality Control for Generated Conformations
Checks: pLDDT scores, RMSD diversity, structural validity

Usage:
    python scripts/protein_quality_control.py

Inputs:
    data/conformations/poc_consolidated/PROTEIN_ID/conf_*.pdb

Output:
    data/conformations/poc_consolidated/quality_report.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser, Superimposer


class ConformationQC:
    def __init__(self, conformation_dir: str = "data/conformations/poc_consolidated"):
        self.conformation_dir = Path(conformation_dir)
        self.parser = PDBParser(QUIET=True)

    # ------------------------------------------------------------------ #
    # pLDDT extraction
    # ------------------------------------------------------------------ #
    def extract_plddt(self, pdb_file: Path) -> float:
        """Extract average pLDDT from PDB file (stored in B-factor)."""
        structure = self.parser.get_structure("protein", pdb_file)
        plddts = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        plddts.append(atom.bfactor)

        return float(np.mean(plddts)) if plddts else 0.0

    # ------------------------------------------------------------------ #
    # RMSD computation
    # ------------------------------------------------------------------ #
    def calculate_rmsd(self, pdb1: Path, pdb2: Path) -> float | None:
        """Calculate RMSD between two conformations using Cα atoms."""
        try:
            struct1 = self.parser.get_structure("s1", pdb1)
            struct2 = self.parser.get_structure("s2", pdb2)

            atoms1 = [atom for atom in struct1.get_atoms() if atom.name == "CA"]
            atoms2 = [atom for atom in struct2.get_atoms() if atom.name == "CA"]

            if len(atoms1) != len(atoms2) or len(atoms1) == 0:
                return None

            sup = Superimposer()
            sup.set_atoms(atoms1, atoms2)
            return float(sup.rms)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Per‑protein QC
    # ------------------------------------------------------------------ #
    def check_protein_quality(self, protein_dir: Path) -> dict:
        """Check quality for one protein (handles 2 or 3 conformations)."""
        protein_id = protein_dir.name

        pdb_files = sorted(protein_dir.glob("conf_*.pdb"))
        num_confs = len(pdb_files)
        
        # CHANGED: Accept 2 or 3 conformations
        if num_confs < 2:
            return {
                "protein_id": protein_id,
                "status": "INCOMPLETE",
                "num_conformations": num_confs,
                "avg_plddt": np.nan,
                "min_plddt": np.nan,
                "conf1_plddt": np.nan,
                "conf2_plddt": np.nan,
                "conf3_plddt": np.nan,
                "avg_rmsd": np.nan,
                "rmsd_1-2": np.nan,
                "rmsd_1-3": np.nan,
                "rmsd_2-3": np.nan,
                "issues": f"Insufficient conformations ({num_confs})",
            }

        # pLDDT per conformation (handle 2 or 3)
        plddts = [self.extract_plddt(pdb) for pdb in pdb_files]
        avg_plddt = float(np.mean(plddts))
        min_plddt = float(np.min(plddts))

        # Pairwise RMSDs
        rmsds = []
        rmsd_12 = self.calculate_rmsd(pdb_files[0], pdb_files[1])
        
        # CHANGED: Only calculate conf3 RMSDs if it exists
        if num_confs >= 3:
            rmsd_13 = self.calculate_rmsd(pdb_files[0], pdb_files[2])
            rmsd_23 = self.calculate_rmsd(pdb_files[1], pdb_files[2])
        else:
            rmsd_13 = None
            rmsd_23 = None

        if rmsd_12 is not None:
            rmsds.append(rmsd_12)
        if rmsd_13 is not None:
            rmsds.append(rmsd_13)
        if rmsd_23 is not None:
            rmsds.append(rmsd_23)

        avg_rmsd = float(np.mean(rmsds)) if rmsds else 0.0

        # ------------------------------------------------------------------ #
        # Assign qualitative status
        # ------------------------------------------------------------------ #
        status = "GOOD"
        issues = []

        # pLDDT rules
        if avg_plddt < 70:
            status = "WARNING"
            issues.append(f"Low avg pLDDT ({avg_plddt:.1f})")

        if min_plddt < 60:
            status = "POOR"
            issues.append(f"Very low min pLDDT ({min_plddt:.1f})")

        # RMSD rules (diversity)
        if avg_rmsd < 0.5:
            status = "WARNING"
            issues.append(f"Low diversity (avg RMSD={avg_rmsd:.2f} Å)")

        if avg_rmsd > 5.0:
            if status == "GOOD":
                status = "WARNING"
            issues.append(f"Very high diversity (avg RMSD={avg_rmsd:.2f} Å)")
        
        # CHANGED: Note if only 2 conformations
        if num_confs == 2:
            issues.append("Only 2 conformations available")

        # CHANGED: Build result dict dynamically based on num_confs
        result = {
            "protein_id": protein_id,
            "status": status,
            "num_conformations": num_confs,
            "avg_plddt": avg_plddt,
            "min_plddt": min_plddt,
            "conf1_plddt": plddts[0],
            "conf2_plddt": plddts[1],
            "conf3_plddt": plddts[2] if num_confs >= 3 else np.nan,
            "avg_rmsd": avg_rmsd,
            "rmsd_1-2": rmsd_12,
            "rmsd_1-3": rmsd_13 if num_confs >= 3 else np.nan,
            "rmsd_2-3": rmsd_23 if num_confs >= 3 else np.nan,
            "issues": "; ".join(issues) if issues else "None",
        }
        
        return result

    # ------------------------------------------------------------------ #
    # Run QC on all proteins
    # ------------------------------------------------------------------ #
    def run_qc(self) -> pd.DataFrame:
        print("=" * 70)
        print("CONFORMATIONAL QUALITY CONTROL")
        print("=" * 70)

        protein_dirs = [
            d
            for d in self.conformation_dir.iterdir()
            if d.is_dir() and not d.name.startswith("batch_results")
        ]

        print(f"\nChecking {len(protein_dirs)} proteins...")

        results = []
        for protein_dir in tqdm(protein_dirs, desc="QC Progress"):
            results.append(self.check_protein_quality(protein_dir))

        df = pd.DataFrame(results)

        # Save detailed report
        output_path = self.conformation_dir / "quality_report.csv"
        df.to_csv(output_path, index=False)

        # Summary
        print("\n" + "=" * 70)
        print("QUALITY CONTROL SUMMARY")
        print("=" * 70)
        print(f"Total proteins: {len(df)}")
        
        # CHANGED: Show conformation distribution
        print(f"\nConformation distribution:")
        print(f"  3 conformations: {len(df[df['num_conformations'] == 3])}")
        print(f"  2 conformations: {len(df[df['num_conformations'] == 2])}")
        print(f"  <2 conformations: {len(df[df['num_conformations'] < 2])}")
        
        print(f"\nQuality status:")
        for status in ["GOOD", "WARNING", "POOR", "INCOMPLETE"]:
            count = len(df[df['status'] == status])
            if count > 0:
                print(f"  {status}: {count}")

        # Calculate metrics only for valid proteins (2+ conformations)
        valid_df = df[df['num_conformations'] >= 2]
        
        if len(valid_df) > 0:
            print(f"\nMetrics (proteins with ≥2 conformations, n={len(valid_df)}):")
            if "avg_plddt" in valid_df.columns and valid_df["avg_plddt"].notna().any():
                print(f"  Average pLDDT: {valid_df['avg_plddt'].mean():.2f}")
            if "avg_rmsd" in valid_df.columns and valid_df["avg_rmsd"].notna().any():
                print(f"  Average RMSD: {valid_df['avg_rmsd'].mean():.2f} Å")

        print(f"\n✓ Full report saved: {output_path}")

        # Flag poor cases
        poor = df[df["status"] == "POOR"]
        if len(poor) > 0:
            print(f"\n⚠️  {len(poor)} proteins need attention:")
            for _, row in poor.iterrows():
                print(f"  - {row['protein_id']}: {row['issues']}")
        
        # Flag incomplete cases
        incomplete = df[df["status"] == "INCOMPLETE"]
        if len(incomplete) > 0:
            print(f"\n⚠️  {len(incomplete)} proteins incomplete:")
            for _, row in incomplete.iterrows():
                print(f"  - {row['protein_id']}: {row['issues']}")

        return df


if __name__ == "__main__":
    qc = ConformationQC()
    qc.run_qc()
