"""
Week 1: Create Protein FASTA File for ColabFold
================================================
Purpose: Convert protein sequences to FASTA format for conformational generation
Input: data/poc/protein_mapping.csv
Output: data/poc/poc_proteins.fasta (ready for ColabFold)
"""

import pandas as pd
from pathlib import Path

def create_protein_fasta():
    """
    Create FASTA file sorted by protein length (shortest first for efficient processing).
    """
    
    poc_dir = Path('data/poc')
    
    # Load protein mapping
    protein_path = poc_dir / 'protein_mapping.csv'
    if not protein_path.exists():
        raise FileNotFoundError(
            f"Protein mapping not found: {protein_path}\n"
            f"Run: python scripts/create_poc_subset.py"
        )
    
    proteins = pd.read_csv(protein_path)
    print(f"Loaded {len(proteins)} proteins from {protein_path}")
    
    # Sort by length (shortest first for faster ColabFold processing)
    proteins = proteins.sort_values('Length').reset_index(drop=True)
    
    print(f"Length range: {proteins['Length'].min()} - {proteins['Length'].max()} residues")
    print(f"Processing order: shortest to longest (for efficient ColabFold batching)")
    
    # Create FASTA content
    fasta_lines = []
    for idx, row in proteins.iterrows():
        protein_id = row['Protein_ID']
        sequence = row['Sequence']
        length = row['Length']
        
        # FASTA header: >sequence_identifier description
        # Format: >POC_PROT_001|Length_456|Original_ID_123
        header = f">POC_PROT_{protein_id}|Length_{length}|Original_{protein_id}"
        
        fasta_lines.append(header)
        fasta_lines.append(sequence)
    
    # Write FASTA file
    fasta_path = poc_dir / 'poc_proteins.fasta'
    with open(fasta_path, 'w') as f:
        f.write('\n'.join(fasta_lines))
    
    print(f"\n✓ FASTA file created: {fasta_path}")
    print(f"  Sequences: {len(proteins)}")
    print(f"  Format: FASTA with headers containing protein metadata")
    print(f"\n📋 FASTA Header Format:")
    print(f"   >POC_PROT_NNN|Length_LLL|Original_III")
    print(f"   NNN = POC protein index (001-050)")
    print(f"   LLL = Protein length in residues")
    print(f"   III = Original Davis protein ID")
    
    # Print first 3 and last 3 entries as verification
    print(f"\nFirst 3 entries (shortest proteins):")
    for i in range(min(3, len(proteins))):
        row = proteins.iloc[i]
        print(f"  POC_PROT_{row['Protein_ID']:03d}: {row['Length']} residues")
    
    print(f"\nLast 3 entries (longest proteins):")
    for i in range(max(0, len(proteins)-3), len(proteins)):
        row = proteins.iloc[i]
        print(f"  POC_PROT_{row['Protein_ID']:03d}: {row['Length']} residues")
    
    # Save processing metadata
    metadata = {
        'total_proteins': len(proteins),
        'length_range': [int(proteins['Length'].min()), int(proteins['Length'].max())],
        'sorted_by': 'length_ascending',
        'reason': 'Shorter proteins process faster in ColabFold',
        'fasta_file': str(fasta_path),
        'created': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = poc_dir / 'fasta_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved: {metadata_path}")
    print(f"\n✅ Ready for Week 2: Conformational Generation with ColabFold")
    print(f"\nNext steps:")
    print(f"  1. Open: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb")
    print(f"  2. Copy content of {fasta_path}")
    print(f"  3. Paste into ColabFold 'Input protein sequences' cell")
    print(f"  4. Set: msa_mode='MMseqs2', num_recycles=3")


if __name__ == "__main__":
    create_protein_fasta()
