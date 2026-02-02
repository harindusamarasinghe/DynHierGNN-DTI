#!/usr/bin/env python3
"""
Create FASTA files for batch2 proteins in SMALLER BATCHES
Split 50 proteins into 5 FASTA files (10 proteins each)
"""

import pandas as pd
from pathlib import Path

# Load batch2 protein mapping
batch2 = pd.read_csv('data/poc/batch2_protein_mapping.csv')

print("="*70)
print("BATCH 2 FASTA FILE CREATION - MULTIPLE BATCHES")
print("="*70)
print(f"Total proteins: {len(batch2)}")

# Sort by length (shortest first - more efficient for ColabFold)
batch2_sorted = batch2.sort_values('Length')

# Create output directory
Path('data/poc').mkdir(parents=True, exist_ok=True)

# Split into batches of 10 proteins each
batch_size = 10
num_batches = (len(batch2_sorted) + batch_size - 1) // batch_size  # Ceiling division

print(f"Creating {num_batches} FASTA files ({batch_size} proteins each)")
print("="*70)

fasta_files = []

for batch_num in range(num_batches):
    # Get proteins for this batch
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, len(batch2_sorted))
    batch_proteins = batch2_sorted.iloc[start_idx:end_idx]
    
    # Create FASTA content
    fasta_lines = []
    for _, row in batch_proteins.iterrows():
        # Header format: >ProteinID|Length_XXX
        header = f">{row['Protein_ID']}|Length_{row['Length']}"
        sequence = row['Sequence']
        
        fasta_lines.append(header)
        fasta_lines.append(sequence)
    
    # Save FASTA file
    fasta_path = f'data/poc/batch2_proteins_part{batch_num+1}.fasta'
    
    with open(fasta_path, 'w') as f:
        f.write('\n'.join(fasta_lines))
    
    fasta_files.append(fasta_path)
    
    # Print batch info
    print(f"\nBatch {batch_num+1}/{num_batches}:")
    print(f"  File: {fasta_path}")
    print(f"  Proteins: {len(batch_proteins)}")
    print(f"  Length range: {batch_proteins['Length'].min()}-{batch_proteins['Length'].max()} aa")
    print(f"  Total residues: {batch_proteins['Length'].sum()}")
    print(f"  Proteins: {', '.join(batch_proteins['Protein_ID'].head(3).tolist())}...")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total FASTA files created: {num_batches}")
print(f"Total proteins: {len(batch2_sorted)}")
print(f"Overall length range: {batch2_sorted['Length'].min()}-{batch2_sorted['Length'].max()} aa")
print(f"Total residues: {batch2_sorted['Length'].sum()}")

print("\n✓ FASTA files ready for ColabFold!")
print("\nFiles created:")
for i, fasta_file in enumerate(fasta_files, 1):
    print(f"  {i}. {fasta_file}")

print("\n" + "="*70)
print("COLABFOLD EXECUTION PLAN")
print("="*70)
print("Run each batch separately in ColabFold:")
print("1. Upload batch2_proteins_part1.fasta → Generate → Download")
print("2. Upload batch2_proteins_part2.fasta → Generate → Download")
print("3. ... continue for all batches")
print("\nAdvantage: Smaller batches = faster per-batch, easier monitoring")
