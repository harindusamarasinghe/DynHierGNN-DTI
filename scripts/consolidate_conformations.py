#!/usr/bin/env python3
"""
Consolidate batch2 conformations (Batch1-6)
"""

import os
import shutil
import zipfile
from pathlib import Path
import re
from collections import defaultdict


def consolidate_all_batches():
    """
    Merge all batch ZIP files into unified conformation folders
    Priority: Batch6 > Batch5 > Batch4 > Batch3 > Batch2 > Batch1 (latest first)
    """
    
    # CHANGED: New output path for full dataset
    output_dir = Path("data/conformations/full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CONSOLIDATING BATCH2 CONFORMATIONS (Batches 1-6)")
    print("="*70)
    
    # CHANGED: Point to your batch2 folder
    zip_pattern = Path("./batch2") / "*_conf*_results.zip"
    zip_files = sorted(Path("./confirmation-zips/batch2").glob("Batch*_conf*_results.zip"))
    
    print(f"\nFound {len(zip_files)} ZIP files to process\n")
    
    # Step 1: Extract all ZIPs to temp directory
    temp_dir = Path("temp_extraction_batch2")
    temp_dir.mkdir(exist_ok=True)
    
    extracted_batches = {}  # {batch_name: {conf_idx: extraction_path}}
    
    for zip_file in zip_files:
        print(f"Extracting: {zip_file.name}")
        
        # CHANGED: Updated regex for your naming (Batch1_conf1_results.zip)
        match = re.search(r'Batch(\d+)_conf(\d+)', zip_file.name)
        if not match:
            print(f"  ✗ Skipping (cannot parse filename): {zip_file.name}")
            continue
        
        batch_num = int(match.group(1))
        conf_num = int(match.group(2))
        batch_key = f"Batch{batch_num}"
        
        # Extract to temp folder
        extract_path = temp_dir / batch_key / f"conf_{conf_num}"
        extract_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(extract_path)
        
        if batch_key not in extracted_batches:
            extracted_batches[batch_key] = {}
        extracted_batches[batch_key][conf_num] = extract_path
        
        print(f"  ✓ Extracted to: {extract_path}")
    
    # Step 2: Consolidate by protein with priority (Batch6 > ... > Batch1)
    print("\n" + "="*70)
    print("MERGING BY PROTEIN (Priority: Batch6 > Batch5 > Batch4 > Batch3 > Batch2 > Batch1)")
    print("="*70 + "\n")
    
    protein_data = defaultdict(lambda: {1: None, 2: None, 3: None})
    
    # Process in priority order (reverse: Batch6 first)
    for batch_key in sorted(extracted_batches.keys(), key=lambda x: int(x[5:]), reverse=True):
        print(f"\nProcessing {batch_key}:")
        
        for conf_num in sorted(extracted_batches[batch_key].keys()):
            extract_path = extracted_batches[batch_key][conf_num]
            pdb_files = list(extract_path.rglob("*.pdb"))
            
            print(f"  Conf {conf_num}: {len(pdb_files)} PDB files")
            
            for pdb_file in pdb_files:
                # CHANGED: Updated protein ID extraction for batch2 naming
                # Expected: "POC_PROT_CSNK1A1L_Length476_results.pdb" or similar
                match = re.search(r'POC_PROT_(\w+)_Length', pdb_file.name)
                if not match:
                    # Fallback pattern
                    match = re.search(r'(\w+?)_Length', pdb_file.name)
                if not match:
                    continue
                
                protein_id = match.group(1)
                
                # Only assign if this conformation slot is empty (priority to later batches)
                if protein_data[protein_id][conf_num] is None:
                    protein_data[protein_id][conf_num] = pdb_file
    
    # Step 3: Copy consolidated data to final location
    print("\n" + "="*70)
    print("CREATING FINAL PROTEIN FOLDERS")
    print("="*70 + "\n")
    
    complete_proteins = []
    incomplete_proteins = []
    
    for protein_id in sorted(protein_data.keys()):
        protein_folder = output_dir / f"POC_PROT_{protein_id}"
        protein_folder.mkdir(exist_ok=True)
        
        confs = protein_data[protein_id]
        conf_count = sum(1 for v in confs.values() if v is not None)
        
        # Copy PDB files
        for conf_num, pdb_path in confs.items():
            if pdb_path is not None:
                dest_pdb = protein_folder / f"conf_{conf_num}.pdb"
                shutil.copy(pdb_path, dest_pdb)
                
                # Also copy associated JSON files
                json_pattern = pdb_path.stem + "*.json"
                for json_file in pdb_path.parent.glob(json_pattern):
                    dest_json = protein_folder / f"conf_{conf_num}_{json_file.stem.split('_')[-1]}.json"
                    shutil.copy(json_file, dest_json)
        
        # Track completeness
        if conf_count == 3:
            complete_proteins.append(protein_id)
            status = "✓ Complete (3 confs)"
        elif conf_count == 2:
            incomplete_proteins.append(protein_id)
            status = "⚠ Incomplete (2 confs)"
        else:
            status = f"⚠ Partial ({conf_count} confs)"
        
        print(f"  POC_PROT_{protein_id}: {status}")
    
    # Summary
    print("\n" + "="*70)
    print("CONSOLIDATION SUMMARY")
    print("="*70)
    print(f"Total proteins: {len(protein_data)}")
    print(f"  Complete (3 conformations): {len(complete_proteins)}")
    print(f"  Incomplete (2 conformations): {len(incomplete_proteins)}")
    
    if incomplete_proteins:
        print(f"\n⚠️ Proteins with only 2 conformations:")
        for prot_id in incomplete_proteins[:10]:  # Show first 10
            print(f"  - POC_PROT_{prot_id}")
        if len(incomplete_proteins) > 10:
            print(f"  ... and {len(incomplete_proteins)-10} more")
        print("\n✅ These are still usable (we'll use 2 confs)")
    
    # Cleanup temp directory
    print(f"\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print(f"\n🎉 Final structure: {output_dir}/POC_PROT_{protein_id}/conf_1.pdb")
    print(f"\n✅ Ready for ESM-2 graph construction!")
    
    return output_dir, complete_proteins, incomplete_proteins


if __name__ == "__main__":
    consolidate_all_batches()
