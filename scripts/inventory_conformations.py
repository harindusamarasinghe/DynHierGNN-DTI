import os
from pathlib import Path
import zipfile
import re

def inventory_all_batches():
    """
    Scan current directory for all batch ZIP files and analyze contents
    """
    print("="*70)
    print("CONFORMATIONAL DATA INVENTORY")
    print("="*70)
    
    # Find all batch ZIP files
    zip_files = []
    for file in Path("./confirmation-zips").glob("*.zip"):
        if "batch" in file.name.lower() and "conf" in file.name.lower():
            zip_files.append(file)
    
    if not zip_files:
        print("\n✗ No batch ZIP files found in current directory")
        print("  Make sure you're in the directory with downloaded ZIPs")
        return
    
    # Sort by batch and conformation
    zip_files.sort()
    
    print(f"\nFound {len(zip_files)} ZIP files:\n")
    
    batch_data = {}  # {batch_name: {conf_idx: {protein_id: pdb_count}}}
    
    for zip_file in zip_files:
        print(f"Analyzing: {zip_file.name}")
        
        # Extract batch and conformation info
        # Examples: batch_conf_1_results.zip, batch2_conf_1_results.zip, batch3(2)_conf_1_results.zip
        match = re.search(r'batch(\d+)?(?:\(\d+\))?_conf_(\d+)', zip_file.name)
        
        if match:
            batch_num = match.group(1) or "1"
            conf_num = match.group(2)
            batch_key = f"batch{batch_num}"
            
            if batch_key not in batch_data:
                batch_data[batch_key] = {}
            
            # Analyze ZIP contents
            with zipfile.ZipFile(zip_file, 'r') as zf:
                pdb_files = [f for f in zf.namelist() if f.endswith('.pdb')]
                
                # Extract protein IDs
                proteins = set()
                for pdb in pdb_files:
                    pdb_name = Path(pdb).name
                    # Extract protein ID: POC_PROT_AURKA_Length_414_unrelaxed...
                    match_prot = re.search(r'POC_PROT_(\w+)_Length', pdb_name)
                    if match_prot:
                        proteins.add(match_prot.group(1))
                
                batch_data[batch_key][int(conf_num)] = {
                    'zip_file': zip_file.name,
                    'pdb_count': len(pdb_files),
                    'proteins': sorted(proteins)
                }
                
                print(f"  ✓ Conf {conf_num}: {len(proteins)} proteins, {len(pdb_files)} PDB files")
        else:
            print(f"  ✗ Could not parse filename pattern")
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY BY BATCH")
    print("="*70)
    
    all_proteins = set()
    
    for batch_key in sorted(batch_data.keys()):
        print(f"\n{batch_key.upper()}:")
        confs = batch_data[batch_key]
        
        for conf_idx in sorted(confs.keys()):
            info = confs[conf_idx]
            proteins = info['proteins']
            all_proteins.update(proteins)
            print(f"  Conformation {conf_idx}: {len(proteins)} proteins")
            print(f"    File: {info['zip_file']}")
        
        # Check completeness
        conf_counts = list(confs.keys())
        if conf_counts == [1, 2, 3]:
            print(f"  Status: ✓ Complete (3 conformations)")
        elif conf_counts == [1, 2]:
            print(f"  Status: ⚠ Incomplete (2 conformations, missing conf 3)")
        else:
            print(f"  Status: ⚠ Partial ({len(conf_counts)} conformations)")
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Total unique proteins across all batches: {len(all_proteins)}")
    print(f"Target: 50 proteins")
    
    if len(all_proteins) >= 50:
        print("✓ Sufficient protein coverage")
    else:
        missing = 50 - len(all_proteins)
        print(f"⚠ Missing {missing} proteins")
    
    # Save detailed report
    report_path = "conformation_inventory_report.txt"
    with open(report_path, 'w') as f:
        f.write("CONFORMATIONAL DATA INVENTORY REPORT\n")
        f.write("="*70 + "\n\n")
        
        for batch_key in sorted(batch_data.keys()):
            f.write(f"\n{batch_key.upper()}:\n")
            confs = batch_data[batch_key]
            
            for conf_idx in sorted(confs.keys()):
                info = confs[conf_idx]
                f.write(f"\n  Conformation {conf_idx}:\n")
                f.write(f"    File: {info['zip_file']}\n")
                f.write(f"    Proteins ({len(info['proteins'])}):\n")
                for prot in info['proteins']:
                    f.write(f"      - {prot}\n")
    
    print(f"\nDetailed report saved: {report_path}")
    
    return batch_data, all_proteins

if __name__ == "__main__":
    inventory_all_batches()
