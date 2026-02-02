#!/usr/bin/env python3
"""
Extract ESM-2 embeddings for ALL proteins (batch1 + batch2)
Runs independently from conformation generation
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_esm2_embeddings():
    """Extract ESM-2 embeddings for ALL proteins (batch1 + batch2)"""
    
    start_time = time.time()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cpu":
        logger.warning("Running on CPU - will take 3-5 hours for 100 proteins")
        logger.info("For faster execution, use GPU (30-45 min)")
    
    # Load ESM-2 model
    logger.info("="*70)
    logger.info("LOADING ESM-2 MODEL (650M parameters)")
    logger.info("="*70)
    logger.info("First time: Downloads ~2.5 GB model")
    logger.info("Subsequent runs: Loads from cache")
    
    try:
        model, alphabet = torch.hub.load(
            "facebookresearch/esm:main",
            "esm2_t33_650M_UR50D"
        )
        logger.info("✓ Loaded ESM-2 from torch hub")
    except Exception as e:
        logger.error(f"Torch hub load failed: {e}")
        logger.info("Trying alternative method...")
        try:
            import esm
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            logger.info("✓ Loaded ESM-2 from esm library")
        except Exception as e2:
            logger.error(f"ESM library load failed: {e2}")
            logger.error("Please install: pip install fair-esm")
            return
    
    model = model.to(device)
    model.eval()
    logger.info("✓ ESM-2 model loaded and ready")
    
    # Load ALL protein sequences (batch1 + batch2)
    logger.info("\n" + "="*70)
    logger.info("LOADING ALL PROTEINS (BATCH1 + BATCH2)")
    logger.info("="*70)
    
    # Load batch1 (POC proteins)
    batch1_proteins = pd.read_csv('data/poc/protein_mapping.csv')
    logger.info(f"Batch1 proteins: {len(batch1_proteins)}")
    
    # Load batch2 (new proteins)
    batch2_proteins = pd.read_csv('data/poc/batch2_protein_mapping.csv')
    logger.info(f"Batch2 proteins: {len(batch2_proteins)}")
    
    # Combine
    all_proteins = pd.concat([batch1_proteins, batch2_proteins], ignore_index=True)
    logger.info(f"Total proteins: {len(all_proteins)}")
    logger.info(f"Length range: {all_proteins['Length'].min()}-{all_proteins['Length'].max()} aa")
    
    # Create embeddings directory
    embeddings_dir = Path("embeddings/all_proteins")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {embeddings_dir}")
    
    # Check for existing embeddings
    existing_embeddings = list(embeddings_dir.glob("*_esm2.npy"))
    if existing_embeddings:
        logger.info(f"Found {len(existing_embeddings)} existing embeddings")
        logger.info("Will skip already processed proteins")
    
    # Extract embeddings
    batch_converter = alphabet.get_batch_converter()
    
    logger.info("\n" + "="*70)
    logger.info("EXTRACTING ESM-2 EMBEDDINGS FOR ALL 100 PROTEINS")
    logger.info("="*70)
    logger.info("Progress will be saved after each protein")
    logger.info("Safe to interrupt and resume later")
    logger.info("="*70)
    
    successful = 0
    skipped = 0
    failed = 0
    
    for idx, row in tqdm(all_proteins.iterrows(), total=len(all_proteins), desc="Proteins"):
        protein_id = row['Protein_ID']
        sequence = row['Sequence']
        
        # Check if already processed
        save_path = embeddings_dir / f"{protein_id}_esm2.npy"
        if save_path.exists():
            skipped += 1
            continue
        
        try:
            # Prepare input
            data = [(protein_id, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            # Extract embeddings
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            
            # Get layer 33 embeddings [batch, seq_len, 1280]
            embeddings = results["representations"][33]
            
            # Remove BOS/EOS tokens
            token_embeddings = embeddings[0, 1:-1, :]  # [seq_len, 1280]
            
            # Save immediately
            np.save(save_path, token_embeddings.cpu().numpy())
            
            successful += 1
            
            # Log progress every 10 proteins
            if (successful + skipped) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = len(all_proteins) - (successful + skipped + failed)
                avg_time = elapsed / (successful + skipped) if (successful + skipped) > 0 else 0
                est_remaining = avg_time * remaining / 60  # minutes
                
                logger.info(f"\nProgress: {successful + skipped}/{len(all_proteins)} | "
                          f"Success: {successful} | Skipped: {skipped} | Failed: {failed}")
                logger.info(f"Estimated time remaining: {est_remaining:.1f} minutes")
        
        except Exception as e:
            logger.error(f"Error processing {protein_id}: {e}")
            failed += 1
            continue
    
    # Final summary
    elapsed_total = time.time() - start_time
    
    logger.info("\n" + "="*70)
    logger.info("ESM-2 EXTRACTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total proteins: {len(all_proteins)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {elapsed_total/60:.1f} minutes")
    if successful > 0:
        logger.info(f"Average time per protein: {elapsed_total/successful:.1f} seconds")
    logger.info(f"\nOutput location: {embeddings_dir}")
    
    # Verify embeddings
    saved_files = list(embeddings_dir.glob("*_esm2.npy"))
    logger.info(f"\n✓ Saved {len(saved_files)} embedding files")
    
    # Check one file
    if saved_files:
        sample = np.load(saved_files[0])
        logger.info(f"✓ Sample embedding shape: {sample.shape} (seq_len, 1280)")
        logger.info(f"✓ Data type: {sample.dtype}")
    
    logger.info("\n✓ Ready for graph construction with ESM-2 features!")
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("1. Wait for conformations to finish generating")
    logger.info("2. Build protein graphs with ESM-2 features (1300-dim)")
    logger.info("3. Train model and validate improvement")

if __name__ == "__main__":
    extract_esm2_embeddings()
