"""
Download Davis dataset from Therapeutics Data Commons
"""
from tdc.multi_pred import DTI
import pandas as pd
import os

def download_davis():
    print("Downloading Davis dataset from TDC...")
    
    # Initialize DTI dataset
    data = DTI(name='Davis')
    
    # Get data splits
    split = data.get_split()
    
    # Combine all splits
    train = split['train']
    valid = split['valid']
    test = split['test']
    
    full_data = pd.concat([train, valid, test], ignore_index=True)
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)
    
    # Save full dataset
    full_data.to_csv('data/raw/davis_full.csv', index=False)
    
    # Save splits
    train.to_csv('data/splits/davis_train.csv', index=False)
    valid.to_csv('data/splits/davis_valid.csv', index=False)
    test.to_csv('data/splits/davis_test.csv', index=False)
    
    # Print statistics
    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Total interactions: {len(full_data)}")
    print(f"  Unique proteins: {full_data['Target'].nunique()}")
    print(f"  Unique drugs: {full_data['Drug'].nunique()}")
    print(f"  Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    print(f"\nColumns: {list(full_data.columns)}")
    print(f"\nFirst few rows:")
    print(full_data.head())
    
    return full_data

if __name__ == "__main__":
    download_davis()
