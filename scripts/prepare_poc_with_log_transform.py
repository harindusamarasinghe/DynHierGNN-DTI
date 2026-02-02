import pandas as pd
import numpy as np

def convert_to_pIC50(ic50_nm):
    """Convert IC50 (nM) to pIC50 = -log10(IC50 in Molar)"""
    # IC50 in nM -> Molar: divide by 1e9
    # pIC50 = -log10(IC50_M) = -log10(IC50_nM / 1e9) = 9 - log10(IC50_nM)
    if ic50_nm <= 0:
        return np.nan
    return 9 - np.log10(ic50_nm)

for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/poc/poc_{split}.csv')
    
    # Convert IC50 to pIC50
    df['Y_original'] = df['Y']  # Keep original for reference
    df['Y'] = df['Y'].apply(convert_to_pIC50)
    
    # Remove invalid values
    df = df.dropna(subset=['Y'])
    
    print(f"{split}: {len(df)} samples, pIC50 range: {df['Y'].min():.2f} - {df['Y'].max():.2f}")
    
    df.to_csv(f'data/poc/poc_{split}.csv', index=False)

print("\n✓ Data transformed successfully!")
