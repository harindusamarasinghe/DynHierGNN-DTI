# Create a new Python file: data_exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the BindingDB TSV file
df = pd.read_csv('BindingDB_All_202509.tsv', sep='\t')

# Basic inspection
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"First 5 rows:\n{df.head()}")
print(f"Data types:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")
