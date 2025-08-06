import pandas as pd
import numpy as np

# Load and check the dataset structure
df = pd.read_csv('chembl_pretraining.csv')

print('Dataset shape:', df.shape)
print('First few columns:', df.columns[:10].tolist())

# Get protein columns (exclude smiles)
protein_columns = [col for col in df.columns if col != 'smiles']
print('Number of protein targets:', len(protein_columns))

# Convert protein columns to numeric and check sparsity
for col in protein_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check how many non-null values each target has
non_null_counts = df[protein_columns].count()
print()
print('Top 10 targets by number of compounds:')
print(non_null_counts.nlargest(10))

print()
print('Bottom 10 targets by number of compounds:')
print(non_null_counts.nsmallest(10))

print()
print('Targets with 0 compounds:', (non_null_counts == 0).sum())
print('Targets with 1-10 compounds:', ((non_null_counts > 0) & (non_null_counts <= 10)).sum())
print('Targets with >100 compounds:', (non_null_counts > 100).sum())