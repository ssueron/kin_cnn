"""
Generate mask files for kinase datasets if they don't exist
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_mask_file(csv_path, output_path=None):
    """Generate mask file from data CSV (True = has data, False = missing)"""
    
    df = pd.read_csv(csv_path)
    
    # Get SMILES column
    if 'curated_smiles' in df.columns:
        smiles_col = 'curated_smiles'
    elif 'smiles' in df.columns:
        smiles_col = 'smiles'
    elif 'molecule' in df.columns:
        smiles_col = 'molecule'
    else:
        raise ValueError(f"No SMILES column found in {csv_path}. Expected 'curated_smiles', 'smiles', or 'molecule'")
    
    # Get kinase columns
    kinase_cols = [col for col in df.columns if col != smiles_col]
    
    # Create mask dataframe efficiently using pd.concat
    mask_data = {smiles_col: df[smiles_col]}
    mask_data.update({col: ~df[col].isna() for col in kinase_cols})
    mask_df = pd.DataFrame(mask_data)
    
    # Save mask file
    if output_path is None:
        output_path = csv_path.replace('.csv', '_mask.csv')
    
    mask_df.to_csv(output_path, index=False)
    
    # Print statistics
    total_values = len(mask_df) * len(kinase_cols)
    present_values = mask_df[kinase_cols].sum().sum()
    sparsity = (1 - present_values / total_values) * 100
    
    print(f"Generated mask for {Path(csv_path).name}:")
    print(f"  Compounds: {len(mask_df)}")
    print(f"  Kinases: {len(kinase_cols)}")
    print(f"  Sparsity: {sparsity:.1f}%")
    print(f"  Saved to: {output_path}")
    
    return mask_df

def generate_all_masks():
    """Generate mask files for all datasets"""
    
    datasets = [
        'datasets/chembl_pretraining_train.csv',
        'datasets/chembl_pretraining_val.csv',
        'datasets/chembl_pretraining_test.csv',
        'datasets/pkis2_finetuning_train.csv',
        'datasets/pkis2_finetuning_val.csv',
        'datasets/pkis2_finetuning_test.csv'
    ]
    
    for dataset_path in datasets:
        if Path(dataset_path).exists():
            mask_path = dataset_path.replace('.csv', '_mask.csv')
            if not Path(mask_path).exists():
                generate_mask_file(dataset_path)
            else:
                print(f"Mask already exists for {Path(dataset_path).name}")
        else:
            print(f"WARNING: Dataset not found: {dataset_path}")

def custom_smiles_encoding(smiles: str, vocab: dict):
    """Custom SMILES encoding function for experiment_runner.py"""
    import re
    
    # Create regex pattern with tokens sorted by decreasing length
    tokens = sorted([k for k in vocab.keys() if k not in ["*", "^", "$", "UNK"]], key=len, reverse=True)
    pattern = '|'.join(re.escape(token) for token in tokens)
    
    # Tokenize with regex
    matches = re.findall(f'({pattern})', smiles)
    
    # Convert to IDs with handling of unknown tokens
    token_ids = []
    for token in matches:
        token_ids.append(vocab.get(token, vocab.get("UNK", 3)))
    
    return token_ids

def verify_data_structure():
    """Verify all required files exist"""
    print("Verifying data structure...")
    print("="*50)
    
    required_files = {
        'ChEMBL Pretraining': [
            'datasets/chembl_pretraining_train.csv',
            'datasets/chembl_pretraining_val.csv',
            'datasets/chembl_pretraining_test.csv'
        ],
        'PKIS2 Finetuning': [
            'datasets/pkis2_finetuning_train.csv',
            'datasets/pkis2_finetuning_val.csv',
            'datasets/pkis2_finetuning_test.csv'
        ],
        'Vocabularies': [
            'data/smiles_vocab.json',
            'data/selfies_vocab.json'
        ]
    }
    
    all_good = True
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file_path in files:
            if Path(file_path).exists():
                df_shape = ""
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, nrows=1)
                    n_rows = len(pd.read_csv(file_path))
                    n_cols = len(df.columns)
                    df_shape = f" ({n_rows} rows, {n_cols} cols)"
                print(f"  ✓ {file_path}{df_shape}")
            else:
                print(f"  ✗ {file_path} NOT FOUND")
                all_good = False
    
    if all_good:
        print("\n✓ All required files found!")
    else:
        print("\n✗ Some files are missing. Please check your data directory.")
    
    return all_good

if __name__ == "__main__":
    import sys
    
    print("Data Preparation Script")
    print("="*50)
    
    # First verify structure
    if verify_data_structure():
        print("\n" + "="*50)
        print("Generating mask files...")
        print("="*50 + "\n")
        
        generate_all_masks()
        
        print("\n" + "="*50)
        print("Data preparation complete!")
        print("You can now run experiments with:")
        print("  python run_experiments.py test")
        print("  python run_experiments.py baseline")
        print("  python run_experiments.py all")
    else:
        print("\nPlease ensure all data files are in place before proceeding.")