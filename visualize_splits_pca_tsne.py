#!/usr/bin/env python3
"""
PCA/t-SNE Visualization of Molecular Splits

This script creates PCA and t-SNE visualizations to analyze the chemical space
coverage of train/validation/test splits using molecular fingerprints.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore')

def smiles_to_fingerprint(smiles, radius=3, fp_size=2048, use_features=False):
    """
    Convert SMILES to modern Morgan fingerprint using MorganGenerator
    
    Parameters:
    - radius: Morgan fingerprint radius (default 3, equivalent to ECFP6)
    - fp_size: Fingerprint bit vector size
    - use_features: Use feature-based Morgan fingerprints (considers atom features)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.uint8)
        
        # Create Morgan fingerprint generator (modern API with correct parameter names)
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius,  # positional argument
            countSimulation=False,
            includeChirality=True,
            useBondTypes=True,
            onlyNonzeroInvariants=False,
            includeRingMembership=True,
            countBounds=None,
            fpSize=fp_size,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if use_features else None
        )
        
        # Generate fingerprint
        fp = generator.GetFingerprint(mol)
        return np.array(fp, dtype=np.uint8)
        
    except Exception as e:
        print(f"Warning: Failed to generate fingerprint for SMILES '{smiles}': {e}")
        return np.zeros(fp_size, dtype=np.uint8)

def calculate_molecular_descriptors(smiles):
    """Calculate basic molecular descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0, 0, 0, 0]
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rotbonds = Descriptors.NumRotatableBonds(mol)
        
        return [mw, logp, tpsa, rotbonds]
    except:
        return [0, 0, 0, 0]

def load_split_data(train_file, val_file, test_file, smiles_col='smiles'):
    """Load and combine split data with labels"""
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return combined_df

def create_pca_tsne_visualization(df, smiles_col='smiles', use_descriptors=False, use_features=False):
    """Create PCA and t-SNE visualizations"""
    print("Computing molecular representations...")
    
    if use_descriptors:
        # Use molecular descriptors
        descriptors = []
        for smiles in df[smiles_col]:
            desc = calculate_molecular_descriptors(smiles)
            descriptors.append(desc)
        
        X = np.array(descriptors)
        feature_names = ['MW', 'LogP', 'TPSA', 'RotBonds']
        print(f"Using {len(feature_names)} molecular descriptors")
    else:
        # Use modern Morgan fingerprints
        fingerprints = []
        fp_type = "feature-based" if use_features else "standard"
        print(f"Generating {fp_type} Morgan fingerprints (radius=3, ECFP6 equivalent)...")
        
        for smiles in df[smiles_col]:
            fp = smiles_to_fingerprint(smiles, radius=3, use_features=use_features)
            fingerprints.append(fp)
        
        X = np.array(fingerprints)
        print(f"Using Morgan fingerprints with {X.shape[1]} bits ({fp_type})")
    
    # Remove any rows with all zeros
    valid_mask = ~np.all(X == 0, axis=1)
    X_clean = X[valid_mask]
    df_clean = df[valid_mask].copy()
    
    print(f"Valid molecules: {len(df_clean)} / {len(df)}")
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_clean)
    
    # t-SNE
    print("Computing t-SNE...")
    # Use PCA preprocessing for t-SNE if using fingerprints
    if not use_descriptors and X_clean.shape[1] > 50:
        pca_prep = PCA(n_components=50)
        X_prep = pca_prep.fit_transform(X_clean)
    else:
        X_prep = X_clean
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_prep)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'train': 'blue', 'val': 'orange', 'test': 'red'}
    splits = df_clean['split'].unique()
    
    # PCA plot
    for split in splits:
        mask = df_clean['split'] == split
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[split], label=f'{split} ({mask.sum()})', 
                       alpha=0.6, s=20)
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].set_title('PCA: Chemical Space Coverage')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE plot
    for split in splits:
        mask = df_clean['split'] == split
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[split], label=f'{split} ({mask.sum()})', 
                       alpha=0.6, s=20)
    
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE: Chemical Space Coverage')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if use_descriptors:
        method = "descriptors"
    else:
        method = "fingerprints_features" if use_features else "fingerprints_standard"
    
    plt.savefig(f'splits_chemical_space_{method}.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization as 'splits_chemical_space_{method}.png'")
    
    # Print summary statistics
    print(f"\nPCA Explained Variance:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  Total: {pca.explained_variance_ratio_[:2].sum():.1%}")
    
    return X_pca, X_tsne, df_clean

def main():
    """Main function to run PCA/t-SNE analysis"""
    print("PCA/t-SNE Analysis of Molecular Splits")
    print("=" * 50)
    
    # File paths - adjust as needed
    train_file = 'chembl_cluster_split_train.csv'
    val_file = 'chembl_cluster_split_val.csv'  
    test_file = 'chembl_cluster_split_test.csv'
    
    try:
        # Load data
        print("Loading split data...")
        df = load_split_data(train_file, val_file, test_file)
        print(f"Loaded {len(df)} compounds across splits:")
        print(df['split'].value_counts())
        
        # Create visualizations using standard Morgan fingerprints
        print("\n" + "="*40)
        print("Using Standard Morgan Fingerprints:")
        create_pca_tsne_visualization(df, use_descriptors=False, use_features=False)
        
        # Create visualizations using feature-based Morgan fingerprints
        print("\n" + "="*40)
        print("Using Feature-Based Morgan Fingerprints:")
        create_pca_tsne_visualization(df, use_descriptors=False, use_features=True)
        
        # Create visualizations using molecular descriptors
        print("\n" + "="*40) 
        print("Using Molecular Descriptors:")
        create_pca_tsne_visualization(df, use_descriptors=True)
        
        print("\n" + "="*50)
        print("Analysis complete! Generated files:")
        print("  - splits_chemical_space_fingerprints_standard.png")
        print("  - splits_chemical_space_fingerprints_features.png")
        print("  - splits_chemical_space_descriptors.png")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find split files. Please check file paths.")
        print(f"Expected files: {train_file}, {val_file}, {test_file}")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()