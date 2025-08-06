#!/usr/bin/env python3
"""
ECFP6 Tanimoto Similarity Analysis for 357 Protein Targets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def compute_ecfp6_similarities(df):
    """Compute ECFP6 fingerprints and Tanimoto similarities"""
    print("Computing ECFP6 fingerprints...")
    
    # Generate molecules and filter valid ones
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notna()].reset_index(drop=True)
    
    # Generate ECFP6 fingerprints (radius=3)
    generator = GetMorganGenerator(radius=3, fpSize=2048)
    df["ecfp6"] = df["mol"].apply(generator.GetFingerprint)
    
    print(f"Generated fingerprints for {len(df)} compounds")
    return df

def analyze_target_similarities(df):
    """Analyze similarity distributions across protein targets"""
    print("Analyzing target distributions...")
    
    # Get protein target columns (exclude non-target columns)
    protein_cols = [col for col in df.columns if col not in ['smiles', 'mol', 'ecfp6']]
    print(f"Found {len(protein_cols)} protein targets")
    
    # Convert to numeric and create melted dataframe
    for col in protein_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_melted = df.melt(
        id_vars=['smiles', 'ecfp6'], 
        value_vars=protein_cols,
        var_name='target', 
        value_name='pIC50'
    ).dropna(subset=['pIC50'])
    
    # Compute pairwise similarities for compounds with target data
    target_stats = []
    
    for target in protein_cols:
        target_data = df_melted[df_melted['target'] == target]
        if len(target_data) < 2:
            continue
            
        fps = list(target_data['ecfp6'])
        n = len(fps)
        
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        if similarities:
            target_stats.append({
                'target': target,
                'n_compounds': n,
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities)
            })
        
        print(f"Processed {target}: {n} compounds")
    
    return pd.DataFrame(target_stats), df_melted

def main():
    # Load dataset
    print("Loading ChEMBL dataset...")
    df = pd.read_csv("datasets/chembl_pretraining.csv")
    print(f"Loaded {len(df)} compounds with {df.shape[1]-1} targets")
    
    # Compute fingerprints and similarities
    df = compute_ecfp6_similarities(df)
    
    # Analyze target distributions
    target_stats, df_melted = analyze_target_similarities(df)
    
    # Generate summary
    print("\n=== SIMILARITY ANALYSIS SUMMARY ===")
    print(target_stats.to_string(index=False))
    
    # Simple visualization
    if len(target_stats) > 0:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(target_stats)), target_stats['mean_similarity'])
        plt.xlabel('Target Index')
        plt.ylabel('Mean Tanimoto Similarity')
        plt.title('Mean Similarity by Target')
        
        plt.subplot(1, 2, 2)
        plt.hist(target_stats['mean_similarity'], bins=10)
        plt.xlabel('Mean Tanimoto Similarity')
        plt.ylabel('Number of Targets')
        plt.title('Distribution of Mean Similarities')
        
        plt.tight_layout()
        plt.savefig('tanimoto_target_analysis.png', dpi=150)
        plt.show()
    
    # Save results
    target_stats.to_csv('target_similarity_stats.csv', index=False)
    print("\nResults saved to target_similarity_stats.csv")

if __name__ == "__main__":
    main()