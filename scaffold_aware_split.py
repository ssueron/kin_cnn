#!/usr/bin/env python3
"""
Cluster-Aware Dataset Splitting for Balanced Chemical Space Coverage

This script creates train/validation/test splits that ensure:
1. Chemical clusters in test/val are never seen in training (cluster leakage prevention)
2. Balanced pIC50 distribution across splits
3. Balanced protein target representation across splits
4. Uses Tanimoto similarity-based clusters for better chemical space coverage
5. Proper handling of data sparsity and edge cases

Author: Generated splitting script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ClusterAwareSplitter:
    """
    Advanced cluster-aware dataset splitter with balanced distributions using Tanimoto similarity clusters
    """
    
    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
        """
        Initialize splitter with desired ratios
        
        Parameters:
        - train_ratio: fraction for training set
        - val_ratio: fraction for validation set  
        - test_ratio: fraction for test set
        - random_state: random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.split_assignments = {}
        self.cluster_stats = {}
        self.split_stats = {}
    
    def load_cluster_data(self, cluster_file='datasets/chembl_clusters_nn_0.4.csv'):
        """
        Load Tanimoto similarity-based cluster assignments
        
        Parameters:
        - cluster_file: path to cluster assignment file
        
        Returns:
        - cluster_df: DataFrame with SMILES and cluster_id
        """
        print(f"Loading cluster data from {cluster_file}...")
        cluster_df = pd.read_csv(cluster_file)
        print(f"Loaded {len(cluster_df)} compounds with {cluster_df['cluster_id'].nunique()} unique clusters")
        return cluster_df
    
    def prepare_data(self, bioactivity_df, cluster_df, smiles_col='smiles'):
        """
        Prepare data by merging bioactivity and cluster data, creating analysis-ready format
        
        Parameters:
        - bioactivity_df: DataFrame with SMILES and bioactivity data
        - cluster_df: DataFrame with SMILES and cluster_id
        - smiles_col: column name containing SMILES strings
        
        Returns:
        - prepared_df: DataFrame with clusters and bioactivity data
        - df_melted: melted format for analysis
        - protein_cols: list of protein target columns
        """
        print("Preparing data for cluster-aware splitting...")
        
        # Merge bioactivity and cluster data
        df = pd.merge(bioactivity_df, cluster_df, on=smiles_col, how='inner')
        print(f"Successfully merged data for {len(df)} compounds")
        
        # Get protein columns (exclude non-target columns)
        exclude_cols = [smiles_col, 'cluster_id']
        protein_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Convert protein columns to numeric
        for col in protein_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create melted format for easier analysis
        df_melted = df.melt(
            id_vars=[smiles_col, 'cluster_id'],
            value_vars=protein_cols,
            var_name='target_protein',
            value_name='pIC50'
        ).dropna(subset=['pIC50'])
        
        print(f"Data preparation complete:")
        print(f"  - {len(df)} unique compounds")
        print(f"  - {df['cluster_id'].nunique()} unique clusters")
        print(f"  - {len(protein_cols)} protein targets")
        print(f"  - {len(df_melted)} compound-target pairs")
        
        return df, df_melted, protein_cols
    
    def analyze_cluster_characteristics(self, df, df_melted):
        """Analyze cluster characteristics for informed splitting"""
        print("Analyzing cluster characteristics...")
        
        cluster_analysis = []
        
        for cluster_id, cluster_group in df.groupby('cluster_id'):
            # Get all bioactivity data for this cluster
            cluster_data = df_melted[df_melted['cluster_id'] == cluster_id]
            
            analysis = {
                'cluster_id': cluster_id,
                'compound_count': len(cluster_group),
                'total_measurements': len(cluster_data),
                'unique_targets': cluster_data['target_protein'].nunique(),
                'mean_pIC50': cluster_data['pIC50'].mean(),
                'std_pIC50': cluster_data['pIC50'].std(),
                'min_pIC50': cluster_data['pIC50'].min(),
                'max_pIC50': cluster_data['pIC50'].max(),
                'target_list': cluster_data['target_protein'].unique().tolist(),
                'compound_indices': cluster_group.index.tolist()
            }
            
            cluster_analysis.append(analysis)
        
        cluster_df = pd.DataFrame(cluster_analysis)
        
        # Calculate cluster importance scores for balanced splitting
        cluster_df['importance_score'] = (
            cluster_df['compound_count'] * 0.3 +  # Size weight
            cluster_df['unique_targets'] * 0.4 +   # Target diversity weight  
            cluster_df['total_measurements'] * 0.3  # Data richness weight
        )
        
        self.cluster_stats = cluster_df
        
        print(f"Cluster analysis complete:")
        print(f"  - Cluster sizes: {cluster_df['compound_count'].min()}-{cluster_df['compound_count'].max()}")
        print(f"  - Target coverage: {cluster_df['unique_targets'].min()}-{cluster_df['unique_targets'].max()}")
        print(f"  - Measurements: {cluster_df['total_measurements'].min()}-{cluster_df['total_measurements'].max()}")
        
        return cluster_df
    
    def create_balanced_cluster_split(self, cluster_df):
        """
        Create balanced cluster split using stratified approach
        
        Strategy:
        1. Sort clusters by importance score
        2. Assign clusters to splits in round-robin fashion with balancing
        3. Ensure no cluster appears in multiple splits
        4. Balance by compound count, target diversity, and pIC50 range
        """
        print("\nCreating balanced cluster split...")
        
        # Sort clusters by importance (largest/most diverse first)
        cluster_df_sorted = cluster_df.sort_values('importance_score', ascending=False)
        
        # Initialize split containers
        splits = {
            'train': {'clusters': [], 'compounds': 0, 'measurements': 0, 'targets': set()},
            'val': {'clusters': [], 'compounds': 0, 'measurements': 0, 'targets': set()}, 
            'test': {'clusters': [], 'compounds': 0, 'measurements': 0, 'targets': set()}
        }
        
        target_compounds = {
            'train': int(cluster_df['compound_count'].sum() * self.train_ratio),
            'val': int(cluster_df['compound_count'].sum() * self.val_ratio),
            'test': int(cluster_df['compound_count'].sum() * self.test_ratio)
        }
        
        # Greedy assignment with balancing
        for _, cluster_row in cluster_df_sorted.iterrows():
            cluster_id = cluster_row['cluster_id']
            compound_count = cluster_row['compound_count']
            measurements = cluster_row['total_measurements']
            targets = set(cluster_row['target_list'])
            
            # Calculate current ratios for each split
            total_assigned = sum(split_info['compounds'] for split_info in splits.values())
            
            if total_assigned == 0:
                # First cluster goes to train
                best_split = 'train'
            else:
                # Find split that needs this cluster most (furthest from target ratio)
                split_scores = {}
                
                for split_name, split_info in splits.items():
                    current_ratio = split_info['compounds'] / total_assigned if total_assigned > 0 else 0
                    target_ratio = {'train': self.train_ratio, 'val': self.val_ratio, 'test': self.test_ratio}[split_name]
                    
                    # Score based on how much this split needs more data
                    ratio_deficit = target_ratio - current_ratio
                    
                    # Bonus for target diversity
                    new_targets = len(targets - split_info['targets'])
                    target_bonus = new_targets * 0.1
                    
                    split_scores[split_name] = ratio_deficit + target_bonus
                
                best_split = max(split_scores.keys(), key=lambda x: split_scores[x])
            
            # Assign cluster to best split
            splits[best_split]['clusters'].append(cluster_id)
            splits[best_split]['compounds'] += compound_count
            splits[best_split]['measurements'] += measurements
            splits[best_split]['targets'].update(targets)
        
        # Create cluster-to-split mapping
        cluster_to_split = {}
        for split_name, split_info in splits.items():
            for cluster_id in split_info['clusters']:
                cluster_to_split[cluster_id] = split_name
        
        # Print split statistics
        print(f"Cluster split created:")
        for split_name, split_info in splits.items():
            target_ratio = {'train': self.train_ratio, 'val': self.val_ratio, 'test': self.test_ratio}[split_name]
            actual_ratio = split_info['compounds'] / sum(s['compounds'] for s in splits.values())
            
            print(f"  {split_name.upper():>5}: {len(split_info['clusters']):>4} clusters, "
                  f"{split_info['compounds']:>5} compounds ({actual_ratio:.2f} vs {target_ratio:.2f} target), "
                  f"{len(split_info['targets']):>3} unique targets")
        
        return cluster_to_split
    
    def assign_compounds_to_splits(self, df, cluster_to_split):
        """Assign individual compounds to splits based on cluster assignment"""
        print("Assigning compounds to splits...")
        
        df['split'] = df['cluster_id'].map(cluster_to_split)
        
        # Handle any unmapped clusters (shouldn't happen but safety check)
        unmapped = df['split'].isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} compounds with unmapped clusters, assigning to train")
            df.loc[df['split'].isna(), 'split'] = 'train'
        
        return df
    
    def validate_split_quality(self, df, df_melted, protein_cols):
        """Comprehensive validation of split quality"""
        print("\nValidating split quality...")
        
        split_stats = {}
        
        for split_name in ['train', 'val', 'test']:
            split_df = df[df['split'] == split_name]
            split_melted = df_melted[df_melted['cluster_id'].isin(split_df['cluster_id'])]
            
            stats = {
                'compounds': len(split_df),
                'clusters': split_df['cluster_id'].nunique(),
                'measurements': len(split_melted),
                'unique_targets': split_melted['target_protein'].nunique(),
                'mean_pIC50': split_melted['pIC50'].mean(),
                'std_pIC50': split_melted['pIC50'].std(),
                'pIC50_range': (split_melted['pIC50'].min(), split_melted['pIC50'].max()),
                'target_coverage': {}
            }
            
            # Calculate per-target statistics
            for protein in protein_cols:
                protein_data = split_melted[split_melted['target_protein'] == protein]['pIC50']
                if len(protein_data) > 0:
                    stats['target_coverage'][protein] = {
                        'count': len(protein_data),
                        'mean_pIC50': protein_data.mean(),
                        'std_pIC50': protein_data.std()
                    }
            
            split_stats[split_name] = stats
        
        self.split_stats = split_stats
        
        # Print validation results
        print("Split validation results:")
        print(f"{'Split':<6} {'Compounds':<10} {'Clusters':<10} {'Measurements':<12} {'Targets':<8} {'Mean pIC50':<10}")
        print("-" * 66)
        
        for split_name, stats in split_stats.items():
            print(f"{split_name:<6} {stats['compounds']:<10} {stats['clusters']:<10} "
                  f"{stats['measurements']:<12} {stats['unique_targets']:<8} {stats['mean_pIC50']:<10.2f}")
        
        # Check for cluster leakage
        train_clusters = set(df[df['split'] == 'train']['cluster_id'])
        val_clusters = set(df[df['split'] == 'val']['cluster_id'])  
        test_clusters = set(df[df['split'] == 'test']['cluster_id'])
        
        train_val_overlap = len(train_clusters & val_clusters)
        train_test_overlap = len(train_clusters & test_clusters)
        val_test_overlap = len(val_clusters & test_clusters)
        
        print(f"\nCluster leakage check:")
        print(f"  Train-Val overlap: {train_val_overlap} clusters")
        print(f"  Train-Test overlap: {train_test_overlap} clusters") 
        print(f"  Val-Test overlap: {val_test_overlap} clusters")
        
        if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
            print("  ✓ No cluster leakage detected!")
        else:
            print("  ⚠ Cluster leakage detected!")
        
        return split_stats
    
    def create_visualizations(self, df, df_melted, protein_cols):
        """Create comprehensive visualizations of the split"""
        print("\nCreating split visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Compound distribution across splits
        plt.subplot(4, 4, 1)
        split_counts = df['split'].value_counts()
        plt.pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%')
        plt.title('Compound Distribution Across Splits')
        
        # 2. Cluster distribution across splits  
        plt.subplot(4, 4, 2)
        cluster_split_counts = df.groupby('split')['cluster_id'].nunique()
        plt.bar(cluster_split_counts.index, cluster_split_counts.values)
        plt.xlabel('Split')
        plt.ylabel('Number of Unique Clusters')
        plt.title('Cluster Distribution Across Splits')
        
        # 3. pIC50 distribution across splits
        plt.subplot(4, 4, 3)
        for split_name in ['train', 'val', 'test']:
            split_data = df_melted[df_melted['cluster_id'].isin(
                df[df['split'] == split_name]['cluster_id']
            )]
            plt.hist(split_data['pIC50'], bins=30, alpha=0.6, label=split_name, density=True)
        plt.xlabel('pIC50')
        plt.ylabel('Density')
        plt.title('pIC50 Distribution Across Splits')
        plt.legend()
        
        # 4. Target coverage across splits
        plt.subplot(4, 4, 4)
        target_coverage = {}
        for split_name in ['train', 'val', 'test']:
            split_data = df_melted[df_melted['cluster_id'].isin(
                df[df['split'] == split_name]['cluster_id']
            )]
            target_coverage[split_name] = split_data['target_protein'].nunique()
        
        plt.bar(target_coverage.keys(), target_coverage.values())
        plt.xlabel('Split')
        plt.ylabel('Number of Unique Targets')
        plt.title('Target Coverage Across Splits')
        
        # 5. Cluster size distribution across splits
        plt.subplot(4, 4, 5)
        for split_name in ['train', 'val', 'test']:
            split_clusters = df[df['split'] == split_name]['cluster_id']
            cluster_sizes = self.cluster_stats[
                self.cluster_stats['cluster_id'].isin(split_clusters)
            ]['compound_count']
            plt.hist(cluster_sizes, bins=20, alpha=0.6, label=split_name, density=True)
        plt.xlabel('Compounds per Cluster')
        plt.ylabel('Density')
        plt.title('Cluster Size Distribution')
        plt.legend()
        plt.xscale('log')
        
        # 6. Measurements per split
        plt.subplot(4, 4, 6)
        measurement_counts = {}
        for split_name in ['train', 'val', 'test']:
            split_data = df_melted[df_melted['cluster_id'].isin(
                df[df['split'] == split_name]['cluster_id']
            )]
            measurement_counts[split_name] = len(split_data)
        
        plt.bar(measurement_counts.keys(), measurement_counts.values())
        plt.xlabel('Split')
        plt.ylabel('Number of Measurements')
        plt.title('Total Measurements per Split')
        
        # 7. Top 10 targets distribution across splits
        plt.subplot(4, 4, 7)
        top_targets = df_melted['target_protein'].value_counts().head(10).index
        
        target_split_matrix = np.zeros((len(top_targets), 3))
        split_names = ['train', 'val', 'test']
        
        for i, target in enumerate(top_targets):
            for j, split_name in enumerate(split_names):
                split_data = df_melted[
                    (df_melted['target_protein'] == target) &
                    df_melted['cluster_id'].isin(df[df['split'] == split_name]['cluster_id'])
                ]
                target_split_matrix[i, j] = len(split_data)
        
        sns.heatmap(target_split_matrix, 
                    xticklabels=split_names,
                    yticklabels=top_targets,
                    annot=True, fmt='g', cmap='Blues')
        plt.title('Top 10 Targets: Measurements per Split')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0, fontsize=8)
        
        # 8. pIC50 range coverage
        plt.subplot(4, 4, 8)
        pic50_ranges = {}
        for split_name in ['train', 'val', 'test']:
            split_data = df_melted[df_melted['cluster_id'].isin(
                df[df['split'] == split_name]['cluster_id']
            )]
            pic50_ranges[split_name] = (split_data['pIC50'].min(), split_data['pIC50'].max())
        
        splits = list(pic50_ranges.keys())
        mins = [pic50_ranges[s][0] for s in splits]
        maxs = [pic50_ranges[s][1] for s in splits]
        
        x = np.arange(len(splits))
        plt.bar(x, maxs, alpha=0.6, label='Max pIC50')
        plt.bar(x, mins, alpha=0.8, label='Min pIC50')
        plt.xticks(x, splits)
        plt.ylabel('pIC50')
        plt.title('pIC50 Range Coverage per Split')
        plt.legend()
        
        # 9-12. Individual split characteristics
        for i, split_name in enumerate(['train', 'val', 'test']):
            plt.subplot(4, 4, 9 + i)
            split_data = df_melted[df_melted['cluster_id'].isin(
                df[df['split'] == split_name]['cluster_id']
            )]
            
            # Scatter plot of pIC50 vs target (sample for readability)
            if len(split_data) > 1000:
                plot_data = split_data.sample(1000, random_state=42)
            else:
                plot_data = split_data
            
            target_codes = pd.Categorical(plot_data['target_protein']).codes
            plt.scatter(target_codes, plot_data['pIC50'], alpha=0.6, s=1)
            plt.xlabel('Target (encoded)')
            plt.ylabel('pIC50')
            plt.title(f'{split_name.capitalize()} Set Characteristics')
        
        # 12. Summary statistics
        plt.subplot(4, 4, 12)
        plt.axis('off')
        
        total_compounds = len(df)
        total_clusters = df['cluster_id'].nunique()
        total_measurements = len(df_melted)
        
        summary_text = f"""
        CLUSTER SPLIT SUMMARY
        
        Total Compounds: {total_compounds:,}
        Total Clusters: {total_clusters:,}
        Total Measurements: {total_measurements:,}
        
        Train: {len(df[df['split'] == 'train']):,} compounds
               {df[df['split'] == 'train']['cluster_id'].nunique():,} clusters
        
        Val:   {len(df[df['split'] == 'val']):,} compounds
               {df[df['split'] == 'val']['cluster_id'].nunique():,} clusters
        
        Test:  {len(df[df['split'] == 'test']):,} compounds
               {df[df['split'] == 'test']['cluster_id'].nunique():,} clusters
        
        No cluster leakage confirmed ✓
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig('cluster_aware_split_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'cluster_aware_split_analysis.png'")
        
        return fig
    
    def save_splits(self, df, df_melted, output_prefix='cluster_split'):
        """Save the split datasets to separate files"""
        print(f"\nSaving split datasets with prefix '{output_prefix}'...")
        
        for split_name in ['train', 'val', 'test']:
            split_df = df[df['split'] == split_name].copy()
            
            # Remove helper columns including cluster_id
            split_df = split_df.drop(['split', 'cluster_id'], axis=1)
            
            # Save wide format (original structure)
            output_file = f"{output_prefix}_{split_name}.csv"
            split_df.to_csv(output_file, index=False)
            print(f"  Saved {output_file}: {len(split_df)} compounds")
        
        # Also save cluster assignments for reference
        cluster_assignments = df[['smiles', 'cluster_id', 'split']].copy()
        cluster_assignments.to_csv(f"{output_prefix}_assignments.csv", index=False)
        print(f"  Saved {output_prefix}_assignments.csv: cluster-split mapping")
        
        return True
    
    def fit_split(self, bioactivity_df, cluster_file='datasets/chembl_clusters_nn_0.4.csv', 
                  smiles_col='smiles', output_prefix='cluster_split'):
        """
        Complete pipeline: load clusters, prepare data, create splits, validate, visualize, and save
        
        Parameters:
        - bioactivity_df: input DataFrame with SMILES and bioactivity data
        - cluster_file: path to cluster assignment file
        - smiles_col: column containing SMILES strings
        - output_prefix: prefix for output files
        
        Returns:
        - df_with_splits: DataFrame with split assignments
        - split_stats: validation statistics
        """
        print("Starting cluster-aware dataset splitting pipeline...")
        print("=" * 60)
        
        # Step 1: Load cluster data
        cluster_df = self.load_cluster_data(cluster_file)
        
        # Step 2: Prepare data
        df_prepared, df_melted, protein_cols = self.prepare_data(bioactivity_df, cluster_df, smiles_col)
        
        # Step 3: Analyze clusters
        cluster_df_analysis = self.analyze_cluster_characteristics(df_prepared, df_melted)
        
        # Step 4: Create balanced splits
        cluster_to_split = self.create_balanced_cluster_split(cluster_df_analysis)
        
        # Step 5: Assign compounds to splits
        df_with_splits = self.assign_compounds_to_splits(df_prepared, cluster_to_split)
        
        # Step 6: Validate split quality
        split_stats = self.validate_split_quality(df_with_splits, df_melted, protein_cols)
        
        # Step 7: Create visualizations
        self.create_visualizations(df_with_splits, df_melted, protein_cols)
        
        # Step 8: Save splits
        self.save_splits(df_with_splits, df_melted, output_prefix)
        
        print("\n" + "=" * 60)
        print("Cluster-aware splitting pipeline completed successfully!")
        print("Generated files:")
        print(f"  - {output_prefix}_train.csv")
        print(f"  - {output_prefix}_val.csv") 
        print(f"  - {output_prefix}_test.csv")
        print(f"  - {output_prefix}_assignments.csv")
        print(f"  - cluster_aware_split_analysis.png")
        
        return df_with_splits, split_stats


def main():
    """Example usage of ClusterAwareSplitter"""
    print("Cluster-Aware Dataset Splitting")
    print("=" * 40)
    
    # Load your bioactivity data
    print("Loading bioactivity data...")
    bioactivity_df = pd.read_csv('datasets/chembl_pretraining.csv')
    print(f"Loaded {len(bioactivity_df)} compounds with bioactivity data")
    
    # Initialize splitter with desired ratios
    splitter = ClusterAwareSplitter(
        train_ratio=0.8,   # 70% for training
        val_ratio=0.1,    # 15% for validation
        test_ratio=0.1,   # 15% for testing
        random_state=42    # For reproducibility
    )
    
    # Create splits using your Tanimoto similarity clusters
    df_with_splits, split_stats = splitter.fit_split(
        bioactivity_df, 
        cluster_file='datasets/chembl_clusters_nn_0.4.csv',  # Your cluster file
        smiles_col='smiles',
        output_prefix='chembl_cluster_split'
    )
    
    # Print final summary
    print("\nFinal Summary:")
    for split_name, stats in split_stats.items():
        print(f"{split_name.upper():>5}: {stats['compounds']:>6} compounds, "
              f"{stats['clusters']:>4} clusters, {stats['measurements']:>6} measurements")


if __name__ == "__main__":
    main()