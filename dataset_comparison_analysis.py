#!/usr/bin/env python3
"""
Comprehensive Dataset Comparison: ChEMBL Pretraining vs PKIS2 Finetuning

This script performs a detailed comparison between the ChEMBL pretraining dataset
and the PKIS2 finetuning dataset using:
1. ECFP6 (Morgan) fingerprints with radius=3
2. Bemis-Murcko scaffolds
3. Chemical space analysis and overlap assessment
4. Diversity metrics and visualization

Author: Dataset Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Scaffolds, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')
# Specifically suppress RDKit deprecation warnings
import os
os.environ['RDKIT_SILENCE_WARNINGS'] = '1'

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class DatasetComparator:
    """
    Comprehensive dataset comparison using ECFP6 fingerprints and Bemis-Murcko scaffolds
    """
    
    def __init__(self, random_state=42):
        """Initialize the comparator"""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Storage for datasets and analysis results
        self.datasets = {}
        self.fingerprints = {}
        self.scaffolds = {}
        self.analysis_results = {}
    
    def load_datasets(self, chembl_path='datasets/chembl_pretraining.csv', 
                     pkis2_path='datasets/pkis2_finetuning.csv'):
        """Load both datasets"""
        print("Loading datasets...")
        
        # Load ChEMBL pretraining dataset
        chembl_df = pd.read_csv(chembl_path)
        print(f"ChEMBL pretraining: {len(chembl_df)} compounds")
        
        # Load PKIS2 finetuning dataset (note: different SMILES column name)
        pkis2_df = pd.read_csv(pkis2_path)
        # Standardize column name to 'smiles'
        if 'Smiles' in pkis2_df.columns:
            pkis2_df = pkis2_df.rename(columns={'Smiles': 'smiles'})
        print(f"PKIS2 finetuning: {len(pkis2_df)} compounds")
        
        # Store datasets
        self.datasets['chembl'] = chembl_df
        self.datasets['pkis2'] = pkis2_df
        
        print(f"Total unique compounds across both datasets: {len(set(chembl_df['smiles'].tolist() + pkis2_df['smiles'].tolist()))}")
        
        return chembl_df, pkis2_df
    
    def validate_smiles(self, dataset_name, smiles_list):
        """Validate and clean SMILES strings"""
        print(f"Validating SMILES for {dataset_name}...")
        
        valid_smiles = []
        invalid_count = 0
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Canonicalize SMILES
                    canonical_smiles = Chem.MolToSmiles(mol)
                    valid_smiles.append(canonical_smiles)
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
        
        print(f"  Valid SMILES: {len(valid_smiles)}")
        print(f"  Invalid SMILES: {invalid_count}")
        
        return valid_smiles
    
    def calculate_ecfp6_fingerprints(self, dataset_name, smiles_list):
        """Calculate ECFP6 (Morgan) fingerprints with radius=3"""
        print(f"Calculating ECFP6 fingerprints for {dataset_name}...")
        
        fingerprints = []
        valid_smiles = []
        
        # Suppress deprecation warnings temporarily
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for smiles in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # ECFP6 = Morgan fingerprint with radius=3
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                        fingerprints.append(fp)
                        valid_smiles.append(smiles)
                except Exception as e:
                    print(f"Error processing {smiles}: {e}")
                    continue
        
        print(f"  Generated {len(fingerprints)} fingerprints")
        
        self.fingerprints[dataset_name] = fingerprints
        return fingerprints, valid_smiles
    
    def calculate_bemis_murcko_scaffolds(self, dataset_name, smiles_list):
        """Calculate Bemis-Murcko scaffolds"""
        print(f"Calculating Bemis-Murcko scaffolds for {dataset_name}...")
        
        scaffolds = []
        scaffold_smiles = []
        valid_smiles = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Generate Bemis-Murcko scaffold
                    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                    if scaffold_mol is not None:
                        scaffold_smiles_str = Chem.MolToSmiles(scaffold_mol)
                        scaffolds.append(scaffold_mol)
                        scaffold_smiles.append(scaffold_smiles_str)
                        valid_smiles.append(smiles)
                    else:
                        # If no scaffold, use original molecule
                        scaffolds.append(mol)
                        scaffold_smiles.append(smiles)
                        valid_smiles.append(smiles)
            except Exception as e:
                print(f"Error processing scaffold for {smiles}: {e}")
                continue
        
        print(f"  Generated {len(scaffolds)} scaffolds")
        print(f"  Unique scaffolds: {len(set(scaffold_smiles))}")
        
        self.scaffolds[dataset_name] = {
            'molecules': scaffolds,
            'smiles': scaffold_smiles,
            'original_smiles': valid_smiles
        }
        
        return scaffolds, scaffold_smiles, valid_smiles
    
    def calculate_fingerprint_similarities(self):
        """Calculate inter- and intra-dataset fingerprint similarities"""
        print("Calculating fingerprint similarities...")
        
        results = {}
        
        for dataset_name, fps in self.fingerprints.items():
            print(f"  Processing {dataset_name} ({len(fps)} compounds)...")
            
            # Calculate intra-dataset similarities (sample if too large)
            if len(fps) > 1000:
                sample_indices = np.random.choice(len(fps), 1000, replace=False)
                fps_sample = [fps[i] for i in sample_indices]
            else:
                fps_sample = fps
            
            similarities = []
            for i in range(len(fps_sample)):
                for j in range(i+1, len(fps_sample)):
                    sim = TanimotoSimilarity(fps_sample[i], fps_sample[j])
                    similarities.append(sim)
            
            results[f'{dataset_name}_intra'] = similarities
        
        # Calculate inter-dataset similarities
        if 'chembl' in self.fingerprints and 'pkis2' in self.fingerprints:
            print("  Calculating inter-dataset similarities...")
            
            chembl_fps = self.fingerprints['chembl']
            pkis2_fps = self.fingerprints['pkis2']
            
            # Sample if datasets are large
            if len(chembl_fps) > 500:
                chembl_sample = np.random.choice(len(chembl_fps), 500, replace=False)
                chembl_fps_sample = [chembl_fps[i] for i in chembl_sample]
            else:
                chembl_fps_sample = chembl_fps
            
            if len(pkis2_fps) > 500:
                pkis2_sample = np.random.choice(len(pkis2_fps), 500, replace=False)
                pkis2_fps_sample = [pkis2_fps[i] for i in pkis2_sample]
            else:
                pkis2_fps_sample = pkis2_fps
            
            inter_similarities = []
            for chembl_fp in chembl_fps_sample:
                for pkis2_fp in pkis2_fps_sample:
                    sim = TanimotoSimilarity(chembl_fp, pkis2_fp)
                    inter_similarities.append(sim)
            
            results['inter_dataset'] = inter_similarities
        
        self.analysis_results['fingerprint_similarities'] = results
        return results
    
    def analyze_scaffold_overlap(self):
        """Analyze scaffold overlap between datasets"""
        print("Analyzing scaffold overlap...")
        
        if 'chembl' not in self.scaffolds or 'pkis2' not in self.scaffolds:
            print("Scaffold data not available for both datasets")
            return {}
        
        chembl_scaffolds = set(self.scaffolds['chembl']['smiles'])
        pkis2_scaffolds = set(self.scaffolds['pkis2']['smiles'])
        
        # Calculate overlap statistics
        overlap = chembl_scaffolds & pkis2_scaffolds
        chembl_unique = chembl_scaffolds - pkis2_scaffolds
        pkis2_unique = pkis2_scaffolds - chembl_scaffolds
        
        results = {
            'chembl_total_scaffolds': len(chembl_scaffolds),
            'pkis2_total_scaffolds': len(pkis2_scaffolds),
            'overlap_scaffolds': len(overlap),
            'chembl_unique_scaffolds': len(chembl_unique),
            'pkis2_unique_scaffolds': len(pkis2_unique),
            'overlap_percentage_chembl': len(overlap) / len(chembl_scaffolds) * 100,
            'overlap_percentage_pkis2': len(overlap) / len(pkis2_scaffolds) * 100,
            'jaccard_index': len(overlap) / len(chembl_scaffolds | pkis2_scaffolds),
            'overlap_scaffold_smiles': list(overlap),
            'chembl_unique_scaffold_smiles': list(chembl_unique),
            'pkis2_unique_scaffold_smiles': list(pkis2_unique)
        }
        
        print(f"  ChEMBL scaffolds: {results['chembl_total_scaffolds']}")
        print(f"  PKIS2 scaffolds: {results['pkis2_total_scaffolds']}")
        print(f"  Overlapping scaffolds: {results['overlap_scaffolds']}")
        print(f"  Jaccard index: {results['jaccard_index']:.4f}")
        
        self.analysis_results['scaffold_overlap'] = results
        return results
    
    def calculate_diversity_metrics(self):
        """Calculate diversity metrics for both datasets"""
        print("Calculating diversity metrics...")
        
        results = {}
        
        for dataset_name, fps in self.fingerprints.items():
            print(f"  Processing {dataset_name}...")
            
            # Sample if dataset is too large
            if len(fps) > 1000:
                sample_indices = np.random.choice(len(fps), 1000, replace=False)
                fps_sample = [fps[i] for i in sample_indices]
            else:
                fps_sample = fps
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(fps_sample)):
                for j in range(i+1, len(fps_sample)):
                    sim = TanimotoSimilarity(fps_sample[i], fps_sample[j])
                    similarities.append(sim)
            
            # Diversity metrics
            mean_similarity = np.mean(similarities)
            diversity_index = 1 - mean_similarity  # Simple diversity index
            
            # Scaffold diversity
            if dataset_name in self.scaffolds:
                scaffold_smiles = self.scaffolds[dataset_name]['smiles']
                unique_scaffolds = len(set(scaffold_smiles))
                scaffold_diversity = unique_scaffolds / len(scaffold_smiles)
            else:
                unique_scaffolds = 0
                scaffold_diversity = 0
            
            results[dataset_name] = {
                'mean_tanimoto_similarity': mean_similarity,
                'diversity_index': diversity_index,
                'unique_scaffolds': unique_scaffolds,
                'scaffold_diversity': scaffold_diversity,
                'total_compounds': len(fps)
            }
        
        self.analysis_results['diversity_metrics'] = results
        return results
    
    def create_chemical_space_visualization(self):
        """Create chemical space visualization using PCA and t-SNE"""
        print("Creating chemical space visualizations...")
        
        if len(self.fingerprints) != 2:
            print("Need exactly 2 datasets for comparison visualization")
            return None
        
        # Define color mapping: ChEMBL = blue, PKIS2 = red
        color_map = {'chembl': 'blue', 'pkis2': 'red'}
        
        # Combine fingerprints from both datasets
        all_fps = []
        labels = []
        
        for dataset_name, fps in self.fingerprints.items():
            # Sample if dataset is too large
            if len(fps) > 1000:
                sample_indices = np.random.choice(len(fps), 1000, replace=False)
                fps_sample = [fps[i] for i in sample_indices]
            else:
                fps_sample = fps
            
            # Convert fingerprints to numpy arrays
            fp_arrays = [np.array(fp) for fp in fps_sample]
            all_fps.extend(fp_arrays)
            labels.extend([dataset_name] * len(fp_arrays))
        
        # Convert to numpy array
        X = np.array(all_fps)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # PCA visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        # Plot PCA
        for dataset_name in self.fingerprints.keys():
            mask = np.array(labels) == dataset_name
            color = color_map.get(dataset_name, 'gray')
            axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             label=dataset_name.upper(), alpha=0.6, s=20, c=color)
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 0].set_title('PCA of ECFP6 Fingerprints')
        axes[0, 0].legend()
        
        # t-SNE visualization
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        
        for dataset_name in self.fingerprints.keys():
            mask = np.array(labels) == dataset_name
            color = color_map.get(dataset_name, 'gray')
            axes[0, 1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                             label=dataset_name.upper(), alpha=0.6, s=20, c=color)
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        axes[0, 1].set_title('t-SNE of ECFP6 Fingerprints')
        axes[0, 1].legend()
        
        # Similarity distributions
        if 'fingerprint_similarities' in self.analysis_results:
            similarities = self.analysis_results['fingerprint_similarities']
            
            # Intra-dataset similarities
            for dataset_name, fps in self.fingerprints.items():
                if f'{dataset_name}_intra' in similarities:
                    color = color_map.get(dataset_name, 'gray')
                    axes[1, 0].hist(similarities[f'{dataset_name}_intra'], 
                                   bins=50, alpha=0.6, label=f'{dataset_name.upper()} intra-dataset',
                                   density=True, color=color)
            
            # Inter-dataset similarities
            if 'inter_dataset' in similarities:
                axes[1, 0].hist(similarities['inter_dataset'], 
                               bins=50, alpha=0.6, label='Inter-dataset',
                               density=True, color='purple')
            
            axes[1, 0].set_xlabel('Tanimoto Similarity')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Fingerprint Similarity Distributions')
            axes[1, 0].legend()
        
        # Scaffold overlap visualization
        if 'scaffold_overlap' in self.analysis_results:
            overlap_data = self.analysis_results['scaffold_overlap']
            
            categories = ['ChEMBL\nUnique', 'Shared', 'PKIS2\nUnique']
            values = [overlap_data['chembl_unique_scaffolds'], 
                     overlap_data['overlap_scaffolds'],
                     overlap_data['pkis2_unique_scaffolds']]
            colors = ['blue', 'purple', 'red']
            
            bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel('Number of Scaffolds')
            axes[1, 1].set_title('Scaffold Overlap Analysis')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('dataset_comparison_chemical_space.png', dpi=300, bbox_inches='tight')
        print("  Saved visualization as 'dataset_comparison_chemical_space.png'")
        
        return fig
    
    def create_comprehensive_analysis_plots(self):
        """Create comprehensive analysis plots"""
        print("Creating comprehensive analysis plots...")
        
        # Define color mapping: ChEMBL = blue, PKIS2 = red
        color_map = {'chembl': 'blue', 'pkis2': 'red'}
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        # 1. Dataset size comparison
        if self.datasets:
            dataset_names = list(self.datasets.keys())
            dataset_sizes = [len(df) for df in self.datasets.values()]
            colors = [color_map.get(name, 'gray') for name in dataset_names]
            
            bars = axes[0, 0].bar([name.upper() for name in dataset_names], dataset_sizes, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Number of Compounds')
            axes[0, 0].set_title('Dataset Size Comparison')
            
            # Add value labels
            for bar, size in zip(bars, dataset_sizes):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{size:,}', ha='center', va='bottom')
        
        # 2. Scaffold diversity comparison
        if 'diversity_metrics' in self.analysis_results:
            diversity_data = self.analysis_results['diversity_metrics']
            
            datasets = list(diversity_data.keys())
            scaffold_diversities = [diversity_data[d]['scaffold_diversity'] for d in datasets]
            colors = [color_map.get(name, 'gray') for name in datasets]
            
            bars = axes[0, 1].bar([name.upper() for name in datasets], scaffold_diversities, color=colors, alpha=0.7)
            axes[0, 1].set_ylabel('Scaffold Diversity (Unique/Total)')
            axes[0, 1].set_title('Scaffold Diversity Comparison')
            axes[0, 1].set_ylim(0, 1)
            
            # Add value labels
            for bar, diversity in zip(bars, scaffold_diversities):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{diversity:.3f}', ha='center', va='bottom')
        
        # 3. Mean Tanimoto similarity comparison
        if 'diversity_metrics' in self.analysis_results:
            diversity_data = self.analysis_results['diversity_metrics']
            
            datasets = list(diversity_data.keys())
            mean_similarities = [diversity_data[d]['mean_tanimoto_similarity'] for d in datasets]
            colors = [color_map.get(name, 'gray') for name in datasets]
            
            bars = axes[0, 2].bar([name.upper() for name in datasets], mean_similarities, color=colors, alpha=0.7)
            axes[0, 2].set_ylabel('Mean Tanimoto Similarity')
            axes[0, 2].set_title('Intra-Dataset Similarity Comparison')
            
            # Add value labels
            for bar, similarity in zip(bars, mean_similarities):
                height = bar.get_height()
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{similarity:.3f}', ha='center', va='bottom')
        
        # 4. Scaffold overlap pie chart
        if 'scaffold_overlap' in self.analysis_results:
            overlap_data = self.analysis_results['scaffold_overlap']
            
            sizes = [overlap_data['chembl_unique_scaffolds'], 
                    overlap_data['overlap_scaffolds'],
                    overlap_data['pkis2_unique_scaffolds']]
            labels = ['ChEMBL Unique', 'Shared', 'PKIS2 Unique']
            colors = ['blue', 'purple', 'red']
            
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Scaffold Distribution')
        
        # 5. Unique scaffold counts
        if 'diversity_metrics' in self.analysis_results:
            diversity_data = self.analysis_results['diversity_metrics']
            
            datasets = list(diversity_data.keys())
            unique_scaffolds = [diversity_data[d]['unique_scaffolds'] for d in datasets]
            colors = [color_map.get(name, 'gray') for name in datasets]
            
            bars = axes[1, 1].bar([name.upper() for name in datasets], unique_scaffolds, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel('Number of Unique Scaffolds')
            axes[1, 1].set_title('Unique Scaffold Count Comparison')
            
            # Add value labels
            for bar, count in zip(bars, unique_scaffolds):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{count:,}', ha='center', va='bottom')
        
        # 6. Similarity distribution comparison
        if 'fingerprint_similarities' in self.analysis_results:
            similarities = self.analysis_results['fingerprint_similarities']
            
            for dataset_name in self.fingerprints.keys():
                if f'{dataset_name}_intra' in similarities:
                    color = color_map.get(dataset_name, 'gray')
                    axes[1, 2].hist(similarities[f'{dataset_name}_intra'], 
                                   bins=30, alpha=0.6, label=f'{dataset_name.upper()}',
                                   density=True, color=color)
            
            axes[1, 2].set_xlabel('Tanimoto Similarity')
            axes[1, 2].set_ylabel('Density')
            axes[1, 2].set_title('Intra-Dataset Similarity Distributions')
            axes[1, 2].legend()
        
        # 7. Inter-dataset similarity distribution
        if 'fingerprint_similarities' in self.analysis_results:
            similarities = self.analysis_results['fingerprint_similarities']
            
            if 'inter_dataset' in similarities:
                axes[2, 0].hist(similarities['inter_dataset'], bins=50, alpha=0.7, 
                               color='purple', density=True)
                axes[2, 0].axvline(np.mean(similarities['inter_dataset']), 
                                  color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(similarities["inter_dataset"]):.3f}')
                axes[2, 0].set_xlabel('Tanimoto Similarity')
                axes[2, 0].set_ylabel('Density')
                axes[2, 0].set_title('Inter-Dataset Similarity Distribution')
                axes[2, 0].legend()
        
        # 8. Jaccard index visualization
        if 'scaffold_overlap' in self.analysis_results:
            overlap_data = self.analysis_results['scaffold_overlap']
            jaccard = overlap_data['jaccard_index']
            
            # Create a simple visualization of Jaccard index
            axes[2, 1].bar(['Jaccard Index'], [jaccard], color='purple', alpha=0.7)
            axes[2, 1].set_ylabel('Jaccard Index')
            axes[2, 1].set_title('Scaffold Overlap Jaccard Index')
            axes[2, 1].set_ylim(0, 1)
            axes[2, 1].text(0, jaccard + 0.01, f'{jaccard:.4f}', ha='center', va='bottom')
        
        # 9. Summary statistics
        axes[2, 2].axis('off')
        
        # Create summary text
        summary_text = "DATASET COMPARISON SUMMARY\n\n"
        
        if self.datasets:
            for name, df in self.datasets.items():
                summary_text += f"{name.upper()}:\n"
                summary_text += f"  Compounds: {len(df):,}\n"
                
                if 'diversity_metrics' in self.analysis_results and name in self.analysis_results['diversity_metrics']:
                    metrics = self.analysis_results['diversity_metrics'][name]
                    summary_text += f"  Unique Scaffolds: {metrics['unique_scaffolds']:,}\n"
                    summary_text += f"  Scaffold Diversity: {metrics['scaffold_diversity']:.3f}\n"
                    summary_text += f"  Mean Similarity: {metrics['mean_tanimoto_similarity']:.3f}\n"
                summary_text += "\n"
        
        if 'scaffold_overlap' in self.analysis_results:
            overlap = self.analysis_results['scaffold_overlap']
            summary_text += f"OVERLAP ANALYSIS:\n"
            summary_text += f"  Shared Scaffolds: {overlap['overlap_scaffolds']:,}\n"
            summary_text += f"  Jaccard Index: {overlap['jaccard_index']:.4f}\n"
            summary_text += f"  ChEMBL Coverage: {overlap['overlap_percentage_chembl']:.1f}%\n"
            summary_text += f"  PKIS2 Coverage: {overlap['overlap_percentage_pkis2']:.1f}%\n"
        
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig('dataset_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
        print("  Saved comprehensive analysis as 'dataset_comparison_comprehensive.png'")
        
        return fig
    
    def generate_detailed_report(self):
        """Generate a detailed text report"""
        print("Generating detailed report...")
        
        report_filename = 'dataset_comparison_report.txt'
        
        with open(report_filename, 'w') as f:
            f.write("COMPREHENSIVE DATASET COMPARISON REPORT\n")
            f.write("ChEMBL Pretraining vs PKIS2 Finetuning\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            
            for name, df in self.datasets.items():
                f.write(f"{name.upper()} Dataset:\n")
                f.write(f"  Total compounds: {len(df):,}\n")
                f.write(f"  Columns: {df.shape[1]}\n")
                
                # Target analysis
                if name == 'chembl':
                    protein_cols = [col for col in df.columns if col != 'smiles']
                    non_null_counts = df[protein_cols].count()
                    f.write(f"  Protein targets: {len(protein_cols)}\n")
                    f.write(f"  Most active targets: {', '.join(non_null_counts.nlargest(5).index.tolist())}\n")
                elif name == 'pkis2':
                    protein_cols = [col for col in df.columns if col != 'smiles']
                    f.write(f"  Protein targets: {len(protein_cols)}\n")
                
                f.write("\n")
            
            # Fingerprint analysis
            if 'diversity_metrics' in self.analysis_results:
                f.write("MOLECULAR FINGERPRINT ANALYSIS (ECFP6)\n")
                f.write("-" * 40 + "\n")
                
                for dataset_name, metrics in self.analysis_results['diversity_metrics'].items():
                    f.write(f"{dataset_name.upper()}:\n")
                    f.write(f"  Mean Tanimoto similarity: {metrics['mean_tanimoto_similarity']:.4f}\n")
                    f.write(f"  Diversity index: {metrics['diversity_index']:.4f}\n")
                    f.write(f"  (Higher diversity = more chemically diverse dataset)\n\n")
            
            # Scaffold analysis
            if 'scaffold_overlap' in self.analysis_results:
                f.write("BEMIS-MURCKO SCAFFOLD ANALYSIS\n")
                f.write("-" * 32 + "\n")
                
                overlap = self.analysis_results['scaffold_overlap']
                f.write(f"ChEMBL scaffolds: {overlap['chembl_total_scaffolds']:,}\n")
                f.write(f"PKIS2 scaffolds: {overlap['pkis2_total_scaffolds']:,}\n")
                f.write(f"Shared scaffolds: {overlap['overlap_scaffolds']:,}\n")
                f.write(f"ChEMBL unique: {overlap['chembl_unique_scaffolds']:,}\n")
                f.write(f"PKIS2 unique: {overlap['pkis2_unique_scaffolds']:,}\n\n")
                
                f.write(f"Overlap percentages:\n")
                f.write(f"  ChEMBL coverage: {overlap['overlap_percentage_chembl']:.2f}%\n")
                f.write(f"  PKIS2 coverage: {overlap['overlap_percentage_pkis2']:.2f}%\n")
                f.write(f"  Jaccard index: {overlap['jaccard_index']:.4f}\n\n")
            
            # Similarity analysis
            if 'fingerprint_similarities' in self.analysis_results:
                f.write("CHEMICAL SIMILARITY ANALYSIS\n")
                f.write("-" * 28 + "\n")
                
                similarities = self.analysis_results['fingerprint_similarities']
                
                for dataset_name in self.fingerprints.keys():
                    if f'{dataset_name}_intra' in similarities:
                        intra_sims = similarities[f'{dataset_name}_intra']
                        f.write(f"{dataset_name.upper()} intra-dataset similarities:\n")
                        f.write(f"  Mean: {np.mean(intra_sims):.4f}\n")
                        f.write(f"  Median: {np.median(intra_sims):.4f}\n")
                        f.write(f"  Std: {np.std(intra_sims):.4f}\n")
                        f.write(f"  Range: {np.min(intra_sims):.4f} - {np.max(intra_sims):.4f}\n\n")
                
                if 'inter_dataset' in similarities:
                    inter_sims = similarities['inter_dataset']
                    f.write(f"Inter-dataset similarities (ChEMBL vs PKIS2):\n")
                    f.write(f"  Mean: {np.mean(inter_sims):.4f}\n")
                    f.write(f"  Median: {np.median(inter_sims):.4f}\n")
                    f.write(f"  Std: {np.std(inter_sims):.4f}\n")
                    f.write(f"  Range: {np.min(inter_sims):.4f} - {np.max(inter_sims):.4f}\n\n")
            
            # Diversity comparison
            if 'diversity_metrics' in self.analysis_results:
                f.write("CHEMICAL DIVERSITY COMPARISON\n")
                f.write("-" * 30 + "\n")
                
                diversity_data = self.analysis_results['diversity_metrics']
                
                for dataset_name, metrics in diversity_data.items():
                    f.write(f"{dataset_name.upper()}:\n")
                    f.write(f"  Scaffold diversity: {metrics['scaffold_diversity']:.4f}\n")
                    f.write(f"  Unique scaffolds: {metrics['unique_scaffolds']:,}\n")
                    f.write(f"  Total compounds: {metrics['total_compounds']:,}\n")
                    f.write(f"  Scaffolds per compound: {metrics['unique_scaffolds']/metrics['total_compounds']:.4f}\n\n")
            
            # Conclusions
            f.write("CONCLUSIONS AND INSIGHTS\n")
            f.write("-" * 24 + "\n")
            
            if 'scaffold_overlap' in self.analysis_results:
                overlap = self.analysis_results['scaffold_overlap']
                jaccard = overlap['jaccard_index']
                
                if jaccard > 0.5:
                    f.write("• HIGH scaffold overlap - datasets share substantial chemical space\n")
                elif jaccard > 0.2:
                    f.write("• MODERATE scaffold overlap - datasets have some chemical space overlap\n")
                else:
                    f.write("• LOW scaffold overlap - datasets explore different chemical spaces\n")
            
            if 'diversity_metrics' in self.analysis_results:
                diversity_data = self.analysis_results['diversity_metrics']
                
                # Compare diversities
                if len(diversity_data) == 2:
                    datasets = list(diversity_data.keys())
                    div1 = diversity_data[datasets[0]]['diversity_index']
                    div2 = diversity_data[datasets[1]]['diversity_index']
                    
                    if abs(div1 - div2) > 0.05:
                        more_diverse = datasets[0] if div1 > div2 else datasets[1]
                        f.write(f"• {more_diverse.upper()} shows higher chemical diversity\n")
                    else:
                        f.write("• Both datasets show similar chemical diversity levels\n")
            
            f.write("\n")
            f.write("Generated files:\n")
            f.write("  - dataset_comparison_chemical_space.png\n")
            f.write("  - dataset_comparison_comprehensive.png\n")
            f.write("  - dataset_comparison_report.txt\n")
        
        print(f"Detailed report saved as '{report_filename}'")
    
    def run_complete_analysis(self):
        """Run the complete comparison analysis pipeline"""
        print("Starting comprehensive dataset comparison...")
        print("=" * 60)
        
        # Step 1: Load datasets
        chembl_df, pkis2_df = self.load_datasets()
        
        # Step 2: Process ChEMBL dataset
        chembl_smiles = self.validate_smiles('chembl', chembl_df['smiles'].tolist())
        chembl_fps, chembl_valid_smiles = self.calculate_ecfp6_fingerprints('chembl', chembl_smiles)
        chembl_scaffolds, chembl_scaffold_smiles, _ = self.calculate_bemis_murcko_scaffolds('chembl', chembl_valid_smiles)
        
        # Step 3: Process PKIS2 dataset
        pkis2_smiles = self.validate_smiles('pkis2', pkis2_df['smiles'].tolist())
        pkis2_fps, pkis2_valid_smiles = self.calculate_ecfp6_fingerprints('pkis2', pkis2_smiles)
        pkis2_scaffolds, pkis2_scaffold_smiles, _ = self.calculate_bemis_murcko_scaffolds('pkis2', pkis2_valid_smiles)
        
        # Step 4: Calculate similarities and overlaps
        self.calculate_fingerprint_similarities()
        self.analyze_scaffold_overlap()
        self.calculate_diversity_metrics()
        
        # Step 5: Create visualizations
        self.create_chemical_space_visualization()
        self.create_comprehensive_analysis_plots()
        
        # Step 6: Generate report
        self.generate_detailed_report()
        
        print("\n" + "=" * 60)
        print("Dataset comparison analysis completed successfully!")
        print("Generated files:")
        print("  - dataset_comparison_chemical_space.png")
        print("  - dataset_comparison_comprehensive.png") 
        print("  - dataset_comparison_report.txt")
        
        return self.analysis_results


def main():
    """Main analysis function"""
    print("Dataset Comparison Analysis: ChEMBL vs PKIS2")
    print("=" * 50)
    
    # Initialize comparator
    comparator = DatasetComparator(random_state=42)
    
    # Run complete analysis
    results = comparator.run_complete_analysis()
    
    # Print key findings
    print("\n" + "=" * 50)
    print("KEY FINDINGS:")
    
    if 'scaffold_overlap' in results:
        overlap = results['scaffold_overlap']
        print(f"• Scaffold Jaccard Index: {overlap['jaccard_index']:.4f}")
        print(f"• Shared Scaffolds: {overlap['overlap_scaffolds']:,}")
        
    if 'diversity_metrics' in results:
        diversity = results['diversity_metrics']
        print("• Chemical Diversity:")
        for dataset, metrics in diversity.items():
            print(f"  - {dataset}: {metrics['diversity_index']:.4f}")
            
    if 'fingerprint_similarities' in results:
        similarities = results['fingerprint_similarities']
        if 'inter_dataset' in similarities:
            inter_mean = np.mean(similarities['inter_dataset'])
            print(f"• Mean Inter-dataset Similarity: {inter_mean:.4f}")


if __name__ == "__main__":
    main()