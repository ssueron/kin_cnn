#!/usr/bin/env python3
"""
Chemical Compound Cluster Analysis Script

Analyzes the distribution of chemical compound clusters relative to:
1. Protein target activity
2. pIC50 values
3. Tanimoto similarity patterns

Author: Generated analysis script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load cluster and bioactivity data"""
    print("Loading data...")
    
    # Load cluster assignments
    clusters_df = pd.read_csv('datasets/chembl_clusters_nn_0.4.csv')
    print(f"Loaded {len(clusters_df)} compounds with cluster assignments")
    
    # Load bioactivity data
    bioactivity_df = pd.read_csv('datasets/chembl_pretraining.csv')
    print(f"Loaded {len(bioactivity_df)} compounds with bioactivity data")
    
    return clusters_df, bioactivity_df

def merge_data(clusters_df, bioactivity_df):
    """Merge cluster and bioactivity data on SMILES"""
    print("Merging datasets...")
    
    merged_df = pd.merge(clusters_df, bioactivity_df, on='smiles', how='inner')
    print(f"Successfully merged data for {len(merged_df)} compounds")
    
    return merged_df

def analyze_cluster_distribution(merged_df):
    """Analyze basic cluster distribution statistics"""
    print("\n=== CLUSTER DISTRIBUTION ANALYSIS ===")
    
    cluster_counts = merged_df['cluster_id'].value_counts()
    
    print(f"Total number of clusters: {len(cluster_counts)}")
    print(f"Average compounds per cluster: {cluster_counts.mean():.2f}")
    print(f"Median compounds per cluster: {cluster_counts.median():.2f}")
    print(f"Largest cluster size: {cluster_counts.max()}")
    print(f"Smallest cluster size: {cluster_counts.min()}")
    
    # Show top 10 largest clusters
    print("\nTop 10 largest clusters:")
    for i, (cluster_id, count) in enumerate(cluster_counts.head(10).items()):
        print(f"  {i+1}. Cluster {cluster_id}: {count} compounds")
    
    return cluster_counts

def analyze_protein_activity_distribution(merged_df):
    """Analyze how clusters distribute across different proteins"""
    print("\n=== PROTEIN ACTIVITY DISTRIBUTION ANALYSIS ===")
    
    # Get protein columns (exclude smiles and cluster_id)
    protein_cols = [col for col in merged_df.columns if col not in ['smiles', 'cluster_id']]
    
    # Calculate activity counts per protein
    protein_activity_counts = {}
    for protein in protein_cols:
        # Count non-null values (compounds with activity data for this protein)
        activity_count = merged_df[protein].notna().sum()
        protein_activity_counts[protein] = activity_count
    
    # Sort by activity count
    sorted_proteins = sorted(protein_activity_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Total proteins analyzed: {len(protein_cols)}")
    print("\nTop 10 proteins by number of active compounds:")
    for i, (protein, count) in enumerate(sorted_proteins[:10]):
        print(f"  {i+1}. {protein}: {count} compounds")
    
    return protein_activity_counts, protein_cols

def analyze_cluster_protein_overlap(merged_df, protein_cols):
    """Analyze how clusters overlap with protein activities"""
    print("\n=== CLUSTER-PROTEIN OVERLAP ANALYSIS ===")
    
    # Create a summary table
    cluster_protein_summary = []
    
    for cluster_id in merged_df['cluster_id'].unique():
        cluster_data = merged_df[merged_df['cluster_id'] == cluster_id]
        cluster_size = len(cluster_data)
        
        # Count activity per protein in this cluster
        protein_activities = {}
        for protein in protein_cols:
            active_count = cluster_data[protein].notna().sum()
            protein_activities[protein] = active_count
        
        # Find most active proteins in this cluster
        most_active_proteins = sorted(protein_activities.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
        
        cluster_protein_summary.append({
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'total_activities': sum(protein_activities.values()),
            'top_proteins': most_active_proteins
        })
    
    # Sort by cluster size
    cluster_protein_summary.sort(key=lambda x: x['cluster_size'], reverse=True)
    
    print("Top 10 clusters by size and their protein activity patterns:")
    for i, cluster_info in enumerate(cluster_protein_summary[:10]):
        print(f"\nCluster {cluster_info['cluster_id']} ({cluster_info['cluster_size']} compounds):")
        print(f"  Total activities: {cluster_info['total_activities']}")
        print("  Top proteins:")
        for j, (protein, count) in enumerate(cluster_info['top_proteins']):
            if count > 0:
                print(f"    {j+1}. {protein}: {count} compounds")
    
    return cluster_protein_summary

def analyze_pic50_distribution(merged_df, protein_cols):
    """Analyze pIC50 value distributions across clusters"""
    print("\n=== pIC50 DISTRIBUTION ANALYSIS ===")
    
    # Calculate statistics for each cluster
    cluster_pic50_stats = []
    
    for cluster_id in merged_df['cluster_id'].unique():
        cluster_data = merged_df[merged_df['cluster_id'] == cluster_id]
        
        # Collect all pIC50 values for this cluster
        all_pic50_values = []
        for protein in protein_cols:
            values = cluster_data[protein].dropna().values
            all_pic50_values.extend(values)
        
        if len(all_pic50_values) > 0:
            cluster_pic50_stats.append({
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_data),
                'total_pic50_measurements': len(all_pic50_values),
                'mean_pic50': np.mean(all_pic50_values),
                'median_pic50': np.median(all_pic50_values),
                'std_pic50': np.std(all_pic50_values),
                'min_pic50': np.min(all_pic50_values),
                'max_pic50': np.max(all_pic50_values)
            })
    
    # Sort by number of measurements
    cluster_pic50_stats.sort(key=lambda x: x['total_pic50_measurements'], reverse=True)
    
    print("Top 10 clusters by number of pIC50 measurements:")
    for i, stats in enumerate(cluster_pic50_stats[:10]):
        print(f"\nCluster {stats['cluster_id']}:")
        print(f"  Size: {stats['cluster_size']} compounds")
        print(f"  pIC50 measurements: {stats['total_pic50_measurements']}")
        print(f"  Mean pIC50: {stats['mean_pic50']:.2f}")
        print(f"  Std pIC50: {stats['std_pic50']:.2f}")
        print(f"  Range: {stats['min_pic50']:.2f} - {stats['max_pic50']:.2f}")
    
    return cluster_pic50_stats

def create_visualizations(merged_df, cluster_counts, protein_activity_counts, 
                         cluster_pic50_stats, protein_cols):
    """Create comprehensive visualizations"""
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Set up the plotting area
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Cluster size distribution
    plt.subplot(3, 3, 1)
    plt.hist(cluster_counts.values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Cluster Size')
    plt.ylabel('Number of Clusters')
    plt.title('Distribution of Cluster Sizes')
    plt.yscale('log')
    
    # 2. Top 20 proteins by activity count
    plt.subplot(3, 3, 2)
    top_proteins = sorted(protein_activity_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    proteins, counts = zip(*top_proteins)
    plt.barh(range(len(proteins)), counts)
    plt.yticks(range(len(proteins)), proteins, fontsize=8)
    plt.xlabel('Number of Active Compounds')
    plt.title('Top 20 Proteins by Activity Count')
    
    # 3. pIC50 distribution across all data
    plt.subplot(3, 3, 3)
    all_pic50_values = []
    for protein in protein_cols:
        values = merged_df[protein].dropna().values
        all_pic50_values.extend(values)
    
    plt.hist(all_pic50_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('pIC50 Value')
    plt.ylabel('Frequency')
    plt.title('Overall pIC50 Distribution')
    
    # 4. Cluster size vs number of activities
    plt.subplot(3, 3, 4)
    cluster_sizes = []
    total_activities = []
    
    for cluster_id in merged_df['cluster_id'].unique():
        cluster_data = merged_df[merged_df['cluster_id'] == cluster_id]
        cluster_sizes.append(len(cluster_data))
        
        activity_count = 0
        for protein in protein_cols:
            activity_count += cluster_data[protein].notna().sum()
        total_activities.append(activity_count)
    
    plt.scatter(cluster_sizes, total_activities, alpha=0.6)
    plt.xlabel('Cluster Size')
    plt.ylabel('Total Activities in Cluster')
    plt.title('Cluster Size vs Total Activities')
    plt.xscale('log')
    plt.yscale('log')
    
    # 5. Mean pIC50 by cluster size (for clusters with stats)
    plt.subplot(3, 3, 5)
    if cluster_pic50_stats:
        cluster_sizes_with_stats = [stats['cluster_size'] for stats in cluster_pic50_stats]
        mean_pic50_values = [stats['mean_pic50'] for stats in cluster_pic50_stats]
        
        plt.scatter(cluster_sizes_with_stats, mean_pic50_values, alpha=0.6)
        plt.xlabel('Cluster Size')
        plt.ylabel('Mean pIC50')
        plt.title('Cluster Size vs Mean pIC50')
        plt.xscale('log')
    
    # 6. Top 10 largest clusters breakdown
    plt.subplot(3, 3, 6)
    top_10_clusters = cluster_counts.head(10)
    plt.bar(range(len(top_10_clusters)), top_10_clusters.values)
    plt.xlabel('Cluster Rank')
    plt.ylabel('Number of Compounds')
    plt.title('Top 10 Largest Clusters')
    
    # 7. Activity distribution heatmap (top clusters vs top proteins)
    plt.subplot(3, 3, 7)
    
    # Get top 10 clusters and top 10 proteins
    top_clusters = cluster_counts.head(10).index
    top_proteins_list = [p[0] for p in sorted(protein_activity_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]]
    
    # Create activity matrix
    activity_matrix = np.zeros((len(top_clusters), len(top_proteins_list)))
    
    for i, cluster_id in enumerate(top_clusters):
        cluster_data = merged_df[merged_df['cluster_id'] == cluster_id]
        for j, protein in enumerate(top_proteins_list):
            activity_matrix[i, j] = cluster_data[protein].notna().sum()
    
    sns.heatmap(activity_matrix, 
                xticklabels=top_proteins_list,
                yticklabels=[f'Cluster {c}' for c in top_clusters],
                annot=True, fmt='g', cmap='YlOrRd')
    plt.title('Activity Heatmap: Top Clusters vs Top Proteins')
    plt.xticks(rotation=45, ha='right')
    
    # 8. pIC50 variance within clusters
    plt.subplot(3, 3, 8)
    if cluster_pic50_stats:
        measurements = [stats['total_pic50_measurements'] for stats in cluster_pic50_stats 
                       if stats['total_pic50_measurements'] > 10]  # Only clusters with >10 measurements
        variances = [stats['std_pic50']**2 for stats in cluster_pic50_stats 
                    if stats['total_pic50_measurements'] > 10]
        
        if measurements and variances:
            plt.scatter(measurements, variances, alpha=0.6)
            plt.xlabel('Number of pIC50 Measurements')
            plt.ylabel('pIC50 Variance')
            plt.title('pIC50 Variance vs Measurement Count')
            plt.xscale('log')
    
    # 9. Summary statistics box
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Calculate summary statistics
    total_compounds = len(merged_df)
    total_clusters = len(cluster_counts)
    total_proteins = len([p for p, c in protein_activity_counts.items() if c > 0])
    total_activities = sum(protein_activity_counts.values())
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    Total Compounds: {total_compounds:,}
    Total Clusters: {total_clusters:,}
    Active Proteins: {total_proteins}
    Total Activities: {total_activities:,}
    
    Avg Cluster Size: {cluster_counts.mean():.1f}
    Max Cluster Size: {cluster_counts.max():,}
    
    Avg pIC50: {np.mean(all_pic50_values):.2f}
    pIC50 Range: {np.min(all_pic50_values):.2f} - {np.max(all_pic50_values):.2f}
    """
    
    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('cluster_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive visualization as 'cluster_analysis_comprehensive.png'")
    
    return fig

def generate_report(merged_df, cluster_counts, protein_activity_counts, 
                   cluster_protein_summary, cluster_pic50_stats):
    """Generate a comprehensive text report"""
    print("\n=== GENERATING COMPREHENSIVE REPORT ===")
    
    report_filename = 'cluster_analysis_report.txt'
    
    with open(report_filename, 'w') as f:
        f.write("CHEMICAL COMPOUND CLUSTER ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total compounds analyzed: {len(merged_df):,}\n")
        f.write(f"Total clusters identified: {len(cluster_counts):,}\n")
        f.write(f"Total proteins with activity data: {len(protein_activity_counts)}\n\n")
        
        # Cluster statistics
        f.write("CLUSTER STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average cluster size: {cluster_counts.mean():.2f}\n")
        f.write(f"Median cluster size: {cluster_counts.median():.2f}\n")
        f.write(f"Largest cluster: {cluster_counts.max()} compounds\n")
        f.write(f"Smallest cluster: {cluster_counts.min()} compounds\n")
        f.write(f"Standard deviation: {cluster_counts.std():.2f}\n\n")
        
        # Top clusters
        f.write("TOP 20 LARGEST CLUSTERS\n")
        f.write("-" * 25 + "\n")
        for i, (cluster_id, count) in enumerate(cluster_counts.head(20).items()):
            f.write(f"{i+1:2d}. Cluster {cluster_id:>6}: {count:>5} compounds\n")
        f.write("\n")
        
        # Protein activity overview
        f.write("PROTEIN ACTIVITY OVERVIEW\n")
        f.write("-" * 25 + "\n")
        sorted_proteins = sorted(protein_activity_counts.items(), key=lambda x: x[1], reverse=True)
        f.write("Top 20 proteins by number of active compounds:\n")
        for i, (protein, count) in enumerate(sorted_proteins[:20]):
            f.write(f"{i+1:2d}. {protein:<15}: {count:>5} compounds\n")
        f.write("\n")
        
        # pIC50 statistics
        if cluster_pic50_stats:
            f.write("pIC50 STATISTICS\n")
            f.write("-" * 16 + "\n")
            all_measurements = sum(stats['total_pic50_measurements'] for stats in cluster_pic50_stats)
            f.write(f"Total pIC50 measurements: {all_measurements:,}\n")
            
            # Overall pIC50 statistics
            all_pic50_values = []
            for stats in cluster_pic50_stats:
                # This is approximate since we don't have individual values
                pass
            
            f.write("\nTop 10 clusters by pIC50 measurement count:\n")
            for i, stats in enumerate(cluster_pic50_stats[:10]):
                f.write(f"{i+1:2d}. Cluster {stats['cluster_id']:>6}: "
                       f"{stats['total_pic50_measurements']:>4} measurements, "
                       f"mean pIC50: {stats['mean_pic50']:>6.2f}\n")
        f.write("\n")
        
        # Cluster-protein relationships
        f.write("CLUSTER-PROTEIN RELATIONSHIPS\n")
        f.write("-" * 30 + "\n")
        f.write("Analysis of how chemical similarity clusters relate to protein activity:\n\n")
        
        for i, cluster_info in enumerate(cluster_protein_summary[:10]):
            f.write(f"Cluster {cluster_info['cluster_id']} "
                   f"({cluster_info['cluster_size']} compounds):\n")
            f.write(f"  Total activities: {cluster_info['total_activities']}\n")
            f.write("  Most active proteins:\n")
            for j, (protein, count) in enumerate(cluster_info['top_proteins'][:3]):
                if count > 0:
                    percentage = (count / cluster_info['cluster_size']) * 100
                    f.write(f"    {protein}: {count} compounds ({percentage:.1f}%)\n")
            f.write("\n")
    
    print(f"Comprehensive report saved as '{report_filename}'")

def main():
    """Main analysis function"""
    print("Chemical Compound Cluster Analysis")
    print("=" * 40)
    
    # Load and merge data
    clusters_df, bioactivity_df = load_data()
    merged_df = merge_data(clusters_df, bioactivity_df)
    
    # Perform analyses
    cluster_counts = analyze_cluster_distribution(merged_df)
    protein_activity_counts, protein_cols = analyze_protein_activity_distribution(merged_df)
    cluster_protein_summary = analyze_cluster_protein_overlap(merged_df, protein_cols)
    cluster_pic50_stats = analyze_pic50_distribution(merged_df, protein_cols)
    
    # Create visualizations
    fig = create_visualizations(merged_df, cluster_counts, protein_activity_counts, 
                               cluster_pic50_stats, protein_cols)
    
    # Generate comprehensive report
    generate_report(merged_df, cluster_counts, protein_activity_counts, 
                   cluster_protein_summary, cluster_pic50_stats)
    
    print("\n" + "=" * 40)
    print("Analysis complete!")
    print("Generated files:")
    print("  - cluster_analysis_comprehensive.png")
    print("  - cluster_analysis_report.txt")
    
    plt.show()

if __name__ == "__main__":
    main()