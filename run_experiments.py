#!/usr/bin/env python
"""
Main script to run all kinase prediction experiments
"""

import json
from pathlib import Path
from experiment_runner import ExperimentRunner, run_all_experiments
from hyperparameter_optimization import run_full_hpo, run_best_models
from rf_baseline import train_rf_baseline, compare_all_models
import pandas as pd

# Default configuration
DEFAULT_CONFIG = {
    'token_encoding': 'learnable',
    'embedding_dim': 128,
    'n_layers': 3,
    'kernel_size': 5,
    'n_filters': 128,
    'dense_layer_size': 512,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'pretrain_epochs': 50,
    'finetune_epochs': 30,
    'finetune_lr_multiplier': 0.1,
    'augmentation_factor': 0,
    'vocab_size': 53,
    'maxlen': 200,
    'vocab_path': None
}

def quick_test(force_cpu=False):
    """Run quick test with small epochs to verify everything works"""
    test_config = DEFAULT_CONFIG.copy()
    test_config.update({
        'pretrain_epochs': 2,
        'finetune_epochs': 2,
        'experiment_name': 'quick_test',
        'force_cpu': force_cpu
    })
    
    print("Running quick test to verify setup...")
    
    # Test each strategy
    for strategy in ["pretrain_only", "finetune_only", "transfer_full", "transfer_frozen"]:
        print(f"\nTesting {strategy}...")
        test_config['strategy'] = strategy
        test_config['experiment_name'] = f'test_{strategy}'
        
        try:
            runner = ExperimentRunner(test_config)
            model, metrics = runner.run()
            print(f"✓ {strategy} works! R2: {metrics['r2']:.4f}")
        except Exception as e:
            print(f"✗ {strategy} failed: {e}")
    
    print("\nQuick test complete!")

def run_baseline_experiments():
    """Run all experiments with default hyperparameters"""
    print("="*60)
    print("RUNNING BASELINE EXPERIMENTS")
    print("="*60)
    
    results = run_all_experiments(DEFAULT_CONFIG)
    
    print("\nBaseline experiments complete!")
    return results

def run_hpo_experiments(n_trials=30):
    """Run hyperparameter optimization"""
    print("="*60)
    print("RUNNING HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Run HPO for all strategies
    hpo_results = run_full_hpo()
    
    # Train best models
    print("\n" + "="*60)
    print("TRAINING BEST MODELS FROM HPO")
    print("="*60)
    
    best_results = run_best_models()
    
    return hpo_results, best_results

def run_rf_baselines():
    """Train Random Forest baselines"""
    print("="*60)
    print("TRAINING RANDOM FOREST BASELINES")
    print("="*60)
    
    # Train on both datasets
    print("\n1. Training on pretraining data...")
    rf_pretrain = train_rf_baseline("pretrain")
    
    print("\n2. Training on finetuning data...")
    rf_finetune = train_rf_baseline("finetune")
    
    return rf_pretrain, rf_finetune

def generate_final_report():
    """Generate comprehensive comparison report"""
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)
    
    # Collect all results
    all_results = []
    
    # Load CNN baseline results
    baseline_path = Path("experiments/comparison_summary.csv")
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        for _, row in df.iterrows():
            all_results.append({
                'model_type': 'CNN',
                'strategy': row['strategy'],
                'augmentation': row['augmentation'],
                'optimized': False,
                'rmse': row['rmse'],
                'r2': row['r2']
            })
    
    # Load HPO results
    for strategy in ["pretrain_only", "finetune_only", "transfer_full", "transfer_frozen"]:
        hpo_path = Path(f"experiments/hpo_{strategy}_results.json")
        if hpo_path.exists():
            with open(hpo_path, 'r') as f:
                hpo = json.load(f)
            all_results.append({
                'model_type': 'CNN',
                'strategy': strategy,
                'augmentation': 'optimized',
                'optimized': True,
                'rmse': None,
                'r2': hpo['best_value']
            })
    
    # Load RF results
    for dataset in ["pretrain", "finetune"]:
        rf_path = Path(f"experiments/rf_baseline_{dataset}/metrics.json")
        if rf_path.exists():
            with open(rf_path, 'r') as f:
                metrics = json.load(f)
            all_results.append({
                'model_type': 'RF_ECFP6',
                'strategy': f'{dataset}_only',
                'augmentation': 0,
                'optimized': False,
                'rmse': metrics['test'].get('rmse'),
                'r2': metrics['test']['r2']
            })
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values('r2', ascending=False)
    
    # Save report
    df_summary.to_csv("experiments/final_report.csv", index=False)
    
    # Print summary
    print("\nTop 10 Models by R2:")
    print(df_summary.head(10).to_string(index=False))
    
    # Best model by category
    print("\n" + "="*40)
    print("BEST MODELS BY CATEGORY:")
    print("="*40)
    
    for strategy in df_summary['strategy'].unique():
        best = df_summary[df_summary['strategy'] == strategy].iloc[0]
        print(f"{strategy:20s}: R2 = {best['r2']:.4f} ({best['model_type']})")
    
    return df_summary

def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_experiments.py test [--cpu]   # Quick test")
        print("  python run_experiments.py baseline       # Run baseline experiments")
        print("  python run_experiments.py hpo [n]        # Run HPO with n trials (default 30)")
        print("  python run_experiments.py rf             # Train RF baselines")
        print("  python run_experiments.py all            # Run everything")
        print("  python run_experiments.py report         # Generate final report")
        print("  Add --cpu to any command to force CPU usage")
        return
    
    command = sys.argv[1]
    force_cpu = '--cpu' in sys.argv
    
    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)
    
    if command == "test":
        quick_test(force_cpu=force_cpu)
    
    elif command == "baseline":
        run_baseline_experiments()
    
    elif command == "hpo":
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        run_hpo_experiments(n_trials)
    
    elif command == "rf":
        run_rf_baselines()
    
    elif command == "all":
        print("Running complete experimental pipeline...")
        print("This will take several hours. Press Ctrl+C to cancel.\n")
        
        # 1. Baseline experiments
        run_baseline_experiments()
        
        # 2. HPO (reduced trials for complete run)
        run_hpo_experiments(n_trials=20)
        
        # 3. RF baselines
        run_rf_baselines()
        
        # 4. Final report
        generate_final_report()
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETE!")
        print("Check experiments/final_report.csv for results")
        print("="*60)
    
    elif command == "report":
        generate_final_report()
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()