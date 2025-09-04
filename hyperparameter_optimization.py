import optuna
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from experiment_runner import ExperimentRunner

def create_objective(strategy: str, dataset: str = "finetune"):
    """Create Optuna objective function for given strategy"""
    
    def objective(trial):
        config = {
            'strategy': strategy,
            'experiment_name': f"hpo_{strategy}_{trial.number}",
            
            # Model architecture
            'token_encoding': 'learnable',
            'embedding_dim': trial.suggest_categorical('embedding_dim', [64, 128, 256]),
            'n_layers': trial.suggest_int('n_layers', 2, 5),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'n_filters': trial.suggest_categorical('n_filters', [32, 64, 128, 256]),
            'dense_layer_size': trial.suggest_categorical('dense_layer_size', [128, 256, 512, 1024]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
            
            # Training
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'pretrain_epochs': trial.suggest_int('pretrain_epochs', 20, 100, step=10) if 'transfer' in strategy or strategy == 'pretrain_only' else 50,
            'finetune_epochs': trial.suggest_int('finetune_epochs', 10, 50, step=10) if 'transfer' in strategy or strategy == 'finetune_only' else 20,
            'finetune_lr_multiplier': trial.suggest_float('finetune_lr_multiplier', 0.01, 0.5) if 'transfer' in strategy else 0.1,
            
            # Augmentation
            'augmentation_factor': trial.suggest_categorical('augmentation_factor', [0, 5, 10]),
            
            # Fixed params
            'vocab_size': 53,
            'maxlen': 200,
            'vocab_path': None
        }
        
        try:
            print(f"\nStarting trial {trial.number} for {strategy}...")
            runner = ExperimentRunner(config)
            model, metrics = runner.run()
            # Convert numpy types to Python types
            metrics = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in metrics.items()}
            print(f"Trial {trial.number} completed - R2: {metrics['r2']:.4f}, Profile Pearson: {metrics.get('profile_pearson', 'N/A'):.4f}")
            
            # Store detailed metrics in trial
            trial.set_user_attr('r2', metrics['r2'])
            trial.set_user_attr('rmse', metrics['rmse'])
            trial.set_user_attr('mse', metrics['mse'])
            trial.set_user_attr('profile_pearson', metrics.get('profile_pearson', np.nan))
            trial.set_user_attr('profile_spearman', metrics.get('profile_spearman', np.nan))
            trial.set_user_attr('per_kinase_r2_mean', metrics.get('per_kinase_r2_mean', np.nan))
            trial.set_user_attr('per_kinase_r2_std', metrics.get('per_kinase_r2_std', np.nan))
            trial.set_user_attr('per_kinase_r2_dict', metrics.get('per_kinase_r2_dict', {}))
            trial.set_user_attr('per_kinase_rmse_dict', metrics.get('per_kinase_rmse_dict', {}))
            
            # Return negative R2 (we want to maximize R2, Optuna minimizes)
            return -metrics['r2']
            
        except (ValueError, RuntimeError, tf.errors.ResourceExhaustedError) as e:
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    return objective

def optimize_strategy(strategy: str, n_trials: int = 50):
    """Run HPO for a specific strategy"""
    
    study_name = f"{strategy}_optimization"
    db_path = Path(f"experiments/hpo_{strategy}.db")
    
    # Ensure experiments directory exists with proper permissions
    db_path.parent.mkdir(exist_ok=True)
    
    # Create database file with write permissions if it doesn't exist
    if not db_path.exists():
        db_path.touch()
        db_path.chmod(0o664)  # rw-rw-r--
    
    storage_path = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    objective = create_objective(strategy)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    # Get best trial metrics
    best_trial = study.best_trial
    best_metrics = {}
    if hasattr(best_trial, 'user_attrs'):
        best_metrics = best_trial.user_attrs
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': -study.best_value,  # Convert back to R2
        'best_metrics': best_metrics,
        'n_trials': len(study.trials),
        'strategy': strategy
    }
    
    with open(f"experiments/hpo_{strategy}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBest params for {strategy}:")
    print(json.dumps(study.best_params, indent=2))
    print(f"Best R2: {-study.best_value:.4f}")
    
    return study

def run_full_hpo():
    """Run HPO for all strategies"""
    strategies = ["pretrain_only", "finetune_only", "transfer_full", "transfer_frozen"]
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Optimizing: {strategy}")
        print(f"{'='*60}")
        
        study = optimize_strategy(strategy, n_trials=30)
        all_results[strategy] = {
            'best_params': study.best_params,
            'best_r2': -study.best_value
        }
    
    # Compare results
    print("\n\nHPO Results Comparison:")
    print("-" * 40)
    for strategy, results in all_results.items():
        print(f"{strategy}: R2 = {results['best_r2']:.4f}")
    
    with open("experiments/hpo_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def run_best_models():
    """Train final models with best hyperparameters from HPO"""
    
    # Load best params for each strategy
    strategies = ["pretrain_only", "finetune_only", "transfer_full", "transfer_frozen"]
    
    final_results = []
    
    for strategy in strategies:
        try:
            with open(f"experiments/hpo_{strategy}_results.json", 'r') as f:
                hpo_results = json.load(f)
            
            best_config = {
                'strategy': strategy,
                'experiment_name': f"{strategy}_best_final",
                'token_encoding': 'learnable',
                'vocab_size': 53,
                'maxlen': 200,
                'vocab_path': None
            }
            best_config.update(hpo_results['best_params'])
            
            print(f"\nTraining best model for {strategy}")
            runner = ExperimentRunner(best_config)
            model, metrics = runner.run()
            
            final_results.append({
                'strategy': strategy,
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'mse': metrics['mse'],
                'profile_pearson': metrics.get('profile_pearson', np.nan),
                'profile_spearman': metrics.get('profile_spearman', np.nan),
                'per_kinase_r2_mean': metrics.get('per_kinase_r2_mean', np.nan),
                'per_kinase_r2_std': metrics.get('per_kinase_r2_std', np.nan),
                'per_kinase_r2_dict': metrics.get('per_kinase_r2_dict', {}),
                'per_kinase_rmse_dict': metrics.get('per_kinase_rmse_dict', {})
            })
            
        except FileNotFoundError:
            print(f"No HPO results found for {strategy}, skipping...")
    
    return final_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "optimize":
            run_full_hpo()
        elif sys.argv[1] == "best":
            run_best_models()
        else:
            strategy = sys.argv[1]
            n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            optimize_strategy(strategy, n_trials)
    else:
        print("Usage:")
        print("  python hyperparameter_optimization.py optimize  # Run HPO for all strategies")
        print("  python hyperparameter_optimization.py best      # Train best models")
        print("  python hyperparameter_optimization.py <strategy> [n_trials]  # Optimize specific strategy")
