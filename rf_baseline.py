import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import pickle
from pathlib import Path
import optuna

def generate_ecfp6(smiles_list, radius=3, n_bits=2048):
    """Generate ECFP6 fingerprints"""
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros(n_bits))
    
    return np.array(fingerprints)

def load_kinase_data(csv_path, mask_path=None):
    """Load kinase data and return SMILES, targets, and masks"""
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
    smiles = df[smiles_col].values
    
    # Get kinase columns
    kinase_cols = [col for col in df.columns if col != smiles_col]
    y = df[kinase_cols].values
    
    # Load mask if provided
    if mask_path and Path(mask_path).exists():
        mask_df = pd.read_csv(mask_path)
        mask = mask_df[kinase_cols].values.astype(bool)
    else:
        # No mask means all values are valid
        mask = ~np.isnan(y)
    
    return smiles, y, mask, kinase_cols

class ECFPRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 radius=3, n_bits=2048, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.radius = radius
        self.n_bits = n_bits
        self.n_jobs = n_jobs
        self.models = {}
        self.kinase_names = []
        
    def fit(self, X_smiles, y, mask=None):
        """Train RF models for each kinase"""
        # Generate fingerprints
        X = generate_ecfp6(X_smiles, self.radius, self.n_bits)
        
        n_samples, n_kinases = y.shape
        
        # Train separate model for each kinase
        for i in range(n_kinases):
            if mask is not None:
                valid_idx = mask[:, i]
            else:
                valid_idx = ~np.isnan(y[:, i])
            
            if np.sum(valid_idx) > 10:  # Need minimum samples
                X_train = X[valid_idx]
                y_train = y[valid_idx, i]
                
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    n_jobs=self.n_jobs,
                    random_state=42
                )
                rf.fit(X_train, y_train)
                self.models[i] = rf
            else:
                self.models[i] = None
    
    def predict(self, X_smiles):
        """Predict for all kinases"""
        X = generate_ecfp6(X_smiles, self.radius, self.n_bits)
        n_samples = len(X)
        n_kinases = len(self.models)
        
        predictions = np.full((n_samples, n_kinases), np.nan)
        
        for i, model in self.models.items():
            if model is not None:
                predictions[:, i] = model.predict(X)
        
        return predictions
    
    def evaluate(self, X_smiles, y_true, mask=None):
        """Evaluate model performance"""
        y_pred = self.predict(X_smiles)
        
        metrics = {'rmse': [], 'r2': [], 'mae': []}
        
        for i in range(y_true.shape[1]):
            # Only evaluate kinases that have trained models
            if i not in self.models or self.models[i] is None:
                continue
                
            if mask is not None:
                valid_idx = mask[:, i]
            else:
                valid_idx = ~np.isnan(y_true[:, i])
            
            # Additional check to ensure predictions are not NaN
            valid_idx = valid_idx & ~np.isnan(y_pred[:, i])
            
            if np.sum(valid_idx) > 0:
                y_true_i = y_true[valid_idx, i]
                y_pred_i = y_pred[valid_idx, i]
                
                # Double check for NaN values before computing metrics
                if not (np.any(np.isnan(y_true_i)) or np.any(np.isnan(y_pred_i))):
                    metrics['rmse'].append(np.sqrt(mean_squared_error(y_true_i, y_pred_i)))
                    metrics['r2'].append(r2_score(y_true_i, y_pred_i))
                    metrics['mae'].append(mean_absolute_error(y_true_i, y_pred_i))
        
        # Return NaN if no valid metrics computed
        if not metrics['rmse']:
            return {
                'rmse': np.nan,
                'r2': np.nan,
                'mae': np.nan,
                'rmse_std': np.nan,
                'r2_std': np.nan,
                'mae_std': np.nan
            }
        
        # Average metrics
        return {
            'rmse': np.mean(metrics['rmse']),
            'r2': np.mean(metrics['r2']),
            'mae': np.mean(metrics['mae']),
            'rmse_std': np.std(metrics['rmse']),
            'r2_std': np.std(metrics['r2']),
            'mae_std': np.std(metrics['mae'])
        }

def train_rf_baseline(dataset="pretrain"):
    """Train RF baseline on specified dataset"""
    
    # Paths
    if dataset == "pretrain":
        paths = {
            'train': 'datasets/chembl_pretraining_train.csv',
            'val': 'datasets/chembl_pretraining_val.csv',
            'test': 'datasets/chembl_pretraining_test.csv',
            'train_mask': 'datasets/chembl_pretraining_train_mask.csv',
            'val_mask': 'datasets/chembl_pretraining_val_mask.csv',
            'test_mask': 'datasets/chembl_pretraining_test_mask.csv'
        }
    else:
        paths = {
            'train': 'datasets/pkis2_finetuning_train.csv',
            'val': 'datasets/pkis2_finetuning_val.csv',
            'test': 'datasets/pkis2_finetuning_test.csv',
            'train_mask': 'datasets/pkis2_finetuning_train_mask.csv',
            'val_mask': 'datasets/pkis2_finetuning_val_mask.csv',
            'test_mask': 'datasets/pkis2_finetuning_test_mask.csv'
        }
    
    # Load data
    print(f"Loading {dataset} data...")
    X_train, y_train, mask_train, kinase_names = load_kinase_data(paths['train'], paths['train_mask'])
    X_val, y_val, mask_val, _ = load_kinase_data(paths['val'], paths['val_mask'])
    X_test, y_test, mask_test, _ = load_kinase_data(paths['test'], paths['test_mask'])
    
    # Train model
    print("Training Random Forest...")
    model = ECFPRandomForest(n_estimators=100, max_depth=20, n_jobs=-1)
    model.kinase_names = kinase_names
    model.fit(X_train, y_train, mask_train)
    
    # Evaluate
    print("Evaluating...")
    val_metrics = model.evaluate(X_val, y_val, mask_val)
    test_metrics = model.evaluate(X_test, y_test, mask_test)
    
    # Save results
    results_dir = Path(f"experiments/rf_baseline_{dataset}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump({
            'validation': val_metrics,
            'test': test_metrics
        }, f, indent=2)
    
    with open(results_dir / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nResults for {dataset}:")
    print(f"Validation R2: {val_metrics['r2']:.4f} ± {val_metrics['r2_std']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f} ± {test_metrics['r2_std']:.4f}")
    
    return model, test_metrics

def optimize_rf_hyperparameters(dataset="finetune", n_trials=30):
    """Optimize RF hyperparameters using Optuna"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        
        # Load data
        if dataset == "pretrain":
            train_path = 'datasets/chembl_pretraining_train.csv'
            val_path = 'datasets/chembl_pretraining_val.csv'
            train_mask_path = 'datasets/chembl_pretraining_train_mask.csv'
            val_mask_path = 'datasets/chembl_pretraining_val_mask.csv'
        else:
            train_path = 'datasets/pkis2_finetuning_train.csv'
            val_path = 'datasets/pkis2_finetuning_val.csv'
            train_mask_path = 'datasets/pkis2_finetuning_train_mask.csv'
            val_mask_path = 'datasets/pkis2_finetuning_val_mask.csv'
        
        X_train, y_train, mask_train, _ = load_kinase_data(train_path, train_mask_path)
        X_val, y_val, mask_val, _ = load_kinase_data(val_path, val_mask_path)
        
        # Train model
        model = ECFPRandomForest(**params, n_jobs=-1)
        model.fit(X_train, y_train, mask_train)
        
        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, mask_val)
        
        return -val_metrics['r2']  # Minimize negative R2
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best params for {dataset}:")
    print(study.best_params)
    print(f"Best R2: {-study.best_value:.4f}")
    
    with open(f"experiments/rf_hpo_{dataset}.json", 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_r2': -study.best_value
        }, f, indent=2)
    
    return study.best_params

def compare_all_models():
    """Compare RF baseline with CNN results"""
    
    results = []
    
    # Train RF on both datasets
    for dataset in ["pretrain", "finetune"]:
        model, metrics = train_rf_baseline(dataset)
        results.append({
            'model': f'RF_{dataset}',
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'mae': metrics['mae']
        })
    
    # Load CNN results if available
    cnn_results_path = Path("experiments/comparison_summary.csv")
    if cnn_results_path.exists():
        cnn_df = pd.read_csv(cnn_results_path)
        for _, row in cnn_df.iterrows():
            results.append({
                'model': f"CNN_{row['strategy']}_aug{row['augmentation']}",
                'rmse': row['rmse'],
                'r2': row['r2'],
                'mae': row.get('mae', np.nan)
            })
    
    # Create comparison table
    df = pd.DataFrame(results)
    df = df.sort_values('r2', ascending=False)
    df.to_csv("experiments/full_comparison.csv", index=False)
    
    print("\n" + "="*60)
    print("FULL MODEL COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            dataset = sys.argv[2] if len(sys.argv) > 2 else "pretrain"
            train_rf_baseline(dataset)
        elif sys.argv[1] == "optimize":
            dataset = sys.argv[2] if len(sys.argv) > 2 else "finetune"
            optimize_rf_hyperparameters(dataset)
        elif sys.argv[1] == "compare":
            compare_all_models()
    else:
        # Train baselines for both datasets
        print("Training RF baseline on pretraining data...")
        train_rf_baseline("pretrain")
        print("\nTraining RF baseline on finetuning data...")
        train_rf_baseline("finetune")
        print("\nComparing all models...")
        compare_all_models()