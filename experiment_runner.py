import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf

from setup_environment import setup_tf_environment
from deepclp.models.multi_kinase_cnn import MultiKinaseCNN
from deepclp.kinase_training import (
    kinase_csv_to_matrix_with_mask,
    train_multikinase_predictor_with_mask,
    evaluate_multikinase_predictor
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"experiments/{experiment_name}_{self.timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.exp_dir / "metrics.json"
        self.config_file = self.exp_dir / "config.json"
        
    def log_config(self, config: Dict):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict, phase: str = ""):
        metrics_path = self.exp_dir / f"metrics_{phase}.json" if phase else self.metrics_file
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    def save_model(self, model: MultiKinaseCNN, name: str):
        model_dir = self.exp_dir / "models"
        model_dir.mkdir(exist_ok=True)
        model.save_weights(str(model_dir / f"{name}.weights.h5"))
    
    def log_history(self, history: Dict, phase: str):
        df = pd.DataFrame(history)
        df.to_csv(self.exp_dir / f"history_{phase}.csv", index=False)

class ExperimentRunner:
    def __init__(self, config: Dict):
        force_cpu = config.get('force_cpu', False)
        setup_tf_environment(force_cpu=force_cpu, verbose=True)
        
        self.config = config
        self.tracker = ExperimentTracker(config['experiment_name'])
        self.tracker.log_config(config)
        
        self.pretrain_paths = {
            'train': 'datasets/chembl_pretraining_train.csv',
            'val': 'datasets/chembl_pretraining_val.csv',
            'test': 'datasets/chembl_pretraining_test.csv'
        }
        self.finetune_paths = {
            'train': 'datasets/pkis2_finetuning_train.csv',
            'val': 'datasets/pkis2_finetuning_val.csv',
            'test': 'datasets/pkis2_finetuning_test.csv'
        }
        
    def load_data(self, dataset: str, augmentation_factor: int = 0) -> Dict:
        paths = self.pretrain_paths if dataset == "pretrain" else self.finetune_paths
        
        data = {}
        kinase_names = None
        for split in ['train', 'val', 'test']:
            X, y, mask, names = kinase_csv_to_matrix_with_mask(
                csv_path=paths[split],
                mask_path=paths[split].replace('.csv', '_mask.csv'),
                representation_name="smiles",
                maxlen=self.config['maxlen'],
                custom_vocab_path=self.config.get('vocab_path')
            )
            
            if kinase_names is None:
                kinase_names = names
            
            if split == 'train' and augmentation_factor > 0:
                X, y, mask = self.augment_data(X, y, mask, paths[split], augmentation_factor)
            
            data[f'X_{split}'] = X
            data[f'y_{split}'] = y
            data[f'mask_{split}'] = mask
            
        data['kinase_names'] = kinase_names
        return data
    
    def augment_data(self, X, y, mask, csv_path, factor):
        from rdkit import Chem
        import random
        import gc
        
        df = pd.read_csv(csv_path)
        
        if 'smiles' in df.columns:
            smiles_col = 'smiles'
        elif 'curated_smiles' in df.columns:
            smiles_col = 'curated_smiles'
        elif 'molecule' in df.columns:
            smiles_col = 'molecule'
        else:
            raise ValueError(f"No SMILES column found in {csv_path}. Expected 'smiles', 'curated_smiles', or 'molecule'")
        
        total_size = len(df) * factor
        augmented_X = np.zeros((total_size, X.shape[1]), dtype=X.dtype)
        augmented_y = np.zeros((total_size, y.shape[1]), dtype=y.dtype)
        augmented_mask = np.zeros((total_size, mask.shape[1]), dtype=mask.dtype)
        
        idx = 0
        for i, smiles in enumerate(df[smiles_col].values):
            augmented_X[idx] = X[i]
            augmented_y[idx] = y[i]
            augmented_mask[idx] = mask[i]
            idx += 1
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    for _ in range(factor - 1):
                        aug_smiles = Chem.MolToSmiles(mol, doRandom=True)
                        from deepclp import sequence_utils
                        if self.config.get('vocab_path'):
                            with open(self.config['vocab_path']) as f:
                                vocab = json.load(f)
                            aug_encoded = custom_smiles_encoding(aug_smiles, vocab)
                        else:
                            with open('data/smiles_vocab_extended.json') as f:
                                vocab = json.load(f)
                            aug_encoded = sequence_utils.smiles_label_encoding(aug_smiles, vocab)
                        
                        aug_encoded = tf.keras.preprocessing.sequence.pad_sequences(
                            [aug_encoded], padding="post", maxlen=self.config['maxlen'], value=0
                        )[0]
                        
                        augmented_X[idx] = aug_encoded
                        augmented_y[idx] = y[i]
                        augmented_mask[idx] = mask[i]
                        idx += 1
            except (ValueError, TypeError, AttributeError) as e:
                continue
        
        result_X = augmented_X[:idx].copy()
        result_y = augmented_y[:idx].copy()
        result_mask = augmented_mask[:idx].copy()
        
        del augmented_X, augmented_y, augmented_mask, df
        gc.collect()
        
        return result_X, result_y, result_mask
    
    def create_model(self, n_kinases: int = 318) -> MultiKinaseCNN:
        return MultiKinaseCNN(
            token_encoding=self.config['token_encoding'],
            embedding_dim=self.config['embedding_dim'],
            n_layers=self.config['n_layers'],
            kernel_size=self.config['kernel_size'],
            n_filters=self.config['n_filters'],
            dense_layer_size=self.config['dense_layer_size'],
            dropout=self.config['dropout'],
            vocab_size=self.config['vocab_size'],
            maxlen=self.config['maxlen'],
            n_kinases=n_kinases
        )
    
    def run_strategy_pretrain_only(self):
        print("Running Strategy: Pretrain Only")
        data = self.load_data("pretrain", self.config['augmentation_factor'])
        
        n_kinases = data['y_train'].shape[1]
        model = self.create_model(n_kinases)
        history = train_multikinase_predictor_with_mask(
            model=model,
            X_train=data['X_train'],
            y_train=data['y_train'],
            mask_train=data['mask_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            mask_val=data['mask_val'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            epochs=self.config['pretrain_epochs']
        )
        
        self.tracker.log_history(history, "pretrain")
        self.tracker.save_model(model, "pretrain_final")
        
        test_metrics = evaluate_multikinase_predictor(
            model, data['X_test'], data['y_test'], data['mask_test'], data['kinase_names']
        )
        self.tracker.log_metrics(test_metrics, "pretrain_test")
        
        return model, test_metrics
    
    def run_strategy_finetune_only(self):
        print("Running Strategy: Finetune Only")
        data = self.load_data("finetune", self.config['augmentation_factor'])
        
        n_kinases = data['y_train'].shape[1]
        model = self.create_model(n_kinases)
        history = train_multikinase_predictor_with_mask(
            model=model,
            X_train=data['X_train'],
            y_train=data['y_train'],
            mask_train=data['mask_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            mask_val=data['mask_val'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            epochs=self.config['finetune_epochs']
        )
        
        self.tracker.log_history(history, "finetune")
        self.tracker.save_model(model, "finetune_final")
        
        test_metrics = evaluate_multikinase_predictor(
            model, data['X_test'], data['y_test'], data['mask_test'], data['kinase_names']
        )
        self.tracker.log_metrics(test_metrics, "finetune_test")
        
        return model, test_metrics
    
    def run_strategy_transfer_full(self):
        print("Running Strategy: Transfer Learning (Full)")
        
        pretrain_data = self.load_data("pretrain", self.config['augmentation_factor'])
        n_kinases = pretrain_data['y_train'].shape[1]
        model = self.create_model(n_kinases)
        
        history_pretrain = train_multikinase_predictor_with_mask(
            model=model,
            X_train=pretrain_data['X_train'],
            y_train=pretrain_data['y_train'],
            mask_train=pretrain_data['mask_train'],
            X_val=pretrain_data['X_val'],
            y_val=pretrain_data['y_val'],
            mask_val=pretrain_data['mask_val'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            epochs=self.config['pretrain_epochs']
        )
        
        self.tracker.log_history(history_pretrain, "pretrain")
        self.tracker.save_model(model, "pretrain_checkpoint")
        
        finetune_data = self.load_data("finetune", self.config['augmentation_factor'])
        finetune_lr = self.config['learning_rate'] * self.config['finetune_lr_multiplier']
        
        history_finetune = train_multikinase_predictor_with_mask(
            model=model,
            X_train=finetune_data['X_train'],
            y_train=finetune_data['y_train'],
            mask_train=finetune_data['mask_train'],
            X_val=finetune_data['X_val'],
            y_val=finetune_data['y_val'],
            mask_val=finetune_data['mask_val'],
            learning_rate=finetune_lr,
            batch_size=self.config['batch_size'],
            epochs=self.config['finetune_epochs']
        )
        
        self.tracker.log_history(history_finetune, "finetune")
        self.tracker.save_model(model, "transfer_full_final")
        
        pretrain_test_metrics = evaluate_multikinase_predictor(
            model, pretrain_data['X_test'], pretrain_data['y_test'], pretrain_data['mask_test'], pretrain_data['kinase_names']
        )
        finetune_test_metrics = evaluate_multikinase_predictor(
            model, finetune_data['X_test'], finetune_data['y_test'], finetune_data['mask_test'], finetune_data['kinase_names']
        )
        
        self.tracker.log_metrics({
            'pretrain_test': pretrain_test_metrics,
            'finetune_test': finetune_test_metrics
        }, "transfer_full_test")
        
        return model, finetune_test_metrics
    
    def run_strategy_transfer_frozen(self):
        print("Running Strategy: Transfer Learning (Frozen)")
        
        pretrain_data = self.load_data("pretrain", self.config['augmentation_factor'])
        n_kinases = pretrain_data['y_train'].shape[1]
        model = self.create_model(n_kinases)
        
        history_pretrain = train_multikinase_predictor_with_mask(
            model=model,
            X_train=pretrain_data['X_train'],
            y_train=pretrain_data['y_train'],
            mask_train=pretrain_data['mask_train'],
            X_val=pretrain_data['X_val'],
            y_val=pretrain_data['y_val'],
            mask_val=pretrain_data['mask_val'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            epochs=self.config['pretrain_epochs']
        )
        
        self.tracker.log_history(history_pretrain, "pretrain")
        self.tracker.save_model(model, "pretrain_checkpoint")
        
        for layer in model.convs:
            layer.trainable = False
        model.embedding.trainable = False
        
        finetune_data = self.load_data("finetune", self.config['augmentation_factor'])
        finetune_lr = self.config['learning_rate'] * self.config['finetune_lr_multiplier']
        
        history_finetune = train_multikinase_predictor_with_mask(
            model=model,
            X_train=finetune_data['X_train'],
            y_train=finetune_data['y_train'],
            mask_train=finetune_data['mask_train'],
            X_val=finetune_data['X_val'],
            y_val=finetune_data['y_val'],
            mask_val=finetune_data['mask_val'],
            learning_rate=finetune_lr,
            batch_size=self.config['batch_size'],
            epochs=self.config['finetune_epochs']
        )
        
        self.tracker.log_history(history_finetune, "finetune")
        self.tracker.save_model(model, "transfer_frozen_final")
        
        finetune_test_metrics = evaluate_multikinase_predictor(
            model, finetune_data['X_test'], finetune_data['y_test'], finetune_data['mask_test']
        )
        self.tracker.log_metrics(finetune_test_metrics, "transfer_frozen_test")
        
        return model, finetune_test_metrics
    
    def run(self):
        strategy = self.config['strategy']
        
        if strategy == "pretrain_only":
            return self.run_strategy_pretrain_only()
        elif strategy == "finetune_only":
            return self.run_strategy_finetune_only()
        elif strategy == "transfer_full":
            return self.run_strategy_transfer_full()
        elif strategy == "transfer_frozen":
            return self.run_strategy_transfer_frozen()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

def run_all_experiments(base_config: Dict):
    strategies = ["pretrain_only", "finetune_only", "transfer_full", "transfer_frozen"]
    augmentation_factors = [0, 5, 10]
    
    results = []
    
    for strategy in strategies:
        for aug_factor in augmentation_factors:
            config = base_config.copy()
            config['strategy'] = strategy
            config['augmentation_factor'] = aug_factor
            config['experiment_name'] = f"{strategy}_aug{aug_factor}"
            
            print(f"\n{'='*60}")
            print(f"Running: {config['experiment_name']}")
            print(f"{'='*60}")
            
            runner = ExperimentRunner(config)
            model, metrics = runner.run()
            
            results.append({
                'strategy': strategy,
                'augmentation': aug_factor,
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'mse': metrics['mse']
            })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiments/comparison_summary.csv", index=False)
    print("\n\nResults Summary:")
    print(df_results)
    
    return df_results
