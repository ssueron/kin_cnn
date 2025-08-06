"""
CNN Hyperparameter Optimization for Bioactivity Prediction
==========================================================

This script trains CNN models on kinase bioactivity data using three scenarios:
1. Training on finetuning dataset only (small dataset)
2. Training on pretraining dataset only (large dataset)  
3. Transfer learning (pretrain on large, then finetune on small)

For beginners:
- Hyperparameter optimization automatically finds the best model settings
- We test different combinations to find what works best
- The results help you understand which approach gives the best predictions

Dataset: 355 kinases with activity data in both datasets
- Pretraining: 79,492 molecules
- Finetuning: 645 molecules

Hyperparameter Search Space (as requested):
- No. layers: 1, 2, 3
- Dropout: 0.25 (fixed)
- Batch size: 32 (fixed)  
- No. filters: 32, 64, 128
- Kernel length: 3, 5, 7
- Learning rate: 10^-2, 10^-3, 5×10^-3, 10^-4, 5×10^-5
"""

import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import pickle
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from deepclp import models, training


class CNNKinaseOptimizer:
    """
    CNN hyperparameter optimizer for kinase bioactivity prediction.
    
    For beginners: This class handles all the optimization automatically.
    It finds the best settings for your CNN model.
    """
    
    def __init__(self, pretraining_path: str, finetuning_path: str, results_dir: str = "cnn_optimization_results"):
        """
        Initialize the optimizer.
        
        Args:
            pretraining_path: Path to large pretraining dataset
            finetuning_path: Path to smaller finetuning dataset  
            results_dir: Directory to save results
        """
        self.pretraining_path = pretraining_path
        self.finetuning_path = finetuning_path
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print("🔬 Loading datasets...")
        self.pretraining_data = self._load_dataset_with_special_header(pretraining_path)
        self.finetuning_data = self._load_dataset_with_special_header(finetuning_path)
        
        # Find common kinases (targets present in both datasets)
        pre_kinases = set([col for col in self.pretraining_data.columns if col != 'Smiles'])
        fine_kinases = set([col for col in self.finetuning_data.columns if col != 'Smiles'])
        self.common_kinases = list(pre_kinases.intersection(fine_kinases))
        self.n_kinases = len(self.common_kinases)
        
        print(f"📊 Dataset loaded:")
        print(f"   • Pretraining: {len(self.pretraining_data)} molecules")
        print(f"   • Finetuning: {len(self.finetuning_data)} molecules") 
        print(f"   • Common kinases: {self.n_kinases}")
        
        # Prepare vocabulary for SMILES encoding
        self.vocab, self.vocab_size = self._build_vocabulary()
        print(f"   • Vocabulary size: {self.vocab_size}")
    
    def _load_dataset_with_special_header(self, filepath: str) -> pd.DataFrame:
        """Load dataset with special header format (handles quotes)."""
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
        
        # Parse header with nested quotes
        header_clean = header_line.strip('"')
        headers = []
        current_header = ''
        in_quotes = False

        for char in header_clean:
            if char == '"' and not in_quotes:
                in_quotes = True
                current_header += char
            elif char == '"' and in_quotes:
                in_quotes = False
                current_header += char
            elif char == ',' and not in_quotes:
                headers.append(current_header.strip('"'))
                current_header = ''
            else:
                current_header += char

        if current_header:
            headers.append(current_header.strip('"'))
        
        df = pd.read_csv(filepath, skiprows=1, names=headers)
        return df
    
    def _build_vocabulary(self, maxlen: int = 100) -> Tuple[Dict, int]:
        """
        Build vocabulary from SMILES strings.
        
        For beginners: This converts chemical structures (SMILES) into numbers
        that the neural network can understand.
        """
        # Collect all unique characters from SMILES strings
        vocab_chars = set()
        
        for smiles in self.pretraining_data['Smiles']:
            if pd.notna(smiles):
                vocab_chars.update(list(str(smiles)))
                
        for smiles in self.finetuning_data['Smiles']:
            if pd.notna(smiles):
                vocab_chars.update(list(str(smiles)))
        
        # Create character to index mapping (0 reserved for padding)
        vocab_chars = sorted(list(vocab_chars))
        vocab = {char: i+1 for i, char in enumerate(vocab_chars)}
        vocab['<PAD>'] = 0  # Padding token
        
        return vocab, len(vocab)
    
    def _smiles_to_sequence(self, smiles: str, maxlen: int = 100) -> np.ndarray:
        """Convert SMILES string to numerical sequence."""
        if pd.isna(smiles):
            return np.zeros(maxlen, dtype=int)
        
        sequence = np.zeros(maxlen, dtype=int)
        smiles_str = str(smiles)
        
        for i, char in enumerate(smiles_str[:maxlen]):
            sequence[i] = self.vocab.get(char, 0)  # Unknown chars become 0
            
        return sequence
    
    def _prepare_data(self, data: pd.DataFrame, maxlen: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare molecular data for training.
        
        For beginners: This converts SMILES strings and activity values 
        into arrays that the CNN can process.
        """
        # Convert SMILES to sequences
        X = []
        for smiles in data['Smiles']:
            sequence = self._smiles_to_sequence(smiles, maxlen)
            X.append(sequence)
        X = np.array(X)
        
        # Extract target values for common kinases only
        y = data[self.common_kinases].values.astype(float)
        
        # Handle missing values (NaN) by setting them to 0
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def _create_cnn_model(self, trial, maxlen: int = 100) -> keras.Model:
        """
        Create CNN model with hyperparameters from Optuna trial.
        
        For beginners: This builds the neural network with different settings
        to find the best combination.
        """
        # Hyperparameters with your specified ranges
        n_layers = trial.suggest_categorical('n_layers', [1, 2, 3])
        n_filters = trial.suggest_categorical('n_filters', [32, 64, 128])  
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.001, 0.005, 0.0001, 0.00005])
        
        # Fixed hyperparameters as requested
        dropout = 0.25
        batch_size = 32  # Will be used in training, not model creation
        
        # Build CNN model  
        model = keras.Sequential([
            # Embedding layer to convert indices to dense vectors
            keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=64,  # Fixed embedding dimension
                input_length=maxlen,
                name='embedding'
            )
        ])
        
        # Add convolutional layers
        for i in range(n_layers):
            model.add(keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(keras.layers.BatchNormalization(name=f'batch_norm_{i+1}'))
            if i < n_layers - 1:  # No pooling after last conv layer
                model.add(keras.layers.MaxPooling1D(
                    pool_size=2,
                    name=f'maxpool_{i+1}'
                ))
        
        # Global pooling and fully connected layers
        model.add(keras.layers.GlobalMaxPooling1D(name='global_maxpool'))
        model.add(keras.layers.Dense(128, activation='relu', name='dense_1'))
        model.add(keras.layers.Dropout(dropout, name='dropout'))
        model.add(keras.layers.Dense(64, activation='relu', name='dense_2'))
        
        # Output layer for multi-target regression
        model.add(keras.layers.Dense(self.n_kinases, activation='linear', name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _evaluate_model(self, model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # R² score (handle potential numerical issues)
        try:
            r2 = r2_score(y_test, y_pred)
        except:
            r2 = -999  # Very bad score if calculation fails
            
        return {
            'mse': mse,
            'mae': mae,  
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def _objective_scenario1(self, trial) -> float:
        """
        Scenario 1: Train CNN on finetuning dataset only.
        
        For beginners: This trains a model from scratch using only the small dataset.
        """
        print(f"\\n🎯 Scenario 1 - Trial {trial.number}: Finetuning data only")
        
        try:
            # Prepare data
            X, y = self._prepare_data(self.finetuning_data)
            
            # Split into train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"   📊 Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Create model
            model = self._create_cnn_model(trial)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,  # Moderate epochs for optimization
                batch_size=32,  # Fixed as requested
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=8,
                        restore_best_weights=True,
                        monitor='val_loss'
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=4,
                        factor=0.5,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            print(f"   📈 Val MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
            return metrics['mse']  # Minimize MSE
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            return float('inf')
    
    def _objective_scenario2(self, trial) -> float:
        """
        Scenario 2: Train CNN on pretraining dataset only.
        
        For beginners: This trains a model using only the large dataset.
        """
        print(f"\\n🎯 Scenario 2 - Trial {trial.number}: Pretraining data only")
        
        try:
            # Prepare data
            X, y = self._prepare_data(self.pretraining_data)
            
            # Split into train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42  # Smaller val set due to large data
            )
            
            print(f"   📊 Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Create model
            model = self._create_cnn_model(trial)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,  # Fewer epochs due to large dataset
                batch_size=32,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                        monitor='val_loss'
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=3,
                        factor=0.5,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            print(f"   📈 Val MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
            return metrics['mse']
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            return float('inf')
    
    def _objective_scenario3(self, trial) -> float:
        """
        Scenario 3: Transfer learning - Pretrain then finetune.
        
        For beginners: This first trains on the large dataset, then fine-tunes
        on the small dataset. Often gives the best results!
        """
        print(f"\\n🎯 Scenario 3 - Trial {trial.number}: Transfer learning")
        
        try:
            # Phase 1: Pretrain on large dataset
            print("   📚 Phase 1: Pretraining...")
            X_pre, y_pre = self._prepare_data(self.pretraining_data)
            X_pre_train, X_pre_val, y_pre_train, y_pre_val = train_test_split(
                X_pre, y_pre, test_size=0.15, random_state=42
            )
            
            model = self._create_cnn_model(trial)
            
            # Pretrain
            model.fit(
                X_pre_train, y_pre_train,
                validation_data=(X_pre_val, y_pre_val),
                epochs=15,  # Moderate pretraining
                batch_size=32,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Phase 2: Fine-tune on small dataset
            print("   🔧 Phase 2: Fine-tuning...")
            X_fine, y_fine = self._prepare_data(self.finetuning_data)
            X_fine_train, X_fine_val, y_fine_train, y_fine_val = train_test_split(
                X_fine, y_fine, test_size=0.2, random_state=42
            )
            
            # Lower learning rate for fine-tuning
            finetune_lr = trial.params['learning_rate'] * 0.1
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=finetune_lr),
                loss='mse',
                metrics=['mae']
            )
            
            # Fine-tune
            history = model.fit(
                X_fine_train, y_fine_train,
                validation_data=(X_fine_val, y_fine_val),
                epochs=25,
                batch_size=32,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=8,
                        restore_best_weights=True,
                        monitor='val_loss'
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=4,
                        factor=0.5,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Evaluate on finetuning validation set
            metrics = self._evaluate_model(model, X_fine_val, y_fine_val)
            print(f"   📈 Val MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
            return metrics['mse']
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            return float('inf')
    
    def optimize_scenario(self, scenario: int, n_trials: int = 30) -> optuna.Study:
        """
        Run hyperparameter optimization for a specific scenario.
        
        For beginners: This automatically tries different model configurations
        to find the best one for your data.
        """
        print(f"\\n🚀 Starting optimization for Scenario {scenario}")
        print(f"🔄 Will try {n_trials} different configurations")
        print(f"🎛️  Search space:")
        print(f"   • Layers: [1, 2, 3]")
        print(f"   • Filters: [32, 64, 128]")
        print(f"   • Kernel size: [3, 5, 7]")
        print(f"   • Learning rate: [0.01, 0.001, 0.005, 0.0001, 0.00005]")
        print(f"   • Dropout: 0.25 (fixed)")
        print(f"   • Batch size: 32 (fixed)")
        
        # Choose objective function
        if scenario == 1:
            objective_func = self._objective_scenario1
        elif scenario == 2:
            objective_func = self._objective_scenario2
        elif scenario == 3:
            objective_func = self._objective_scenario3
        else:
            raise ValueError("Scenario must be 1, 2, or 3")
        
        # Create Optuna study
        study_name = f"cnn_scenario_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='minimize',  # Minimize MSE
            study_name=study_name
        )
        
        # Run optimization
        study.optimize(objective_func, n_trials=n_trials)
        
        # Save results
        results = {
            'scenario': scenario,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'timestamp': datetime.now().isoformat(),
            'search_space': {
                'n_layers': [1, 2, 3],
                'n_filters': [32, 64, 128],
                'kernel_size': [3, 5, 7],
                'learning_rate': [0.01, 0.001, 0.005, 0.0001, 0.00005],
                'dropout': 0.25,
                'batch_size': 32
            }
        }
        
        results_file = os.path.join(self.results_dir, f"cnn_scenario_{scenario}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study object
        study_file = os.path.join(self.results_dir, f"cnn_scenario_{scenario}_study.pkl")
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        print(f"\\n✅ Optimization completed!")
        print(f"📊 Best validation MSE: {study.best_value:.4f}")
        print(f"🎛️  Best parameters:")
        for param, value in study.best_params.items():
            print(f"   • {param}: {value}")
        print(f"💾 Results saved to: {results_file}")
        
        return study


def main():
    """Main function to run the optimization."""
    print("🧬 CNN Hyperparameter Optimization for Kinase Bioactivity Prediction")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = CNNKinaseOptimizer(
        pretraining_path="datasets/chembl_pretraining.csv",
        finetuning_path="datasets/pkis2_finetuning.csv"
    )
    
    print("\\nChoose which scenario to optimize:")
    print("1. Train on finetuning dataset only (small dataset)")
    print("2. Train on pretraining dataset only (large dataset)")
    print("3. Transfer learning (pretrain then finetune)")
    print("4. Optimize all scenarios")
    
    choice = input("\\nEnter your choice (1-4): ").strip()
    
    if choice in ['1', '2', '3']:
        scenario = int(choice)
        n_trials = int(input(f"Number of trials for scenario {scenario} (recommend 20-50): ") or "30")
        
        print(f"\\n🎯 Starting optimization for Scenario {scenario}")
        study = optimizer.optimize_scenario(scenario, n_trials=n_trials)
        
    elif choice == '4':
        print("\\n🎯 Optimizing all scenarios...")
        n_trials = int(input("Number of trials per scenario (recommend 20-30): ") or "25")
        
        for scenario in [1, 2, 3]:
            print(f"\\n{'='*50}")
            print(f"SCENARIO {scenario}")
            print(f"{'='*50}")
            study = optimizer.optimize_scenario(scenario, n_trials=n_trials)
    
    else:
        print("❌ Invalid choice. Please run again and select 1-4.")
        return
    
    print(f"\\n🎉 All optimizations completed!")
    print(f"📁 Results saved in: {optimizer.results_dir}")
    print(f"\\n💡 Next steps:")
    print(f"   1. Check the results JSON files for best hyperparameters")
    print(f"   2. Use the visualization script to compare scenarios")
    print(f"   3. Train final models with the best parameters")


if __name__ == "__main__":
    main()