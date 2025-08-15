import optuna
import pandas as pd
import numpy as np
import json
import keras
import tensorflow as tf
from deepclp.models.multi_kinase_cnn import MultiKinaseCNN
from deepclp.models.single_kinase_cnn import smiles_label_encoding, WANDB_AVAILABLE

if WANDB_AVAILABLE:
    import wandb

def get_n_kinases(csv_path):
    df = pd.read_csv(csv_path, nrows=1)
    return len([col for col in df.columns if col != "smiles"])

def objective(trial):
    n_kinases = get_n_kinases("datasets/train_clean.csv")
    
    model = MultiKinaseCNN(
        token_encoding="learnable",
        embedding_dim=trial.suggest_categorical('embedding_dim', [64, 128, 256, 512]),
        n_layers=trial.suggest_int('n_layers', 2, 5),
        kernel_size=trial.suggest_categorical('kernel_size', [4, 5, 6, 7]),
        n_filters=trial.suggest_categorical('n_filters', [32, 64, 128]),
        dense_layer_size=trial.suggest_categorical('dense_layer_size', [128, 256, 512, 1024]),
        dropout=trial.suggest_float('dropout', 0.0, 0.6),
        vocab_size=47,
        maxlen=85,
        n_kinases=n_kinases
    )
    
    config = {
        "learning_rate": trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        "epochs": 300,
        "batch_size": trial.suggest_categorical('batch_size', [16, 32, 64]),
        "n_kinases": n_kinases
    }
    
    if WANDB_AVAILABLE:
        wandb.init(project="multi-kinase-cnn", config=config)
    
    def load_multi_kinase_data_with_mask(csv_path):
        df = pd.read_csv(csv_path)
        X = df["smiles"].values.tolist()
        
        with open("custom_vocab.json") as f:
            vocab = json.load(f)
        X = [smiles_label_encoding(s, vocab) for s in X]
        
        kinase_columns = [col for col in df.columns if col != "smiles"]
        y = df[kinase_columns].values.astype(np.float32)
        
        # Create mask: True = valid, False = missing
        mask = ~np.isnan(y)
        
        # Replace NaN with 0 (ignored by mask)
        y = np.nan_to_num(y, nan=0.0)
        
        X = keras.preprocessing.sequence.pad_sequences(X, padding="post", maxlen=85, value=0)
        return X, y, mask
    
    X_train, y_train, mask_train = load_multi_kinase_data_with_mask("datasets/train_clean.csv")
    X_val, y_val, mask_val = load_multi_kinase_data_with_mask("datasets/val_clean.csv")
    
    optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # Training phase
        epoch_losses = []
        
        for i in range(0, len(X_train), config["batch_size"]):
            batch_X = X_train[i:i+config["batch_size"]]
            batch_y = y_train[i:i+config["batch_size"]]
            batch_mask = mask_train[i:i+config["batch_size"]]
            
            with tf.GradientTape() as tape:
                predictions = model(batch_X, training=True)
                loss = model.compute_masked_loss(batch_y, predictions, batch_mask)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_losses.append(float(loss))
        
        # Validation phase
        val_predictions = model(X_val, training=False)
        val_loss = model.compute_masked_loss(y_val, val_predictions, mask_val)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break
    
    final_val_loss = float(best_val_loss)
    
    if WANDB_AVAILABLE:
        wandb.finish()
    
    return final_val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best trial:")
print(study.best_params)
print(f"Best val_loss: {study.best_value:.4f}")