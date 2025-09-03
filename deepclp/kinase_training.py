import json
import re
import logging
from typing import List, Tuple, Dict

import keras
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from deepclp import metrics, sequence_utils
from deepclp.models.multi_kinase_cnn import MultiKinaseCNN

logger = logging.getLogger(__name__)


def custom_smiles_encoding(smiles: str, vocab: dict) -> List[int]:
    """Encode SMILES with custom vocabulary
    
    Args:
        smiles (str): SMILES string to encode
        vocab (dict): Dictionary token -> id
        
    Returns:
        List[int]: List of token IDs
    """
    # Create regex pattern with tokens sorted by decreasing length
    tokens = sorted([k for k in vocab.keys() if k not in ["*", "^", "$", "UNK"]], key=len, reverse=True)
    pattern = '|'.join(re.escape(token) for token in tokens)
    
    # Tokenize with regex
    matches = re.findall(f'({pattern})', smiles)
    
    # Convert to IDs with handling of unknown tokens
    token_ids = []
    for token in matches:
        token_ids.append(vocab.get(token, vocab.get("UNK", 3)))  # 3 for unknown token
    
    return token_ids


def get_n_kinases_from_csv(csv_path: str) -> int:
    """Automatically detect the number of kinases in a CSV file
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        int: Number of kinases (columns - 1 for 'molecule')
    """
    df = pd.read_csv(csv_path, nrows=1)  # Read only the first line for columns
    kinase_columns = [col for col in df.columns if col != "molecule"]
    return len(kinase_columns)


def kinase_csv_to_matrix_with_mask(
    csv_path: str,
    mask_path: str,
    representation_name: str,
    maxlen: int,
    custom_vocab_path: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a CSV file to matrices with mask for multi-kinase training.

    Args:
        csv_path (str): Path to the CSV file with data.
        mask_path (str): Path to the CSV file with masks.
        representation_name (str): Name of the molecular representation.
        maxlen (int): Maximum length of sequences.
        custom_vocab_path (str): Path to custom vocabulary (optional).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: X, y, and mask matrices.
    """
    df = pd.read_csv(csv_path)
    mask_df = pd.read_csv(mask_path)
    
    # Detect SMILES column (smiles, curated_smiles, or molecule)
    smiles_col = None
    if "smiles" in df.columns:
        smiles_col = "smiles"
    elif "curated_smiles" in df.columns:
        smiles_col = "curated_smiles"
    elif "molecule" in df.columns:
        smiles_col = "molecule"
    else:
        raise ValueError("No SMILES column found (smiles, curated_smiles, or molecule)")
    
    # Extract molecules (SMILES)
    X = df[smiles_col].values.tolist()
    
    # Encode SMILES or SELFIES
    if representation_name == "smiles":
        # Use custom vocabulary if provided, otherwise default vocabulary
        if custom_vocab_path:
            with open(custom_vocab_path) as f:
                vocab = json.load(f)
            X = [custom_smiles_encoding(s, vocab) for s in X]
        else:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vocab_path = os.path.join(base_dir, "data", "smiles_vocab_extended.json")
            with open(vocab_path) as f:
                smiles_vocab = json.load(f)
            X = [sequence_utils.smiles_label_encoding(s, smiles_vocab) for s in X]
    elif representation_name == "selfies":
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab_path = os.path.join(base_dir, "data", "selfies_vocab.json")
        with open(vocab_path) as f:
            selfies_vocab = json.load(f)
        X = [sequence_utils.selfies_label_encoding(s, selfies_vocab) for s in X]
    else:
        raise ValueError(
            f"Invalid representation name: {representation_name}. Choose from {'smiles', 'selfies'}."
        )

    # Extract all kinase columns (all except SMILES column)
    kinase_columns = [col for col in df.columns if col != smiles_col]
    y = df[kinase_columns].values.astype(np.float32)
    
    # Extract mask (same columns as y)
    mask_columns = [col for col in mask_df.columns if col != smiles_col]
    mask = mask_df[mask_columns].values.astype(bool)
    
    # Padding input sequences
    X = keras.preprocessing.sequence.pad_sequences(
        X, padding="post", maxlen=maxlen, value=0
    )
    
    return X, y, mask


def kinase_csv_to_matrix(
    csv_path: str,
    representation_name: str,
    maxlen: int,
    custom_vocab_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a CSV file to matrices for multi-kinase training and prediction.
    Compatibility version - uses 'molecule' or 'curated_smiles' as SMILES column.

    Args:
        csv_path (str): Path to the CSV file.
        representation_name (str): Name of the molecular representation.
        maxlen (int): Maximum length of sequences.
        custom_vocab_path (str): Path to custom vocabulary (optional).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Input and output matrices, X and y.
    """
    df = pd.read_csv(csv_path)
    
    # Detect SMILES column (molecule or curated_smiles)
    smiles_col = None
    if "molecule" in df.columns:
        smiles_col = "molecule"
    elif "curated_smiles" in df.columns:
        smiles_col = "curated_smiles"
    else:
        raise ValueError("No SMILES column found (molecule or curated_smiles)")
    
    # Extract molecules
    X = df[smiles_col].values.tolist()
    
    # Encode SMILES or SELFIES
    if representation_name == "smiles":
        # Use custom vocabulary if provided, otherwise default vocabulary
        if custom_vocab_path:
            with open(custom_vocab_path) as f:
                vocab = json.load(f)
            X = [custom_smiles_encoding(s, vocab) for s in X]
        else:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vocab_path = os.path.join(base_dir, "data", "smiles_vocab_extended.json")
            with open(vocab_path) as f:
                smiles_vocab = json.load(f)
            X = [sequence_utils.smiles_label_encoding(s, smiles_vocab) for s in X]
    elif representation_name == "selfies":
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab_path = os.path.join(base_dir, "data", "selfies_vocab.json")
        with open(vocab_path) as f:
            selfies_vocab = json.load(f)
        X = [sequence_utils.selfies_label_encoding(s, selfies_vocab) for s in X]
    else:
        raise ValueError(
            f"Invalid representation name: {representation_name}. Choose from {'smiles', 'selfies'}."
        )

    # Extract all kinase columns (all except SMILES column)
    kinase_columns = [col for col in df.columns if col != smiles_col]
    y = df[kinase_columns].values
    
    # Padding input sequences
    X = keras.preprocessing.sequence.pad_sequences(
        X, padding="post", maxlen=maxlen, value=0
    )
    
    return X, y


def train_multikinase_predictor_with_mask(
    model: MultiKinaseCNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray,
    learning_rate: float,
    batch_size: int,
    epochs: int = 100,
) -> Dict[str, List[float]]:
    """Trains a multi-kinase predictor with mask handling.

    Args:
        model (MultiKinaseCNN): Model to train.
        X_train (np.ndarray): Training input matrix.
        y_train (np.ndarray): Training output matrix.
        mask_train (np.ndarray): Training mask.
        X_val (np.ndarray): Validation input matrix.
        y_val (np.ndarray): Validation output matrix.
        mask_val (np.ndarray): Validation mask.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.

    Returns:
        Dict[str, List[float]]: Training history.
    """
    import tensorflow as tf
    
    # Configure optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Initialize model
    _ = model(X_train[:1])
    
    history = {"loss": [], "val_loss": []}
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    min_delta = 1e-4
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        train_losses = []
        
        # Training by batch
        n_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size > 0 else 0)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            batch_mask = mask_train[start_idx:end_idx]
            
            # Forward pass with gradients
            with tf.GradientTape() as tape:
                predictions = model(batch_X, training=True)
                loss_value = model.compute_masked_loss(batch_y, predictions, batch_mask)
            
            # Backward pass and weight update
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_losses.append(float(loss_value))
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        val_losses = []
        
        val_batches = len(X_val) // batch_size + (1 if len(X_val) % batch_size > 0 else 0)
        
        for i in range(val_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X_val))
            
            batch_X = X_val[start_idx:end_idx]
            batch_y = y_val[start_idx:end_idx]
            batch_mask = mask_val[start_idx:end_idx]
            
            predictions = model(batch_X, training=False)
            loss_value = model.compute_masked_loss(batch_y, predictions, batch_mask)
            val_losses.append(float(loss_value))
        
        avg_val_loss = np.mean(val_losses)
        
        # Record history
        history["loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping with plateau detection
        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return history


def spearman_r(y_true, y_pred):
    if len(y_true) < 2 or np.var(y_true) == 0 or np.var(y_pred) == 0:
        return np.nan
    return spearmanr(y_true, y_pred)[0]


def pearson_r(y_true, y_pred):
    if len(y_true) < 2 or np.var(y_true) == 0 or np.var(y_pred) == 0:
        return np.nan
    return pearsonr(y_true, y_pred)[0]


def per_kinase_r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def train_multikinase_predictor(
    model: MultiKinaseCNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float,
    batch_size: int,
) -> Dict[str, List[float]]:
    """Trains a multi-kinase predictor (version without mask for compatibility).

    Args:
        model (MultiKinaseCNN): Model to train.
        X_train (np.ndarray): Training input matrix.
        y_train (np.ndarray): Training output matrix.
        X_val (np.ndarray): Validation input matrix.
        y_val (np.ndarray): Validation output matrix.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.

    Returns:
        Dict[str, List[float]]: Training history.
    """
    optimizer = keras.optimizers.get("adam")
    optimizer.learning_rate.assign(learning_rate)
    
    # Use MSE as loss function for regression
    loss = "mean_squared_error"
    
    # Appropriate metrics for regression
    metrics_list = [
        keras.metrics.RootMeanSquaredError(),
        keras.metrics.MeanAbsoluteError()
    ]

    model.compile(
        loss=loss,
        metrics=metrics_list,
        optimizer=optimizer,
    )
    
    # Initialize model with a forward pass
    model(X_val[:1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-5,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=500,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=callbacks,
    )

    return history.history


def evaluate_multikinase_predictor(
    model: MultiKinaseCNN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask_test: np.ndarray = None,
) -> Dict[str, float]:
    """Evaluates a multi-kinase predictor.

    Args:
        model (MultiKinaseCNN): Model to evaluate.
        X_test (np.ndarray): Test input matrix.
        y_test (np.ndarray): Test output matrix.
        mask_test (np.ndarray): Test mask matrix (True = valid value).

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    predictions = model.predict(X_test)
    
    # Calculate per-kinase metrics
    kinase_metrics = []
    kinase_r2_values = []
    
    for i in range(model.n_kinases):
        y_true_kinase = y_test[:, i]
        y_pred_kinase = predictions[:, i]
        
        # Apply mask if provided
        if mask_test is not None:
            mask_kinase = mask_test[:, i]
            valid_indices = mask_kinase
            y_true_kinase = y_true_kinase[valid_indices]
            y_pred_kinase = y_pred_kinase[valid_indices]
        
        # Skip kinases with insufficient data points
        if len(y_true_kinase) < 2 or np.isnan(y_true_kinase).all() or np.isnan(y_pred_kinase).all():
            kinase_metrics.append({"rmse": np.nan, "r2": np.nan, "mse": np.nan})
            kinase_r2_values.append(np.nan)
            continue
            
        kinase_metric = metrics.evaluate_predictions(
            y_true_kinase, y_pred_kinase, metrics=["rmse", "r2", "mse"]
        )
        kinase_metrics.append(kinase_metric)
        kinase_r2_values.append(kinase_metric["r2"])
    
    # Calculate profile-level correlations (per molecule)
    profile_pearson = []
    profile_spearman = []
    
    for i in range(len(X_test)):
        y_true_mol = y_test[i, :]
        y_pred_mol = predictions[i, :]
        
        if mask_test is not None:
            mask_mol = mask_test[i, :]
            valid_idx = mask_mol & ~np.isnan(y_true_mol) & ~np.isnan(y_pred_mol)
        else:
            valid_idx = ~np.isnan(y_true_mol) & ~np.isnan(y_pred_mol)
        
        if np.sum(valid_idx) > 1:
            profile_pearson.append(pearson_r(y_true_mol[valid_idx], y_pred_mol[valid_idx]))
            profile_spearman.append(spearman_r(y_true_mol[valid_idx], y_pred_mol[valid_idx]))
    
    # Filter valid kinases for averaging
    valid_kinases = [m for m in kinase_metrics if not (np.isnan(m["rmse"]) or np.isnan(m["r2"]) or np.isnan(m["mse"]))]
    valid_r2_values = [r2 for r2 in kinase_r2_values if not np.isnan(r2)]
    
    if len(valid_kinases) == 0:
        return {"rmse": np.nan, "r2": np.nan, "mse": np.nan, "profile_pearson": np.nan, "profile_spearman": np.nan}
    
    # Aggregate metrics
    avg_metrics = {
        "rmse": np.mean([m["rmse"] for m in valid_kinases]),
        "r2": np.mean([m["r2"] for m in valid_kinases]),
        "mse": np.mean([m["mse"] for m in valid_kinases]),
        "profile_pearson": np.nanmean(profile_pearson) if profile_pearson else np.nan,
        "profile_spearman": np.nanmean(profile_spearman) if profile_spearman else np.nan,
        "per_kinase_r2_mean": np.mean(valid_r2_values) if valid_r2_values else np.nan,
        "per_kinase_r2_std": np.std(valid_r2_values) if valid_r2_values else np.nan,
        "per_kinase_r2_values": valid_r2_values
    }
    
    return avg_metrics