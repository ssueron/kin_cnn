import keras
import pandas as pd
import numpy as np
import json
import re
from typing import List
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from deepclp.models.utils import get_embedding_layer

_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
_SMILES_REGEX = re.compile(rf"(\[|\]|{_ELEMENTS_STR}|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%\d{{2}}|\d)")

def segment_smiles(smiles: str) -> List[str]:
    return _SMILES_REGEX.findall(smiles)

def smiles_label_encoding(smiles: str, token_to_label: dict) -> List[int]:
    return [token_to_label.get(token, token_to_label.get("UNK", 3)) for token in segment_smiles(smiles)]

def evaluate_bioactivity(y_true, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'pearson_r': pearsonr(y_true, y_pred)[0],
        'spearman_r': spearmanr(y_true, y_pred)[0]
    }


class SingleKinaseCNN(keras.Model):
    def __init__(
        self,
        token_encoding: str,
        embedding_dim: int,
        n_layers: int,
        kernel_size: int,
        n_filters: int,
        dense_layer_size: int,
        dropout: float,
        vocab_size: int,
        maxlen: int,
    ):
        """Single-task CNN for individual kinase inhibition prediction.

        Args:
            token_encoding (str): Type of token encoding. Choose from {"onehot", "learnable", "random"}.
            embedding_dim (int): Dimension of token embeddings.
            n_layers (int): Number of convolutional layers.
            kernel_size (int): Size of convolutional kernel.
            n_filters (int): Number of filters in convolutional layers.
            dense_layer_size (int): Size of dense layer.
            dropout (float): Dropout rate.
            maxlen (int): Maximum length of input sequences.
            vocab_size (int): Size of vocabulary.
        """
        super().__init__()
        
        # Input validation
        if dense_layer_size < 4:
            raise ValueError(f"dense_layer_size must be at least 4, got {dense_layer_size}")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.token_encoding = token_encoding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.dense_layer_size = dense_layer_size
        self.dropout = dropout
        self.is_classification = False  # Always regression for this model

        self.embedding = get_embedding_layer(
            token_encoding,
            vocab_size,
            embedding_dim
        )
        self.convs = [
            keras.layers.Conv1D(
                filters=(layer_ix + 1) * n_filters,
                kernel_size=kernel_size,
                strides=1,
                activation="relu",
                padding="valid",
            )
            for layer_ix in range(n_layers)
        ]
        self.dropout_layer = keras.layers.Dropout(dropout)
        self.pooling = keras.layers.GlobalMaxPooling1D()
        self.dense1 = keras.layers.Dense(
            dense_layer_size,
            activation="relu",
        )
        self.dense2 = keras.layers.Dense(
            dense_layer_size // 2,
            activation="relu",
        )
        self.dense3 = keras.layers.Dense(
            dense_layer_size // 4,
            activation="relu",
        )
        # Output layer with single neuron and linear activation for regression
        self.output_layer = keras.layers.Dense(1, activation="linear")

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        for conv in self.convs:
            x = conv(x)
            x = self.dropout_layer(x, training=training)

        x = self.pooling(x)
        x = self.dense1(x)
        x = self.dropout_layer(x, training=training)
        x = self.dense2(x)
        x = self.dropout_layer(x, training=training)
        x = self.dense3(x)
        predictions = self.output_layer(x)
        
        return predictions

    def load_vegfr2_data(self, csv_path, maxlen=85):
        df = pd.read_csv(csv_path)
        X = df["smiles"].values.tolist()
        
        with open("custom_vocab.json") as f:
            vocab = json.load(f)
        X = [smiles_label_encoding(s, vocab) for s in X]
        
        y = df["VEGFR2"].values
        X = keras.preprocessing.sequence.pad_sequences(
            X, padding="post", maxlen=maxlen, value=0
        )
        return X, y

    def train_vegfr2(self, train_path, val_path, config):
        X_train, y_train = self.load_vegfr2_data(train_path)
        X_val, y_val = self.load_vegfr2_data(val_path)
        
        optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
        self.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[keras.metrics.RootMeanSquaredError()]
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=25,
                restore_best_weights=True
            )
        ]
        
        if WANDB_AVAILABLE:
            run = wandb.init(project="vegfr2-cnn", config=config)
            print(f"W&B run initialized: {run.name}")
            print(f"W&B project URL: {run.url}")
        
        history = self.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        if WANDB_AVAILABLE:
            for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                wandb.log({"epoch": epoch, "loss": loss, "val_loss": val_loss})
        
        return history

    def evaluate_vegfr2(self, test_path):
        X_test, y_test = self.load_vegfr2_data(test_path)
        y_pred = self.predict(X_test).flatten()
        
        metrics = evaluate_bioactivity(y_test, y_pred)
        
        return metrics