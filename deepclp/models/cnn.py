import keras

from deepclp.models.utils import get_embedding_layer


class CNN(keras.Model):
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
        task_type: str,
    ):
        """Convolutional Neural Network model.

        Args:
            token_encoding (str): Type of token encoding to use. Choose from {"onehot", "learnable", "random"}.
            embedding_dim (int): Dimension of the token embeddings.
            n_layers (int): Number of convolutional layers.
            kernel_size (int): Size of the convolutional kernel.
            n_filters (int): Number of filters in the convolutional layers.
            dense_layer_size (int): Size of the dense layer.
            dropout (float): Dropout rate.
            maxlen (int): Maximum length of the input sequences.
            vocab_size (int): Size of the vocabulary.
            task_type (str): Type of prediction task. Choose from:
                - "bounded_regression": Predicts values in [0,1] (e.g., percentage inhibition)
                - "classification": Binary classification 
                - "regression": Unbounded regression (e.g., IC50 values)

        Returns:
            keras.Model: CNN model
        """
        super().__init__()
        self.token_encoding = token_encoding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.dense_layer_size = dense_layer_size
        self.dropout = dropout
        self.task_type = task_type

        self.embedding = get_embedding_layer(
            token_encoding,
            vocab_size,
            embedding_dim,
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
        self.dropout = keras.layers.Dropout(dropout)
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
        if task_type == "bounded_regression":
            output_activation = "sigmoid"  # Constrains output to [0,1] for percentage inhibition
        elif task_type == "classification":
            output_activation = "sigmoid"  # For binary classification probabilities
        elif task_type == "regression":
            output_activation = "linear"   # Unbounded output for IC50, binding affinity, etc.
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Choose from 'bounded_regression', 'classification', 'regression'")
        
        self.output_layer = keras.layers.Dense(1, activation=output_activation)

    def call(self, inputs):
        x = self.embedding(inputs)
        for conv in self.convs:
            x = conv(x)
            x = self.dropout(x)

        x = self.pooling(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return self.output_layer(x)
