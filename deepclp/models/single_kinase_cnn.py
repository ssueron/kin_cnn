import keras

from deepclp.models.utils import get_embedding_layer


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