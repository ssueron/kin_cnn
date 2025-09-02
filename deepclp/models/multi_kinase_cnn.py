import keras

from deepclp.models.utils import get_embedding_layer


class MultiKinaseCNN(keras.Model):
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
        n_kinases: int,  # Number of kinases to predict (adaptive)
    ):
        """Multi-task CNN for kinase inhibition prediction.

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
            n_kinases (int): Number of kinases to predict (tasks).
        """
        super().__init__()
        
        # Input validation
        if dense_layer_size < 4:
            raise ValueError(f"dense_layer_size must be at least 4, got {dense_layer_size}")
        if n_kinases <= 0:
            raise ValueError(f"n_kinases must be positive, got {n_kinases}")
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
        self.n_kinases = n_kinases
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
                activation=None,
                padding="valid",
            )
            for layer_ix in range(n_layers)
        ]
        self.batch_norms = [
            keras.layers.BatchNormalization()
            for _ in range(n_layers)
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
        # Output layer with n_kinases neurons and linear activation for regression
        self.output_layer = keras.layers.Dense(n_kinases, activation="linear")

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x)
            x = bn(x, training=training)
            x = keras.activations.relu(x)
            x = self.dropout_layer(x, training=training)

        x = self.pooling(x)
        x = self.dense1(x)
        x = self.dropout_layer(x, training=training)
        x = self.dense2(x)
        x = self.dropout_layer(x, training=training)
        x = self.dense3(x)
        predictions = self.output_layer(x)
        
        return predictions
    
    def compute_masked_loss(self, y_true, y_pred, mask):
        """Computes loss ignoring masked values.
        
        Args:
            y_true: Target values (batch_size, n_kinases)
            y_pred: Predictions (batch_size, n_kinases) 
            mask: Binary mask (batch_size, n_kinases) - True = valid value
        
        Returns:
            Average loss over non-masked values
        """
        import keras
        
        # Input validation
        y_true_shape = keras.ops.shape(y_true)
        y_pred_shape = keras.ops.shape(y_pred)
        mask_shape = keras.ops.shape(mask)
        
        # Validate dimensions match
        if len(y_true_shape) != 2 or len(y_pred_shape) != 2 or len(mask_shape) != 2:
            raise ValueError("All inputs must be 2D tensors")
        
        # Convert to backend tensors
        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(y_pred)
        mask = keras.ops.convert_to_tensor(mask)
        
        # Apply mask before computing loss to avoid NaN propagation
        mask_float = keras.ops.cast(mask, dtype=y_true.dtype)
        
        # Only compute squared difference on valid (non-masked) positions
        # Set masked positions to 0 in both y_true and y_pred for computation
        y_true_masked = keras.ops.where(mask, y_true, 0.0)
        y_pred_masked = keras.ops.where(mask, y_pred, 0.0)
        
        # Calculate squared difference (manual MSE)
        squared_diff = keras.ops.square(y_true_masked - y_pred_masked)  # (batch_size, n_kinases)
        
        # Apply mask to zero out invalid positions (redundant but explicit)
        masked_losses = squared_diff * mask_float
        
        # Calculate average only on non-masked values
        valid_count = keras.ops.sum(mask_float)
        total_loss = keras.ops.sum(masked_losses)
        
        # Avoid division by zero
        return keras.ops.where(valid_count > 0, total_loss / valid_count, 0.0)
    
