from deepclp.models.single_kinase_cnn import SingleKinaseCNN, WANDB_AVAILABLE

print(f"W&B available: {WANDB_AVAILABLE}")
if WANDB_AVAILABLE:
    import wandb
    try:
        user = wandb.api.default_entity
        print(f"W&B authenticated user: {user}")
    except:
        print("W&B authentication check failed")

model = SingleKinaseCNN(
    token_encoding="learnable",
    embedding_dim=64,
    n_layers=2,
    kernel_size=5,
    n_filters=64,
    dense_layer_size=128,
    dropout=0.25,
    vocab_size=47,
    maxlen=85
)

config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "embedding_dim": 64,
    "n_layers": 2,
    "kernel_size": 5,
    "n_filters": 64,
    "dense_layer_size": 128,
    "dropout": 0.25
}

history = model.train_vegfr2(
    "datasets/vegfr2_train.csv",
    "datasets/vegfr2_val.csv",
    config
)