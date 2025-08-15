import optuna
from deepclp.models.single_kinase_cnn import SingleKinaseCNN, WANDB_AVAILABLE

if WANDB_AVAILABLE:
    import wandb

def objective(trial):
    model = SingleKinaseCNN(
        token_encoding="learnable",
        embedding_dim=trial.suggest_categorical('embedding_dim', [64, 128, 256, 512, 1024]),
        n_layers=trial.suggest_int('n_layers', 2, 5),
        kernel_size=trial.suggest_categorical('kernel_size', [4, 5, 6, 7, 8, 9]),
        n_filters=trial.suggest_categorical('n_filters', [32, 64, 128]),
        dense_layer_size=trial.suggest_categorical('dense_layer_size', [128, 256, 512, 1024, 2048]),
        dropout=trial.suggest_float('dropout', 0.0, 0.6),
        vocab_size=47,
        maxlen=85
    )
    
    config = {
        "learning_rate": trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        "epochs": 300,
        "batch_size": trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        "embedding_dim": trial.params['embedding_dim'],
        "n_layers": trial.params['n_layers'],
        "kernel_size": trial.params['kernel_size'],
        "n_filters": trial.params['n_filters'],
        "dense_layer_size": trial.params['dense_layer_size'],
        "dropout": trial.params['dropout']
    }
    
    history = model.train_vegfr2(
        "datasets/vegfr2_train.csv",
        "datasets/vegfr2_val.csv", 
        config
    )
    
    metrics = model.evaluate_vegfr2("datasets/vegfr2_val.csv")
    
    if WANDB_AVAILABLE:
        wandb.log(metrics)
        wandb.finish()
    
    return -metrics['pearson_r']  # Maximize Pearson R

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_params)
print(f"Best Pearson R: {-study.best_value:.4f}")