# CNN Hyperparameter Optimization for Kinase Bioactivity Prediction

## 📋 Overview

This project implements CNN-based models for predicting kinase bioactivity using three different training approaches:

1. **Finetuning Only**: Train CNN on small finetuning dataset (645 molecules)
2. **Pretraining Only**: Train CNN on large pretraining dataset (79,492 molecules)  
3. **Transfer Learning**: Pretrain on large dataset, then finetune on small dataset

## 📊 Dataset Information

- **Pretraining Dataset**: `datasets/chembl_pretraining.csv` (79,492 molecules)
- **Finetuning Dataset**: `datasets/pkis2_finetuning.csv` (645 molecules)
- **Common Kinases**: 355 kinase targets present in both datasets
- **Input**: SMILES molecular representations
- **Output**: Bioactivity values (regression task)

## 🎛️ Hyperparameter Search Space

As requested, the optimization explores these specific ranges:

- **Number of layers**: [1, 2, 3]
- **Dropout**: 0.25 (fixed)
- **Batch size**: 32 (fixed)
- **Number of filters**: [32, 64, 128]
- **Kernel length**: [3, 5, 7]
- **Learning rate**: [10⁻², 10⁻³, 5×10⁻³, 10⁻⁴, 5×10⁻⁵]

## 🚀 Quick Start

### 1. Install Dependencies

First, make sure you have the required packages:

```bash
pip install optuna matplotlib seaborn pandas numpy scikit-learn
```

### 2. Run Hyperparameter Optimization

```bash
python cnn_hyperparameter_optimization.py
```

Choose from the menu:
- `1`: Optimize finetuning-only scenario
- `2`: Optimize pretraining-only scenario  
- `3`: Optimize transfer learning scenario
- `4`: Optimize all scenarios

**Recommended**: Start with option `4` to compare all approaches.

### 3. Visualize Results

After optimization is complete:

```bash
python visualize_results.py
```

This creates comparison plots and detailed analysis of your results.

## 📁 File Structure

```
📦 CNN Optimization Project
├── 📄 cnn_hyperparameter_optimization.py  # Main optimization script
├── 📄 visualize_results.py                # Results visualization  
├── 📄 analyze_kinase_coverage.py          # Dataset analysis
├── 📄 quick_analysis.py                   # Quick dataset stats
├── 📂 datasets/
│   ├── 📄 chembl_pretraining.csv         # Large pretraining dataset
│   └── 📄 pkis2_finetuning.csv           # Small finetuning dataset
└── 📂 cnn_optimization_results/           # Results directory (created automatically)
    ├── 📄 cnn_scenario_1_results.json    # Scenario 1 best parameters
    ├── 📄 cnn_scenario_1_study.pkl       # Scenario 1 optimization history
    ├── 📄 cnn_scenario_2_results.json    # Scenario 2 best parameters
    ├── 📄 cnn_scenario_2_study.pkl       # Scenario 2 optimization history
    ├── 📄 cnn_scenario_3_results.json    # Scenario 3 best parameters
    ├── 📄 cnn_scenario_3_study.pkl       # Scenario 3 optimization history
    └── 📊 *.png                          # Visualization plots
```

## 🧠 Understanding the Models

### For Beginners: What Each Scenario Does

**Scenario 1: Finetuning Only**
- Trains a CNN from scratch using only the small dataset (645 molecules)
- Pros: Simple, fast training
- Cons: Limited data may lead to overfitting

**Scenario 2: Pretraining Only**  
- Trains a CNN using only the large dataset (79,492 molecules)
- Pros: Lots of training data, good generalization
- Cons: May not be optimized for your specific task

**Scenario 3: Transfer Learning**
- First trains on large dataset (pretraining)
- Then fine-tunes on small dataset (adaptation to specific task)
- Pros: Best of both worlds - learns general patterns then specializes
- Cons: More complex, takes longer to train

### CNN Architecture

The CNN model consists of:

1. **Embedding Layer**: Converts SMILES characters to dense vectors
2. **Convolutional Layers**: Extract molecular patterns (1-3 layers)
3. **Batch Normalization**: Stabilizes training
4. **Max Pooling**: Reduces dimensionality
5. **Global Max Pooling**: Summarizes features
6. **Dense Layers**: Final prediction layers
7. **Output Layer**: 355 kinase activity predictions

## 📈 Expected Results

Transfer learning (Scenario 3) typically performs best because:
- It learns general molecular patterns from the large dataset
- Then adapts to the specific kinases in the small dataset
- Combines the benefits of both large-scale and task-specific learning

## 🎯 Next Steps After Optimization

1. **Check Results**: Look at the JSON files to see best hyperparameters
2. **Compare Scenarios**: Use the visualization script to understand performance differences
3. **Train Final Model**: Use the best hyperparameters to train your production model
4. **Validate Results**: Test on held-out data to confirm performance

## 💡 Tips for Beginners

- **Start Small**: Begin with 10-20 trials per scenario to get quick results
- **Monitor Progress**: The script shows real-time optimization progress
- **Understand Trade-offs**: Lower MSE = better predictions
- **Be Patient**: Transfer learning takes longer but often gives best results
- **Save Results**: All results are automatically saved for later analysis

## 🔧 Customization

To modify the search space, edit the `_create_cnn_model` method in `cnn_hyperparameter_optimization.py`:

```python
# Current hyperparameter suggestions
n_layers = trial.suggest_categorical('n_layers', [1, 2, 3])
n_filters = trial.suggest_categorical('n_filters', [32, 64, 128])  
kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.001, 0.005, 0.0001, 0.00005])
```

## 📚 Further Reading

- [DeepCLP Documentation](./README.md)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Transfer Learning Guide](https://keras.io/guides/transfer_learning/)

## 🆘 Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce batch size or model size
2. **Slow Training**: Reduce number of trials or use GPU
3. **Poor Results**: Try different hyperparameter ranges
4. **Import Errors**: Install missing packages with pip

**Getting Help:**

If you encounter issues, check:
1. Are all dependencies installed?
2. Are the dataset files in the correct location?
3. Do you have enough disk space for results?

---

🎉 **Happy Optimizing!** Remember: machine learning is about experimentation and learning from results!