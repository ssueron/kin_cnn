#!/usr/bin/env python3
"""
Script principal pour lancer l'entraînement final optimisé du modèle MultiKinaseCNN.
Combine toutes les optimisations et permet de choisir entre différentes configurations.
"""

import os
import sys
import argparse
from datetime import datetime
import json

# Configuration des variables d'environnement
os.environ['KERAS_BACKEND'] = 'torch'

# Imports des modules d'entraînement
from optimized_kinase_training import AdvancedKinaseTrainer
from memory_optimized_training import MemoryOptimizedTrainer
from advanced_evaluation import AdvancedKinaseEvaluator


def setup_training_environment():
    """Configure l'environnement d'entraînement."""
    print("🔧 Configuration de l'environnement d'entraînement...")
    
    # Détection GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU détecté: {gpu_name} ({gpu_memory:.1f} GB)")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            print("   Utilisation du CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    except ImportError:
        print("   PyTorch non disponible, utilisation du CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Vérification des fichiers de données
    required_files = [
        "processed_data/train_data.csv",
        "processed_data/train_mask.csv", 
        "processed_data/val_data.csv",
        "processed_data/val_mask.csv",
        "processed_data/test_data.csv",
        "processed_data/test_mask.csv",
        "data/smiles_vocab_extended.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Fichiers manquants: {missing_files}")
        return False
    
    print("✅ Tous les fichiers requis sont présents")
    return True


def choose_training_mode():
    """Permet de choisir le mode d'entraînement."""
    print("\n🎯 CHOIX DU MODE D'ENTRAÎNEMENT")
    print("=" * 50)
    print("1. 🚀 Mode Performance  - Entraînement optimisé avec toutes les fonctionnalités")
    print("2. 💾 Mode Mémoire     - Entraînement optimisé pour systèmes à mémoire limitée")
    print("3. ⚡ Mode Rapide      - Entraînement rapide pour tests (paramètres réduits)")
    print("4. 🔬 Mode Recherche   - Grid search automatique des hyperparamètres")
    
    while True:
        try:
            choice = int(input("\nVotre choix (1-4): "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("❌ Choix invalide. Entrez un nombre entre 1 et 4.")
        except ValueError:
            print("❌ Entrée invalide. Entrez un nombre.")


def get_training_parameters(mode):
    """Retourne les paramètres d'entraînement selon le mode choisi."""
    configs = {
        1: {  # Mode Performance
            'name': 'Performance',
            'epochs': 200,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'patience': 25,
            'model_size': 'large'
        },
        2: {  # Mode Mémoire  
            'name': 'Mémoire',
            'epochs': 150,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'patience': 20,
            'model_size': 'medium'
        },
        3: {  # Mode Rapide
            'name': 'Rapide',
            'epochs': 50,
            'learning_rate': 2e-3,
            'batch_size': 128,
            'patience': 10,
            'model_size': 'small'
        },
        4: {  # Mode Recherche
            'name': 'Recherche',
            'epochs': 100,
            'learning_rate': 'auto',
            'batch_size': 'auto',
            'patience': 15,
            'model_size': 'auto'
        }
    }
    
    return configs[mode]


def run_performance_mode(config):
    """Lance l'entraînement en mode performance."""
    print(f"\n🚀 LANCEMENT EN MODE {config['name'].upper()}")
    print("=" * 60)
    
    # Utiliser le trainer avancé
    trainer = AdvancedKinaseTrainer()
    
    # Adapter la configuration
    trainer.optimized_params['training']['epochs'] = config['epochs']
    trainer.optimized_params['training']['learning_rate'] = config['learning_rate']
    trainer.optimized_params['training']['batch_size'] = config['batch_size']
    trainer.optimized_params['training']['patience'] = config['patience']
    
    # Lancer l'entraînement
    model, history, model_path = trainer.run_complete_training()
    
    return model, history, model_path, trainer.optimized_params


def run_memory_mode(config):
    """Lance l'entraînement en mode mémoire optimisée."""
    print(f"\n💾 LANCEMENT EN MODE {config['name'].upper()}")
    print("=" * 60)
    
    # Utiliser le trainer optimisé mémoire
    trainer = MemoryOptimizedTrainer()
    
    # Adapter la configuration
    trainer.memory_config['batch_size'] = config['batch_size']
    
    # Lancer l'entraînement
    model, history, model_path = trainer.run_optimized_training(
        epochs=config['epochs'],
        learning_rate=config['learning_rate']
    )
    
    return model, history, model_path, trainer.model_config


def run_fast_mode(config):
    """Lance l'entraînement en mode rapide."""
    print(f"\n⚡ LANCEMENT EN MODE {config['name'].upper()}")
    print("=" * 60)
    
    # Utiliser le trainer avancé avec paramètres réduits
    trainer = AdvancedKinaseTrainer()
    
    # Configuration allégée pour mode rapide
    trainer.optimized_params['model']['embedding_dim'] = 64
    trainer.optimized_params['model']['n_layers'] = 2
    trainer.optimized_params['model']['n_filters'] = 32
    trainer.optimized_params['model']['dense_layer_size'] = 256
    
    trainer.optimized_params['training']['epochs'] = config['epochs']
    trainer.optimized_params['training']['learning_rate'] = config['learning_rate']
    trainer.optimized_params['training']['batch_size'] = config['batch_size']
    trainer.optimized_params['training']['patience'] = config['patience']
    
    # Lancer l'entraînement
    model, history, model_path = trainer.run_complete_training()
    
    return model, history, model_path, trainer.optimized_params


def run_research_mode(config):
    """Lance la recherche automatique d'hyperparamètres."""
    print(f"\n🔬 LANCEMENT EN MODE {config['name'].upper()}")
    print("=" * 60)
    print("🔍 Grid search des hyperparamètres...")
    
    # Définir la grille de recherche
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3],
        'batch_size': [32, 64, 128],
        'dropout': [0.3, 0.4, 0.5],
        'embedding_dim': [96, 128, 160],
        'n_filters': [48, 64, 80]
    }
    
    best_score = float('inf')
    best_config = None
    best_model_path = None
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"🎯 {total_combinations} combinaisons à tester...")
    
    combination = 0
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for dropout in param_grid['dropout']:
                for emb_dim in param_grid['embedding_dim']:
                    for n_filters in param_grid['n_filters']:
                        combination += 1
                        print(f"\n🧪 Test {combination}/{total_combinations}: LR={lr}, BS={bs}, Dropout={dropout}, Emb={emb_dim}, Filters={n_filters}")
                        
                        # Créer trainer avec cette config
                        trainer = AdvancedKinaseTrainer()
                        
                        # Appliquer les paramètres
                        trainer.optimized_params['training']['learning_rate'] = lr
                        trainer.optimized_params['training']['batch_size'] = bs
                        trainer.optimized_params['training']['epochs'] = config['epochs']
                        trainer.optimized_params['training']['patience'] = config['patience']
                        trainer.optimized_params['model']['dropout'] = dropout
                        trainer.optimized_params['model']['embedding_dim'] = emb_dim
                        trainer.optimized_params['model']['n_filters'] = n_filters
                        
                        try:
                            # Entraînement
                            model, history, model_path = trainer.run_complete_training()
                            
                            # Évaluer la performance
                            final_val_loss = history['val_loss'][-1]
                            
                            if final_val_loss < best_score:
                                best_score = final_val_loss
                                best_config = {
                                    'learning_rate': lr,
                                    'batch_size': bs, 
                                    'dropout': dropout,
                                    'embedding_dim': emb_dim,
                                    'n_filters': n_filters
                                }
                                best_model_path = model_path
                                
                                print(f"✅ Nouveau meilleur score: {best_score:.6f}")
                            else:
                                print(f"📊 Score: {final_val_loss:.6f} (meilleur: {best_score:.6f})")
                                
                        except Exception as e:
                            print(f"❌ Erreur lors de l'entraînement: {e}")
                            continue
    
    print(f"\n🏆 MEILLEURE CONFIGURATION TROUVÉE:")
    print(f"   Score (val_loss): {best_score:.6f}")
    print(f"   Paramètres: {best_config}")
    print(f"   Modèle: {best_model_path}")
    
    return None, None, best_model_path, best_config


def run_evaluation(model_path, results_path=None):
    """Lance l'évaluation du modèle entraîné."""
    print(f"\n📊 ÉVALUATION DU MODÈLE")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return None, None
    
    try:
        evaluator = AdvancedKinaseEvaluator(model_path, results_path)
        kinase_metrics, overall_metrics = evaluator.run_complete_evaluation()
        return kinase_metrics, overall_metrics
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation: {e}")
        return None, None


def save_session_summary(mode_config, model_path, training_params, evaluation_results=None):
    """Sauvegarde un résumé complet de la session d'entraînement."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'session_timestamp': timestamp,
        'training_mode': mode_config['name'],
        'training_config': mode_config,
        'model_path': model_path,
        'training_parameters': training_params,
        'evaluation_results': evaluation_results,
        'system_info': {
            'python_version': sys.version,
            'has_cuda': os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
        }
    }
    
    summary_path = f"training_session_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📋 Résumé de session sauvegardé: {summary_path}")
    return summary_path


def main():
    """Point d'entrée principal."""
    print("🧬 ENTRAÎNEMENT FINAL MULTIKINASECNN")
    print("=" * 60)
    print("🎯 Prédiction d'activité biomoléculaire multi-kinases")
    print("📊 Dataset: 79,492 molécules × 357 kinases (sparsité 0.43%)")
    print("=" * 60)
    
    # Configuration de l'environnement
    if not setup_training_environment():
        print("❌ Échec de la configuration. Arrêt.")
        return
    
    # Choix du mode d'entraînement
    mode = choose_training_mode()
    config = get_training_parameters(mode)
    
    print(f"\n📋 Configuration sélectionnée: {config}")
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Lancement de l'entraînement selon le mode
    model = None
    history = None
    model_path = None
    training_params = None
    
    try:
        if mode == 1:  # Performance
            model, history, model_path, training_params = run_performance_mode(config)
        elif mode == 2:  # Mémoire
            model, history, model_path, training_params = run_memory_mode(config)
        elif mode == 3:  # Rapide
            model, history, model_path, training_params = run_fast_mode(config)
        elif mode == 4:  # Recherche
            model, history, model_path, training_params = run_research_mode(config)
        
        if model_path is None:
            print("❌ Échec de l'entraînement")
            return
            
        print(f"\n✅ ENTRAÎNEMENT RÉUSSI!")
        print(f"📁 Modèle sauvegardé: {model_path}")
        
        # Évaluation automatique
        print("\n🤔 Lancer l'évaluation du modèle? (y/n): ", end="")
        if input().lower() == 'y':
            # Trouver le fichier de résultats correspondant
            results_path = model_path.replace('.weights.h5', '').replace('model', 'results') + '.json'
            if not os.path.exists(results_path):
                results_path = None
                
            evaluation_results = run_evaluation(model_path, results_path)
        else:
            evaluation_results = None
        
        # Sauvegarder le résumé de session
        summary_path = save_session_summary(config, model_path, training_params, evaluation_results)
        
        print(f"\n🎉 SESSION TERMINÉE AVEC SUCCÈS!")
        print(f"📁 Fichiers générés:")
        print(f"   Modèle: {model_path}")
        print(f"   Résumé: {summary_path}")
        
        if evaluation_results:
            print(f"📊 Performance globale: R² = {evaluation_results[1]['r2']:.4f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()