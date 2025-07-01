"""Configuration file for Adversarial Mol-CycleGAN - following mol-cycle-gan structure"""

# JT-VAE Configuration (mol-cycle-gan style)
JTVAE_CONFIG = {
    'jtvae_path': './jtvae/',
    'vocab_path': 'data/zinc/vocab.txt',
    'model_path': 'molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4',
    'hidden_size': 450,
    'latent_size': 56,
    'depth': 3
}

# Black-box Model Configuration
BLACKBOX_CONFIG = {
    'model_path': './models/blackbox_model.pt',
    'model_type': 'classification',  # or 'regression'
    'scaler_path': './models/scaler.pkl'
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'lambda_adv': 1.0,
    'lambda_sim': 0.1,
    'patience': 20
}

# Generator Architecture
GENERATOR_CONFIG = {
    'hidden_size': 256,
    'num_layers': 3,
    'dropout': 0.1
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'similarity_threshold': 0.8,
    'prediction_threshold': 0.1,
    'num_samples': 5
}

# File Paths (mol-cycle-gan structure)
PATHS = {
    'models_dir': './models',
    'data_dir': './data',
    'results_dir': './results',
    'jtvae_repo': './jtvae'
}
