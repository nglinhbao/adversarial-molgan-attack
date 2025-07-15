# Adversarial Mol-CycleGAN Attack

This repository implements adversarial attacks on molecular property prediction models using JT-VAE encoding and adversarial perturbations in latent space. The system generates adversarial molecular examples that can fool black-box models while maintaining chemical validity and similarity to original molecules.

## Features

- **Multi-dataset Support**: ESOL (Delaney), FreeSolv, Lipophilicity, BBBP, Tox21, ToxCast, SIDER, ClinTox
- **Multi-architecture Attack**: BFGNN, GNN, GREA, GRIN, IRM, LSTM, RPGNN, SMILESTransformer
- **Transferability Testing**: Evaluate attack success across multiple model architectures
- **Comprehensive Evaluation**: Validity, uniqueness, novelty, and similarity metrics

## Quick Start

### 1. Setup

```bash
conda create -n mol_adv python=3.10 -y
conda activate mol_adv
./setup.sh
```

### 2. Train Black-box Models

First, train the target models that will be attacked:

```bash
cd black_box_model_training
python train.py --dataset delaney
```

Trained models are saved to models:
```
black_box_model_training/models/
├── delaney_BFGNNMolecularPredictor.pt
├── delaney_results.csv
├── bace_results.csv
└── bbbp_results.csv
```

### 3. Run Adversarial Attack

```bash
python main.py --dataset delaney --model_architecture BFGNN
```

**Optional parameters:**
```bash
python main.py \
    --dataset delaney \
    --model_architecture BFGNN \
    --epochs 50 \
    --batch_size 8 \
    --success_threshold 0.2
```

### 3. Results

Results are saved to timestamped directories:
```
results/
├── delaney_BFGNN_2025-07-01_20-22-50/
├── delaney_BFGNN_2025-07-02_17-03-18/
│   ├── model_summary.txt
│   ├── training_progress.png
│   ├── training_results.csv
│   ├── adversarial_results_delaney_BFGNN.csv
│   └── evaluation_report.txt
└── delaney_BFGNN_2025-07-05_10-17-50/
```

## Available Datasets

| Dataset | Task | Property |
|---------|------|----------|
| delaney | Regression | Aqueous solubility |
| bbbp | Classification | Blood-brain barrier |
| qm7 | Regression | Atomization energy |
| bace | Classification | BACE-1 inhibition |

## Model Architectures

- BFGNN, GNN, GREA, GRIN, IRM, LSTM, RPGNN, SMILESTransformer

## Utility Commands

```bash
# List available datasets
python main.py --list_datasets

# List trained models
python main.py --list_models --dataset delaney

# Clean cached data
python main.py --clean_datasets all
```
```

This version is much more concise while still covering the essential information users need to get started and understand what the repository does.This version is much more concise while still covering the essential information users need to get started and understand what the repository does.