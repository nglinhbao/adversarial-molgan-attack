# Adversarial Mol-CycleGAN Attack Implementation

This repository implements an adversarial attack framework for molecular property prediction models using JT-VAE encoding and adversarial perturbations in latent space.

## Overview

The system generates adversarial molecular examples that can fool black-box molecular property prediction models while maintaining chemical validity and similarity to original molecules. It combines Junction Tree Variational Autoencoder (JT-VAE) for molecular encoding/decoding with adversarial training in latent space.

## Features

- **Multi-dataset Support**: ESOL (Delaney), FreeSolv, Lipophilicity, BBBP, Tox21, ToxCast, SIDER, ClinTox
- **Multi-architecture Attack**: Support for 8 different model architectures (BFGNN, GNN, GREA, GRIN, IRM, LSTM, RPGNN, SMILESTransformer)
- **Transferability Testing**: Evaluate attack success across multiple model architectures
- **Comprehensive Evaluation**: Validity, uniqueness, novelty, and similarity metrics
- **Live Training Visualization**: Real-time plotting of training progress
- **Automatic Caching**: Efficient dataset filtering and caching system

## Project Structure

```
adversarial-molgan-attack/
├── main.py                    # Main execution script
├── adversarial_generator.py   # Core adversarial generation logic
├── jtvae_wrapper.py          # JT-VAE integration wrapper
├── black_box_model.py        # Black-box model interface
├── data_loader.py            # Dataset loading and filtering
├── evaluation.py             # Comprehensive evaluation metrics
├── training_plotter.py       # Live training visualization
├── black_box_model_training/ # Model training scripts
│   ├── train.py             # Train models for different datasets
│   └── models/              # Saved trained models
├── jtvae/                   # JT-VAE submodule
└── results/                 # Training results and outputs
```

## Setup

1. **Initialize the project:**
```bash
bash setup.sh
```

2. **Install dependencies:**
```bash
pip install torch torchvision rdkit-pypi deepchem pandas numpy matplotlib scikit-learn tqdm
```

3. **Download JT-VAE pretrained models:**
```bash
# Download vocab file and model weights
mkdir -p data/zinc15
wget -O data/zinc15/vocab.txt [VOCAB_URL]
mkdir -p molvae/vae_model
wget -O molvae/vae_model/model.iter-4 [MODEL_URL]
```

## Complete Workflow

### Overview of the Attack Pipeline

The adversarial attack follows this multi-stage workflow:

```
1. Data Preparation → 2. Model Loading → 3. Adversarial Training → 4. Attack Generation → 5. Evaluation
```

### Stage 1: Data Preparation and Filtering

```python
# data_loader.py workflow
def load_molecules_from_dataset(dataset_name, max_molecules=None):
    """
    1. Load raw dataset (ESOL, FreeSolv, etc.)
    2. Filter molecules for JT-VAE compatibility
    3. Cache filtered results
    4. Return valid SMILES and properties
    """
```

**Data filtering criteria:**
- Valid RDKit molecular structure
- Molecular weight < 500 Da
- Contains only allowed atoms (C, N, O, S, P, F, Cl, Br, I)
- Successfully encodable by JT-VAE
- Valid property values (no NaN)

### Stage 2: Model Initialization

```python
# main.py workflow initialization
def main():
    """
    1. Parse command line arguments
    2. Load and filter dataset
    3. Initialize JT-VAE wrapper
    4. Load black-box target model
    5. Initialize adversarial generator
    """
```

**Component initialization:**
- **JT-VAE Wrapper**: Loads pretrained VAE for molecular encoding/decoding
- **Black-box Model**: Loads trained property prediction model
- **Adversarial Generator**: Neural network for generating latent perturbations

### Stage 3: Adversarial Training Loop

```python
# adversarial_generator.py training workflow
def train(self, training_smiles, epochs, batch_size, lambda_adv, lambda_sim, success_threshold):
    """
    For each epoch:
        For each batch:
            1. Encode molecules to latent space (JT-VAE)
            2. Get original predictions (black-box model)
            3. Generate adversarial perturbations (generator)
            4. Decode perturbed latents to molecules (JT-VAE)
            5. Get adversarial predictions (black-box model)
            6. Calculate losses:
               - Adversarial loss (maximize prediction change)
               - Similarity loss (minimize latent distance)
            7. Backpropagate and update generator
            8. Track and visualize progress
    """
```

**Training components:**
- **Adversarial Loss**: `L_adv = max(0, success_threshold - |pred_orig - pred_adv|)`
- **Similarity Loss**: `L_sim = ||z_orig - z_adv||_2`
- **Total Loss**: `L_total = λ_adv * L_adv + λ_sim * L_sim`

### Stage 4: Attack Generation

```python
# adversarial_generator.py generation workflow
def generate_adversarial(self, smiles, num_samples=1):
    """
    For each target molecule:
        1. Encode original SMILES to latent vector
        2. Generate adversarial perturbation
        3. Apply perturbation to latent vector
        4. Decode perturbed vector to adversarial SMILES
        5. Validate molecular structure
        6. Return adversarial examples
    """
```

### Stage 5: Comprehensive Evaluation

```python
# evaluation.py evaluation workflow
def evaluate_adversarial_attack(original_smiles, adversarial_smiles, 
                               original_preds, adversarial_preds, success_threshold):
    """
    Calculate attack metrics:
        1. Success Rate: fraction exceeding threshold
        2. Prediction Changes: magnitude of property changes
        3. Molecular Validity: RDKit validation
        4. Chemical Similarity: Tanimoto coefficients
        5. Uniqueness: fraction of unique molecules
        6. Novelty: fraction not in training data
    """
```

### Stage 6: Transferability Testing

```python
# main.py transferability workflow
def test_transferability():
    """
    For each available model architecture:
        1. Load different trained model
        2. Apply same adversarial molecules
        3. Calculate success rates
        4. Measure cross-architecture effectiveness
    """
```

## Detailed Code Workflow

### 1. Main Execution Flow (`main.py`)

```python
def main():
    # 1. Parse arguments and setup
    args = parse_arguments()
    setup_directories()
    
    # 2. Load and prepare data
    training_smiles, training_properties = load_molecules_from_dataset(args.dataset)
    test_smiles, test_properties = split_test_data(training_smiles, training_properties)
    
    # 3. Initialize models
    jtvae = JTVAEWrapper()
    black_box_model = BlackBoxModel(get_blackbox_model_path(args.dataset, args.model_architecture))
    adversarial_model = AdversarialMolCycleGAN(jtvae, black_box_model, args.learning_rate)
    
    # 4. Train adversarial generator
    adversarial_model.train(
        training_smiles=training_smiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_adv=args.lambda_adv,
        lambda_sim=args.lambda_sim,
        success_threshold=args.success_threshold
    )
    
    # 5. Generate adversarial examples
    adversarial_results = []
    for smiles in test_smiles:
        adversarial_molecules = adversarial_model.generate_adversarial(smiles, num_samples=5)
        adversarial_results.extend(adversarial_molecules)
    
    # 6. Evaluate attack effectiveness
    evaluation_results = evaluate_adversarial_attack(
        test_smiles, adversarial_results, 
        original_predictions, adversarial_predictions,
        args.success_threshold
    )
    
    # 7. Test transferability across models
    transferability_results = test_transferability(adversarial_results)
    
    # 8. Save results and generate reports
    save_results(evaluation_results, transferability_results)
```

### 2. Adversarial Generator Training (`adversarial_generator.py`)

```python
def train_step(self, smiles_list, lambda_adv, lambda_sim, epoch, success_threshold):
    # Forward pass
    original_z = self.jtvae.encode_smiles_list(smiles_list)  # [batch_size, latent_dim]
    original_pred = self.black_box_model.predict_from_latent(original_z)
    
    # Generate adversarial examples
    adversarial_z = self.generator(original_z)  # Apply learned perturbations
    adversarial_smiles = self.jtvae.decode_from_numpy(adversarial_z.detach().cpu().numpy())
    adversarial_pred = self.black_box_model.predict_from_smiles(adversarial_smiles)
    
    # Calculate losses
    validity_scores = self.check_molecule_validity(adversarial_smiles)
    adv_loss = self.adversarial_loss(original_pred, adversarial_pred, success_threshold)
    sim_loss = self.similarity_loss(original_z, adversarial_z)
    
    total_loss = lambda_adv * adv_loss + lambda_sim * sim_loss
    
    # Backward pass
    self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()
    
    return total_loss, adv_loss, sim_loss, validity_scores.mean()
```

### 3. JT-VAE Integration (`jtvae_wrapper.py`)

```python
class JTVAEWrapper:
    def encode_smiles_list(self, smiles_list):
        """Convert SMILES to latent vectors"""
        latent_vectors = []
        for smiles in smiles_list:
            mol_tree = MolTree(smiles)
            latent_vec = self.model.encode(mol_tree)
            latent_vectors.append(latent_vec)
        return torch.stack(latent_vectors)
    
    def decode_from_numpy(self, latent_array):
        """Convert latent vectors back to SMILES"""
        decoded_smiles = []
        for latent_vec in latent_array:
            try:
                smiles = self.model.decode(latent_vec)
                decoded_smiles.append(smiles)
            except:
                decoded_smiles.append(None)  # Invalid molecule
        return decoded_smiles
```

### 4. Black-box Model Interface (`black_box_model.py`)

```python
class BlackBoxModel:
    def predict_from_smiles(self, smiles_list):
        """Get property predictions from SMILES"""
        features = [self.smiles_to_features(smiles) for smiles in smiles_list]
        features_tensor = torch.stack([f for f in features if f is not None])
        
        with torch.no_grad():
            predictions = self.model(features_tensor)
        
        return predictions
    
    def smiles_to_features(self, smiles):
        """Convert SMILES to molecular descriptors/fingerprints"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Extract features (fingerprints, descriptors, etc.)
        return self.feature_extractor(mol)
```

### 5. Real-time Training Visualization (`training_plotter.py`)

```python
class TrainingPlotter:
    def update_plots(self, epoch, total_loss, adv_loss, sim_loss, validity_rate, success_rate):
        """Update live training plots"""
        self.losses['total'].append(total_loss)
        self.losses['adversarial'].append(adv_loss)
        self.losses['similarity'].append(sim_loss)
        self.metrics['validity'].append(validity_rate)
        self.metrics['success'].append(success_rate)
        
        # Refresh matplotlib plots
        self.update_loss_plot()
        self.update_metrics_plot()
        plt.pause(0.01)  # Non-blocking plot update
```

## Usage

### 1. Train Black-box Models (Required)

```bash
cd black_box_model_training
python train.py --dataset delaney --model BFGNN
python train.py --dataset delaney --model GNN
# ... train additional models for transferability testing
```

### 2. Run Adversarial Attack

Basic usage:
```bash
python main.py --dataset delaney --model_architecture BFGNN
```

Advanced usage:
```bash
python main.py \
    --dataset delaney \
    --model_architecture BFGNN \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --lambda_adv 1.0 \
    --lambda_sim 0.01 \
    --success_threshold 0.2 \
    --max_train_molecules 1000 \
    --max_test_molecules 100
```

### 3. Available Datasets

| Dataset | Task Type | Property | Molecules |
|---------|-----------|----------|-----------|
| delaney | Regression | Aqueous solubility | 1,128 |
| freesolv | Regression | Hydration free energy | 643 |
| lipophilicity | Regression | Lipophilicity | 4,200 |
| bbbp | Classification | Blood-brain barrier | 2,050 |
| tox21 | Classification | Toxicity (12 targets) | 7,831 |
| toxcast | Classification | Toxicity (617 targets) | 8,576 |
| sider | Classification | Side effects (27 targets) | 1,427 |
| clintox | Classification | Clinical toxicity | 1,484 |

### 4. Model Architectures

- **BFGNN**: Bidirectional Feature Graph Neural Network
- **GNN**: Standard Graph Neural Network  
- **GREA**: Graph Representation Enhanced Attention
- **GRIN**: Graph Interaction Network
- **IRM**: Invariant Risk Minimization
- **LSTM**: Long Short-Term Memory
- **RPGNN**: Random Path Graph Neural Network
- **SMILESTransformer**: Transformer on SMILES strings

## Evaluation Metrics

### Attack Success Metrics
- **Success Rate**: Fraction of attacks exceeding success threshold
- **Average Prediction Change**: Mean absolute change in property predictions
- **Transferability**: Attack success across different model architectures

### Molecular Quality Metrics
- **Validity**: Fraction of chemically valid molecules (RDKit validation)
- **Uniqueness**: Fraction of unique adversarial molecules
- **Novelty**: Fraction of molecules not seen in training data
- **Similarity**: Average Tanimoto similarity to original molecules

## Results

Training results are saved to timestamped directories in `results/`:

```
results/delaney_BFGNN_2025-07-01_14-30-15/
├── training_progress.png              # Live training plots
├── adversarial_results_delaney_BFGNN.csv  # Detailed results
├── evaluation_metrics_delaney_BFGNN.csv   # Summary metrics
├── comprehensive_statistics.csv       # Dataset statistics
├── generator_model.pth                # Trained generator
├── generator_checkpoint_epoch_*.pth   # Training checkpoints
└── model_summary.txt                  # Model configuration
```

## Example Output

```
=== Adversarial Mol-CycleGAN Implementation ===
Dataset: DELANEY
Model Architecture: BFGNN
Task type: regression

=== Training Progress ===
Epoch 10/50: Loss=0.234, Adv=0.156, Sim=0.078, Valid=0.95, Success=0.67
Epoch 20/50: Loss=0.189, Adv=0.134, Sim=0.055, Valid=0.97, Success=0.73
Epoch 30/50: Loss=0.167, Adv=0.120, Sim=0.047, Valid=0.98, Success=0.76

=== Attack Success Metrics ===
Success Rate: 0.750
Successful Attacks: 15/20
Average Prediction Change: 1.243
Average Tanimoto Similarity: 0.842

=== Molecular Quality Metrics ===
Validity: 0.950
Uniqueness: 0.900
Novelty: 0.800

=== Transferability Results ===
GNN:
  Success Rate: 0.650 (13/20)
  Avg Prediction Change: 1.156
LSTM:
  Success Rate: 0.400 (8/20)
  Avg Prediction Change: 0.892
```

## Utility Commands

List available datasets:
```bash
python main.py --list_datasets
```

List trained models:
```bash
python main.py --list_models
```

Clean cached datasets:
```bash
python main.py --clean_datasets all
python main.py --clean_datasets delaney
```

##