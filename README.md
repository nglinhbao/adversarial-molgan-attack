# Adversarial Mol-CycleGAN Implementation

This repository implements the Adversarial Mol-CycleGAN method for generating adversarial molecular examples that can fool black-box molecular property prediction models.

## Features

- Integration with JT-VAE for molecular encoding/decoding (following mol-cycle-gan approach)
- Support for both classification and regression black-box models
- Adversarial training in latent space
- Comprehensive evaluation metrics
- Visualization tools for results analysis

## Setup

1. **Run the setup script:**
```bash
bash setup.sh
```

2. **Download JT-VAE pretrained models:**
```bash
# You'll need to obtain the pretrained JT-VAE model
# Place it in jtvae/molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4
```

3. **Prepare your black-box model:**
   - Place your PyTorch model file in `models/blackbox_model.pt`
   - Ensure it accepts molecular descriptors as input
   - Update the `smiles_to_features` method in `black_box_model.py` if needed

## Usage

### Basic Usage

```python
from jtvae_wrapper import JTVAEWrapper
from black_box_model import DummyBlackBoxModel
from adversarial_generator import AdversarialMolCycleGAN

# Initialize components
jtvae = JTVAEWrapper()
black_box_model = DummyBlackBoxModel()
adversarial_model = AdversarialMolCycleGAN(jtvae, black_box_model)

# Train
training_smiles = ["CCO", "CCN", "CCC"]
adversarial_model.train(training_smiles, epochs=50)

# Generate adversarial examples
target_smiles = "CCO"
adversarial_molecules = adversarial_model.generate_adversarial(target_smiles, num_samples=5)
```

### Running the Full Pipeline

```bash
python main.py
```

## Configuration

Modify `config.py` to adjust:
- Model architectures
- Training hyperparameters
- File paths
- Evaluation thresholds

## Requirements

- Python 3.7+
- PyTorch 1.9+
- RDKit
- NumPy, SciPy, Matplotlib, Pandas
- scikit-learn

## License

This implementation is provided for research purposes.
