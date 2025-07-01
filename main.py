import os
import numpy as np
import pandas as pd
from rdkit import Chem
import deepchem as dc
from jtvae_wrapper import JTVAEWrapper
from black_box_model import BlackBoxModel
from adversarial_generator import AdversarialMolCycleGAN
from evaluation import evaluate_adversarial_attack, plot_evaluation_results
from datetime import datetime

def _to_xy(ds):
    """Return (list_of_smiles, numpy_targets) for a DeepChem Dataset."""
    smiles = [s.replace(" ", "") for s in ds.ids]  # Remove spaces from SMILES
    y = ds.y
    # Ensure 2‑D targets (n_samples, n_tasks)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return smiles, y

def load_delaney(root="data/delaney"):
    """Load Delaney (regression dataset) via DeepChem."""
    tasks, (tr, va, te), _ = dc.molnet.load_delaney(
        featurizer="Raw",        # keeps original SMILES in ds.ids
        splitter="random",       # 80/10/10 split below
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        data_dir=root,
        reload=False,
    )
    # Delaney has 1 regression task by default – keep it.
    return tasks, *map(_to_xy, (tr, va, te))

def load_molecules_from_delaney(subset, max_molecules=None, jtvae_wrapper=None):
    """Load molecules from Delaney dataset for adversarial training"""
    print(f"Loading Delaney dataset...")
    tasks, train, valid, test = load_delaney()
    
    print(f"Delaney dataset loaded:")
    print(f"  Tasks: {tasks}")
    print(f"  Train: {len(train[0])} molecules")
    print(f"  Valid: {len(valid[0])} molecules") 
    print(f"  Test: {len(test[0])} molecules")
    
    # Select subset
    if subset == 'train':
        smiles_list, targets = train
    elif subset == 'valid':
        smiles_list, targets = valid
    elif subset == 'test':
        smiles_list, targets = test
    else:
        # Combine all for more diversity
        all_smiles = train[0] + valid[0] + test[0]
        all_targets = np.vstack([train[1], valid[1], test[1]])
        smiles_list, targets = all_smiles, all_targets
    
    # Limit number of molecules if specified (before filtering to have more candidates)
    if max_molecules and len(smiles_list) > max_molecules * 3:  # Get 3x more for filtering
        indices = np.random.choice(len(smiles_list), max_molecules * 3, replace=False)
        smiles_list = [smiles_list[i] for i in indices]
        targets = targets[indices]

    print(f"Selected {len(smiles_list)} molecules from {subset} subset for filtering")
    
    # Filter out invalid SMILES and molecules that JT-VAE cannot process
    valid_smiles = []
    valid_targets = []
    jtvae_processable_count = 0
    
    for i, smiles in enumerate(smiles_list):
        # First check RDKit validity
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        # If JT-VAE wrapper is provided, test encoding/decoding
        if jtvae_wrapper is not None:
            try:
                # Test encoding
                latent_array = jtvae_wrapper.encode_to_numpy([smiles])
                if latent_array.size == 0:
                    continue
                
                # Test decoding
                decoded_smiles = jtvae_wrapper.decode_from_numpy(latent_array)
                if not decoded_smiles or decoded_smiles[0] is None:
                    continue
                
                # Molecule is JT-VAE processable
                jtvae_processable_count += 1
                valid_smiles.append(smiles)
                valid_targets.append(targets[i])
                
                # Stop if we have enough molecules
                if max_molecules and len(valid_smiles) >= max_molecules:
                    break
                    
            except Exception as e:
                # Skip molecules that cause JT-VAE errors
                continue
        else:
            # If no JT-VAE wrapper provided, just use RDKit filtering
            valid_smiles.append(smiles)
            valid_targets.append(targets[i])
    
    if jtvae_wrapper is not None:
        print(f"JT-VAE filtering results:")
        print(f"  RDKit valid molecules tested: {i+1}")
        print(f"  JT-VAE processable molecules: {jtvae_processable_count}")
        print(f"  Final dataset: {len(valid_smiles)} molecules")
    else:
        print(f"RDKit filtering results:")
        print(f"  Final dataset: {len(valid_smiles)} valid molecules")
    
    # Show some examples
    print("Example molecules:")
    for i, smiles in enumerate(valid_smiles[:5]):
        target_value = valid_targets[i][0] if len(valid_targets[i]) > 0 else "N/A"
        print(f"  {i+1}. {smiles} (solubility: {target_value:.3f})")
    
    return valid_smiles, np.array(valid_targets)

def main():
    print("=== Adversarial Mol-CycleGAN Implementation ===")
    print("Loading molecules from DeepChem Delaney dataset")
    
    # Create timestamped results directory name
    results_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join('results', results_name)
    os.makedirs(results_path, exist_ok=True)
    print(f"Results will be saved to: {results_path}")
    
    # Configuration following mol-cycle-gan structure
    jtvae_path = "./jtvae/"
    vocab_path = "data/zinc15/vocab.txt"
    model_path = "molvae/vae_model/model.iter-4"
    blackbox_model_path = "black_box_model_training/models/delaney_BFGNNMolecularPredictor.pt"
    
    # Initialize JT-VAE wrapper using mol-cycle-gan approach
    print("Initializing JT-VAE wrapper (mol-cycle-gan style)...")
    jtvae = JTVAEWrapper(
        jtvae_path=jtvae_path,
        hidden_size=450,
        latent_size=56,
        depth=3,
        jtnn_model_path=model_path,
        vocab_path=vocab_path
    )
    
    print("Initializing black-box model...")
    black_box_model = BlackBoxModel(model_path=blackbox_model_path, model_type='regression')
    
    print("Initializing adversarial generator...")
    adversarial_model = AdversarialMolCycleGAN(
        jtvae_wrapper=jtvae,
        black_box_model=black_box_model,
        results_name=results_name  # Pass the results directory name
    )
    
    # Load training data from Delaney dataset with JT-VAE filtering
    print("\n=== Loading Delaney Dataset with JT-VAE Filtering ===")
    training_smiles, training_targets = load_molecules_from_delaney(
        subset='train', 
        # max_molecules=50,  # Limit to 50 molecules for training
        jtvae_wrapper=jtvae  # Pass JT-VAE for filtering
    )
    
    # Load test data from Delaney dataset with JT-VAE filtering
    test_smiles, test_targets = load_molecules_from_delaney(
        subset='test',
        # max_molecules=20,
        jtvae_wrapper=jtvae  # Pass JT-VAE for filtering
    )
    
    # Test encoding/decoding first
    print("\n=== Testing JT-VAE Encoding/Decoding ===")
    test_molecule = training_smiles[0]  # Use first molecule from Delaney
    print(f"Test molecule: {test_molecule}")
    
    # Test encoding
    latent_array = jtvae.encode_to_numpy([test_molecule])
    print(f"Encoded to latent vector of shape: {latent_array.shape}")
    
    # Test decoding
    decoded_smiles = jtvae.decode_from_numpy(latent_array)
    print(f"Decoded back to: {decoded_smiles[0] if decoded_smiles else 'Failed'}")
    
    # Train the adversarial generator
    print("\n=== Training Adversarial Generator ===")
    adversarial_model.train(
        training_smiles=training_smiles,
        epochs=20,
        batch_size=4,
        lambda_adv=1.0,
        lambda_sim=0.01,
        success_threshold=0.7
    )
    
    # Generate adversarial examples
    print("\n=== Generating Adversarial Examples ===")
    test_molecules = test_smiles[:5]  # Use first 5 test molecules
    
    original_smiles = []
    adversarial_smiles = []
    original_targets = []
    
    for i, smiles in enumerate(test_molecules):
        target_value = test_targets[i][0] if len(test_targets) > i else "N/A"
        print(f"Generating adversarial molecule for: {smiles} (solubility: {target_value:.3f})")
        adv_molecules = adversarial_model.generate_adversarial(smiles, num_samples=1)
        
        if adv_molecules:
            original_smiles.append(smiles)
            adversarial_smiles.append(adv_molecules[0])
            original_targets.append(target_value)
            print(f"  -> Generated: {adv_molecules[0]}")
        else:
            print(f"  -> Failed to generate adversarial molecule")
    
    # Evaluate results
    if original_smiles and adversarial_smiles:
        print("\n=== Evaluating Adversarial Attacks ===")
        evaluation_results = evaluate_adversarial_attack(
            original_smiles, 
            adversarial_smiles, 
            black_box_model
        )
        
        # Add original targets to results dictionary
        evaluation_results['original_target'] = []
        for i in range(len(evaluation_results['original_smiles'])):
            if i < len(original_targets):
                evaluation_results['original_target'].append(original_targets[i])
            else:
                evaluation_results['original_target'].append(None)
        
        # Plot results
        plot_evaluation_results(evaluation_results)
        
        # Save results
        results_df = pd.DataFrame(evaluation_results)
        results_df.to_csv('results/adversarial_results_delaney.csv', index=False)
        print("Results saved to results/adversarial_results_delaney.csv")
    
    # Example of single molecule attack
    print("\n=== Single Molecule Attack Example ===")
    target_smiles = training_smiles[0]  # Use first Delaney molecule
    target_value = training_targets[0][0] if len(training_targets) > 0 else "N/A"
    print(f"Target molecule: {target_smiles}")
    print(f"Original Delaney solubility: {target_value:.3f}")
    
    original_pred = black_box_model.predict(target_smiles)
    print(f"Black-box model prediction: {original_pred:.3f}")
    
    adv_molecules = adversarial_model.generate_adversarial(target_smiles, num_samples=3)
    print(f"Generated {len(adv_molecules)} adversarial molecules:")
    
    if adv_molecules:
        for i, adv_smiles in enumerate(adv_molecules):
            adv_pred = black_box_model.predict(adv_smiles)
            from evaluation import calculate_tanimoto_similarity
            similarity = calculate_tanimoto_similarity(target_smiles, adv_smiles)
            
            print(f"Adversarial molecule {i+1}: {adv_smiles}")
            print(f"  Predicted solubility: {adv_pred:.3f}")
            print(f"  Tanimoto similarity: {similarity:.3f}")
            print(f"  Solubility difference: {abs(original_pred - adv_pred):.3f}")
    
    # Save dataset statistics
    print("\n=== Dataset Statistics ===")
    stats = {
        'dataset': 'Delaney',
        'total_training_molecules': len(training_smiles),
        'total_test_molecules': len(test_smiles),
        'successful_adversarial_generations': len(adversarial_smiles),
        'average_original_prediction': np.mean([black_box_model.predict(s) for s in original_smiles]) if original_smiles else 0,
        'average_adversarial_prediction': np.mean([black_box_model.predict(s) for s in adversarial_smiles]) if adversarial_smiles else 0,
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(results_path, 'dataset_statistics.csv'), index=False)
    print(f"Dataset statistics saved to {os.path.join(results_path, 'dataset_statistics.csv')}")
    
    print("=== Attack Complete ===")

if __name__ == "__main__":
    main()