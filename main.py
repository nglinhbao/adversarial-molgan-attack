import os
import argparse
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
import deepchem as dc
from jtvae_wrapper import JTVAEWrapper
from black_box_model import BlackBoxModel
from adversarial_generator import AdversarialMolCycleGAN
from evaluation import comprehensive_evaluation, evaluate_molecular_quality
from datetime import datetime
from data_loader import DATASETS, load_molecules_from_dataset, get_property_name, list_filtered_datasets, clean_filtered_datasets

def get_available_models_dict(dataset_name):
    """Get dictionary of all available trained models for transferability testing"""
    models_dir = "black_box_model_training/models"
    models_dict = {}
    
    if not os.path.exists(models_dir):
        print("Models directory not found!")
        return models_dict
    
    # Available model architectures
    available_models = [
        "BFGNN", "GNN", "GREA", "GRIN", "IRM", 
        "LSTM", "RPGNN", "SMILESTransformer"
    ]
    
    # Map architecture names to file names
    model_filename_map = {
        "BFGNN": "BFGNNMolecularPredictor",
        "GNN": "GNNMolecularPredictor", 
        "GREA": "GREAMolecularPredictor",
        "GRIN": "GRINMolecularPredictor",
        "IRM": "IRMMolecularPredictor",
        "LSTM": "LSTMMolecularPredictor",
        "RPGNN": "RPGNNMolecularPredictor",
        "SMILESTransformer": "SMILESTransformerMolecularPredictor"
    }
    
    dataset_cfg = DATASETS[dataset_name]
    
    print(f"Loading available models for transferability testing...")
    for arch in available_models:
        model_filename = model_filename_map[arch]
        model_path = f"{models_dir}/{dataset_name}_{model_filename}.pt"
        
        if os.path.exists(model_path):
            try:
                # Create BlackBoxModel instance for this architecture
                model = BlackBoxModel(
                    model_path=model_path,
                    model_type=dataset_cfg['task_type']
                )
                models_dict[arch] = model
                print(f"  ✓ Loaded {arch} model")
            except Exception as e:
                print(f"  ✗ Failed to load {arch} model: {e}")
        else:
            print(f"  - {arch} model not available")
    
    print(f"Loaded {len(models_dict)} models for transferability testing")
    return models_dict

def get_blackbox_model_path(dataset_name, model_architecture="BFGNN"):
    """Get the appropriate black-box model path for the dataset and architecture"""
    
    # Available model architectures
    available_models = [
        "BFGNN", "GNN", "GREA", "GRIN", "IRM", 
        "LSTM", "RPGNN", "SMILESTransformer"
    ]
    
    if model_architecture not in available_models:
        print(f"Warning: Unknown model architecture '{model_architecture}'")
        print(f"Available models: {available_models}")
        model_architecture = "BFGNN"  # Default fallback
    
    # Map architecture names to file names
    model_filename_map = {
        "BFGNN": "BFGNNMolecularPredictor",
        "GNN": "GNNMolecularPredictor", 
        "GREA": "GREAMolecularPredictor",
        "GRIN": "GRINMolecularPredictor",
        "IRM": "IRMMolecularPredictor",
        "LSTM": "LSTMMolecularPredictor",
        "RPGNN": "RPGNNMolecularPredictor",
        "SMILESTransformer": "SMILESTransformerMolecularPredictor"
    }
    
    model_filename = model_filename_map[model_architecture]
    model_path = f"black_box_model_training/models/{dataset_name}_{model_filename}.pt"
    
    return model_path

def list_available_models(dataset_name=None):
    """List all available trained models"""
    models_dir = "black_box_model_training/models"
    
    if not os.path.exists(models_dir):
        print("Models directory not found!")
        return
    
    files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    
    if dataset_name:
        files = [f for f in files if f.startswith(dataset_name)]
    
    if not files:
        print(f"No models found{f' for dataset {dataset_name}' if dataset_name else ''}!")
        return
    
    print(f"Available models{f' for {dataset_name}' if dataset_name else ''}:")
    
    # Group by dataset
    datasets = {}
    for file in sorted(files):
        parts = file.replace('.pt', '').split('_', 1)
        if len(parts) == 2:
            ds, model = parts
            if ds not in datasets:
                datasets[ds] = []
            # Extract just the model name (remove "MolecularPredictor")
            model_name = model.replace("MolecularPredictor", "")
            datasets[ds].append(model_name)
    
    for ds, models in datasets.items():
        print(f"  {ds.upper()}:")
        for model in sorted(models):
            model_path = f"{models_dir}/{ds}_{model}MolecularPredictor.pt"
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024*1024)  # MB
                print(f"    - {model} ({size:.1f} MB)")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Adversarial Mol-CycleGAN Attack on Molecular Property Prediction")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="delaney",
                       help="Dataset to use for adversarial attack")
    parser.add_argument("--model_architecture", type=str, default="BFGNN",
                       choices=["BFGNN", "GNN", "GREA", "GRIN", "IRM", "LSTM", "RPGNN", "SMILESTransformer"],
                       help="Black-box model architecture to attack")
    parser.add_argument("--max_train_molecules", type=int, default=None,
                       help="Maximum number of training molecules to use")
    parser.add_argument("--max_test_molecules", type=int, default=None,
                       help="Maximum number of test molecules to use")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Training batch size (increased for better GPU utilization)")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate for the generator (reduced for stability)")
    parser.add_argument("--lambda_adv", type=float, default=2.0,
                       help="Adversarial loss weight (reduced for better balance)")
    parser.add_argument("--lambda_sim", type=float, default=1.0,
                       help="Similarity loss weight (increased for better balance)")
    parser.add_argument("--success_threshold", type=float, default=0.7,
                       help="Success threshold for adversarial examples")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                       help="Interval for saving checkpoints")
    parser.add_argument("--force_refilter", action="store_true",
                       help="Force re-filtering even if cached dataset exists")
    parser.add_argument("--list_datasets", action="store_true",
                       help="List available filtered datasets and exit")
    parser.add_argument("--list_models", action="store_true",
                       help="List available trained models and exit")
    parser.add_argument("--clean_datasets", type=str, nargs="?", const="all",
                       help="Clean filtered datasets (specify dataset name or 'all')")
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_datasets:
        list_filtered_datasets()
        return
    
    if args.list_models:
        list_available_models(args.dataset if args.dataset != "delaney" else None)
        return
    
    if args.clean_datasets:
        if args.clean_datasets == "all":
            clean_filtered_datasets()
        else:
            clean_filtered_datasets(args.clean_datasets)
        return
    
    print(f"=== Adversarial Mol-CycleGAN Implementation ===")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model Architecture: {args.model_architecture}")
    print(f"Task type: {DATASETS[args.dataset]['task_type']}")
    print(f"Metric: {DATASETS[args.dataset]['metric_name']}")
    print(f"Force refilter: {args.force_refilter}")
    
    # Create timestamped results directory name
    results_name = f"{args.dataset}_{args.model_architecture}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    results_path = os.path.join('results', results_name)
    os.makedirs(results_path, exist_ok=True)
    print(f"Results will be saved to: {results_path}")
    
    # Configuration following mol-cycle-gan structure
    jtvae_path = "./jtvae/"
    vocab_path = "data/zinc15/vocab.txt"
    model_path = "molvae/vae_model/model.iter-4"
    blackbox_model_path = get_blackbox_model_path(args.dataset, args.model_architecture)
    
    # Check if black-box model exists
    if not os.path.exists(blackbox_model_path):
        print(f"Error: Black-box model not found at {blackbox_model_path}")
        print(f"Available models for {args.dataset}:")
        list_available_models(args.dataset)
        print(f"\nPlease train the model first using:")
        print(f"  cd black_box_model_training && python train.py --dataset {args.dataset}")
        return
    
    print(f"Using black-box model: {blackbox_model_path}")
    
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
    dataset_cfg = DATASETS[args.dataset]
    black_box_model = BlackBoxModel(
        model_path=blackbox_model_path, 
        model_type=dataset_cfg['task_type']
    )
    
    print("Initializing adversarial generator...")
    adversarial_model = AdversarialMolCycleGAN(
        jtvae_wrapper=jtvae,
        black_box_model=black_box_model,
        learning_rate=args.learning_rate,
        results_name=results_name  # Pass the results directory name
    )
    
    # Load training data from specified dataset with JT-VAE filtering
    print(f"\n=== Loading {args.dataset.upper()} Training Dataset ===")
    training_smiles, training_targets, train_cfg = load_molecules_from_dataset(
        dataset_name=args.dataset,
        subset='train', 
        max_molecules=args.max_train_molecules,
        jtvae_wrapper=jtvae,  # Pass JT-VAE for filtering
        force_refilter=args.force_refilter
    )
    
    # Load test data from specified dataset with JT-VAE filtering
    print(f"\n=== Loading {args.dataset.upper()} Test Dataset ===")
    test_smiles, test_targets, test_cfg = load_molecules_from_dataset(
        dataset_name=args.dataset,
        subset='test',
        max_molecules=args.max_test_molecules,
        jtvae_wrapper=jtvae,  # Pass JT-VAE for filtering
        force_refilter=args.force_refilter
    )
    
    # Test encoding/decoding first
    print("\n=== Testing JT-VAE Encoding/Decoding ===")
    test_molecule = training_smiles[0]  # Use first molecule from dataset
    property_name = get_property_name(args.dataset)
    target_value = training_targets[0][0] if len(training_targets) > 0 else "N/A"
    print(f"Test molecule: {test_molecule}")
    if isinstance(target_value, (int, float)):
        print(f"Target {property_name}: {target_value:.3f}")
    else:
        print(f"Target {property_name}: {target_value}")
    
    # Test encoding
    latent_array = jtvae.encode_to_numpy([test_molecule])
    print(f"Encoded to latent vector of shape: {latent_array.shape}")
    
    # Test decoding
    decoded_smiles = jtvae.decode_from_numpy(latent_array)
    print(f"Decoded back to: {decoded_smiles[0] if decoded_smiles else 'Failed'}")
    
    # Test the black-box model on the test molecule
    print(f"\n=== Testing Black-box Model ({args.model_architecture}) ===")
    test_prediction = black_box_model.predict(test_molecule)
    print(f"Black-box model prediction: {test_prediction:.3f}")
    
    # Train the adversarial generator
    print(f"\n=== Training Adversarial Generator (targeting {args.model_architecture}) ===")
    # The generator model will be automatically saved after training completes
    # Checkpoints will also be saved every 50 epochs by default
    adversarial_model.train(
        training_smiles=training_smiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_adv=args.lambda_adv,
        lambda_sim=args.lambda_sim,
        success_threshold=args.success_threshold,
        save_checkpoints=True,      # Save checkpoints during training
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Generate adversarial examples
    print("\n=== Generating Adversarial Examples ===")
    test_molecules = test_smiles[:20]  # Use first 20 test molecules for better evaluation
    
    original_smiles = []
    adversarial_smiles = []
    original_targets = []
    
    for i, smiles in enumerate(test_molecules):
        target_value = test_targets[i][0] if len(test_targets) > i else "N/A"
        if isinstance(target_value, (int, float)):
            print(f"Generating adversarial molecule for: {smiles} ({property_name}: {target_value:.3f})")
        else:
            print(f"Generating adversarial molecule for: {smiles} ({property_name}: {target_value})")
        adv_molecules = adversarial_model.generate_adversarial(smiles, num_samples=1)
        
        if adv_molecules:
            original_smiles.append(smiles)
            adversarial_smiles.append(adv_molecules[0])
            original_targets.append(target_value)
            print(f"  -> Generated: {adv_molecules[0]}")
        else:
            print(f"  -> Failed to generate adversarial molecule")
    
    # Comprehensive evaluation using new evaluation.py
    if original_smiles and adversarial_smiles:
        print(f"\n=== Comprehensive Adversarial Attack Evaluation on {args.model_architecture} ===")
        
        # Get all available models for transferability testing
        models_dict = get_available_models_dict(args.dataset)
        
        # Remove the target model from transferability testing to avoid redundancy
        if args.model_architecture in models_dict:
            target_model = models_dict.pop(args.model_architecture)
            print(f"Target model ({args.model_architecture}) removed from transferability testing")
        
        if models_dict:
            print(f"Models for transferability testing: {list(models_dict.keys())}")
        else:
            print("No additional models available for transferability testing")
            models_dict = None
        
        # Perform comprehensive evaluation
        comprehensive_results = comprehensive_evaluation(
            original_smiles=original_smiles,
            adversarial_smiles=adversarial_smiles,
            black_box_model=black_box_model,
            training_smiles=training_smiles,  # For novelty calculation
            models_dict=models_dict,  # Now includes all available models
            success_threshold=args.success_threshold,
            task_type=dataset_cfg['task_type'],
            save_path=results_path,
            dataset_name=args.dataset,
            model_name=args.model_architecture
        )
        
        # Extract individual results
        attack_results = comprehensive_results['attack_results']
        quality_results = comprehensive_results['quality_results']
        transferability_results = comprehensive_results['transferability_results']
        
        # Print summary statistics
        print(f"\n=== Attack Success Metrics ===")
        print(f"Success Rate: {attack_results['success_rate']:.3f}")
        print(f"Successful Attacks: {attack_results['successful_attacks']}/{attack_results['total_attempts']}")
        print(f"Average Prediction Change: {attack_results['avg_prediction_change']:.3f}")
        print(f"Average Tanimoto Similarity: {attack_results['avg_tanimoto_similarity']:.3f}")
        
        print(f"\n=== Molecular Quality Metrics ===")
        print(f"Validity: {quality_results.get('validity_validity', 0):.3f}")
        print(f"Uniqueness: {quality_results.get('uniqueness_uniqueness', 0):.3f}")
        print(f"Novelty: {quality_results.get('novelty_novelty', 0):.3f}")
        
        # Print transferability results
        if transferability_results:
            print(f"\n=== Transferability Results ===")
            for model_name, results in transferability_results.items():
                success_rate = results.get('success_rate', 0)
                avg_change = results.get('avg_prediction_change', 0)
                successful = results.get('successful_attacks', 0)
                total = results.get('total_attempts', 0)
                print(f"{model_name}:")
                print(f"  Success Rate: {success_rate:.3f} ({successful}/{total})")
                print(f"  Avg Prediction Change: {avg_change:.3f}")
        
        # Create detailed results DataFrame
        detailed_results = {
            'original_smiles': attack_results['original_smiles'],
            'adversarial_smiles': attack_results['adversarial_smiles'],
            'original_predictions': attack_results['original_predictions'],
            'adversarial_predictions': attack_results['adversarial_predictions'],
            'tanimoto_similarities': attack_results['tanimoto_similarities'],
            'prediction_changes': attack_results['prediction_changes'],
            'original_targets': original_targets[:len(attack_results['original_smiles'])],
            'model_architecture': [args.model_architecture] * len(attack_results['original_smiles']),
            'dataset': [args.dataset] * len(attack_results['original_smiles'])
        }
        
        # Add transferability results to detailed DataFrame (with proper length checking)
        if transferability_results:
            main_length = len(attack_results['original_smiles'])
            for model_name, results in transferability_results.items():
                # Get predictions and changes, padding with None if needed
                model_adv_preds = results.get('adversarial_predictions', [])
                model_pred_changes = results.get('prediction_changes', [])
                model_orig_preds = results.get('original_predictions', [])
                
                # Ensure arrays are the same length as main results
                if len(model_adv_preds) >= main_length:
                    detailed_results[f'{model_name}_adv_predictions'] = model_adv_preds[:main_length]
                else:
                    detailed_results[f'{model_name}_adv_predictions'] = model_adv_preds + [None] * (main_length - len(model_adv_preds))
                
                if len(model_orig_preds) >= main_length:
                    detailed_results[f'{model_name}_orig_predictions'] = model_orig_preds[:main_length]
                else:
                    detailed_results[f'{model_name}_orig_predictions'] = model_orig_preds + [None] * (main_length - len(model_orig_preds))
                
                if len(model_pred_changes) >= main_length:
                    detailed_results[f'{model_name}_pred_changes'] = model_pred_changes[:main_length]
                else:
                    detailed_results[f'{model_name}_pred_changes'] = model_pred_changes + [None] * (main_length - len(model_pred_changes))
                
                # Calculate success for each example
                success_flags = []
                for i in range(main_length):
                    if i < len(model_pred_changes):
                        success_flags.append(1 if model_pred_changes[i] >= args.success_threshold else 0)
                    else:
                        success_flags.append(None)
                
                detailed_results[f'{model_name}_attack_success'] = success_flags
        
        # Save detailed results
        results_df = pd.DataFrame(detailed_results)
        results_file = os.path.join(results_path, f'adversarial_results_{args.dataset}_{args.model_architecture}.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Detailed results saved to {results_file}")
        
        # Save comprehensive metrics summary
        metrics_summary = {
            'dataset': args.dataset,
            'model_architecture': args.model_architecture,
            'attack_success_rate': attack_results['success_rate'],
            'avg_prediction_change': attack_results['avg_prediction_change'],
            'avg_tanimoto_similarity': attack_results['avg_tanimoto_similarity'],
            'min_tanimoto_similarity': attack_results['min_tanimoto_similarity'],
            'max_tanimoto_similarity': attack_results['max_tanimoto_similarity'],
            'validity': quality_results.get('validity_validity', 0),
            'uniqueness': quality_results.get('uniqueness_uniqueness', 0),
            'novelty': quality_results.get('novelty_novelty', 0),
            'successful_attacks': attack_results['successful_attacks'],
            'total_attempts': attack_results['total_attempts'],
            'valid_molecules': quality_results.get('validity_valid_molecules', 0),
            'unique_molecules': quality_results.get('uniqueness_unique_molecules', 0),
            'novel_molecules': quality_results.get('novelty_novel_molecules', 0),
            'models_tested_for_transferability': len(transferability_results) if transferability_results else 0
        }
        
        # Add transferability metrics to summary
        if transferability_results:
            for model_name, results in transferability_results.items():
                metrics_summary[f'{model_name}_transferability_success_rate'] = results.get('success_rate', 0)
                metrics_summary[f'{model_name}_transferability_avg_change'] = results.get('avg_prediction_change', 0)
        
        metrics_df = pd.DataFrame([metrics_summary])
        metrics_file = os.path.join(results_path, f'evaluation_metrics_{args.dataset}_{args.model_architecture}.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Evaluation metrics saved to {metrics_file}")
        
        # Print individual adversarial examples with detailed analysis
        print(f"\n=== Individual Adversarial Examples Analysis ===")
        for i in range(min(5, len(attack_results['original_smiles']))):  # Show first 5 examples
            orig_smiles = attack_results['original_smiles'][i]
            adv_smiles = attack_results['adversarial_smiles'][i]
            orig_pred = attack_results['original_predictions'][i]
            adv_pred = attack_results['adversarial_predictions'][i]
            similarity = attack_results['tanimoto_similarities'][i]
            pred_change = attack_results['prediction_changes'][i]
            
            print(f"\nExample {i+1}:")
            print(f"  Original:     {orig_smiles}")
            print(f"  Adversarial:  {adv_smiles}")
            print(f"  Original {property_name}: {orig_pred:.3f}")
            print(f"  Adversarial {property_name}: {adv_pred:.3f}")
            print(f"  Prediction change: {pred_change:.3f}")
            print(f"  Tanimoto similarity: {similarity:.3f}")
            
            # Show transferability for this example
            if transferability_results:
                print(f"  Transferability:")
                for model_name, results in transferability_results.items():
                    if i < len(results.get('adversarial_predictions', [])):
                        transfer_pred = results['adversarial_predictions'][i]
                        transfer_change = results['prediction_changes'][i]
                        transfer_success = "✓" if transfer_change >= args.success_threshold else "✗"
                        print(f"    {model_name}: {transfer_pred:.3f} (Δ{transfer_change:.3f}) {transfer_success}")
            
            # Determine if attack was successful
            if pred_change >= args.success_threshold:
                print(f"  Status: ✓ SUCCESSFUL ATTACK")
            else:
                print(f"  Status: ✗ Failed attack") 
    
    else:
        print("No adversarial molecules generated for evaluation.")
    
    # Save comprehensive dataset statistics
    print("\n=== Dataset Statistics ===")
    stats = {
        'dataset': args.dataset,
        'model_architecture': args.model_architecture,
        'task_type': dataset_cfg['task_type'],
        'metric': dataset_cfg['metric_name'],
        'total_training_molecules': len(training_smiles),
        'total_test_molecules': len(test_smiles),
        'molecules_attempted': len(test_molecules),
        'successful_adversarial_generations': len(adversarial_smiles),
        'generation_success_rate': len(adversarial_smiles) / len(test_molecules) if test_molecules else 0,
        'average_original_prediction': np.mean([black_box_model.predict(s) for s in original_smiles]) if original_smiles else 0,
        'average_adversarial_prediction': np.mean([black_box_model.predict(s) for s in adversarial_smiles]) if adversarial_smiles else 0,
        'cached_datasets_used': not args.force_refilter,
        'blackbox_model_path': blackbox_model_path,
        'success_threshold': args.success_threshold,
        'epochs_trained': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lambda_adv': args.lambda_adv,
        'lambda_sim': args.lambda_sim
    }
    
    # Add evaluation metrics to stats if available
    if original_smiles and adversarial_smiles:
        stats.update({
            'attack_success_rate': attack_results['success_rate'],
            'avg_prediction_change': attack_results['avg_prediction_change'],
            'avg_tanimoto_similarity': attack_results['avg_tanimoto_similarity'],
            'validity': quality_results.get('validity_validity', 0),
            'uniqueness': quality_results.get('uniqueness_uniqueness', 0),
            'novelty': quality_results.get('novelty_novelty', 0)
        })
    
    stats_df = pd.DataFrame([stats])
    stats_file = os.path.join(results_path, 'comprehensive_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Comprehensive statistics saved to {stats_file}")
    
    print("=== Adversarial Attack Evaluation Complete ===")

if __name__ == "__main__":
    main()