import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Tuple, Optional, Union
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def is_valid_molecule(smiles: str) -> bool:
    """Check if a SMILES string represents a valid molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two molecules using Morgan fingerprints."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

def calculate_attack_success_rate(original_predictions: List[float], 
                                adversarial_predictions: List[float],
                                success_threshold: float = 0.7,
                                task_type: str = "regression") -> Dict[str, float]:
    """
    Calculate attack success rate based on prediction differences.
    
    Args:
        original_predictions: Predictions on original molecules
        adversarial_predictions: Predictions on adversarial molecules
        success_threshold: Threshold for considering an attack successful
        task_type: "regression" or "classification"
    
    Returns:
        Dictionary with success metrics
    """
    if len(original_predictions) != len(adversarial_predictions):
        raise ValueError("Length of original and adversarial predictions must match")
    
    if len(original_predictions) == 0:
        return {"success_rate": 0.0, "avg_prediction_change": 0.0, "successful_attacks": 0}
    
    prediction_changes = []
    successful_attacks = 0
    
    for orig_pred, adv_pred in zip(original_predictions, adversarial_predictions):
        if task_type == "regression":
            # For regression, measure absolute change in prediction
            change = abs(orig_pred - adv_pred)
            prediction_changes.append(change)
            
            # Success if change exceeds threshold
            if change >= success_threshold:
                successful_attacks += 1
        else:  # classification
            # For classification, success if prediction class changes
            orig_class = int(orig_pred > 0.5)
            adv_class = int(adv_pred > 0.5)
            
            change = abs(orig_pred - adv_pred)
            prediction_changes.append(change)
            
            if orig_class != adv_class:
                successful_attacks += 1
    
    success_rate = successful_attacks / len(original_predictions)
    avg_change = np.mean(prediction_changes)
    
    return {
        "success_rate": success_rate,
        "avg_prediction_change": avg_change,
        "successful_attacks": successful_attacks,
        "total_attempts": len(original_predictions),
        "prediction_changes": prediction_changes
    }

def evaluate_validity(generated_smiles: List[str]) -> Dict[str, float]:
    """
    Evaluate validity of generated molecules.
    
    Args:
        generated_smiles: List of generated SMILES strings
    
    Returns:
        Dictionary with validity metrics
    """
    if not generated_smiles:
        return {"validity": 0.0, "valid_molecules": 0, "total_molecules": 0}
    
    valid_count = sum(1 for smiles in generated_smiles if is_valid_molecule(smiles))
    validity = valid_count / len(generated_smiles)
    
    return {
        "validity": validity,
        "valid_molecules": valid_count,
        "total_molecules": len(generated_smiles),
        "invalid_molecules": len(generated_smiles) - valid_count
    }

def evaluate_novelty(generated_smiles: List[str], 
                    training_smiles: List[str]) -> Dict[str, float]:
    """
    Evaluate novelty of generated molecules (not in training set).
    
    Args:
        generated_smiles: List of generated SMILES strings
        training_smiles: List of training set SMILES strings
    
    Returns:
        Dictionary with novelty metrics
    """
    # Filter to only valid molecules
    valid_generated = [smiles for smiles in generated_smiles if is_valid_molecule(smiles)]
    
    if not valid_generated:
        return {"novelty": 0.0, "novel_molecules": 0, "valid_molecules": 0}
    
    training_set = set(training_smiles)
    novel_count = sum(1 for smiles in valid_generated if smiles not in training_set)
    novelty = novel_count / len(valid_generated)
    
    return {
        "novelty": novelty,
        "novel_molecules": novel_count,
        "valid_molecules": len(valid_generated),
        "training_set_size": len(training_set)
    }

def evaluate_uniqueness(generated_smiles: List[str]) -> Dict[str, float]:
    """
    Evaluate uniqueness of generated molecules.
    
    Args:
        generated_smiles: List of generated SMILES strings
    
    Returns:
        Dictionary with uniqueness metrics
    """
    # Filter to only valid molecules
    valid_generated = [smiles for smiles in generated_smiles if is_valid_molecule(smiles)]
    
    if not valid_generated:
        return {"uniqueness": 0.0, "unique_molecules": 0, "valid_molecules": 0}
    
    unique_molecules = len(set(valid_generated))
    uniqueness = unique_molecules / len(valid_generated)
    
    return {
        "uniqueness": uniqueness,
        "unique_molecules": unique_molecules,
        "valid_molecules": len(valid_generated),
        "duplicate_molecules": len(valid_generated) - unique_molecules
    }

def evaluate_transferability(original_smiles: List[str],
                           adversarial_smiles: List[str],
                           models: Dict[str, object],
                           success_threshold: float = 0.7) -> Dict[str, Dict]:
    """
    Evaluate transferability of adversarial examples across different models.
    
    Args:
        original_smiles: List of original molecule SMILES
        adversarial_smiles: List of adversarial molecule SMILES
        models: Dictionary of model_name -> model_object
        success_threshold: Threshold for attack success
    
    Returns:
        Dictionary with transferability results for each model
    """
    transferability_results = {}
    
    print(f"Evaluating transferability across {len(models)} models...")
    print(f"Input: {len(original_smiles)} original, {len(adversarial_smiles)} adversarial molecules")
    
    for model_name, model in models.items():
        try:
            print(f"  Evaluating model: {model_name}")
            
            # Get predictions for original and adversarial molecules
            original_preds = []
            adversarial_preds = []
            
            # Process each molecule pair
            for i, (orig_smiles, adv_smiles) in enumerate(zip(original_smiles, adversarial_smiles)):
                try:
                    orig_pred = model.predict(orig_smiles)
                    adv_pred = model.predict(adv_smiles)
                    original_preds.append(orig_pred)
                    adversarial_preds.append(adv_pred)
                except Exception as e:
                    print(f"    Error predicting molecule {i}: {e}")
                    # Skip this molecule pair
                    continue
            
            print(f"    Successfully processed {len(original_preds)} molecule pairs")
            
            if len(original_preds) == 0:
                print(f"    No valid predictions for {model_name}")
                transferability_results[model_name] = {
                    "success_rate": 0.0,
                    "avg_prediction_change": 0.0,
                    "successful_attacks": 0,
                    "total_attempts": 0,
                    "original_predictions": [],
                    "adversarial_predictions": [],
                    "prediction_changes": [],
                    "error": "No valid predictions"
                }
                continue
            
            # Calculate attack success rate for this model
            success_metrics = calculate_attack_success_rate(
                original_preds, adversarial_preds, success_threshold
            )
            
            # Add the prediction arrays to the results
            success_metrics["original_predictions"] = original_preds
            success_metrics["adversarial_predictions"] = adversarial_preds
            
            transferability_results[model_name] = success_metrics
            print(f"    Success rate: {success_metrics['success_rate']:.3f}")
            
        except Exception as e:
            print(f"  Error evaluating model {model_name}: {e}")
            transferability_results[model_name] = {
                "success_rate": 0.0,
                "avg_prediction_change": 0.0,
                "successful_attacks": 0,
                "total_attempts": 0,
                "original_predictions": [],
                "adversarial_predictions": [],
                "prediction_changes": [],
                "error": str(e)
            }
    
    return transferability_results

def evaluate_adversarial_attack(original_smiles: List[str],
                               adversarial_smiles: List[str],
                               black_box_model,
                               success_threshold: float = 0.7,
                               task_type: str = "regression") -> Dict:
    """
    Comprehensive evaluation of adversarial attack results.
    
    Args:
        original_smiles: List of original molecule SMILES
        adversarial_smiles: List of adversarial molecule SMILES
        black_box_model: Black box model for predictions
        success_threshold: Threshold for attack success
        task_type: "regression" or "classification"
    
    Returns:
        Dictionary with comprehensive evaluation results
    """
    if len(original_smiles) != len(adversarial_smiles):
        raise ValueError("Length of original and adversarial SMILES must match")
    
    # Get predictions
    original_predictions = []
    adversarial_predictions = []
    tanimoto_similarities = []
    valid_pairs = []
    
    for orig_smiles, adv_smiles in zip(original_smiles, adversarial_smiles):
        try:
            if is_valid_molecule(orig_smiles) and is_valid_molecule(adv_smiles):
                orig_pred = black_box_model.predict(orig_smiles)
                adv_pred = black_box_model.predict(adv_smiles)
                similarity = calculate_tanimoto_similarity(orig_smiles, adv_smiles)
                
                original_predictions.append(orig_pred)
                adversarial_predictions.append(adv_pred)
                tanimoto_similarities.append(similarity)
                valid_pairs.append((orig_smiles, adv_smiles))
        except Exception as e:
            print(f"Error processing pair {orig_smiles} -> {adv_smiles}: {e}")
            continue
    
    if not valid_pairs:
        return {
            "error": "No valid molecule pairs found",
            "original_smiles": [],
            "adversarial_smiles": [],
            "original_predictions": [],
            "adversarial_predictions": [],
            "tanimoto_similarities": [],
            "success_rate": 0.0
        }
    
    # Calculate attack success rate
    success_metrics = calculate_attack_success_rate(
        original_predictions, adversarial_predictions, success_threshold, task_type
    )
    
    # Calculate similarity statistics
    avg_similarity = np.mean(tanimoto_similarities)
    min_similarity = np.min(tanimoto_similarities)
    max_similarity = np.max(tanimoto_similarities)
    
    # Compile results
    results = {
        "original_smiles": [pair[0] for pair in valid_pairs],
        "adversarial_smiles": [pair[1] for pair in valid_pairs],
        "original_predictions": original_predictions,
        "adversarial_predictions": adversarial_predictions,
        "tanimoto_similarities": tanimoto_similarities,
        "success_rate": success_metrics["success_rate"],
        "avg_prediction_change": success_metrics["avg_prediction_change"],
        "successful_attacks": success_metrics["successful_attacks"],
        "total_attempts": success_metrics["total_attempts"],
        "avg_tanimoto_similarity": avg_similarity,
        "min_tanimoto_similarity": min_similarity,
        "max_tanimoto_similarity": max_similarity,
        "valid_pairs": len(valid_pairs),
        "prediction_changes": success_metrics["prediction_changes"]
    }
    
    return results

def evaluate_molecular_quality(generated_smiles: List[str],
                              training_smiles: List[str] = None) -> Dict[str, float]:
    """
    Evaluate molecular quality metrics: validity, novelty, and uniqueness.
    
    Args:
        generated_smiles: List of generated SMILES strings
        training_smiles: List of training set SMILES (for novelty calculation)
    
    Returns:
        Dictionary with quality metrics
    """
    results = {}
    
    # Validity
    validity_metrics = evaluate_validity(generated_smiles)
    results.update({f"validity_{k}": v for k, v in validity_metrics.items()})
    
    # Uniqueness
    uniqueness_metrics = evaluate_uniqueness(generated_smiles)
    results.update({f"uniqueness_{k}": v for k, v in uniqueness_metrics.items()})
    
    # Novelty (if training set provided)
    if training_smiles is not None:
        novelty_metrics = evaluate_novelty(generated_smiles, training_smiles)
        results.update({f"novelty_{k}": v for k, v in novelty_metrics.items()})
    
    return results

def plot_transferability_results(transferability_results: Dict[str, Dict], 
                                save_path: str = None):
    """
    Plot transferability results across different models.
    
    Args:
        transferability_results: Results from evaluate_transferability
        save_path: Path to save plots
    """
    if not transferability_results:
        print("No transferability results to plot")
        return
    
    models = list(transferability_results.keys())
    success_rates = [transferability_results[model].get('success_rate', 0) for model in models]
    avg_changes = [transferability_results[model].get('avg_prediction_change', 0) for model in models]
    
    # Limit figure size and handle many models
    n_models = len(models)
    if n_models == 0:
        print("No models to plot")
        return
    
    # Calculate appropriate figure size
    fig_width = max(12, min(20, 3 + n_models * 1.5))
    fig_height = 6
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Truncate model names if too long
    display_models = [model[:15] + '...' if len(model) > 15 else model for model in models]
    
    # Success rates
    bars1 = ax1.bar(range(len(models)), success_rates, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Attack Success Rate Across Models')
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(display_models, rotation=45, ha='right')
    
    # Add value labels (only if not too many models)
    if n_models <= 10:
        for i, (bar, value) in enumerate(zip(bars1, success_rates)):
            height = bar.get_height()
            ax1.text(i, height + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Average prediction changes
    bars2 = ax2.bar(range(len(models)), avg_changes, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Average Prediction Change')
    ax2.set_title('Average Prediction Change Across Models')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(display_models, rotation=45, ha='right')
    
    # Add value labels (only if not too many models)
    if n_models <= 10:
        for i, (bar, value) in enumerate(zip(bars2, avg_changes)):
            height = bar.get_height()
            ax2.text(i, height + max(avg_changes) * 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        try:
            os.makedirs(save_path, exist_ok=True)
            plot_file = os.path.join(save_path, 'transferability_plots.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Transferability plots saved to {plot_file}")
        except Exception as e:
            print(f"Error saving transferability plots: {e}")
            # Try saving with lower DPI
            try:
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                print(f"Transferability plots saved with lower DPI to {plot_file}")
            except Exception as e2:
                print(f"Failed to save transferability plots even with lower DPI: {e2}")
    
    try:
        plt.show()
    except:
        print("Could not display plot")
    finally:
        plt.close()

def plot_evaluation_results(evaluation_results: Dict, save_path: str = None):
    """
    Plot comprehensive evaluation results.
    
    Args:
        evaluation_results: Results from evaluate_adversarial_attack
        save_path: Path to save plots
    """
    if not evaluation_results or 'error' in evaluation_results:
        print("No valid evaluation results to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Adversarial Attack Evaluation Results', fontsize=16)
    
    # Plot 1: Prediction changes
    if 'prediction_changes' in evaluation_results and evaluation_results['prediction_changes']:
        try:
            axes[0, 0].hist(evaluation_results['prediction_changes'], bins=20, alpha=0.7, color='skyblue')
            mean_change = np.mean(evaluation_results['prediction_changes'])
            axes[0, 0].axvline(mean_change, color='red', linestyle='--', label=f'Mean: {mean_change:.3f}')
            axes[0, 0].set_xlabel('Prediction Change')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Prediction Changes')
            axes[0, 0].legend()
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Error plotting prediction changes:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Prediction Changes (Error)')
    else:
        axes[0, 0].text(0.5, 0.5, 'No prediction changes data', ha='center', va='center', 
                       transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Prediction Changes (No Data)')
    
    # Plot 2: Tanimoto similarities
    if 'tanimoto_similarities' in evaluation_results and evaluation_results['tanimoto_similarities']:
        try:
            axes[0, 1].hist(evaluation_results['tanimoto_similarities'], bins=20, alpha=0.7, color='lightgreen')
            mean_sim = np.mean(evaluation_results['tanimoto_similarities'])
            axes[0, 1].axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
            axes[0, 1].set_xlabel('Tanimoto Similarity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Tanimoto Similarities')
            axes[0, 1].legend()
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Error plotting similarities:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Tanimoto Similarities (Error)')
    else:
        axes[0, 1].text(0.5, 0.5, 'No similarity data', ha='center', va='center', 
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Tanimoto Similarities (No Data)')
    
    # Plot 3: Scatter plot of original vs adversarial predictions
    if ('original_predictions' in evaluation_results and 'adversarial_predictions' in evaluation_results and
        evaluation_results['original_predictions'] and evaluation_results['adversarial_predictions']):
        try:
            scatter = axes[1, 0].scatter(evaluation_results['original_predictions'], 
                                       evaluation_results['adversarial_predictions'],
                                       c=evaluation_results.get('tanimoto_similarities', 'blue'), 
                                       cmap='viridis', alpha=0.7)
            
            min_pred = min(min(evaluation_results['original_predictions']), 
                          min(evaluation_results['adversarial_predictions']))
            max_pred = max(max(evaluation_results['original_predictions']), 
                          max(evaluation_results['adversarial_predictions']))
            
            axes[1, 0].plot([min_pred, max_pred], [min_pred, max_pred], 'r--', alpha=0.5, label='y=x')
            axes[1, 0].set_xlabel('Original Predictions')
            axes[1, 0].set_ylabel('Adversarial Predictions')
            axes[1, 0].set_title('Original vs Adversarial Predictions')
            axes[1, 0].legend()
            
            if 'tanimoto_similarities' in evaluation_results:
                plt.colorbar(scatter, ax=axes[1, 0], label='Tanimoto Similarity')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Error plotting scatter:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Predictions Scatter (Error)')
    else:
        axes[1, 0].text(0.5, 0.5, 'No prediction data', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Predictions Scatter (No Data)')
    
    # Plot 4: Success metrics summary
    try:
        success_rate = evaluation_results.get('success_rate', 0)
        avg_similarity = evaluation_results.get('avg_tanimoto_similarity', 0)
        avg_change = evaluation_results.get('avg_prediction_change', 0)
        
        metrics = ['Success Rate', 'Avg Similarity', 'Avg Change']
        values = [success_rate, avg_similarity, avg_change]
        colors = ['coral', 'lightblue', 'lightcoral']
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Summary Metrics')
        axes[1, 1].set_ylim(0, max(1, max(values) * 1.1))
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'Error plotting metrics:\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Summary Metrics (Error)')
    
    plt.tight_layout()
    
    if save_path:
        try:
            os.makedirs(save_path, exist_ok=True)
            plot_file = os.path.join(save_path, 'evaluation_plots.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to {plot_file}")
        except Exception as e:
            print(f"Error saving evaluation plots: {e}")
            # Try saving with lower DPI
            try:
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                print(f"Evaluation plots saved with lower DPI to {plot_file}")
            except Exception as e2:
                print(f"Failed to save evaluation plots even with lower DPI: {e2}")
    
    try:
        plt.show()
    except:
        print("Could not display plot")
    finally:
        plt.close()

def save_evaluation_report(evaluation_results: Dict,
                          molecular_quality_results: Dict = None,
                          transferability_results: Dict = None,
                          save_path: str = None,
                          dataset_name: str = "Unknown",
                          model_name: str = "Unknown") -> str:
    """
    Save a comprehensive evaluation report.
    
    Args:
        evaluation_results: Results from evaluate_adversarial_attack
        molecular_quality_results: Results from evaluate_molecular_quality
        transferability_results: Results from evaluate_transferability
        save_path: Path to save the report
        dataset_name: Name of the dataset
        model_name: Name of the target model
    
    Returns:
        Path to the saved report
    """
    if save_path is None:
        save_path = "."
    
    os.makedirs(save_path, exist_ok=True)
    report_path = os.path.join(save_path, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ADVERSARIAL MOLECULAR ATTACK EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Target Model: {model_name}\n")
        f.write(f"Report Generated: {pd.Timestamp.now()}\n\n")
        
        # Attack Success Metrics
        f.write("ATTACK SUCCESS METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Success Rate: {evaluation_results.get('success_rate', 0):.3f}\n")
        f.write(f"Successful Attacks: {evaluation_results.get('successful_attacks', 0)}\n")
        f.write(f"Total Attempts: {evaluation_results.get('total_attempts', 0)}\n")
        f.write(f"Average Prediction Change: {evaluation_results.get('avg_prediction_change', 0):.3f}\n")
        f.write(f"Valid Pairs Generated: {evaluation_results.get('valid_pairs', 0)}\n\n")
        
        # Similarity Metrics
        f.write("SIMILARITY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Tanimoto Similarity: {evaluation_results.get('avg_tanimoto_similarity', 0):.3f}\n")
        f.write(f"Min Tanimoto Similarity: {evaluation_results.get('min_tanimoto_similarity', 0):.3f}\n")
        f.write(f"Max Tanimoto Similarity: {evaluation_results.get('max_tanimoto_similarity', 0):.3f}\n\n")
        
        # Molecular Quality Metrics
        if molecular_quality_results:
            f.write("MOLECULAR QUALITY METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Validity: {molecular_quality_results.get('validity_validity', 0):.3f}\n")
            f.write(f"Valid Molecules: {molecular_quality_results.get('validity_valid_molecules', 0)}\n")
            f.write(f"Total Molecules: {molecular_quality_results.get('validity_total_molecules', 0)}\n")
            
            if 'uniqueness_uniqueness' in molecular_quality_results:
                f.write(f"Uniqueness: {molecular_quality_results.get('uniqueness_uniqueness', 0):.3f}\n")
                f.write(f"Unique Molecules: {molecular_quality_results.get('uniqueness_unique_molecules', 0)}\n")
            
            if 'novelty_novelty' in molecular_quality_results:
                f.write(f"Novelty: {molecular_quality_results.get('novelty_novelty', 0):.3f}\n")
                f.write(f"Novel Molecules: {molecular_quality_results.get('novelty_novel_molecules', 0)}\n")
            f.write("\n")
        
        # Transferability Results
        if transferability_results:
            f.write("TRANSFERABILITY RESULTS\n")
            f.write("-" * 40 + "\n")
            for model, results in transferability_results.items():
                f.write(f"{model}:\n")
                f.write(f"  Success Rate: {results.get('success_rate', 0):.3f}\n")
                f.write(f"  Avg Prediction Change: {results.get('avg_prediction_change', 0):.3f}\n")
                f.write(f"  Successful Attacks: {results.get('successful_attacks', 0)}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Evaluation report saved to {report_path}")
    return report_path

# Main evaluation function for comprehensive assessment
def comprehensive_evaluation(original_smiles: List[str],
                           adversarial_smiles: List[str],
                           black_box_model,
                           training_smiles: List[str] = None,
                           models_dict: Dict = None,
                           success_threshold: float = 0.7,
                           task_type: str = "regression",
                           save_path: str = None,
                           dataset_name: str = "Unknown",
                           model_name: str = "Unknown") -> Dict:
    """
    Perform comprehensive evaluation including all metrics.
    
    Args:
        original_smiles: List of original molecule SMILES
        adversarial_smiles: List of adversarial molecule SMILES
        black_box_model: Primary black box model
        training_smiles: Training set SMILES for novelty calculation
        models_dict: Dictionary of additional models for transferability
        success_threshold: Threshold for attack success
        task_type: "regression" or "classification"
        save_path: Path to save results
        dataset_name: Name of dataset
        model_name: Name of target model
    
    Returns:
        Dictionary with all evaluation results
    """
    print("Performing comprehensive evaluation...")
    
    # Basic adversarial attack evaluation
    attack_results = evaluate_adversarial_attack(
        original_smiles, adversarial_smiles, black_box_model, 
        success_threshold, task_type
    )
    
    # Molecular quality evaluation
    quality_results = evaluate_molecular_quality(
        adversarial_smiles, training_smiles
    )
    
    # Transferability evaluation (if additional models provided)
    transferability_results = None
    if models_dict:
        transferability_results = evaluate_transferability(
            original_smiles, adversarial_smiles, models_dict, success_threshold
        )
    
    # Generate plots
    if save_path:
        plot_evaluation_results(attack_results, save_path)
        if transferability_results:
            plot_transferability_results(transferability_results, save_path)
    
    # Save comprehensive report
    if save_path:
        save_evaluation_report(
            attack_results, quality_results, transferability_results,
            save_path, dataset_name, model_name
        )
    
    # Combine all results
    comprehensive_results = {
        "attack_results": attack_results,
        "quality_results": quality_results,
        "transferability_results": transferability_results
    }
    
    return comprehensive_results