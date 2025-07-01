import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit import DataStructs
import matplotlib.pyplot as plt

def calculate_tanimoto_similarity(smiles1, smiles2):
    """Calculate Tanimoto similarity between two molecules"""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def evaluate_adversarial_attack(original_smiles, adversarial_smiles, black_box_model):
    """Evaluate the effectiveness of adversarial attack"""
    results = {
        'original_smiles': [],
        'adversarial_smiles': [],
        'tanimoto_similarity': [],
        'original_prediction': [],
        'adversarial_prediction': [],
        'prediction_difference': [],
        'attack_success': []
    }
    
    for orig_smi, adv_smi in zip(original_smiles, adversarial_smiles):
        if adv_smi is None:
            continue
        
        # Calculate similarity
        similarity = calculate_tanimoto_similarity(orig_smi, adv_smi)
        
        # Get predictions
        orig_pred = black_box_model.predict([orig_smi])[0]
        adv_pred = black_box_model.predict([adv_smi])[0]
        
        pred_diff = abs(orig_pred - adv_pred)
        
        # Determine attack success (threshold can be adjusted)
        if black_box_model.model_type == 'classification':
            attack_success = (orig_pred > 0.5) != (adv_pred > 0.5)  # Class flip
        else:
            attack_success = pred_diff > 0.1  # Significant change for regression
        
        results['original_smiles'].append(orig_smi)
        results['adversarial_smiles'].append(adv_smi)
        results['tanimoto_similarity'].append(similarity)
        results['original_prediction'].append(orig_pred)
        results['adversarial_prediction'].append(adv_pred)
        results['prediction_difference'].append(pred_diff)
        results['attack_success'].append(attack_success)
    
    return results

def plot_evaluation_results(evaluation_results):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Tanimoto similarity distribution
    axes[0, 0].hist(evaluation_results['tanimoto_similarity'], bins=20, alpha=0.7)
    axes[0, 0].set_xlabel('Tanimoto Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Molecular Similarities')
    
    # Prediction differences
    axes[0, 1].hist(evaluation_results['prediction_difference'], bins=20, alpha=0.7)
    axes[0, 1].set_xlabel('Prediction Difference')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prediction Differences')
    
    # Similarity vs Prediction Difference
    axes[1, 0].scatter(evaluation_results['tanimoto_similarity'], 
                      evaluation_results['prediction_difference'], alpha=0.6)
    axes[1, 0].set_xlabel('Tanimoto Similarity')
    axes[1, 0].set_ylabel('Prediction Difference')
    axes[1, 0].set_title('Similarity vs Prediction Difference')
    
    # Attack success rate
    success_rate = np.mean(evaluation_results['attack_success'])
    axes[1, 1].bar(['Failed', 'Successful'], 
                  [1-success_rate, success_rate], 
                  color=['red', 'green'], alpha=0.7)
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].set_title(f'Attack Success Rate: {success_rate:.2f}')
    
    plt.tight_layout()
    plt.savefig('results/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("=== Evaluation Summary ===")
    print(f"Total molecules evaluated: {len(evaluation_results['original_smiles'])}")
    print(f"Average Tanimoto similarity: {np.mean(evaluation_results['tanimoto_similarity']):.3f}")
    print(f"Average prediction difference: {np.mean(evaluation_results['prediction_difference']):.3f}")
    print(f"Attack success rate: {success_rate:.3f}")
    print(f"High similarity (>0.8) attacks: {np.sum(np.array(evaluation_results['tanimoto_similarity']) > 0.8)}")
