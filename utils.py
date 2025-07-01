import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import pickle

def validate_smiles(smiles):
    """Validate SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def calculate_molecular_properties(smiles):
    """Calculate basic molecular properties"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Crippen.MolLogP(mol),
        'hbd': Lipinski.NumHDonors(mol),
        'hba': Lipinski.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol)
    }
    
    return properties

def save_model(model, path):
    """Save model to file"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load model from file"""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def save_results(results, path):
    """Save results to pickle file"""
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {path}")

def load_results(path):
    """Load results from pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded from {path}")
    return results

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
