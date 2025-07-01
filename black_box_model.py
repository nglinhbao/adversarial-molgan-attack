import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Union, List

# Suppress RDKit warnings about invalid molecules
# RDLogger.DisableLog('rdApp.*')

# Import torch-molecule classes
try:
    from torch_molecule import (
        GRINMolecularPredictor, BFGNNMolecularPredictor, GREAMolecularPredictor, 
        IRMMolecularPredictor, RPGNNMolecularPredictor, GNNMolecularPredictor,
        SMILESTransformerMolecularPredictor, LSTMMolecularPredictor,
    )
except ImportError:
    print("Warning: torch-molecule not installed. Some functionality may be limited.")
    # Define dummy classes for fallback
    class GRINMolecularPredictor: pass
    class BFGNNMolecularPredictor: pass
    class GREAMolecularPredictor: pass
    class IRMMolecularPredictor: pass
    class RPGNNMolecularPredictor: pass
    class GNNMolecularPredictor: pass
    class SMILESTransformerMolecularPredictor: pass
    class LSTMMolecularPredictor: pass

class BlackBoxModel:
    """A black box model wrapper that can load and use pre-trained molecular predictors."""
    
    def __init__(self, model_path: str, model_type: str):
        """
        Initialize the black box model with a pre-trained model.
        
        Args:
            model_path: Path to the saved model (.pt file)
        """
        self.model_path = model_path
        self.model_type = model_type  # 'regression' or 'classification'
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # First, try to load the checkpoint to inspect its structure
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            print(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
                
                # Extract model name from checkpoint or filename
                if 'model_name' in checkpoint:
                    model_name = checkpoint['model_name']
                else:
                    # Extract from filename (e.g., bace_BFGNNMolecularPredictor.pt)
                    filename = os.path.basename(self.model_path)
                    if '_' in filename:
                        model_name = filename.split('_', 1)[1].replace('.pt', '')
                    else:
                        model_name = filename.replace('.pt', '')
                
                print(f"Model name: {model_name}")
            
            # Map model names to classes
            model_classes = {
                'GRINMolecularPredictor': GRINMolecularPredictor,
                'BFGNNMolecularPredictor': BFGNNMolecularPredictor,
                'GREAMolecularPredictor': GREAMolecularPredictor,
                'IRMMolecularPredictor': IRMMolecularPredictor,
                'RPGNNMolecularPredictor': RPGNNMolecularPredictor,
                'GNNMolecularPredictor': GNNMolecularPredictor,
                'SMILESTransformerMolecularPredictor': SMILESTransformerMolecularPredictor,
                'LSTMMolecularPredictor': LSTMMolecularPredictor,
            }
            
            # Find the correct model class
            model_class = None
            for class_name, cls in model_classes.items():
                if class_name in model_name:
                    model_class = cls
                    break
            
            if model_class is None:
                raise ValueError(f"Unknown model type in filename: {model_name}")
            
            print(f"Using model class: {model_class.__name__}")
            
            # Try different methods to load the model
            try:
                # Method 1: Use load_from_local as a class method
                self.model = model_class.load_from_local(path=self.model_path)
            except Exception as e1:
                print(f"Method 1 failed: {e1}")
                try:
                    # Method 2: Create instance and load
                    self.model = model_class()
                    self.model.load_from_local(self.model_path)
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    try:
                        # Method 3: Load from checkpoint manually
                        if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
                            hyperparams = checkpoint['hyperparameters']
                            print(f"Hyperparameters: {hyperparams}")
                            
                            # Create model with hyperparameters
                            self.model = model_class(**hyperparams)
                            
                            # Load state dict if available
                            if 'model_state_dict' in checkpoint:
                                self.model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            raise Exception("No hyperparameters found in checkpoint")
                    except Exception as e3:
                        print(f"Method 3 failed: {e3}")
                        raise Exception(f"All loading methods failed: {e1}, {e2}, {e3}")
            
            # Ensure model is in eval mode and on correct device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            print(f"Successfully loaded model: {type(self.model)}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def _preprocess_smiles(self, smiles_list: List[str]) -> List[str]:
        """
        Preprocess SMILES strings for the model.
        Based on the training code, models expect raw SMILES with spaces removed.
        """
        # Remove spaces from SMILES (as done in training)
        processed_smiles = [smiles.replace(" ", "") for smiles in smiles_list]
        return processed_smiles
    
    def predict(self, inputs: Union[str, List[str], Chem.Mol, List[Chem.Mol]]) -> Union[float, np.ndarray]:
        """
        Make predictions using the loaded model.
        
        Args:
            inputs: Input data (single SMILES string, single RDKit Mol, 
                    list of SMILES strings, or list of RDKit Mols)
            
        Returns:
            float or np.ndarray: Model predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Convert inputs to SMILES strings
            if isinstance(inputs, str):
                # Single SMILES string - use as is
                processed_inputs = self._preprocess_smiles([inputs])
                single_input = True
            elif hasattr(inputs, 'GetNumAtoms'):
                # Single RDKit Mol object - convert to SMILES
                smiles = Chem.MolToSmiles(inputs)
                if smiles is None:
                    raise ValueError(f"Could not convert RDKit Mol to SMILES: {inputs}")
                processed_inputs = self._preprocess_smiles([smiles])
                single_input = True
            elif isinstance(inputs, list):
                # List of inputs - convert each to SMILES
                smiles_list = []
                for item in inputs:
                    if isinstance(item, str):
                        # Already a SMILES string
                        smiles_list.append(item)
                    elif hasattr(item, 'GetNumAtoms'):
                        # RDKit Mol object - convert to SMILES
                        smiles = Chem.MolToSmiles(item)
                        if smiles is None:
                            print(f"Warning: Could not convert RDKit Mol to SMILES, skipping: {item}")
                            continue
                        smiles_list.append(smiles)
                    else:
                        print(f"Warning: Unsupported item type in list: {type(item)}, skipping")
                        continue
                
                if not smiles_list:
                    raise ValueError("No valid SMILES strings could be extracted from input list")
                    
                processed_inputs = self._preprocess_smiles(smiles_list)
                single_input = False
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}. Expected str, RDKit Mol, or List of either")
            
            # Use the model's predict method (torch-molecule models have this method)
            if hasattr(self.model, 'predict'):
                # torch-molecule models return a dict with 'prediction' key
                result = self.model.predict(processed_inputs)
                
                if isinstance(result, dict) and 'prediction' in result:
                    predictions = result['prediction']
                else:
                    predictions = result
            else:
                # Fallback: direct model call
                with torch.no_grad():
                    predictions = self.model(processed_inputs)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]
                    if hasattr(predictions, 'cpu'):
                        predictions = predictions.cpu().numpy()
            
            # Convert to numpy if tensor
            if hasattr(predictions, 'cpu'):
                predictions = predictions.cpu().numpy()
            elif torch.is_tensor(predictions):
                predictions = predictions.numpy()
            
            # Return single value for single input, array for batch
            if single_input:
                return float(predictions.squeeze())
            else:
                return predictions
                
        except Exception as e:
            print(f"Prediction error details: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_proba(self, inputs: Union[str, List[str], Chem.Mol, List[Chem.Mol]]) -> Union[float, np.ndarray]:
        """
        Get prediction probabilities (alias for predict for compatibility).
        
        Args:
            inputs: Input data (single SMILES string, single RDKit Mol, 
                    list of SMILES strings, or list of RDKit Mols)
            
        Returns:
            float or np.ndarray: Model prediction probabilities
        """
        return self.predict(inputs)