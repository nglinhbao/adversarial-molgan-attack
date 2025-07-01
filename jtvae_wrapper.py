import sys
import os
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

# Add JT-VAE to path - using submodule approach like mol-cycle-gan
sys.path.append('./jtvae')
from jtvae import Vocab, MolTree, JTNNVAE

class Options:
    def __init__(self,
                 jtvae_path="./jtvae/",
                 hidden_size=450,
                 latent_size=56,
                 depth=3,
                 jtnn_model_path="molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4",
                 vocab_path="data/zinc/vocab.txt"):
        self.jtvae_path = jtvae_path
        self.vocab_path = os.path.join(jtvae_path, vocab_path)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.model_path = os.path.join(jtvae_path, jtnn_model_path)

class JTVAEWrapper:
    """Wrapper for JT-VAE encoder/decoder functionality - following mol-cycle-gan approach"""
    
    def __init__(self, jtvae_path="./jtvae/", hidden_size=450, latent_size=56, depth=3,
                 jtnn_model_path="molvae/vae_model/model.iter-4",
                 vocab_path="data/zinc/vocab.txt"):
        
        self.opts = Options(
            jtvae_path=jtvae_path,
            hidden_size=hidden_size,
            latent_size=latent_size,
            depth=depth,
            jtnn_model_path=jtnn_model_path,
            vocab_path=vocab_path
        )
        
        # Load model using the same approach as mol-cycle-gan
        self.model = self.load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
    
    def load_model(self):
        """Load JT-VAE model following mol-cycle-gan approach"""
        # Load vocabulary exactly like mol-cycle-gan
        vocab = [x.strip("\r\n ") for x in open(self.opts.vocab_path)]
        vocab = Vocab(vocab)
        
        hidden_size = int(self.opts.hidden_size)
        latent_size = int(self.opts.latent_size)
        depth = int(self.opts.depth)
        
        # Create model
        model = JTNNVAE(vocab, hidden_size, latent_size, depth)
        
        # Load pretrained weights if available
        if os.path.exists(self.opts.model_path):
            model.load_state_dict(torch.load(self.opts.model_path, map_location='cpu'))
            print(f"Loaded JT-VAE model from {self.opts.model_path}")
        else:
            print(f"Model file {self.opts.model_path} not found. Using random initialization.")
        
        model.eval()
        return model
    
    def encode(self, smiles_list):
        """Encode SMILES strings to latent vectors"""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        latent_vectors = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    latent_vectors.append(None)
                    continue
                
                # Create molecular tree
                mol_tree = MolTree(smiles)
                
                # Encode to latent space
                with torch.no_grad():
                    _, tree_vec, mol_vec = self.model.encode([mol_tree])
                    # Apply the mean transformations to get the actual latent vectors
                    tree_mean = self.model.T_mean(tree_vec)
                    mol_mean = self.model.G_mean(mol_vec)
                    # Concatenate the transformed vectors
                    latent_vec = torch.cat([tree_mean, mol_mean], dim=1)
                    latent_vectors.append(latent_vec.cpu().numpy())
            except Exception as e:
                print(f"Error encoding {smiles}: {e}")
                latent_vectors.append(None)
        
        return latent_vectors
    
    def decode(self, latent_vectors):
        """Decode latent vectors to SMILES strings - following mol-cycle-gan approach"""
        if isinstance(latent_vectors, np.ndarray):
            if len(latent_vectors.shape) == 1:
                latent_vectors = latent_vectors.reshape(1, -1)
            latent_vectors = torch.FloatTensor(latent_vectors)
        
        if len(latent_vectors.shape) == 1:
            latent_vectors = latent_vectors.unsqueeze(0)
        
        smiles_list = []
        tree_dims = int(self.opts.latent_size / 2)  # Following mol-cycle-gan split
        
        with torch.no_grad():
            for i in range(latent_vectors.shape[0]):
                try:
                    # Split latent vector like in mol-cycle-gan decode function
                    tree_vec = latent_vectors[i, 0:tree_dims].to(self.device)
                    mol_vec = latent_vectors[i, tree_dims:].to(self.device)
                    
                    # Add batch dimension
                    tree_vec = tree_vec.unsqueeze(0)
                    mol_vec = mol_vec.unsqueeze(0)
                    
                    # Decode from latent space - model.decode returns a string directly
                    decoded_smiles = self.model.decode(tree_vec, mol_vec, prob_decode=False)
                    
                    # Handle the decoded SMILES
                    if isinstance(decoded_smiles, str) and decoded_smiles.strip():
                        smiles_list.append(decoded_smiles)
                    elif isinstance(decoded_smiles, list) and len(decoded_smiles) > 0:
                        smiles_list.append(decoded_smiles[0])
                    else:
                        smiles_list.append(None)
                except Exception as e:
                    print(f"Error decoding latent vector: {e}")
                    smiles_list.append(None)
        
        return smiles_list
    
    def get_latent_size(self):
        """Get the actual size of the latent space by checking a test encoding"""
        test_smiles = "C"  # Simple methane
        test_encoding = self.encode_to_numpy([test_smiles])
        if test_encoding.size > 0:
            return test_encoding.shape[1]
        return self.opts.latent_size  # Fallback to parameter if encoding fails
    
    def encode_to_numpy(self, smiles_list):
        """Encode SMILES to numpy arrays for compatibility with mol-cycle-gan format"""
        latent_vectors = self.encode(smiles_list)
        valid_vectors = [v for v in latent_vectors if v is not None]
        
        if valid_vectors:
            return np.vstack(valid_vectors)
        else:
            return np.array([])
    
    def decode_from_numpy(self, latent_array):
        """Decode from numpy array format (mol-cycle-gan compatibility)"""
        if latent_array.size == 0:
            return []
        
        return self.decode(latent_array)
