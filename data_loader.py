import os
import numpy as np
import pickle
from rdkit import Chem
import deepchem as dc
from datetime import datetime

def _to_xy(ds):
    """Return (list_of_smiles, numpy_targets) for a DeepChem Dataset."""
    smiles = [s.replace(" ", "") for s in ds.ids]  # Remove spaces from SMILES
    y = ds.y
    # Ensure 2‑D targets (n_samples, n_tasks)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return smiles, y

def load_qm7(root="data/qm7"):
    """Load QM7 via DeepChem."""
    tasks, (tr, va, te), _ = dc.molnet.load_qm7(
        featurizer="Raw",        # keeps original SMILES in ds.ids
        splitter="random",       # 80/10/10 split below
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        data_dir=root,
        reload=False,
    )
    # QM7 has 1 regression task by default – keep it.
    return tasks, *map(_to_xy, (tr, va, te))

def load_bbbp(root="data/bbbp"):
    """Load BBBP (blood‑brain barrier penetration) with DeepChem."""
    tasks, (tr, va, te), _ = dc.molnet.load_bbbp(
        featurizer="Raw",
        splitter="scaffold",    # more realistic split for classification
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        data_dir=root,
        reload=False,
    )
    return tasks, *map(_to_xy, (tr, va, te))

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

def load_bace(root="data/bace"):
    """Load BACE (classification dataset) via DeepChem."""
    tasks, (tr, va, te), _ = dc.molnet.load_bace_classification(
        featurizer="Raw",        # keeps original SMILES in ds.ids
        splitter="random",       # 80/10/10 split below
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        data_dir=root,
        reload=False,
    )
    # BACE has 1 classification task by default – keep it.
    return tasks, *map(_to_xy, (tr, va, te))

# Dataset configuration matching train.py
DATASETS = {
    "qm7": dict(loader=load_qm7, task_type="regression",
                metric_name="mae", better="lower"),
    "bbbp": dict(loader=load_bbbp, task_type="classification",
                 metric_name="roc_auc", better="higher"),
    "delaney": dict(loader=load_delaney, task_type="regression",
                    metric_name="mae", better="lower"),
    "bace": dict(loader=load_bace, task_type="classification",
                 metric_name="roc_auc", better="higher"),
}

def get_filtered_dataset_path(dataset_name, subset, max_molecules=None):
    """Generate path for saved filtered dataset"""
    os.makedirs("data/filtered_datasets", exist_ok=True)
    
    if max_molecules:
        filename = f"{dataset_name}_{subset}_max{max_molecules}_jtvae_filtered.pkl"
    else:
        filename = f"{dataset_name}_{subset}_jtvae_filtered.pkl"
    
    return os.path.join("data/filtered_datasets", filename)

def save_filtered_dataset(smiles_list, targets, cfg, filepath):
    """Save filtered dataset to pickle file"""
    dataset_data = {
        'smiles': smiles_list,
        'targets': targets,
        'config': cfg,
        'filtered_at': datetime.now().isoformat(),
        'num_molecules': len(smiles_list)
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset_data, f)
    
    print(f"Filtered dataset saved to: {filepath}")

def load_filtered_dataset(filepath):
    """Load filtered dataset from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            dataset_data = pickle.load(f)
        
        print(f"Loaded filtered dataset from: {filepath}")
        print(f"  Filtered at: {dataset_data['filtered_at']}")
        print(f"  Number of molecules: {dataset_data['num_molecules']}")
        
        return dataset_data['smiles'], dataset_data['targets'], dataset_data['config']
    
    except (FileNotFoundError, pickle.PickleError, KeyError) as e:
        print(f"Could not load filtered dataset from {filepath}: {e}")
        return None, None, None

def load_molecules_from_dataset(dataset_name, subset, max_molecules=None, jtvae_wrapper=None, force_refilter=False):
    """Load molecules from specified dataset for adversarial training with caching"""
    
    # Check if filtered dataset already exists
    filtered_path = get_filtered_dataset_path(dataset_name, subset, max_molecules)
    
    if not force_refilter and os.path.exists(filtered_path):
        print(f"Found existing filtered dataset for {dataset_name} {subset}")
        smiles_list, targets, cfg = load_filtered_dataset(filtered_path)
        
        if smiles_list is not None:
            # Show some examples
            property_name = get_property_name(dataset_name)
            print(f"Example molecules from cached dataset:")
            for i, smiles in enumerate(smiles_list[:5]):
                target_value = targets[i][0] if len(targets[i]) > 0 else "N/A"
                if isinstance(target_value, (int, float)):
                    print(f"  {i+1}. {smiles} ({property_name}: {target_value:.3f})")
                else:
                    print(f"  {i+1}. {smiles} ({property_name}: {target_value})")
            
            return smiles_list, np.array(targets), cfg
    
    # If no cached dataset or force refiltering, perform filtering
    print(f"Loading {dataset_name.upper()} dataset...")
    
    # Get dataset configuration and loader
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    cfg = DATASETS[dataset_name]
    tasks, train, valid, test = cfg["loader"]()
    
    print(f"{dataset_name.upper()} dataset loaded:")
    print(f"  Tasks: {tasks}")
    print(f"  Task type: {cfg['task_type']}")
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
    rdkit_valid_count = 0
    
    print("Starting molecule filtering...")
    print("Progress: ", end="", flush=True)
    
    for i, smiles in enumerate(smiles_list):
        # Progress indicator
        if i % 100 == 0:
            print(f"{i}", end="", flush=True)
        elif i % 20 == 0:
            print(".", end="", flush=True)
        
        # First check RDKit validity
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        rdkit_valid_count += 1
        
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
            
            # Stop if we have enough molecules
            if max_molecules and len(valid_smiles) >= max_molecules:
                break
    
    print()  # New line after progress indicators
    
    if jtvae_wrapper is not None:
        print(f"JT-VAE filtering results:")
        print(f"  Total molecules tested: {i+1}")
        print(f"  RDKit valid molecules: {rdkit_valid_count}")
        print(f"  JT-VAE processable molecules: {jtvae_processable_count}")
        print(f"  Final dataset: {len(valid_smiles)} molecules")
        print(f"  JT-VAE success rate: {(jtvae_processable_count/rdkit_valid_count*100):.1f}%")
    else:
        print(f"RDKit filtering results:")
        print(f"  Total molecules tested: {i+1}")
        print(f"  Final dataset: {len(valid_smiles)} valid molecules")
        print(f"  RDKit success rate: {(len(valid_smiles)/len(smiles_list)*100):.1f}%")
    
    # Convert to numpy array
    valid_targets = np.array(valid_targets)
    
    # Save filtered dataset for future use
    try:
        save_filtered_dataset(valid_smiles, valid_targets, cfg, filtered_path)
    except Exception as e:
        print(f"Warning: Could not save filtered dataset: {e}")
    
    # Show some examples with appropriate property names
    property_name = get_property_name(dataset_name)
    print(f"Example molecules:")
    for i, smiles in enumerate(valid_smiles[:5]):
        target_value = valid_targets[i][0] if len(valid_targets[i]) > 0 else "N/A"
        if isinstance(target_value, (int, float)):
            print(f"  {i+1}. {smiles} ({property_name}: {target_value:.3f})")
        else:
            print(f"  {i+1}. {smiles} ({property_name}: {target_value})")
    
    return valid_smiles, valid_targets, cfg

def list_filtered_datasets():
    """List all available filtered datasets"""
    filtered_dir = "data/filtered_datasets"
    if not os.path.exists(filtered_dir):
        print("No filtered datasets directory found.")
        return
    
    files = [f for f in os.listdir(filtered_dir) if f.endswith('.pkl')]
    
    if not files:
        print("No filtered datasets found.")
        return
    
    print("Available filtered datasets:")
    for file in sorted(files):
        filepath = os.path.join(filtered_dir, file)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"  {file}")
            print(f"    Molecules: {data['num_molecules']}")
            print(f"    Filtered at: {data['filtered_at']}")
            print(f"    Task type: {data['config']['task_type']}")
        except Exception as e:
            print(f"  {file} (corrupted: {e})")

def clean_filtered_datasets(dataset_name=None):
    """Remove filtered datasets (optionally for specific dataset)"""
    filtered_dir = "data/filtered_datasets"
    if not os.path.exists(filtered_dir):
        print("No filtered datasets directory found.")
        return
    
    files = [f for f in os.listdir(filtered_dir) if f.endswith('.pkl')]
    
    if dataset_name:
        files = [f for f in files if f.startswith(dataset_name)]
    
    if not files:
        print(f"No filtered datasets found{f' for {dataset_name}' if dataset_name else ''}.")
        return
    
    for file in files:
        filepath = os.path.join(filtered_dir, file)
        os.remove(filepath)
        print(f"Removed: {file}")

def get_property_name(dataset_name):
    """Get the property name for display purposes"""
    property_names = {
        "qm7": "atomization energy",
        "bbbp": "BBB permeability",
        "delaney": "solubility",
        "bace": "BACE inhibition"
    }
    return property_names.get(dataset_name, "property")