#!/usr/bin/env python
"""
Train the torch‑molecule model zoo on QM9 or BBBP *with* DeepChem and save every
 tuned model into ./models.

$ python train_torch_molecule_deepchem.py --dataset qm9
$ python train_torch_molecule_deepchem.py --dataset bbbp
"""

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ 0. Imports                                                                 │
# ╰────────────────────────────────────────────────────────────────────────────╯
import argparse, pathlib, numpy as np, torch
import deepchem as dc
from sklearn.metrics import mean_absolute_error, roc_auc_score
import csv
from torch_molecule import (
    GRINMolecularPredictor, BFGNNMolecularPredictor, GREAMolecularPredictor, 
    IRMMolecularPredictor, RPGNNMolecularPredictor, GNNMolecularPredictor,
    SMILESTransformerMolecularPredictor, LSTMMolecularPredictor,
)


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ 1. DeepChem dataset helpers                                                │
# ╰────────────────────────────────────────────────────────────────────────────╯

# DeepChem already handles splitting, so we just convert each Dataset into a
# (smiles_list, y) pair that the torch‑molecule models accept.

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
    """Load delaney (solubility dataset) via DeepChem."""
    tasks, (tr, va, te), _ = dc.molnet.load_delaney(
        featurizer="Raw",        # keeps original SMILES in ds.ids
        splitter="random",       # 80/10/10 split below
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        data_dir=root,
        reload=False,
    )
    # delaney has 1 regression task by default – keep it.
    return tasks, *map(_to_xy, (tr, va, te))


def load_bace(root="data/bace"):
    """Load bace (classification dataset) via DeepChem."""
    tasks, (tr, va, te), _ = dc.molnet.load_bace_classification(
        featurizer="Raw",        # keeps original SMILES in ds.ids
        splitter="random",       # 80/10/10 split below
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        data_dir=root,
        reload=False,
    )
    # bace has 1 classification task by default – keep it.
    return tasks, *map(_to_xy, (tr, va, te))

# Update the DATASETS dictionary to include delaney and bace
DATASETS = {
    "qm7" : dict(loader=load_qm7,  task_type="regression",
                 metric_name="mae",     better="lower"),
    "bbbp": dict(loader=load_bbbp, task_type="classification",
                 metric_name="roc_auc", better="higher"),
    "delaney": dict(loader=load_delaney, task_type="regression",
                 metric_name="mae",     better="lower"),
    "bace" : dict(loader=load_bace,  task_type="classification",
                 metric_name="roc_auc", better="higher"),
}


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ 2. Model zoo (unchanged)                                                   │
# ╰────────────────────────────────────────────────────────────────────────────╯
MODEL_ZOO = {
    "GRIN"        : GRINMolecularPredictor,
    "BFGNN"       : BFGNNMolecularPredictor,
    "GREA"        : GREAMolecularPredictor,
    "IRM"         : IRMMolecularPredictor,
    "RPGNN"       : RPGNNMolecularPredictor,
    "GNNs"        : GNNMolecularPredictor,
    "Transformer" : SMILESTransformerMolecularPredictor,
    "LSTM"        : LSTMMolecularPredictor,
}


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ 3. Metrics (unchanged)                                                     │
# ╰────────────────────────────────────────────────────────────────────────────╯

def _metric(y_true, y_pred, name):
    if name == "mae":      return mean_absolute_error(y_true, y_pred)
    if name == "roc_auc":  return roc_auc_score(y_true, y_pred, average="weighted")
    raise ValueError(name)


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ 4. Training / saving (modified for IRM single-task)                       │
# ╰────────────────────────────────────────────────────────────────────────────╯

def train_eval_save(model_cls, cfg, tasks, train, valid, test,
                    out_dir, tag):
    """Train and evaluate a model, handling IRM with single tasks."""
    
    # Check if this is IRM model and multi-task scenario
    if model_cls.__name__ == "IRMMolecularPredictor" and len(tasks) > 1:
        return train_eval_save_irm_single_task(model_cls, cfg, tasks, train, valid, test, out_dir, tag)
    
    # Standard training for other models or single-task IRM
    model = model_cls(
        num_task=len(tasks),
        task_type=cfg["task_type"],
        model_name=model_cls.__name__,
        evaluate_criterion=cfg["metric_name"],
        evaluate_higher_better=(cfg["better"] == "higher"),
        verbose=False,
    )
    model.autofit(
        X_train=train[0],  y_train=train[1],
        X_val=valid[0],    y_val=valid[1],
        n_trials=1,       # adjust for speed if needed
    )
    y_pred = model.predict(test[0])["prediction"]
    score  = _metric(test[1], y_pred, cfg["metric_name"])

    out_dir.mkdir(exist_ok=True)
    fpath = out_dir / f"{tag}_{model_cls.__name__}.pt"
    try:
        model.save_to_local(str(fpath))
    except AttributeError:  # fallback for older versions
        torch.save(model, fpath)
    return score, fpath


def train_eval_save_irm_single_task(model_cls, cfg, tasks, train, valid, test, out_dir, tag):
    """Train IRM model one task at a time and average the results."""
    
    all_scores = []
    all_predictions = []
    saved_models = []
    
    for task_idx in range(len(tasks)):
        print(f"  Training IRM for task {task_idx + 1}/{len(tasks)}: {tasks[task_idx]}")
        
        # Extract single task data
        train_single = (train[0], train[1][:, task_idx:task_idx+1])
        valid_single = (valid[0], valid[1][:, task_idx:task_idx+1])
        test_single = (test[0], test[1][:, task_idx:task_idx+1])
        
        # Create model for single task
        model = model_cls(
            num_task=1,  # Single task
            task_type=cfg["task_type"],
            model_name=f"{model_cls.__name__}_task_{task_idx}",
            evaluate_criterion=cfg["metric_name"],
            evaluate_higher_better=(cfg["better"] == "higher"),
            verbose=False,
        )
        
        try:
            model.autofit(
                X_train=train_single[0],  y_train=train_single[1],
                X_val=valid_single[0],    y_val=valid_single[1],
                n_trials=1,
            )
            
            y_pred_single = model.predict(test_single[0])["prediction"]
            score_single = _metric(test_single[1], y_pred_single, cfg["metric_name"])
            all_scores.append(score_single)
            all_predictions.append(y_pred_single)
            
            # Save individual task model
            out_dir.mkdir(exist_ok=True)
            fpath = out_dir / f"{tag}_{model_cls.__name__}_task_{task_idx}.pt"
            try:
                model.save_to_local(str(fpath))
            except AttributeError:
                torch.save(model, fpath)
            saved_models.append(fpath)
            
        except Exception as e:
            print(f"    Task {task_idx} failed: {e}")
            # Use dummy values for failed tasks
            all_scores.append(float('inf') if cfg["better"] == "lower" else 0.0)
            all_predictions.append(np.zeros_like(test_single[1]))
    
    # Calculate average score
    avg_score = np.mean(all_scores)
    
    # Return average score and list of saved model paths
    return avg_score, saved_models


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ 5. Main (minor tweaks)                                                     │
# ╰────────────────────────────────────────────────────────────────────────────╯

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--dataset", choices=DATASETS, default="bbbp")
    args = argp.parse_args()

    cfg = DATASETS[args.dataset]
    tasks, train, valid, test = cfg["loader"]()

    print(f"\nDataset: {args.dataset.upper()} | Tasks: {len(tasks)} | "
          f"Metric: {cfg['metric_name'].upper()}\n" + "-"*72)

    # Prepare CSV file for saving results
    csv_file = pathlib.Path("models") / f"{args.dataset}_results.csv"
    csv_file.parent.mkdir(exist_ok=True)  # Ensure the directory exists
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Metric", "Score", "Direction", "Saved File"])

        for name, cls in MODEL_ZOO.items():
            try:
                score, saved = train_eval_save(
                    cls, cfg, tasks, train, valid, test,
                    pathlib.Path("models"), tag=args.dataset
                )
                arrow = "↑" if cfg["better"] == "higher" else "↓"
                
                # Handle different saved file formats (single file vs list of files)
                if isinstance(saved, list):
                    saved_files_str = ", ".join([f.name for f in saved])
                    print(f"{name:<13} {cfg['metric_name'].upper():>8}: "
                          f"{score:.4f} {arrow}  📝 {saved_files_str}")
                else:
                    saved_files_str = saved.name
                    print(f"{name:<13} {cfg['metric_name'].upper():>8}: "
                          f"{score:.4f} {arrow}  📝 {saved_files_str}")
                
                # Write results to CSV
                writer.writerow([name, cfg["metric_name"].upper(), f"{score:.4f}", arrow, saved_files_str])
            except Exception as e:
                print(f"{name:<13} ⚠️  skipped  ({e})")
                writer.writerow([name, cfg["metric_name"].upper(), "skipped", "-", "-"])

    print(f"\nResults saved to: {csv_file}")


if __name__ == "__main__":
    main()