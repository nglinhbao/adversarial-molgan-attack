import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from typing import List, Optional
from training_plotter import TrainingPlotter
from matplotlib import pyplot as plt
from datetime import datetime
import os

class AdversarialGenerator(nn.Module):
    """Generator network for adversarial perturbations in latent space"""
    
    def __init__(self, latent_size, hidden_size=256, num_layers=3, actual_input_size=None):
        super(AdversarialGenerator, self).__init__()
        
        # Use actual input size if provided, otherwise use latent_size
        self.input_size = actual_input_size if actual_input_size is not None else latent_size
        self.latent_size = latent_size
        
        # Add a projection layer if input and latent sizes differ
        if self.input_size != self.latent_size:
            self.projection = nn.Linear(self.input_size, self.latent_size)
            print(f"Added projection layer from {self.input_size} to {self.latent_size}")
        else:
            self.projection = None
        
        layers = []
        current_size = latent_size
        
        # Create the network architecture
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size
        
        # Final layer to output perturbation
        layers.append(nn.Linear(current_size, latent_size))
        layers.append(nn.Tanh())  # Bounded perturbation
        
        self.network = nn.Sequential(*layers)
        
        # Scale factor for perturbations (can be adjusted)
        self.perturbation_scale = 10.0  # Further increased to ensure molecular diversity
        self.use_random_noise = False  # Option to use random noise instead of learned perturbations
        self.adaptive_scaling = True  # Disable adaptive scaling since 5.0x works well
        self.max_perturbation_attempts = 3  # Maximum attempts to find effective perturbations
        
    def forward(self, z):
        """Generate adversarial perturbation and add to input"""
        if self.use_random_noise:
            # Use random noise for testing
            perturbation = torch.randn_like(z) * self.perturbation_scale * 0.1
            return z + perturbation
            
        # Project input if needed
        if self.projection is not None:
            z_projected = self.projection(z)
            perturbation = self.network(z_projected)
            # Project perturbation back to original space if necessary
            if perturbation.shape != z.shape:
                # This is a simple approach - use transpose of projection matrix
                with torch.no_grad():
                    projection_transpose = nn.Linear(self.latent_size, self.input_size)
                    projection_transpose.weight.copy_(self.projection.weight.t())
                    perturbation = projection_transpose(perturbation)
        else:
            perturbation = self.network(z)
            
        # Apply scaling to perturbation
        perturbation = perturbation * self.perturbation_scale
        return z + perturbation
    
    def generate_adaptive_perturbation(self, z, jtvae_wrapper):
        """Generate perturbations with adaptive scaling to ensure molecular changes"""
        if not self.adaptive_scaling:
            return self.forward(z)
        
        # Get original molecules
        original_smiles = jtvae_wrapper.decode_from_numpy(z.cpu().numpy())
        
        # Try more aggressive perturbation scales
        scales = [self.perturbation_scale * 2, self.perturbation_scale * 4, self.perturbation_scale * 8, self.perturbation_scale * 12]
        
        for scale in scales:
            # Generate perturbation with current scale
            if self.projection is not None:
                z_projected = self.projection(z)
                perturbation = self.network(z_projected)
                if perturbation.shape != z.shape:
                    with torch.no_grad():
                        projection_transpose = nn.Linear(self.latent_size, self.input_size)
                        projection_transpose.weight.copy_(self.projection.weight.t())
                        perturbation = projection_transpose(perturbation)
            else:
                perturbation = self.network(z)
            
            # Apply current scale with some random noise to break symmetries
            noise = torch.randn_like(perturbation) * 0.1
            scaled_perturbation = (perturbation + noise) * scale
            adversarial_z = z + scaled_perturbation
            
            # Check if molecules changed
            adversarial_smiles = jtvae_wrapper.decode_from_numpy(adversarial_z.cpu().detach().numpy())
            
            # Count how many molecules actually changed
            changed_count = 0
            for orig, adv in zip(original_smiles, adversarial_smiles):
                if orig != adv and orig is not None and adv is not None:
                    changed_count += 1
            
            # Lower threshold - accept if at least 10% of molecules changed
            if changed_count >= max(1, len(original_smiles) * 0.1):
                return adversarial_z
        
        # If no scale worked well, use the largest scale with maximum noise
        final_perturbation = perturbation * scales[-1]
        noise = torch.randn_like(final_perturbation) * 0.3
        return z + final_perturbation + noise

class AdversarialMolCycleGAN:
    """Main class for adversarial molecular generation with validity discriminator"""
    
    def __init__(self, jtvae_wrapper, black_box_model, results_name=None):
        self.jtvae = jtvae_wrapper
        self.black_box_model = black_box_model
        
        # Get the latent size reported by the wrapper
        self.latent_size = jtvae_wrapper.get_latent_size()
        print(f"JTVAE reported latent size: {self.latent_size}")
        
        # Test encode a molecule to get the actual size
        test_smiles = "C"  # Simple methane molecule
        test_latent = jtvae_wrapper.encode_to_numpy([test_smiles])
        if test_latent.size > 0:
            actual_latent_size = test_latent.shape[1]
            print(f"Actual encoded latent size: {actual_latent_size}")
        else:
            actual_latent_size = self.latent_size
            print("Warning: Could not determine actual latent size, using reported size")
        
        # Initialize the generator
        self.generator = AdversarialGenerator(self.latent_size, actual_input_size=actual_latent_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)

        # Print model architecture for debugging
        print(f"Generator architecture:\n{self.generator}")

        # Optimizer for the generator
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.001)

        # Create results directory with provided name or default timestamp
        if results_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_name = f"adversarial_training_{timestamp}"
        
        self.results_path = os.path.join('results', results_name)
        os.makedirs(self.results_path, exist_ok=True)
        print(f"Results will be saved to: {self.results_path}")

        # Initialize plotter
        self.plotter = TrainingPlotter(os.path.join(self.results_path, 'training_progress.png'))
    
    def check_molecule_validity(
        self,
        smiles_list: List[Optional[str]]
    ) -> torch.FloatTensor:
        """
        Check which molecules are valid and return a tensor of validity labels (1.0 = valid, 0.0 = invalid).

        Args:
            smiles_list (List[Optional[str]]): List of SMILES strings (or None).

        Returns:
            torch.FloatTensor: 1D tensor of shape (len(smiles_list),) with 1.0 for valid molecules, 0.0 for invalid.
        """
        validity_labels: List[float] = []
        valid_count = 0
        total_count = 0

        for smiles in smiles_list:
            total_count += 1

            # Immediately invalid if missing
            if smiles is None:
                validity_labels.append(0.0)
                continue

            # RDKit parsing without automatic sanitization
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                validity_labels.append(0.0)
                continue

            try:
                # Full sanitization (valence, aromaticity, etc.)
                Chem.SanitizeMol(mol)

                # Round-trip SMILES to ensure consistency
                canon = Chem.MolToSmiles(mol, isomericSmiles=True)

                # Molecule is valid if it round-trips and has at least one atom
                if canon and mol.GetNumAtoms() > 0:
                    validity_labels.append(1.0)
                    valid_count += 1
                else:
                    validity_labels.append(0.0)
            except Exception:
                validity_labels.append(0.0)

        return torch.FloatTensor(validity_labels).to(self.device)
    
    # Convert numpy arrays to torch tensors for adversarial loss

    def adversarial_loss(self, original_pred, adversarial_pred, target_pred=None, success_threshold=0.2):
        """
        Compute adversarial loss - now designed to be minimized
        
        Args:
            success_threshold: For regression, the minimum relative change needed to consider attack successful (default: 20%)
        """
        original_pred = torch.tensor(original_pred, device=self.device)
        adversarial_pred = torch.tensor(adversarial_pred, device=self.device)

        if self.black_box_model.model_type == 'classification':
            # For classification: minimize negative prediction difference (maximize difference)
            prediction_diff = torch.abs(original_pred - adversarial_pred).mean()
            return 1.0 / (prediction_diff + 1e-8)  # Convert to minimization problem
        else:
            # For regression: threshold-based success criterion
            if target_pred is not None:
                target_pred = torch.tensor(target_pred, device=self.device)
                return torch.abs(adversarial_pred - target_pred).mean()
            else:
                # Calculate relative change as percentage
                prediction_diff = torch.abs(original_pred - adversarial_pred)
                relative_change = prediction_diff / (torch.abs(original_pred) + 1e-8)
                
                # If relative change exceeds threshold, attack is successful (low loss)
                # If below threshold, penalize proportionally to encourage reaching threshold
                success_mask = relative_change >= success_threshold
                
                # For successful attacks: small constant loss (attack achieved)
                success_loss = torch.ones_like(relative_change) * 0.1
                
                # For unsuccessful attacks: inversely proportional to progress toward threshold
                # This encourages reaching the threshold but doesn't push beyond unnecessarily
                failure_loss = (success_threshold - relative_change) / success_threshold
                
                # Combine losses based on success mask
                total_loss = torch.where(success_mask, success_loss, failure_loss)
                
                return total_loss.mean()

    
    def similarity_loss(self, original_z, adversarial_z):
        """Compute similarity loss in latent space"""
        return torch.norm(original_z - adversarial_z, p=1, dim=1).mean()
    
    def train_step(self, smiles_list, lambda_adv, lambda_sim, epoch, success_threshold):
        """Single training step adapted for mol-cycle-gan style encoding"""
        # Encode original molecules using mol-cycle-gan style
        latent_array = self.jtvae.encode_to_numpy(smiles_list)

        if latent_array.size == 0:
            return None

        # Convert to tensor
        original_z = torch.FloatTensor(latent_array).to(self.device)

        # Generate adversarial latent vectors with adaptive scaling
        if hasattr(self.generator, 'adaptive_scaling') and self.generator.adaptive_scaling:
            adversarial_z = self.generator.generate_adaptive_perturbation(original_z, self.jtvae)
        else:
            adversarial_z = self.generator(original_z)

        # Decode latent vectors to SMILES
        original_smiles = self.jtvae.decode_from_numpy(original_z.detach().cpu().numpy())
        adversarial_smiles = self.jtvae.decode_from_numpy(adversarial_z.detach().cpu().numpy())

        # Calculate molecular change rate
        total_molecules = len(original_smiles)
        changed_molecules = 0
        
        for orig_smi, adv_smi in zip(original_smiles, adversarial_smiles):
            if orig_smi is not None and adv_smi is not None and orig_smi != adv_smi:
                changed_molecules += 1
        
        molecular_change_rate = (changed_molecules / total_molecules * 100) if total_molecules > 0 else 0.0

        # Filter out None values and ensure matching lengths
        valid_pairs = []
        for orig_smi, adv_smi in zip(original_smiles, adversarial_smiles):
            if orig_smi is not None and adv_smi is not None:
                valid_pairs.append((orig_smi, adv_smi))
        
        if not valid_pairs:
            return None
        
        # Extract valid SMILES for prediction
        valid_original_smiles = [pair[0] for pair in valid_pairs]
        valid_adversarial_smiles = [pair[1] for pair in valid_pairs]

        # Compute adversarial loss
        original_preds = self.black_box_model.predict(valid_original_smiles)
        adversarial_preds = self.black_box_model.predict(valid_adversarial_smiles)
        adv_loss = self.adversarial_loss(original_preds, adversarial_preds, success_threshold)

        # Compute similarity loss
        sim_loss = self.similarity_loss(original_z, adversarial_z)

        # Total loss - both components are now positive and minimized
        # This prevents cancellation and ensures both objectives are optimized
        total_loss = lambda_adv * adv_loss + lambda_sim * sim_loss

        # Backpropagation
        self.generator_optimizer.zero_grad()
        total_loss.backward()
        self.generator_optimizer.step()

        # Calculate validity rate and molecular change rate
        validity_rate = len(valid_pairs) / len(original_smiles) if len(original_smiles) > 0 else 0.0
        
        # Calculate molecular change rate
        changed_molecules = 0
        for orig_smi, adv_smi in valid_pairs:
            if orig_smi != adv_smi:
                changed_molecules += 1
        molecular_change_rate = (changed_molecules / len(valid_pairs) * 100) if valid_pairs else 0.0

        # Don't plot here - move plotting to epoch level in train() method

        return {
            'adversarial_loss': adv_loss.item(),
            'similarity_loss': sim_loss.item(),
            'total_loss': total_loss.item(),
            'validity_rate': validity_rate,
            'molecular_change_rate': molecular_change_rate
        }
    
    def generate_adversarial(self, smiles, num_samples=1):
        """Generate adversarial molecules for a given SMILES"""
        self.generator.eval()
        
        with torch.no_grad():
            # Encode original molecule
            latent_array = self.jtvae.encode_to_numpy([smiles])
            
            if latent_array.size == 0:
                return []
            
            original_z = torch.FloatTensor(latent_array).to(self.device)
            
            adversarial_molecules = []
            for _ in range(num_samples):
                # Generate adversarial latent vector
                adversarial_z = self.generator(original_z)
                
                # Decode to SMILES
                adversarial_smiles = self.jtvae.decode_from_numpy(adversarial_z.cpu().numpy())
                
                if adversarial_smiles and adversarial_smiles[0]:
                    adversarial_molecules.append(adversarial_smiles[0])
            
            return adversarial_molecules
    
    def train(self, training_smiles, epochs, batch_size, lambda_adv, lambda_sim, success_threshold, plot_progress=True):
        """Train the adversarial generator with validity discriminator and optional live plotting"""
        self.generator.train()
                
        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_losses = []
            epoch_change_rates = []
            
            # Shuffle training data
            np.random.shuffle(training_smiles)
            
            for i in range(0, len(training_smiles), batch_size):
                batch_smiles = training_smiles[i:i+batch_size]
                
                loss_dict = self.train_step(
                    batch_smiles, 
                    lambda_adv=lambda_adv, 
                    lambda_sim=lambda_sim,
                    epoch=epoch,
                    success_threshold=success_threshold
                )
                
                if loss_dict and loss_dict['total_loss'] != float('inf'):
                    epoch_losses.append(loss_dict)
                    if 'molecular_change_rate' in loss_dict:
                        epoch_change_rates.append(loss_dict['molecular_change_rate'])
            
            # Print epoch statistics and update plot
            if epoch_losses:
                avg_total_loss = np.mean([l['total_loss'] for l in epoch_losses])
                avg_adv_loss = np.mean([l['adversarial_loss'] for l in epoch_losses])
                avg_sim_loss = np.mean([l['similarity_loss'] for l in epoch_losses])
                avg_validity_rate = np.mean([l['validity_rate'] for l in epoch_losses])
                avg_change_rate = np.mean(epoch_change_rates) if epoch_change_rates else 0

                print(f"Epoch {epoch}: Total Loss: {avg_total_loss:.4f}, "
                        f"Adv Loss: {avg_adv_loss:.4f}, Sim Loss: {avg_sim_loss:.4f}, "
                        f"Validity Rate: {avg_validity_rate:.2%}, "
                        f"Change Rate: {avg_change_rate:.1f}%")

                # Update live plot with epoch-averaged data
                if self.plotter is not None:
                    avg_loss_dict = {
                        'total_loss': avg_total_loss,
                        'adversarial_loss': avg_adv_loss,
                        'similarity_loss': avg_sim_loss,
                        'validity_rate': avg_validity_rate
                    }
                    self.plotter.update_data(epoch, avg_loss_dict, avg_change_rate)
                    self.plotter.update_plots()
        
        # Save final plot and keep it open
        if self.plotter is not None:
            print("Training complete! Saving final training progress plot...")
            self.plotter.save_final_plot()
            try:
                plt.show(block=True)
            except KeyboardInterrupt:
                pass