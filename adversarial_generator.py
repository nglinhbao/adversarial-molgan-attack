import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from typing import List, Optional
from training_plotter import TrainingPlotter
from matplotlib import pyplot as plt
from datetime import datetime
import os
import csv

class AdversarialGenerator(nn.Module):
    """Generator network for adversarial perturbations in latent space"""
    
    def __init__(self, latent_size, hidden_size=512, num_layers=5, actual_input_size=None):
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
        
        # Create the network architecture with more layers for GPU utilization
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Add batch norm for better GPU utilization
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size
        
        # Final layer to output perturbation
        layers.append(nn.Linear(current_size, latent_size))
        layers.append(nn.Tanh())  # Bounded perturbation
        
        self.network = nn.Sequential(*layers)
        
        # Scale factor for perturbations (start smaller for stability)
        self.perturbation_scale = 5.0  # Increased from 1.0 for better molecular diversity
        self.use_random_noise = False  # Option to use random noise instead of learned perturbations
        self.adaptive_scaling = False  # Disable adaptive scaling since 5.0x works well
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
    
    def __init__(self, jtvae_wrapper, black_box_model, learning_rate, results_name=None):
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

        # Optimizer for the generator with improved settings
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=learning_rate,
            betas=(0.5, 0.999),  # More stable for adversarial training
            weight_decay=1e-5    # Small regularization
        )
        
        # Learning rate scheduler for stability
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.generator_optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            # verbose=True,
            min_lr=1e-6
        )
        
        # Exponential moving averages for loss tracking
        self.ema_alpha = 0.1
        self.adv_loss_ema = None
        self.sim_loss_ema = None
        
        # Early stopping parameters
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 15  # Stop if no improvement for 15 epochs

        # Create results directory with provided name or default timestamp
        if results_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_name = f"adversarial_training_{timestamp}"
        
        self.results_path = os.path.join('results', results_name)
        os.makedirs(self.results_path, exist_ok=True)
        print(f"Results will be saved to: {self.results_path}")

        # Initialize plotter
        self.plotter = TrainingPlotter(os.path.join(self.results_path, 'training_progress.png'))
        
        # Initialize CSV logging
        self.csv_file_path = os.path.join(self.results_path, 'training_results.csv')
        self.csv_fieldnames = [
            'epoch', 
            'total_loss', 
            'adversarial_loss', 
            'similarity_loss', 
            'validity_rate', 
            'molecular_change_rate',
            'learning_rate',
            'timestamp'
        ]
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        try:
            with open(self.csv_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
                writer.writeheader()
            print(f"Initialized CSV log file: {self.csv_file_path}")
        except Exception as e:
            print(f"Warning: Could not initialize CSV file: {e}")
    
    def _log_to_csv(self, epoch, loss_dict, change_rate, learning_rate):
        """Log training results to CSV file"""
        try:
            # Prepare row data
            row_data = {
                'epoch': epoch,
                'total_loss': loss_dict.get('total_loss', 0.0),
                'adversarial_loss': loss_dict.get('adversarial_loss', 0.0),
                'similarity_loss': loss_dict.get('similarity_loss', 0.0),
                'validity_rate': loss_dict.get('validity_rate', 0.0),
                'molecular_change_rate': change_rate,
                'learning_rate': learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            # Append to CSV file
            with open(self.csv_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
                writer.writerow(row_data)
                
        except Exception as e:
            print(f"Warning: Could not log to CSV file: {e}")
    
    def get_training_statistics(self):
        """Load and return training statistics from CSV file"""
        try:
            if os.path.exists(self.csv_file_path):
                df = pd.read_csv(self.csv_file_path)
                return df
            else:
                print("No training results CSV file found.")
                return None
        except Exception as e:
            print(f"Error loading training statistics: {e}")
            return None
    
    def save_training_summary(self):
        """Save a summary of training results"""
        try:
            df = self.get_training_statistics()
            if df is not None and len(df) > 0:
                summary_path = os.path.join(self.results_path, 'training_summary.txt')
                
                with open(summary_path, 'w') as f:
                    f.write("Training Results Summary\n")
                    f.write("========================\n\n")
                    f.write(f"Total epochs: {len(df)}\n")
                    f.write(f"Final total loss: {df['total_loss'].iloc[-1]:.6f}\n")
                    f.write(f"Best total loss: {df['total_loss'].min():.6f}\n")
                    f.write(f"Final adversarial loss: {df['adversarial_loss'].iloc[-1]:.6f}\n")
                    f.write(f"Final similarity loss: {df['similarity_loss'].iloc[-1]:.6f}\n")
                    f.write(f"Final validity rate: {df['validity_rate'].iloc[-1]:.2%}\n")
                    f.write(f"Final molecular change rate: {df['molecular_change_rate'].iloc[-1]:.1f}%\n")
                    f.write(f"Average validity rate: {df['validity_rate'].mean():.2%}\n")
                    f.write(f"Average molecular change rate: {df['molecular_change_rate'].mean():.1f}%\n")
                    f.write(f"Final learning rate: {df['learning_rate'].iloc[-1]:.2e}\n")
                    
                print(f"Training summary saved to: {summary_path}")
                
        except Exception as e:
            print(f"Error saving training summary: {e}")
    
    # Convert numpy arrays to torch tensors for adversarial loss

    def adversarial_loss(self, original_pred, adversarial_pred, target_pred=None, success_threshold=0.2):
        """
        Compute adversarial loss with improved stability
        
        Args:
            success_threshold: For regression, the minimum relative change needed to consider attack successful (default: 20%)
        """
        original_pred = torch.tensor(original_pred, device=self.device)
        adversarial_pred = torch.tensor(adversarial_pred, device=self.device)

        if self.black_box_model.model_type == 'classification':
            # For classification: use smooth loss that encourages change but avoids instability
            prediction_diff = torch.abs(original_pred - adversarial_pred).mean()
            # Use negative log to encourage larger differences, but with stability
            return -torch.log(prediction_diff + 1e-6)
        else:
            # For regression: improved threshold-based success criterion
            if target_pred is not None:
                target_pred = torch.tensor(target_pred, device=self.device)
                return torch.abs(adversarial_pred - target_pred).mean()
            else:
                # Calculate relative change as percentage
                prediction_diff = torch.abs(original_pred - adversarial_pred)
                relative_change = prediction_diff / (torch.abs(original_pred) + 1e-8)
                
                # Smooth transition around success threshold using sigmoid
                # This avoids sharp discontinuities that cause instability
                sigmoid_factor = 10.0  # Controls smoothness of transition
                success_probability = torch.sigmoid(sigmoid_factor * (relative_change - success_threshold))
                
                # Combine smooth success and failure components
                success_loss = 0.1  # Low loss when successful
                failure_loss = torch.clamp(success_threshold - relative_change, min=0.0) / success_threshold
                
                # Smoothly interpolate between success and failure loss
                total_loss = success_probability * success_loss + (1 - success_probability) * failure_loss
                
                return total_loss.mean()

    
    def similarity_loss(self, original_z, adversarial_z):
        """Compute similarity loss in latent space"""
        return torch.norm(original_z - adversarial_z, p=1, dim=1).mean()
    
    def train_step(self, smiles_list, lambda_adv, lambda_sim, epoch, success_threshold):
        """Single training step adapted for mol-cycle-gan style encoding with GPU optimization"""
        # Encode original molecules using mol-cycle-gan style
        latent_array = self.jtvae.encode_to_numpy(smiles_list)

        if latent_array.size == 0:
            return None

        # Convert to tensor and keep on GPU for all operations
        original_z = torch.FloatTensor(latent_array).to(self.device)

        # Generate adversarial latent vectors - always use forward for better GPU utilization
        adversarial_z = self.generator(original_z)

        # Batch decode on CPU to minimize transfers
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

        # Compute similarity loss with L2 norm for smoother gradients
        sim_loss = torch.norm(original_z - adversarial_z, p=2, dim=1).mean()

        # Update exponential moving averages
        if self.adv_loss_ema is None:
            self.adv_loss_ema = adv_loss.item()
            self.sim_loss_ema = sim_loss.item()
        else:
            self.adv_loss_ema = self.ema_alpha * adv_loss.item() + (1 - self.ema_alpha) * self.adv_loss_ema
            self.sim_loss_ema = self.ema_alpha * sim_loss.item() + (1 - self.ema_alpha) * self.sim_loss_ema

        # Adaptive loss balancing using EMA to prevent one loss from dominating
        adv_scale = 1.0 / (self.adv_loss_ema + 1e-8)
        sim_scale = 1.0 / (self.sim_loss_ema + 1e-8)
        
        # Normalize scales
        total_scale = adv_scale + sim_scale
        adv_weight = adv_scale / total_scale
        sim_weight = sim_scale / total_scale
        
        # Total loss with adaptive weighting and original lambdas
        total_loss = lambda_adv * adv_weight * adv_loss + lambda_sim * sim_weight * sim_loss

        # Backpropagation with gradient clipping for stability
        self.generator_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
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
    
    def train(self, training_smiles, epochs, batch_size, lambda_adv, lambda_sim, success_threshold, plot_progress=True, save_checkpoints=True, checkpoint_interval=50):
        """Train the adversarial generator with validity discriminator and optional live plotting
        
        Args:
            training_smiles: List of SMILES strings for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            lambda_adv: Weight for adversarial loss
            lambda_sim: Weight for similarity loss
            success_threshold: Threshold for adversarial success
            plot_progress: Whether to show live plots during training
            save_checkpoints: Whether to save model checkpoints during training
            checkpoint_interval: Save checkpoint every N epochs
        """
        
        self.generator.train()
        
        # Print training info
        print(f"\n=== Training Configuration ===")
        print(f"Batch size: {batch_size}")
        print(f"Total batches per epoch: {len(training_smiles) // batch_size}")
                
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
                
                # Update learning rate scheduler
                self.scheduler.step(avg_total_loss)

                # Log to CSV
                current_lr = self.generator_optimizer.param_groups[0]['lr']
                avg_loss_dict = {
                    'total_loss': avg_total_loss,
                    'adversarial_loss': avg_adv_loss,
                    'similarity_loss': avg_sim_loss,
                    'validity_rate': avg_validity_rate
                }
                self._log_to_csv(epoch, avg_loss_dict, avg_change_rate, current_lr)
                
                # Early stopping check
                if avg_total_loss < self.best_loss:
                    self.best_loss = avg_total_loss
                    self.patience_counter = 0
                    # Save best model
                    if save_checkpoints:
                        best_model_path = f"best_generator_epoch_{epoch}.pth"
                        self.save_generator_model(best_model_path)
                        print(f"New best model saved at epoch {epoch}")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch} - no improvement for {self.early_stopping_patience} epochs")
                    break

                # Update live plot with epoch-averaged data
                if self.plotter is not None:
                    self.plotter.update_data(epoch, avg_loss_dict, avg_change_rate)
                    self.plotter.update_plots()
                
                # Save checkpoint at specified intervals
                if save_checkpoints and (epoch + 1) % checkpoint_interval == 0:
                    checkpoint_filename = f"generator_checkpoint_epoch_{epoch + 1}.pth"
                    self.save_generator_model(checkpoint_filename)
                    print(f"Checkpoint saved at epoch {epoch + 1}")
        
        # Save final training summary
        self.save_training_summary()
        
        # Save final plot and keep it open
        if self.plotter is not None:
            print("Training complete! Saving final training progress plot...")
            self.plotter.save_final_plot()
            try:
                plt.show(block=True)
            except KeyboardInterrupt:
                pass
        
        # Save the trained generator model
        self.save_generator_model()

    def save_generator_model(self, filename=None):
        """Save the trained generator model to disk"""
        if filename is None:
            filename = f"generator_model.pth"
        
        model_path = os.path.join(self.results_path, filename)
        
        # Save model state dict along with configuration
        save_dict = {
            'model_state_dict': self.generator.state_dict(),
            'model_config': {
                'latent_size': self.latent_size,
                'hidden_size': 256,  # Default from __init__
                'num_layers': 3,     # Default from __init__
                'actual_input_size': getattr(self.generator, 'input_size', self.latent_size),
                'perturbation_scale': self.generator.perturbation_scale,
                'adaptive_scaling': getattr(self.generator, 'adaptive_scaling', True),
                'use_random_noise': getattr(self.generator, 'use_random_noise', False)
            },
            'optimizer_state_dict': self.generator_optimizer.state_dict()
        }
        
        torch.save(save_dict, model_path)
        print(f"Generator model saved to: {model_path}")
        
        # Also save a human-readable summary
        summary_path = os.path.join(self.results_path, "model_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Adversarial Generator Model Summary\n")
            f.write("==================================\n\n")
            f.write(f"Model saved at: {model_path}\n")
            f.write(f"Latent size: {self.latent_size}\n")
            f.write(f"Actual input size: {save_dict['model_config']['actual_input_size']}\n")
            f.write(f"Hidden size: {save_dict['model_config']['hidden_size']}\n")
            f.write(f"Number of layers: {save_dict['model_config']['num_layers']}\n")
            f.write(f"Perturbation scale: {save_dict['model_config']['perturbation_scale']}\n")
            f.write(f"Adaptive scaling: {save_dict['model_config']['adaptive_scaling']}\n")
            f.write(f"Use random noise: {save_dict['model_config']['use_random_noise']}\n")
            f.write(f"\nModel architecture:\n{self.generator}\n")
        
        print(f"Model summary saved to: {summary_path}")
        return model_path

    def load_generator_model(self, model_path):
        """Load a previously saved generator model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        save_dict = torch.load(model_path, map_location=self.device)
        
        # Load model configuration
        model_config = save_dict.get('model_config', {})
        
        # Recreate generator with saved configuration
        self.generator = AdversarialGenerator(
            latent_size=model_config.get('latent_size', self.latent_size),
            hidden_size=model_config.get('hidden_size', 256),
            num_layers=model_config.get('num_layers', 3),
            actual_input_size=model_config.get('actual_input_size', None)
        )
        
        # Set additional attributes
        self.generator.perturbation_scale = model_config.get('perturbation_scale', 10.0)
        self.generator.adaptive_scaling = model_config.get('adaptive_scaling', True)
        self.generator.use_random_noise = model_config.get('use_random_noise', False)
        
        # Load model weights
        self.generator.load_state_dict(save_dict['model_state_dict'])
        self.generator.to(self.device)
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in save_dict:
            self.generator_optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        
        print(f"Generator model loaded from: {model_path}")
        return self.generator