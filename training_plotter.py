import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import os

class TrainingPlotter:
    """Real-time training progress plotter with save functionality"""
    
    def __init__(self, save_path="results/training_progress.png"):
        self.save_path = save_path
        self.epochs = []
        self.total_losses = []
        self.adv_losses = []
        self.sim_losses = []
        self.validity_rates = []
        self.molecular_changes = []
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Set up the plot with 2x3 layout for separate validity and change rate plots
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('Adversarial Molecular Generator Training Progress', fontsize=16)
        
        # Hide the bottom right subplot (axes[1, 2])
        self.axes[1, 2].set_visible(False)
        
        # Individual subplot titles
        self.axes[0, 0].set_title('Total Loss')
        self.axes[0, 1].set_title('Adversarial Loss')
        self.axes[0, 2].set_title('Similarity Loss')
        self.axes[1, 0].set_title('Validity Rate')
        self.axes[1, 1].set_title('Molecular Change Rate')
        
        # Set up axis labels for visible plots only
        for i in range(2):
            for j in range(3):
                if not (i == 1 and j == 2):  # Skip the hidden plot
                    self.axes[i, j].set_xlabel('Epoch')
                    self.axes[i, j].grid(True, alpha=0.3)
        
        # Y-axis labels
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 2].set_ylabel('Loss')
        self.axes[1, 0].set_ylabel('Validity Rate (%)')
        self.axes[1, 1].set_ylabel('Change Rate (%)')
        
        plt.tight_layout()
        plt.show()
    
    def update_data(self, epoch, loss_dict, molecular_change_rate=None):
        """Update training data"""
        self.epochs.append(epoch)
        self.total_losses.append(loss_dict['total_loss'])
        self.adv_losses.append(loss_dict['adversarial_loss'])
        self.sim_losses.append(loss_dict['similarity_loss'])
        self.validity_rates.append(loss_dict['validity_rate'] * 100)  # Convert to percentage
        
        if molecular_change_rate is not None:
            self.molecular_changes.append(molecular_change_rate)
        else:
            self.molecular_changes.append(0)
    
    def update_plots(self):
        """Update all plots with current data"""
        if len(self.epochs) == 0:
            return
        
        # Clear visible axes only
        for i in range(2):
            for j in range(3):
                if not (i == 1 and j == 2):  # Skip the hidden plot
                    self.axes[i, j].clear()
                    self.axes[i, j].grid(True, alpha=0.3)
        
        # Plot 1: Total Loss
        self.axes[0, 0].plot(self.epochs, self.total_losses, 'b-', linewidth=2, label='Total Loss')
        self.axes[0, 0].set_title('Total Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        
        # Plot 2: Adversarial Loss
        self.axes[0, 1].plot(self.epochs, self.adv_losses, 'r-', linewidth=2, label='Adversarial Loss')
        self.axes[0, 1].set_title('Adversarial Loss')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].legend()
        
        # Plot 3: Similarity Loss
        self.axes[0, 2].plot(self.epochs, self.sim_losses, 'g-', linewidth=2, label='Similarity Loss')
        self.axes[0, 2].set_title('Similarity Loss')
        self.axes[0, 2].set_xlabel('Epoch')
        self.axes[0, 2].set_ylabel('Loss')
        self.axes[0, 2].legend()
        
        # Plot 4: Validity Rate (separate plot)
        self.axes[1, 0].plot(self.epochs, self.validity_rates, 'orange', linewidth=2, label='Validity Rate')
        self.axes[1, 0].set_title('Validity Rate')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Validity Rate (%)')
        self.axes[1, 0].legend()
        self.axes[1, 0].set_ylim(0, 105)  # Set y-axis limit for percentage
        
        # Plot 5: Molecular Change Rate (separate plot)
        self.axes[1, 1].plot(self.epochs, self.molecular_changes, 'teal', linewidth=2, label='Molecular Changes')
        self.axes[1, 1].set_title('Molecular Change Rate')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Change Rate (%)')
        self.axes[1, 1].legend()
        self.axes[1, 1].set_ylim(0, 105)  # Set y-axis limit for percentage
        
        # Add current values as text
        if len(self.epochs) > 0:
            current_epoch = self.epochs[-1]
            current_total = self.total_losses[-1]
            current_validity = self.validity_rates[-1]
            current_change = self.molecular_changes[-1]
            
            info_text = f"Epoch: {current_epoch} | Total Loss: {current_total:.4f} | Validity: {current_validity:.1f}% | Change Rate: {current_change:.1f}%"
            self.fig.suptitle(f'Adversarial Molecular Generator Training Progress\n{info_text}', fontsize=14)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to allow plot update
        
        # Save the plot
        self.save_plot()
    
    def save_plot(self):
        """Save the current plot to file"""
        try:
            self.fig.savefig(self.save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
        except Exception as e:
            print(f"Warning: Could not save plot to {self.save_path}: {e}")
    
    def save_final_plot(self):
        """Save the final plot with higher quality"""
        try:
            # Create a clean version for final save
            plt.ioff()  # Turn off interactive mode for final save
            
            # Create a new figure for final save (to avoid any interactive artifacts)
            final_fig, final_axes = plt.subplots(2, 3, figsize=(15, 8))
            final_fig.suptitle('Adversarial Molecular Generator Training Progress - Final Results', fontsize=16)
            
            # Hide the bottom right subplot in final figure too
            final_axes[1, 2].set_visible(False)
            
            if len(self.epochs) == 0:
                return
            
            # Plot all data on final figure
            # Plot 1: Total Loss
            final_axes[0, 0].plot(self.epochs, self.total_losses, 'b-', linewidth=2, label='Total Loss')
            final_axes[0, 0].set_title('Total Loss')
            final_axes[0, 0].set_xlabel('Epoch')
            final_axes[0, 0].set_ylabel('Loss')
            final_axes[0, 0].legend()
            final_axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Adversarial Loss
            final_axes[0, 1].plot(self.epochs, self.adv_losses, 'r-', linewidth=2, label='Adversarial Loss')
            final_axes[0, 1].set_title('Adversarial Loss')
            final_axes[0, 1].set_xlabel('Epoch')
            final_axes[0, 1].set_ylabel('Loss')
            final_axes[0, 1].legend()
            final_axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Similarity Loss
            final_axes[0, 2].plot(self.epochs, self.sim_losses, 'g-', linewidth=2, label='Similarity Loss')
            final_axes[0, 2].set_title('Similarity Loss')
            final_axes[0, 2].set_xlabel('Epoch')
            final_axes[0, 2].set_ylabel('Loss')
            final_axes[0, 2].legend()
            final_axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Validity Rate
            final_axes[1, 0].plot(self.epochs, self.validity_rates, 'orange', linewidth=2, label='Validity Rate')
            final_axes[1, 0].set_title('Validity Rate')
            final_axes[1, 0].set_xlabel('Epoch')
            final_axes[1, 0].set_ylabel('Validity Rate (%)')
            final_axes[1, 0].legend()
            final_axes[1, 0].grid(True, alpha=0.3)
            final_axes[1, 0].set_ylim(0, 105)
            
            # Plot 5: Molecular Change Rate
            final_axes[1, 1].plot(self.epochs, self.molecular_changes, 'teal', linewidth=2, label='Molecular Changes')
            final_axes[1, 1].set_title('Molecular Change Rate')
            final_axes[1, 1].set_xlabel('Epoch')
            final_axes[1, 1].set_ylabel('Change Rate (%)')
            final_axes[1, 1].legend()
            final_axes[1, 1].grid(True, alpha=0.3)
            final_axes[1, 1].set_ylim(0, 105)
            
            # Add final statistics
            if len(self.epochs) > 0:
                final_epoch = self.epochs[-1]
                final_total = self.total_losses[-1]
                final_validity = self.validity_rates[-1]
                final_change = self.molecular_changes[-1]
                
                info_text = f"Final Results - Epoch: {final_epoch} | Total Loss: {final_total:.4f} | Validity: {final_validity:.1f}% | Change Rate: {final_change:.1f}%"
                final_fig.suptitle(f'Adversarial Molecular Generator Training Progress\n{info_text}', fontsize=14)
            
            plt.tight_layout()
            
            # Save final plot
            final_fig.savefig(self.save_path, dpi=300, bbox_inches='tight', 
                            facecolor='white', edgecolor='none')
            
            # # Also save a timestamped version
            # import datetime
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # timestamped_path = self.save_path.replace('.png', f'_{timestamp}.png')
            # final_fig.savefig(timestamped_path, dpi=300, bbox_inches='tight', 
            #                 facecolor='white', edgecolor='none')
            
            print(f"Training progress plots saved to:")
            print(f"  - {self.save_path}")
            # print(f"  - {timestamped_path}")
            
            plt.close(final_fig)
            plt.ion()  # Turn interactive mode back on
            
        except Exception as e:
            print(f"Warning: Could not save final plot: {e}")