#!/usr/bin/env python3
"""
PyTorch Checkpoint Layer Extraction and Visualization Script

This script extracts layers from PyTorch checkpoint files and saves them as images
to visualize what features the neural network has learned during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import cv2
from PIL import Image
import os
from collections import defaultdict

class CheckpointLayerExtractor:
    """Extracts and visualizes layers from PyTorch checkpoint files."""
    
    def __init__(self, checkpoint_path, output_dir="checkpoint_visualizations"):
        """
        Initialize the extractor.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            output_dir (str): Directory to save visualizations
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self):
        """Load the checkpoint file."""
        print(f"Loading checkpoint: {self.checkpoint_path}")
        
        try:
            # Load on CPU to avoid GPU memory issues
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            print("Checkpoint loaded successfully!")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def analyze_checkpoint_structure(self, checkpoint):
        """Analyze and print the structure of the checkpoint."""
        print("\n" + "="*80)
        print("CHECKPOINT STRUCTURE ANALYSIS")
        print("="*80)
        
        # Print top-level keys
        print("Top-level keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key])}")
        
        # Analyze model state dict if available
        if 'model_state_dict' in checkpoint:
            print(f"\nModel State Dict ({len(checkpoint['model_state_dict'])} parameters):")
            state_dict = checkpoint['model_state_dict']
            
            # Group parameters by layer type
            layer_groups = {
                'encoder': [],
                'decoder': [],
                'attention': [],
                'embedding': [],
                'linear': [],
                'conv': [],
                'other': []
            }
            
            for name, param in state_dict.items():
                if 'encoder' in name.lower():
                    layer_groups['encoder'].append((name, param.shape))
                elif 'decoder' in name.lower():
                    layer_groups['decoder'].append((name, param.shape))
                elif 'attention' in name.lower() or 'attn' in name.lower():
                    layer_groups['attention'].append((name, param.shape))
                elif 'embedding' in name.lower() or 'embed' in name.lower():
                    layer_groups['embedding'].append((name, param.shape))
                elif 'linear' in name.lower() or 'fc' in name.lower():
                    layer_groups['linear'].append((name, param.shape))
                elif 'conv' in name.lower():
                    layer_groups['conv'].append((name, param.shape))
                else:
                    layer_groups['other'].append((name, param.shape))
            
            # Print grouped layers
            for group_name, layers in layer_groups.items():
                if layers:
                    print(f"\n{group_name.upper()} LAYERS ({len(layers)}):")
                    for name, shape in layers[:10]:  # Show first 10
                        print(f"  {name}: {shape}")
                    if len(layers) > 10:
                        print(f"  ... and {len(layers) - 10} more")
        
        return checkpoint
    
    def tensor_to_image(self, tensor, title="Layer Visualization", normalize=True):
        """
        Convert a tensor to a visualizable image.
        
        Args:
            tensor (torch.Tensor): Input tensor
            title (str): Title for the visualization
            normalize (bool): Whether to normalize the tensor values
            
        Returns:
            numpy.ndarray: Image array
        """
        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().numpy()
        else:
            data = tensor
        
        # Handle different tensor shapes
        if len(data.shape) == 4:  # Conv weights: (out_channels, in_channels, height, width)
            # Visualize first few filters
            data = data[:16]  # Take first 16 filters
            # Arrange in a grid
            grid_size = int(np.ceil(np.sqrt(len(data))))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            fig.suptitle(title, fontsize=16)
            
            for i in range(grid_size * grid_size):
                row, col = i // grid_size, i % grid_size
                if i < len(data):
                    # For multi-channel, take mean across input channels
                    if data[i].shape[0] > 1:
                        img = np.mean(data[i], axis=0)
                    else:
                        img = data[i][0]
                    
                    if normalize:
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    
                    axes[row, col].imshow(img, cmap='viridis')
                    axes[row, col].set_title(f'Filter {i}', fontsize=8)
                    axes[row, col].axis('off')
                else:
                    axes[row, col].axis('off')
            
        elif len(data.shape) == 3:  # 3D tensor
            # Take mean across one dimension
            data = np.mean(data, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            if normalize:
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            ax.set_title(title, fontsize=14)
            plt.colorbar(im, ax=ax)
            
        elif len(data.shape) == 2:  # 2D tensor (weight matrix)
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            if normalize:
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Input Dimension')
            ax.set_ylabel('Output Dimension')
            plt.colorbar(im, ax=ax)
            
        elif len(data.shape) == 1:  # 1D tensor (bias, embeddings)
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.plot(data)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
        else:
            print(f"Cannot visualize tensor with shape {data.shape}")
            return None
        
        plt.tight_layout()
        
        # Convert to image array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img_array
    
    def create_weight_distribution_plot(self, tensor, title="Weight Distribution"):
        """Create a histogram of weight values."""
        data = tensor.detach().cpu().numpy().flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title(f'{title} - Distribution')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f"""
        Mean: {np.mean(data):.4f}
        Std: {np.std(data):.4f}
        Min: {np.min(data):.4f}
        Max: {np.max(data):.4f}
        Shape: {tensor.shape}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Convert to image array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img_array
    
    def visualize_weights(self, weights, param_name, category):
        """
        Create visualizations for weight matrices and tensors.
        
        Args:
            weights (torch.Tensor): The weight tensor to visualize
            param_name (str): Name of the parameter
            category (str): Category of the layer
        """
        weights_np = weights.detach().cpu().numpy()
        
        # Handle different types of layers
        if category == 'conv_layers':
            return self.visualize_conv_kernels(weights_np, param_name)
        elif category == 'attention_layers':
            return self.visualize_attention_weights(weights_np, param_name)
        elif category == 'batch_norm_layers':
            return self.visualize_batch_norm_params(weights_np, param_name)
        elif category == 'embedding_layers':
            return self.visualize_embedding_matrix(weights_np, param_name)
        else:
            return self.visualize_weight_matrix(weights_np, param_name)
    
    def visualize_conv_kernels(self, weights, param_name):
        """Visualize convolutional kernels as grids of filters."""
        if len(weights.shape) != 4:  # Should be [out_channels, in_channels, height, width]
            return self.visualize_weight_matrix(weights, param_name)
        
        out_channels, in_channels, kh, kw = weights.shape
        
        # Create a grid visualization of the filters
        grid_size = int(np.ceil(np.sqrt(out_channels)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'Convolutional Kernels: {param_name}\n{out_channels} filters, {in_channels} input channels, {kh}x{kw} kernel', fontsize=14)
        
        for i in range(grid_size * grid_size):
            row, col = i // grid_size, i % grid_size
            ax = axes[row, col] if grid_size > 1 else axes
            
            if i < out_channels:
                # Average across input channels for visualization
                kernel_vis = np.mean(weights[i], axis=0)
                im = ax.imshow(kernel_vis, cmap='viridis', aspect='equal')
                ax.set_title(f'Filter {i}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def visualize_attention_weights(self, weights, param_name):
        """Visualize attention weight matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Weight matrix heatmap
        im1 = axes[0].imshow(weights, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Attention Weights Matrix\n{param_name}')
        axes[0].set_xlabel('Input Dimension')
        axes[0].set_ylabel('Output Dimension')
        plt.colorbar(im1, ax=axes[0])
        
        # Weight distribution histogram
        axes[1].hist(weights.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1].set_title('Weight Distribution')
        axes[1].set_xlabel('Weight Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_batch_norm_params(self, weights, param_name):
        """Visualize batch normalization parameters."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parameter values plot
        axes[0].plot(weights, 'o-', markersize=3, linewidth=1)
        axes[0].set_title(f'Batch Norm Parameters\n{param_name}')
        axes[0].set_xlabel('Channel Index')
        axes[0].set_ylabel('Parameter Value')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution histogram
        axes[1].hist(weights, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title('Parameter Distribution')
        axes[1].set_xlabel('Parameter Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_embedding_matrix(self, weights, param_name):
        """Visualize embedding matrices."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Embedding matrix heatmap
        im1 = axes[0].imshow(weights, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Embedding Matrix\n{param_name}')
        axes[0].set_xlabel('Embedding Dimension')
        axes[0].set_ylabel('Vocabulary Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Mean embeddings per dimension
        mean_per_dim = np.mean(weights, axis=0)
        axes[1].plot(mean_per_dim, 'o-', markersize=3)
        axes[1].set_title('Mean Embedding per Dimension')
        axes[1].set_xlabel('Embedding Dimension')
        axes[1].set_ylabel('Mean Value')
        axes[1].grid(True, alpha=0.3)
        
        # Distribution histogram
        axes[2].hist(weights.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[2].set_title('Embedding Weight Distribution')
        axes[2].set_xlabel('Weight Value')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_1d_tensor(self, weights, param_name):
        """Visualize 1D tensors like biases."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parameter values plot
        axes[0].plot(weights, 'o-', markersize=3, linewidth=1)
        axes[0].set_title(f'1D Parameter Values\n{param_name}')
        axes[0].set_xlabel('Parameter Index')
        axes[0].set_ylabel('Parameter Value')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution histogram
        axes[1].hist(weights, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_title('Parameter Distribution')
        axes[1].set_xlabel('Parameter Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_weight_matrix(self, weights, param_name):
        """Visualize general weight matrices."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Weight matrix heatmap
        im1 = axes[0].imshow(weights, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Weight Matrix\n{param_name}')
        axes[0].set_xlabel('Input Dimension')
        axes[0].set_ylabel('Output Dimension')
        plt.colorbar(im1, ax=axes[0])
        
        # Weight distribution histogram
        axes[1].hist(weights.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1].set_title('Weight Distribution')
        axes[1].set_xlabel('Weight Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f'Shape: {weights.shape}\n'
        stats_text += f'Min: {weights.min():.4f}\n'
        stats_text += f'Max: {weights.max():.4f}\n'
        stats_text += f'Mean: {weights.mean():.4f}\n'
        stats_text += f'Std: {weights.std():.4f}'
        
        axes[2].text(0.1, 0.5, stats_text, transform=axes[2].transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[2].set_title('Statistics')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_scalar_value(self, value, param_name):
        """Visualize scalar values."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Display the scalar value prominently
        ax.text(0.5, 0.5, f'{param_name}\n\nValue: {value}', 
                transform=ax.transAxes, fontsize=16, 
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        ax.set_title(f'Scalar Parameter: {param_name}', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def extract_and_visualize_layers(self, checkpoint):
        """Extract and visualize all layers from the checkpoint."""
        if 'model_state_dict' not in checkpoint:
            print("No model_state_dict found in checkpoint")
            return
        
        state_dict = checkpoint['model_state_dict']
        # Initialize layer info dictionary
        layer_info = defaultdict(list)
        
        print(f"Extracting and visualizing {len(state_dict)} parameters...")
        
        # Create overview statistics first
        self.create_model_overview(state_dict)
        
        for name, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                continue
            
            print(f"  Processing: {name} {tuple(param.shape)}")
            
            # Determine layer category
            category = self.categorize_layer(name)
            
            # Skip very large tensors to avoid memory issues
            if param.numel() > 50_000_000:  # Skip tensors larger than 50M elements
                print(f"    ⚠️  Skipping large tensor: {param.numel():,} elements")
                continue
            
            try:
                # Create visualization
                if len(param.shape) == 0:  # Scalar values
                    fig = self.visualize_scalar_value(param.item(), name)
                elif len(param.shape) >= 2:
                    # Weight matrices and higher-dimensional tensors
                    fig = self.visualize_weights(param, name, category)
                else:
                    # 1D tensors (biases, etc.)
                    fig = self.visualize_1d_tensor(param.detach().cpu().numpy(), name)
                
                # Save the visualization
                safe_name = name.replace('.', '_').replace('/', '_')
                output_path = self.output_dir / f"{category}_layers" / f"{safe_name}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                layer_info[category].append({
                    'name': name,
                    'shape': list(param.shape) if len(param.shape) > 0 else 'scalar',
                    'file': f"{category}_layers/{safe_name}.png"
                })
                
            except Exception as e:
                print(f"    ⚠️  Failed to visualize {name}: {e}")
                continue
            
        print(f"\nVisualization complete! Results saved to: {self.output_dir}")
    
    def categorize_layer(self, param_name):
        """
        Categorize a parameter by its name into visualization categories.
        
        Args:
            param_name (str): Name of the parameter
            
        Returns:
            str: Category name for organization
        """
        param_lower = param_name.lower()
        
        # CNN/Convolutional layers (PRIORITY - these were missing!)
        if 'conv' in param_lower and 'weight' in param_lower:
            return 'conv_layers'
        
        # Attention layers (PRIORITY - these were missing!)
        if 'attention' in param_lower:
            return 'attention_layers'
        
        # Batch normalization layers (these were missing!)
        if any(bn_term in param_lower for bn_term in ['batch_norm', 'bn.']):
            return 'batch_norm_layers'
        
        # Embedding layers (this was missing!)
        if 'embedding' in param_lower:
            return 'embedding_layers'
        
        # Encoder layers (already working)
        if 'encoder' in param_lower:
            if 'rnn' in param_lower or 'gru' in param_lower or 'lstm' in param_lower:
                return 'encoder_layers'
            else:
                return 'encoder_other'
        
        # Decoder layers (already working)
        if 'decoder' in param_lower:
            if 'rnn' in param_lower or 'gru' in param_lower or 'lstm' in param_lower:
                return 'decoder_layers'
            else:
                return 'decoder_other'
        
        # Linear/Projection layers
        if any(term in param_lower for term in ['projection', 'linear', 'fc']):
            return 'linear_layers'
        
        return 'other_layers'
    
    def create_model_overview(self, state_dict):
        """Create an overview visualization of the entire model."""
        # Count parameters by category
        category_counts = {}
        category_params = {}
        
        for name, param in state_dict.items():
            category = self.categorize_layer(name)
            if category not in category_counts:
                category_counts[category] = 0
                category_params[category] = 0
            category_counts[category] += 1
            category_params[category] += param.numel()
        
        # Create overview plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Layer count by category
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        ax1.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Layers by Category')
        
        # Parameter count by category
        param_counts = list(category_params.values())
        ax2.pie(param_counts, labels=categories, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Parameters by Category')
        
        # Bar chart of layer counts
        ax3.bar(categories, counts, color=colors)
        ax3.set_title('Number of Layers per Category')
        ax3.set_ylabel('Count')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Bar chart of parameter counts
        ax4.bar(categories, param_counts, color=colors)
        ax4.set_title('Parameters per Category')
        ax4.set_ylabel('Parameter Count')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Add total statistics
        total_params = sum(category_params.values())
        total_layers = sum(category_counts.values())
        
        fig.suptitle(f'Model Overview\nTotal Layers: {total_layers:,} | Total Parameters: {total_params:,}', 
                     fontsize=16)
        
        plt.tight_layout()
        
        # Save overview
        overview_path = self.output_dir / 'overview' / 'model_overview.png'
        overview_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Model overview saved to: {overview_path}")
    
    def run(self):
        """Run the complete extraction and visualization process."""
        print("="*80)
        print("PYTORCH CHECKPOINT LAYER EXTRACTOR")
        print("="*80)
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return
        
        # Analyze structure
        checkpoint = self.analyze_checkpoint_structure(checkpoint)
        
        # Extract and visualize layers
        self.extract_and_visualize_layers(checkpoint)
        
        print(f"\n{'='*80}")
        print("EXTRACTION COMPLETE!")
        print(f"All visualizations saved to: {self.output_dir}")
        print(f"{'='*80}")

def main():
    """Main function to run the layer extractor."""
    parser = argparse.ArgumentParser(description="Extract and visualize layers from PyTorch checkpoints")
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument("--output-dir", default="checkpoint_visualizations", 
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create and run extractor
    extractor = CheckpointLayerExtractor(args.checkpoint, args.output_dir)
    extractor.run()

if __name__ == "__main__":
    main() 