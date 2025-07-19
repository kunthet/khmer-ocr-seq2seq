#!/usr/bin/env python3
"""
PyTorch Checkpoint Structure Inspector

This script thoroughly inspects the complete structure of checkpoint files
to identify all available layers and parameters.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

class CheckpointInspector:
    """Inspects the complete structure of PyTorch checkpoint files."""
    
    def __init__(self, checkpoint_path):
        """
        Initialize the inspector.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = None
        
    def load_checkpoint(self):
        """Load the checkpoint file."""
        try:
            print(f"Loading checkpoint: {self.checkpoint_path}")
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            print("✅ Checkpoint loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def inspect_top_level_structure(self):
        """Inspect the top-level structure of the checkpoint."""
        print("\n" + "="*80)
        print("TOP-LEVEL CHECKPOINT STRUCTURE")
        print("="*80)
        
        for key, value in self.checkpoint.items():
            if isinstance(value, dict):
                print(f"📁 {key}: dict with {len(value)} keys")
            elif isinstance(value, torch.Tensor):
                print(f"🎯 {key}: tensor {tuple(value.shape)}")
            elif isinstance(value, (int, float)):
                print(f"🔢 {key}: {type(value).__name__} = {value}")
            elif isinstance(value, str):
                print(f"📝 {key}: str = '{value}'")
            else:
                print(f"❓ {key}: {type(value)} = {value}")
    
    def inspect_model_state_dict(self):
        """Inspect the model state dictionary in detail."""
        if 'model_state_dict' not in self.checkpoint:
            print("❌ No 'model_state_dict' found in checkpoint")
            return
        
        state_dict = self.checkpoint['model_state_dict']
        print("\n" + "="*80)
        print("MODEL STATE DICT ANALYSIS")
        print("="*80)
        print(f"Total parameters: {len(state_dict)}")
        
        # Categorize parameters by their names
        categories = defaultdict(list)
        
        for param_name, param_tensor in state_dict.items():
            # Determine category based on parameter name
            if 'encoder' in param_name.lower():
                if 'conv' in param_name.lower() or 'cnn' in param_name.lower():
                    categories['Encoder CNN'].append((param_name, param_tensor.shape))
                elif 'rnn' in param_name.lower() or 'lstm' in param_name.lower() or 'gru' in param_name.lower():
                    categories['Encoder RNN'].append((param_name, param_tensor.shape))
                elif 'attention' in param_name.lower() or 'attn' in param_name.lower():
                    categories['Encoder Attention'].append((param_name, param_tensor.shape))
                else:
                    categories['Encoder Other'].append((param_name, param_tensor.shape))
            elif 'decoder' in param_name.lower():
                if 'attention' in param_name.lower() or 'attn' in param_name.lower():
                    categories['Decoder Attention'].append((param_name, param_tensor.shape))
                elif 'rnn' in param_name.lower() or 'lstm' in param_name.lower() or 'gru' in param_name.lower():
                    categories['Decoder RNN'].append((param_name, param_tensor.shape))
                else:
                    categories['Decoder Other'].append((param_name, param_tensor.shape))
            elif 'conv' in param_name.lower() or 'cnn' in param_name.lower():
                categories['CNN Layers'].append((param_name, param_tensor.shape))
            elif 'attention' in param_name.lower() or 'attn' in param_name.lower():
                categories['Attention Layers'].append((param_name, param_tensor.shape))
            elif 'embedding' in param_name.lower() or 'embed' in param_name.lower():
                categories['Embedding Layers'].append((param_name, param_tensor.shape))
            elif 'fc' in param_name.lower() or 'linear' in param_name.lower() or 'projection' in param_name.lower():
                categories['Linear/FC Layers'].append((param_name, param_tensor.shape))
            elif 'bn' in param_name.lower() or 'batch_norm' in param_name.lower() or 'batchnorm' in param_name.lower():
                categories['Batch Norm'].append((param_name, param_tensor.shape))
            else:
                categories['Uncategorized'].append((param_name, param_tensor.shape))
        
        # Print categorized results
        for category, params in categories.items():
            if params:
                print(f"\n🔍 {category} ({len(params)} parameters):")
                for param_name, shape in sorted(params):
                    print(f"  {param_name}: {tuple(shape)}")
    
    def inspect_detailed_layer_structure(self):
        """Provide detailed analysis of each layer type."""
        if 'model_state_dict' not in self.checkpoint:
            return
        
        state_dict = self.checkpoint['model_state_dict']
        print("\n" + "="*80)
        print("DETAILED LAYER STRUCTURE ANALYSIS")
        print("="*80)
        
        # Group by layer prefix
        layer_groups = defaultdict(list)
        
        for param_name, param_tensor in state_dict.items():
            # Extract the layer name (everything before the last dot)
            parts = param_name.split('.')
            if len(parts) > 1:
                layer_name = '.'.join(parts[:-1])
                param_type = parts[-1]
                layer_groups[layer_name].append((param_type, param_tensor.shape))
        
        # Sort and display layer groups
        for layer_name in sorted(layer_groups.keys()):
            params = layer_groups[layer_name]
            print(f"\n📦 Layer: {layer_name}")
            for param_type, shape in sorted(params):
                print(f"  └── {param_type}: {tuple(shape)}")
    
    def generate_extraction_recommendations(self):
        """Generate recommendations for what should be extracted."""
        if 'model_state_dict' not in self.checkpoint:
            return
        
        state_dict = self.checkpoint['model_state_dict']
        print("\n" + "="*80)
        print("EXTRACTION RECOMMENDATIONS")
        print("="*80)
        
        # Count different types of layers
        conv_layers = [name for name in state_dict.keys() if 'conv' in name.lower()]
        attention_layers = [name for name in state_dict.keys() if 'attention' in name.lower() or 'attn' in name.lower()]
        encoder_layers = [name for name in state_dict.keys() if 'encoder' in name.lower()]
        decoder_layers = [name for name in state_dict.keys() if 'decoder' in name.lower()]
        embedding_layers = [name for name in state_dict.keys() if 'embedding' in name.lower() or 'embed' in name.lower()]
        
        print(f"🔍 Found Layer Types:")
        print(f"  Convolutional layers: {len(conv_layers)}")
        print(f"  Attention layers: {len(attention_layers)}")
        print(f"  Encoder layers: {len(encoder_layers)}")
        print(f"  Decoder layers: {len(decoder_layers)}")
        print(f"  Embedding layers: {len(embedding_layers)}")
        
        if conv_layers:
            print(f"\n🖼️  Convolutional layers that should be visualized:")
            for layer in conv_layers[:10]:  # Show first 10
                print(f"  {layer}")
            if len(conv_layers) > 10:
                print(f"  ... and {len(conv_layers) - 10} more")
        
        if attention_layers:
            print(f"\n🎯 Attention layers that should be visualized:")
            for layer in attention_layers[:10]:  # Show first 10
                print(f"  {layer}")
            if len(attention_layers) > 10:
                print(f"  ... and {len(attention_layers) - 10} more")

def main():
    parser = argparse.ArgumentParser(description='Inspect PyTorch checkpoint structure')
    parser.add_argument('checkpoint_path', help='Path to the checkpoint file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed layer structure')
    
    args = parser.parse_args()
    
    inspector = CheckpointInspector(args.checkpoint_path)
    
    if not inspector.load_checkpoint():
        return
    
    inspector.inspect_top_level_structure()
    inspector.inspect_model_state_dict()
    
    if args.detailed:
        inspector.inspect_detailed_layer_structure()
    
    inspector.generate_extraction_recommendations()

if __name__ == "__main__":
    main() 