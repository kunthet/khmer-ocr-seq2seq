#!/usr/bin/env python3
"""
Test script for the synthetic dataset implementation
"""

import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.synthetic_dataset import SyntheticImageDataset, create_synthetic_dataloaders
from src.utils.config import ConfigManager


def test_synthetic_dataset():
    """Test the synthetic dataset functionality"""
    print("=== Testing Synthetic Dataset ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset creation
    config = ConfigManager()
    
    try:
        # Create datasets
        train_dataset = SyntheticImageDataset(
            split="train",
            synthetic_dir="data/synthetic",
            config_manager=config,
            max_samples=10
        )
        
        val_dataset = SyntheticImageDataset(
            split="val",
            synthetic_dir="data/synthetic",
            config_manager=config,
            max_samples=10
        )
        
        print(f"✅ Datasets created successfully")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        
        # Test data loading
        if len(train_dataset) > 0:
            print("\n=== Testing Data Loading ===")
            
            # Load first sample
            image_tensor, target_tensor, text = train_dataset[0]
            
            print(f"✅ Sample loaded successfully")
            print(f"   Image shape: {image_tensor.shape}")
            print(f"   Target shape: {target_tensor.shape}")
            print(f"   Text: '{text}'")
            print(f"   Target: {target_tensor.tolist()}")
            
            # Test sample info
            sample_info = train_dataset.get_sample_info(0)
            print(f"   Sample info: {sample_info}")
            
            # Test statistics
            stats = train_dataset.get_statistics()
            print(f"   Dataset stats: {stats}")
        
        # Test data loaders
        print("\n=== Testing Data Loaders ===")
        
        train_loader, val_loader, test_loader = create_synthetic_dataloaders(
            config_manager=config,
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            synthetic_dir="data/synthetic",
            max_samples=10
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test batch loading
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            
            print(f"✅ Batch loaded successfully")
            print(f"   Images shape: {batch['images'].shape}")
            print(f"   Targets shape: {batch['targets'].shape}")
            print(f"   Image lengths: {batch['image_lengths']}")
            print(f"   Target lengths: {batch['target_lengths']}")
            print(f"   Texts: {batch['texts']}")
        
        print("\n🎉 All tests passed! Synthetic dataset is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    success = test_synthetic_dataset()
    
    if success:
        print("\n✅ Synthetic dataset is ready for training!")
        print("\nNext steps:")
        print("1. Generate full dataset: python generate_synthetic_images.py")
        print("2. Start training: python src/training/train_production.py")
    else:
        print("\n❌ Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main() 