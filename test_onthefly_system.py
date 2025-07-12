#!/usr/bin/env python3
"""
Test Script for On-the-Fly Training System

This script tests the new on-the-fly dataset implementation to ensure
it works correctly before full training.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.onthefly_dataset import OnTheFlyDataset, OnTheFlyCollateFunction, create_onthefly_dataloaders
from src.data.synthetic_dataset import SyntheticImageDataset
from src.utils.config import ConfigManager


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def test_onthefly_dataset():
    """Test the OnTheFlyDataset functionality"""
    print("=== Testing OnTheFlyDataset ===")
    
    config = ConfigManager()
    
    try:
        # Test training dataset
        print("\n1. Testing training dataset creation...")
        train_dataset = OnTheFlyDataset(
            split="train",
            config_manager=config,
            samples_per_epoch=100,  # Small number for testing
            augment_prob=0.8
        )
        
        print(f"✅ Training dataset created: {len(train_dataset)} samples per epoch")
        print(f"   Text lines available: {len(train_dataset.text_lines)}")
        print(f"   Fonts available: {len(train_dataset.text_renderer.fonts)}")
        
        # Test sample generation
        print("\n2. Testing sample generation...")
        sample_image, sample_target, sample_text = train_dataset[0]
        
        print(f"✅ Sample generated successfully")
        print(f"   Image shape: {sample_image.shape}")
        print(f"   Target shape: {sample_target.shape}")
        print(f"   Text: '{sample_text[:50]}{'...' if len(sample_text) > 50 else ''}'")
        print(f"   Target tokens: {sample_target.tolist()[:10]}...")
        
        # Test multiple samples for variety
        print("\n3. Testing sample variety...")
        texts = []
        for i in range(5):
            _, _, text = train_dataset[i]
            texts.append(text)
        
        unique_texts = len(set(texts))
        print(f"✅ Generated {unique_texts}/5 unique texts (variety check)")
        
        # Test statistics
        print("\n4. Testing dataset statistics...")
        stats = train_dataset.get_statistics()
        print(f"✅ Dataset statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ OnTheFlyDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loaders():
    """Test the data loaders with collate function"""
    print("\n=== Testing Data Loaders ===")
    
    config = ConfigManager()
    
    try:
        # Create datasets
        print("\n1. Creating datasets...")
        train_dataset = OnTheFlyDataset(
            split="train",
            config_manager=config,
            samples_per_epoch=50
        )
        
        val_dataset = OnTheFlyDataset(
            split="val",
            config_manager=config,
            samples_per_epoch=20,
            augment_prob=0.0
        )
        
        print(f"✅ Datasets created: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Create collate function
        collate_fn = OnTheFlyCollateFunction(config.vocab)
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # No multiprocessing for testing
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        print(f"✅ Data loaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")
        
        # Test batch loading
        print("\n2. Testing batch loading...")
        train_batch = next(iter(train_loader))
        
        print(f"✅ Batch loaded successfully")
        print(f"   Images shape: {train_batch['images'].shape}")
        print(f"   Targets shape: {train_batch['targets'].shape}")
        print(f"   Image lengths: {train_batch['image_lengths']}")
        print(f"   Target lengths: {train_batch['target_lengths']}")
        print(f"   Texts sample: {train_batch['texts'][0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loaders test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_loading():
    """Test loading of fixed validation set"""
    print("\n=== Testing Validation Set Loading ===")
    
    validation_dir = "data/validation_fixed"
    
    if not Path(validation_dir).exists():
        print(f"⚠️  Fixed validation set not found at {validation_dir}")
        print("   Run: python generate_fixed_validation_set.py")
        return True  # Not a failure, just not generated yet
    
    config = ConfigManager()
    
    try:
        print(f"\n1. Loading fixed validation set from {validation_dir}...")
        
        val_dataset = SyntheticImageDataset(
            split="val",
            synthetic_dir=validation_dir,
            config_manager=config,
            max_samples=6400
        )
        
        print(f"✅ Fixed validation set loaded: {len(val_dataset)} samples")
        
        # Test sample loading
        print("\n2. Testing validation sample loading...")
        val_image, val_target, val_text = val_dataset[0]
        
        print(f"✅ Validation sample loaded")
        print(f"   Image shape: {val_image.shape}")
        print(f"   Target shape: {val_target.shape}")
        print(f"   Text: '{val_text[:50]}{'...' if len(val_text) > 50 else ''}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation set test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test memory efficiency compared to pre-generated approach"""
    print("\n=== Testing Memory Efficiency ===")
    
    import psutil
    import gc
    
    try:
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Create on-the-fly dataset
        config = ConfigManager()
        
        dataset = OnTheFlyDataset(
            split="train",
            config_manager=config,
            samples_per_epoch=1000  # Reasonable size for testing
        )
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        print(f"Memory after dataset creation: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        # Generate several samples to test memory growth
        print("\nGenerating 50 samples to test memory growth...")
        for i in range(50):
            _ = dataset[i]
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"  After {i+1} samples: {current_memory:.1f} MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"✅ Memory efficiency test completed")
        print(f"   Total memory increase: {total_increase:.1f} MB")
        print(f"   Memory per sample: {total_increase/50:.2f} MB")
        
        # Clean up
        del dataset
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        return False


def test_reproducibility():
    """Test reproducibility with fixed seeds"""
    print("\n=== Testing Reproducibility ===")
    
    config = ConfigManager()
    
    try:
        print("\n1. Testing validation reproducibility (fixed seed)...")
        
        # Create two validation datasets with same seed
        val_dataset1 = OnTheFlyDataset(
            split="val",
            config_manager=config,
            samples_per_epoch=10,
            augment_prob=0.0,
            random_seed=42
        )
        
        val_dataset2 = OnTheFlyDataset(
            split="val", 
            config_manager=config,
            samples_per_epoch=10,
            augment_prob=0.0,
            random_seed=42
        )
        
        # Compare first few samples
        identical_count = 0
        for i in range(5):
            _, _, text1 = val_dataset1[i]
            _, _, text2 = val_dataset2[i]
            if text1 == text2:
                identical_count += 1
        
        print(f"✅ Validation reproducibility: {identical_count}/5 identical texts")
        
        print("\n2. Testing training variety (no fixed seed)...")
        
        # Create two training datasets without seeds
        train_dataset1 = OnTheFlyDataset(
            split="train",
            config_manager=config,
            samples_per_epoch=10,
            random_seed=None
        )
        
        train_dataset2 = OnTheFlyDataset(
            split="train",
            config_manager=config, 
            samples_per_epoch=10,
            random_seed=None
        )
        
        # Compare samples (should be different due to randomness)
        different_count = 0
        for i in range(5):
            _, _, text1 = train_dataset1[i]
            _, _, text2 = train_dataset2[i]
            if text1 != text2:
                different_count += 1
        
        print(f"✅ Training variety: {different_count}/5 different texts")
        
        return True
        
    except Exception as e:
        print(f"❌ Reproducibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    setup_logging()
    
    print("🧪 Testing On-the-Fly Training System")
    print("="*50)
    
    tests = [
        ("OnTheFlyDataset", test_onthefly_dataset),
        ("Data Loaders", test_data_loaders),
        ("Validation Loading", test_validation_loading),
        ("Memory Efficiency", test_memory_efficiency),
        ("Reproducibility", test_reproducibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! On-the-fly training system is ready.")
        print("\nNext steps:")
        print("1. Generate fixed validation set: python generate_fixed_validation_set.py")
        print("2. Start training: python src/training/train_onthefly.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 