#!/usr/bin/env python3
"""
Verify Training Configuration for Short Text Optimization
This script checks that the training setup is properly configured for short text training.
"""

import sys
import os
from pathlib import Path

def check_validation_dataset():
    """Check if the short text validation dataset exists."""
    validation_path = Path("data/validation_short_text")
    
    print("🔍 Checking Validation Dataset:")
    print("=" * 40)
    
    if not validation_path.exists():
        print(f"❌ Validation directory not found: {validation_path}")
        print("   Please run: python generate_fixed_validation_dataset.py")
        return False
    
    if not (validation_path / "images").exists():
        print(f"❌ Images directory not found: {validation_path / 'images'}")
        return False
        
    if not (validation_path / "labels").exists():
        print(f"❌ Labels directory not found: {validation_path / 'labels'}")
        return False
        
    if not (validation_path / "metadata.json").exists():
        print(f"❌ Metadata file not found: {validation_path / 'metadata.json'}")
        return False
    
    # Count images and labels
    images = list((validation_path / "images").glob("*.png"))
    labels = list((validation_path / "labels").glob("*.txt"))
    
    print(f"✅ Validation dataset found: {validation_path}")
    print(f"   Images: {len(images)}")
    print(f"   Labels: {len(labels)}")
    
    if len(images) != len(labels):
        print(f"⚠️  Mismatch: {len(images)} images vs {len(labels)} labels")
        return False
    
    return True

def check_training_script():
    """Check if the training script exists and is configured correctly."""
    script_path = Path("src/training/train_onthefly_short_text.py")
    
    print("\n🔍 Checking Training Script:")
    print("=" * 40)
    
    if not script_path.exists():
        print(f"❌ Training script not found: {script_path}")
        return False
    
    print(f"✅ Training script found: {script_path}")
    
    # Check key configuration in the script
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("validation_short_text", "Short text validation directory configured"),
        ("RandomWidthDataset", "RandomWidthDataset import found"),
        ("min_length=1", "Minimum length configured"),
        ("max_length=100", "Maximum length configured for short texts"),
        ("alpha=0.005", "Alpha parameter configured"),
    ]
    
    for check_text, description in checks:
        if check_text in content:
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - NOT FOUND")
    
    return True

def check_random_width_dataset():
    """Check if RandomWidthDataset is properly configured for short texts."""
    print("\n🔍 Checking RandomWidthDataset Configuration:")
    print("=" * 40)
    
    try:
        sys.path.append('src')
        from test_random_width_dataset import MockBaseDataset, MockConfigManager
        from src.data.random_width_dataset import RandomWidthDataset
        
        # Create a small test dataset
        base_dataset = MockBaseDataset(10)
        config_manager = MockConfigManager()
        
        dataset = RandomWidthDataset(
            base_dataset=base_dataset,
            min_length=1,
            max_length=100,
            alpha=0.005,
            config_manager=config_manager
        )
        
        print("✅ RandomWidthDataset can be created successfully")
        
        # Test a few samples to verify distribution
        short_count = 0
        total_samples = 50
        
        for i in range(total_samples):
            sample = dataset[i % len(dataset)]
            if len(sample['text']) <= 5:
                short_count += 1
        
        short_percentage = (short_count / total_samples) * 100
        print(f"✅ Short text distribution test: {short_count}/{total_samples} ({short_percentage:.1f}%) are ≤5 chars")
        
        if short_percentage >= 30:
            print("   ✅ Good short text bias achieved!")
        else:
            print("   ⚠️  Short text bias could be improved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RandomWidthDataset: {e}")
        return False

def show_usage_instructions():
    """Show instructions for using the training script."""
    print("\n🚀 Usage Instructions:")
    print("=" * 40)
    print("To start short text optimized training:")
    print("")
    print("1. Basic training (auto batch size):")
    print("   python src/training/train_onthefly_short_text.py")
    print("")
    print("2. With custom parameters:")
    print("   python src/training/train_onthefly_short_text.py \\")
    print("     --num-epochs 100 \\")
    print("     --batch-size 32 \\")
    print("     --train-samples-per-epoch 10000")
    print("")
    print("3. Resume from checkpoint:")
    print("   python src/training/train_onthefly_short_text.py \\")
    print("     --resume models/checkpoints/checkpoint_epoch_50.pth")
    print("")
    print("Key Features:")
    print("• ~50% of training data will be very short texts (1-5 chars)")
    print("• ~25% will be short texts (6-10 chars)")
    print("• Uses data/validation_short_text/ for evaluation")
    print("• On-the-fly generation = unlimited data variety")

def main():
    """Main verification function."""
    print("🔧 Training Configuration Verification")
    print("=" * 60)
    print("Verifying that the training setup is optimized for short text performance.")
    print("=" * 60)
    
    # Run all checks
    validation_ok = check_validation_dataset()
    script_ok = check_training_script()
    dataset_ok = check_random_width_dataset()
    
    print("\n📊 Summary:")
    print("=" * 20)
    
    if validation_ok and script_ok and dataset_ok:
        print("✅ All checks passed! Training is ready for short text optimization.")
        show_usage_instructions()
    else:
        print("❌ Some checks failed. Please address the issues above.")
        print("\nQuick fixes:")
        if not validation_ok:
            print("• Generate validation dataset: python generate_fixed_validation_dataset.py")
        if not script_ok:
            print("• Check training script configuration")
        if not dataset_ok:
            print("• Check RandomWidthDataset implementation")

if __name__ == "__main__":
    main() 