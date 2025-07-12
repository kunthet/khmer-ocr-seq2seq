#!/usr/bin/env python3
"""
Test script for the integrated Khmer OCR synthetic generator.

This script tests the KhmerOCRSyntheticGenerator with a small subset of corpus data
to verify that the integration works correctly before running full generation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.synthetic_data_generator.khmer_ocr_generator import KhmerOCRSyntheticGenerator
from src.utils.config import ConfigManager
from src.data.synthetic_dataset import SyntheticImageDataset


def test_basic_generation():
    """Test basic synthetic image generation"""
    print("🧪 Testing basic synthetic image generation...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "test_output"
        
        try:
            # Initialize generator
            generator = KhmerOCRSyntheticGenerator(
                corpus_dir="data/processed",
                output_dir=str(temp_output),
                config_manager=ConfigManager(),
                fonts_dir="fonts",
                augment_train=False,  # Disable augmentation for testing
                use_advanced_backgrounds=False,  # Disable advanced backgrounds for testing
                random_seed=42
            )
            
            # Generate a small test set
            print("  📝 Generating test images...")
            metadata = generator.generate_split("train", max_samples=5)
            
            # Verify output
            assert metadata["total_samples"] == 5
            assert temp_output.exists()
            assert (temp_output / "train" / "images").exists()
            assert (temp_output / "train" / "labels").exists()
            
            # Check generated files
            image_files = list((temp_output / "train" / "images").glob("*.png"))
            label_files = list((temp_output / "train" / "labels").glob("*.txt"))
            
            assert len(image_files) == 5
            assert len(label_files) == 5
            
            print("  ✅ Basic generation test passed!")
            
            # Test image properties
            from PIL import Image
            for image_file in image_files:
                img = Image.open(image_file)
                assert img.height == 32, f"Image height should be 32px, got {img.height}"
                assert img.width >= 64, f"Image width should be >= 64px, got {img.width}"
                print(f"  📏 Image {image_file.name}: {img.width}x{img.height}")
            
            print("  ✅ Image dimensions test passed!")
            
        except Exception as e:
            print(f"  ❌ Basic generation test failed: {e}")
            raise


def test_dataset_integration():
    """Test integration with SyntheticImageDataset"""
    print("🧪 Testing dataset integration...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "test_output"
        
        try:
            # Generate test data
            generator = KhmerOCRSyntheticGenerator(
                corpus_dir="data/processed",
                output_dir=str(temp_output),
                config_manager=ConfigManager(),
                fonts_dir="fonts",
                augment_train=False,
                use_advanced_backgrounds=False,
                random_seed=42
            )
            
            # Generate small datasets for all splits
            for split in ["train", "val", "test"]:
                generator.generate_split(split, max_samples=3)
            
            # Test dataset loading
            config = ConfigManager()
            
            for split in ["train", "val", "test"]:
                dataset = SyntheticImageDataset(
                    split=split,
                    synthetic_dir=str(temp_output),
                    config_manager=config,
                    max_samples=3
                )
                
                assert len(dataset) == 3, f"Dataset should have 3 samples, got {len(dataset)}"
                
                # Test getting a sample
                image_tensor, target_tensor, text = dataset[0]
                assert image_tensor.shape[0] == 1, f"Image should be grayscale, got {image_tensor.shape[0]} channels"
                assert image_tensor.shape[1] == 32, f"Image height should be 32px, got {image_tensor.shape[1]}"
                assert len(text) > 0, "Text should not be empty"
                
                print(f"  ✅ {split} dataset integration test passed!")
            
            print("  ✅ Dataset integration test passed!")
            
        except Exception as e:
            print(f"  ❌ Dataset integration test failed: {e}")
            raise


def test_advanced_features():
    """Test advanced features like backgrounds and augmentation"""
    print("🧪 Testing advanced features...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "test_output"
        
        try:
            # Test with advanced backgrounds
            generator = KhmerOCRSyntheticGenerator(
                corpus_dir="data/processed",
                output_dir=str(temp_output),
                config_manager=ConfigManager(),
                fonts_dir="fonts",
                augment_train=True,
                use_advanced_backgrounds=True,
                random_seed=42
            )
            
            # Generate with advanced features
            metadata = generator.generate_split("train", max_samples=3)
            
            assert metadata["total_samples"] == 3
            assert metadata["augmentation_applied"] == True
            assert metadata["advanced_backgrounds"] == True
            
            print("  ✅ Advanced features test passed!")
            
        except Exception as e:
            print(f"  ❌ Advanced features test failed: {e}")
            raise


def test_statistics():
    """Test statistics generation"""
    print("🧪 Testing statistics generation...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "test_output"
        
        try:
            # Generate test data
            generator = KhmerOCRSyntheticGenerator(
                corpus_dir="data/processed",
                output_dir=str(temp_output),
                config_manager=ConfigManager(),
                fonts_dir="fonts",
                augment_train=False,
                use_advanced_backgrounds=False,
                random_seed=42
            )
            
            # Generate small datasets
            generator.generate_split("train", max_samples=2)
            generator.generate_split("val", max_samples=2)
            
            # Get statistics
            stats = generator.get_statistics()
            
            assert stats["total_images"] == 4
            assert len(stats["splits"]) == 2
            assert stats["splits"]["train"]["images"] == 2
            assert stats["splits"]["val"]["images"] == 2
            
            print("  ✅ Statistics test passed!")
            
        except Exception as e:
            print(f"  ❌ Statistics test failed: {e}")
            raise


def main():
    """Run all tests"""
    print("🚀 Starting Khmer OCR Synthetic Generator Tests...")
    
    try:
        # Check prerequisites
        if not Path("data/processed").exists():
            print("❌ data/processed directory not found. Please run corpus processing first.")
            sys.exit(1)
        
        if not Path("fonts").exists():
            print("❌ fonts directory not found. Please add Khmer fonts to fonts/ directory.")
            sys.exit(1)
        
        # Run tests
        test_basic_generation()
        test_dataset_integration()
        test_advanced_features()
        test_statistics()
        
        print("\n🎉 All tests passed!")
        print("The integrated Khmer OCR synthetic generator is working correctly.")
        print("You can now run the full generation with:")
        print("  python generate_khmer_synthetic_images.py --max-samples 100")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 