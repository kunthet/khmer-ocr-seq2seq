#!/usr/bin/env python3
"""
Test Script for Corpus Integration

This script tests the corpus dataset integration with the existing training pipeline
to ensure everything works before starting full production training.
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.data.corpus_dataset import KhmerCorpusDataset, create_corpus_dataloaders
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.utils.config import ConfigManager


def test_dataset_creation():
    """Test basic dataset creation and data loading"""
    print("🔧 Testing dataset creation...")
    
    config = ConfigManager()
    
    # Create small dataset for testing
    dataset = KhmerCorpusDataset(
        split="train",
        config_manager=config,
        corpus_dir="data/processed",
        max_lines=100  # Limit for testing
    )
    
    print(f"   ✅ Dataset created with {len(dataset)} samples")
    
    # Test sample loading
    sample_image, sample_target, sample_text = dataset[0]
    print(f"   ✅ Sample loaded:")
    print(f"      Image shape: {sample_image.shape}")
    print(f"      Target shape: {sample_target.shape}")
    print(f"      Text: '{sample_text[:50]}{'...' if len(sample_text) > 50 else ''}'")
    print(f"      Target indices: {sample_target.tolist()[:10]}...")
    
    # Test character distribution
    char_dist = dataset.get_character_distribution()
    top_chars = list(char_dist.items())[:10]
    print(f"   ✅ Top 10 characters: {top_chars}")
    
    return True


def test_dataloader_creation():
    """Test data loader creation with collate function"""
    print("\n🔧 Testing data loader creation...")
    
    config = ConfigManager()
    
    # Create data loaders with small datasets
    train_loader, val_loader, test_loader = create_corpus_dataloaders(
        config_manager=config,
        batch_size=4,
        num_workers=0,  # Avoid multiprocessing issues in test
        corpus_dir="data/processed",
        max_lines=50  # Small for testing
    )
    
    print(f"   ✅ Data loaders created:")
    print(f"      Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"      Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"      Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    # Test batch loading
    batch = next(iter(train_loader))
    print(f"   ✅ Sample batch loaded:")
    print(f"      Images shape: {batch['images'].shape}")
    print(f"      Targets shape: {batch['targets'].shape}")
    print(f"      Image lengths: {batch['image_lengths']}")
    print(f"      Target lengths: {batch['target_lengths']}")
    print(f"      Sample texts: {batch['texts'][:2]}")
    
    return batch


def test_model_compatibility(batch):
    """Test model compatibility with corpus data"""
    print("\n🔧 Testing model compatibility...")
    
    config = ConfigManager()
    device = torch.device('cpu')  # Use CPU for testing
    
    # Create model
    model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
    model.to(device)
    model.eval()
    
    # Move batch to device
    images = batch['images'].to(device)
    targets = batch['targets'].to(device)
    
    print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    with torch.no_grad():
        output = model(images, targets)
    
    print(f"   ✅ Forward pass successful:")
    print(f"      Loss: {output['loss'].item():.4f}")
    print(f"      Logits shape: {output['logits'].shape}")
    
    # Test inference
    with torch.no_grad():
        inference_output = model.generate(images, max_length=50)  # Generate sequences
        predictions = inference_output['sequences'][0]  # First sample
        predicted_text = config.vocab.decode(predictions.tolist())
    
    print(f"   ✅ Inference successful:")
    print(f"      Original text: '{batch['texts'][0]}'")
    print(f"      Predicted text: '{predicted_text}'")
    print(f"      (Note: Untrained model output is expected to be poor)")
    
    return True


def test_font_handling():
    """Test font handling for text rendering"""
    print("\n🔧 Testing font handling...")
    
    # Check if fonts directory exists
    fonts_dir = Path("fonts")
    if fonts_dir.exists():
        font_files = list(fonts_dir.glob("*.ttf"))
        print(f"   ✅ Found {len(font_files)} font files:")
        for font in font_files[:5]:  # Show first 5
            print(f"      - {font.name}")
    else:
        print(f"   ⚠️  Fonts directory not found: {fonts_dir}")
        print(f"      Using system default fonts (may not render Khmer properly)")
        print(f"      Consider adding Khmer fonts to fonts/ directory")
    
    return True


def main():
    """Run all integration tests"""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise for testing
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 CORPUS INTEGRATION TESTING")
    print("="*50)
    
    try:
        # Test 1: Dataset creation
        test_dataset_creation()
        
        # Test 2: Data loader creation
        batch = test_dataloader_creation()
        
        # Test 3: Model compatibility
        test_model_compatibility(batch)
        
        # Test 4: Font handling
        test_font_handling()
        
        print("\n✅ ALL TESTS PASSED!")
        print("🎯 Ready for production training!")
        print("\nNext steps:")
        print("1. Add Khmer fonts to fonts/ directory for better text rendering")
        print("2. Run small-scale training test: python src/training/train_production.py --max-lines 1000 --num-epochs 5")
        print("3. Run full production training: python src/training/train_production.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        logging.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit(main()) 