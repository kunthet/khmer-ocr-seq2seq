#!/usr/bin/env python3
"""
Quick test script for Khmer OCR model.
Tests the trained model on a few synthetic samples.
"""

import torch
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.inference.ocr_engine import KhmerOCREngine
from src.utils.config import ConfigManager
from src.data.synthetic_dataset import SyntheticImageDataset


def main():
    """Quick test of the trained model."""
    print("Khmer OCR Quick Test")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model checkpoint path
    checkpoint_path = "models/checkpoints/best_model.pth"
    
    try:
        # Load model
        print(f"Loading model from: {checkpoint_path}")
        config = ConfigManager()
        
        # Always try to load checkpoint first
        if Path(checkpoint_path).exists():
            try:
                engine = KhmerOCREngine.from_checkpoint(
                    checkpoint_path,
                    config_manager=config,
                    device=device
                )
                print("✓ Trained model loaded successfully")
                model_type = "trained"
            except Exception as e:
                print(f"⚠ Checkpoint loading failed: {e}")
                print("Creating untrained model for testing")
                from src.models.seq2seq import KhmerOCRSeq2Seq
                model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
                engine = KhmerOCREngine(
                    model=model,
                    vocab=config.vocab,
                    device=device
                )
                model_type = "untrained"
        else:
            print("⚠ Checkpoint not found, creating untrained model for testing")
            from src.models.seq2seq import KhmerOCRSeq2Seq
            model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
            engine = KhmerOCREngine(
                model=model,
                vocab=config.vocab,
                device=device
            )
            model_type = "untrained"
        
        # Load test dataset
        print("\nLoading test dataset...")
        dataset = SyntheticImageDataset(
            split="test",
            synthetic_dir="data/synthetic",
            config_manager=config,
            max_samples=10
        )
        print(f"Loaded {len(dataset)} test samples")
        
        # Test a few samples
        print("\nTesting samples:")
        print("-" * 50)
        
        total_time = 0
        correct = 0
        
        for i in range(min(5, len(dataset))):
            image, target_tensor, target_text = dataset[i]
            
            # Convert tensor to PIL Image if needed
            if isinstance(image, torch.Tensor):
                import numpy as np
                from PIL import Image
                image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                image_pil = Image.fromarray(image_np)
            else:
                image_pil = image
            
            # Time the inference
            start_time = time.time()
            
            result = engine.recognize(
                image_pil,
                method='greedy',
                return_confidence=True,
                preprocess=True
            )
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Check accuracy
            exact_match = result['text'].strip() == target_text.strip()
            if exact_match:
                correct += 1
            
            print(f"Sample {i+1}:")
            print(f"  Target:     '{target_text}'")
            print(f"  Prediction: '{result['text']}'")
            print(f"  Confidence: {result['confidence']:.3f}" if result['confidence'] is not None else "  Confidence: N/A")
            print(f"  Time:       {inference_time:.3f}s")
            print(f"  Correct:    {exact_match}")
            print()
        
        # Summary
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Model type:          {model_type}")
        print(f"Samples tested:      {min(5, len(dataset))}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy:           {correct/min(5, len(dataset)):.1%}")
        print(f"Average time:       {total_time/min(5, len(dataset)):.3f}s")
        print(f"Total time:         {total_time:.3f}s")
        
        if model_type == "untrained":
            if correct == 0:
                print("\n⚠ No correct predictions - this is expected for an untrained model")
                print("  The model generates mostly special tokens (SOS, EOS, PAD) which are filtered out")
                print("  resulting in empty predictions. This is normal behavior for untrained models.")
            else:
                print(f"\n⚠ Unexpected accuracy for untrained model: {correct}/{min(5, len(dataset))}")
        else:  # trained model
            if correct == 0:
                print("\n⚠ No correct predictions from trained model")
                print("  This may indicate:")
                print("  - The model needs more training")
                print("  - The model is overfitting to certain patterns")
                print("  - There's a mismatch between training and test data")
            elif correct == min(5, len(dataset)):
                print("\n✓ Perfect accuracy!")
            else:
                print(f"\n✓ Model is working - {correct}/{min(5, len(dataset))} correct")
                
        print(f"\nTo improve model performance:")
        print("  1. Train the model longer if using untrained model")
        print("  2. Check training data quality and diversity")
        print("  3. Tune hyperparameters (learning rate, batch size)")
        print("  4. Ensure proper text preprocessing in training")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 