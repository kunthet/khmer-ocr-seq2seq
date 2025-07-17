#!/usr/bin/env python3
"""
Debug script to investigate the inference issue with empty predictions.
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


def debug_model_generation():
    """Debug the model generation process."""
    print("Debugging Khmer OCR Model Generation")
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
        
        # Debug vocabulary
        print(f"\nVocabulary Info:")
        print(f"  Total vocab size: {len(config.vocab)}")
        print(f"  SOS_IDX: {config.vocab.SOS_IDX}")
        print(f"  EOS_IDX: {config.vocab.EOS_IDX}")
        print(f"  PAD_IDX: {config.vocab.PAD_IDX}")
        print(f"  UNK_IDX: {config.vocab.UNK_IDX}")
        print(f"  First 10 tokens: {config.vocab.vocab[:10]}")
        print(f"  Last 10 tokens: {config.vocab.vocab[-10:]}")
        
        # Create model (always use untrained for debugging)
        print("\nCreating untrained model for debugging...")
        from src.models.seq2seq import KhmerOCRSeq2Seq
        model = KhmerOCRSeq2Seq(config_or_vocab_size=config)
        engine = KhmerOCREngine(
            model=model,
            vocab=config.vocab,
            device=device
        )
        
        # Load test dataset
        print("\nLoading test dataset...")
        dataset = SyntheticImageDataset(
            split="test",
            synthetic_dir="data/synthetic",
            config_manager=config,
            max_samples=3
        )
        print(f"Loaded {len(dataset)} test samples")
        
        # Debug a single sample
        print("\nDebugging single sample:")
        print("-" * 50)
        
        image, target_tensor, target_text = dataset[0]
        print(f"Target text: '{target_text}'")
        print(f"Target tensor shape: {target_tensor.shape}")
        print(f"Target tensor: {target_tensor.tolist()}")
        
        # Decode target tensor to verify vocabulary
        target_decoded = config.vocab.decode(target_tensor.tolist())
        print(f"Target decoded: '{target_decoded}'")
        
        # Convert tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            import numpy as np
            from PIL import Image
            image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
        else:
            image_pil = image
        
        print(f"Image shape: {image_pil.size}")
        
        # Debug the inference process step by step
        print("\nDebugging inference process:")
        
        # Step 1: Preprocess image
        print("Step 1: Preprocessing image...")
        preprocessed = engine.preprocess_image(image_pil)
        print(f"  Preprocessed shape: {preprocessed.shape}")
        print(f"  Preprocessed min/max: {preprocessed.min():.3f}/{preprocessed.max():.3f}")
        
        # Step 2: Generate raw output
        print("Step 2: Generating raw output...")
        with torch.no_grad():
            result = engine.model.generate(
                images=preprocessed,
                max_length=50,
                method='greedy'
            )
        
        print(f"  Raw generation result keys: {result.keys()}")
        sequences = result['sequences']
        print(f"  Generated sequences shape: {sequences.shape}")
        print(f"  Generated sequences: {sequences}")
        
        # Step 3: Decode sequences
        print("Step 3: Decoding sequences...")
        for i, seq in enumerate(sequences):
            seq_list = seq.cpu().tolist()
            print(f"  Sequence {i}: {seq_list}")
            
            # Show raw decode
            raw_decoded = config.vocab.decode(seq_list)
            print(f"  Raw decoded: '{raw_decoded}'")
            
            # Show filtered decode
            filtered_decoded = engine._decode_sequence(seq_list)
            print(f"  Filtered decoded: '{filtered_decoded}'")
            
            # Manual filtering debug
            cleaned_sequence = []
            for token_id in seq_list:
                print(f"    Token {token_id}: '{config.vocab.idx_to_char.get(token_id, 'UNKNOWN')}'")
                if token_id == config.vocab.EOS_IDX:
                    print(f"    -> EOS found, breaking")
                    break
                if token_id not in [config.vocab.SOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]:
                    cleaned_sequence.append(token_id)
                    print(f"    -> Kept token {token_id}")
                else:
                    print(f"    -> Filtered out special token {token_id}")
            
            print(f"  Cleaned sequence: {cleaned_sequence}")
            manual_decoded = config.vocab.decode(cleaned_sequence)
            print(f"  Manual decoded: '{manual_decoded}'")
        
        # Step 4: Test with trained model if available
        if Path(checkpoint_path).exists():
            print("\nStep 4: Testing with trained model (if available)...")
            try:
                trained_engine = KhmerOCREngine.from_checkpoint(
                    checkpoint_path,
                    config_manager=config,
                    device=device
                )
                
                trained_result = trained_engine.recognize(
                    image_pil,
                    method='greedy',
                    return_confidence=True,
                    preprocess=True
                )
                
                print(f"  Trained model prediction: '{trained_result['text']}'")
                print(f"  Trained model confidence: {trained_result['confidence']}")
                
            except Exception as e:
                print(f"  Failed to load trained model: {e}")
        
        # Step 5: Test vocabulary encoding/decoding
        print("\nStep 5: Testing vocabulary encoding/decoding...")
        test_text = "កម្ពុជា"
        encoded = config.vocab.encode(test_text)
        decoded = config.vocab.decode(encoded)
        print(f"  Test text: '{test_text}'")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Round-trip successful: {test_text == decoded}")
        
        print("\n" + "=" * 50)
        print("DEBUG ANALYSIS:")
        print("=" * 50)
        
        # Check if model is generating only special tokens
        all_tokens = [token for seq in sequences for token in seq.cpu().tolist()]
        special_tokens = [config.vocab.SOS_IDX, config.vocab.EOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]
        special_count = sum(1 for token in all_tokens if token in special_tokens)
        total_count = len(all_tokens)
        
        print(f"Total tokens generated: {total_count}")
        print(f"Special tokens: {special_count}")
        print(f"Content tokens: {total_count - special_count}")
        print(f"Special token ratio: {special_count/total_count:.2%}")
        
        if special_count / total_count > 0.8:
            print("❌ ISSUE: Model is generating mostly special tokens")
            print("   This suggests the model is not properly trained")
        else:
            print("✅ Model is generating content tokens")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    debug_model_generation() 