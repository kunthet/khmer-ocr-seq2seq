#!/usr/bin/env python3
"""
Debug script to reproduce and analyze the batch vs individual inference issue.

This script compares:
1. Batch inference using model.generate() (as in validation analysis)
2. Individual inference using engine.recognize() 
3. Shows the difference in EOS token handling
"""

import torch
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config import ConfigManager
from src.inference.ocr_engine import KhmerOCREngine
from src.data.synthetic_dataset import SyntheticImageDataset, SyntheticCollateFunction
from torch.utils.data import DataLoader

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_test_samples(config_manager, num_samples=5):
    """Load a few test samples for comparison."""
    dataset = SyntheticImageDataset(
        split="val",
        synthetic_dir="data/validation_fixed",
        config_manager=config_manager,
        max_samples=num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=False,
        collate_fn=SyntheticCollateFunction(config_manager.vocab)
    )
    
    # Get single batch
    batch = next(iter(dataloader))
    return batch

def test_batch_inference(model, batch, config_manager):
    """Test batch inference (like in validation analysis)."""
    print("\n" + "="*60)
    print("BATCH INFERENCE TEST (like validation analysis)")
    print("="*60)
    
    images = batch['images']
    targets = batch['targets']
    texts = batch.get('texts', [])
    
    # Direct batch inference
    with torch.no_grad():
        result = model.generate(
            images=images,
            max_length=256,
            method='greedy'
        )
    
    predictions = result['sequences']
    
    print(f"Generated sequences shape: {predictions.shape}")
    print(f"Generated sequences raw:\n{predictions}")
    
    # Decode like in validation analysis (PROBLEMATIC WAY)
    print("\nBatch decoding results (validation analysis style):")
    for i in range(len(predictions)):
        pred_sequence = predictions[i].cpu().tolist()
        target_sequence = targets[i].cpu().tolist()
        
        print(f"\nSample {i}:")
        print(f"  Raw prediction tokens: {pred_sequence}")
        print(f"  Raw target tokens: {target_sequence}")
        
        # Find EOS positions
        eos_positions = [j for j, token in enumerate(pred_sequence) if token == config_manager.vocab.EOS_IDX]
        print(f"  EOS positions in prediction: {eos_positions}")
        
        # Validation analysis way - filter out ALL special tokens
        validation_prediction = config_manager.vocab.decode([
            t for t in pred_sequence 
            if t not in [config_manager.vocab.PAD_IDX, 
                        config_manager.vocab.SOS_IDX, 
                        config_manager.vocab.EOS_IDX]
        ])
        
        # Correct way - stop at first EOS
        correct_prediction = ""
        clean_tokens = []
        for token in pred_sequence:
            if token == config_manager.vocab.EOS_IDX:
                break
            if token not in [config_manager.vocab.SOS_IDX, config_manager.vocab.PAD_IDX]:
                clean_tokens.append(token)
        correct_prediction = config_manager.vocab.decode(clean_tokens)
        
        # Target text
        target_tokens = [t for t in target_sequence 
                        if t not in [config_manager.vocab.PAD_IDX, 
                                   config_manager.vocab.SOS_IDX, 
                                   config_manager.vocab.EOS_IDX]]
        target_text = config_manager.vocab.decode(target_tokens)
        
        print(f"  Target text: '{target_text}'")
        print(f"  Validation prediction: '{validation_prediction}'")
        print(f"  Correct prediction: '{correct_prediction}'")
        print(f"  Match (validation): {validation_prediction.strip() == target_text.strip()}")
        print(f"  Match (correct): {correct_prediction.strip() == target_text.strip()}")
        
        if validation_prediction != correct_prediction:
            print(f"  ⚠️  DIFFERENCE DETECTED! Validation method includes post-EOS tokens")
            print(f"  Extra content: '{validation_prediction[len(correct_prediction):]}'")

def test_individual_inference(engine, batch, config_manager):
    """Test individual inference (like in engine.recognize)."""
    print("\n" + "="*60)
    print("INDIVIDUAL INFERENCE TEST (like engine.recognize)")
    print("="*60)
    
    images = batch['images']
    targets = batch['targets']
    texts = batch.get('texts', [])
    
    for i in range(len(images)):
        print(f"\nSample {i}:")
        
        # Individual inference
        result = engine.recognize(
            images[i],
            method='greedy',
            return_confidence=True,
            preprocess=False
        )
        
        # Target text
        target_sequence = targets[i].cpu().tolist()
        target_tokens = [t for t in target_sequence 
                        if t not in [config_manager.vocab.PAD_IDX, 
                                   config_manager.vocab.SOS_IDX, 
                                   config_manager.vocab.EOS_IDX]]
        target_text = config_manager.vocab.decode(target_tokens)
        
        print(f"  Target text: '{target_text}'")
        print(f"  Individual prediction: '{result['text']}'")
        print(f"  Match: {result['text'].strip() == target_text.strip()}")

def analyze_generation_behavior(model, config_manager):
    """Analyze the raw generation behavior step by step."""
    print("\n" + "="*60)
    print("GENERATION BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Create a simple test case
    text = "តេស្ត"  # Simple Khmer text
    print(f"Test text: '{text}'")
    
    # Encode to see expected tokens
    tokens = config_manager.vocab.encode(text)
    expected_sequence = [config_manager.vocab.SOS_IDX] + tokens + [config_manager.vocab.EOS_IDX]
    print(f"Expected sequence: {expected_sequence}")
    
    # Create a synthetic image for this text
    from src.synthetic_data_generator.text_renderer import TextRenderer
    renderer = TextRenderer(config_manager)
    pil_image = renderer.render_text(text, font_size=32)
    
    # Convert to tensor
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(pil_image).unsqueeze(0)
    
    # Generate with model
    with torch.no_grad():
        result = model.generate(
            images=image_tensor,
            max_length=50,  # Short max length for analysis
            method='greedy'
        )
    
    sequence = result['sequences'][0].cpu().tolist()
    print(f"Generated sequence: {sequence}")
    
    # Find where EOS appears
    eos_positions = [i for i, token in enumerate(sequence) if token == config_manager.vocab.EOS_IDX]
    print(f"EOS positions: {eos_positions}")
    
    if eos_positions:
        first_eos = eos_positions[0]
        before_eos = sequence[:first_eos]
        at_eos = [sequence[first_eos]]
        after_eos = sequence[first_eos + 1:]
        
        print(f"Tokens before first EOS: {before_eos}")
        print(f"EOS token: {at_eos}")
        print(f"Tokens after first EOS: {after_eos}")
        
        if after_eos:
            print("⚠️  PROBLEM: Tokens generated after EOS!")
            after_eos_text = config_manager.vocab.decode([t for t in after_eos if t not in [
                config_manager.vocab.PAD_IDX, config_manager.vocab.SOS_IDX, config_manager.vocab.EOS_IDX
            ]])
            print(f"Post-EOS text: '{after_eos_text}'")

def main():
    setup_logging()
    
    print("🔍 Debug: Batch vs Individual Inference Issue")
    print("=" * 80)
    
    # Load configuration
    config = ConfigManager("configs/config.yaml")
    
    # Check for checkpoint
    checkpoint_path = "./models/checkpoints/full2/checkpoint_epoch_039.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please provide a valid checkpoint path.")
        return
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    engine = KhmerOCREngine.from_checkpoint(
        checkpoint_path,
        config_manager=config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Load test samples
    print("Loading test samples...")
    batch = load_test_samples(config, num_samples=3)
    
    # Test batch inference (problematic)
    test_batch_inference(engine.model, batch, config)
    
    # Test individual inference (working)
    test_individual_inference(engine, batch, config)
    
    # Analyze generation behavior
    analyze_generation_behavior(engine.model, config)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("- Batch inference includes tokens generated AFTER EOS")
    print("- Individual inference correctly stops at first EOS")
    print("- The issue is in the sequence decoding logic, not the generation itself")
    print("=" * 80)

if __name__ == "__main__":
    main() 