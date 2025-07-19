#!/usr/bin/env python3
"""
Simple script to verify that the EOS handling fix works correctly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config import ConfigManager
from src.inference.ocr_engine import KhmerOCREngine
from src.data.synthetic_dataset import SyntheticImageDataset, SyntheticCollateFunction
from torch.utils.data import DataLoader

def test_eos_fix():
    print("🧪 Testing EOS handling fix...")
    
    # Load configuration
    config = ConfigManager("configs/config.yaml")
    
    print(f"Vocab special tokens:")
    print(f"  SOS_IDX: {config.vocab.SOS_IDX}")
    print(f"  EOS_IDX: {config.vocab.EOS_IDX}")
    print(f"  PAD_IDX: {config.vocab.PAD_IDX}")
    print(f"  UNK_IDX: {config.vocab.UNK_IDX}")
    
    # Check for checkpoint
    checkpoint_path = "./models/checkpoints/full2/checkpoint_epoch_039.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    engine = KhmerOCREngine.from_checkpoint(
        checkpoint_path,
        config_manager=config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Load test sample
    dataset = SyntheticImageDataset(
        split="val",
        synthetic_dir="data/validation_fixed",
        config_manager=config,
        max_samples=1
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=SyntheticCollateFunction(config.vocab)
    )
    
    batch = next(iter(dataloader))
    images = batch['images']
    targets = batch['targets']
    
    # Test batch inference
    with torch.no_grad():
        result = engine.model.generate(
            images=images,
            max_length=256,
            method='greedy'
        )
    
    predictions = result['sequences']
    pred_sequence = predictions[0].cpu().tolist()
    target_sequence = targets[0].cpu().tolist()
    
    print(f"\nRaw prediction tokens: {pred_sequence}")
    print(f"Raw target tokens: {target_sequence}")
    
    # Find EOS positions for all possible EOS token IDs
    all_eos_candidates = [config.vocab.EOS_IDX, 1, 114]  # Check multiple possibilities
    for eos_candidate in all_eos_candidates:
        eos_positions = [i for i, token in enumerate(pred_sequence) if token == eos_candidate]
        if eos_positions:
            print(f"Token {eos_candidate} positions: {eos_positions}")
    
    # OLD WAY (problematic) - filters out ALL special tokens
    old_prediction = config.vocab.decode([
        t for t in pred_sequence 
        if t not in [config.vocab.PAD_IDX, config.vocab.SOS_IDX, config.vocab.EOS_IDX]
    ])
    
    # NEW WAY (fixed) - stops at first EOS
    clean_tokens = []
    for token_id in pred_sequence:
        if token_id == config.vocab.EOS_IDX:
            break  # Stop at first EOS
        if token_id not in [config.vocab.SOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]:
            clean_tokens.append(token_id)
    
    new_prediction = config.vocab.decode(clean_tokens)
    
    # Target
    clean_target_tokens = []
    for token_id in target_sequence:
        if token_id == config.vocab.EOS_IDX:
            break
        if token_id not in [config.vocab.SOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]:
            clean_target_tokens.append(token_id)
    
    target_text = config.vocab.decode(clean_target_tokens)
    
    print(f"\nTarget text: '{target_text}'")
    print(f"OLD prediction: '{old_prediction}'")
    print(f"NEW prediction: '{new_prediction}'")
    print(f"Match (OLD): {old_prediction.strip() == target_text.strip()}")
    print(f"Match (NEW): {new_prediction.strip() == target_text.strip()}")
    
    if old_prediction != new_prediction:
        print(f"✅ FIX WORKING! Removed post-EOS content: '{old_prediction[len(new_prediction):]}'")
        print(f"Length difference: {len(old_prediction)} -> {len(new_prediction)} characters")
    else:
        print("🤔 Same result - investigating...")
        print(f"Clean tokens: {clean_tokens}")
        
        # Let's try with token 114 as EOS (from the debug output)
        alt_clean_tokens = []
        for token_id in pred_sequence:
            if token_id == 114:  # Try 114 as EOS
                break
            if token_id not in [config.vocab.SOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]:
                alt_clean_tokens.append(token_id)
        
        alt_prediction = config.vocab.decode(alt_clean_tokens)
        print(f"ALT prediction (stop at 114): '{alt_prediction}'")
        
        if alt_prediction != old_prediction:
            print(f"✅ ALT FIX WORKING! Token 114 might be EOS")

if __name__ == "__main__":
    test_eos_fix() 