#!/usr/bin/env python3
"""
Comprehensive verification of the EOS handling fix.
Tests multiple samples comparing batch vs individual inference.
"""

import torch
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config import ConfigManager
from src.inference.ocr_engine import KhmerOCREngine
from src.data.synthetic_dataset import SyntheticImageDataset, SyntheticCollateFunction
from torch.utils.data import DataLoader

def decode_sequence_old_way(sequence, vocab):
    """Old problematic way - filters out ALL special tokens."""
    return vocab.decode([
        t for t in sequence 
        if t not in [vocab.PAD_IDX, vocab.SOS_IDX, vocab.EOS_IDX]
    ])

def decode_sequence_new_way(sequence, vocab):
    """New fixed way - stops at first EOS."""
    clean_tokens = []
    for token_id in sequence:
        if token_id == vocab.EOS_IDX:
            break  # Stop at first EOS
        if token_id not in [vocab.SOS_IDX, vocab.PAD_IDX, vocab.UNK_IDX]:
            clean_tokens.append(token_id)
    return vocab.decode(clean_tokens)

def test_comprehensive_fix(num_samples=10):
    print("🔬 Comprehensive EOS Fix Verification")
    print("=" * 60)
    
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
    print(f"\nLoading model from: {checkpoint_path}")
    engine = KhmerOCREngine.from_checkpoint(
        checkpoint_path,
        config_manager=config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Load test samples
    dataset = SyntheticImageDataset(
        split="val",
        synthetic_dir="data/validation_fixed",
        config_manager=config,
        max_samples=num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=False,
        collate_fn=SyntheticCollateFunction(config.vocab)
    )
    
    batch = next(iter(dataloader))
    images = batch['images']
    targets = batch['targets']
    texts = batch.get('texts', [])
    
    print(f"\nTesting {num_samples} samples...")
    print("-" * 60)
    
    # Batch inference
    start_time = time.time()
    with torch.no_grad():
        result = engine.model.generate(
            images=images,
            max_length=256,
            method='greedy'
        )
    batch_time = time.time() - start_time
    
    predictions = result['sequences']
    
    # Statistics
    fixes_applied = 0
    old_vs_new_diffs = []
    individual_matches = 0
    total_samples = len(predictions)
    
    for i in range(total_samples):
        pred_sequence = predictions[i].cpu().tolist()
        target_sequence = targets[i].cpu().tolist()
        
        # Decode target
        target_text = decode_sequence_new_way(target_sequence, config.vocab)
        
        # Old way (problematic)
        old_prediction = decode_sequence_old_way(pred_sequence, config.vocab)
        
        # New way (fixed)
        new_prediction = decode_sequence_new_way(pred_sequence, config.vocab)
        
        # Individual inference for comparison
        start_individual = time.time()
        individual_result = engine.recognize(
            images[i],
            method='greedy',
            return_confidence=False,
            preprocess=False  # Already preprocessed
        )
        individual_time = time.time() - start_individual
        individual_prediction = individual_result['text']
        
        # Check if individual matches our fix
        if individual_prediction.strip() == new_prediction.strip():
            individual_matches += 1
        
        # Track differences
        if old_prediction != new_prediction:
            fixes_applied += 1
            diff_length = len(old_prediction) - len(new_prediction)
            old_vs_new_diffs.append(diff_length)
            
            print(f"\nSample {i}: FIX APPLIED")
            print(f"  Target:        '{target_text}'")
            print(f"  OLD batch:     '{old_prediction}'")
            print(f"  NEW batch:     '{new_prediction}'")
            print(f"  Individual:    '{individual_prediction}'")
            print(f"  Removed chars: {diff_length}")
            print(f"  Individual matches NEW: {individual_prediction.strip() == new_prediction.strip()}")
        else:
            # Check if this sample actually has the issue
            eos_positions = [j for j, token in enumerate(pred_sequence) if token == config.vocab.EOS_IDX]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0]
                tokens_after_eos = len(pred_sequence) - first_eos - 1
                if tokens_after_eos > 0:
                    print(f"\nSample {i}: No difference but has post-EOS tokens ({tokens_after_eos})")
                    print(f"  Sequence length: {len(pred_sequence)}")
                    print(f"  First EOS at: {first_eos}")
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE RESULTS:")
    print("=" * 60)
    
    print(f"Total samples tested: {total_samples}")
    print(f"Fixes applied: {fixes_applied} ({fixes_applied/total_samples*100:.1f}%)")
    print(f"Individual matches NEW batch: {individual_matches}/{total_samples} ({individual_matches/total_samples*100:.1f}%)")
    
    if old_vs_new_diffs:
        avg_chars_removed = sum(old_vs_new_diffs) / len(old_vs_new_diffs)
        print(f"Average characters removed per fix: {avg_chars_removed:.1f}")
        print(f"Max characters removed: {max(old_vs_new_diffs)}")
        print(f"Min characters removed: {min(old_vs_new_diffs)}")
    
    print(f"\nPerformance:")
    print(f"Batch inference time: {batch_time:.3f}s ({batch_time/total_samples*1000:.1f}ms per sample)")
    
    print(f"\n🎯 CONCLUSION:")
    if fixes_applied > 0:
        print(f"✅ Fix is WORKING! {fixes_applied} samples had post-EOS content removed.")
        if individual_matches == total_samples:
            print(f"✅ NEW batch method perfectly matches individual inference!")
        else:
            print(f"⚠️  NEW batch method matches individual inference in {individual_matches}/{total_samples} cases")
    else:
        print(f"🤔 No fixes needed in this batch - all samples may already be correct")

if __name__ == "__main__":
    test_comprehensive_fix(num_samples=20) 