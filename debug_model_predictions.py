#!/usr/bin/env python3
"""
Debug script to examine what the model is actually generating during validation.
This will help us understand why CER is >100% despite decreasing loss.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.seq2seq import KhmerOCRSeq2Seq
from src.utils.config import ConfigManager
from src.data.synthetic_dataset import SyntheticImageDataset
from src.inference.ocr_engine import KhmerOCREngine
from src.inference.metrics import OCRMetrics
import Levenshtein


def analyze_model_predictions():
    """Analyze what the model is actually generating"""
    print("=" * 80)
    print("🔍 KHMER OCR MODEL PREDICTION ANALYSIS")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load configuration
    config = ConfigManager()
    print(f"Vocabulary size: {len(config.vocab)}")
    print(f"Special tokens: SOS={config.vocab.SOS_IDX}, EOS={config.vocab.EOS_IDX}, PAD={config.vocab.PAD_IDX}, UNK={config.vocab.UNK_IDX}")
    
    # Create untrained model (to simulate early training)
    print("\n📊 Creating untrained model (simulating early training)...")
    model = KhmerOCRSeq2Seq(config)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load validation dataset
    print("\n📁 Loading validation samples...")
    
    # Create a quick validation dataset using OnTheFlyDataset
    print("🔄 Creating on-the-fly validation samples...")
    from src.data.onthefly_dataset import OnTheFlyDataset
    
    val_dataset = OnTheFlyDataset(
        split="val",
        config_manager=config,
        corpus_dir="data/processed",
        samples_per_epoch=10,
        augment_prob=0.0,  # No augmentation for validation
        shuffle_texts=False,
        random_seed=42
    )
    print(f"✅ Created on-the-fly validation set: 10 samples")
    
    # Analyze predictions
    print("\n🔍 ANALYZING MODEL PREDICTIONS")
    print("=" * 80)
    
    predictions = []
    targets = []
    raw_sequences = []
    token_stats = {'special_tokens': 0, 'content_tokens': 0, 'total_tokens': 0}
    
    with torch.no_grad():
        for i in range(min(10, len(val_dataset))):
            # Get sample
            try:
                sample = val_dataset[i]
                if isinstance(sample, dict):
                    image = sample['image']
                    target_text = sample['text']
                elif isinstance(sample, tuple) and len(sample) >= 2:
                    image, target_text = sample[0], sample[1]
                    # Handle case where target_text might be a tensor
                    if hasattr(target_text, 'cpu'):
                        # Decode tensor to text if needed
                        target_text = config.vocab.decode(target_text.cpu().tolist())
                else:
                    print(f"   ⚠️ Unexpected sample format: {type(sample)}")
                    continue
            except Exception as e:
                print(f"   ❌ Error getting sample {i}: {e}")
                continue
            
            # Convert to proper format
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # Add batch dimension
                    image = image.unsqueeze(0)
                image = image.to(device)
            else:
                # Convert PIL to tensor
                image_array = np.array(image.convert('L'))
                image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0) / 255.0
                image = image_tensor.to(device)
            
            # Generate prediction
            result = model.generate(image, max_length=50, method='greedy')
            sequence = result['sequences'][0].cpu().tolist()
            
            # Decode sequence
            decoded_text = config.vocab.decode(sequence)
            
            # Store results
            predictions.append(decoded_text)
            targets.append(target_text)
            raw_sequences.append(sequence)
            
            # Analyze tokens
            for token in sequence:
                token_stats['total_tokens'] += 1
                if token in [config.vocab.SOS_IDX, config.vocab.EOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]:
                    token_stats['special_tokens'] += 1
                else:
                    token_stats['content_tokens'] += 1
            
            # Show detailed analysis for first few samples
            if i < 5:
                print(f"\n📝 Sample {i+1}:")
                print(f"   Target: '{target_text}' (len: {len(target_text)})")
                print(f"   Prediction: '{decoded_text}' (len: {len(decoded_text)})")
                print(f"   Raw sequence: {sequence[:20]}{'...' if len(sequence) > 20 else ''}")
                
                # Calculate CER for this sample
                cer = Levenshtein.distance(decoded_text, target_text) / len(target_text) if len(target_text) > 0 else 0
                print(f"   CER: {cer:.1%}")
                
                # Analyze tokens in detail
                print(f"   Token analysis:")
                for j, token in enumerate(sequence[:15]):  # First 15 tokens
                    token_char = config.vocab.idx_to_char.get(token, f"UNKNOWN_{token}")
                    if token == config.vocab.EOS_IDX:
                        print(f"     {j}: {token} -> 'EOS' (sequence ends here)")
                        break
                    elif token == config.vocab.SOS_IDX:
                        print(f"     {j}: {token} -> 'SOS'")
                    elif token == config.vocab.PAD_IDX:
                        print(f"     {j}: {token} -> 'PAD'")
                    elif token == config.vocab.UNK_IDX:
                        print(f"     {j}: {token} -> 'UNK'")
                    else:
                        print(f"     {j}: {token} -> '{token_char}'")
                
                if len(sequence) > 15:
                    print(f"     ... {len(sequence) - 15} more tokens")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print("=" * 80)
    
    # Token distribution
    special_ratio = token_stats['special_tokens'] / token_stats['total_tokens'] if token_stats['total_tokens'] > 0 else 0
    print(f"Token Distribution:")
    print(f"   Total tokens generated: {token_stats['total_tokens']}")
    print(f"   Special tokens: {token_stats['special_tokens']} ({special_ratio:.1%})")
    print(f"   Content tokens: {token_stats['content_tokens']} ({1-special_ratio:.1%})")
    
    # Length analysis
    pred_lengths = [len(p) for p in predictions]
    target_lengths = [len(t) for t in targets]
    
    print(f"\nLength Analysis:")
    print(f"   Avg target length: {np.mean(target_lengths):.1f}")
    print(f"   Avg prediction length: {np.mean(pred_lengths):.1f}")
    print(f"   Length ratio: {np.mean(pred_lengths)/np.mean(target_lengths):.2f}x")
    
    # CER analysis
    metrics = OCRMetrics()
    overall_cer = metrics.character_error_rate(predictions, targets)
    print(f"\nAccuracy Metrics:")
    print(f"   Overall CER: {overall_cer:.1%}")
    print(f"   Expected CER for untrained model: 80-120%")
    
    # Pattern analysis
    print(f"\nPattern Analysis:")
    
    # Check for repetitive patterns
    repetitive_count = 0
    for seq in raw_sequences:
        # Check if sequence has repetitive patterns
        if len(seq) >= 4:
            # Look for repeating 2-token patterns
            for i in range(len(seq) - 3):
                if seq[i] == seq[i+2] and seq[i+1] == seq[i+3]:
                    repetitive_count += 1
                    break
    
    print(f"   Sequences with repetitive patterns: {repetitive_count}/{len(raw_sequences)}")
    
    # Check for predominantly special tokens
    mostly_special = 0
    for seq in raw_sequences:
        special_count = sum(1 for token in seq if token in [config.vocab.SOS_IDX, config.vocab.EOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX])
        if special_count / len(seq) > 0.8:
            mostly_special += 1
    
    print(f"   Sequences mostly special tokens (>80%): {mostly_special}/{len(raw_sequences)}")
    
    # Recommendations
    print(f"\n💡 DIAGNOSTIC RECOMMENDATIONS")
    print("=" * 80)
    
    if special_ratio > 0.7:
        print("❌ HIGH SPECIAL TOKEN RATIO")
        print("   - Model generating too many PAD/UNK tokens")
        print("   - Recommendations:")
        print("     • Reduce teacher forcing ratio (0.7 → 0.5)")
        print("     • Increase learning rate (1e-4 → 2e-4)")
        print("     • Add more regularization")
    
    if np.mean(pred_lengths) > np.mean(target_lengths) * 1.5:
        print("❌ SEQUENCES TOO LONG")
        print("   - Model not learning to generate EOS tokens")
        print("   - Recommendations:")
        print("     • Add length penalty during training")
        print("     • Increase EOS token weight in loss")
        print("     • Check attention mechanism alignment")
    
    if repetitive_count > len(raw_sequences) * 0.3:
        print("❌ REPETITIVE PATTERNS DETECTED")
        print("   - Model stuck in loops")
        print("   - Recommendations:")
        print("     • Add coverage mechanism")
        print("     • Reduce temperature during inference")
        print("     • Add attention dropout")
    
    if overall_cer > 1.5:  # 150%
        print("❌ EXTREMELY HIGH CER")
        print("   - Model predictions much longer than targets")
        print("   - This is causing CER > 100%")
        print("   - Priority: Fix sequence length generation")
    
    print(f"\n✅ Analysis complete. Use recommendations to improve training.")


if __name__ == "__main__":
    analyze_model_predictions() 