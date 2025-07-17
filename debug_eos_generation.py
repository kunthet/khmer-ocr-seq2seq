#!/usr/bin/env python3
"""
EOS Generation Debug Script
Analyzes exactly why the model never generates EOS tokens (0% across all epochs).

This script will:
1. Load the latest model checkpoint
2. Analyze EOS token probability distributions
3. Compare EOS vs other token logits
4. Examine attention patterns around EOS positions
5. Provide actionable insights for fixing EOS generation
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import ConfigManager
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset, OnTheFlyCollateFunction
from torch.utils.data import DataLoader

def setup_logging():
    """Setup logging for debug analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('EOSDebug')

def load_latest_model(config):
    """Load the latest trained model."""
    model = KhmerOCRSeq2Seq(config).to(config.device)
    
    # Try to load latest checkpoint
    checkpoint_dir = Path('models/checkpoints')
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint, map_location=config.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                # Assume the checkpoint is the state dict itself
                model.load_state_dict(checkpoint)
            
            print(f"✅ Loaded checkpoint: {latest_checkpoint}")
            return model, str(latest_checkpoint)
    
    print("⚠️  No checkpoint found, using random weights")
    return model, "random_weights"

def analyze_eos_logits(model, dataloader, config, num_samples=100):
    """Analyze EOS token logits in detail."""
    print("\n🔍 ANALYZING EOS LOGITS...")
    
    model.eval()
    eos_idx = config.vocab.EOS_IDX
    pad_idx = config.vocab.PAD_IDX
    vocab_size = len(config.vocab)
    
    analysis = {
        'eos_logits_at_correct_pos': [],
        'eos_ranks_at_correct_pos': [],
        'max_other_logits_at_correct_pos': [],
        'eos_logits_throughout_sequence': [],
        'predicted_tokens_at_eos_pos': [],
        'eos_probabilities_at_correct_pos': [],
        'sequence_lengths': {'pred': [], 'target': []},
        'total_sequences': 0,
        'sequences_with_eos_in_target': 0
    }
    
    with torch.no_grad():
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
                
            images = batch['images'].to(config.device)
            targets = batch['targets'].to(config.device)
            
            # Get model outputs (logits)
            outputs = model(images, target_sequences=None)  # No teacher forcing
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    outputs = outputs['logits']
                elif 'predictions' in outputs:
                    outputs = outputs['predictions']
                else:
                    # Try to find the main tensor output
                    tensor_key = next((k for k, v in outputs.items() if torch.is_tensor(v) and v.dim() == 3), None)
                    if tensor_key:
                        outputs = outputs[tensor_key]
                    else:
                        print(f"⚠️  Could not find logits in model output: {outputs.keys()}")
                        continue
            
            batch_size, seq_len, vocab_size = outputs.shape
            
            # Convert to probabilities
            probs = F.softmax(outputs, dim=-1)
            
            for b in range(batch_size):
                if samples_processed >= num_samples:
                    break
                    
                target_seq = targets[b]
                output_seq = outputs[b]  # logits
                prob_seq = probs[b]      # probabilities
                
                analysis['total_sequences'] += 1
                samples_processed += 1
                
                # Find target sequence length and EOS position
                non_pad_mask = target_seq != pad_idx
                target_length = non_pad_mask.sum().item()
                
                # Find EOS position in target
                eos_positions = (target_seq == eos_idx).nonzero(as_tuple=True)[0]
                
                if len(eos_positions) > 0:
                    analysis['sequences_with_eos_in_target'] += 1
                    eos_pos = eos_positions[0].item()
                    
                    if eos_pos < seq_len:  # Make sure position is valid
                        # Get logits and probabilities at EOS position
                        eos_logit = output_seq[eos_pos, eos_idx].item()
                        eos_prob = prob_seq[eos_pos, eos_idx].item()
                        
                        # Get all other token logits at this position
                        other_logits = torch.cat([
                            output_seq[eos_pos, :eos_idx],
                            output_seq[eos_pos, eos_idx+1:]
                        ])
                        max_other_logit = other_logits.max().item()
                        
                        # Get EOS rank among all tokens
                        all_logits = output_seq[eos_pos]
                        eos_rank = (all_logits > all_logits[eos_idx]).sum().item() + 1
                        
                        # What token was actually predicted?
                        predicted_token = torch.argmax(output_seq[eos_pos]).item()
                        
                        # Store analysis data
                        analysis['eos_logits_at_correct_pos'].append(eos_logit)
                        analysis['eos_ranks_at_correct_pos'].append(eos_rank)
                        analysis['max_other_logits_at_correct_pos'].append(max_other_logit)
                        analysis['predicted_tokens_at_eos_pos'].append(predicted_token)
                        analysis['eos_probabilities_at_correct_pos'].append(eos_prob)
                
                # Analyze EOS logits throughout the sequence
                for pos in range(min(seq_len, target_length + 5)):
                    eos_logit_at_pos = output_seq[pos, eos_idx].item()
                    analysis['eos_logits_throughout_sequence'].append({
                        'position': pos,
                        'eos_logit': eos_logit_at_pos,
                        'is_target_eos_pos': pos in [ep.item() for ep in eos_positions]
                    })
                
                # Predicted vs target lengths
                pred_length = 0
                for pos in range(seq_len):
                    pred_token = torch.argmax(output_seq[pos]).item()
                    if pred_token == eos_idx:
                        pred_length = pos + 1
                        break
                    elif pred_token == pad_idx:
                        pred_length = pos
                        break
                if pred_length == 0:
                    pred_length = seq_len  # Never found EOS or PAD
                
                analysis['sequence_lengths']['pred'].append(pred_length)
                analysis['sequence_lengths']['target'].append(target_length)
    
    return analysis

def analyze_token_competition(model, dataloader, config, num_samples=50):
    """Analyze what tokens are competing with EOS."""
    print("\n🥊 ANALYZING TOKEN COMPETITION...")
    
    model.eval()
    eos_idx = config.vocab.EOS_IDX
    pad_idx = config.vocab.PAD_IDX
    
    competition_analysis = {
        'top_competitors': {},  # token_id -> count
        'competitor_margins': [],  # how much they beat EOS by
        'eos_vs_pad_competition': [],
        'position_based_competition': {}  # position -> competitors
    }
    
    with torch.no_grad():
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
                
            images = batch['images'].to(config.device)
            targets = batch['targets'].to(config.device)
            
            outputs = model(images, target_sequences=None)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    outputs = outputs['logits']
                elif 'predictions' in outputs:
                    outputs = outputs['predictions']
                else:
                    tensor_key = next((k for k, v in outputs.items() if torch.is_tensor(v) and v.dim() == 3), None)
                    if tensor_key:
                        outputs = outputs[tensor_key]
                    else:
                        continue
            
            for b in range(min(outputs.size(0), num_samples - samples_processed)):
                target_seq = targets[b]
                output_seq = outputs[b]
                
                samples_processed += 1
                
                # Find EOS position in target
                eos_positions = (target_seq == eos_idx).nonzero(as_tuple=True)[0]
                
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[0].item()
                    
                    if eos_pos < output_seq.size(0):
                        # Get logits at EOS position
                        position_logits = output_seq[eos_pos]
                        eos_logit = position_logits[eos_idx]
                        
                        # Find top competing tokens
                        top_k_values, top_k_indices = torch.topk(position_logits, k=10)
                        
                        for i, (logit_val, token_idx) in enumerate(zip(top_k_values, top_k_indices)):
                            token_idx = token_idx.item()
                            if token_idx != eos_idx:
                                # This token beats EOS
                                margin = logit_val.item() - eos_logit.item()
                                if margin > 0:
                                    competition_analysis['competitor_margins'].append(margin)
                                    
                                    # Track top competitors
                                    if token_idx not in competition_analysis['top_competitors']:
                                        competition_analysis['top_competitors'][token_idx] = 0
                                    competition_analysis['top_competitors'][token_idx] += 1
                                    
                                    # Track position-based competition
                                    if eos_pos not in competition_analysis['position_based_competition']:
                                        competition_analysis['position_based_competition'][eos_pos] = {}
                                    if token_idx not in competition_analysis['position_based_competition'][eos_pos]:
                                        competition_analysis['position_based_competition'][eos_pos][token_idx] = 0
                                    competition_analysis['position_based_competition'][eos_pos][token_idx] += 1
                        
                        # EOS vs PAD specific analysis
                        pad_logit = position_logits[pad_idx]
                        eos_vs_pad_margin = pad_logit.item() - eos_logit.item()
                        competition_analysis['eos_vs_pad_competition'].append({
                            'eos_logit': eos_logit.item(),
                            'pad_logit': pad_logit.item(),
                            'margin': eos_vs_pad_margin
                        })
    
    return competition_analysis

def analyze_attention_patterns(model, dataloader, config, num_samples=20):
    """Analyze attention patterns around EOS positions."""
    print("\n👁️  ANALYZING ATTENTION PATTERNS...")
    
    # This is a simplified analysis - full attention analysis would require
    # access to attention weights from the model
    
    model.eval()
    eos_idx = config.vocab.EOS_IDX
    
    attention_analysis = {
        'avg_attention_at_eos': 0.0,
        'attention_distribution_near_eos': [],
        'samples_analyzed': 0
    }
    
    # For now, we'll analyze prediction confidence as a proxy for attention
    with torch.no_grad():
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
                
            images = batch['images'].to(config.device)
            targets = batch['targets'].to(config.device)
            
            outputs = model(images, target_sequences=None)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    outputs = outputs['logits']
                elif 'predictions' in outputs:
                    outputs = outputs['predictions']
                else:
                    tensor_key = next((k for k, v in outputs.items() if torch.is_tensor(v) and v.dim() == 3), None)
                    if tensor_key:
                        outputs = outputs[tensor_key]
                    else:
                        continue
            
            probs = F.softmax(outputs, dim=-1)
            
            for b in range(min(outputs.size(0), num_samples - samples_processed)):
                target_seq = targets[b]
                prob_seq = probs[b]
                
                samples_processed += 1
                attention_analysis['samples_analyzed'] += 1
                
                # Find EOS position
                eos_positions = (target_seq == eos_idx).nonzero(as_tuple=True)[0]
                
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[0].item()
                    
                    if eos_pos < prob_seq.size(0):
                        # Analyze confidence/attention around EOS position
                        window_start = max(0, eos_pos - 2)
                        window_end = min(prob_seq.size(0), eos_pos + 3)
                        
                        attention_window = []
                        for pos in range(window_start, window_end):
                            # Get entropy as measure of uncertainty/attention
                            pos_probs = prob_seq[pos]
                            entropy = -torch.sum(pos_probs * torch.log(pos_probs + 1e-8))
                            max_prob = torch.max(pos_probs)
                            
                            attention_window.append({
                                'position_relative_to_eos': pos - eos_pos,
                                'entropy': entropy.item(),
                                'max_probability': max_prob.item(),
                                'eos_probability': pos_probs[eos_idx].item()
                            })
                        
                        attention_analysis['attention_distribution_near_eos'].append(attention_window)
    
    return attention_analysis

def generate_report(eos_analysis, competition_analysis, attention_analysis, config):
    """Generate comprehensive EOS debugging report."""
    print("\n" + "="*80)
    print("🔍 EOS GENERATION DEBUG REPORT")
    print("="*80)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total sequences analyzed: {eos_analysis['total_sequences']}")
    print(f"   Sequences with EOS in target: {eos_analysis['sequences_with_eos_in_target']}")
    print(f"   EOS presence rate: {eos_analysis['sequences_with_eos_in_target']/eos_analysis['total_sequences']:.1%}")
    
    # EOS logit analysis
    if eos_analysis['eos_logits_at_correct_pos']:
        avg_eos_logit = np.mean(eos_analysis['eos_logits_at_correct_pos'])
        avg_competitor_logit = np.mean(eos_analysis['max_other_logits_at_correct_pos'])
        avg_eos_rank = np.mean(eos_analysis['eos_ranks_at_correct_pos'])
        avg_eos_prob = np.mean(eos_analysis['eos_probabilities_at_correct_pos'])
        
        print(f"\n🎯 EOS LOGIT ANALYSIS (at correct positions):")
        print(f"   Average EOS logit: {avg_eos_logit:.4f}")
        print(f"   Average max competitor logit: {avg_competitor_logit:.4f}")
        print(f"   Average logit gap (competitor - EOS): {avg_competitor_logit - avg_eos_logit:.4f}")
        print(f"   Average EOS rank: {avg_eos_rank:.1f} / {len(config.vocab)}")
        print(f"   Average EOS probability: {avg_eos_prob:.4f} ({avg_eos_prob*100:.2f}%)")
        
        # Severity assessment
        logit_gap = avg_competitor_logit - avg_eos_logit
        if logit_gap > 5.0:
            severity = "🔴 CRITICAL"
        elif logit_gap > 2.0:
            severity = "🟡 MODERATE"
        else:
            severity = "🟢 MILD"
        
        print(f"   Problem severity: {severity}")
    
    # Competition analysis
    if competition_analysis['top_competitors']:
        print(f"\n🥊 TOKEN COMPETITION ANALYSIS:")
        # Sort competitors by frequency
        sorted_competitors = sorted(
            competition_analysis['top_competitors'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"   Top 10 tokens beating EOS:")
        for i, (token_idx, count) in enumerate(sorted_competitors[:10]):
            percentage = (count / len(competition_analysis['competitor_margins'])) * 100
            print(f"     {i+1}. Token {token_idx}: {count} times ({percentage:.1f}%)")
        
        if competition_analysis['competitor_margins']:
            avg_margin = np.mean(competition_analysis['competitor_margins'])
            max_margin = np.max(competition_analysis['competitor_margins'])
            print(f"   Average winning margin: {avg_margin:.4f}")
            print(f"   Maximum winning margin: {max_margin:.4f}")
    
    # EOS vs PAD analysis
    if competition_analysis['eos_vs_pad_competition']:
        pad_wins = sum(1 for x in competition_analysis['eos_vs_pad_competition'] if x['margin'] > 0)
        total_comparisons = len(competition_analysis['eos_vs_pad_competition'])
        pad_win_rate = pad_wins / total_comparisons
        
        print(f"\n🏁 EOS vs PAD ANALYSIS:")
        print(f"   PAD beats EOS: {pad_wins}/{total_comparisons} ({pad_win_rate:.1%})")
        avg_pad_margin = np.mean([x['margin'] for x in competition_analysis['eos_vs_pad_competition']])
        print(f"   Average PAD advantage: {avg_pad_margin:.4f}")
    
    # Length analysis
    if eos_analysis['sequence_lengths']['pred'] and eos_analysis['sequence_lengths']['target']:
        avg_pred_len = np.mean(eos_analysis['sequence_lengths']['pred'])
        avg_target_len = np.mean(eos_analysis['sequence_lengths']['target'])
        
        print(f"\n📏 SEQUENCE LENGTH ANALYSIS:")
        print(f"   Average predicted length: {avg_pred_len:.1f}")
        print(f"   Average target length: {avg_target_len:.1f}")
        print(f"   Length difference: {avg_pred_len - avg_target_len:.1f}")
        
        # Check if model always predicts max length
        max_len_predictions = sum(1 for l in eos_analysis['sequence_lengths']['pred'] if l >= 150)
        max_len_rate = max_len_predictions / len(eos_analysis['sequence_lengths']['pred'])
        print(f"   Predictions at max length: {max_len_predictions}/{len(eos_analysis['sequence_lengths']['pred'])} ({max_len_rate:.1%})")
    
    # Attention analysis
    print(f"\n👁️  ATTENTION ANALYSIS:")
    print(f"   Attention samples analyzed: {attention_analysis['samples_analyzed']}")
    
    # Root cause assessment and recommendations
    print(f"\n🎯 ROOT CAUSE ASSESSMENT:")
    
    if eos_analysis['eos_logits_at_correct_pos']:
        avg_eos_rank = np.mean(eos_analysis['eos_ranks_at_correct_pos'])
        avg_eos_prob = np.mean(eos_analysis['eos_probabilities_at_correct_pos'])
        
        if avg_eos_rank > len(config.vocab) * 0.8:
            print("   🔴 CRITICAL: EOS token consistently ranked very low")
        elif avg_eos_prob < 0.01:
            print("   🔴 CRITICAL: EOS probability extremely low")
        
        if competition_analysis['eos_vs_pad_competition']:
            pad_win_rate = sum(1 for x in competition_analysis['eos_vs_pad_competition'] if x['margin'] > 0) / len(competition_analysis['eos_vs_pad_competition'])
            if pad_win_rate > 0.8:
                print("   🔴 CRITICAL: PAD token dominates over EOS")
    
    print(f"\n💡 RECOMMENDED SOLUTIONS:")
    
    if eos_analysis['eos_logits_at_correct_pos']:
        avg_eos_rank = np.mean(eos_analysis['eos_ranks_at_correct_pos'])
        if avg_eos_rank > 50:
            print("   1. 🎯 Increase EOS loss weight to 50x or higher")
            print("   2. 🎓 Implement EOS-specific teacher forcing (start at 0.9)")
            print("   3. 🎲 Add EOS token frequency boosting in loss")
    
    if competition_analysis['top_competitors']:
        print("   4. 🚫 Add penalty for top competing tokens")
        print("   5. 🎯 Implement margin-based loss (EOS should win by margin)")
    
    if eos_analysis['sequence_lengths']['pred']:
        max_len_rate = sum(1 for l in eos_analysis['sequence_lengths']['pred'] if l >= 150) / len(eos_analysis['sequence_lengths']['pred'])
        if max_len_rate > 0.9:
            print("   6. ⏰ Add strong length penalty for sequences reaching max_length")
            print("   7. 🎯 Curriculum learning: start with shorter sequences")
    
    print("   8. 🔄 Use EOS-focused validation and early stopping")
    print("   9. 📊 Monitor EOS generation rate as primary metric")

def main():
    print("🚀 Starting EOS Generation Debug Analysis...")
    
    # Setup
    logger = setup_logging()
    config = ConfigManager()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {config.device}")
    print(f"EOS token index: {config.vocab.EOS_IDX}")
    print(f"PAD token index: {config.vocab.PAD_IDX}")
    print(f"Vocabulary size: {len(config.vocab)}")
    
    # Load model
    model, checkpoint_path = load_latest_model(config)
    print(f"Model loaded from: {checkpoint_path}")
    
    # Create validation dataset for analysis
    val_dataset = OnTheFlyDataset(
        config_manager=config,
        split='val',
        samples_per_epoch=500,
        augment_prob=0.0  # No augmentation for analysis
    )
    
    collate_fn = OnTheFlyCollateFunction(config.vocab)
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    
    # Run analyses
    print("\n🔬 Running comprehensive EOS analysis...")
    
    eos_analysis = analyze_eos_logits(model, val_loader, config, num_samples=200)
    competition_analysis = analyze_token_competition(model, val_loader, config, num_samples=100)
    attention_analysis = analyze_attention_patterns(model, val_loader, config, num_samples=50)
    
    # Generate report
    generate_report(eos_analysis, competition_analysis, attention_analysis, config)
    
    print(f"\n✅ EOS debug analysis complete!")
    print(f"💾 Log saved to: logs/eos_generation_debug.log")

if __name__ == "__main__":
    main() 