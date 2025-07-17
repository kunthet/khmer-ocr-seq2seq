#!/usr/bin/env python3
"""
EOS-Focused Training Script V1
Specialized training to fix the critical issue of 0% EOS token generation.

Key features:
- Strong EOS loss weighting (20x pattern loss)
- EOS-specific teacher forcing schedule
- Enhanced EOS monitoring and debugging
- Built on successful pattern-fix foundations
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import ConfigManager
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset, OnTheFlyCollateFunction
from src.training.checkpoint_manager import CheckpointManager

class EOSFocusedLoss(nn.Module):
    """Enhanced loss function specifically designed to encourage EOS token generation."""
    
    def __init__(self, config, eos_weight=20.0, pattern_weight=2.0, sequence_weight=0.5):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=config.vocab.PAD_IDX)
        self.eos_idx = config.vocab.EOS_IDX
        self.pad_idx = config.vocab.PAD_IDX
        self.vocab_size = len(config.vocab)
        
        # EOS-focused weights (EOS gets 10x more importance than patterns)
        self.eos_weight = eos_weight
        self.pattern_weight = pattern_weight
        self.sequence_weight = sequence_weight
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_repetitive_patterns(self, sequences):
        """Detect repetitive patterns in generated sequences."""
        batch_size, seq_len = sequences.shape
        repetitive_count = 0
        
        for seq in sequences:
            # Convert to list, excluding padding
            tokens = []
            for token in seq:
                if token == self.pad_idx:
                    break
                tokens.append(token.item())
            
            if len(tokens) < 6:  # Too short to have patterns
                continue
                
            # Check for repetitive patterns (2-4 token cycles)
            is_repetitive = False
            for pattern_len in range(2, min(5, len(tokens)//3)):
                for start in range(len(tokens) - pattern_len * 2):
                    pattern = tokens[start:start+pattern_len]
                    # Check if pattern repeats at least 3 times
                    repeats = 1
                    pos = start + pattern_len
                    while pos + pattern_len <= len(tokens):
                        if tokens[pos:pos+pattern_len] == pattern:
                            repeats += 1
                            pos += pattern_len
                        else:
                            break
                    
                    if repeats >= 3:
                        is_repetitive = True
                        break
                
                if is_repetitive:
                    break
            
            if is_repetitive:
                repetitive_count += 1
        
        return repetitive_count
    
    def compute_eos_encouragement_loss(self, logits, targets):
        """Compute specialized loss to encourage EOS token generation at appropriate positions."""
        batch_size, seq_len, vocab_size = logits.shape
        eos_loss = 0.0
        eos_predictions = 0
        eos_targets = 0
        
        for b in range(batch_size):
            target_seq = targets[b]
            logit_seq = logits[b]
            
            # Find where EOS should be in target
            eos_positions = (target_seq == self.eos_idx).nonzero(as_tuple=True)[0]
            
            if len(eos_positions) > 0:
                eos_pos = eos_positions[0].item()
                eos_targets += 1
                
                # Strong encouragement for EOS at correct position
                eos_logit = logit_seq[eos_pos, self.eos_idx]
                max_other_logit = torch.max(torch.cat([
                    logit_seq[eos_pos, :self.eos_idx],
                    logit_seq[eos_pos, self.eos_idx+1:]
                ]))
                
                # Margin loss: EOS should be higher than any other token
                margin_loss = torch.clamp(max_other_logit - eos_logit + 1.0, min=0.0)
                eos_loss += margin_loss
                
                # Check if model actually predicts EOS
                pred_token = torch.argmax(logit_seq[eos_pos])
                if pred_token == self.eos_idx:
                    eos_predictions += 1
        
        eos_loss = eos_loss / max(batch_size, 1)
        eos_success_rate = eos_predictions / max(eos_targets, 1)
        
        return eos_loss, eos_success_rate, eos_predictions, eos_targets
    
    def forward(self, logits, targets):
        # Handle different output formats from model
        if isinstance(logits, dict):
            if 'logits' in logits:
                logits = logits['logits']
            elif 'predictions' in logits:
                logits = logits['predictions']
            else:
                # Try to find the main tensor output
                tensor_key = next((k for k, v in logits.items() if torch.is_tensor(v) and v.dim() == 3), None)
                if tensor_key:
                    logits = logits[tensor_key]
                else:
                    raise ValueError(f"Could not find logits in model output: {logits.keys()}")
        
        # Main cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        target_batch, target_seq = targets.shape
        
        # Handle shape mismatch during validation (different sequence lengths)
        if seq_len != target_seq:
            min_len = min(seq_len, target_seq)
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]
            seq_len = min_len
        
        main_loss = self.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        # Get predictions for analysis
        predictions = torch.argmax(logits, dim=-1)
        
        # Pattern detection
        repetitive_count = self.detect_repetitive_patterns(predictions)
        pattern_loss = (repetitive_count / batch_size) * 100.0  # Convert to percentage penalty
        
        # EOS encouragement loss
        eos_loss, eos_success_rate, eos_preds, eos_targets = self.compute_eos_encouragement_loss(logits, targets)
        
        # Sequence completion bonus (encourage shorter, complete sequences)
        avg_pred_length = torch.sum((predictions != self.pad_idx).float(), dim=1).mean()
        avg_target_length = torch.sum((targets != self.pad_idx).float(), dim=1).mean()
        length_penalty = torch.abs(avg_pred_length - avg_target_length) / avg_target_length
        
        # Combined loss with EOS focus
        total_loss = (main_loss + 
                     self.eos_weight * eos_loss + 
                     self.pattern_weight * pattern_loss + 
                     self.sequence_weight * length_penalty)
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'eos_loss': eos_loss,
            'pattern_loss': pattern_loss,
            'length_penalty': length_penalty,
            'eos_success_rate': eos_success_rate,
            'eos_predictions': eos_preds,
            'eos_targets': eos_targets,
            'repetitive_count': repetitive_count
        }

def format_loss_value(loss):
    """Format loss value for logging (handle both tensors and floats)."""
    if hasattr(loss, 'item'):
        return f"{loss.item():.4f}"
    else:
        return f"{loss:.4f}"

def analyze_eos_patterns(dataloader, model, config, num_samples=50):
    """Analyze EOS generation patterns in detail."""
    model.eval()
    eos_analysis = {
        'total_sequences': 0,
        'eos_generated': 0,
        'eos_at_correct_position': 0,
        'sequences_with_eos': 0,
        'avg_pred_length': 0.0,
        'avg_target_length': 0.0,
        'sample_predictions': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= num_samples:
                break
                
            images = batch['images'].to(config.device)
            targets = batch['targets'].to(config.device)
            
            # Generate predictions
            outputs = model(images, target_sequences=None)  # No teacher forcing
            
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
            
            predictions = torch.argmax(outputs, dim=-1)
            
            batch_size, seq_len = predictions.shape
            eos_analysis['total_sequences'] += batch_size
            
            for b in range(batch_size):
                pred_seq = predictions[b]
                target_seq = targets[b]
                
                # Find lengths (excluding padding)
                pred_length = torch.sum((pred_seq != config.vocab.PAD_IDX).float()).item()
                target_length = torch.sum((target_seq != config.vocab.PAD_IDX).float()).item()
                
                eos_analysis['avg_pred_length'] += pred_length
                eos_analysis['avg_target_length'] += target_length
                
                # Check EOS generation
                pred_eos_positions = (pred_seq == config.vocab.EOS_IDX).nonzero(as_tuple=True)[0]
                target_eos_positions = (target_seq == config.vocab.EOS_IDX).nonzero(as_tuple=True)[0]
                
                if len(pred_eos_positions) > 0:
                    eos_analysis['sequences_with_eos'] += 1
                    eos_analysis['eos_generated'] += len(pred_eos_positions)
                    
                    # Check if EOS is at correct position
                    if len(target_eos_positions) > 0:
                        pred_eos_pos = pred_eos_positions[0].item()
                        target_eos_pos = target_eos_positions[0].item()
                        if abs(pred_eos_pos - target_eos_pos) <= 1:  # Allow 1 position tolerance
                            eos_analysis['eos_at_correct_position'] += 1
                
                # Store sample for debugging
                if len(eos_analysis['sample_predictions']) < 10:
                    pred_tokens = pred_seq[:int(pred_length)].cpu().tolist()
                    target_tokens = target_seq[:int(target_length)].cpu().tolist()
                    eos_analysis['sample_predictions'].append({
                        'prediction': pred_tokens,
                        'target': target_tokens,
                        'pred_length': pred_length,
                        'target_length': target_length
                    })
    
    # Calculate averages
    total_seqs = eos_analysis['total_sequences']
    if total_seqs > 0:
        eos_analysis['avg_pred_length'] /= total_seqs
        eos_analysis['avg_target_length'] /= total_seqs
        eos_analysis['eos_generation_rate'] = eos_analysis['sequences_with_eos'] / total_seqs
        eos_analysis['eos_accuracy_rate'] = eos_analysis['eos_at_correct_position'] / total_seqs
    
    return eos_analysis

def train_epoch(model, dataloader, criterion, optimizer, scheduler, config, epoch, logger):
    """Train one epoch with EOS-focused monitoring."""
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_eos_loss = 0.0
    total_pattern_loss = 0.0
    total_eos_success = 0.0
    total_samples = 0
    
    print("="*100)
    print(f"Epoch {epoch}")
    logger.info(f"Epoch {epoch}")
    print("="*100)

    # EOS-specific teacher forcing schedule
    # Start high (0.8) to show EOS examples, reduce quickly
    teacher_forcing_ratio = max(0.8 - epoch * 0.08, 0.1)
    print(f"Teacher forcing ratio: {teacher_forcing_ratio}")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        images = batch['images'].to(config.device)
        targets = batch['targets'].to(config.device)
        
        # Forward pass with EOS-focused teacher forcing
        outputs = model(images, target_sequences=targets, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Compute EOS-focused loss
        loss_dict = criterion(outputs, targets)
        
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Accumulate metrics
        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss_dict['total_loss'].item() * batch_size
        total_main_loss += loss_dict['main_loss'].item() * batch_size
        total_eos_loss += loss_dict['eos_loss'].item() * batch_size
        total_pattern_loss += loss_dict['pattern_loss'] * batch_size
        total_eos_success += loss_dict['eos_success_rate'] * batch_size
        
        # Log progress
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            speed = total_samples / elapsed if elapsed > 0 else 0
            
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                       f"Loss: {format_loss_value(loss_dict['total_loss'])} "
                       f"(Main: {format_loss_value(loss_dict['main_loss'])}, "
                       f"EOS: {format_loss_value(loss_dict['eos_loss'])}, "
                       f"Pattern: {loss_dict['pattern_loss']:.1f}), "
                       f"EOS Success: {loss_dict['eos_success_rate']:.1%}, "
                       f"TF: {teacher_forcing_ratio:.3f}, "
                       f"Speed: {speed:.1f} samples/sec")
    
    # Return epoch averages
    return {
        'total_loss': total_loss / total_samples,
        'main_loss': total_main_loss / total_samples,
        'eos_loss': total_eos_loss / total_samples,
        'pattern_loss': total_pattern_loss / total_samples,
        'eos_success': total_eos_success / total_samples,
        'teacher_forcing': teacher_forcing_ratio
    }

def validate_epoch(model, dataloader, criterion, config, logger):
    """Validate with detailed EOS analysis."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(config.device)
            targets = batch['targets'].to(config.device)
            
            outputs = model(images, target_sequences=None)  # No teacher forcing
            loss_dict = criterion(outputs, targets)
            
            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss_dict['total_loss'].item() * batch_size
    
    # Detailed EOS analysis
    eos_analysis = analyze_eos_patterns(dataloader, model, config, num_samples=200)
    
    avg_loss = total_loss / total_samples
    return avg_loss, eos_analysis

def main():
    parser = argparse.ArgumentParser(description='EOS-Focused Training V1')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--samples-per-epoch', type=int, default=2000, help='Training samples per epoch')
    parser.add_argument('--val-samples', type=int, default=200, help='Validation samples')
    parser.add_argument('--eos-weight', type=float, default=20.0, help='EOS loss weight')
    parser.add_argument('--pattern-weight', type=float, default=2.0, help='Pattern loss weight')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/eos_focused_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('EOSFocusedV1')
    
    logger.info(">> Starting EOS-Focused Training V1...")
    logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"EOS weight: {args.eos_weight}, Pattern weight: {args.pattern_weight}")
    
    # Load configuration
    config = ConfigManager()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {config.device}")
    
    # Create datasets with EOS focus
    logger.info(">> Creating EOS-optimized datasets...")
    
    train_dataset = OnTheFlyDataset(
        config_manager=config,
        split='train',
        samples_per_epoch=args.samples_per_epoch,
        augment_prob=0.8
    )
    
    val_dataset = OnTheFlyDataset(
        config_manager=config,
        split='val',
        samples_per_epoch=args.val_samples,
        augment_prob=0.0
    )
    
    # Create data loaders
    collate_fn = OnTheFlyCollateFunction(config.vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    logger.info(f"[OK] Training dataset: {args.samples_per_epoch} samples per epoch")
    logger.info(f"[OK] Validation dataset: {args.val_samples} samples")
    
    # Load model
    logger.info(">> Loading model...")
    model = KhmerOCRSeq2Seq(config).to(config.device)
    
    # Load existing checkpoint if available
    checkpoint_manager = CheckpointManager(checkpoint_dir='models/checkpoints')
    
    try:
        checkpoint_path = checkpoint_manager.get_latest_checkpoint()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        else:
            logger.info("No checkpoint found, starting from scratch")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Setup training components
    criterion = EOSFocusedLoss(
        config, 
        eos_weight=args.eos_weight, 
        pattern_weight=args.pattern_weight
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # OneCycle scheduler for EOS training
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Quick warmup for EOS focus
        anneal_strategy='cos'
    )
    
    logger.info(">> Starting EOS-focused training...")
    
    best_eos_score = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch+1}/{args.epochs} - EOS-FOCUSED TRAINING V1")
        logger.info(f"{'='*60}")
        
        # Train epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch, logger)
        
        # Validate epoch
        val_loss, eos_analysis = validate_epoch(model, val_loader, criterion, config, logger)
        
        # Calculate EOS-focused score
        eos_score = (eos_analysis['eos_generation_rate'] * 100 + 
                    eos_analysis['eos_accuracy_rate'] * 100 + 
                    max(0, 100 - eos_analysis.get('repetitive_patterns', 100)))
        
        # Log results
        logger.info(f"\n[RESULTS] EPOCH {epoch+1} RESULTS:")
        logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                   f"(Main: {train_metrics['main_loss']:.4f}, "
                   f"EOS: {train_metrics['eos_loss']:.4f}, "
                   f"Pattern: {train_metrics['pattern_loss']:.1f})")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  EOS Analysis:")
        logger.info(f"    Sequences with EOS: {eos_analysis['sequences_with_eos']}/{eos_analysis['total_sequences']} "
                   f"({eos_analysis['eos_generation_rate']:.1%})")
        logger.info(f"    EOS at correct position: {eos_analysis['eos_at_correct_position']}/{eos_analysis['total_sequences']} "
                   f"({eos_analysis['eos_accuracy_rate']:.1%})")
        logger.info(f"    Avg length - Pred: {eos_analysis['avg_pred_length']:.1f}, "
                   f"Target: {eos_analysis['avg_target_length']:.1f}")
        logger.info(f"    EOS Score: {eos_score:.1f}")
        
        # Save best model based on EOS performance
        if eos_score > best_eos_score:
            best_eos_score = eos_score
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_loss,
                'eos_score': eos_score,
                'eos_analysis': eos_analysis
            }
            
            checkpoint_manager.save_checkpoint(checkpoint, f'eos_focused_epoch_{epoch+1}')
            logger.info(f"  [BEST] New best EOS score: {eos_score:.1f}")
        
        # Sample prediction analysis
        if len(eos_analysis['sample_predictions']) > 0:
            logger.info(f"  Sample Predictions:")
            for i, sample in enumerate(eos_analysis['sample_predictions'][:3]):
                logger.info(f"    Sample {i+1}: Pred len={sample['pred_length']}, "
                           f"Target len={sample['target_length']}")
                # Show first few tokens
                pred_str = str(sample['prediction'][:10])
                target_str = str(sample['target'][:10])
                logger.info(f"      Pred:   {pred_str}...")
                logger.info(f"      Target: {target_str}...")
    
    logger.info(f"\n[COMPLETE] EOS-focused training completed!")
    logger.info(f"Best EOS score: {best_eos_score:.1f} at epoch {best_epoch}")

if __name__ == "__main__":
    main() 