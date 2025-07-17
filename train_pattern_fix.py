#!/usr/bin/env python3
"""
Pattern-Fix Training Script for Khmer OCR

This script specifically addresses the repetitive pattern generation issue
where the model gets stuck generating sequences like "ិកឬិកឬិកឬ..." 
instead of learning proper Khmer text generation.

Key Features:
- Anti-repetition loss components
- EOS token encouragement  
- Progressive teacher forcing reduction
- Pattern detection and intervention
- Enhanced learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import logging
import time
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config import ConfigManager
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset, OnTheFlyCollateFunction
from src.data.synthetic_dataset import SyntheticImageDataset, SyntheticCollateFunction
from src.training.validator import Validator
import numpy as np


class AntiPatternLoss(nn.Module):
    """
    Custom loss component that penalizes repetitive patterns
    """
    def __init__(self, vocab_size, repetition_penalty=1.2, ngram_size=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.repetition_penalty = repetition_penalty
        self.ngram_size = ngram_size
        
    def forward(self, logits, targets):
        """
        Calculate anti-pattern loss
        
        Args:
            logits: Model output logits [B, T, vocab_size]
            targets: Target sequences [B, T]
        """
        batch_size, seq_len, _ = logits.shape
        
        # Get predicted tokens
        predicted_tokens = logits.argmax(dim=-1)  # [B, T]
        
        # Calculate repetition penalty
        repetition_loss = 0.0
        pattern_count = 0
        
        for b in range(batch_size):
            for t in range(seq_len - self.ngram_size + 1):
                # Check for ngram repetition
                ngram1 = predicted_tokens[b, t:t+self.ngram_size]
                
                # Look for this pattern in the rest of the sequence
                for t2 in range(t + self.ngram_size, seq_len - self.ngram_size + 1):
                    ngram2 = predicted_tokens[b, t2:t2+self.ngram_size]
                    
                    if torch.equal(ngram1, ngram2):
                        # Found repetition - add penalty
                        pattern_count += 1
                        
                        # Penalty is proportional to pattern probability
                        pattern_prob = torch.softmax(logits[b, t2:t2+self.ngram_size], dim=-1)
                        pattern_penalty = torch.mean(torch.gather(
                            pattern_prob, 
                            dim=-1, 
                            index=ngram2.unsqueeze(-1)
                        ))
                        
                        repetition_loss += pattern_penalty * self.repetition_penalty
        
        # Normalize by batch size and sequence length
        if pattern_count > 0:
            repetition_loss = repetition_loss / (batch_size * seq_len)
        
        return repetition_loss


class EOSEncouragementLoss(nn.Module):
    """
    Loss component that encourages proper EOS token generation
    """
    def __init__(self, eos_idx, bonus_weight=0.1):
        super().__init__()
        self.eos_idx = eos_idx
        self.bonus_weight = bonus_weight
        
    def forward(self, logits, targets, target_lengths):
        """
        Calculate EOS encouragement loss
        
        Args:
            logits: Model output logits [B, T, vocab_size]
            targets: Target sequences [B, T]
            target_lengths: Actual target lengths [B]
        """
        batch_size = logits.shape[0]
        eos_bonus = 0.0
        
        for b in range(batch_size):
            target_length = target_lengths[b].item()
            if target_length > 1:
                # Expected EOS position
                expected_eos_pos = target_length - 1
                
                if expected_eos_pos < logits.shape[1]:
                    # Bonus for predicting EOS at the right position
                    eos_prob = torch.softmax(logits[b, expected_eos_pos], dim=-1)[self.eos_idx]
                    eos_bonus += eos_prob * self.bonus_weight
        
        # Negative because we want to maximize this (minimize negative)
        return -eos_bonus / batch_size


class PatternFixTrainer:
    """
    Specialized trainer that addresses repetitive pattern generation
    """
    
    def __init__(self, config_manager, device):
        self.config = config_manager
        self.device = device
        
        # Initialize model
        self.model = KhmerOCRSeq2Seq(config_manager).to(device)
        
        # Enhanced optimizer with higher learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=2e-4,  # Higher learning rate for faster pattern learning
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss components
        self.criterion = nn.NLLLoss(ignore_index=config_manager.vocab.PAD_IDX)
        self.anti_pattern_loss = AntiPatternLoss(
            vocab_size=len(config_manager.vocab),
            repetition_penalty=1.2,
            ngram_size=3
        )
        self.eos_loss = EOSEncouragementLoss(
            eos_idx=config_manager.vocab.EOS_IDX,
            bonus_weight=0.1
        )
        
        # Progressive teacher forcing
        self.initial_teacher_forcing = 0.7
        self.final_teacher_forcing = 0.3
        self.teacher_forcing_decay_steps = 10000
        self.global_step = 0
        
        # Pattern detection
        self.pattern_warnings = 0
        self.max_pattern_warnings = 5
        
        # Validator
        self.validator = Validator(self.model, config_manager, device)
        
        # Logger
        self.logger = logging.getLogger('PatternFixTrainer')
        
    def get_current_teacher_forcing_ratio(self):
        """Calculate current teacher forcing ratio with progressive reduction"""
        progress = min(self.global_step / self.teacher_forcing_decay_steps, 1.0)
        
        # Exponential decay
        ratio = self.initial_teacher_forcing * (
            (self.final_teacher_forcing / self.initial_teacher_forcing) ** progress
        )
        
        return max(ratio, self.final_teacher_forcing)
    
    def detect_repetitive_patterns(self, predictions):
        """
        Detect if model is generating repetitive patterns
        
        Args:
            predictions: List of predicted text strings
            
        Returns:
            bool: True if repetitive patterns detected
        """
        repetitive_count = 0
        
        for pred in predictions:
            if len(pred) > 6:
                # Check for simple repetitions (like "ាបាបាប")
                for i in range(2, min(6, len(pred) // 3)):
                    pattern = pred[:i]
                    if pred.startswith(pattern * (len(pred) // i)):
                        repetitive_count += 1
                        break
        
        repetition_ratio = repetitive_count / len(predictions)
        
        if repetition_ratio > 0.3:
            self.pattern_warnings += 1
            self.logger.warning(f"High repetition detected: {repetition_ratio:.1%} of predictions")
            return True
        
        return False
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with pattern monitoring"""
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_pattern_loss = 0.0
        total_eos_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self.global_step += 1
            
            # Move batch to device
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Current teacher forcing ratio
            teacher_forcing_ratio = self.get_current_teacher_forcing_ratio()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                images=images,
                target_sequences=targets,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            logits = output['logits']  # [B, T, vocab_size]
            
            # Main loss (NLL)
            batch_size, seq_len, vocab_size = logits.shape
            main_loss = 0.0
            
            for t in range(1, seq_len):  # Skip SOS
                target_t = targets[:, t]
                mask = target_t != self.config.vocab.PAD_IDX
                
                if mask.sum() > 0:
                    step_loss = self.criterion(logits[mask, t], target_t[mask])
                    main_loss += step_loss
            
            main_loss = main_loss / (seq_len - 1)
            
            # Anti-pattern loss
            pattern_loss = self.anti_pattern_loss(logits, targets)
            
            # EOS encouragement loss
            eos_loss = self.eos_loss(logits, targets, target_lengths)
            
            # Combined loss
            total_batch_loss = main_loss + 0.1 * pattern_loss + 0.05 * eos_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += total_batch_loss.item()
            total_main_loss += main_loss.item()
            total_pattern_loss += pattern_loss.item() if isinstance(pattern_loss, torch.Tensor) else pattern_loss
            total_eos_loss += eos_loss.item() if isinstance(eos_loss, torch.Tensor) else eos_loss
            batch_count += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {total_batch_loss.item():.4f} "
                    f"(Main: {main_loss.item():.4f}, Pattern: {pattern_loss:.4f}, EOS: {eos_loss:.4f}), "
                    f"TF: {teacher_forcing_ratio:.3f}"
                )
                
                # Pattern detection check
                if batch_idx % 100 == 0 and batch_idx > 0:
                    self.check_for_patterns(images[:2], targets[:2])
        
        return {
            'total_loss': total_loss / batch_count,
            'main_loss': total_main_loss / batch_count,
            'pattern_loss': total_pattern_loss / batch_count,
            'eos_loss': total_eos_loss / batch_count,
            'teacher_forcing': teacher_forcing_ratio
        }
    
    def check_for_patterns(self, sample_images, sample_targets):
        """Check current model predictions for patterns"""
        self.model.eval()
        
        with torch.no_grad():
            # Generate predictions
            result = self.model.generate(sample_images, max_length=50, method='greedy')
            sequences = result['sequences']
            
            predictions = []
            for seq in sequences:
                seq_list = seq.cpu().tolist()
                text = self.config.vocab.decode(seq_list)
                predictions.append(text)
            
            # Check for repetitive patterns
            if self.detect_repetitive_patterns(predictions):
                self.logger.warning("Repetitive patterns detected in current predictions:")
                for i, pred in enumerate(predictions):
                    target_text = self.config.vocab.decode(sample_targets[i].cpu().tolist())
                    self.logger.warning(f"  Sample {i}: '{pred[:50]}...' | Target: '{target_text[:50]}...'")
        
        self.model.train()
    
    def train(self, train_dataloader, val_dataloader, epochs=50):
        """Main training loop"""
        
        # OneCycleLR scheduler
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=2e-4,
            steps_per_epoch=len(train_dataloader),
            epochs=epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000
        )
        
        best_cer = float('inf')
        
        for epoch in range(epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"EPOCH {epoch+1}/{epochs} - PATTERN FIX TRAINING")
            self.logger.info(f"{'='*60}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_metrics = self.validator.validate(val_dataloader)
            
            # Update scheduler
            scheduler.step()
            
            # Log epoch results
            self.logger.info(f"\nEpoch {epoch+1} Results:")
            self.logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            self.logger.info(f"    Main: {train_metrics['main_loss']:.4f}")
            self.logger.info(f"    Pattern: {train_metrics['pattern_loss']:.4f}")
            self.logger.info(f"    EOS: {train_metrics['eos_loss']:.4f}")
            self.logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"  Val CER: {val_metrics['cer']*100:.2f}%")
            self.logger.info(f"  Teacher Forcing: {train_metrics['teacher_forcing']:.3f}")
            self.logger.info(f"  Pattern Warnings: {self.pattern_warnings}")
            
            # Save best model
            if val_metrics['cer'] < best_cer:
                best_cer = val_metrics['cer']
                self.logger.info(f"  NEW BEST CER: {best_cer*100:.2f}%")
                
                # Save checkpoint
                checkpoint_path = Path("models/checkpoints/pattern_fix_best.pth")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_cer': best_cer,
                    'config': self.config
                }, checkpoint_path)
            
            # Early intervention for excessive patterns
            if self.pattern_warnings > self.max_pattern_warnings:
                self.logger.warning(f"Too many pattern warnings ({self.pattern_warnings}). Adjusting training...")
                
                # Reset teacher forcing to higher value temporarily
                self.initial_teacher_forcing = min(0.8, self.initial_teacher_forcing + 0.1)
                self.pattern_warnings = 0
                
                # Reset optimizer with higher learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 1.2
                
                self.logger.info(f"  Increased teacher forcing to {self.initial_teacher_forcing}")
                self.logger.info(f"  Increased learning rate to {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping for very good results
            if val_metrics['cer'] <= 0.05:  # 5% CER
                self.logger.info(f"Excellent CER achieved: {val_metrics['cer']*100:.2f}%. Stopping early!")
                break
        
        self.logger.info(f"\nPattern-fix training completed!")
        self.logger.info(f"Best CER: {best_cer*100:.2f}%")
        self.logger.info(f"Final pattern warnings: {self.pattern_warnings}")


def main():
    parser = argparse.ArgumentParser(description="Pattern-Fix Training for Khmer OCR")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--samples-per-epoch', type=int, default=5000, help='Training samples per epoch')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('PatternFixMain')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Configuration
    config = ConfigManager()
    
    # Create datasets
    train_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config,
        corpus_dir="data/processed",
        samples_per_epoch=args.samples_per_epoch,
        augment_prob=0.8,
        shuffle_texts=True
    )
    
    # Try to use fixed validation set
    try:
        val_dataset = SyntheticImageDataset(
            split="val",
            synthetic_dir="data/validation_fixed",
            config_manager=config,
            max_samples=256
        )
        collate_fn_val = SyntheticCollateFunction(config.vocab)
        logger.info("Using fixed validation set")
    except:
        val_dataset = OnTheFlyDataset(
            split="val",
            config_manager=config,
            corpus_dir="data/processed",
            samples_per_epoch=256,
            augment_prob=0.0,
            shuffle_texts=False,
            random_seed=42
        )
        collate_fn_val = OnTheFlyCollateFunction(config.vocab)
        logger.info("Using on-the-fly validation set")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=OnTheFlyCollateFunction(config.vocab),
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_val,
        pin_memory=True
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples per epoch")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create trainer
    trainer = PatternFixTrainer(config, device)
    
    # Start training
    logger.info("Starting pattern-fix training...")
    trainer.train(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main() 