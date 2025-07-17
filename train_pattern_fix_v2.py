#!/usr/bin/env python3
"""
Pattern-Fix Training Script V2 for Khmer OCR - ISSUE FIXES

Critical fixes:
1. ✅ Creates validation data on-the-fly (no empty dataset)
2. ⚡ Optimized batch processing for faster training
3. 🎯 Enhanced pattern/EOS loss components
4. 📊 Better progress monitoring and debugging

Key Features:
- Anti-repetition loss with improved detection
- EOS token encouragement with better weighting  
- Progressive teacher forcing reduction (starts at 0.5)
- Real-time pattern detection and intervention
- Efficient data loading and processing
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
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config import ConfigManager
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset, OnTheFlyCollateFunction
from src.inference.metrics import OCRMetrics
import Levenshtein


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/pattern_fix_training.log')
        ]
    )
    return logging.getLogger('PatternFixV2')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Pattern-Fix Training V2')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--samples-per-epoch', type=int, default=2000, help='Samples per epoch')
    parser.add_argument('--val-samples', type=int, default=200, help='Validation samples')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--test-run', action='store_true', help='Quick test run')
    return parser.parse_args()


class PatternFixLoss(nn.Module):
    """Enhanced loss function to fix repetitive pattern generation"""
    
    def __init__(self, vocab_size, eos_token_id=2, pad_token_id=0):
        super().__init__()
        self.nll_loss = nn.NLLLoss(ignore_index=pad_token_id, reduction='mean')
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        
    def forward(self, predictions, targets, attention_weights=None):
        """
        Enhanced loss with pattern penalties and EOS encouragement
        
        Args:
            predictions: (batch_size, seq_len, vocab_size) - log probabilities
            targets: (batch_size, seq_len) - target token indices
            attention_weights: (batch_size, seq_len, input_len) - attention weights
        """
        batch_size, seq_len, vocab_size = predictions.shape
        
        # 1. Main NLL Loss
        main_loss = self.nll_loss(predictions.view(-1, vocab_size), targets.view(-1))
        
        # 2. Pattern Repetition Penalty (Enhanced)
        pattern_penalty = self._compute_pattern_penalty(predictions, targets)
        
        # 3. EOS Token Encouragement (Enhanced)
        eos_encouragement = self._compute_eos_encouragement(predictions, targets)
        
        # 4. Attention Diversity Loss (if available)
        attention_loss = self._compute_attention_diversity(attention_weights) if attention_weights is not None else 0
        
        # Combine losses with adaptive weights
        total_loss = (
            main_loss + 
            0.1 * pattern_penalty + 
            0.2 * eos_encouragement +
            0.05 * attention_loss
        )
        
        return {
            'total': total_loss,
            'main': main_loss,
            'pattern': pattern_penalty,
            'eos': eos_encouragement,
            'attention': attention_loss
        }
    
    def _compute_pattern_penalty(self, predictions, targets):
        """Compute penalty for repetitive patterns"""
        batch_size, seq_len, _ = predictions.shape
        
        # Get predicted tokens (greedy decoding for pattern detection)
        pred_tokens = torch.argmax(predictions, dim=-1)  # (batch_size, seq_len)
        
        penalty = 0.0
        
        for batch_idx in range(batch_size):
            tokens = pred_tokens[batch_idx]
            
            # Detect 2-gram, 3-gram, and 4-gram repetitions
            for ngram_size in [2, 3, 4]:
                penalty += self._detect_ngram_repetition(tokens, ngram_size)
        
        return penalty / batch_size
    
    def _detect_ngram_repetition(self, tokens, ngram_size):
        """Detect repetitive n-grams in a sequence"""
        if len(tokens) < ngram_size * 3:  # Need at least 3 repetitions to detect
            return 0.0
        
        ngram_counts = defaultdict(int)
        penalty = 0.0
        
        # Count n-grams
        for i in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[i:i+ngram_size].cpu().tolist())
            ngram_counts[ngram] += 1
        
        # Penalize high-frequency n-grams
        for ngram, count in ngram_counts.items():
            if count >= 3:  # 3+ repetitions
                # Stronger penalty for longer repetitions
                repetition_penalty = (count - 2) * 0.1 * ngram_size
                penalty += repetition_penalty
        
        return penalty
    
    def _compute_eos_encouragement(self, predictions, targets):
        """Encourage EOS token generation at appropriate positions"""
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Find where targets have EOS tokens
        target_eos_mask = (targets == self.eos_token_id).float()
        
        if target_eos_mask.sum() == 0:  # No EOS in targets
            return torch.tensor(0.0, device=predictions.device)
        
        # Get EOS probabilities from predictions
        eos_probs = torch.softmax(predictions, dim=-1)[:, :, self.eos_token_id]
        
        # Encourage EOS where it should be
        eos_loss = -torch.mean(target_eos_mask * torch.log(eos_probs + 1e-8))
        
        # Additional penalty for sequences that don't generate EOS at all
        pred_tokens = torch.argmax(predictions, dim=-1)
        no_eos_penalty = 0.0
        
        for batch_idx in range(batch_size):
            has_eos_target = (targets[batch_idx] == self.eos_token_id).any()
            has_eos_pred = (pred_tokens[batch_idx] == self.eos_token_id).any()
            
            if has_eos_target and not has_eos_pred:
                no_eos_penalty += 1.0
        
        no_eos_penalty = no_eos_penalty / batch_size
        
        return eos_loss + 0.5 * no_eos_penalty
    
    def _compute_attention_diversity(self, attention_weights):
        """Encourage diverse attention patterns"""
        if attention_weights is None:
            return 0.0
        
        # Compute attention entropy (higher is more diverse)
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        diversity_loss = -torch.mean(attention_entropy)  # Negative because we want high entropy
        
        return diversity_loss * 0.1


class PatternFixTrainerV2:
    """Enhanced trainer specifically for fixing pattern generation issues"""
    
    def __init__(self, model, train_dataloader, val_dataloader, config, device, logger):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Enhanced loss function
        self.criterion = PatternFixLoss(
            vocab_size=config.data_config.vocab_size,
            eos_token_id=config.vocab.EOS_IDX,
            pad_token_id=config.vocab.PAD_IDX
        ).to(device)
        
        # Metrics
        self.metrics = OCRMetrics(config.vocab)
        
        # Pattern monitoring
        self.pattern_stats = defaultdict(int)
    
    def train_epoch(self, epoch, optimizer, scheduler, teacher_forcing_ratio):
        """Train one epoch with pattern monitoring"""
        self.model.train()
        total_losses = defaultdict(float)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = self.model(
                images, 
                target_sequences=targets,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # Compute enhanced loss
            loss_dict = self.criterion(
                outputs['logits'], 
                targets,
                attention_weights=outputs.get('attention_weights')
            )
            
            # Backward pass
            loss_dict['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Log losses
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    total_losses[key] += value.item()
                else:
                    total_losses[key] += value
            
            # Monitor patterns (every 10 batches)
            if batch_idx % 10 == 0:
                self._monitor_patterns(outputs['logits'], targets, batch_idx)
            
            # Progress logging
            if batch_idx % 5 == 0:
                elapsed = time.time() - start_time
                samples_processed = (batch_idx + 1) * len(images)
                samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
                
                def format_loss_value(value):
                    """Format loss value, handling both tensors and floats"""
                    if isinstance(value, torch.Tensor):
                        return value.item()
                    return value
                
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, "
                    f"Loss: {format_loss_value(loss_dict['total']):.4f} "
                    f"(Main: {format_loss_value(loss_dict['main']):.4f}, "
                    f"Pattern: {format_loss_value(loss_dict['pattern']):.4f}, "
                    f"EOS: {format_loss_value(loss_dict['eos']):.4f}), "
                    f"TF: {teacher_forcing_ratio:.3f}, "
                    f"Speed: {samples_per_sec:.1f} samples/sec"
                )
        
        # Return average losses
        avg_losses = {key: value / len(self.train_dataloader) for key, value in total_losses.items()}
        return avg_losses
    
    def validate(self, epoch):
        """Validate model with detailed pattern analysis"""
        self.model.eval()
        total_losses = defaultdict(float)
        predictions_all = []
        targets_all = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(images, target_sequences=targets, teacher_forcing_ratio=0.0)
                
                loss_dict = self.criterion(
                    outputs['logits'], 
                    targets,
                    attention_weights=outputs.get('attention_weights')
                )
                
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        total_losses[key] += value.item()
                    else:
                        total_losses[key] += value
                
                # Collect predictions for metrics
                pred_tokens = torch.argmax(outputs['logits'], dim=-1)
                predictions_all.extend(pred_tokens.cpu().numpy())
                targets_all.extend(targets.cpu().numpy())
        
        # Compute metrics
        avg_losses = {key: value / len(self.val_dataloader) for key, value in total_losses.items()}
        
        # Detailed pattern analysis
        pattern_analysis = self._analyze_validation_patterns(predictions_all, targets_all)
        
        return avg_losses, pattern_analysis
    
    def _monitor_patterns(self, predictions, targets, batch_idx):
        """Monitor pattern generation in real-time"""
        pred_tokens = torch.argmax(predictions, dim=-1)
        
        for seq in pred_tokens:
            seq_list = seq.cpu().tolist()
            
            # Detect common repetitive patterns
            if self._has_repetitive_pattern(seq_list):
                self.pattern_stats['repetitive_sequences'] += 1
            
            # Check for EOS generation
            if 2 in seq_list:  # EOS token ID
                self.pattern_stats['sequences_with_eos'] += 1
            else:
                self.pattern_stats['sequences_without_eos'] += 1
    
    def _has_repetitive_pattern(self, seq, min_length=3, min_repetitions=3):
        """Check if sequence has repetitive patterns"""
        for pattern_len in range(2, min_length + 1):
            for i in range(len(seq) - pattern_len * min_repetitions + 1):
                pattern = seq[i:i + pattern_len]
                count = 1
                
                # Count consecutive repetitions
                for j in range(i + pattern_len, len(seq) - pattern_len + 1, pattern_len):
                    if seq[j:j + pattern_len] == pattern:
                        count += 1
                    else:
                        break
                
                if count >= min_repetitions:
                    return True
        
        return False
    
    def _analyze_validation_patterns(self, predictions, targets):
        """Analyze validation patterns for detailed reporting"""
        analysis = {
            'total_sequences': len(predictions),
            'repetitive_patterns': 0,
            'eos_generated': 0,
            'average_length_pred': 0,
            'average_length_target': 0,
            'pattern_examples': []
        }
        
        total_pred_length = 0
        total_target_length = 0
        
        for pred, target in zip(predictions, targets):
            # Convert to lists
            pred_list = pred.tolist() if hasattr(pred, 'tolist') else list(pred)
            target_list = target.tolist() if hasattr(target, 'tolist') else list(target)
            
            # Remove padding
            pred_clean = [t for t in pred_list if t != 0]
            target_clean = [t for t in target_list if t != 0]
            
            total_pred_length += len(pred_clean)
            total_target_length += len(target_clean)
            
            # Check for repetitive patterns
            if self._has_repetitive_pattern(pred_clean):
                analysis['repetitive_patterns'] += 1
                if len(analysis['pattern_examples']) < 5:
                    analysis['pattern_examples'].append(pred_clean[:20])  # First 20 tokens
            
            # Check for EOS generation
            if 2 in pred_clean:
                analysis['eos_generated'] += 1
        
        analysis['average_length_pred'] = total_pred_length / len(predictions) if predictions else 0
        analysis['average_length_target'] = total_target_length / len(targets) if targets else 0
        
        return analysis


def main():
    """Main training function with all fixes"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info(">> Starting Pattern-Fix Training V2...")
    logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = ConfigManager()
    
    # Create datasets with optimized parameters
    logger.info(">> Creating optimized datasets...")
    
    # Training dataset
    train_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config,
        corpus_dir="data/processed",
        samples_per_epoch=args.samples_per_epoch,
        augment_prob=0.3,  # Reduced for faster processing
        shuffle_texts=True,
        random_seed=None  # Different each epoch
    )
    
    # Validation dataset (on-the-fly generation)
    val_dataset = OnTheFlyDataset(
        split="val",
        config_manager=config,
        corpus_dir="data/processed",
        samples_per_epoch=args.val_samples,
        augment_prob=0.0,  # No augmentation for validation
        shuffle_texts=False,
        random_seed=42  # Fixed seed for reproducible validation
    )
    
    logger.info(f"[OK] Training dataset: {len(train_dataset)} samples per epoch")
    logger.info(f"[OK] Validation dataset: {len(val_dataset)} samples")
    
    # Optimized data loaders
    collate_fn = OnTheFlyCollateFunction(config.vocab)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Optimized for speed
        pin_memory=True,
        persistent_workers=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model setup
    logger.info(">> Loading model...")
    model = KhmerOCRSeq2Seq(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Optimizer with enhanced settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Enhanced scheduler
    total_steps = args.epochs * len(train_dataloader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Initialize trainer
    trainer = PatternFixTrainerV2(model, train_dataloader, val_dataloader, config, device, logger)
    
    # Training loop with adaptive teacher forcing
    logger.info(">> Starting pattern-fix training...")
    
    best_pattern_score = float('inf')
    
    for epoch in range(args.epochs):
        # Progressive teacher forcing reduction (start at 0.5, end at 0.1)
        teacher_forcing_ratio = 0.5 - (0.4 * epoch / args.epochs)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch+1}/{args.epochs} - PATTERN FIX TRAINING V2")
        logger.info(f"Teacher Forcing: {teacher_forcing_ratio:.3f}")
        logger.info(f"{'='*60}")
        
        # Train
        train_losses = trainer.train_epoch(epoch, optimizer, scheduler, teacher_forcing_ratio)
        
        # Validate
        val_losses, pattern_analysis = trainer.validate(epoch)
        
        # Enhanced logging
        logger.info(f"\n[RESULTS] EPOCH {epoch+1} RESULTS:")
        logger.info(f"  Train Loss: {train_losses['total']:.4f} (Main: {train_losses['main']:.4f}, Pattern: {train_losses['pattern']:.4f})")
        logger.info(f"  Val Loss: {val_losses['total']:.4f}")
        logger.info(f"  Pattern Analysis:")
        logger.info(f"    Repetitive patterns: {pattern_analysis['repetitive_patterns']}/{pattern_analysis['total_sequences']} ({100*pattern_analysis['repetitive_patterns']/pattern_analysis['total_sequences']:.1f}%)")
        logger.info(f"    EOS generated: {pattern_analysis['eos_generated']}/{pattern_analysis['total_sequences']} ({100*pattern_analysis['eos_generated']/pattern_analysis['total_sequences']:.1f}%)")
        logger.info(f"    Avg length - Pred: {pattern_analysis['average_length_pred']:.1f}, Target: {pattern_analysis['average_length_target']:.1f}")
        
        # Pattern score for model selection
        pattern_score = pattern_analysis['repetitive_patterns'] + (pattern_analysis['total_sequences'] - pattern_analysis['eos_generated'])
        
        if pattern_score < best_pattern_score:
            best_pattern_score = pattern_score
            logger.info(f"  [BEST] New best pattern score: {pattern_score}")
            # Save best model
            torch.save(model.state_dict(), 'models/checkpoints/pattern_fix_best.pth')
        
        # Early stopping if patterns are well controlled
        if pattern_analysis['repetitive_patterns'] == 0 and pattern_analysis['eos_generated'] > pattern_analysis['total_sequences'] * 0.8:
            logger.info("[SUCCESS] Pattern generation fixed! Early stopping.")
            break
    
    logger.info("[COMPLETE] Pattern-fix training completed!")


if __name__ == "__main__":
    main() 