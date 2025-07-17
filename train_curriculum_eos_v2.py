#!/usr/bin/env python3
"""
Curriculum Learning Trainer for EOS Generation Fix

Implements progressive sequence length training to force the model to learn
proper sequence termination, starting with short sequences and gradually increasing.
"""

import os
import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time
from pathlib import Path
import logging

# Project imports
from src.utils.config import ConfigManager
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset
from src.training.checkpoint_manager import CheckpointManager

# Configure logging with file output
def setup_logging(log_dir="logs/curriculum_eos_v2"):
    """Setup logging with both file and console output"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"curriculum_training_{timestamp}.log")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("CurriculumTrainer")
    logger.info(f"📝 Training log: {log_file}")
    return logger, log_file

logger = logging.getLogger("CurriculumTrainer")

def curriculum_collate_fn(batch):
    """Custom collate function for variable-width images and sequences."""
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['targets'])
    
    # Pad images to max width in batch
    max_width = max(img.shape[-1] for img in images)
    padded_images = []
    
    for img in images:
        if len(img.shape) == 3:  # (C, H, W)
            c, h, w = img.shape
            padded = torch.zeros(c, h, max_width, dtype=img.dtype, device=img.device)
            padded[:, :, :w] = img
        else:  # (H, W) or other
            h, w = img.shape[-2:]
            padded = torch.zeros(*img.shape[:-1], max_width, dtype=img.dtype, device=img.device)
            padded[..., :w] = img
        padded_images.append(padded)
    
    # Stack images
    images_tensor = torch.stack(padded_images, dim=0)
    
    # Pad target sequences
    targets_tensor = pad_sequence(targets, batch_first=True, padding_value=0)  # 0 is PAD_IDX
    
    return {
        'image': images_tensor,
        'targets': targets_tensor
    }

class CurriculumDataset:
    """Wrapper for OnTheFlyDataset that enforces max sequence length using syllable-based truncation."""
    
    def __init__(self, base_dataset, max_length, config_manager):
        self.base_dataset = base_dataset
        self.max_length = max_length
        self.config_manager = config_manager
        self.vocab = config_manager.vocab
        
        # Import syllable segmentation function
        from src.khtext.subword_cluster import split_syllables_advanced
        self.split_syllables = self._split_syllables_preserve_spaces
        
        # Cache only text strings to avoid OnTheFlyDataset randomness (much faster than full samples)
        print(f"🔄 Caching text strings for {len(base_dataset)} samples...")
        self.cached_texts = []
        
        # Access the text_lines directly from OnTheFlyDataset to avoid slow __getitem__ calls
        if hasattr(base_dataset, 'text_lines'):
            # For datasets smaller than text_lines, cycle through them deterministically
            for i in range(len(base_dataset)):
                text_idx = i % len(base_dataset.text_lines)
                self.cached_texts.append(base_dataset.text_lines[text_idx])
        else:
            # Fallback: this might be slow for other dataset types
            print("⚠️  Warning: Base dataset doesn't have text_lines, falling back to slower method")
            for i in range(len(base_dataset)):
                _, _, text = base_dataset[i]
                self.cached_texts.append(text)
        
        print(f"✅ Text caching complete! {len(self.cached_texts)} texts cached.")
    
    def _split_syllables_preserve_spaces(self, text):
        """
        Split syllables while preserving actual space characters instead of converting to <SPACE> tags.
        This is needed for curriculum learning to maintain proper vocabulary encoding.
        """
        from src.khtext.subword_cluster import split_syllables_advanced
        
        # Get syllables using the original function
        syllables = split_syllables_advanced(text)
        
        # Convert <SPACE> tags back to actual space characters
        preserved_syllables = []
        for syllable in syllables:
            if syllable == '<SPACE>':
                preserved_syllables.append(' ')  # Convert back to actual space
            elif syllable == '<SPACES>':
                preserved_syllables.append('  ')  # Convert back to multiple spaces
            elif syllable == '<TAB>':
                preserved_syllables.append('\t')  # Convert back to tab
            elif syllable == '<NEWLINE>':
                preserved_syllables.append('\n')  # Convert back to newline
            elif syllable == '<CRLF>':
                preserved_syllables.append('\r\n')  # Convert back to CRLF
            else:
                preserved_syllables.append(syllable)  # Keep as-is
        
        return preserved_syllables
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get cached text to ensure consistency
        original_text = self.cached_texts[idx]
        
        # Split into syllables for intelligent truncation
        syllables = self.split_syllables(original_text)
        
        # Calculate current length with SOS/EOS tokens (reserve 2 tokens for SOS+EOS)
        max_content_tokens = self.max_length - 2  # Reserve 2 tokens for SOS and EOS
        current_length = 0  # Start counting content tokens only
        selected_syllables = []
        
        for syllable in syllables:
            syllable_tokens = len(self.vocab.encode(syllable))
            if current_length + syllable_tokens <= max_content_tokens:
                selected_syllables.append(syllable)
                current_length += syllable_tokens
            else:
                break
        
        # Join selected syllables
        truncated_text = ''.join(selected_syllables)
        
        # Encode the truncated text with SOS/EOS
        target_indices = [self.vocab.SOS_IDX] + self.vocab.encode(truncated_text) + [self.vocab.EOS_IDX]
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        
        # Verify final length doesn't exceed max_length (safety check)
        if len(target_indices) > self.max_length:
            # Emergency truncation if still too long (should not happen with correct logic)
            truncated_indices = target_indices[:self.max_length-1] + [self.vocab.EOS_IDX]
            target_tensor = torch.tensor(truncated_indices, dtype=torch.long)
            # Decode back to get the actual truncated text
            truncated_text = self.vocab.decode(truncated_indices[1:-1])  # Remove SOS/EOS for decoding
        
        # Generate synthetic image for truncated text using the base dataset's text generator
        try:
            if hasattr(self.base_dataset, 'text_generator'):
                # Use the same text generator as the base dataset
                text_generator = self.base_dataset.text_generator
                
                # Generate image for the truncated text with minimal augmentation for curriculum learning
                from PIL import ImageFont, Image
                import torchvision.transforms as transforms
                
                # Select appropriate font
                font_path = text_generator._select_font(truncated_text, self.base_dataset.split)
                temp_font = ImageFont.truetype(font_path, text_generator.font_size)
                image_width = text_generator._calculate_optimal_width(truncated_text, temp_font)
                
                # Render the text
                pil_image = text_generator._render_text_image(truncated_text, font_path, image_width)
                
                # Apply light augmentation (reduce for curriculum learning clarity)
                if self.base_dataset.use_augmentation and text_generator.augmentor:
                    # Temporarily reduce augmentation
                    original_aug_prob = self.base_dataset.augment_prob
                    self.base_dataset.augment_prob = 0  # disable augmentation
                    pil_image = text_generator._apply_augmentation(pil_image)
                    self.base_dataset.augment_prob = original_aug_prob
                
                # Convert to tensor
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                image = transform(pil_image)
                
            else:
                # Fallback: create a dummy image 
                image = torch.zeros((3, 64, 256))
                
        except Exception as e:
            # Fallback on any error
            image = torch.zeros((3, 64, 256))
        
        return {
            'image': image,
            'targets': target_tensor,
            'text': truncated_text
        }

class CurriculumEOSLoss(nn.Module):
    """Loss function with curriculum-aware EOS weighting."""
    
    def __init__(self, vocab_size, pad_idx, eos_idx, eos_weight=10.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.eos_weight = eos_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')
    
    def forward(self, predictions, targets, current_max_length):
        """
        Compute curriculum-aware loss with strong EOS weighting for short sequences.
        
        Args:
            predictions: Model predictions [batch, seq_len, vocab_size]
            targets: Target sequences [batch, seq_len]
            current_max_length: Current curriculum max length
        """
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Reshape for loss computation
        predictions_flat = predictions.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Compute base cross-entropy loss
        losses = self.ce_loss(predictions_flat, targets_flat)
        losses = losses.view(batch_size, seq_len)
        
        # Create EOS mask
        eos_mask = (targets_flat == self.eos_idx).view(batch_size, seq_len)
        
        # Adaptive EOS weighting based on curriculum stage
        # Shorter sequences get higher EOS weight to force learning
        adaptive_eos_weight = self.eos_weight * (20.0 / max(current_max_length, 10.0))
        
        # Apply EOS weighting
        eos_weighted_losses = torch.where(
            eos_mask,
            losses * adaptive_eos_weight,
            losses
        )
        
        # Compute sequence-level loss (ignore padding)
        mask = (targets != self.pad_idx).float()
        sequence_losses = (eos_weighted_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        total_loss = sequence_losses.mean()
        
        # Compute metrics
        with torch.no_grad():
            # EOS prediction accuracy
            predictions_tokens = torch.argmax(predictions_flat, dim=-1)
            eos_predicted = (predictions_tokens == self.eos_idx).view(batch_size, seq_len)
            eos_targets = eos_mask
            
            eos_correct = (eos_predicted & eos_targets).sum().float()
            eos_total = eos_targets.sum().float()
            eos_accuracy = eos_correct / eos_total.clamp(min=1) * 100
            
            # Sequence completion rate (ends with EOS)
            last_predictions = predictions_tokens.view(batch_size, seq_len)[:, -1]
            completion_rate = (last_predictions == self.eos_idx).float().mean() * 100
        
        return {
            'total_loss': total_loss,
            'eos_accuracy': eos_accuracy.item(),
            'completion_rate': completion_rate.item(),
            'adaptive_weight': adaptive_eos_weight
        }

class CurriculumTrainer:
    """Curriculum learning trainer for EOS generation."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Curriculum schedule
        self.curriculum_schedule = [
            # (epochs, max_length, description)
            (4, 10, "Phase 1: Short sequences"),
            (4, 15, "Phase 2: Slightly longer"),  
            (4, 20, "Phase 3: Medium sequences"),
            (8, 30, "Phase 4: Extended sequences"),  # Extended from 4 to 8 epochs
            (16, 50, "Phase 5: Full sequences")
        ]
        
        # Initialize model
        self.model = KhmerOCRSeq2Seq(config_manager)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Initialize base dataset
        self.base_train_dataset = OnTheFlyDataset(config_manager=config_manager)
        self.base_val_dataset = OnTheFlyDataset(config_manager=config_manager)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir="models/checkpoints/curriculum_eos_v2",
            max_checkpoints=5  # Keep only latest 5 checkpoints
        )
        
        # Initialize loss function
        vocab = config_manager.vocab
        self.criterion = CurriculumEOSLoss(
            vocab_size=len(vocab),
            pad_idx=vocab.PAD_IDX,
            eos_idx=vocab.EOS_IDX,
            eos_weight=15.0
        )
        
        # Training state
        self.current_epoch = 0
        self.best_eos_rate = 0.0
        
        # Curriculum phase tracking
        self.current_phase = 0  # 0-based index into curriculum_schedule
        self.phase_epoch = 0   # Current epoch within the current phase
        self.total_epochs = sum(phase[0] for phase in self.curriculum_schedule)  # Total epochs across all phases
    
    def create_curriculum_dataset(self, max_length, is_training=True):
        """Create dataset wrapper with curriculum max length."""
        base_dataset = self.base_train_dataset if is_training else self.base_val_dataset
        return CurriculumDataset(base_dataset, max_length, self.config_manager)
    
    def train_phase(self, max_length, epochs, phase_description):
        """Train for a specific curriculum phase."""
        print(f"\n{'='*60}")
        print(f"{phase_description} (max_length={max_length})")
        print(f"{'='*60}")
        
        # Create curriculum datasets
        train_dataset = self.create_curriculum_dataset(max_length, is_training=True)
        val_dataset = self.create_curriculum_dataset(max_length, is_training=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32,  # Larger batch for curriculum 
            shuffle=True,
            collate_fn=curriculum_collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=curriculum_collate_fn
        )
        
        phase_best_eos = 0.0
        
        for epoch in range(epochs):
            self.current_epoch += 1
            self.phase_epoch += 1  # Track epoch within current phase
            
            # Print epoch announcement before starting
            logger.info(f"🔄 Starting Epoch {self.current_epoch}/{sum(s[0] for s in self.curriculum_schedule)} (Phase {self.current_phase + 1}: {phase_description})")
            logger.info(f"📊 Max Length: {max_length} | Adaptive EOS Weight: {15.0 * (20.0 / max(max_length, 10.0)):.2f}")
            print(f"\n🔄 Starting Epoch {self.current_epoch}/{sum(s[0] for s in self.curriculum_schedule)} (Phase {self.current_phase + 1}: {phase_description})")
            print(f"📊 Max Length: {max_length} | Adaptive EOS Weight: {15.0 * (20.0 / max(max_length, 10.0)):.2f}")
            
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, max_length)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, max_length)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics (both console and file)
            epoch_summary = f"Epoch {self.current_epoch}/{sum(s[0] for s in self.curriculum_schedule)}"
            time_info = f"Time: {epoch_time:.1f}s | Max Length: {max_length}"
            train_info = f"Train - Loss: {train_metrics['loss']:.4f} | EOS: {train_metrics['eos_acc']:.1f}% | Complete: {train_metrics['complete']:.1f}%"
            val_info = f"Val   - Loss: {val_metrics['loss']:.4f} | EOS: {val_metrics['eos_acc']:.1f}% | Complete: {val_metrics['complete']:.1f}%"
            weight_info = f"Adaptive EOS Weight: {train_metrics['weight']:.2f}"
            
            print(f"\n{epoch_summary}")
            print(time_info)
            print(train_info)
            print(val_info)
            print(weight_info)
            
            # Log to file
            logger.info(f"{epoch_summary} | {time_info}")
            logger.info(f"{train_info}")
            logger.info(f"{val_info}")
            logger.info(f"{weight_info}")
            
            # Track best EOS rate for this phase
            current_eos_rate = val_metrics['eos_acc']
            if current_eos_rate > phase_best_eos:
                phase_best_eos = current_eos_rate
            
            # Check if this is a new global best
            is_best = current_eos_rate > self.best_eos_rate
            if is_best:
                self.best_eos_rate = current_eos_rate
                best_msg = f"✅ New best EOS rate: {current_eos_rate:.1f}%"
                print(best_msg)
                logger.info(best_msg)
            
            # Create checkpoint data (save every epoch)
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'loss': val_metrics['loss'],
                'eos_rate': current_eos_rate,
                'max_length': max_length,
                'config': self.config_manager.to_dict(),  # Save as dictionary for compatibility
                'curriculum_phase': f"max_length_{max_length}",
                'phase_best_eos': phase_best_eos,
                'global_best_eos': self.best_eos_rate,
                # Add curriculum phase tracking
                'current_phase': self.current_phase,
                'phase_epoch': self.phase_epoch,
                'curriculum_schedule': self.curriculum_schedule
            }
            
            # Save checkpoint every epoch
            self.checkpoint_manager.save_checkpoint(
                checkpoint_data=checkpoint_data,
                epoch=self.current_epoch,
                is_best=is_best
            )
            checkpoint_msg = f"💾 Checkpoint saved: epoch_{self.current_epoch:03d}.pth"
            print(checkpoint_msg)
            logger.info(checkpoint_msg)
        
        # Reset phase epoch counter for next phase
        self.phase_epoch = 0
        
        print(f"\nPhase completed! Best EOS rate: {phase_best_eos:.1f}%")
        return phase_best_eos
    
    def _calculate_phase_from_epoch(self):
        """Calculate current phase and phase epoch from total epoch number (for backward compatibility)."""
        total_epochs_completed = self.current_epoch
        epochs_processed = 0
        
        # Find which phase we're in based on completed epochs
        for phase_idx, (phase_epochs, _, _) in enumerate(self.curriculum_schedule):
            if total_epochs_completed <= epochs_processed + phase_epochs:
                # We're in this phase
                self.current_phase = phase_idx
                self.phase_epoch = total_epochs_completed - epochs_processed
                return
            epochs_processed += phase_epochs
        
        # If we've gone beyond all phases, set to last phase as completed
        self.current_phase = len(self.curriculum_schedule) - 1
        last_phase_epochs = self.curriculum_schedule[-1][0]
        self.phase_epoch = last_phase_epochs
    
    def train_epoch(self, train_loader, max_length):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_eos_acc = 0.0
        total_complete = 0.0
        total_weight = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch['image'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, targets, teacher_forcing_ratio=0.9)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                predictions = outputs.get('predictions', outputs.get('logits', outputs.get('output', outputs)))
                if predictions is None:
                    # If still None, try to find any tensor in the dict
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 3:
                            predictions = value
                            break
                    if predictions is None:
                        raise ValueError(f"Could not find predictions in model output: {list(outputs.keys())}")
            else:
                predictions = outputs
            
            # Compute curriculum loss
            loss_dict = self.criterion(predictions, targets, max_length)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total_loss'].item()
            total_eos_acc += loss_dict['eos_accuracy']
            total_complete += loss_dict['completion_rate']
            total_weight += loss_dict['adaptive_weight']
            num_batches += 1
            
            # Progress update
            if batch_idx % 50 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss_dict['total_loss'].item():.4f} | EOS: {loss_dict['eos_accuracy']:.1f}%")
                # print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss_dict['total_loss'].item():.4f} | EOS: {loss_dict['eos_accuracy']:.1f}%")
        
        return {
            'loss': total_loss / num_batches,
            'eos_acc': total_eos_acc / num_batches,
            'complete': total_complete / num_batches,
            'weight': total_weight / num_batches
        }
    
    def validate_epoch(self, val_loader, max_length):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_eos_acc = 0.0
        total_complete = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                images = batch['image'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, targets, teacher_forcing_ratio=0.0)  # No teacher forcing for validation
                
                if isinstance(outputs, dict):
                    predictions = outputs.get('predictions', outputs.get('logits', outputs.get('output', outputs)))
                    if predictions is None:
                        # If still None, try to find any tensor in the dict
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor) and value.dim() == 3:
                                predictions = value
                                break
                        if predictions is None:
                            raise ValueError(f"Could not find predictions in model output: {list(outputs.keys())}")
                else:
                    predictions = outputs
                
                # Compute loss
                loss_dict = self.criterion(predictions, targets, max_length)
                
                # Accumulate metrics
                total_loss += loss_dict['total_loss'].item()
                total_eos_acc += loss_dict['eos_accuracy']
                total_complete += loss_dict['completion_rate']
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'eos_acc': total_eos_acc / num_batches,
            'complete': total_complete / num_batches
        }
    
    def run_curriculum_training(self):
        """Run complete curriculum training."""
        header = "🎓 CURRICULUM LEARNING FOR EOS GENERATION"
        separator = "="*80
        model_params = f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        device_info = f"Device: {self.device}"
        
        print(header)
        print(separator)
        print(model_params)
        print(device_info)
        
        # Log to file
        logger.info(header)
        logger.info(model_params)
        logger.info(device_info)
        
        total_phases = len(self.curriculum_schedule)
        phase_results = []
        
        try:
            # Resume from correct phase if continuing training
            for phase_idx in range(self.current_phase, total_phases):
                self.current_phase = phase_idx
                epochs, max_length, description = self.curriculum_schedule[phase_idx]
                
                phase_msg = f"🔄 Phase {phase_idx + 1}/{total_phases}"
                print(f"\n{phase_msg}")
                logger.info(phase_msg)
                
                # Adjust epochs if resuming mid-phase
                remaining_epochs = epochs
                if phase_idx == self.current_phase and self.phase_epoch > 0:
                    remaining_epochs = epochs - self.phase_epoch
                    resume_msg = f"🔄 Resuming Phase {phase_idx + 1} from epoch {self.phase_epoch + 1}/{epochs}"
                    print(resume_msg)
                    logger.info(resume_msg)
                elif phase_idx == self.current_phase and self.phase_epoch == 0:
                    # Starting this phase fresh
                    start_msg = f"🔄 Starting Phase {phase_idx + 1}"
                    print(start_msg)
                    logger.info(start_msg)
                
                if remaining_epochs > 0:
                    phase_eos_rate = self.train_phase(max_length, remaining_epochs, description)
                    phase_results.append({
                        'phase': phase_idx + 1,
                        'max_length': max_length,
                        'epochs': epochs,
                        'best_eos_rate': phase_eos_rate,
                        'description': description
                    })
                    
                    # Early success check
                    if phase_eos_rate > 80.0 and max_length >= 20:
                        early_success_msg = f"🎉 Early success! EOS rate {phase_eos_rate:.1f}% at length {max_length}"
                        print(early_success_msg)
                        logger.info(early_success_msg)
                        break
                
                # Reset phase epoch counter for next phase
                self.phase_epoch = 0
                    
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        
        # Final summary
        print("\n" + "="*80)
        print("📊 CURRICULUM TRAINING SUMMARY")
        print("="*80)
        for result in phase_results:
            print(f"Phase {result['phase']}: {result['description']}")
            print(f"  Max Length: {result['max_length']} | EOS Rate: {result['best_eos_rate']:.1f}%")
        
        best_rate_msg = f"🏆 Overall Best EOS Rate: {self.best_eos_rate:.1f}%"
        checkpoint_info = f"📁 Checkpoints saved to: models/checkpoints/curriculum_eos_v2"
        
        print(f"\n{best_rate_msg}")
        print(checkpoint_info)
        
        # Log final results
        logger.info("="*80)
        logger.info("CURRICULUM TRAINING COMPLETED")
        logger.info(best_rate_msg)
        logger.info(checkpoint_info)
        if self.best_eos_rate > 50.0:
            logger.info("✅ Training SUCCESSFUL! EOS generation target achieved.")
        else:
            logger.info("⚠️ Training completed but EOS target not fully achieved.")
        logger.info("="*80)
        
        return self.best_eos_rate > 50.0  # Success threshold

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Curriculum EOS Training for Khmer OCR")
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from (e.g., models/checkpoints/curriculum_eos_v2/best_model.pth)"
    )
    parser.add_argument(
        "--auto-resume", 
        action="store_true",
        help="Automatically resume from the latest checkpoint if available"
    )
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=22,
        help="Number of training epochs (default: 22 for full curriculum)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs/curriculum_eos_v2",
        help="Directory for training logs"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="models/checkpoints/curriculum_eos_v2",
        help="Directory for saving checkpoints"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config_manager = ConfigManager()
    
    # Setup logging with file output
    logger, log_file = setup_logging(args.log_dir)
    logger.info("🎓 Curriculum EOS Training Started")
    logger.info(f"Arguments: {vars(args)}")
    
    # Update configuration with command line arguments
    config_manager.training_config.epochs = args.num_epochs
    config_manager.training_config.batch_size = args.batch_size
    
    # Create curriculum trainer
    trainer = CurriculumTrainer(config_manager)
    
    # Update checkpoint directory if specified
    if args.checkpoint_dir != "models/checkpoints/curriculum_eos_v2":
        trainer.checkpoint_manager.checkpoint_dir = Path(args.checkpoint_dir)
    
    # Handle resume functionality
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = args.resume
        print(f"🔄 Resuming training from: {resume_checkpoint}")
    elif args.auto_resume:
        # Try to find latest checkpoint
        latest_checkpoint = trainer.checkpoint_manager.load_latest_checkpoint()
        if latest_checkpoint:
            resume_checkpoint = "auto"
            print(f"🔄 Auto-resuming from latest checkpoint")
        else:
            print("📁 No checkpoints found for auto-resume, starting fresh training")
    
    # Load checkpoint if specified
    if resume_checkpoint and resume_checkpoint != "auto":
        try:
            checkpoint_data = trainer.checkpoint_manager.load_checkpoint(resume_checkpoint)
            trainer.model.load_state_dict(checkpoint_data['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            trainer.current_epoch = checkpoint_data.get('epoch', 0)
            trainer.best_eos_rate = checkpoint_data.get('eos_rate', 0.0)
            
            # Restore curriculum phase progress
            trainer.current_phase = checkpoint_data.get('current_phase', 0)
            trainer.phase_epoch = checkpoint_data.get('phase_epoch', 0)
            
            # Calculate which phase we're in based on epoch if phase info missing
            if 'current_phase' not in checkpoint_data:
                trainer._calculate_phase_from_epoch()
            
            phase_name = trainer.curriculum_schedule[trainer.current_phase][2] if trainer.current_phase < len(trainer.curriculum_schedule) else "Completed"
            print(f"✅ Resumed from epoch {trainer.current_epoch}, best EOS rate: {trainer.best_eos_rate:.1f}%")
            print(f"📍 Current phase: {trainer.current_phase + 1}/5 - {phase_name}")
            print(f"📍 Phase progress: {trainer.phase_epoch}/{trainer.curriculum_schedule[trainer.current_phase][0] if trainer.current_phase < len(trainer.curriculum_schedule) else 'N/A'}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("Starting fresh training...")
    elif resume_checkpoint == "auto":
        try:
            latest_checkpoint = trainer.checkpoint_manager.load_latest_checkpoint()
            trainer.model.load_state_dict(latest_checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])
            trainer.current_epoch = latest_checkpoint.get('epoch', 0)
            trainer.best_eos_rate = latest_checkpoint.get('eos_rate', 0.0)
            
            # Restore curriculum phase progress
            trainer.current_phase = latest_checkpoint.get('current_phase', 0)
            trainer.phase_epoch = latest_checkpoint.get('phase_epoch', 0)
            
            # Calculate which phase we're in based on epoch if phase info missing
            if 'current_phase' not in latest_checkpoint:
                trainer._calculate_phase_from_epoch()
            
            phase_name = trainer.curriculum_schedule[trainer.current_phase][2] if trainer.current_phase < len(trainer.curriculum_schedule) else "Completed"
            print(f"✅ Auto-resumed from epoch {trainer.current_epoch}, best EOS rate: {trainer.best_eos_rate:.1f}%")
            print(f"📍 Current phase: {trainer.current_phase + 1}/5 - {phase_name}")
            print(f"📍 Phase progress: {trainer.phase_epoch}/{trainer.curriculum_schedule[trainer.current_phase][0] if trainer.current_phase < len(trainer.curriculum_schedule) else 'N/A'}")
        except Exception as e:
            print(f"❌ Failed to auto-resume: {e}")
            print("Starting fresh training...")
    
    # Run curriculum training
    success = trainer.run_curriculum_training()
    
    if success:
        final_msg = "✅ Curriculum training SUCCESSFUL! EOS generation learned."
        print(f"\n{final_msg}")
        logger.info(final_msg)
    else:
        final_msg = "❌ Curriculum training needs more work. Consider adjusting parameters."
        print(f"\n{final_msg}")
        logger.warning(final_msg) 