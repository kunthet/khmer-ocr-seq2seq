"""
Training infrastructure for Khmer OCR Seq2Seq model.
Implements training loop with teacher forcing, loss computation, and gradient updates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging

from ..models.seq2seq import KhmerOCRSeq2Seq
from ..utils.config import ConfigManager
from .validator import Validator
from .checkpoint_manager import CheckpointManager


class Trainer:
    """
    Main trainer class for Khmer OCR Seq2Seq model.
    
    Implements:
    - Training loop with teacher forcing (ratio = 1.0)
    - Adam optimizer with learning rate 1e-6
    - Cross-entropy loss (NLLLoss)
    - Gradient clipping and checkpoint management
    """
    
    def __init__(
        self,
        model: KhmerOCRSeq2Seq,
        config_manager: ConfigManager,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        log_dir: str = "logs",
        checkpoint_dir: str = "models/checkpoints",
        gdrive_backup: bool = True
    ):
        self.model = model
        self.config = config_manager
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer according to PRD specs
        self.optimizer = self._initialize_optimizer()
        
        # Initialize loss function (NLLLoss for LogSoftmax output)
        self.criterion = nn.NLLLoss(ignore_index=self.config.vocab.PAD_IDX)
        
        # Initialize components with Google Drive backup
        self.validator = Validator(model, config_manager, device)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            gdrive_backup=gdrive_backup
        )
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup TensorBoard
        run_name = f"khmer_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(os.path.join(log_dir, run_name))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_cer = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_cer': [],
            'learning_rate': []
        }
        
        # Check for Google Drive backup status
        gdrive_info = self.checkpoint_manager.get_gdrive_info()
        if gdrive_info.get("enabled"):
            self.logger.info(f"Google Drive backup: {'✅ Ready' if gdrive_info.get('gdrive_accessible') else '❌ Not accessible'}")
            if gdrive_info.get('gdrive_accessible'):
                self.logger.info(f"Google Drive dir: {gdrive_info.get('gdrive_dir')}")
                # Try to sync from Google Drive if no local checkpoints
                if not self.checkpoint_manager.list_checkpoints():
                    synced = self.checkpoint_manager.sync_from_gdrive()
                    if synced:
                        self.logger.info("Synced checkpoints from Google Drive")
        else:
            self.logger.info("Google Drive backup: ❌ Disabled")
    
    def _initialize_optimizer(self) -> optim.Optimizer:
        """Initialize Adam optimizer with PRD specifications."""
        if self.config.training_config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training_config.optimizer}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training process."""
        logger = logging.getLogger("KhmerOCRTrainer")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train model for one epoch with teacher forcing.
        
        Returns:
            Dict containing training metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            images = batch['images'].to(self.device)  # [B, 1, H, W]
            targets = batch['targets'].to(self.device)  # [B, max_seq_len]
            target_lengths = batch['target_lengths'].to(self.device)  # [B]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with teacher forcing (ratio = 1.0 as per PRD)
            batch_size = images.size(0)
            max_seq_len = targets.size(1)
            
            # Initialize decoder input with SOS tokens
            decoder_input = torch.full(
                (batch_size, 1), 
                self.config.vocab.SOS_IDX, 
                dtype=torch.long, 
                device=self.device
            )
            
            # Get encoder outputs
            encoder_outputs, decoder_hidden = self.model.encode(images)
            
            # Initialize loss for this batch
            batch_loss = 0.0
            total_predictions = 0
            
            # Teacher forcing: use ground truth as decoder input
            for t in range(1, max_seq_len):  # Skip SOS, predict from index 1
                # Decode next token
                decoder_output, decoder_hidden, attention_weights = self.model.decode_step(
                    decoder_input[:, -1:],  # Last predicted token
                    decoder_hidden,
                    encoder_outputs
                )
                
                # Get targets for this time step
                target_t = targets[:, t]  # [B]
                
                # Calculate loss only for non-padded positions
                mask = target_t != self.config.vocab.PAD_IDX
                valid_predictions = mask.sum().item()
                
                if valid_predictions > 0:
                    # decoder_output: [B, vocab_size] (LogSoftmax output - log probabilities)
                    step_loss = self.criterion(decoder_output[mask], target_t[mask])
                    batch_loss += step_loss
                    total_predictions += valid_predictions
                
                # Teacher forcing: use ground truth as next input
                next_input = targets[:, t:t+1]  # [B, 1]
                decoder_input = torch.cat([decoder_input, next_input], dim=1)
            
            # Average loss over sequence length
            if total_predictions > 0:
                batch_loss = batch_loss / max_seq_len
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping as per PRD specifications
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training_config.gradient_clip
            )
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            self.global_step += 1
            
            # Log batch progress
            if batch_idx % 100 == 0:
                progress = 100.0 * batch_idx / num_batches
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches} "
                    f"({progress:.1f}%), Loss: {batch_loss.item():.4f}"
                )
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/Train_Batch', batch_loss.item(), self.global_step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        return {
            'train_loss': avg_loss,
            'epoch_time': epoch_time,
            'num_batches': num_batches
        }
    
    def train(self, resume_from_checkpoint: str = None) -> None:
        """
        Main training loop for the specified number of epochs.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        self.logger.info("Starting training...")
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Epochs: {self.config.training_config.epochs}")
        self.logger.info(f"  Batch size: {self.config.training_config.batch_size}")
        self.logger.info(f"  Learning rate: {self.config.training_config.learning_rate}")
        self.logger.info(f"  Teacher forcing ratio: {self.config.training_config.teacher_forcing_ratio}")
        self.logger.info(f"  Device: {self.device}")
        
        total_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training_config.epochs):
            self.current_epoch = epoch
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training_config.epochs}")
            self.logger.info(f"{'='*60}")
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validator.validate(self.val_dataloader)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['val_cer'].append(val_metrics['cer'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log epoch results
            self.logger.info(f"\nEpoch {epoch + 1} Results:")
            self.logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"  Val CER: {val_metrics['cer']:.2%}")
            self.logger.info(f"  Epoch Time: {train_metrics['epoch_time']:.2f}s")
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train_Epoch', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/Val_Epoch', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Metrics/CER', val_metrics['cer'], epoch)
            self.writer.add_scalar('Metrics/Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['cer'] < self.best_cer
            if is_best:
                self.best_cer = val_metrics['cer']
                self.logger.info(f"  New best CER: {self.best_cer:.2%}")
            
            self.save_checkpoint(
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best
            )
            
            # Early stopping check (if CER reaches target)
            if val_metrics['cer'] <= 0.01:  # 1.0% CER target
                self.logger.info(f"\nTarget CER achieved: {val_metrics['cer']:.2%}")
                self.logger.info("Stopping training early!")
                break
        
        total_time = time.time() - total_start_time
        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best CER: {self.best_cer:.2%}")
        
        # Close TensorBoard writer
        self.writer.close()
    
    def save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict[str, float], 
        is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'model': self.config.model_config,
                'training': self.config.training_config,
                'data': self.config.data_config
            },
            'metrics': metrics,
            'best_cer': self.best_cer,
            'global_step': self.global_step
        }
        
        self.checkpoint_manager.save_checkpoint(
            checkpoint_data, 
            epoch, 
            is_best=is_best
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint and resume training."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint_data['epoch'] + 1
        self.training_history = checkpoint_data['training_history']
        self.best_cer = checkpoint_data['best_cer']
        self.global_step = checkpoint_data['global_step']
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
        self.logger.info(f"Best CER so far: {self.best_cer:.2%}") 