#!/usr/bin/env python3
"""
Advanced Khmer OCR Training Script with Performance Optimizations
Implements curriculum learning, mixed precision, and advanced techniques
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
import logging
import yaml
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset, OnTheFlyCollateFunction
from src.data.synthetic_dataset import SyntheticImageDataset, SyntheticCollateFunction
from src.training.trainer import Trainer
from src.utils.config import ConfigManager

class AdvancedKhmerOCRTrainer(Trainer):
    """Advanced trainer with performance optimizations"""
    
    def __init__(self, *args, **kwargs):
        # Extract advanced training parameters
        self.use_mixed_precision = kwargs.pop('use_mixed_precision', True)
        self.curriculum_learning = kwargs.pop('curriculum_learning', True)
        self.warmup_epochs = kwargs.pop('warmup_epochs', 5)
        
        super().__init__(*args, **kwargs)
        
        # Initialize mixed precision scaler
        if self.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logging.info("✅ Mixed precision training enabled")
        else:
            self.scaler = None
            
        # Curriculum learning parameters
        if self.curriculum_learning:
            self.base_samples_per_epoch = 64000  # Start smaller
            self.max_samples_per_epoch = 256000  # End with full size
    
    def _initialize_optimizer(self):
        """Override optimizer initialization to support AdamW and other optimizers."""
        optimizer_name = self.config.training_config.optimizer.lower()
        
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate
            )
        elif optimizer_name == "adamw":
            weight_decay = getattr(self.config.training_config, 'weight_decay', 1e-4)
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            logging.info("✅ Curriculum learning enabled")
    
    def create_advanced_optimizer(self):
        """Create optimizer with advanced settings"""
        # Use AdamW with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training_config.learning_rate,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Advanced learning rate scheduler
        total_steps = len(self.train_dataloader) * self.config.training_config.epochs
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.training_config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
        )
        
        return optimizer, scheduler
    
    def get_curriculum_samples_per_epoch(self, epoch):
        """Get number of samples based on curriculum learning"""
        if not self.curriculum_learning:
            return self.max_samples_per_epoch
            
        if epoch < self.warmup_epochs:
            # Gradually increase from base to max
            progress = epoch / self.warmup_epochs
            samples = int(self.base_samples_per_epoch + 
                         (self.max_samples_per_epoch - self.base_samples_per_epoch) * progress)
        else:
            samples = self.max_samples_per_epoch
            
        return samples
    
    def create_dynamic_dataloader(self, epoch, config_manager, corpus_dir):
        """Create dataloader with curriculum learning"""
        samples_per_epoch = self.get_curriculum_samples_per_epoch(epoch)
        
        logging.info(f"Epoch {epoch}: Using {samples_per_epoch} samples per epoch")
        
        # Create on-the-fly training dataset
        train_dataset = OnTheFlyDataset(
            split="train",
            config_manager=config_manager,
            corpus_dir=corpus_dir,
            samples_per_epoch=samples_per_epoch,
            augment_prob=0.8 + (epoch * 0.02),  # Gradually increase augmentation
            shuffle_texts=True,
            random_seed=None  # Maximum variety
        )
        
        # Create collate function
        train_collate_fn = OnTheFlyCollateFunction(config_manager.vocab)
        
        # Create dataloader
        return DataLoader(
            train_dataset,
            batch_size=self.config.training_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_collate_fn
        )
    
    def advanced_training_step(self, batch, epoch):
        """Advanced training step with mixed precision"""
        images, targets, target_lengths = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)
        
        # Dynamic teacher forcing ratio
        teacher_forcing_ratio = max(0.5, 1.0 - (epoch * 0.02))
        
        if self.scaler and self.use_mixed_precision:
            # Mixed precision training
            with autocast():
                outputs = self.model(images, targets, teacher_forcing_ratio)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            if hasattr(self.config_manager.training_config, 'gradient_clip'):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config_manager.training_config.gradient_clip
                )
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        else:
            # Standard precision training
            outputs = self.model(images, targets, teacher_forcing_ratio)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            
            if hasattr(self.config.training_config, 'gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training_config.gradient_clip
                )
            
            self.optimizer.step()
        
        return loss.item()


def create_data_loaders(config_manager, batch_size, corpus_dir="data/processed", validation_dir="data/validation_fixed"):
    """Create training and validation data loaders using OnTheFlyDataset"""
    logging.info("Creating datasets...")
    
    # Create initial on-the-fly training dataset (will be recreated dynamically)
    train_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config_manager,
        corpus_dir=corpus_dir,
        samples_per_epoch=64000,  # Starting size for curriculum learning
        augment_prob=0.8,
        shuffle_texts=True,
        random_seed=None
    )
    
    # Create fixed validation dataset
    validation_path = Path(validation_dir)
    
    # Try to load from SyntheticImageDataset first
    val_dataset = None
    val_collate_fn = None
    
    if (validation_path / "val" / "images").exists():
        try:
            logging.info(f"Loading fixed validation set from {validation_dir}")
            val_dataset = SyntheticImageDataset(
                split="val",
                synthetic_dir=validation_dir,
                config_manager=config_manager,
                max_samples=6400
            )
            if len(val_dataset) > 0:
                val_collate_fn = SyntheticCollateFunction(config_manager.vocab)
            else:
                val_dataset = None
        except Exception as e:
            logging.warning(f"Failed to load SyntheticImageDataset: {e}")
            val_dataset = None
    
    elif (validation_path / "images").exists():
        try:
            logging.info(f"Loading fixed validation set from {validation_dir}")
            val_dataset = SyntheticImageDataset(
                split="val",
                synthetic_dir=validation_dir,
                config_manager=config_manager,
                max_samples=6400
            )
            if len(val_dataset) > 0:
                val_collate_fn = SyntheticCollateFunction(config_manager.vocab)
            else:
                val_dataset = None
        except Exception as e:
            logging.warning(f"Failed to load SyntheticImageDataset: {e}")
            val_dataset = None
    
    # Fallback to OnTheFlyDataset if SyntheticImageDataset failed
    if val_dataset is None:
        logging.warning(f"Using OnTheFlyDataset for validation instead")
        val_dataset = OnTheFlyDataset(
            split="val",
            config_manager=config_manager,
            corpus_dir=corpus_dir,
            samples_per_epoch=6400,
            augment_prob=0.0,
            shuffle_texts=False,
            random_seed=42
        )
        val_collate_fn = OnTheFlyCollateFunction(config_manager.vocab)
    
    # Create collate functions
    train_collate_fn = OnTheFlyCollateFunction(config_manager.vocab)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_collate_fn
    )
    
    logging.info(f"Created train dataloader: {len(train_dataloader)} batches ({len(train_dataset)} samples)")
    logging.info(f"Created val dataloader: {len(val_dataloader)} batches ({len(val_dataset)} samples)")
    
    return train_dataloader, val_dataloader


def main():
    """Main training function with advanced optimizations"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("AdvancedKhmerOCRTraining")
    
    logger.info("🚀 Starting Advanced Khmer OCR Training...")
    logger.info("=" * 60)
    logger.info("Advanced Features:")
    logger.info("  ✅ Mixed Precision Training (AMP)")
    logger.info("  ✅ Curriculum Learning")
    logger.info("  ✅ Advanced Learning Rate Scheduling")
    logger.info("  ✅ Dynamic Teacher Forcing")
    logger.info("  ✅ Enhanced Regularization")
    logger.info("  ✅ OnTheFlyDataset Integration")
    logger.info("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    try:
        # Load configuration
        config_path = "configs/train_config.yaml"
        logger.info(f"Loading configuration from: {config_path}")
        config = ConfigManager(config_path)
        
        # Create model
        logger.info("Creating Khmer OCR Seq2Seq model...")
        model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
        model = model.to(device)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
        # Create data loaders
        batch_size = config.training_config.batch_size
        train_dataloader, val_dataloader = create_data_loaders(
            config_manager=config,
            batch_size=batch_size,
            corpus_dir="data/processed",
            validation_dir="data/validation_fixed"
        )
        
        # Create advanced trainer
        logger.info("Initializing advanced trainer...")
        trainer = AdvancedKhmerOCRTrainer(
            model=model,
            config_manager=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            log_dir="logs/advanced_training",
            checkpoint_dir="models/checkpoints",
            use_mixed_precision=True,
            curriculum_learning=True,
            warmup_epochs=5,
            gdrive_backup=True
        )
        
        # Override optimizer and scheduler
        trainer.optimizer, trainer.scheduler = trainer.create_advanced_optimizer()
        
        logger.info("Starting advanced training...")
        logger.info(f"Target: <=5.0% CER with advanced optimizations")
        
        # Training loop with curriculum learning
        for epoch in range(config.training_config.epochs):
            # Create dynamic dataloader for curriculum learning
            if trainer.curriculum_learning and epoch > 0:
                train_dataloader = trainer.create_dynamic_dataloader(
                    epoch, config, "data/processed"
                )
                trainer.train_dataloader = train_dataloader
            
            # Train epoch
            trainer.train_epoch()
            
            # Validate
            val_metrics = trainer.validate()
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, val_metrics)
            
            # Check for early stopping
            if val_metrics['cer'] <= 0.05:  # 5% CER target
                logger.info(f"🎉 Target CER achieved at epoch {epoch}!")
                break
        
        logger.info("✅ Advanced training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 