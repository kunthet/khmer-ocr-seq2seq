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
import argparse
import time

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
    val_dataset = None
    val_collate_fn = None
    
    # Try to load fixed validation set (images are directly in validation_dir/images/)
    if (validation_path / "images").exists():
        image_count = len(list((validation_path / "images").glob("*.png")))
        if image_count > 0:
            try:
                logging.info(f"Loading fixed validation set from {validation_dir} (found {image_count} images)")
                val_dataset = SyntheticImageDataset(
                    split="",  # No split subdirectory, images are directly in validation_dir
                    synthetic_dir=validation_dir,
                    config_manager=config_manager,
                    max_samples=6400
                )
                if len(val_dataset) > 0:
                    val_collate_fn = SyntheticCollateFunction(config_manager.vocab)
                    logging.info(f"✅ Successfully loaded {len(val_dataset)} validation samples")
                else:
                    val_dataset = None
                    logging.warning("Validation dataset is empty, falling back to OnTheFlyDataset")
            except Exception as e:
                logging.warning(f"Failed to load SyntheticImageDataset: {e}")
                val_dataset = None
    
    # Check if val/ subdirectory structure exists as fallback
    elif (validation_path / "val" / "images").exists():
        image_count = len(list((validation_path / "val" / "images").glob("*.png")))
        if image_count > 0:
            try:
                logging.info(f"Loading fixed validation set from {validation_dir}/val (found {image_count} images)")
                val_dataset = SyntheticImageDataset(
                    split="val",
                    synthetic_dir=validation_dir,
                    config_manager=config_manager,
                    max_samples=6400
                )
                if len(val_dataset) > 0:
                    val_collate_fn = SyntheticCollateFunction(config_manager.vocab)
                    logging.info(f"✅ Successfully loaded {len(val_dataset)} validation samples")
                else:
                    val_dataset = None
            except Exception as e:
                logging.warning(f"Failed to load SyntheticImageDataset from val/ subdirectory: {e}")
                val_dataset = None
    
    # Fallback to OnTheFlyDataset if SyntheticImageDataset failed
    if val_dataset is None:
        logging.warning(f"Fixed validation set not found or failed to load. Using OnTheFlyDataset for validation instead")
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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Advanced Khmer OCR Training with Performance Optimizations")
    
    # Training mode arguments
    parser.add_argument('--test-run', action='store_true', 
                       help='Run a quick test with small dataset (128 train, 64 val samples, 2 epochs)')
    parser.add_argument('--fresh-start', action='store_true',
                       help='Start training from scratch (ignore existing checkpoints)')
    parser.add_argument('--continue-training', action='store_true',
                       help='Continue training from the latest checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--target-cer', type=float, default=5.0,
                       help='Target CER for early stopping (default: 5.0 percent)')
    
    # Dataset parameters
    parser.add_argument('--train-samples', type=int, default=None,
                       help='Number of training samples per epoch (overrides curriculum learning)')
    parser.add_argument('--val-samples', type=int, default=None,
                       help='Number of validation samples')
    
    # Advanced features
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs for curriculum learning')
    
    # Paths
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--corpus-dir', type=str, default='data/processed',
                       help='Path to corpus directory')
    parser.add_argument('--validation-dir', type=str, default='data/validation_fixed',
                       help='Path to validation directory')
    parser.add_argument('--log-dir', type=str, default='logs/advanced_training',
                       help='Path to log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                       help='Path to checkpoint directory')
    
    return parser.parse_args()


def create_test_data_loaders(config_manager, args):
    """Create smaller datasets for testing"""
    logging.info("Creating TEST datasets...")
    
    # Create small on-the-fly training dataset
    train_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config_manager,
        corpus_dir=args.corpus_dir,
        samples_per_epoch=args.train_samples or 128,  # Small test size
        augment_prob=0.8,
        shuffle_texts=True,
        random_seed=None
    )
    
    # Create small validation dataset
    val_dataset = OnTheFlyDataset(
        split="val",
        config_manager=config_manager,
        corpus_dir=args.corpus_dir,
        samples_per_epoch=args.val_samples or 64,  # Small test size
        augment_prob=0.0,
        shuffle_texts=False,
        random_seed=42
    )
    
    # Create collate functions
    train_collate_fn = OnTheFlyCollateFunction(config_manager.vocab)
    val_collate_fn = OnTheFlyCollateFunction(config_manager.vocab)
    
    # Create data loaders with smaller batch size and no workers for testing
    batch_size = args.batch_size or 16  # Smaller batch for testing
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for testing
        pin_memory=True,
        collate_fn=train_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for testing
        pin_memory=True,
        collate_fn=val_collate_fn
    )
    
    logging.info(f"Created TEST train dataloader: {len(train_dataloader)} batches ({len(train_dataset)} samples)")
    logging.info(f"Created TEST val dataloader: {len(val_dataloader)} batches ({len(val_dataset)} samples)")
    
    return train_dataloader, val_dataloader


def check_gdrive_availability(gdrive_dir="/content/drive/MyDrive/KhmerOCR_Checkpoints"):
    """
    Check if Google Drive is available and accessible.
    
    Args:
        gdrive_dir: Google Drive directory path to check
        
    Returns:
        dict: Information about Google Drive availability
    """
    import platform
    from pathlib import Path
    
    # Google Drive is typically only available in Colab or mounted drives
    is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
    is_windows = platform.system() == "Windows"
    is_linux = platform.system() == "Linux"
    
    gdrive_path = Path(gdrive_dir)
    
    # Check if path exists and is writable
    gdrive_accessible = False
    gdrive_writable = False
    mount_detected = False
    
    try:
        # Check if Google Drive mount point exists (typical Colab pattern)
        if Path("/content/drive").exists():
            mount_detected = True
            
        # Check if the specific directory exists and is accessible
        if gdrive_path.exists():
            gdrive_accessible = True
            
            # Test write access
            test_file = gdrive_path / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
                gdrive_writable = True
            except (PermissionError, OSError):
                gdrive_writable = False
                
    except Exception:
        gdrive_accessible = False
        gdrive_writable = False
    
    return {
        "available": gdrive_accessible and gdrive_writable,
        "accessible": gdrive_accessible,
        "writable": gdrive_writable,
        "mount_detected": mount_detected,
        "is_colab": is_colab,
        "is_windows": is_windows,
        "is_linux": is_linux,
        "gdrive_dir": str(gdrive_dir),
        "recommendation": get_gdrive_recommendation(is_colab, is_windows, gdrive_accessible, gdrive_writable)
    }

def get_gdrive_recommendation(is_colab, is_windows, accessible, writable):
    """Get recommendation for Google Drive setup based on environment."""
    if is_colab and not accessible:
        return "Mount Google Drive in Colab: from google.colab import drive; drive.mount('/content/drive')"
    elif is_windows and not accessible:
        return "Google Drive not available on local Windows. Using local checkpoints only."
    elif accessible and not writable:
        return "Google Drive accessible but not writable. Check permissions."
    elif accessible and writable:
        return "Google Drive fully functional for backup."
    else:
        return "Google Drive not available. Using local checkpoints only."

def create_trainer_with_safe_gdrive(model, config, train_dataloader, val_dataloader, device, args):
    """
    Create trainer with safe Google Drive backup handling.
    
    Args:
        model: Khmer OCR model
        config: Configuration manager
        train_dataloader: Training data loader
        val_dataloader: Validation data loader  
        device: Torch device
        args: Command line arguments
        
    Returns:
        Trainer instance with properly configured backup settings
    """
    logger = logging.getLogger("AdvancedKhmerOCRTraining")
    
    # Check Google Drive availability
    gdrive_info = check_gdrive_availability()
    
    logger.info("🔄 Google Drive Backup Status:")
    logger.info(f"  Environment: {'Colab' if gdrive_info['is_colab'] else 'Windows' if gdrive_info['is_windows'] else 'Linux'}")
    logger.info(f"  Mount detected: {'✅' if gdrive_info['mount_detected'] else '❌'}")
    logger.info(f"  Accessible: {'✅' if gdrive_info['accessible'] else '❌'}")
    logger.info(f"  Writable: {'✅' if gdrive_info['writable'] else '❌'}")
    logger.info(f"  Status: {'🟢 Enabled' if gdrive_info['available'] else '🔴 Disabled'}")
    logger.info(f"  Info: {gdrive_info['recommendation']}")
    
    # Determine backup settings
    enable_gdrive_backup = gdrive_info['available'] and not args.test_run
    
    if not enable_gdrive_backup:
        if args.test_run:
            logger.info("📁 Google Drive backup disabled for test run")
        else:
            logger.warning("⚠️  Google Drive backup disabled - checkpoints will be saved locally only")
            logger.info("💡 To enable Google Drive backup:")
            if gdrive_info['is_windows']:
                logger.info("   • Use Google Colab for automatic Google Drive integration")
                logger.info("   • Or manually map a Google Drive folder and update gdrive_dir path")
            else:
                logger.info("   • Mount Google Drive if in Colab environment")
                logger.info("   • Ensure write permissions to Google Drive folder")
    
    try:
        # Create trainer with safe backup settings
        trainer = AdvancedKhmerOCRTrainer(
            model=model,
            config_manager=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            use_mixed_precision=not args.no_mixed_precision,
            curriculum_learning=not args.no_curriculum,
            warmup_epochs=args.warmup_epochs,
            gdrive_backup=enable_gdrive_backup
        )
        
        # Log backup configuration
        backup_status = "Google Drive + Local" if enable_gdrive_backup else "Local Only"
        logger.info(f"💾 Checkpoint backup: {backup_status}")
        logger.info(f"📂 Local checkpoint dir: {args.checkpoint_dir}")
        if enable_gdrive_backup:
            logger.info(f"☁️  Google Drive backup dir: {gdrive_info['gdrive_dir']}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Failed to create trainer with Google Drive backup: {e}")
        logger.info("🔄 Retrying with local backup only...")
        
        # Fallback: Create trainer with local backup only
        trainer = AdvancedKhmerOCRTrainer(
            model=model,
            config_manager=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            use_mixed_precision=not args.no_mixed_precision,
            curriculum_learning=not args.no_curriculum,
            warmup_epochs=args.warmup_epochs,
            gdrive_backup=False  # Force disable on fallback
        )
        
        logger.info("✅ Trainer created with local backup only")
        return trainer


def main():
    """Main training function with advanced optimizations"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging with proper encoding for Windows
    if sys.platform == "win32":
        # Use ASCII-safe format for Windows
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("AdvancedKhmerOCRTraining")
    
    # Test run mode
    if args.test_run:
        logger.info("=" * 42)
        logger.info("TEST RUN MODE")
        logger.info("Running a test with small dataset:")
        logger.info(f"  Epochs: {args.epochs or 2}")
        logger.info(f"  Train samples: {args.train_samples or 128}")
        logger.info(f"  Val samples: {args.val_samples or 64}")
        logger.info(f"  Batch size: {args.batch_size or 16}")
        logger.info(f"  Num workers: 0")
        logger.info("=" * 42)
    
    logger.info("Starting Advanced Khmer OCR Training...")
    logger.info("=" * 60)
    logger.info("Advanced Features:")
    logger.info("  - Mixed Precision Training (AMP)")
    logger.info("  - Curriculum Learning")
    logger.info("  - Advanced Learning Rate Scheduling")
    logger.info("  - Dynamic Teacher Forcing")
    logger.info("  - Enhanced Regularization")
    logger.info("  - OnTheFlyDataset Integration")
    logger.info("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = ConfigManager(args.config)
        
        # Override config with command line arguments
        if args.epochs is not None:
            config.training_config.epochs = args.epochs
        elif args.test_run:
            config.training_config.epochs = 2  # Default for test run
            
        if args.batch_size is not None:
            config.training_config.batch_size = args.batch_size
        elif args.test_run:
            config.training_config.batch_size = 16  # Smaller batch for test
            
        if args.learning_rate is not None:
            config.training_config.learning_rate = args.learning_rate
        
        # Create model
        logger.info("Creating Khmer OCR Seq2Seq model...")
        model = KhmerOCRSeq2Seq(config_or_vocab_size=config)
        model = model.to(device)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
        # Create data loaders
        if args.test_run:
            train_dataloader, val_dataloader = create_test_data_loaders(config, args)
        else:
            train_dataloader, val_dataloader = create_data_loaders(
                config_manager=config,
                batch_size=config.training_config.batch_size,
                corpus_dir=args.corpus_dir,
                validation_dir=args.validation_dir
            )
        
        # Create trainer with safe Google Drive handling
        logger.info("🏋️  Initializing advanced trainer...")
        trainer = create_trainer_with_safe_gdrive(model, config, train_dataloader, val_dataloader, device, args)
        
        # Override optimizer and scheduler for non-test runs
        if not args.test_run:
            trainer.optimizer, trainer.scheduler = trainer.create_advanced_optimizer()
            logger.info("🔧 Using advanced optimizer: AdamW with OneCycleLR")
        else:
            logger.info("🔧 Using standard optimizer for test run")
        
        # Training loop
        epochs = args.epochs if args.epochs else config.training_config.epochs
        logger.info(f"🎯 Starting advanced training for {epochs} epochs...")
        logger.info(f"🏆 Target: <={args.target_cer}% CER with advanced optimizations")
        
        try:
            # Training loop
            for epoch in range(epochs):
                logger.info(f"🔄 Starting epoch {epoch}/{epochs}")
                epoch_start_time = time.time()
                
                # Update curriculum learning parameters
                if args.test_run:
                    # For test run, keep simple settings
                    pass
                else:
                    # Dynamic curriculum learning
                    if hasattr(trainer, 'curriculum_learning') and trainer.curriculum_learning:
                        trainer.update_curriculum(epoch)
                
                # Train epoch
                train_metrics = trainer.train_epoch()
                
                # Validate
                val_metrics = trainer.validator.validate(trainer.val_dataloader)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Print results in the requested format
                logger.info(f"📊 Epoch {epoch} Results:")
                logger.info(f"  🚂 Train Loss: {train_metrics['train_loss']:.4f}")
                logger.info(f"  📊 Val Loss: {val_metrics['val_loss']:.4f}")
                logger.info(f"  🎯 Val CER: {val_metrics['cer']*100:.2f}%")
                logger.info(f"  ⏱️  Epoch Time: {epoch_time:.2f}s")
                
                # Save checkpoint
                is_best = val_metrics['cer'] < getattr(trainer, 'best_cer', float('inf'))
                trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
                
                # Early stopping check
                if val_metrics['cer'] <= (args.target_cer / 100.0):  # Convert percentage to decimal
                    logger.info(f"🎉 Target CER achieved at epoch {epoch}!")
                    logger.info(f"🏆 Final CER: {val_metrics['cer']*100:.2f}% (Target: <={args.target_cer}%)")
                    break
                    
            logger.info("✅ Training completed successfully!")
            
            # Final summary
            final_cer = val_metrics['cer'] * 100
            logger.info("=" * 80)
            logger.info("📋 TRAINING SUMMARY")
            logger.info("=" * 80)
            logger.info(f"🎯 Final CER: {final_cer:.2f}%")
            logger.info(f"🏆 Target CER: <={args.target_cer}%")
            logger.info(f"✅ Success: {'Yes' if final_cer <= args.target_cer else 'No'}")
            logger.info(f"📊 Total Epochs: {epoch + 1}")
            logger.info(f"💾 Checkpoints saved to: {args.checkpoint_dir}")
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("⚠️  Training interrupted by user")
            logger.info("💾 Saving current progress...")
            try:
                trainer.save_checkpoint(epoch, val_metrics, is_best=False)
                logger.info("✅ Progress saved successfully")
            except:
                logger.warning("❌ Failed to save progress")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize training: {e}")
        import traceback
        logger.error("🔍 Full error traceback:")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 