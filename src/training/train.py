"""
Main training script for Khmer OCR Seq2Seq model.
Integrates all training infrastructure components for end-to-end training.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import logging
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.dataset import KhmerDataset
from src.utils.config import ConfigManager
from .trainer import Trainer
from .validator import Validator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup global logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def create_data_loaders(config: ConfigManager, num_train_samples: int = 10000, num_val_samples: int = 1000) -> tuple:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration manager
        num_train_samples: Number of training samples per epoch
        num_val_samples: Number of validation samples
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logging.info("Creating datasets...")
    
    # Create training dataset with augmentation
    train_dataset = KhmerDataset(
        config=config,
        num_samples=num_train_samples,
        apply_augmentation=True,
        mode='train'
    )
    
    # Create validation dataset without augmentation
    val_dataset = KhmerDataset(
        config=config,
        num_samples=num_val_samples,
        apply_augmentation=False,  # No augmentation for validation
        mode='val'
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    logging.info(f"Created train dataloader: {len(train_dataloader)} batches")
    logging.info(f"Created val dataloader: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader


def create_model(config: ConfigManager, device: torch.device) -> KhmerOCRSeq2Seq:
    """
    Create and initialize the Seq2Seq model.
    
    Args:
        config: Configuration manager
        device: Device to place model on
        
    Returns:
        Initialized Seq2Seq model
    """
    logging.info("Creating model...")
    
    model = KhmerOCRSeq2Seq(config)
    
    # Move to device
    model = model.to(device)
    
    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created:")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")
    logging.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Khmer OCR Seq2Seq Model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--train-samples", 
        type=int, 
        default=10000,
        help="Number of training samples per epoch"
    )
    parser.add_argument(
        "--val-samples", 
        type=int, 
        default=1000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=None,
        help="GPU device ID (None for auto-detect)"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="models/checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="Directory to save logs and TensorBoard"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("KhmerOCRTraining")
    
    logger.info("="*60)
    logger.info("Khmer OCR Seq2Seq Training")
    logger.info("="*60)
    
    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            logger.warning(f"GPU {args.gpu} not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = ConfigManager(args.config)
        
        # Log training configuration
        logger.info("Training Configuration:")
        logger.info(f"  Epochs: {config.training_config.epochs}")
        logger.info(f"  Batch size: {config.training_config.batch_size}")
        logger.info(f"  Learning rate: {config.training_config.learning_rate}")
        logger.info(f"  Optimizer: {config.training_config.optimizer}")
        logger.info(f"  Teacher forcing ratio: {config.training_config.teacher_forcing_ratio}")
        logger.info(f"  Gradient clip: {config.training_config.gradient_clip}")
        
        # Create directories
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Create data loaders
        train_dataloader, val_dataloader = create_data_loaders(
            config, 
            args.train_samples, 
            args.val_samples
        )
        
        # Create model
        model = create_model(config, device)
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            config_manager=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Start training
        logger.info("Starting training process...")
        trainer.train(resume_from_checkpoint=args.resume)
        
        logger.info("Training completed successfully!")
        
        # Validate on a few samples for inspection
        logger.info("Running sample validation...")
        sample_results = trainer.validator.validate_samples(val_dataloader, num_samples=5)
        
        logger.info("Sample validation results:")
        for i, result in enumerate(sample_results):
            logger.info(f"  Sample {i+1}:")
            logger.info(f"    Target: '{result['target_clean']}'")
            logger.info(f"    Prediction: '{result['prediction_clean']}'")
            logger.info(f"    Correct: {result['correct']}")
            logger.info(f"    CER: {result['cer']:.2%}")
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 