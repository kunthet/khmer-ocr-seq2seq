"""
Production Training Script for Khmer OCR Seq2Seq Model

This script runs full-scale training using the processed corpus data
for 150 epochs targeting ≤1.0% CER performance.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import ConfigManager
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.synthetic_dataset import create_synthetic_dataloaders
from src.training.trainer import Trainer

def setup_logging(log_dir: str, log_level: str = "INFO") -> str:
    """Setup logging configuration with UTF-8 encoding."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"production_training_{timestamp}.log")
    
    # Setup logging with UTF-8 encoding
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def estimate_memory_usage(model, batch_size: int, device: torch.device) -> float:
    """Estimate memory usage for model and batch."""
    # Model parameters
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Rough estimation for activations (very approximate)
    activation_size = batch_size * 32 * 128 * 4  # Height * Width * float32
    
    total_bytes = param_size + activation_size
    total_gb = total_bytes / (1024**3)
    
    return total_gb

def calculate_batch_size(model, device: torch.device, max_memory_gb: float = 8.0) -> int:
    """Calculate optimal batch size based on available memory."""
    if device.type == 'cuda':
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = gpu_memory * 0.8  # Use 80% of GPU memory
        max_memory_gb = available_memory / (1024**3)
    
    # Start with a reasonable batch size and adjust
    batch_sizes = [32, 16, 8, 4, 2, 1]
    
    for batch_size in batch_sizes:
        estimated_memory = estimate_memory_usage(model, batch_size, device)
        if estimated_memory <= max_memory_gb:
            return batch_size
    
    return 1  # Fallback to smallest batch size

def main():
    """Main production training function."""
    parser = argparse.ArgumentParser(description="Khmer OCR Production Training")
    parser.add_argument("--synthetic-dir", type=str, default="data/synthetic", help="Synthetic images directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto-detect if not specified)")
    parser.add_argument("--num-epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs/production", help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Checkpoint directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per split (for testing)")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Production training log: {log_file}")
    logger.info("=== Khmer OCR Production Training ===")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Check device
        device = get_device()
        if device.type == 'cpu':
            logger.warning("No GPU available - training will be very slow on CPU")
        else:
            logger.info(f"Using device: {device}")
            if device.type == 'cuda':
                logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")
        
        # Check if synthetic images exist
        synthetic_dir = Path(args.synthetic_dir)
        if not synthetic_dir.exists():
            logger.error(f"Synthetic images directory not found: {synthetic_dir}")
            logger.error("Please run: python generate_synthetic_images.py")
            return
        
        # Check for required splits
        required_splits = ["train", "val", "test"]
        missing_splits = []
        for split in required_splits:
            split_dir = synthetic_dir / split
            if not split_dir.exists() or not (split_dir / "images").exists():
                missing_splits.append(split)
        
        if missing_splits:
            logger.error(f"Missing synthetic image splits: {missing_splits}")
            logger.error("Please run: python generate_synthetic_images.py")
            return
        
        logger.info("Synthetic images found and ready")
        
        # Load configuration
        config = ConfigManager()
        
        # Override epochs in config with command line argument
        config.training_config.epochs = args.num_epochs
        
        logger.info("Configuration loaded successfully")
        
        # Create model
        logger.info("Creating Khmer OCR Seq2Seq model...")
        model = KhmerOCRSeq2Seq(config_or_vocab_size=config)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Memory estimation
        memory_gb = estimate_memory_usage(model, 32, device)
        logger.info(f"Model memory: ~{memory_gb:.2f}GB (FP32)")
        
        # Determine batch size
        if args.batch_size is None:
            batch_size = calculate_batch_size(model, device)
            logger.info(f"Using batch size: {batch_size}")
        else:
            batch_size = args.batch_size
            logger.info(f"Using specified batch size: {batch_size}")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_dataloader, val_dataloader, test_dataloader = create_synthetic_dataloaders(
            config_manager=config,
            batch_size=batch_size,
            num_workers=4,
            synthetic_dir=args.synthetic_dir,
            max_samples=args.max_samples
        )
        
        train_size = len(train_dataloader.dataset)
        val_size = len(val_dataloader.dataset)
        test_size = len(test_dataloader.dataset)
        
        logger.info(f"Data loaders created - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            config_manager=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir
        )
        
        if args.validate_only:
            logger.info("Running validation only...")
            val_metrics = trainer.validator.validate(val_dataloader)
            logger.info(f"Validation Results:")
            logger.info(f"  Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  CER: {val_metrics['cer']:.2%}")
            return
        
        # Production training info
        logger.info("Starting production training for {} epochs...".format(args.num_epochs))
        logger.info("Model: Attention-based Seq2Seq with CRNN Encoder")
        logger.info("Dataset: Pre-generated Synthetic Images ({:,} samples)".format(train_size))
        logger.info("Target: <=1.0% CER (Character Error Rate)")
        
        # Start training
        trainer.train(resume_from_checkpoint=args.resume)
        
        logger.info("Production training completed successfully!")
        
        # Final validation
        logger.info("Running final validation...")
        val_metrics = trainer.validator.validate(val_dataloader)
        logger.info(f"Final Validation Results:")
        logger.info(f"  Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  CER: {val_metrics['cer']:.2%}")
        
        if val_metrics['cer'] <= 0.01:
            logger.info("SUCCESS: Target CER achieved!")
        else:
            logger.info("Target CER not yet achieved. Consider additional training.")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Full traceback:", exc_info=True)
        return
    
    finally:
        logger.info("Training session ended")

if __name__ == "__main__":
    main() 