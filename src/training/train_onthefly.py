"""
On-the-Fly Training Script for Khmer OCR Seq2Seq Model

This training script uses:
1. On-the-fly image generation during training (no pre-generated training images)
2. Fixed validation set of exactly 6,400 images for consistent evaluation
3. Reduced storage requirements and unlimited augmentation variety
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
from pathlib import Path
import logging
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset, OnTheFlyCollateFunction
from src.data.synthetic_dataset import SyntheticImageDataset, SyntheticCollateFunction
from src.data.curriculum_dataset import CurriculumDataset
from src.utils.config import ConfigManager
from src.training.trainer import Trainer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup global logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def estimate_memory_usage(model: torch.nn.Module, batch_size: int, device: torch.device) -> float:
    """Estimate GPU memory usage in GB"""
    # Basic memory estimation
    model_params = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Approximate forward pass memory (rough estimate)
    # Image tensor: batch_size * 1 * 32 * max_width * 4 bytes (max_width now configurable, typically 1600)
    max_width = 1600  # Configurable width from config, doubled from 800
    image_memory = batch_size * 1 * 32 * max_width * 4
    
    # Target tensor and other activations (rough estimate)
    other_memory = batch_size * 256 * 4 * 10  # approximate
    
    total_bytes = model_params + image_memory + other_memory
    return total_bytes / (1024**3)  # Convert to GB


class CompatibleCollateFunction:
    """
    Efficient collate function that handles data directly from batch items
    to match trainer expectations. Picklable for multiprocessing.
    """
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.pad_idx = vocab.PAD_IDX
    
    def __call__(self, batch):
        # Extract data from batch items in single pass
        images = []
        targets = []
        texts = []
        image_lengths = []
        target_lengths = []
        
        for item in batch:
            # Extract image and record original width
            img = item['image']
            images.append(img)
            image_lengths.append(img.shape[-1])  # Original width before padding
            
            # Extract target and record original length
            target = item['targets']
            targets.append(target)
            target_lengths.append(len(target))  # Original length before padding
            
            # Extract text if available
            texts.append(item.get('text', ''))
        
        # Pad images to max width in batch
        max_width = max(image_lengths)
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
        targets_tensor = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        # Return in format expected by trainer
        return {
            'images': images_tensor,  # Plural to match trainer expectation
            'targets': targets_tensor,
            'image_lengths': torch.tensor(image_lengths),
            'target_lengths': torch.tensor(target_lengths),
            'texts': texts
        }


def calculate_batch_size(model: torch.nn.Module, device: torch.device) -> int:
    """Calculate optimal batch size based on available memory"""
    if device.type == "cuda":
        # Get available GPU memory
        total_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = total_memory * 0.8  # Use 80% of available memory
        
        # Start with a reasonable batch size and check if it fits
        for batch_size in [64, 32, 16, 8, 4]:
            estimated_usage = estimate_memory_usage(model, batch_size, device)
            if estimated_usage * (1024**3) < available_memory:
                return batch_size
        
        return 2  # Minimum batch size
    else:
        return 8  # Conservative batch size for CPU/MPS


def create_data_loaders(
    config_manager: ConfigManager,
    batch_size: int,
    num_workers: int = 4,
    corpus_dir: str = "data/processed",
    validation_dir: str = "data/validation_fixed",
    train_samples_per_epoch: int = 10000
) -> tuple:
    """
    Create training and validation data loaders.
    
    Args:
        config_manager: Configuration manager
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        corpus_dir: Directory with processed corpus data
        validation_dir: Directory with fixed validation images
        train_samples_per_epoch: Number of training samples per epoch
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logging.info("Creating datasets...")
    
    # Create base on-the-fly training dataset
    base_train_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config_manager,
        corpus_dir=corpus_dir,
        samples_per_epoch=train_samples_per_epoch,
        augment_prob=0.5,
        shuffle_texts=True,
        random_seed=None  # No seed for maximum variety
    )
    
    # Wrap with CurriculumDataset for length control (max_length=150)
    train_dataset = CurriculumDataset(base_train_dataset, max_length=150, config_manager=config_manager)
    logging.info(f"Applied CurriculumDataset wrapper with max_length=150 for training")
    
    # Create fixed validation dataset
    validation_path = Path(validation_dir)
    if validation_path.exists() and (validation_path / "images").exists():
        # Load pre-generated fixed validation set
        logging.info(f"Loading fixed validation set from {validation_dir}")
        val_dataset = SyntheticImageDataset(
            split="val",
            synthetic_dir=validation_dir,
            config_manager=config_manager,
            max_samples=6400  # Ensure exactly 6,400 samples
        )
        val_collate_fn = SyntheticCollateFunction(config_manager.vocab)
    else:
        # Fallback to on-the-fly validation (not recommended)
        logging.warning(f"Fixed validation set not found at {validation_dir}")
        logging.warning("Using on-the-fly validation generation (not recommended for reproducibility)")
        base_val_dataset = OnTheFlyDataset(
            split="val",
            config_manager=config_manager,
            corpus_dir=corpus_dir,
            samples_per_epoch=6400,
            augment_prob=0.0,  # No augmentation for validation
            shuffle_texts=False,
            random_seed=42  # Fixed seed for validation
        )
        # Wrap with CurriculumDataset for consistent length control
        val_dataset = CurriculumDataset(base_val_dataset, max_length=150, config_manager=config_manager)
        val_collate_fn = CompatibleCollateFunction(config_manager.vocab)
    
    # Use compatible collate function that adapts curriculum output to trainer expectations
    train_collate_fn = CompatibleCollateFunction(config_manager.vocab)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_collate_fn
    )
    
    logging.info(f"Created train dataloader: {len(train_dataloader)} batches ({len(train_dataset)} samples)")
    logging.info(f"Created val dataloader: {len(val_dataloader)} batches ({len(val_dataset)} samples)")
    
    return train_dataloader, val_dataloader


def create_model(config: ConfigManager, device: torch.device) -> KhmerOCRSeq2Seq:
    """Create and initialize the model"""
    logging.info("Creating Khmer OCR Seq2Seq model...")
    
    # Pass the ConfigManager object to use the correct configuration
    model = KhmerOCRSeq2Seq(config_or_vocab_size=config)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created successfully")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Khmer OCR Seq2Seq Model with On-the-Fly Generation")
    
    # Training parameters
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=150,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size (auto-calculated if not provided)"
    )
    parser.add_argument(
        "--train-samples-per-epoch", 
        type=int, 
        default=10000,
        help="Number of training samples to generate per epoch"
    )
    
    # Data parameters
    parser.add_argument(
        "--corpus-dir", 
        type=str, 
        default="data/processed",
        help="Directory with processed corpus text files"
    )
    parser.add_argument(
        "--validation-dir", 
        type=str, 
        default="data/validation_fixed",
        help="Directory with fixed validation images"
    )
    
    # Configuration
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
    
    # Output directories
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
    
    # Utility options
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
        "--validate-only", 
        action="store_true",
        help="Only run validation without training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("KhmerOCROnTheFlyTraining")
    
    logger.info("="*70)
    logger.info("Khmer OCR Seq2Seq On-the-Fly Training")
    logger.info("="*70)
    
    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            logger.warning(f"GPU {args.gpu} not available, using auto-detection")
            device = get_device()
    else:
        device = get_device()
    
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    try:
        # Check validation directory
        validation_path = Path(args.validation_dir)
        if not validation_path.exists():
            logger.error(f"Fixed validation set not found: {validation_path}")
            logger.error("Please generate fixed validation set first:")
            logger.error("  python generate_fixed_validation_set.py")
            return
        
        # Validate validation set
        if not (validation_path / "images").exists() or not (validation_path / "metadata.json").exists():
            logger.error(f"Invalid validation set structure at: {validation_path}")
            logger.error("Please regenerate fixed validation set:")
            logger.error("  python generate_fixed_validation_set.py")
            return
        
        logger.info(f"Found fixed validation set at: {validation_path}")
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = ConfigManager(args.config)
        
        # Override epochs in config with command line argument
        config.training_config.epochs = args.num_epochs
        
        # Log training configuration
        logger.info("Training Configuration:")
        logger.info(f"  Epochs: {config.training_config.epochs}")
        logger.info(f"  Learning rate: {config.training_config.learning_rate}")
        logger.info(f"  Optimizer: {config.training_config.optimizer}")
        logger.info(f"  Teacher forcing ratio: {config.training_config.teacher_forcing_ratio}")
        logger.info(f"  Gradient clip: {config.training_config.gradient_clip}")
        logger.info(f"  Training samples per epoch: {args.train_samples_per_epoch}")
        
        # Get gradient accumulation steps from config (default to 1 if not present)
        gradient_accumulation_steps = getattr(config.training_config, 'gradient_accumulation_steps', 1)
        if gradient_accumulation_steps > 1:
            logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        
        # Create directories
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Create model
        model = create_model(config, device)
        
        # Determine batch size
        if args.batch_size is None:
            batch_size = calculate_batch_size(model, device)
            logger.info(f"Auto-calculated batch size: {batch_size}")
        else:
            batch_size = args.batch_size
            logger.info(f"Using specified batch size: {batch_size}")
        
        # Update config with actual batch size
        config.training_config.batch_size = batch_size
        
        # Create data loaders
        train_dataloader, val_dataloader = create_data_loaders(
            config_manager=config,
            batch_size=batch_size,
            corpus_dir=args.corpus_dir,
            validation_dir=args.validation_dir,
            train_samples_per_epoch=args.train_samples_per_epoch
        )
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            config_manager=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            gdrive_backup=True,  # Enable Google Drive backup for Colab training
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        if args.validate_only:
            logger.info("Running validation only...")
            val_metrics = trainer.validator.validate(val_dataloader)
            logger.info(f"Validation Results:")
            logger.info(f"  Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  CER: {val_metrics['cer']:.2%}")
            return
        
        # Training info
        logger.info("Starting on-the-fly training...")
        logger.info("Training Method: On-the-fly image generation")
        logger.info(f"Training Samples: {args.train_samples_per_epoch} per epoch (unlimited variety)")
        logger.info(f"Validation Samples: {len(val_dataloader.dataset)} (fixed set)")
        logger.info("Target: <=1.0% CER (Character Error Rate)")
        
        # Start training
        trainer.train(resume_from_checkpoint=args.resume)
        
        logger.info("Training completed successfully!")
        
        # Final validation
        logger.info("Running final validation...")
        val_metrics = trainer.validator.validate(val_dataloader)
        logger.info(f"Final Validation Results:")
        logger.info(f"  Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  CER: {val_metrics['cer']:.2%}")
        
        if val_metrics['cer'] <= 0.01:
            logger.info("🎉 SUCCESS: Target CER achieved!")
        else:
            logger.info("Target CER not yet achieved. Consider additional training.")
            
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