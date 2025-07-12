"""
On-the-Fly Dataset for Khmer OCR Training

This module provides a PyTorch Dataset that generates synthetic images dynamically
during training instead of loading pre-generated images from disk. This approach
reduces storage requirements and allows for unlimited data augmentation variety.
"""

import os
import random
import hashlib
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

from .text_renderer import TextRenderer
from .augmentation import DataAugmentation
from .corpus_processor import KhmerCorpusProcessor
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class OnTheFlyDataset(Dataset):
    """
    Dataset that generates synthetic images on-the-fly during training.
    
    This dataset:
    - Loads text lines from corpus
    - Generates images dynamically per batch
    - Applies real-time augmentation
    - Reduces storage requirements
    - Provides unlimited augmentation variety
    """
    
    def __init__(
        self,
        split: str = "train",
        config_manager: Optional[ConfigManager] = None,
        corpus_dir: str = "data/processed",
        samples_per_epoch: int = 10000,
        augment_prob: float = 0.8,
        fonts: Optional[List[str]] = None,
        shuffle_texts: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize on-the-fly dataset
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            config_manager: Configuration manager instance
            corpus_dir: Directory with processed corpus data
            samples_per_epoch: Number of samples to generate per epoch
            augment_prob: Probability of applying augmentation (0.0-1.0)
            fonts: List of font paths to use for rendering
            shuffle_texts: Whether to shuffle text lines for variety
            random_seed: Random seed for reproducibility
        """
        self.split = split
        self.config = config_manager or ConfigManager()
        self.corpus_dir = Path(corpus_dir)
        self.samples_per_epoch = samples_per_epoch
        self.augment_prob = augment_prob
        self.shuffle_texts = shuffle_texts
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        # Load text lines from corpus
        self.text_lines = self._load_text_lines()
        logger.info(f"Loaded {len(self.text_lines)} text lines for {split} split")
        
        # Initialize text renderer
        if fonts is None:
            fonts = self._get_default_fonts()
        
        self.text_renderer = TextRenderer(
            fonts=fonts,
            image_height=self.config.data_config.image_height
        )
        
        # Initialize augmentation
        self.use_augmentation = (split == "train" and augment_prob > 0.0)
        if self.use_augmentation:
            self.augmentation = DataAugmentation(
                blur_prob=augment_prob,
                morph_prob=augment_prob,
                noise_prob=augment_prob,
                concat_prob=0.0  # Disable concatenation for single text rendering
            )
        else:
            self.augmentation = None
        
        # Vocabulary
        self.vocab = self.config.vocab
        
        logger.info(f"Initialized on-the-fly {split} dataset:")
        logger.info(f"  Text lines: {len(self.text_lines)}")
        logger.info(f"  Samples per epoch: {samples_per_epoch}")
        logger.info(f"  Augmentation: {'enabled' if self.use_augmentation else 'disabled'}")
        logger.info(f"  Fonts: {len(self.text_renderer.fonts)}")
        
    def _load_text_lines(self) -> List[str]:
        """Load text lines from corpus files"""
        lines = []
        
        # For training split, check for multiple files (train_0.txt, train_1.txt, etc.)
        if self.split == "train":
            # Look for train_*.txt files
            train_files = sorted(self.corpus_dir.glob(f"{self.split}_*.txt"))
            
            if train_files:
                # Load from multiple training files
                logger.info(f"Found {len(train_files)} training files: {[f.name for f in train_files]}")
                
                for train_file in train_files:
                    try:
                        with open(train_file, 'r', encoding='utf-8') as f:
                            file_lines = [line.strip() for line in f if line.strip()]
                            lines.extend(file_lines)
                            logger.info(f"Loaded {len(file_lines)} lines from {train_file.name}")
                    except Exception as e:
                        logger.warning(f"Error loading {train_file}: {e}")
                        continue
            else:
                # Fallback to single train.txt file
                corpus_file = self.corpus_dir / f"{self.split}.txt"
                if corpus_file.exists():
                    with open(corpus_file, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(lines)} lines from {corpus_file.name}")
                else:
                    raise FileNotFoundError(f"No training corpus files found. Expected {self.corpus_dir}/train_*.txt or {corpus_file}")
        else:
            # For val/test splits, use single file
            corpus_file = self.corpus_dir / f"{self.split}.txt"
            
            if not corpus_file.exists():
                raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
            
            with open(corpus_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(lines)} lines from {corpus_file.name}")
        
        if not lines:
            raise ValueError(f"No text lines loaded for {self.split} split")
        
        # Shuffle for variety if enabled
        if self.shuffle_texts:
            random.shuffle(lines)
        
        return lines
    
    def _get_default_fonts(self) -> List[str]:
        """Get default Khmer fonts"""
        fonts_dir = Path("fonts")
        if not fonts_dir.exists():
            raise FileNotFoundError(f"Fonts directory not found: {fonts_dir}")
        
        font_files = list(fonts_dir.glob("*.ttf"))
        if not font_files:
            raise FileNotFoundError("No TTF fonts found in fonts directory")
        
        return [str(font) for font in font_files]
    
    def __len__(self) -> int:
        """Return dataset size (samples per epoch)"""
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Generate a training sample on-the-fly
        
        Args:
            idx: Sample index (used for text selection)
            
        Returns:
            Tuple of (image_tensor, target_tensor, text)
        """
        # Select text line (cycle through available lines)
        text_idx = idx % len(self.text_lines)
        text = self.text_lines[text_idx]
        
        # For training, add some randomness to text selection
        if self.split == "train" and random.random() < 0.3:
            text = random.choice(self.text_lines)
        
        # Select font
        if self.split == "train":
            # Random font selection for training (data augmentation)
            font_path = random.choice(self.text_renderer.fonts)
        else:
            # Consistent font selection for val/test based on text hash
            font_idx = hash(text + str(idx)) % len(self.text_renderer.fonts)
            font_path = self.text_renderer.fonts[font_idx]
        
        # Render text to image
        try:
            image = self.text_renderer.render_text(text, font_path)
        except Exception as e:
            logger.warning(f"Error rendering text '{text}': {e}")
            # Fallback to simple text if rendering fails
            fallback_text = "ខ្មែរ"  # Simple Khmer text
            image = self.text_renderer.render_text(fallback_text, font_path)
            text = fallback_text
        
        # Apply augmentation if enabled
        if self.use_augmentation and self.augmentation:
            try:
                image = self.augmentation.apply_augmentations(image)
            except Exception as e:
                logger.warning(f"Error applying augmentation: {e}")
                # Continue with original image if augmentation fails
        
        # Convert image to tensor
        image_tensor = self._image_to_tensor(image)
        
        # Encode text to indices with SOS/EOS tokens
        try:
            target_indices = [self.vocab.SOS_IDX] + self.vocab.encode(text) + [self.vocab.EOS_IDX]
        except Exception as e:
            logger.warning(f"Error encoding text '{text}': {e}")
            # Fallback encoding
            target_indices = [self.vocab.SOS_IDX, self.vocab.EOS_IDX]
        
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        
        return image_tensor, target_tensor, text
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor"""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to tensor and normalize
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        
        return transform(image)
    
    def get_text_sample(self, num_samples: int = 5) -> List[str]:
        """Get a sample of text lines for inspection"""
        sample_indices = random.sample(range(len(self.text_lines)), 
                                     min(num_samples, len(self.text_lines)))
        return [self.text_lines[i] for i in sample_indices]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics"""
        text_lengths = [len(text) for text in self.text_lines]
        
        return {
            "total_text_lines": len(self.text_lines),
            "samples_per_epoch": self.samples_per_epoch,
            "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "split": self.split,
            "augmentation_enabled": self.use_augmentation,
            "fonts_available": len(self.text_renderer.fonts)
        }


class OnTheFlyCollateFunction:
    """
    Custom collate function for on-the-fly dataset with variable-width images
    """
    
    def __init__(self, vocab):
        self.vocab = vocab
    
    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        """
        Collate batch samples
        
        Args:
            batch: List of (image_tensor, target_tensor, text) tuples
            
        Returns:
            Dict with batched tensors
        """
        images, targets, texts = zip(*batch)
        
        # Pad images to same width
        max_width = max(img.shape[2] for img in images)
        batch_size = len(images)
        height = images[0].shape[1]
        
        # Create padded image tensor
        padded_images = torch.zeros(batch_size, 1, height, max_width)
        image_lengths = []
        
        for i, img in enumerate(images):
            width = img.shape[2]
            padded_images[i, :, :, :width] = img
            image_lengths.append(width)
        
        # Pad target sequences
        max_target_len = max(len(target) for target in targets)
        padded_targets = torch.full(
            (batch_size, max_target_len), 
            self.vocab.PAD_IDX, 
            dtype=torch.long
        )
        target_lengths = []
        
        for i, target in enumerate(targets):
            target_len = len(target)
            padded_targets[i, :target_len] = target
            target_lengths.append(target_len)
        
        return {
            'images': padded_images,
            'targets': padded_targets,
            'image_lengths': torch.tensor(image_lengths),
            'target_lengths': torch.tensor(target_lengths),
            'texts': texts
        }


def create_onthefly_dataloaders(
    config_manager: ConfigManager,
    batch_size: int = 32,
    num_workers: int = 4,
    corpus_dir: str = "data/processed",
    train_samples_per_epoch: int = 10000,
    val_dataset_path: Optional[str] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders with on-the-fly training generation and fixed validation set
    
    Args:
        config_manager: Configuration manager
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        corpus_dir: Directory with processed corpus data
        train_samples_per_epoch: Number of training samples per epoch
        val_dataset_path: Path to pre-generated validation dataset (if None, creates on-the-fly)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create on-the-fly training dataset
    train_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config_manager,
        corpus_dir=corpus_dir,
        samples_per_epoch=train_samples_per_epoch,
        augment_prob=0.8,
        shuffle_texts=True,
        random_seed=None  # No seed for training (max variety)
    )
    
    # Create validation dataset
    if val_dataset_path and Path(val_dataset_path).exists():
        # Use pre-generated validation set
        from .synthetic_dataset import SyntheticImageDataset
        val_dataset = SyntheticImageDataset(
            split="val",
            synthetic_dir=val_dataset_path,
            config_manager=config_manager,
            max_samples=6400  # Fixed validation size
        )
    else:
        # Fallback to on-the-fly validation (not recommended for reproducibility)
        logger.warning("No pre-generated validation set found, using on-the-fly generation")
        val_dataset = OnTheFlyDataset(
            split="val",
            config_manager=config_manager,
            corpus_dir=corpus_dir,
            samples_per_epoch=6400,  # Fixed validation size
            augment_prob=0.0,  # No augmentation for validation
            shuffle_texts=False,
            random_seed=42  # Fixed seed for validation reproducibility
        )
    
    # Create collate function
    collate_fn = OnTheFlyCollateFunction(config_manager.vocab)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Created on-the-fly data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples per epoch")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def main():
    """Example usage and testing"""
    logging.basicConfig(level=logging.INFO)
    
    # Test on-the-fly dataset
    config = ConfigManager()
    
    # Create dataset
    dataset = OnTheFlyDataset(
        split="train", 
        config_manager=config,
        samples_per_epoch=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        print("Sample texts:")
        for text in dataset.get_text_sample(5):
            print(f"  '{text}'")
        
        # Test data loading
        print("\nTesting data loading...")
        sample_image, sample_target, sample_text = dataset[0]
        print(f"Image shape: {sample_image.shape}")
        print(f"Target shape: {sample_target.shape}")
        print(f"Text: '{sample_text}'")
        print(f"Encoded: {sample_target.tolist()}")
        
        # Test statistics
        print("\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("No samples found. Please check corpus files.")


if __name__ == "__main__":
    main() 