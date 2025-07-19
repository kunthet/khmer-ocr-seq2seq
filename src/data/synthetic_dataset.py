"""
Synthetic Image Dataset for Khmer OCR Training

This module provides a PyTorch Dataset that loads pre-generated synthetic images
from the data/synthetic/ folder, organized by train/val/test splits.
"""

import os
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class SyntheticImageDataset(Dataset):
    """
    Dataset for loading pre-generated synthetic images for Khmer OCR training.
    
    This dataset loads images and labels from the organized data/synthetic/ folder
    structure created by the generate_synthetic_images.py script.
    """
    
    def __init__(
        self,
        split: str = "train",
        synthetic_dir: str = "data/synthetic",
        config_manager: Optional[ConfigManager] = None,
        max_samples: Optional[int] = None,
        transform=None
    ):
        """
        Initialize synthetic image dataset
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            synthetic_dir: Directory with pre-generated synthetic images
            config_manager: Configuration manager instance
            max_samples: Maximum number of samples to load (for testing)
            transform: Optional image transform
        """
        self.split = split
        self.synthetic_dir = Path(synthetic_dir)
        self.config = config_manager or ConfigManager()
        self.max_samples = max_samples
        self.transform = transform
        
        # Paths for this split
        self.split_dir = self.synthetic_dir / split
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.metadata_path = self.split_dir / "metadata.json"
        
        # Validate directory structure
        self._validate_structure()
        
        # Load samples
        self.samples = self._load_samples()
        
        # Vocabulary
        self.vocab = self.config.vocab
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split from {synthetic_dir}")
    
    def _validate_structure(self):
        """Validate the synthetic dataset directory structure"""
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        if not self.metadata_path.exists():
            logger.warning(f"Metadata file not found: {self.metadata_path}")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample information from metadata or by scanning directories"""
        samples = []
        
        # Try to load from metadata first
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    raw_samples = metadata.get("samples", [])
                    
                    # Convert metadata format to internal format
                    for sample in raw_samples:
                        if "image_path" not in sample and "filename" in sample:
                            # Create image_path from filename
                            filename = sample["filename"]
                            sample["image_path"] = str(self.images_dir / f"{filename}.png")
                            sample["label_path"] = str(self.labels_dir / f"{filename}.txt")
                    
                    samples = raw_samples
                    logger.info(f"Loaded {len(samples)} samples from metadata")
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
                samples = []
        
        # If no metadata or failed to load, scan directories
        if not samples:
            logger.info("Scanning directories for samples...")
            samples = self._scan_directories()
        
        # Apply max_samples limit
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]
            logger.info(f"Limited to {self.max_samples} samples for {self.split}")
        
        return samples
    
    def _scan_directories(self) -> List[Dict]:
        """Scan directories to find image-label pairs"""
        samples = []
        
        # Get all image files
        image_files = sorted(self.images_dir.glob("*.png"))
        
        for image_path in image_files:
            # Find corresponding label file
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            
            if label_path.exists():
                try:
                    # Read label text
                    with open(label_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    # Create sample info
                    sample = {
                        "filename": image_path.stem,
                        "text": text,
                        "image_path": str(image_path),
                        "label_path": str(label_path)
                    }
                    samples.append(sample)
                    
                except Exception as e:
                    logger.warning(f"Error reading label {label_path}: {e}")
                    continue
            else:
                logger.warning(f"No label found for image: {image_path}")
        
        logger.info(f"Found {len(samples)} image-label pairs")
        return samples
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a training sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, target_tensor, text)
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert('L')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Create a dummy image as fallback
            image = Image.new('L', (100, 32), color=255)
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert image to tensor
        image_tensor = self._image_to_tensor(image)
        
        # Get text
        text = sample["text"]
        
        # Encode text to indices with SOS/EOS tokens
        target_indices = [self.vocab.SOS_IDX] + self.vocab.encode(text) + [self.vocab.EOS_IDX]
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
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a sample"""
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range")
        
        sample = self.samples[idx]
        
        # Add image dimensions if available
        try:
            image = Image.open(sample["image_path"])
            sample["image_width"] = image.width
            sample["image_height"] = image.height
        except:
            pass
        
        return sample
    
    def get_text_sample(self, num_samples: int = 5) -> List[str]:
        """Get a sample of text lines for inspection"""
        import random
        sample_indices = random.sample(range(len(self.samples)), 
                                     min(num_samples, len(self.samples)))
        return [self.samples[i]["text"] for i in sample_indices]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.samples:
            return {"total_samples": 0}
        
        # Basic statistics
        stats = {
            "total_samples": len(self.samples),
            "split": self.split,
            "data_dir": str(self.synthetic_dir)
        }
        
        # Text statistics
        texts = [sample["text"] for sample in self.samples]
        if texts:
            text_lengths = [len(text) for text in texts]
            stats.update({
                "avg_text_length": sum(text_lengths) / len(text_lengths),
                "min_text_length": min(text_lengths),
                "max_text_length": max(text_lengths)
            })
        
        # Try to get image statistics
        try:
            sample_image = Image.open(self.samples[0]["image_path"])
            stats.update({
                "image_height": sample_image.height,
                "image_mode": sample_image.mode
            })
        except:
            pass
        
        return stats


class SyntheticCollateFunction:
    """
    Custom collate function for synthetic dataset with variable-width images
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


def create_synthetic_dataloaders(
    config_manager: ConfigManager,
    batch_size: int = 32,
    num_workers: int = 4,
    synthetic_dir: str = "data/synthetic",
    max_samples: Optional[int] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for train/validation/test sets using pre-generated synthetic images
    
    Args:
        config_manager: Configuration manager
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        synthetic_dir: Directory with pre-generated synthetic images
        max_samples: Maximum samples per split (for testing)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SyntheticImageDataset(
        split="train",
        synthetic_dir=synthetic_dir,
        config_manager=config_manager,
        max_samples=max_samples
    )
    
    val_dataset = SyntheticImageDataset(
        split="val",
        synthetic_dir=synthetic_dir,
        config_manager=config_manager,
        max_samples=max_samples
    )
    
    test_dataset = SyntheticImageDataset(
        split="test",
        synthetic_dir=synthetic_dir,
        config_manager=config_manager,
        max_samples=max_samples
    )
    
    # Create collate function
    collate_fn = SyntheticCollateFunction(config_manager.vocab)
    
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
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Created synthetic data loaders - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def main():
    """Example usage and testing"""
    logging.basicConfig(level=logging.INFO)
    
    # Test synthetic dataset
    config = ConfigManager()
    
    # Create dataset
    dataset = SyntheticImageDataset(
        split="train", 
        synthetic_dir="data/synthetic",
        config_manager=config, 
        max_samples=10
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
        print("No samples found. Please run generate_synthetic_images.py first.")


if __name__ == "__main__":
    main() 