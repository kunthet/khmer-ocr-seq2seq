"""
Corpus-based Dataset for Khmer OCR Training

This module provides a PyTorch Dataset that uses preprocessed real Khmer text
from the corpus instead of synthetic text generation.
"""

import os
import random
import hashlib
import json
from typing import List, Tuple, Optional
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


class ImageCache:
    """
    Image cache for storing and retrieving rendered text images.
    
    Stores images in data/synthetic/ with metadata for consistent results
    and faster training iterations.
    """
    
    def __init__(self, cache_dir: str = "data/synthetic"):
        """
        Initialize image cache
        
        Args:
            cache_dir: Directory to store cached images
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.images_dir = self.cache_dir / "images"
        self.metadata_dir = self.cache_dir / "metadata"
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Load existing cache index
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        logger.info(f"Image cache initialized at {self.cache_dir}")
        logger.info(f"Cached images: {len(self.cache_index)}")
    
    def _load_cache_index(self) -> dict:
        """Load cache index from file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache index: {e}")
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save cache index to file"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache index: {e}")
    
    def _get_cache_key(self, text: str, font_path: str, image_height: int) -> str:
        """
        Generate cache key for text + font + height combination
        
        Args:
            text: Text to render
            font_path: Font file path
            image_height: Target image height
            
        Returns:
            Cache key string
        """
        # Create unique key from text, font, and height
        key_data = f"{text}|{font_path}|{image_height}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def get_cached_image(self, text: str, font_path: str, image_height: int) -> Optional[Image.Image]:
        """
        Retrieve cached image if available
        
        Args:
            text: Text to render
            font_path: Font file path
            image_height: Target image height
            
        Returns:
            PIL Image if cached, None otherwise
        """
        cache_key = self._get_cache_key(text, font_path, image_height)
        
        if cache_key in self.cache_index:
            image_path = self.images_dir / f"{cache_key}.png"
            if image_path.exists():
                try:
                    return Image.open(image_path).convert('L')
                except Exception as e:
                    logger.warning(f"Error loading cached image {image_path}: {e}")
                    # Remove invalid entry from index
                    del self.cache_index[cache_key]
                    self._save_cache_index()
        
        return None
    
    def cache_image(self, text: str, font_path: str, image_height: int, image: Image.Image):
        """
        Cache rendered image with metadata
        
        Args:
            text: Text that was rendered
            font_path: Font file path used
            image_height: Target image height
            image: Rendered PIL Image
        """
        cache_key = self._get_cache_key(text, font_path, image_height)
        
        # Save image
        image_path = self.images_dir / f"{cache_key}.png"
        try:
            image.save(image_path, "PNG")
            
            # Save metadata
            metadata = {
                "text": text,
                "font_path": font_path,
                "image_height": image_height,
                "image_width": image.width,
                "cache_key": cache_key,
                "created_at": str(Path(image_path).stat().st_mtime)
            }
            
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Update index
            self.cache_index[cache_key] = {
                "text": text,
                "font": os.path.basename(font_path),
                "image_path": str(image_path),
                "metadata_path": str(metadata_path)
            }
            
            self._save_cache_index()
            
        except Exception as e:
            logger.warning(f"Error caching image: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        total_images = len(self.cache_index)
        total_size = 0
        
        for cache_key in self.cache_index:
            image_path = self.images_dir / f"{cache_key}.png"
            if image_path.exists():
                total_size += image_path.stat().st_size
        
        return {
            "total_images": total_images,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
    
    def clear_cache(self):
        """Clear all cached images"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)
        
        self.cache_index = {}
        self._save_cache_index()
        logger.info("Image cache cleared")


class KhmerCorpusDataset(Dataset):
    """
    Dataset for Khmer OCR using real corpus text instead of synthetic generation.
    
    This dataset loads preprocessed text lines from the corpus and renders them
    into images with augmentation, providing more realistic training data.
    
    Features image caching for consistent results and faster training.
    """
    
    def __init__(
        self,
        split: str = "train",
        config_manager: Optional[ConfigManager] = None,
        corpus_dir: str = "data/processed",
        augment_prob: float = 0.8,
        max_lines: Optional[int] = None,
        fonts: Optional[List[str]] = None,
        use_cache: bool = True,
        cache_dir: str = "data/synthetic"
    ):
        """
        Initialize corpus dataset
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            config_manager: Configuration manager instance
            corpus_dir: Directory with processed corpus data
            augment_prob: Probability of applying augmentation
            max_lines: Maximum number of lines to use (for testing)
            fonts: List of font paths to use for rendering
            use_cache: Whether to use image caching
            cache_dir: Directory for image cache
        """
        self.split = split
        self.config = config_manager or ConfigManager()
        self.corpus_dir = Path(corpus_dir)
        self.max_lines = max_lines
        self.use_cache = use_cache
        
        # Initialize image cache
        if self.use_cache:
            self.image_cache = ImageCache(cache_dir)
        else:
            self.image_cache = None
        
        # Load text lines
        self.text_lines = self._load_text_lines()
        logger.info(f"Loaded {len(self.text_lines)} lines for {split} split")
        
        # Initialize text renderer
        if fonts is None:
            fonts = self._get_default_fonts()
        
        self.text_renderer = TextRenderer(
            fonts=fonts,
            image_height=self.config.data_config.image_height
        )
        
        # Initialize augmentation
        augment_enabled = split == "train"  # Only augment training data
        if augment_enabled:
            self.augmentation = DataAugmentation(
                blur_prob=augment_prob,
                morph_prob=augment_prob,
                noise_prob=augment_prob,
                concat_prob=0.0  # Disable concatenation for single text rendering
            )
        else:
            self.augmentation = DataAugmentation(
                blur_prob=0.0,
                morph_prob=0.0,
                noise_prob=0.0,
                concat_prob=0.0
            )
        
        # Vocabulary
        self.vocab = self.config.vocab
        
        # Cache statistics
        if self.use_cache:
            cache_stats = self.image_cache.get_cache_stats()
            logger.info(f"Image cache: {cache_stats['total_images']} images, "
                       f"{cache_stats['total_size_mb']:.1f}MB")
        
        logger.info(f"Initialized {split} dataset with {len(self.text_lines)} samples")
    
    def _load_text_lines(self) -> List[str]:
        """Load text lines from processed corpus"""
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
                            file_lines = [line.strip() for line in f.readlines() if line.strip()]
                            lines.extend(file_lines)
                            logger.info(f"Loaded {len(file_lines)} lines from {train_file.name}")
                    except Exception as e:
                        logger.warning(f"Error loading {train_file}: {e}")
                        continue
            else:
                # Fallback to single train.txt file
                split_file = self.corpus_dir / f"{self.split}.txt"
                if split_file.exists():
                    with open(split_file, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    logger.info(f"Loaded {len(lines)} lines from {split_file.name}")
                else:
                    logger.error(f"No training corpus files found. Expected {self.corpus_dir}/train_*.txt or {split_file}")
                    logger.info("Processing corpus data first...")
                    
                    # Automatically process corpus if not found
                    processor = KhmerCorpusProcessor()
                    processor.process_corpus(str(self.corpus_dir))
                    
                    # Try loading again
                    if split_file.exists():
                        with open(split_file, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # For val/test splits, use single file
            split_file = self.corpus_dir / f"{self.split}.txt"
            
            if not split_file.exists():
                logger.error(f"Split file not found: {split_file}")
                logger.info("Processing corpus data first...")
                
                # Automatically process corpus if not found
                processor = KhmerCorpusProcessor()
                processor.process_corpus(str(self.corpus_dir))
            
            # Load lines
            try:
                with open(split_file, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Loaded {len(lines)} lines from {split_file.name}")
            except Exception as e:
                logger.error(f"Error loading {split_file}: {e}")
                return []
        
        if not lines:
            logger.warning(f"No text lines loaded for {self.split} split")
            return []
        
        # Apply max_lines limit if specified
        if self.max_lines and len(lines) > self.max_lines:
            lines = lines[:self.max_lines]
            logger.info(f"Limited to {self.max_lines} lines for {self.split}")
        
        return lines
    
    def _get_default_fonts(self) -> List[str]:
        """Get default Khmer fonts for text rendering"""
        # Use actual font files from fonts/ directory
        font_dir = "fonts"
        default_fonts = [
            "KhmerOS.ttf",
            "KhmerOSsiemreap.ttf",
            "KhmerOSbattambang.ttf",
            "KhmerOSbokor.ttf",
            "KhmerOSmuol.ttf",
            "KhmerOSmuollight.ttf",
            "KhmerOSmetalchrieng.ttf",
            "KhmerOSfasthand.ttf"
        ]
        
        # Get full paths and filter to existing fonts
        font_paths = []
        for font_name in default_fonts:
            font_path = os.path.join(font_dir, font_name)
            if os.path.exists(font_path):
                font_paths.append(font_path)
        
        if not font_paths:
            logger.warning(f"No Khmer fonts found in {font_dir}/ directory")
            logger.warning("Please ensure Khmer fonts are installed in the fonts/ directory")
            # Fallback to system font names
            font_paths = [
                "Khmer OS",
                "Khmer OS Siemreap",
                "Khmer OS Battambang", 
                "Khmer OS Muol Light",
                "Hanuman",
                "Nokora"
            ]
        else:
            logger.info(f"Found {len(font_paths)} Khmer fonts in {font_dir}/ directory")
        
        return font_paths
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.text_lines)
    
    def _render_text_with_cache(self, text: str, font_path: str) -> Image.Image:
        """
        Render text with caching support
        
        Args:
            text: Text to render
            font_path: Font file path
            
        Returns:
            Rendered PIL Image
        """
        # Try to get from cache first
        if self.use_cache and self.image_cache:
            cached_image = self.image_cache.get_cached_image(
                text, font_path, self.config.data_config.image_height
            )
            if cached_image is not None:
                return cached_image
        
        # Render new image
        try:
            image = self.text_renderer.render_text(text, font_path)
        except Exception as e:
            logger.warning(f"Error rendering text '{text}' with font '{font_path}': {e}")
            # Fallback to simple text
            fallback_text = "មុខ"  # Simple Khmer word
            image = self.text_renderer.render_text(fallback_text, font_path)
        
        # Cache the image if caching is enabled
        if self.use_cache and self.image_cache:
            self.image_cache.cache_image(
                text, font_path, self.config.data_config.image_height, image
            )
        
        return image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a training sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, target_tensor, text)
        """
        # Get text line
        text = self.text_lines[idx]
        
        # For validation/test sets, use consistent font selection based on text hash
        # For training, use random font selection
        if self.split == "train":
            font_path = random.choice(self.text_renderer.fonts)
        else:
            # Use hash of text to select font consistently
            font_idx = hash(text) % len(self.text_renderer.fonts)
            font_path = self.text_renderer.fonts[font_idx]
        
        # Render text to image with caching
        image = self._render_text_with_cache(text, font_path)
        
        # Apply augmentation (only for training data)
        if self.split == "train":
            image = self.augmentation.apply_augmentations(image)
        
        # Convert image to tensor
        image_tensor = self._image_to_tensor(image)
        
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
    
    def get_text_sample(self, num_samples: int = 5) -> List[str]:
        """Get a sample of text lines for inspection"""
        sample_indices = random.sample(range(len(self.text_lines)), 
                                     min(num_samples, len(self.text_lines)))
        return [self.text_lines[i] for i in sample_indices]
    
    def get_character_distribution(self) -> dict:
        """Get character frequency distribution in the dataset"""
        char_count = {}
        
        for text in self.text_lines:
            for char in text:
                char_count[char] = char_count.get(char, 0) + 1
        
        # Sort by frequency
        sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_chars)
    
    def get_cache_info(self) -> dict:
        """Get cache information"""
        if self.use_cache and self.image_cache:
            return self.image_cache.get_cache_stats()
        return {"cache_enabled": False}
    
    def clear_cache(self):
        """Clear image cache"""
        if self.use_cache and self.image_cache:
            self.image_cache.clear_cache()
            logger.info("Image cache cleared")


class CorpusCollateFunction:
    """
    Custom collate function for corpus dataset with variable-width images
    """
    
    def __init__(self, vocab):
        self.vocab = vocab
    
    def __call__(self, batch) -> dict:
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


def create_corpus_dataloaders(
    config_manager: ConfigManager,
    batch_size: int = 32,
    num_workers: int = 4,
    corpus_dir: str = "data/processed",
    max_lines: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: str = "data/synthetic"
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for train/validation/test sets using corpus data
    
    Args:
        config_manager: Configuration manager
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        corpus_dir: Directory with processed corpus data
        max_lines: Maximum lines per split (for testing)
        use_cache: Whether to use image caching
        cache_dir: Directory for image cache
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = KhmerCorpusDataset(
        split="train",
        config_manager=config_manager,
        corpus_dir=corpus_dir,
        augment_prob=0.8,
        max_lines=max_lines,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    val_dataset = KhmerCorpusDataset(
        split="val", 
        config_manager=config_manager,
        corpus_dir=corpus_dir,
        augment_prob=0.0,  # No augmentation for validation
        max_lines=max_lines,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    test_dataset = KhmerCorpusDataset(
        split="test",
        config_manager=config_manager, 
        corpus_dir=corpus_dir,
        augment_prob=0.0,  # No augmentation for test
        max_lines=max_lines,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    # Create collate function
    collate_fn = CorpusCollateFunction(config_manager.vocab)
    
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
    
    logger.info(f"Created data loaders - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def main():
    """Example usage and testing"""
    logging.basicConfig(level=logging.INFO)
    
    # Test corpus dataset
    config = ConfigManager()
    
    # Create dataset with caching
    dataset = KhmerCorpusDataset(
        split="train", 
        config_manager=config, 
        max_lines=100,
        use_cache=True
    )
    
    print(f"Dataset size: {len(dataset)}")
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
    print(f"Decoded: '{config.vocab.decode_sequence(sample_target.tolist())}'")
    
    # Test cache functionality
    print("\nCache information:")
    cache_info = dataset.get_cache_info()
    print(f"Cache stats: {cache_info}")
    
    # Test loading same sample again (should be cached)
    print("\nTesting cache retrieval...")
    sample_image2, _, _ = dataset[0]
    print(f"Same sample loaded again: {torch.equal(sample_image, sample_image2)}")


if __name__ == "__main__":
    main() 