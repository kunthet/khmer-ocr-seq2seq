"""
PyTorch Dataset for synthetic Khmer OCR data generation.
Integrates vocabulary, text rendering, and data augmentation.
"""
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Import our components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import KhmerVocab
from data.text_renderer import TextRenderer  
from data.augmentation import DataAugmentation


class KhmerDataset(Dataset):
    """
    PyTorch Dataset for generating synthetic Khmer text images.
    
    This dataset:
    1. Generates random Khmer text sequences
    2. Renders them to images using TextRenderer
    3. Applies data augmentation
    4. Converts to PyTorch tensors
    5. Handles variable-width images and sequence padding
    """
    
    def __init__(
        self,
        vocab: KhmerVocab = None,
        text_renderer: TextRenderer = None,
        augmentation: DataAugmentation = None,
        dataset_size: int = 10000,
        min_text_length: int = 1,
        max_text_length: int = 20,
        image_height: int = 32,
        max_width: int = 512,
        use_augmentation: bool = True,
        seed: int = None
    ):
        """
        Initialize Khmer dataset.
        
        Args:
            vocab: KhmerVocab instance for character encoding/decoding
            text_renderer: TextRenderer for converting text to images
            augmentation: DataAugmentation for image augmentation
            dataset_size: Number of samples per epoch
            min_text_length: Minimum characters in generated text
            max_text_length: Maximum characters in generated text
            image_height: Fixed height for all images (32px)
            max_width: Maximum width for images (for padding)
            use_augmentation: Whether to apply augmentations
            seed: Random seed for reproducible generation
        """
        # Initialize components
        self.vocab = vocab or KhmerVocab()
        self.text_renderer = text_renderer or TextRenderer(image_height=image_height)
        self.augmentation = augmentation or DataAugmentation()
        
        # Dataset parameters
        self.dataset_size = dataset_size
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.image_height = image_height
        self.max_width = max_width
        self.use_augmentation = use_augmentation
        
        # Set seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.augmentation:
                self.augmentation.set_seed(seed)
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
            # Note: We'll normalize later in the training loop if needed
        ])
        
        # Get character list for random text generation (excluding special tokens)
        self.generation_chars = (
            self.vocab.khmer_numbers + 
            self.vocab.arabic_numbers + 
            self.vocab.consonants + 
            self.vocab.independent_vowels + 
            self.vocab.dependent_vowels + 
            self.vocab.subscript + 
            self.vocab.diacritics + 
            [s for s in self.vocab.symbols if s not in [' ']]  # Exclude space for now
        )
        
        print(f"KhmerDataset initialized:")
        print(f"  - Dataset size: {self.dataset_size}")
        print(f"  - Text length: {self.min_text_length}-{self.max_text_length}")
        print(f"  - Image size: {self.image_height}px height, max {self.max_width}px width")
        print(f"  - Generation characters: {len(self.generation_chars)}")
        print(f"  - Augmentation enabled: {self.use_augmentation}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size
    
    def generate_random_text(self) -> str:
        """
        Generate random Khmer text sequence.
        
        Returns:
            Random text string
        """
        # Random length
        length = random.randint(self.min_text_length, self.max_text_length)
        
        # Generate random characters
        chars = random.choices(self.generation_chars, k=length)
        
        return ''.join(chars)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a single training sample.
        
        Args:
            idx: Sample index (not used since we generate randomly)
            
        Returns:
            Dictionary containing:
            - 'image': Tensor of shape (1, H, W) 
            - 'text': Original text string
            - 'target': Encoded text as tensor
            - 'target_length': Length of target sequence
        """
        # Generate random text
        text = self.generate_random_text()
        
        # Render text to image
        image = self.text_renderer.render_text(text)
        
        # Apply augmentation if enabled
        if self.use_augmentation and self.augmentation:
            image = self.augmentation.apply_augmentations(image)
        
        # Ensure image has correct height
        if image.height != self.image_height:
            aspect_ratio = image.width / image.height
            new_width = int(self.image_height * aspect_ratio)
            image = image.resize((new_width, self.image_height), Image.Resampling.LANCZOS)
        
        # Clip width if too large
        if image.width > self.max_width:
            image = image.crop((0, 0, self.max_width, self.image_height))
        
        # Convert to tensor
        image_tensor = self.image_transform(image)  # Shape: (1, H, W)
        
        # Encode target text
        target_sequence = self.vocab.encode(text)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'text': text,
            'target': target_tensor,
            'target_length': len(target_sequence),
            'image_width': image.width
        }
    
    def get_sample_batch(self, batch_size: int = 4) -> List[Dict]:
        """
        Generate a sample batch for testing.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        for i in range(batch_size):
            sample = self[i]
            samples.append(sample)
        return samples


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-width images and sequences.
    
    Args:
        batch: List of sample dictionaries from KhmerDataset
        
    Returns:
        Batched dictionary with padded tensors
    """
    # Separate components
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    targets = [item['target'] for item in batch]
    target_lengths = [item['target_length'] for item in batch]
    image_widths = [item['image_width'] for item in batch]
    
    # Find maximum dimensions
    max_width = max(img.shape[2] for img in images)  # Shape is (1, H, W)
    max_target_length = max(len(target) for target in targets)
    
    # Pad images to same width
    padded_images = []
    for img in images:
        # Pad width dimension
        pad_width = max_width - img.shape[2]
        if pad_width > 0:
            # Pad with white (value 1.0 since image is normalized)
            padding = torch.ones(1, img.shape[1], pad_width)
            padded_img = torch.cat([img, padding], dim=2)
        else:
            padded_img = img
        padded_images.append(padded_img)
    
    # Stack images
    image_batch = torch.stack(padded_images)  # Shape: (B, 1, H, W)
    
    # Pad target sequences
    padded_targets = []
    for target in targets:
        pad_length = max_target_length - len(target)
        if pad_length > 0:
            # Pad with PAD token (index 2)
            padded_target = torch.cat([
                target, 
                torch.full((pad_length,), 2, dtype=torch.long)  # PAD_IDX = 2
            ])
        else:
            padded_target = target
        padded_targets.append(padded_target)
    
    # Stack targets
    target_batch = torch.stack(padded_targets)  # Shape: (B, max_length)
    
    return {
        'images': image_batch,
        'texts': texts,
        'targets': target_batch,
        'target_lengths': torch.tensor(target_lengths, dtype=torch.long),
        'image_widths': torch.tensor(image_widths, dtype=torch.long)
    }


class KhmerDataLoader:
    """
    Convenience wrapper for KhmerDataset with DataLoader.
    """
    
    def __init__(
        self,
        dataset: KhmerDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        """
        Initialize data loader.
        
        Args:
            dataset: KhmerDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle samples
            num_workers: Number of worker processes
            pin_memory: Whether to use pinned memory
        """
        from torch.utils.data import DataLoader
        
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Number of batches."""
        return len(self.dataloader)
    
    def get_sample_batch(self):
        """Get a single sample batch."""
        return next(iter(self.dataloader)) 