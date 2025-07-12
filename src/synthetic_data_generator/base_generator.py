"""
Base generator class for Khmer OCR data generation.
"""

import os
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .utils import (
    load_khmer_fonts, validate_font_collection, normalize_khmer_text, 
    create_character_mapping, KhmerTextGenerator
)
from .backgrounds import BackgroundGenerator
from .augmentation import ImageAugmentor


class BaseDataGenerator:
    """
    Base class for Khmer OCR data generators with common functionality.
    """
    
    def __init__(self, 
                 config_path: str,
                 fonts_dir: str,
                 output_dir: str,
                 mode: str = "full_text"):
        """
        Initialize the base data generator.
        
        Args:
            config_path: Path to model configuration file
            fonts_dir: Directory containing Khmer fonts
            output_dir: Directory to save generated data
            mode: Generation mode ('digits', 'full_text', 'mixed')
        """
        self.config_path = config_path
        self.fonts_dir = fonts_dir
        self.output_dir = output_dir
        self.mode = mode
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.image_size = tuple(self.config['model']['input']['image_size'])
        self.max_sequence_length = self.config['model']['characters'].get('max_sequence_length', 20)
        
        self.background_generator = BackgroundGenerator(self.image_size)
        self.augmentor = ImageAugmentor(
            rotation_range=tuple(self.config['data']['augmentation']['rotation']),
            scale_range=tuple(self.config['data']['augmentation']['scaling']),
            noise_std=self.config['data']['augmentation']['noise']['gaussian_std'],
            brightness_range=tuple(self.config['data']['augmentation']['brightness']),
            contrast_range=tuple(self.config['data']['augmentation']['contrast'])
        )
        
        # Load and validate fonts
        self.fonts = load_khmer_fonts(fonts_dir)
        self.font_validation = validate_font_collection(fonts_dir)
        
        # Filter to only working fonts
        self.working_fonts = {
            name: path for name, path in self.fonts.items() 
            if self.font_validation[name]
        }
        
        if not self.working_fonts:
            raise ValueError("No working Khmer fonts found!")
        
        print(f"Loaded {len(self.working_fonts)} working fonts: {list(self.working_fonts.keys())}")
        
        # Create character mappings
        use_full_khmer = (mode in ["full_text", "mixed"])
        self.char_to_idx, self.idx_to_char = create_character_mapping(use_full_khmer)
        
        # Initialize Khmer text generator with proper linguistic rules
        self.khmer_generator = KhmerTextGenerator()
        self.character_frequencies = self.khmer_generator.character_frequencies
        self.khmer_characters = self.khmer_generator.khmer_chars
        
        print(f"Character vocabulary size: {len(self.char_to_idx)}")
        print(f"Generation mode: {mode}")
        print(f"Loaded frequencies for {len(self.character_frequencies)} characters")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _generate_text_content(self, 
                              content_type: str = "auto",
                              length_range: Tuple[int, int] = (1, 15),
                              allowed_characters: Optional[List[str]] = None) -> str:
        """
        Generate text content. Must be implemented by subclasses.
        
        Args:
            content_type: Type of content to generate
            length_range: Range of text lengths
            allowed_characters: Allowed characters for curriculum learning
            
        Returns:
            Generated text content
        """
        raise NotImplementedError("Subclasses must implement _generate_text_content")
    
    def _get_adaptive_font_size(self, text: str) -> int:
        """Get a font size that fits the text within image boundaries."""
        # Start with a reasonable base size
        base_size = int(self.image_size[1] * 0.6)
        
        # Adjust based on text length - longer sequences need smaller fonts
        length_factor = max(0.3, 1.0 - (len(text) - 1) * 0.05)
        target_size = int(base_size * length_factor)
        
        # Test different font sizes to ensure text fits
        for font_size in range(target_size, 12, -2):  # Minimum font size of 12
            # Test with a representative font
            test_font_path = list(self.working_fonts.values())[0]
            test_font = ImageFont.truetype(test_font_path, font_size)
            
            bbox = self._get_text_bbox(text, test_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Check if text fits with some margin (leave 20% margin on each side)
            margin_width = self.image_size[0] * 0.1
            margin_height = self.image_size[1] * 0.1
            
            if (text_width <= self.image_size[0] - 2 * margin_width and 
                text_height <= self.image_size[1] - 2 * margin_height):
                return font_size
        
        # Fallback to minimum size
        return 12
    
    def _get_text_bbox(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
        """Get tight bounding box for text."""
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1000, 1000))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        return bbox
    
    def _safe_position_text(self, text: str, font: ImageFont.FreeTypeFont, 
                           image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate safe position for text ensuring it fits within image bounds."""
        bbox = self._get_text_bbox(text, font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Define safe margins (10% of image size)
        margin_x = int(image_size[0] * 0.1)
        margin_y = int(image_size[1] * 0.1)
        
        # Calculate available space for text placement
        available_width = image_size[0] - 2 * margin_x
        available_height = image_size[1] - 2 * margin_y
        
        # Center text within available space
        x = margin_x + (available_width - text_width) // 2
        y = margin_y + (available_height - text_height) // 2
        
        # Adjust for bbox offset
        x -= bbox[0]
        y -= bbox[1]
        
        # Ensure text doesn't go outside bounds
        x = max(margin_x - bbox[0], min(x, image_size[0] - margin_x - text_width - bbox[0]))
        y = max(margin_y - bbox[1], min(y, image_size[1] - margin_y - text_height - bbox[1]))
        
        return x, y
    
    def _validate_text_fits(self, text: str, font: ImageFont.FreeTypeFont, 
                           position: Tuple[int, int], image_size: Tuple[int, int]) -> bool:
        """Validate that text fits completely within image boundaries."""
        x, y = position
        bbox = self._get_text_bbox(text, font)
        
        # Calculate actual text boundaries in the image
        text_left = x + bbox[0]
        text_top = y + bbox[1]
        text_right = x + bbox[2]
        text_bottom = y + bbox[3]
        
        # Check if text is within image bounds
        return (text_left >= 0 and text_top >= 0 and 
                text_right <= image_size[0] and text_bottom <= image_size[1])
    
    def generate_single_image(self, 
                             text: Optional[str] = None,
                             font_name: Optional[str] = None,
                             content_type: str = "auto",
                             apply_augmentation: bool = True) -> Tuple[Image.Image, Dict]:
        """
        Generate a single synthetic image with text.
        
        Args:
            text: Text to render, if None generates based on mode
            font_name: Font to use, if None chooses random font
            content_type: Type of content to generate ('auto', 'digits', 'characters', 'syllables', 'words', 'phrases', 'mixed')
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Tuple of (image, metadata)
        """
        # Generate text if not provided
        if text is None:
            text = self._generate_text_content(content_type)
        text = normalize_khmer_text(text)
        
        # Choose font
        if font_name is None:
            font_name = random.choice(list(self.working_fonts.keys()))
        font_path = self.working_fonts[font_name]
        
        # Generate background
        background = self.background_generator.generate_random_background()
        
        # Get optimal text color for this background
        text_color = self.background_generator.get_optimal_text_color(background)
        
        # Create font object with adaptive size
        font_size = self._get_adaptive_font_size(text)
        font = ImageFont.truetype(font_path, font_size)
        
        # Calculate safe text position
        text_position = self._safe_position_text(text, font, self.image_size)
        
        # Create image with text
        image = background.copy()
        draw = ImageDraw.Draw(image)
        
        # Draw text
        draw.text(text_position, text, font=font, fill=text_color)
        
        # Validate text positioning
        if not self._validate_text_fits(text, font, text_position, self.image_size):
            print(f"Warning: Text may be cropped - '{text}' with font size {font_size}")
        
        # Apply augmentation if requested
        augmentation_params = {}
        if apply_augmentation:
            image, augmentation_params = self.augmentor.apply_random_augmentation(image)
        
        # Create metadata
        metadata = {
            'label': text,
            'font': font_name,
            'font_size': font_size,
            'text_position': text_position,
            'text_color': text_color,
            'image_size': self.image_size,
            'character_count': len(text),
            'content_type': content_type if content_type != "auto" else self._classify_content_type(text),
            'augmentation': augmentation_params
        }
        
        return image, metadata
    
    def _classify_content_type(self, text: str) -> str:
        """Classify the type of generated content."""
        digits = set("០១២៣៤៥៦៧៨៩")
        khmer_chars = set()
        for category_chars in self.khmer_characters.values():
            khmer_chars.update(category_chars)
        
        text_chars = set(text)
        
        if text_chars.issubset(digits):
            return "digits"
        elif len(text) <= 3:
            return "characters"
        elif len(text) <= 8:
            return "syllables"
        elif len(text) <= 15:
            return "words"
        else:
            return "phrases"
    
    def generate_dataset(self, 
                        num_samples: int,
                        train_split: float = 0.8,
                        save_images: bool = True,
                        show_progress: bool = True) -> Dict:
        """
        Generate a complete dataset.
        
        Args:
            num_samples: Total number of samples to generate
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with dataset information
        """
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        train_dir = Path(self.output_dir) / 'train'
        val_dir = Path(self.output_dir) / 'val'
        
        if save_images:
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)
        
        # Generate samples
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []}
        }
        
        # Progress bar setup
        total_progress = tqdm(total=num_samples, desc="Generating dataset") if show_progress else None
        
        # Generate training samples
        for i in range(num_train):
            image, metadata = self.generate_single_image()
            
            if save_images:
                image_filename = f"train_{i:06d}.png"
                image_path = train_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            all_metadata['train']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        # Generate validation samples
        for i in range(num_val):
            image, metadata = self.generate_single_image()
            
            if save_images:
                image_filename = f"val_{i:06d}.png"
                image_path = val_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            all_metadata['val']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        if total_progress:
            total_progress.close()
        
        # Add dataset-level metadata
        dataset_info = {
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'image_size': list(self.image_size),  # Convert tuple to list for YAML compatibility
            'max_sequence_length': self.max_sequence_length,
            'fonts_used': list(self.working_fonts.keys()),
            'character_set': list(self.char_to_idx.keys()),
            'generator_type': self.__class__.__name__,
            'generated_by': f'{self.__class__.__name__} v2.0'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = Path(self.output_dir) / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_val}")
        print(f"Fonts used: {len(self.working_fonts)}")
        
        return all_metadata
    
    def preview_samples(self, num_samples: int = 10) -> List[Tuple[Image.Image, str]]:
        """
        Generate preview samples for visual inspection.
        
        Args:
            num_samples: Number of preview samples
            
        Returns:
            List of (image, label) tuples
        """
        samples = []
        
        for _ in range(num_samples):
            image, metadata = self.generate_single_image()
            samples.append((image, metadata['label']))
        
        return samples 