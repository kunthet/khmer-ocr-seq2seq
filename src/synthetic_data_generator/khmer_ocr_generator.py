"""
Integrated Khmer OCR Synthetic Data Generator

This module combines the best features from the imported synthetic data generator
with the existing Khmer OCR project architecture. It generates variable-width images
from corpus data in data/processed/ and outputs to data/synthetic/ structure.
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import logging
import traceback

from ..utils.config import ConfigManager
from ..khtext.khnormal_fast import khnormal

# Import components from synthetic_data_generator
from .utils import normalize_khmer_text, load_khmer_fonts, validate_font_collection
from .backgrounds import BackgroundGenerator
from .augmentation import ImageAugmentor

logger = logging.getLogger(__name__)

try:
    from PIL import features
    HAS_RAQM = features.check('raqm')
except ImportError:
    HAS_RAQM = False

if not HAS_RAQM:
    logger.warning("Pillow-Raqm not found. Complex script rendering (Khmer) may be incorrect. "
                   "Please install it for proper text rendering: pip install Pillow-Raqm")


class KhmerOCRSyntheticGenerator:
    """
    Integrated synthetic data generator for Khmer OCR that:
    - Uses corpus data from data/processed/ folder
    - Generates variable-width images with 32px height
    - Outputs to data/synthetic/ structure compatible with existing training pipeline
    - Integrates advanced text rendering and augmentation features
    """
    
    def __init__(
        self,
        corpus_dir: str = "data/processed",
        output_dir: str = "data/synthetic",
        config_manager: Optional[ConfigManager] = None,
        fonts_dir: str = "fonts",
        augment_train: bool = True,
        use_advanced_backgrounds: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the integrated Khmer OCR synthetic data generator
        
        Args:
            corpus_dir: Directory with processed corpus text files (train.txt, val.txt, test.txt)
            output_dir: Output directory for synthetic images
            config_manager: Configuration manager instance
            fonts_dir: Directory containing Khmer fonts
            augment_train: Whether to apply augmentation to training images
            use_advanced_backgrounds: Whether to use advanced background generation
            random_seed: Random seed for reproducibility
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.config = config_manager or ConfigManager()
        self.fonts_dir = fonts_dir
        self.augment_train = augment_train
        self.use_advanced_backgrounds = use_advanced_backgrounds
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Get image configuration
        self.image_height = self.config.data_config.image_height  # 32px as per config
        self.image_channels = self.config.data_config.image_channels  # 1 for grayscale
        
        # Initialize fonts
        self._initialize_fonts()
        
        # Background generator will be created dynamically per image for variable width support
        
        # Initialize augmentation
        if self.augment_train:
            self.augmentor = ImageAugmentor(
                rotation_range=(-2, 2),
                scale_range=(0.9, 1.1),
                noise_std=0.1,
                brightness_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2)
            )
        else:
            self.augmentor = None
        
        # Create output directory structure
        self._create_output_structure()
        
        # Font configuration
        self.font_size = int(self.image_height * 0.5)  # Use 50% of height for better fit
        
        # Performance tracking
        self.generated_count = 0
        self.total_size_mb = 0.0
        self.split_stats = {}
        
        logger.info("Khmer OCR Synthetic Generator initialized")
        logger.info(f"Image height: {self.image_height}px (variable width)")
        logger.info(f"Loaded {len(self.fonts)} fonts")
        logger.info(f"Advanced backgrounds: {self.use_advanced_backgrounds}")
        logger.info(f"Training augmentation: {self.augment_train}")
    
    def _initialize_fonts(self):
        """Initialize Khmer fonts from fonts directory"""
        if not os.path.exists(self.fonts_dir):
            raise ValueError(f"Fonts directory not found: {self.fonts_dir}")
        
        # Load all available Khmer fonts
        fonts_dict = load_khmer_fonts(self.fonts_dir)
        font_validation = validate_font_collection(self.fonts_dir)
        
        # Filter to only working fonts
        self.fonts = {}
        for font_name, font_path in fonts_dict.items():
            if font_validation.get(font_name, False):
                self.fonts[font_name] = font_path
        
        if not self.fonts:
            raise ValueError(f"No working Khmer fonts found in {self.fonts_dir}")
        
        logger.info(f"Loaded working fonts: {list(self.fonts.keys())}")
    
    def _create_output_structure(self):
        """Create output directory structure"""
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            (split_dir / "images").mkdir(exist_ok=True)
            (split_dir / "labels").mkdir(exist_ok=True)
    
    def _load_corpus_text(self, split: str) -> List[str]:
        """Load corpus text from processed files"""
        corpus_file = self.corpus_dir / f"{split}.txt"
        
        if not corpus_file.exists():
            raise ValueError(f"Corpus file not found: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        logger.info(f"Loaded {len(lines)} text lines from {corpus_file}")
        return lines
    
    def _calculate_optimal_width(self, text: str, font: ImageFont.FreeTypeFont) -> int:
        """Calculate optimal image width for given text and font"""
        # Get text bounding box
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        
        # Add padding (5% on each side)
        padding = int(text_width * 0.05)
        optimal_width = text_width + 2 * padding
        
        # Ensure minimum width but allow unlimited maximum width to prevent text cutting
        optimal_width = max(64, optimal_width)
        
        return optimal_width
    
    def _render_text_image(self, text: str, font_path: str, image_width: int) -> Image.Image:
        """
        Render text on a background image with proper Khmer support.
        
        Args:
            text: Text to render
            font_path: Path to font file
            image_width: Width of the image to create
            
        Returns:
            PIL Image with rendered text
        """
        try:
            # Load font
            font = ImageFont.truetype(font_path, size=self.font_size)
            
            # First restore whitespace tags to actual spaces
            try:
                from src.khtext.subword_cluster import restore_whitespace_tags
                text_with_spaces = restore_whitespace_tags(text)
            except ImportError:
                text_with_spaces = text  # Fallback if module not available
            
            # Normalize Khmer text for proper rendering
            normalized_text = khnormal(text_with_spaces)
            
            # Create background with proper size
            # BackgroundGenerator needs to be created with correct dimensions for each image
            bg_generator = BackgroundGenerator(image_size=(image_width, self.image_height))
            background = bg_generator.generate_random_background()
            
            # Determine text color based on background
            text_color = bg_generator.get_optimal_text_color(background)
            
            # Create drawing context
            draw = ImageDraw.Draw(background)
            
            # Get text bounding box for positioning
            try:
                # Try to get bounding box - this works with proper fonts
                bbox = draw.textbbox((0, 0), normalized_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # Fallback if textbbox fails
                text_width, text_height = draw.textsize(normalized_text, font=font)
            
                       
            # Calculate text position (centered)
            x = max(0, (image_width - text_width) // 2)
            y = max(0, (self.image_height - text_height) // 2)
            
            if text_height <=18:
                y = max(0, (self.image_height - 23) // 2)
            
            # Render text directly on background
            draw.text((x, y), normalized_text, font=font, fill=text_color)
            
            # Convert to grayscale if needed
            if self.image_channels == 1 and background.mode != 'L':
                background = background.convert('L')
            
            return background
            
        except Exception as e:
            logger.error(f"Failed to render text '{text}' with font '{font_path}': {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create fallback image
            fallback_image = Image.new('RGB', (image_width, self.image_height), color='white')
            draw = ImageDraw.Draw(fallback_image)
            draw.text((5, 5), text, fill='black')
            # Convert to grayscale if needed
            if self.image_channels == 1:
                fallback_image = fallback_image.convert('L')
            return fallback_image
    
    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply augmentation to image and ensure target height is maintained"""
        if self.augmentor:
            # Apply augmentations
            augmented = self.augmentor.augment_image(image)
            
            # Ensure the augmented image maintains the target height
            if augmented.height != self.image_height:
                # Calculate new width to maintain aspect ratio
                aspect_ratio = augmented.width / augmented.height
                new_width = int(self.image_height * aspect_ratio)
                augmented = augmented.resize((new_width, self.image_height), Image.Resampling.LANCZOS)
            
            return augmented
        return image
    
    def _select_font(self, text: str, split: str) -> str:
        """Select font for text rendering"""
        if split == "train":
            # Random font selection for training (data augmentation)
            font_name = random.choice(list(self.fonts.keys()))
        else:
            # Consistent font selection for val/test based on text hash
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            font_idx = int(text_hash, 16) % len(self.fonts)
            font_name = list(self.fonts.keys())[font_idx]
        
        return self.fonts[font_name]
    
    def _save_sample(self, image: Image.Image, text: str, split: str, index: int, font_name: str) -> Dict:
        """Save image and label files"""
        # Generate filenames
        filename = f"{split}_{index:06d}"
        image_path = self.output_dir / split / "images" / f"{filename}.png"
        label_path = self.output_dir / split / "labels" / f"{filename}.txt"
        
        # Save image
        image.save(image_path)
        
        # Save label
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Return metadata
        return {
            "filename": filename,
            "text": text,
            "font": font_name,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "image_width": image.width,
            "image_height": image.height
        }
    
    def generate_split(self, split: str, max_samples: Optional[int] = None) -> Dict:
        """Generate synthetic images for a specific split"""
        logger.info(f"Generating synthetic images for {split} split...")
        
        # Load corpus text
        text_lines = self._load_corpus_text(split)
        
        if max_samples:
            text_lines = text_lines[:max_samples]
        
        logger.info(f"Processing {len(text_lines)} text lines for {split}")
        
        # Generate images
        samples = []
        apply_augmentation = (split == "train" and self.augment_train)
        
        for i, text in enumerate(tqdm(text_lines, desc=f"Generating {split} images")):
            # Skip empty lines
            if not text.strip():
                continue
            
            # Select font
            font_path = self._select_font(text, split)
            font_name = Path(font_path).stem
            
            # Calculate optimal width
            temp_font = ImageFont.truetype(font_path, self.font_size)
            image_width = self._calculate_optimal_width(text, temp_font)
            
            # Render text image
            image = self._render_text_image(text, font_path, image_width)
            
            # Apply augmentation if needed
            if apply_augmentation:
                image = self._apply_augmentation(image)
            
            # Save sample
            sample_info = self._save_sample(image, text, split, i, font_name)
            samples.append(sample_info)
        
        # Create and save metadata
        metadata = {
            "split": split,
            "total_samples": len(samples),
            "augmentation_applied": apply_augmentation,
            "fonts_used": list(self.fonts.keys()),
            "image_height": self.image_height,
            "variable_width": True,
            "advanced_backgrounds": self.use_advanced_backgrounds,
            "samples": samples
        }
        
        metadata_path = self.output_dir / split / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(samples)} images for {split} split")
        return metadata
    
    def generate_all_splits(self, max_samples: Optional[int] = None) -> Dict:
        """Generate synthetic images for all splits"""
        logger.info("Starting synthetic image generation for all splits...")
        
        all_metadata = {}
        
        # Generate each split
        for split in ["train", "val", "test"]:
            try:
                metadata = self.generate_split(split, max_samples)
                all_metadata[split] = metadata
            except Exception as e:
                logger.error(f"Failed to generate {split} split: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Create overall metadata
        overall_metadata = {
            "generation_config": {
                "corpus_dir": str(self.corpus_dir),
                "output_dir": str(self.output_dir),
                "image_height": self.image_height,
                "variable_width": True,
                "augment_train": self.augment_train,
                "use_advanced_backgrounds": self.use_advanced_backgrounds,
                "fonts": list(self.fonts.keys())
            },
            "splits": all_metadata,
            "total_images": sum(meta["total_samples"] for meta in all_metadata.values())
        }
        
        # Save overall metadata
        overall_metadata_path = self.output_dir / "generation_metadata.json"
        with open(overall_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(overall_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Image generation complete!")
        logger.info(f"Total images generated: {overall_metadata['total_images']}")
        
        return overall_metadata
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated images"""
        stats = {
            "splits": {},
            "total_images": 0,
            "total_size_mb": 0
        }
        
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            if split_dir.exists():
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
                
                if images_dir.exists():
                    image_files = list(images_dir.glob("*.png"))
                    label_files = list(labels_dir.glob("*.txt"))
                    
                    # Calculate size
                    split_size = sum(f.stat().st_size for f in image_files)
                    split_size += sum(f.stat().st_size for f in label_files)
                    
                    stats["splits"][split] = {
                        "images": len(image_files),
                        "labels": len(label_files),
                        "size_mb": split_size / (1024 * 1024)
                    }
                    
                    stats["total_images"] += len(image_files)
                    stats["total_size_mb"] += split_size / (1024 * 1024)
        
        return stats 

 