#!/usr/bin/env python3
"""
Fixed Validation Set Generator for Khmer OCR

This script generates exactly 6,400 validation images that will remain fixed
throughout all training runs to ensure consistent and reproducible evaluation.
The validation set is saved to data/validation_fixed/ for use during training.
"""

import os
import sys
import argparse
import logging
import json
import random
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.synthetic_data_generator.khmer_ocr_generator import KhmerOCRSyntheticGenerator
from src.utils.config import ConfigManager


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_validation_texts(corpus_dir: str, num_samples: int = 6400, random_seed: int = 42) -> List[str]:
    """
    Load and select exactly num_samples validation texts from corpus
    
    Args:
        corpus_dir: Directory with processed corpus files
        num_samples: Number of validation samples to select
        random_seed: Random seed for reproducible selection
        
    Returns:
        List of selected validation texts
    """
    val_file = Path(corpus_dir) / "val.txt"
    
    if not val_file.exists():
        raise FileNotFoundError(f"Validation corpus file not found: {val_file}")
    
    # Load all validation texts
    with open(val_file, 'r', encoding='utf-8') as f:
        all_texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(all_texts)} validation texts in corpus")
    
    # Set random seed for reproducible selection
    random.seed(random_seed)
    
    # Select exactly num_samples texts
    if len(all_texts) >= num_samples:
        selected_texts = random.sample(all_texts, num_samples)
    else:
        # If we don't have enough texts, cycle through them
        selected_texts = []
        cycles_needed = (num_samples + len(all_texts) - 1) // len(all_texts)
        
        for cycle in range(cycles_needed):
            # Shuffle for each cycle
            shuffled_texts = all_texts.copy()
            random.shuffle(shuffled_texts)
            
            remaining_needed = num_samples - len(selected_texts)
            texts_to_take = min(len(shuffled_texts), remaining_needed)
            
            selected_texts.extend(shuffled_texts[:texts_to_take])
            
            if len(selected_texts) >= num_samples:
                break
        
        selected_texts = selected_texts[:num_samples]
    
    print(f"Selected {len(selected_texts)} validation texts")
    return selected_texts


def generate_fixed_validation_set(
    output_dir: str = "data/validation_fixed",
    corpus_dir: str = "data/processed", 
    fonts_dir: str = "fonts",
    num_samples: int = 6400,
    random_seed: int = 42
) -> Dict:
    """
    Generate exactly 6,400 fixed validation images
    
    Args:
        output_dir: Output directory for validation images
        corpus_dir: Directory with processed corpus files
        fonts_dir: Directory with Khmer fonts
        num_samples: Number of validation samples (fixed at 6,400)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with generation metadata
    """
    print(f"Generating {num_samples} fixed validation images...")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {random_seed}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    # Load validation texts
    validation_texts = load_validation_texts(corpus_dir, num_samples, random_seed)
    
    # Initialize configuration and generator
    config = ConfigManager()
    
    # Create a modified generator for validation (no augmentation)
    print("Initializing Khmer OCR generator...")
    generator = KhmerOCRSyntheticGenerator(
        corpus_dir=corpus_dir,
        output_dir=str(output_path),
        config_manager=config,
        fonts_dir=fonts_dir,
        augment_train=False,  # No augmentation for validation
        use_advanced_backgrounds=True,  # Use advanced backgrounds for realism
        random_seed=random_seed
    )
    
    # Generate images
    print(f"Generating {num_samples} validation images...")
    
    metadata_samples = []
    fonts = list(generator.fonts.keys())
    
    for i, text in enumerate(validation_texts):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_samples} ({i/num_samples*100:.1f}%)")
        
        # Select font consistently based on text hash for reproducibility
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        font_idx = int(text_hash, 16) % len(fonts)
        font_name = fonts[font_idx]
        font_path = generator.fonts[font_name]
        
        try:
            # Calculate optimal image width for this text
            from PIL import ImageFont
            temp_font = ImageFont.truetype(font_path, generator.font_size)
            image_width = generator._calculate_optimal_width(text, temp_font)
            
            # Render text to image using the generator's method
            image = generator._render_text_image(text, font_path, image_width)
            
            # Save image and label
            filename = f"val_{i:06d}"
            image_path = images_dir / f"{filename}.png"
            label_path = labels_dir / f"{filename}.txt"
            
            # Save image
            image.save(image_path)
            
            # Save label
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Record metadata
            sample_metadata = {
                "filename": filename,
                "text": text,
                "font": font_name,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_width": image.width,
                "image_height": image.height,
                "text_length": len(text)
            }
            metadata_samples.append(sample_metadata)
            
        except Exception as e:
            print(f"Error generating sample {i} for text '{text}': {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
    print(f"Successfully generated {len(metadata_samples)} validation images")
    
    # Create metadata
    metadata = {
        "dataset_type": "fixed_validation_set",
        "total_samples": len(metadata_samples),
        "target_samples": num_samples,
        "generation_config": {
            "corpus_dir": corpus_dir,
            "fonts_dir": fonts_dir,
            "output_dir": output_dir,
            "random_seed": random_seed,
            "image_height": 32,
            "variable_width": True,
            "augmentation_applied": False,
            "advanced_backgrounds": True
        },
        "fonts_used": fonts,
        "samples": metadata_samples
    }
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Save validation texts for reference
    texts_path = output_path / "validation_texts.txt"
    with open(texts_path, 'w', encoding='utf-8') as f:
        for text in validation_texts:
            f.write(text + '\n')
    
    print(f"Validation texts saved to: {texts_path}")
    
    return metadata


def validate_fixed_set(validation_dir: str = "data/validation_fixed") -> bool:
    """
    Validate the generated fixed validation set
    
    Args:
        validation_dir: Directory with validation images
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"Validating fixed validation set at: {validation_dir}")
    
    validation_path = Path(validation_dir)
    
    if not validation_path.exists():
        print("❌ Validation directory does not exist")
        return False
    
    # Check directory structure
    images_dir = validation_path / "images"
    labels_dir = validation_path / "labels"
    metadata_path = validation_path / "metadata.json"
    
    if not images_dir.exists():
        print("❌ Images directory missing")
        return False
    
    if not labels_dir.exists():
        print("❌ Labels directory missing")
        return False
    
    if not metadata_path.exists():
        print("❌ Metadata file missing")
        return False
    
    # Count files
    image_files = list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(label_files)} label files")
    
    # Check counts
    if len(image_files) != 6400:
        print(f"❌ Expected 6400 images, found {len(image_files)}")
        return False
    
    if len(label_files) != 6400:
        print(f"❌ Expected 6400 labels, found {len(label_files)}")
        return False
    
    # Load and validate metadata
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if metadata["total_samples"] != 6400:
            print(f"❌ Metadata reports {metadata['total_samples']} samples, expected 6400")
            return False
        
        print(f"✅ Metadata validation passed")
        
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return False
    
    # Sample validation - check first few files
    for i in range(min(5, len(image_files))):
        filename = f"val_{i:06d}"
        image_path = images_dir / f"{filename}.png"
        label_path = labels_dir / f"{filename}.txt"
        
        if not image_path.exists():
            print(f"❌ Missing image: {image_path}")
            return False
        
        if not label_path.exists():
            print(f"❌ Missing label: {label_path}")
            return False
        
        try:
            # Verify image can be loaded
            from PIL import Image
            img = Image.open(image_path)
            if img.height != 32:
                print(f"❌ Image {filename} has height {img.height}, expected 32")
                return False
            
            # Verify label can be read
            with open(label_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if not text:
                    print(f"❌ Empty label file: {label_path}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error validating sample {filename}: {e}")
            return False
    
    print("✅ Fixed validation set validation passed!")
    print(f"✅ 6,400 validation images ready for training")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Fixed Validation Set for Khmer OCR")
    
    # Generation options
    parser.add_argument("--output-dir", type=str, default="data/validation_fixed",
                       help="Output directory for validation images")
    parser.add_argument("--corpus-dir", type=str, default="data/processed",
                       help="Directory with processed corpus text files")
    parser.add_argument("--fonts-dir", type=str, default="fonts",
                       help="Directory containing Khmer fonts")
    parser.add_argument("--num-samples", type=int, default=6400,
                       help="Number of validation samples to generate")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Utility options
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing validation set")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.validate_only:
            # Only validate existing set
            logger.info("Validating existing fixed validation set...")
            success = validate_fixed_set(args.output_dir)
            if success:
                logger.info("✅ Validation set is ready for use")
            else:
                logger.error("❌ Validation set validation failed")
                sys.exit(1)
        else:
            # Generate new validation set
            logger.info("=== Fixed Validation Set Generator ===")
            logger.info(f"Target samples: {args.num_samples}")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info(f"Random seed: {args.random_seed}")
            
            # Check if validation set already exists
            if Path(args.output_dir).exists():
                response = input(f"Validation set already exists at {args.output_dir}. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Validation set generation cancelled")
                    return
            
            # Generate validation set
            metadata = generate_fixed_validation_set(
                output_dir=args.output_dir,
                corpus_dir=args.corpus_dir,
                fonts_dir=args.fonts_dir,
                num_samples=args.num_samples,
                random_seed=args.random_seed
            )
            
            # Validate the generated set
            logger.info("Validating generated validation set...")
            success = validate_fixed_set(args.output_dir)
            
            if success:
                logger.info("✅ Fixed validation set generated and validated successfully!")
                logger.info(f"✅ {metadata['total_samples']} validation images ready")
                logger.info(f"✅ Location: {args.output_dir}")
                logger.info("✅ This validation set will remain fixed for all training runs")
            else:
                logger.error("❌ Validation set validation failed")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Fixed validation set generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 