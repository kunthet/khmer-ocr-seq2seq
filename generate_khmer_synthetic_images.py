#!/usr/bin/env python3
"""
Integrated Khmer OCR Synthetic Image Generation Script

This script uses the integrated KhmerOCRSyntheticGenerator to generate synthetic images
from corpus data in data/processed/ folder. It generates variable-width images with
32px height and outputs to data/synthetic/ structure for training.

Features:
- Uses corpus data from data/processed/ (train.txt, val.txt, test.txt)
- Generates variable-width images with 32px height
- Advanced background generation and text rendering
- Khmer text normalization for proper rendering
- Augmentation for training data
- Compatible with existing training pipeline
"""

import os
import sys
import argparse
import logging
from pathlib import Path

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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Synthetic Images for Khmer OCR using Corpus Data")
    
    # Basic configuration
    parser.add_argument("--corpus-dir", type=str, default="data/processed",
                       help="Directory with processed corpus text files (train.txt, val.txt, test.txt)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic",
                       help="Output directory for synthetic images")
    parser.add_argument("--fonts-dir", type=str, default="fonts",
                       help="Directory containing Khmer fonts")
    
    # Generation options
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per split (for testing)")
    parser.add_argument("--no-augment", action="store_true",
                       help="Disable augmentation for training images")
    parser.add_argument("--no-advanced-backgrounds", action="store_true",
                       help="Disable advanced background generation")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                       choices=["train", "val", "test"],
                       help="Splits to generate")
    
    # Utility options
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics of existing generated images")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    fonts_dir = Path(args.fonts_dir)
    
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        sys.exit(1)
    
    if not fonts_dir.exists():
        logger.error(f"Fonts directory not found: {fonts_dir}")
        sys.exit(1)
    
    # Check corpus files
    required_files = ["train.txt", "val.txt", "test.txt"]
    missing_files = []
    for file in required_files:
        if not (corpus_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing corpus files: {missing_files}")
        logger.error("Please run corpus processing first to generate train/val/test splits")
        sys.exit(1)
    
    # Display configuration
    logger.info("=== Khmer OCR Synthetic Image Generator ===")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Fonts directory: {fonts_dir}")
    logger.info(f"Splits to generate: {args.splits}")
    logger.info(f"Max samples per split: {args.max_samples or 'All'}")
    logger.info(f"Augmentation: {'Disabled' if args.no_augment else 'Enabled'}")
    logger.info(f"Advanced backgrounds: {'Disabled' if args.no_advanced_backgrounds else 'Enabled'}")
    logger.info(f"Random seed: {args.random_seed}")
    
    try:
        # Initialize generator
        logger.info("Initializing Khmer OCR Synthetic Generator...")
        generator = KhmerOCRSyntheticGenerator(
            corpus_dir=str(corpus_dir),
            output_dir=str(output_dir),
            config_manager=ConfigManager(),
            fonts_dir=str(fonts_dir),
            augment_train=not args.no_augment,
            use_advanced_backgrounds=not args.no_advanced_backgrounds,
            random_seed=args.random_seed
        )
        
        if args.stats:
            # Show statistics
            stats = generator.get_statistics()
            logger.info("\n=== Synthetic Image Statistics ===")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Total images: {stats['total_images']:,}")
            logger.info(f"Total size: {stats['total_size_mb']:.2f} MB")
            
            for split, split_stats in stats["splits"].items():
                logger.info(f"\n{split.upper()} split:")
                logger.info(f"  Images: {split_stats['images']:,}")
                logger.info(f"  Labels: {split_stats['labels']:,}")
                logger.info(f"  Size: {split_stats['size_mb']:.2f} MB")
            
            return
        
        # Generate images
        if set(args.splits) == {"train", "val", "test"}:
            # Generate all splits
            logger.info("Generating all splits...")
            metadata = generator.generate_all_splits(args.max_samples)
        else:
            # Generate specific splits
            logger.info(f"Generating splits: {args.splits}")
            metadata = {}
            for split in args.splits:
                metadata[split] = generator.generate_split(split, args.max_samples)
        
        # Show final statistics
        stats = generator.get_statistics()
        logger.info("\n=== Generation Complete ===")
        logger.info(f"Total images generated: {stats['total_images']:,}")
        logger.info(f"Total size: {stats['total_size_mb']:.2f} MB")
        logger.info(f"Output directory: {output_dir}")
        
        # Show per-split statistics
        for split, split_stats in stats["splits"].items():
            logger.info(f"{split.upper()}: {split_stats['images']:,} images ({split_stats['size_mb']:.2f} MB)")
        
        logger.info("\n✅ Synthetic image generation completed successfully!")
        logger.info("The generated images are ready for training with the existing pipeline.")
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 