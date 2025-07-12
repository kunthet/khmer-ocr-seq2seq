#!/usr/bin/env python3
"""
Corpus Processing Script

This script processes the Khmer corpus data from both sources and creates
train/validation/test splits for OCR training.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.data.corpus_processor import KhmerCorpusProcessor


def main():
    """Process corpus and display statistics"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Check if corpus directory exists
    corpus_dir = Path("corpus")
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        logger.error("Please ensure the corpus data is in the 'corpus/' directory")
        return
    
    # Check subdirectories
    gemini_dir = corpus_dir / "gemini_generated"
    wiki_dir = corpus_dir / "kmwiki_data"
    
    if not gemini_dir.exists():
        logger.warning(f"Gemini data not found: {gemini_dir}")
    else:
        logger.info(f"Found Gemini data directory: {gemini_dir}")
    
    if not wiki_dir.exists():
        logger.warning(f"Wikipedia data not found: {wiki_dir}")
    else:
        logger.info(f"Found Wikipedia data directory: {wiki_dir}")
    
    # Process corpus
    logger.info("Starting corpus processing...")
    
    processor = KhmerCorpusProcessor(
        corpus_root="corpus",
        min_line_length=5,
        max_line_length=150,
        test_split=0.1,
        val_split=0.1,
        random_seed=42
    )
    
    try:
        stats = processor.process_corpus("data/processed")
        
        # Display results
        print("\n" + "="*50)
        print("CORPUS PROCESSING COMPLETE")
        print("="*50)
        
        print(f"\n📊 STATISTICS:")
        print(f"   Total lines: {stats.total_lines:,}")
        print(f"   Total characters: {stats.total_characters:,}")
        print(f"   Unique characters: {len(stats.unique_characters)}")
        print(f"   Average line length: {stats.avg_line_length:.1f}")
        print(f"   Line length range: {stats.min_line_length}-{stats.max_line_length}")
        
        print(f"\n📁 SOURCE DISTRIBUTION:")
        for source, count in stats.sources.items():
            percentage = (count / stats.total_lines) * 100
            print(f"   {source.capitalize()}: {count:,} lines ({percentage:.1f}%)")
        
        # Calculate splits
        total = stats.total_lines
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        test_size = total - train_size - val_size
        
        print(f"\n📋 DATA SPLITS:")
        print(f"   Train: {train_size:,} lines ({train_size/total*100:.1f}%)")
        print(f"   Validation: {val_size:,} lines ({val_size/total*100:.1f}%)")
        print(f"   Test: {test_size:,} lines ({test_size/total*100:.1f}%)")
        
        # Sample characters
        char_sample = sorted(list(stats.unique_characters))[:50]
        print(f"\n🔤 CHARACTER SAMPLE (first 50):")
        print(f"   {''.join(char_sample)}")
        
        print(f"\n✅ Files saved to: data/processed/")
        print(f"   - train.txt")
        print(f"   - val.txt")
        print(f"   - test.txt")
        print(f"   - corpus_stats.txt")
        
        # Test data loading
        print(f"\n🧪 TESTING DATA LOADING:")
        splits = processor.load_processed_data("data/processed")
        for split_name, lines in splits.items():
            if lines:
                print(f"   {split_name}: {len(lines):,} lines loaded")
                # Show sample
                sample = lines[0] if lines else ""
                print(f"   Sample: '{sample[:50]}{'...' if len(sample) > 50 else ''}'")
            else:
                print(f"   {split_name}: No data loaded")
        
        print(f"\n🎯 READY FOR TRAINING!")
        print(f"   Use: python src/training/train_production.py")
        
    except Exception as e:
        logger.error(f"Error processing corpus: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 