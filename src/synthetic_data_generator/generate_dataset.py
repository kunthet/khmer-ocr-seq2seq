#!/usr/bin/env python3
"""
Script to generate synthetic training data for Khmer OCR (digits and full text).

This script demonstrates the usage of the SyntheticDataGenerator with support for:
- Full Khmer text generation (102+ characters)
- Curriculum learning datasets
- Corpus-based authentic text
- Multi-stage progressive training
- Advanced data generation strategies
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.synthetic_data_generator import SyntheticDataGenerator
from modules.synthetic_data_generator.utils import (
    validate_font_collection, 
    calculate_dataset_statistics,
    load_khmer_corpus,
    analyze_corpus_segments,
    get_full_khmer_characters
)


def main():
    """Main function to generate synthetic dataset."""
    parser = argparse.ArgumentParser(description='Generate synthetic Khmer OCR dataset (digits or full text)')
    
    # Basic configuration
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                      help='Path to model configuration file')
    parser.add_argument('--fonts-dir', type=str, default='src/fonts',
                      help='Directory containing Khmer fonts')
    parser.add_argument('--output-dir', type=str, default='generated_data',
                      help='Output directory for generated dataset')
    
    # Generation mode
    parser.add_argument('--mode', type=str, default='full_text', 
                      choices=['digits', 'full_text', 'mixed'],
                      help='Generation mode: digits only, full text, or mixed')
    parser.add_argument('--use-corpus', action='store_true', default=True,
                      help='Use real Khmer corpus for authentic text generation')
    parser.add_argument('--corpus-file', type=str, default='data/khmer_clean_text.txt',
                      help='Path to Khmer corpus file')
    
    # Dataset size and splits
    parser.add_argument('--num-samples', type=int, default=1000,
                      help='Number of samples to generate')
    parser.add_argument('--train-split', type=float, default=0.8,
                      help='Fraction of samples for training')
    
    # Advanced generation options
    parser.add_argument('--curriculum', action='store_true',
                      help='Generate curriculum learning datasets')
    parser.add_argument('--multi-stage', action='store_true',
                      help='Generate multi-stage progressive training datasets')
    parser.add_argument('--frequency-balanced', action='store_true',
                      help='Generate frequency-balanced character datasets')
    parser.add_argument('--mixed-complexity', action='store_true',
                      help='Generate mixed complexity datasets')
    
    # Content type control
    parser.add_argument('--content-type', type=str, default='auto',
                      choices=['auto', 'characters', 'syllables', 'words', 'phrases', 'mixed'],
                      help='Type of content to generate')
    parser.add_argument('--length-range', type=int, nargs=2, default=[1, 20],
                      help='Range of text lengths (min max)')
    
    # Utility options
    parser.add_argument('--preview-only', action='store_true',
                      help='Only generate preview samples without saving')
    parser.add_argument('--validate-fonts', action='store_true',
                      help='Validate font collection before generation')
    parser.add_argument('--analyze-corpus', action='store_true',
                      help='Analyze corpus characteristics before generation')
    parser.add_argument('--show-samples', type=int, default=5,
                      help='Number of sample texts to display')
    
    args = parser.parse_args()
    
    # Validate paths
    config_path = Path(args.config)
    fonts_dir = Path(args.fonts_dir)
    output_dir = Path(args.output_dir)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    if not fonts_dir.exists():
        print(f"Error: Fonts directory not found: {fonts_dir}")
        return 1
    
    # Display configuration
    mode_descriptions = {
        'digits': 'Khmer Digits (10 characters)',
        'full_text': 'Full Khmer Text (102+ characters)',
        'mixed': 'Mixed Digits + Text'
    }
    
    print("=== Advanced Khmer OCR Dataset Generator ===")
    print(f"Generation Mode: {mode_descriptions[args.mode]}")
    print(f"Configuration: {config_path}")
    print(f"Fonts directory: {fonts_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples:,}")
    print(f"Train/validation split: {args.train_split:.1%}/{1-args.train_split:.1%}")
    print(f"Content type: {args.content_type}")
    print(f"Length range: {args.length_range[0]}-{args.length_range[1]}")
    print(f"Corpus usage: {'Enabled' if args.use_corpus else 'Disabled'}")
    print()
    
    # Validate fonts if requested
    if args.validate_fonts:
        print("🔍 Validating font collection...")
        validation_results = validate_font_collection(str(fonts_dir))
        
        print("Font validation results:")
        working_fonts = 0
        for font_name, is_valid in validation_results.items():
            status = "✅" if is_valid else "❌"
            print(f"  {status} {font_name}")
            if is_valid:
                working_fonts += 1
        
        print(f"\nWorking fonts: {working_fonts}/{len(validation_results)}")
        
        if not working_fonts:
            print("❌ Error: No working fonts found!")
            return 1
        print()
    
    # Analyze corpus if requested
    if args.analyze_corpus and args.use_corpus:
        print("📊 Analyzing corpus characteristics...")
        corpus_analysis = analyze_corpus_segments(args.corpus_file)
        
        if 'error' not in corpus_analysis:
            stats = corpus_analysis['corpus_stats']
            print(f"Corpus statistics:")
            print(f"  Total lines: {stats['total_lines']:,}")
            print(f"  Average line length: {stats['avg_line_length']:.1f} characters")
            print(f"  Total characters: {stats['total_characters']:,}")
            
            for complexity, analysis in corpus_analysis['segment_analysis'].items():
                print(f"\n  {complexity.title()} segments:")
                print(f"    Count: {analysis['count']}")
                print(f"    Length range: {analysis['min_length']}-{analysis['max_length']}")
                print(f"    Unique characters: {analysis['unique_characters']}")
                
                print(f"    Sample segments:")
                for i, segment in enumerate(analysis['sample_segments'][:3]):
                    print(f"      {i+1}. '{segment}'")
        else:
            print(f"❌ Corpus analysis failed: {corpus_analysis['error']}")
        print()
    
    try:
        # Initialize generator
        print("🚀 Initializing synthetic data generator...")
        generator = SyntheticDataGenerator(
            config_path=str(config_path),
            fonts_dir=str(fonts_dir),
            output_dir=str(output_dir),
            mode=args.mode,
            use_corpus=args.use_corpus
        )
        print("✅ Generator initialized successfully!")
        
        # Display character vocabulary info
        if args.mode in ['full_text', 'mixed']:
            khmer_chars = get_full_khmer_characters()
            print(f"📝 Full Khmer character set: {len(khmer_chars)} characters")
            char_categories = {}
            for category, chars in khmer_chars.items():
                char_categories[category] = len(chars)
            print(f"   Character categories: {char_categories}")
        print()
        
        if args.preview_only:
            # Generate preview samples
            print("🔍 Generating preview samples...")
            samples = generator.preview_samples(num_samples=args.show_samples)
            
            print("Preview samples generated:")
            for i, (image, label) in enumerate(samples):
                char_count = len([c for c in label if '\u1780' <= c <= '\u17FF'])
                print(f"  Sample {i+1}: '{label}' ({len(label)} chars, {char_count} Khmer)")
            
            # Save preview samples
            preview_dir = output_dir / 'preview'
            preview_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (image, label) in enumerate(samples):
                # Create safe filename
                safe_label = ''.join(c if c.isalnum() or c in '_-' else '_' for c in label[:10])
                filename = f"preview_{i:02d}_{safe_label}.png"
                image.save(preview_dir / filename)
            
            print(f"💾 Preview samples saved to: {preview_dir}")
        
        elif args.curriculum:
            # Generate curriculum learning datasets
            print("🎓 Generating curriculum learning datasets...")
            
            curriculum_stages = {
                'stage1': {
                    'description': 'High-frequency characters (top 30)',
                    'samples': args.num_samples // 4
                },
                'stage2': {
                    'description': 'Medium-frequency characters (top 60)',
                    'samples': args.num_samples // 4
                },
                'stage3': {
                    'description': 'All characters with complex structures',
                    'samples': args.num_samples // 4
                },
                'mixed': {
                    'description': 'Mixed content including digits',
                    'samples': args.num_samples // 4
                }
            }
            
            total_generated = 0
            for stage_name, config in curriculum_stages.items():
                print(f"\n📚 Generating {stage_name}: {config['description']}")
                print(f"   Target samples: {config['samples']}")
                
                # Generate curriculum stage dataset
                dataset = generator.generate_curriculum_dataset(
                    stage=stage_name,
                    num_samples=config['samples'],
                    train_split=args.train_split,
                    save_images=True,
                    show_progress=True
                )
                
                # Show stage statistics
                train_samples = len(dataset['train']['samples'])
                val_samples = len(dataset['val']['samples'])
                total_generated += train_samples + val_samples
                
                print(f"   ✅ Generated {train_samples} training + {val_samples} validation samples")
                
                # Show sample labels from this stage
                sample_labels = [s['label'] for s in dataset['train']['samples'][:3]]
                print(f"   Sample texts: {sample_labels}")
            
            print(f"\n🎉 Curriculum learning dataset complete!")
            print(f"   Total samples generated: {total_generated:,}")
            print(f"   Stages: {len(curriculum_stages)}")
        
        elif args.multi_stage:
            # Generate multi-stage progressive datasets
            print("📈 Generating multi-stage progressive training datasets...")
            
            # Stage 1: Frequency balanced for core characters
            print("\n🎯 Stage 1: Frequency-balanced core character dataset")
            freq_dataset = generator.generate_frequency_balanced_dataset(
                num_samples=args.num_samples // 3,
                balance_factor=0.7,
                train_split=args.train_split,
                save_images=True
            )
            print(f"   ✅ Generated frequency-balanced dataset")
            
            # Stage 2: Mixed complexity for variety
            print("\n🎯 Stage 2: Mixed complexity dataset")
            mixed_dataset = generator.generate_mixed_complexity_dataset(
                num_samples=args.num_samples // 3,
                train_split=args.train_split,
                save_images=True
            )
            print(f"   ✅ Generated mixed complexity dataset")
            
            # Stage 3: Standard full dataset
            print("\n🎯 Stage 3: Standard full dataset")
            standard_dataset = generator.generate_dataset(
                num_samples=args.num_samples // 3,
                train_split=args.train_split,
                save_images=True,
                show_progress=True
            )
            print(f"   ✅ Generated standard dataset")
            
            total_samples = (
                len(freq_dataset['train']['samples']) + len(freq_dataset['val']['samples']) +
                len(mixed_dataset['train']['samples']) + len(mixed_dataset['val']['samples']) +
                len(standard_dataset['train']['samples']) + len(standard_dataset['val']['samples'])
            )
            
            print(f"\n🎉 Multi-stage dataset complete!")
            print(f"   Total samples generated: {total_samples:,}")
            print(f"   Stages: 3 (frequency-balanced, mixed-complexity, standard)")
        
        elif args.frequency_balanced:
            # Generate frequency-balanced dataset
            print("⚖️ Generating frequency-balanced character dataset...")
            dataset = generator.generate_frequency_balanced_dataset(
                num_samples=args.num_samples,
                balance_factor=0.5,
                train_split=args.train_split,
                save_images=True
            )
            
            print("✅ Frequency-balanced dataset generated!")
        
        elif args.mixed_complexity:
            # Generate mixed complexity dataset
            print("🌀 Generating mixed complexity dataset...")
            dataset = generator.generate_mixed_complexity_dataset(
                num_samples=args.num_samples,
                train_split=args.train_split,
                save_images=True
            )
            
            print("✅ Mixed complexity dataset generated!")
        
        else:
            # Generate standard dataset
            print("📝 Generating standard synthetic dataset...")
            metadata = generator.generate_dataset(
                num_samples=args.num_samples,
                train_split=args.train_split,
                save_images=True,
                show_progress=True
            )
            
            # Show sample texts
            train_samples = metadata['train']['samples']
            print(f"\n📋 Sample generated texts:")
            for i, sample in enumerate(train_samples[:args.show_samples]):
                char_count = len([c for c in sample['label'] if '\u1780' <= c <= '\u17FF'])
                print(f"  {i+1}. '{sample['label']}' ({len(sample['label'])} chars, {char_count} Khmer)")
        
        # Calculate and display comprehensive statistics
        print(f"\n📊 Calculating dataset statistics...")
        stats = calculate_dataset_statistics(str(output_dir))
        
        if 'error' not in stats:
            print("📈 Dataset Statistics:")
            print(f"  Total samples: {stats['total_samples']:,}")
            print(f"  Sequence length range: {stats['sequence_length_distribution']['min']}-{stats['sequence_length_distribution']['max']}")
            print(f"  Average sequence length: {stats['sequence_length_distribution']['mean']:.1f}")
            
            if 'character_coverage' in stats:
                print(f"  Character coverage: {stats['character_coverage']:.1%}")
            
            print(f"\n📊 Font distribution:")
            for font, count in list(stats['font_distribution'].items())[:10]:  # Top 10 fonts
                percentage = (count / stats['total_samples']) * 100
                print(f"    {font}: {count:,} ({percentage:.1f}%)")
            
            print(f"\n🔤 Character frequency (top 10):")
            for char, count in list(stats['character_frequency'].items())[:10]:
                print(f"    '{char}': {count:,}")
            
            # Additional analysis for full text mode
            if args.mode in ['full_text', 'mixed']:
                khmer_chars_in_data = {char for char in stats['character_frequency'].keys() 
                                     if '\u1780' <= char <= '\u17FF'}
                khmer_char_dict = get_full_khmer_characters()
                all_khmer_chars = []
                for category_chars in khmer_char_dict.values():
                    all_khmer_chars.extend(category_chars)
                total_khmer_chars = len(all_khmer_chars)
                coverage = len(khmer_chars_in_data) / total_khmer_chars * 100
                
                print(f"\n🎯 Khmer Character Analysis:")
                print(f"    Unique Khmer characters in dataset: {len(khmer_chars_in_data)}")
                print(f"    Total possible Khmer characters: {total_khmer_chars}")
                print(f"    Character coverage: {coverage:.1f}%")
        else:
            print(f"❌ Error calculating statistics: {stats['error']}")
        
        print(f"\n🎉 Generation completed successfully!")
        print(f"💾 Dataset saved to: {output_dir}")
        return 0
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main()) 