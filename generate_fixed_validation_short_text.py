#!/usr/bin/env python3
"""
Generate Validation Dataset
Creates 3200 validation images using RandomWidthDataset for better length distribution.
"""

import sys
import os
sys.path.append('src')

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import json
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict



def create_output_structure():
    """Create the output directory structure for validation dataset."""
    output_dir = Path("data/validation_short_text")
    
    # Create main directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    
    return output_dir

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    # Denormalize from [-1, 1] to [0, 1]
    denormalized = (tensor + 1.0) / 2.0
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    return to_pil(denormalized)

def generate_validation_dataset():
    """Generate 3200 validation images."""
    print("📸 Generating Validation Dataset (3200 images)")
    print("=" * 60)
    
    # Create dataset
    try:
        from src.utils.config import ConfigManager
        from src.data.onthefly_dataset import OnTheFlyDataset
        from src.data.random_width_dataset import RandomWidthDataset
        
        # Create configuration
        config_manager = ConfigManager()
        
        # Create base validation dataset
        base_dataset = OnTheFlyDataset(
            split='val',  # Use validation split
            config_manager=config_manager,
            samples_per_epoch=3200,  # Generate exactly 3200 samples
            augment_prob=0.0,  # No augmentation for validation
            shuffle_texts=False,  # Consistent validation set
            random_seed=42  # Fixed seed for reproducibility
        )
        
        # Create RandomWidthDataset with balanced length distribution
        dataset = RandomWidthDataset(
            base_dataset=base_dataset,
            min_length=1,
            max_length=100,  # Full range
            alpha=0.005,  # Moderate bias toward shorter texts
            config_manager=config_manager
        )
        
        print("✅ Dataset created successfully!")
        print(f"   Base dataset size: {len(base_dataset)}")
        print(f"   RandomWidth dataset size: {len(dataset)}")
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False
    
    # Create output structure
    output_dir = create_output_structure()
    
    # Generation statistics
    length_stats = defaultdict(int)
    samples_metadata = []
    
    print(f"\n🔄 Generating 3200 validation images...")
    
    # Generate exactly 3200 samples
    for i in tqdm(range(3200), desc="Generating images"):
        try:
            # Get sample from dataset
            sample = dataset[i % len(dataset)]
            text = sample['text']
            text_length = len(text)
            
            # Convert tensor to PIL image
            pil_image = tensor_to_pil(sample['image'])
            
            # Create filenames
            filename = f"val_{i:06d}"
            image_path = output_dir / "images" / f"{filename}.png"
            label_path = output_dir / "labels" / f"{filename}.txt"
            
            # Save image
            pil_image.save(image_path)
            
            # Save label
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Update statistics
            length_stats[text_length] += 1
            
            # Store metadata
            samples_metadata.append({
                "filename": filename,
                "text": text,
                "length": text_length,
                "image_width": pil_image.width,
                "image_height": pil_image.height,
                "target_length": int(sample['target_length']),
                "actual_length": int(sample['actual_length'])
            })
            
        except Exception as e:
            print(f"⚠️  Error generating sample {i}: {e}")
            continue
    
    # Create comprehensive metadata
    create_validation_metadata(output_dir, samples_metadata, length_stats)
    
    print(f"\n✅ Validation dataset generation complete!")
    print(f"📁 Saved to: {output_dir}")
    print(f"📊 Generated {len(samples_metadata)} images")
    
    return True

def create_validation_metadata(output_dir, samples_metadata, length_stats):
    """Create comprehensive metadata for the validation dataset."""
    
    # Calculate length distribution statistics
    total_samples = len(samples_metadata)
    length_distribution = {}
    
    # Group by length ranges
    ranges = {
        "very_short_1_5": (1, 5),
        "short_6_10": (6, 10),
        "medium_11_30": (11, 30),
        "long_31_100": (31, 100),
        "very_long_100_plus": (101, 1000)
    }
    
    range_counts = {range_name: 0 for range_name in ranges}
    
    for length, count in length_stats.items():
        for range_name, (min_len, max_len) in ranges.items():
            if min_len <= length <= max_len:
                range_counts[range_name] += count
                break
    
    # Create summary metadata
    metadata = {
        "dataset_info": {
            "name": "Validation Dataset",
            "total_samples": total_samples,
            "image_height": 32,
            "variable_width": True,
            "created_with": "RandomWidthDataset"
        },
        "length_statistics": {
            "total_samples": total_samples,
            "length_range": f"{min(length_stats.keys())}-{max(length_stats.keys())}",
            "average_length": sum(l * c for l, c in length_stats.items()) / total_samples,
            "length_distribution_by_range": {
                range_name: {
                    "count": count,
                    "percentage": round(count / total_samples * 100, 1)
                }
                for range_name, count in range_counts.items()
            },
            "detailed_length_counts": dict(sorted(length_stats.items()))
        },

        "samples": samples_metadata  # Include all samples
    }
    
    # Save metadata JSON
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Create human-readable summary
    summary_lines = [
        "# Validation Dataset Summary", 
        "",
        f"**Total Samples:** {total_samples}",
        f"**Image Dimensions:** Variable width × 32px height",
        "",
        "## Length Distribution",
        ""
    ]
    
    for range_name, count in range_counts.items():
        percentage = count / total_samples * 100
        range_display = range_name.replace("_", " ").title()
        summary_lines.append(f"- **{range_display}:** {count} samples ({percentage:.1f}%)")
    
    summary_lines.extend([
        "",
        "## Dataset Features",
        "",
        "✅ **Variable Width Images** - Optimized width for each text length",
        "✅ **Balanced Length Distribution** - Good representation of short to long texts",
        "✅ **Consistent Format** - Compatible with existing training pipeline",
        "✅ **Reproducible** - Fixed seed for consistent validation results",
        "",
        "## Expected Performance Impact",
        "",
        "This validation dataset should provide:",
        "- Accurate performance metrics across different text lengths",
        "- Better evaluation of model improvements",
        "- Consistent baseline for comparing different training approaches",
        "",
        f"## Files Generated",
        "",
        f"- `images/`: {total_samples} PNG image files",
        f"- `labels/`: {total_samples} corresponding text labels",
        f"- `metadata.json`: Detailed dataset statistics and sample information",
        f"- `README.md`: This summary document",
        "",
        "Generated using RandomWidthDataset for improved text length distribution."
    ])
    
    # Save summary
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print("\n📊 Length Distribution Summary:")
    for range_name, count in range_counts.items():
        percentage = count / total_samples * 100
        range_display = range_name.replace("_", " ").title()
        print(f"   {range_display:20} {count:5d} samples ({percentage:5.1f}%)")

def main():
    """Main execution function."""
    print("🚀 Validation Dataset Generator")
    print("=" * 60)
    print("Generating 3200 validation images with RandomWidthDataset")
    print("for improved text length distribution and evaluation accuracy.")
    print("=" * 60)
    
    success = generate_validation_dataset()
    
    if success:
        print("\n🎉 Validation dataset generation successful!")
        print("\n📋 Next Steps:")
        print("   1. Use this dataset for consistent validation during training")
        print("   2. Compare performance with previous validation results")
        print("   3. Monitor improvement in short text recognition")
        print("   4. Use for final model evaluation")
        
        print("\n💡 Integration:")
        print("   - Update your training script to use 'data/validation_dataset'")
        print("   - This provides better length distribution for validation")
        print("   - Useful for measuring improvements across different text lengths")
        
    else:
        print("❌ Failed to generate validation dataset. Check error messages above.")

if __name__ == "__main__":
    main() 