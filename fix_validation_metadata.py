#!/usr/bin/env python3
"""
Fix Validation Metadata Script
Regenerate complete metadata.json for validation_short_text dataset
with all 3200 samples instead of just 100.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import re
from PIL import Image

def main():
    print("🔧 Fixing Validation Metadata")
    print("=" * 60)
    
    # Dataset directory
    dataset_dir = Path("data/validation_short_text")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    metadata_file = dataset_dir / "metadata.json"
    
    # Verify directories exist
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # Scan all image files
    image_files = sorted([f for f in images_dir.glob("*.png")])
    print(f"📁 Found {len(image_files)} image files")
    
    if len(image_files) != 3200:
        print(f"⚠️  Expected 3200 images, found {len(image_files)}")
    
    # Generate complete metadata
    samples = []
    length_counts = defaultdict(int)
    
    print("🔄 Processing all image files...")
    
    for i, image_file in enumerate(image_files):
        if i % 500 == 0:
            print(f"   Processed {i}/{len(image_files)} files...")
        
        # Extract filename without extension
        filename_base = image_file.stem
        
        # Load corresponding label
        label_file = labels_dir / f"{filename_base}.txt"
        if not label_file.exists():
            print(f"⚠️  Missing label file: {label_file}")
            continue
        
        # Read label text
        with open(label_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Get image dimensions
        try:
            with Image.open(image_file) as img:
                width, height = img.size
        except Exception as e:
            print(f"⚠️  Error reading image {image_file}: {e}")
            continue
        
        # Calculate text length
        text_length = len(text)
        length_counts[text_length] += 1
        
        # Create sample entry
        sample = {
            "filename": filename_base,
            "text": text,
            "length": text_length,
            "image_width": width,
            "image_height": height,
            "target_length": text_length,
            "actual_length": text_length
        }
        
        samples.append(sample)
    
    print(f"✅ Processed {len(samples)} samples")
    
    # Calculate length distribution
    total_samples = len(samples)
    length_distribution = {
        "very_short_1_5": {"count": 0, "percentage": 0.0},
        "short_6_10": {"count": 0, "percentage": 0.0},
        "medium_11_30": {"count": 0, "percentage": 0.0},
        "long_31_100": {"count": 0, "percentage": 0.0},
        "very_long_100_plus": {"count": 0, "percentage": 0.0}
    }
    
    for length, count in length_counts.items():
        if 1 <= length <= 5:
            length_distribution["very_short_1_5"]["count"] += count
        elif 6 <= length <= 10:
            length_distribution["short_6_10"]["count"] += count
        elif 11 <= length <= 30:
            length_distribution["medium_11_30"]["count"] += count
        elif 31 <= length <= 100:
            length_distribution["long_31_100"]["count"] += count
        else:
            length_distribution["very_long_100_plus"]["count"] += count
    
    # Calculate percentages
    for category in length_distribution.values():
        category["percentage"] = round((category["count"] / total_samples) * 100, 1)
    
    # Calculate average length
    total_chars = sum(length * count for length, count in length_counts.items())
    avg_length = total_chars / total_samples if total_samples > 0 else 0
    
    # Create complete metadata
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
            "length_range": f"1-{max(length_counts.keys()) if length_counts else 100}",
            "average_length": round(avg_length, 3),
            "length_distribution_by_range": length_distribution,
            "detailed_length_counts": dict(sorted(length_counts.items()))
        },
        "samples": samples
    }
    
    # Save complete metadata
    print(f"💾 Saving metadata with {len(samples)} samples...")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("✅ Metadata fix complete!")
    print(f"📊 Summary:")
    print(f"   Total samples: {total_samples}")
    print(f"   Average length: {avg_length:.1f} characters")
    print(f"   Length distribution:")
    for category, data in length_distribution.items():
        print(f"     {category.replace('_', ' ').title()}: {data['count']} ({data['percentage']}%)")
    
    # Verify the fix
    print("\n🔍 Verification:")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        verification_data = json.load(f)
    
    print(f"   Metadata file samples: {len(verification_data['samples'])}")
    print(f"   Expected samples: 3200")
    print(f"   ✅ Fix successful!" if len(verification_data['samples']) == 3200 else "❌ Fix failed!")

if __name__ == "__main__":
    main() 