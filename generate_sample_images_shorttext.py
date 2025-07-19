#!/usr/bin/env python3
"""
Generate sample images using RandomWidthDataset for visual inspection.
This script creates images across different text length categories and saves them to files.
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
from collections import defaultdict

def create_output_directory():
    """Create output directory for sample images."""
    output_dir = Path("sample_images_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different length categories
    categories = ["very_short_1_5", "short_6_10", "medium_11_30", "long_31_plus"]
    for category in categories:
        (output_dir / category).mkdir(exist_ok=True)
    
    return output_dir

def tensor_to_pil(image_tensor):
    """Convert normalized tensor to PIL image."""
    # Denormalize from [-1, 1] to [0, 1]
    denormalized = (image_tensor + 1.0) / 2.0
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    
    # Convert to PIL
    if denormalized.shape[0] == 1:  # Grayscale
        pil_image = transforms.ToPILImage()(denormalized)
    else:  # RGB
        pil_image = transforms.ToPILImage()(denormalized)
    
    return pil_image

def get_category_from_length(length):
    """Determine category based on text length."""
    if 1 <= length <= 5:
        return "very_short_1_5"
    elif 6 <= length <= 10:
        return "short_6_10"
    elif 11 <= length <= 30:
        return "medium_11_30"
    else:
        return "long_31_plus"

def clean_filename(text):
    """Create a clean filename from text."""
    # Replace problematic characters
    clean = text.replace(' ', '_').replace('\n', '_').replace('\t', '_')
    # Keep only first 30 characters for filename
    clean = clean[:30] if len(clean) > 30 else clean
    # Remove any remaining problematic characters
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអេឯៃោៅ័់ាិីុូួើ"
    clean = ''.join(c for c in clean if c in allowed_chars)
    return clean if clean else "unknown_text"

def generate_samples_with_real_dataset():
    """Generate samples using a real dataset setup."""
    print("🔧 Attempting to use real dataset configuration...")
    
    try:
        # Try to import real components
        from src.data.onthefly_dataset import OnTheFlyDataset
        from src.utils.config import ConfigManager
        from src.data.random_width_dataset import RandomWidthDataset
        
        # Initialize real config
        config_manager = ConfigManager()
        
        # Create a small OnTheFlyDataset for testing
        base_dataset = OnTheFlyDataset(
            split='train',
            config_manager=config_manager,
            samples_per_epoch=100,  # Small size for quick testing
            augment_prob=0.5,  # No augmentation for clean samples
            random_seed=42
        )
        
        # Create RandomWidthDataset
        dataset = RandomWidthDataset(
            base_dataset=base_dataset,
            min_length=1,
            max_length=150,  # Keep reasonable for inspection
            alpha=0.02,  # Strong bias toward short texts
            config_manager=config_manager
        )
        
        print("✅ Successfully created real dataset!")
        return dataset, True
        
    except Exception as e:
        print(f"⚠️  Could not create real dataset: {e}")
        print("📝 Falling back to mock dataset...")
        return None, False

def generate_samples_with_mock_dataset():
    """Generate samples using mock dataset."""
    print("🔧 Using mock dataset configuration...")
    
    # Import mock components from test file
    from test_random_width_dataset import MockBaseDataset, MockConfigManager
    from src.data.random_width_dataset import RandomWidthDataset
    
    # Create mock dataset
    base_dataset = MockBaseDataset(200)
    config_manager = MockConfigManager()
    
    # Create RandomWidthDataset
    dataset = RandomWidthDataset(
        base_dataset=base_dataset,
        min_length=1,
        max_length=50,
        alpha=0.05,
        config_manager=config_manager
    )
    
    print("✅ Successfully created mock dataset!")
    return dataset

def generate_and_save_samples(dataset, output_dir, num_samples=50, is_real=True):
    """Generate samples and save them to files."""
    print(f"📸 Generating {num_samples} sample images...")
    
    # Track samples by category
    samples_by_category = defaultdict(list)
    sample_info = []
    
    # Generate samples
    for i in range(num_samples):
        try:
            sample = dataset[i % len(dataset)]
            
            # Extract information
            image = sample['image']
            text = sample['text']
            target_length = sample['target_length']
            actual_length = sample['actual_length']
            
            # Determine category
            category = get_category_from_length(actual_length)
            
            # Convert tensor to PIL image
            pil_image = tensor_to_pil(image)
            
            # Create filename
            clean_text = clean_filename(text)
            filename = f"sample_{i:03d}_len{actual_length:02d}_{clean_text}.png"
            
            # Save image
            save_path = output_dir / category / filename
            pil_image.save(save_path)
            
            # Store sample info
            sample_info.append({
                "index": i,
                "filename": filename,
                "category": category,
                "text": text,
                "target_length": int(target_length),  # Convert numpy types to Python int
                "actual_length": int(actual_length),
                "image_size": f"{pil_image.width}x{pil_image.height}"
            })
            
            samples_by_category[category].append({
                "filename": filename,
                "text": text,
                "length": int(actual_length)  # Convert to Python int
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{num_samples} samples...")
                
        except Exception as e:
            print(f"⚠️  Error generating sample {i}: {e}")
            continue
    
    return samples_by_category, sample_info

def create_summary_report(samples_by_category, sample_info, output_dir):
    """Create a summary report of generated samples."""
    print("📋 Creating summary report...")
    
    # Create detailed JSON report
    report = {
        "total_samples": len(sample_info),
        "categories": {},
        "samples": sample_info,
        "generation_stats": {}
    }
    
    # Category statistics
    for category, samples in samples_by_category.items():
        report["categories"][category] = {
            "count": len(samples),
            "samples": samples[:10]  # First 10 for preview
        }
    
    # Generation statistics
    lengths = [int(s["actual_length"]) for s in sample_info]  # Ensure all are Python ints
    report["generation_stats"] = {
        "length_range": f"{min(lengths)}-{max(lengths)}",
        "average_length": round(sum(lengths) / len(lengths), 1),
        "length_distribution": {
            "1-5": sum(1 for l in lengths if 1 <= l <= 5),
            "6-10": sum(1 for l in lengths if 6 <= l <= 10),
            "11-30": sum(1 for l in lengths if 11 <= l <= 30),
            "31+": sum(1 for l in lengths if l >= 31)
        }
    }
    
    # Save JSON report
    with open(output_dir / "sample_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Create human-readable summary
    summary_lines = [
        "# RandomWidthDataset Sample Images Report",
        f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Statistics",
        f"- Total samples generated: {len(sample_info)}",
        f"- Text length range: {min(lengths)}-{max(lengths)} characters",
        f"- Average text length: {round(sum(lengths) / len(lengths), 1)} characters",
        "",
        "## Length Distribution",
    ]
    
    for category, data in report["categories"].items():
        summary_lines.append(f"- {category.replace('_', ' ').title()}: {data['count']} samples")
    
    summary_lines.extend([
        "",
        "## Directory Structure",
        "```",
        "sample_images_output/",
        "├── very_short_1_5/     # 1-5 character texts",
        "├── short_6_10/         # 6-10 character texts", 
        "├── medium_11_30/       # 11-30 character texts",
        "├── long_31_plus/       # 31+ character texts",
        "├── sample_report.json  # Detailed JSON report",
        "└── README.md           # This summary",
        "```",
        "",
        "## Sample Examples",
    ])
    
    # Add examples from each category
    for category, data in report["categories"].items():
        if data["samples"]:
            summary_lines.extend([
                f"",
                f"### {category.replace('_', ' ').title()}",
            ])
            for sample in data["samples"][:5]:  # Show first 5
                summary_lines.append(f"- `{sample['filename']}`: \"{sample['text'][:50]}{'...' if len(sample['text']) > 50 else ''}\" (length: {sample['length']})")
    
    # Save README
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

def analyze_generated_samples(samples_by_category):
    """Analyze the quality and distribution of generated samples."""
    print("\n📊 Sample Generation Analysis:")
    print("=" * 50)
    
    total_samples = sum(len(samples) for samples in samples_by_category.values())
    
    for category, samples in samples_by_category.items():
        count = len(samples)
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        
        category_name = category.replace('_', ' ').title()
        print(f"{category_name:15} {count:3d} samples ({percentage:5.1f}%)")
        
        # Show a few examples
        if samples:
            print("  Examples:")
            for sample in samples[:3]:
                text_preview = sample['text'][:30] + '...' if len(sample['text']) > 30 else sample['text']
                print(f"    • \"{text_preview}\" (len: {sample['length']})")
    
    print("\n✅ Sample generation complete!")
    print(f"📁 Check the 'sample_images_output' directory for {total_samples} generated images")

def main():
    """Main function to generate sample images."""
    print("🚀 RandomWidthDataset Sample Image Generator")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Try to create dataset (real first, then mock)
    dataset, is_real = generate_samples_with_real_dataset()
    if not is_real:
        dataset = generate_samples_with_mock_dataset()
    
    # Generate and save samples
    samples_by_category, sample_info = generate_and_save_samples(
        dataset, output_dir, num_samples=60, is_real=is_real
    )
    
    # Create summary report
    create_summary_report(samples_by_category, sample_info, output_dir)
    
    # Analyze results
    analyze_generated_samples(samples_by_category)
    
    print(f"\n📋 Summary report saved to: {output_dir / 'README.md'}")
    print(f"📄 Detailed JSON report: {output_dir / 'sample_report.json'}")
    
    # Show file count by category
    print(f"\n📸 Image files generated:")
    for category in ["very_short_1_5", "short_6_10", "medium_11_30", "long_31_plus"]:
        category_dir = output_dir / category
        if category_dir.exists():
            file_count = len(list(category_dir.glob("*.png")))
            print(f"   {category_dir.name}/: {file_count} images")

if __name__ == "__main__":
    main() 