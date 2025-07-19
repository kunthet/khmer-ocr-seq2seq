#!/usr/bin/env python3
"""
Simple script to help inspect and understand the generated sample images.
This provides guidance on what to look for when examining the RandomWidthDataset output.
"""

import os
from pathlib import Path
import json

def inspect_sample_images():
    """Provide guidance for inspecting the generated sample images."""
    print("🔍 RandomWidthDataset Sample Images Inspection Guide")
    print("=" * 60)
    
    output_dir = Path("sample_images_output")
    
    if not output_dir.exists():
        print("❌ No sample images found!")
        print("   Run 'python generate_sample_images.py' first to create sample images.")
        return
    
    # Read the report if available
    report_file = output_dir / "sample_report.json"
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print("📊 Generation Summary:")
        print(f"   Total samples: {report['total_samples']}")
        print(f"   Length range: {report['generation_stats']['length_range']} characters")
        print(f"   Average length: {report['generation_stats']['average_length']} characters")
        
        print("\n📁 Distribution by Category:")
        for category, data in report['categories'].items():
            category_name = category.replace('_', ' ').title()
            print(f"   {category_name}: {data['count']} samples")
    
    print("\n🔍 What to Look For When Inspecting Images:")
    print("-" * 50)
    
    print("\n1. **Very Short Texts (1-5 characters)**")
    print("   • Should show single characters or very short words")
    print("   • Image width should be narrow (proportional to text)")
    print("   • Text should be clear and readable")
    print("   • Examples: Single Khmer characters, short syllables")
    
    print("\n2. **Short Texts (6-10 characters)**")
    print("   • Should show short words or phrases") 
    print("   • This is your target category for improvement")
    print("   • Text should be well-rendered without distortion")
    print("   • Examples: Common Khmer words, short phrases")
    
    print("\n3. **Medium Texts (11-30 characters)**")
    print("   • Should show phrases or short sentences")
    print("   • Good balance between content and image width")
    print("   • Text should flow naturally")
    
    print("\n4. **Long Texts (31+ characters)**")
    print("   • Should show longer sentences or phrases")
    print("   • Image width should expand to accommodate text")
    print("   • Text should remain readable across the width")
    
    print("\n✅ Quality Indicators to Check:")
    print("   ✓ Text is clearly visible and readable")
    print("   ✓ Khmer characters are properly formed")
    print("   ✓ No text cutoff or truncation issues")
    print("   ✓ Appropriate image width for text length")
    print("   ✓ Good contrast between text and background")
    print("   ✓ Consistent font rendering across samples")
    
    print("\n⚠️  Potential Issues to Watch For:")
    print("   ❌ Blurry or distorted characters")
    print("   ❌ Text too small or too large")
    print("   ❌ Inconsistent spacing")
    print("   ❌ Character rendering artifacts")
    print("   ❌ Background noise or distortion")
    
    print("\n🎯 Training Impact Assessment:")
    print("   • Focus on very short and short text quality")
    print("   • Check if short texts have sufficient variety")
    print("   • Ensure text is challenging but readable")
    print("   • Verify the exponential distribution is working")
    
    print(f"\n📂 Files are organized in: {output_dir.absolute()}")
    print("   You can open the PNG files with any image viewer")
    print("   Filenames include length and text content for easy identification")
    
    # Show some specific examples to open
    print("\n💡 Recommended Files to Inspect First:")
    categories = ["very_short_1_5", "short_6_10", "medium_11_30", "long_31_plus"]
    
    for category in categories:
        category_dir = output_dir / category
        if category_dir.exists():
            png_files = list(category_dir.glob("*.png"))
            if png_files:
                example_file = png_files[0]
                print(f"   {category}: {example_file}")
    
    print(f"\n📖 Full report available at: {output_dir / 'README.md'}")

def show_distribution_analysis():
    """Show analysis of the length distribution in generated samples."""
    output_dir = Path("sample_images_output")
    report_file = output_dir / "sample_report.json"
    
    if not report_file.exists():
        return
    
    print("\n📈 Length Distribution Analysis:")
    print("=" * 40)
    
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    dist = report['generation_stats']['length_distribution']
    total = sum(dist.values())
    
    print("Expected distribution for short text training:")
    print(f"   Very short (1-5):   {dist['1-5']:2d} samples ({dist['1-5']/total:.1%})")
    print(f"   Short (6-10):       {dist['6-10']:2d} samples ({dist['6-10']/total:.1%})")
    print(f"   Medium (11-30):     {dist['11-30']:2d} samples ({dist['11-30']/total:.1%})")
    print(f"   Long (31+):         {dist['31+']:2d} samples ({dist['31+']/total:.1%})")
    
    # Calculate improvement over typical dataset
    short_total = dist['1-5'] + dist['6-10']
    short_percentage = short_total / total
    
    print(f"\n✨ Short text representation: {short_percentage:.1%}")
    print("   (Compare to typical OCR datasets: ~5-15%)")
    
    if short_percentage > 0.4:
        print("   🎉 Excellent bias toward short texts!")
    elif short_percentage > 0.25:
        print("   ✅ Good bias toward short texts")
    else:
        print("   ⚠️  Consider increasing alpha for more short text bias")

def main():
    """Main inspection guide."""
    inspect_sample_images()
    show_distribution_analysis()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps for Training:")
    print("1. Visually inspect sample images for quality")
    print("2. If satisfied, integrate RandomWidthDataset into training")
    print("3. Monitor short text performance during training")
    print("4. Adjust alpha parameter if needed for more/fewer short texts")
    print("5. Consider curriculum learning with different alpha stages")

if __name__ == "__main__":
    main() 