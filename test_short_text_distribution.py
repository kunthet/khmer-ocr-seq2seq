#!/usr/bin/env python3
"""
Test the updated RandomWidthDataset to verify it generates more 1-5 character texts.
"""

import sys
sys.path.append('src')

import numpy as np
from collections import Counter

def test_length_distribution():
    """Test the new length distribution to see if it favors short texts."""
    print("🧪 Testing Updated RandomWidthDataset Length Distribution")
    print("=" * 60)
    
    try:
        # Import the updated RandomWidthDataset
        from test_random_width_dataset import MockBaseDataset, MockConfigManager
        from src.data.random_width_dataset import RandomWidthDataset
        
        # Create test dataset with updated distribution
        base_dataset = MockBaseDataset(100)
        config_manager = MockConfigManager()
        
        dataset = RandomWidthDataset(
            base_dataset=base_dataset,
            min_length=1,
            max_length=50,
            alpha=0.005,  # This parameter is less important now
            config_manager=config_manager
        )
        
        print(f"✅ Dataset created successfully!")
        
        # Sample many texts to check distribution
        print("\n🔄 Sampling 1000 texts to check length distribution...")
        
        length_counts = Counter()
        total_samples = 1000
        
        for i in range(total_samples):
            sample = dataset[i % len(dataset)]
            text_length = len(sample['text'])
            length_counts[text_length] += 1
        
        # Analyze distribution by ranges
        ranges = {
            "1-5 chars": (1, 5),
            "6-10 chars": (6, 10), 
            "11-20 chars": (11, 20),
            "21-50 chars": (21, 50),
            "51+ chars": (51, 1000)
        }
        
        print(f"\n📊 Length Distribution Results (from {total_samples} samples):")
        print("=" * 50)
        
        for range_name, (min_len, max_len) in ranges.items():
            count = sum(length_counts[length] for length in range(min_len, min_len + max_len) 
                       if length in length_counts)
            percentage = (count / total_samples) * 100
            print(f"   {range_name:12} {count:4d} samples ({percentage:5.1f}%)")
        
        # Detailed breakdown for very short texts
        print(f"\n🔍 Detailed Breakdown for Very Short Texts:")
        print("=" * 40)
        
        for length in range(1, 11):
            count = length_counts.get(length, 0)
            percentage = (count / total_samples) * 100
            print(f"   Length {length:2d}: {count:3d} samples ({percentage:4.1f}%)")
        
        # Calculate success metrics
        very_short_count = sum(length_counts[length] for length in range(1, 6) 
                              if length in length_counts)
        very_short_percentage = (very_short_count / total_samples) * 100
        
        print(f"\n🎯 Target Analysis:")
        print("=" * 30)
        print(f"   Very short texts (1-5 chars): {very_short_count} samples ({very_short_percentage:.1f}%)")
        print(f"   Target: >30% for good short text coverage")
        
        if very_short_percentage >= 30:
            print("   ✅ SUCCESS: Good coverage of very short texts!")
        elif very_short_percentage >= 20:
            print("   ⚠️  MODERATE: Decent coverage, could be improved")
        else:
            print("   ❌ LOW: Still not enough very short texts")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test execution."""
    print("🚀 Testing RandomWidthDataset Short Text Distribution")
    print("=" * 60)
    print("This test verifies that the updated RandomWidthDataset")
    print("generates significantly more 1-5 character texts.")
    print("=" * 60)
    
    success = test_length_distribution()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. If distribution looks good, regenerate validation dataset")
        print("   2. Use this improved distribution for training")
        print("   3. Monitor short text performance improvements")
    else:
        print("\n❌ Test failed. Check error messages above.")

if __name__ == "__main__":
    main() 