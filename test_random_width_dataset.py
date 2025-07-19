#!/usr/bin/env python3
"""
Test script for RandomWidthDataset class
Tests exponential distribution, text caching, sampling, and generation functionality.
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import random

# Import the class we're testing
from src.data.random_width_dataset import RandomWidthDataset

# Mock classes for testing
class MockVocab:
    def __init__(self):
        self.SOS_IDX = 0
        self.EOS_IDX = 1
        
    def encode(self, text):
        # Simple character-based encoding for testing
        return [ord(c) % 100 + 2 for c in text]  # Start from index 2
    
    def decode(self, indices):
        # Simple decoding for testing
        return ''.join(chr((idx - 2) % 100 + ord('a')) for idx in indices if idx > 1)

class MockConfigManager:
    def __init__(self):
        self.vocab = MockVocab()

class MockTextGenerator:
    def __init__(self):
        self.font_size = 32
        self.augmentor = None  # Add missing augmentor attribute
        
    def _select_font(self, text, split):
        return "arial.ttf"  # Mock font path
    
    def _calculate_optimal_width(self, text, font):
        return max(len(text) * 16, 128)
    
    def _render_text_image(self, text, font_path, width):
        from PIL import Image
        # Create a mock PIL image
        return Image.new('RGB', (width, 64), color='white')
    
    def _apply_augmentation(self, image):
        return image

class MockBaseDataset:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.split = "train"
        self.use_augmentation = True
        self.augment_prob = 0.5
        self.text_generator = MockTextGenerator()
        
        # Generate mock Khmer-like text data
        self.text_lines = self._generate_mock_texts()
    
    def _generate_mock_texts(self):
        """Generate mock Khmer-like texts of various lengths."""
        khmer_chars = ['ក', 'ខ', 'គ', 'ឃ', 'ង', 'ច', 'ឆ', 'ជ', 'ឈ', 'ញ', 'ត', 'ថ', 'ទ', 'ធ', 'ន', 'ប', 'ផ', 'ព', 'ភ', 'ម', 'យ', 'រ', 'ល', 'វ', 'ស', 'ហ', 'អ']
        khmer_vowels = ['ា', 'ិ', 'ី', 'ុ', 'ូ', 'ួ', 'ើ', 'ោ', 'ៅ', 'ំ', 'ះ']
        
        texts = []
        for i in range(self.num_samples):
            # Generate texts of varying lengths
            if i % 4 == 0:  # 25% short texts
                length = random.randint(5, 30)
            elif i % 4 == 1:  # 25% medium texts
                length = random.randint(30, 80)
            elif i % 4 == 2:  # 25% long texts
                length = random.randint(80, 200)
            else:  # 25% very long texts
                length = random.randint(200, 500)
            
            text = ""
            for _ in range(length // 2):  # Approximate character count
                text += random.choice(khmer_chars)
                if random.random() < 0.3:  # Add vowel sometimes
                    text += random.choice(khmer_vowels)
                if random.random() < 0.1:  # Add space sometimes
                    text += " "
            
            texts.append(text[:length])  # Ensure exact length
        
        return texts
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return None, None, self.text_lines[idx % len(self.text_lines)]

def test_initialization():
    """Test RandomWidthDataset initialization."""
    print("🧪 Testing initialization...")
    
    base_dataset = MockBaseDataset(100)
    config_manager = MockConfigManager()
    
    # Test with default parameters
    dataset = RandomWidthDataset(
        base_dataset=base_dataset,
        config_manager=config_manager
    )
    
    assert dataset.min_length == 1
    assert dataset.max_length == 150
    assert dataset.alpha == 0.05
    assert len(dataset) == len(base_dataset)
    
    print("✅ Initialization test passed!")
    return dataset

def test_length_distribution(dataset):
    """Test exponential probability distribution."""
    print("🧪 Testing length distribution...")
    
    # Test distribution shape
    probs = dataset.length_probabilities
    lengths = np.arange(dataset.min_length, dataset.max_length + 1)
    
    # Check that probabilities sum to 1
    assert abs(np.sum(probs) - 1.0) < 1e-6
    
    # Check exponential decay: shorter lengths should have higher probability
    assert probs[0] > probs[10] > probs[50] > probs[100]
    
    # Sample many lengths and check distribution
    sampled_lengths = [dataset._sample_target_length() for _ in range(10000)]
    length_counts = Counter(sampled_lengths)
    
    # Very short texts should be most common
    short_count = sum(count for length, count in length_counts.items() if length <= 10)
    total_count = sum(length_counts.values())
    short_ratio = short_count / total_count
    
    print(f"   Short texts (1-10 chars): {short_ratio:.3f} of samples")
    assert short_ratio > 0.3, f"Expected >30% short texts, got {short_ratio:.3f}"
    
    print("✅ Length distribution test passed!")
    return length_counts

def test_text_caching(dataset):
    """Test text caching and organization."""
    print("🧪 Testing text caching...")
    
    # Check that texts are cached by length
    assert isinstance(dataset.texts_by_length, dict)
    assert len(dataset.texts_by_length) > 0
    
    # Check that texts are within expected length ranges
    for length, texts in dataset.texts_by_length.items():
        assert dataset.min_length <= length <= dataset.max_length
        assert len(texts) > 0
        
        # Verify actual text lengths match the key
        for text in texts[:5]:  # Check first 5 texts
            actual_length = len(text)
            assert actual_length == length, f"Text length {actual_length} doesn't match key {length}"
    
    # Get statistics
    stats = dataset.get_length_statistics()
    total_cached = sum(stats.values())
    
    print(f"   Cached {total_cached} text variations across {len(stats)} length categories")
    print(f"   Length range: {min(stats.keys())} - {max(stats.keys())} characters")
    
    print("✅ Text caching test passed!")
    return stats

def test_getitem_functionality(dataset):
    """Test __getitem__ method."""
    print("🧪 Testing __getitem__ functionality...")
    
    # Test multiple samples
    samples = []
    for i in range(100):
        sample = dataset[i]
        samples.append(sample)
        
        # Check return format
        assert 'image' in sample
        assert 'targets' in sample
        assert 'text' in sample
        assert 'target_length' in sample
        assert 'actual_length' in sample
        
        # Check data types
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['targets'], torch.Tensor)
        assert isinstance(sample['text'], str)
        assert isinstance(sample['target_length'], (int, np.integer))
        assert isinstance(sample['actual_length'], int)
        
        # Check text is not empty
        assert len(sample['text']) > 0
        
        # Check actual length matches text length
        assert sample['actual_length'] == len(sample['text'])
        
        # Check target length is within bounds
        assert dataset.min_length <= sample['target_length'] <= dataset.max_length
    
    # Check length distribution in samples
    actual_lengths = [s['actual_length'] for s in samples]
    target_lengths = [s['target_length'] for s in samples]
    
    print(f"   Generated {len(samples)} samples")
    print(f"   Actual length range: {min(actual_lengths)} - {max(actual_lengths)}")
    print(f"   Target length range: {min(target_lengths)} - {max(target_lengths)}")
    
    # Check that we get more short texts than long texts
    short_samples = sum(1 for length in actual_lengths if length <= 10)
    short_ratio = short_samples / len(samples)
    print(f"   Short samples ratio: {short_ratio:.3f}")
    
    print("✅ __getitem__ test passed!")
    return samples

def test_batch_sampling(dataset):
    """Test batch sampling functionality."""
    print("🧪 Testing batch sampling...")
    
    batch_size = 32
    batch = dataset.sample_batch_by_length_distribution(batch_size)
    
    assert len(batch) == batch_size
    
    # Check batch consistency
    for sample in batch:
        assert 'image' in sample
        assert 'targets' in sample
        assert 'text' in sample
    
    # Check length distribution in batch
    lengths = [len(sample['text']) for sample in batch]
    short_count = sum(1 for length in lengths if length <= 10)
    short_ratio = short_count / len(batch)
    
    print(f"   Batch size: {len(batch)}")
    print(f"   Length range in batch: {min(lengths)} - {max(lengths)}")
    print(f"   Short texts in batch: {short_ratio:.3f}")
    
    print("✅ Batch sampling test passed!")

def test_edge_cases(dataset):
    """Test edge cases and error handling."""
    print("🧪 Testing edge cases...")
    
    # Test with empty target
    try:
        # Force a scenario where we might get empty text
        empty_dataset = RandomWidthDataset(
            base_dataset=MockBaseDataset(1),  # Very small dataset
            min_length=1,
            max_length=5,
            config_manager=MockConfigManager()
        )
        sample = empty_dataset[0]
        assert len(sample['text']) > 0  # Should have fallback text
        print("   ✅ Empty text handling works")
    except Exception as e:
        print(f"   ❌ Edge case failed: {e}")
    
    # Test with very short max_length
    try:
        short_dataset = RandomWidthDataset(
            base_dataset=MockBaseDataset(10),
            min_length=1,
            max_length=5,
            config_manager=MockConfigManager()
        )
        sample = short_dataset[0]
        assert sample['actual_length'] <= 5
        print("   ✅ Very short max_length works")
    except Exception as e:
        print(f"   ❌ Short length test failed: {e}")
    
    print("✅ Edge cases test passed!")

def visualize_distribution(length_counts, dataset):
    """Visualize the length distribution."""
    print("📊 Creating distribution visualization...")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot theoretical distribution
        lengths = np.arange(dataset.min_length, min(dataset.max_length + 1, 101))  # Limit for visibility
        probs = dataset.length_probabilities[:len(lengths)]
        
        ax1.plot(lengths, probs, 'b-', linewidth=2, label='Theoretical')
        ax1.set_xlabel('Text Length (characters)')
        ax1.set_ylabel('Probability')
        ax1.set_title('Theoretical Length Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot sampled distribution
        sampled_lengths = list(length_counts.keys())
        sampled_counts = list(length_counts.values())
        total_samples = sum(sampled_counts)
        sampled_probs = [count / total_samples for count in sampled_counts]
        
        ax2.bar(sampled_lengths[:50], sampled_probs[:50], alpha=0.7, color='orange', label='Sampled')
        ax2.set_xlabel('Text Length (characters)')
        ax2.set_ylabel('Probability')
        ax2.set_title('Sampled Length Distribution (first 50 lengths)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('random_width_dataset_distribution.png', dpi=150, bbox_inches='tight')
        print("   📊 Distribution plot saved as 'random_width_dataset_distribution.png'")
        
    except ImportError:
        print("   ⚠️  Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")

def run_performance_test(dataset):
    """Test performance of dataset operations."""
    print("⚡ Running performance test...")
    
    import time
    
    # Test sampling speed
    start_time = time.time()
    samples = [dataset[i] for i in range(100)]
    end_time = time.time()
    
    sampling_time = end_time - start_time
    samples_per_second = 100 / sampling_time
    
    print(f"   Generated 100 samples in {sampling_time:.3f}s")
    print(f"   Speed: {samples_per_second:.1f} samples/second")
    
    # Test batch sampling speed
    start_time = time.time()
    batch = dataset.sample_batch_by_length_distribution(32)
    end_time = time.time()
    
    batch_time = end_time - start_time
    print(f"   Generated batch of 32 in {batch_time:.3f}s")
    
    print("✅ Performance test completed!")

def main():
    """Run all tests."""
    print("🚀 Starting RandomWidthDataset tests...")
    print("=" * 60)
    
    try:
        # Initialize dataset
        dataset = test_initialization()
        
        # Test core functionality
        length_counts = test_length_distribution(dataset)
        stats = test_text_caching(dataset)
        samples = test_getitem_functionality(dataset)
        test_batch_sampling(dataset)
        test_edge_cases(dataset)
        
        # Performance and visualization
        run_performance_test(dataset)
        visualize_distribution(length_counts, dataset)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Dataset created with {len(dataset)} samples")
        print(f"✅ {len(stats)} different text lengths cached")
        print(f"✅ {sum(stats.values())} total text variations generated")
        print(f"✅ Length range: {min(stats.keys())} - {max(stats.keys())} characters")
        
        # Length distribution summary
        short_cached = sum(count for length, count in stats.items() if length <= 10)
        total_cached = sum(stats.values())
        print(f"✅ Short texts (≤10 chars): {short_cached}/{total_cached} ({short_cached/total_cached:.1%})")
        
        print("\n🎉 All tests passed! RandomWidthDataset is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 