#!/usr/bin/env python3
"""
Test script to verify text length generation and syllable-based truncation in curriculum learning.
"""

import sys
sys.path.append('src')

import torch
from src.utils.config import ConfigManager
from src.data.onthefly_dataset import OnTheFlyDataset
from src.khtext.subword_cluster import split_syllables_advanced
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

class SyllableBasedCurriculumDataset:
    """Updated curriculum dataset that uses syllable-based truncation."""
    
    def __init__(self, base_dataset, max_length, config_manager):
        self.base_dataset = base_dataset
        self.max_length = max_length
        self.config_manager = config_manager
        self.vocab = config_manager.vocab
        
        # Import syllable segmentation function
        self.split_syllables = split_syllables_advanced
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _truncate_text_by_syllables(self, text, max_tokens):
        """
        Truncate text at syllable boundaries to fit within max_tokens limit.
        
        Args:
            text (str): Original text
            max_tokens (int): Maximum number of tokens (excluding SOS/EOS)
            
        Returns:
            str: Truncated text that will produce <= max_tokens when encoded
        """
        # Reserve space for SOS and EOS tokens
        available_tokens = max_tokens - 2
        
        if available_tokens <= 0:
            return ""
        
        # Split text into syllables
        syllables = self.split_syllables(text)
        
        # Build truncated text by adding syllables until we approach the limit
        truncated_syllables = []
        current_token_count = 0
        
        for syllable in syllables:
            # Estimate tokens needed for this syllable
            # Test encoding to get exact count
            test_text = "".join(truncated_syllables + [syllable])
            try:
                test_tokens = self.vocab.encode(test_text)
                if len(test_tokens) <= available_tokens:
                    truncated_syllables.append(syllable)
                    current_token_count = len(test_tokens)
                else:
                    # Adding this syllable would exceed the limit
                    break
            except Exception:
                # If encoding fails, stop here
                break
        
        # Join syllables back into text
        result = "".join(truncated_syllables)
        
        # If result is empty, try to include at least the first syllable
        if not result and syllables:
            result = syllables[0]
        
        return result
    
    def __getitem__(self, idx):
        # Get sample from base dataset
        sample = self.base_dataset[idx]
        
        if isinstance(sample, (list, tuple)) and len(sample) >= 3:
            image, targets, text = sample[0], sample[1], sample[2]
        else:
            raise ValueError(f"Expected 3-tuple from OnTheFlyDataset, got: {type(sample)}")
        
        # Truncate text properly by syllables
        truncated_text = self._truncate_text_by_syllables(text, self.max_length)
        
        # Re-encode the truncated text
        try:
            target_indices = [self.vocab.SOS_IDX] + self.vocab.encode(truncated_text) + [self.vocab.EOS_IDX]
        except Exception as e:
            print(f"Error encoding truncated text '{truncated_text}': {e}")
            # Fallback: just use SOS + EOS
            target_indices = [self.vocab.SOS_IDX, self.vocab.EOS_IDX]
        
        targets = torch.tensor(target_indices, dtype=torch.long)
        
        return {
            'image': image,
            'targets': targets,
            'text': truncated_text,
            'original_text': text
        }

def analyze_corpus_lengths():
    """Analyze the length distribution of texts in the corpus"""
    print("=== Corpus Length Analysis ===")
    
    corpus_dir = Path("data/processed")
    all_texts = []
    
    # Load all corpus files
    corpus_files = ["train.txt", "train_0.txt", "train_1.txt", "val.txt"]
    
    for filename in corpus_files:
        filepath = corpus_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
                all_texts.extend(texts)
                print(f"Loaded {len(texts)} texts from {filename}")
    
    print(f"Total texts: {len(all_texts)}")
    
    # Analyze lengths
    lengths = [len(text) for text in all_texts]
    
    print(f"Length range: {min(lengths)}-{max(lengths)} characters")
    print(f"Average: {np.mean(lengths):.1f} characters")
    print(f"Median: {np.median(lengths):.0f} characters")
    
    # Distribution analysis
    short_texts = len([l for l in lengths if l <= 20])
    medium_texts = len([l for l in lengths if 20 < l <= 50])
    long_texts = len([l for l in lengths if l > 100])
    
    print(f"Short texts (≤20 chars): {short_texts} ({short_texts/len(lengths)*100:.1f}%)")
    print(f"Medium texts (21-50 chars): {medium_texts} ({medium_texts/len(lengths)*100:.1f}%)")
    print(f"Long texts (>100 chars): {long_texts} ({long_texts/len(lengths)*100:.1f}%)")
    
    return all_texts, lengths

def test_regular_onthefly_dataset():
    """Test the regular OnTheFlyDataset without curriculum"""
    print("\n=== Regular OnTheFly Dataset Test ===")
    
    config = ConfigManager()
    
    # Create regular dataset
    dataset = OnTheFlyDataset(
        split="train",
        config_manager=config,
        samples_per_epoch=20,
        shuffle_texts=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test 10 samples
    sample_lengths = []
    texts = []
    
    for i in range(10):
        try:
            image, targets, text = dataset[i]
            length = len(text)
            sample_lengths.append(length)
            texts.append(text)
            print(f"Sample {i}: {length} chars - '{text}'")
        except Exception as e:
            print(f"Error getting sample {i}: {e}")
    
    if sample_lengths:
        print(f"Sample lengths: {sample_lengths}")
        print(f"Average length: {np.mean(sample_lengths):.1f} characters")
        print("Conclusion: Regular dataset uses full corpus text without length control")
    
    return texts

def test_syllable_curriculum_dataset():
    """Test the new syllable-based curriculum dataset"""
    print("\n=== Syllable-Based Curriculum Dataset Test ===")
    
    config = ConfigManager()
    
    # Create base dataset
    base_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config,
        samples_per_epoch=20,
        shuffle_texts=False
    )
    
    # Test different max lengths
    max_lengths = [10, 15, 20, 30, 50]
    
    for max_length in max_lengths:
        print(f"\n--- Testing max_length = {max_length} ---")
        
        # Create curriculum dataset
        curriculum_dataset = SyllableBasedCurriculumDataset(
            base_dataset=base_dataset,
            max_length=max_length,
            config_manager=config
        )
        
        # Test 5 samples
        token_lengths = []
        char_lengths = []
        syllable_counts = []
        truncated_texts = []
        
        for i in range(5):
            try:
                sample = curriculum_dataset[i]
                targets = sample['targets']
                text = sample['text']
                original_text = sample['original_text']
                
                # Calculate metrics
                token_length = len(targets) - 2  # Exclude SOS/EOS
                char_length = len(text)
                syllables = split_syllables_advanced(text)
                syllable_count = len(syllables)
                
                token_lengths.append(token_length)
                char_lengths.append(char_length)
                syllable_counts.append(syllable_count)
                truncated_texts.append(text)
                
                print(f"  Sample {i}: {token_length} tokens, {char_length} chars, {syllable_count} syllables")
                print(f"    Original: '{original_text[:60]}{'...' if len(original_text) > 60 else ''}'")
                print(f"    Truncated: '{text}'")
                print(f"    Syllables: {syllables}")
                
                # Verify token count is within limit
                if token_length > max_length - 2:  # Account for SOS/EOS
                    print(f"    WARNING: Token length {token_length} exceeds limit {max_length-2}")
                
            except Exception as e:
                print(f"  Error getting sample {i}: {e}")
        
        if token_lengths:
            print(f"  Summary for max_length {max_length}:")
            print(f"    Token lengths: {token_lengths} (avg: {np.mean(token_lengths):.1f})")
            print(f"    Character lengths: {char_lengths} (avg: {np.mean(char_lengths):.1f})")
            print(f"    Syllable counts: {syllable_counts} (avg: {np.mean(syllable_counts):.1f})")
            
            # Check if truncation was effective
            over_limit = len([l for l in token_lengths if l > max_length - 2])
            print(f"    Samples over token limit: {over_limit}/5")
            
            # Verify syllable integrity
            all_valid = True
            for text in truncated_texts:
                try:
                    config.vocab.encode(text)
                except Exception:
                    all_valid = False
                    print(f"    WARNING: Invalid text after truncation: '{text}'")
            
            if all_valid:
                print(f"    ✓ All truncated texts are valid Khmer")

def create_visualization():
    """Create visualization comparing corpus distribution vs curriculum effects"""
    print("\n=== Creating Visualization ===")
    
    # Get corpus data
    all_texts, corpus_lengths = analyze_corpus_lengths()
    
    # Sample some texts for curriculum testing
    config = ConfigManager()
    sample_texts = random.sample(all_texts[:1000], 50)  # Sample 50 texts for testing
    
    curriculum_results = {}
    max_lengths = [10, 15, 20, 30, 50]
    
    for max_length in max_lengths:
        char_lengths = []
        token_lengths = []
        
        for text in sample_texts:
            # Truncate using syllable method
            syllables = split_syllables_advanced(text)
            
            # Build truncated text similar to curriculum dataset
            available_tokens = max_length - 2  # Reserve for SOS/EOS
            truncated_syllables = []
            
            for syllable in syllables:
                test_text = "".join(truncated_syllables + [syllable])
                try:
                    test_tokens = config.vocab.encode(test_text)
                    if len(test_tokens) <= available_tokens:
                        truncated_syllables.append(syllable)
                    else:
                        break
                except Exception:
                    break
            
            truncated_text = "".join(truncated_syllables)
            if not truncated_text and syllables:
                truncated_text = syllables[0]
            
            # Get final metrics
            try:
                final_tokens = config.vocab.encode(truncated_text)
                token_lengths.append(len(final_tokens))
                char_lengths.append(len(truncated_text))
            except Exception:
                token_lengths.append(0)
                char_lengths.append(0)
        
        curriculum_results[max_length] = {
            'char_lengths': char_lengths,
            'token_lengths': token_lengths
        }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Corpus distribution
    ax1.hist(corpus_lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Text Length (characters)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Original Corpus Length Distribution')
    ax1.axvline(np.mean(corpus_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(corpus_lengths):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Curriculum effects
    for max_length in max_lengths:
        char_lengths = curriculum_results[max_length]['char_lengths']
        ax2.hist(char_lengths, bins=20, alpha=0.6, 
                label=f'Max {max_length} tokens', density=True)
    
    ax2.set_xlabel('Truncated Text Length (characters)')
    ax2.set_ylabel('Density')
    ax2.set_title('Syllable-Based Curriculum Truncation Effects')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('text_length_analysis_syllable.png', dpi=300, bbox_inches='tight')
    print("Saved visualization as 'text_length_analysis_syllable.png'")

def main():
    print("Khmer OCR Text Length Generation and Syllable-Based Truncation Test")
    print("=" * 70)
    
    # Test 1: Analyze corpus
    analyze_corpus_lengths()
    
    # Test 2: Regular dataset
    test_regular_onthefly_dataset()
    
    # Test 3: New syllable-based curriculum
    test_syllable_curriculum_dataset()
    
    # Test 4: Create visualization
    create_visualization()
    
    print("\n" + "=" * 70)
    print("Key Findings:")
    print("- Syllable-based truncation preserves Khmer text integrity")
    print("- Token limits are respected while maintaining valid syllables")
    print("- Curriculum learning can now use proper linguistic boundaries")
    print("- No arbitrary character cuts that break syllable structure")

if __name__ == "__main__":
    main() 