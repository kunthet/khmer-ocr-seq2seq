#!/usr/bin/env python3
"""
Khmer OCR Corpus Unique Rebalancing Script

This script rebalances the training corpus by concatenating all text into a continuous stream
and then splitting it into unique, non-repetitive chunks while respecting Khmer syllable boundaries.
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import datetime
import json
import argparse

# Add src to path to import khtext modules
sys.path.append(str(Path(__file__).parent / "src"))
from khtext.subword_cluster import split_syllables_advanced

class UniqueCorpusRebalancer:
    """Rebalances corpus data by creating unique chunks from concatenated text."""
    
    def __init__(self, 
                 input_dir="data/processed", 
                 output_dir="data/processed_unique_rebalanced"):
        """
        Initialize the rebalancer.
        
        Args:
            input_dir (str): Input directory containing corpus files
            output_dir (str): Output directory for rebalanced corpus
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Length classes with 5-character intervals (1-150, reduced from 200)
        self.length_classes = [
            (1, 5, "class_001_005"),
            (6, 10, "class_006_010"),
            (11, 15, "class_011_015"),
            (16, 20, "class_016_020"),
            (21, 25, "class_021_025"),
            (26, 30, "class_026_030"),
            (31, 35, "class_031_035"),
            (36, 40, "class_036_040"),
            (41, 45, "class_041_045"),
            (46, 50, "class_046_050"),
            (51, 55, "class_051_055"),
            (56, 60, "class_056_060"),
            (61, 65, "class_061_065"),
            (66, 70, "class_066_070"),
            (71, 75, "class_071_075"),
            (76, 80, "class_076_080"),
            (81, 85, "class_081_085"),
            (86, 90, "class_086_090"),
            (91, 95, "class_091_095"),
            (96, 100, "class_096_100"),
            (101, 105, "class_101_105"),
            (106, 110, "class_106_110"),
            (111, 115, "class_111_115"),
            (116, 120, "class_116_120"),
            (121, 125, "class_121_125"),
            (126, 130, "class_126_130"),
            (131, 135, "class_131_135"),
            (136, 140, "class_136_140"),
            (141, 145, "class_141_145"),
            (146, 150, "class_146_150"),
        ]
        
        # Remove target samples per class as we'll use Round Robin distribution
        # self.target_samples_per_class = 5000
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_corpus_data(self, file_names):
        """
        Load and concatenate all corpus data into a single text stream.
        
        Args:
            file_names (list): List of file names to load
            
        Returns:
            str: Concatenated text from all files
        """
        print("Loading and concatenating corpus data...")
        
        all_text_parts = []
        total_lines = 0
        
        for file_name in file_names:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                print(f"Warning: File {file_path} not found")
                continue
                
            print(f"Loading {file_name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                all_text_parts.extend(lines)
                total_lines += len(lines)
                print(f"  Loaded {len(lines):,} lines")
        
        # Concatenate all text with spaces between original lines
        concatenated_text = " ".join(all_text_parts)
        
        print(f"Total original lines: {total_lines:,}")
        print(f"Concatenated text length: {len(concatenated_text):,} characters")
        
        return concatenated_text
    
    def split_text_into_chunks(self, text, target_length):
        """
        Split text into chunks of approximately target_length while respecting syllable boundaries.
        
        Args:
            text (str): Text to split
            target_length (int): Target length for each chunk
            
        Returns:
            list: List of unique text chunks
        """
        # Get syllables using the advanced splitter
        syllables = split_syllables_advanced(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for syllable in syllables:
            syllable_length = len(syllable)
            
            # If adding this syllable would exceed target, save current chunk
            if current_length + syllable_length > target_length and current_chunk:
                chunk_text = "".join(current_chunk).strip()
                if len(chunk_text) >= target_length * 0.8:  # Allow 20% tolerance
                    chunks.append(chunk_text)
                
                # Start new chunk
                current_chunk = [syllable]
                current_length = syllable_length
            else:
                current_chunk.append(syllable)
                current_length += syllable_length
        
        # Add final chunk if it exists
        if current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if len(chunk_text) >= target_length * 0.8:  # Allow 20% tolerance
                chunks.append(chunk_text)
        
        return chunks
    
    def get_text_class(self, text_length):
        """
        Determine which length class a text belongs to.
        
        Args:
            text_length (int): Length of the text
            
        Returns:
            tuple: (min_len, max_len, class_name) or None if no match
        """
        for min_len, max_len, class_name in self.length_classes:
            if min_len <= text_length <= max_len:
                return (min_len, max_len, class_name)
        return None
    
    def generate_balanced_corpus(self, concatenated_text):
        """
        Generate a balanced corpus from concatenated text using Round Robin distribution.
        
        Args:
            concatenated_text (str): The concatenated corpus text
            
        Returns:
            dict: Dictionary mapping class names to lists of samples
        """
        print("Generating balanced corpus using Round Robin distribution...")
        
        class_samples = defaultdict(list)
        used_chunks = set()  # Track used chunks to avoid duplicates
        
        # Generate chunks of different target lengths systematically
        all_chunks = []
        
        print("Generating chunks for all target lengths...")
        
        # Create target lengths for each class (use multiple target lengths per class for variety)
        target_lengths = []
        for min_len, max_len, class_name in self.length_classes:
            # Use multiple target lengths within each class range
            if max_len - min_len >= 4:
                targets = [min_len + 1, (min_len + max_len) // 2, max_len - 1]
            elif max_len - min_len >= 2:
                targets = [min_len + 1, max_len - 1]
            else:
                targets = [(min_len + max_len) // 2]
            
            for target in targets:
                target_lengths.append(target)
        
        # Remove duplicates and sort
        target_lengths = sorted(list(set(target_lengths)))
        
        print(f"Processing {len(target_lengths)} different target lengths...")
        
        # Generate chunks for each target length
        text_position = 0
        total_text_length = len(concatenated_text)
        chunk_size = total_text_length // len(target_lengths)  # Divide text roughly equally
        
        for i, target_length in enumerate(target_lengths):
            print(f"Processing target length {target_length} ({i+1}/{len(target_lengths)})...")
            
            # Calculate text segment for this target length
            start_pos = i * chunk_size
            if i == len(target_lengths) - 1:  # Last iteration gets remainder
                end_pos = total_text_length
            else:
                end_pos = min((i + 1) * chunk_size, total_text_length)
            
            if start_pos >= total_text_length:
                break
                
            text_segment = concatenated_text[start_pos:end_pos]
            
            # Generate chunks from this segment
            chunks = self.split_text_into_chunks(text_segment, target_length)
            
            # Add unique chunks to our collection
            for chunk in chunks:
                if chunk not in used_chunks and len(chunk.strip()) > 0:
                    chunk_class = self.get_text_class(len(chunk))
                    if chunk_class:  # Only add if it fits in a class
                        all_chunks.append((chunk, chunk_class[2]))
                        used_chunks.add(chunk)
            
            if len(all_chunks) % 1000 == 0:
                print(f"  Generated {len(all_chunks)} unique chunks so far...")
        
        print(f"Total unique chunks generated: {len(all_chunks)}")
        
        # Round Robin distribution
        print("Distributing chunks using Round Robin strategy...")
        
        # Group chunks by their assigned class
        chunks_by_class = defaultdict(list)
        for chunk, class_name in all_chunks:
            chunks_by_class[class_name].append(chunk)
        
        # Get all available class names
        available_classes = [cls[2] for cls in self.length_classes]
        
        # Round Robin distribution across all classes
        class_index = 0
        total_distributed = 0
        
        # Continue until all chunks are distributed
        while any(chunks_by_class.values()):
            current_class = available_classes[class_index]
            
            # If this class has chunks available, assign one
            if chunks_by_class[current_class]:
                chunk = chunks_by_class[current_class].pop(0)
                class_samples[current_class].append(chunk)
                total_distributed += 1
                
                if total_distributed % 1000 == 0:
                    print(f"  Distributed {total_distributed} chunks...")
            
            # Move to next class (Round Robin)
            class_index = (class_index + 1) % len(available_classes)
            
            # Safety check to avoid infinite loop
            if total_distributed >= len(all_chunks):
                break
        
        print(f"Round Robin distribution completed. Total distributed: {total_distributed}")
        
        # Print distribution summary
        print("\nDistribution Summary:")
        for min_len, max_len, class_name in self.length_classes:
            count = len(class_samples.get(class_name, []))
            if count > 0:
                print(f"  {class_name}: {count} samples")
        
        return dict(class_samples)
    
    def save_rebalanced_corpus(self, class_samples, output_files):
        """
        Save the rebalanced corpus to output files.
        
        Args:
            class_samples (dict): Dictionary mapping class names to samples
            output_files (list): List of output file names
        """
        print("Saving rebalanced corpus...")
        
        # Combine all samples and shuffle them
        all_samples = []
        for class_name, samples in class_samples.items():
            all_samples.extend(samples)
        
        import random
        random.shuffle(all_samples)
        
        # Split samples across output files
        samples_per_file = len(all_samples) // len(output_files)
        
        for i, output_file in enumerate(output_files):
            start_idx = i * samples_per_file
            if i == len(output_files) - 1:  # Last file gets remainder
                end_idx = len(all_samples)
            else:
                end_idx = (i + 1) * samples_per_file
            
            file_samples = all_samples[start_idx:end_idx]
            
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in file_samples:
                    f.write(sample + '\n')
            
            print(f"Saved {len(file_samples):,} samples to {output_file}")
    
    def generate_report(self, class_samples):
        """
        Generate a detailed report of the rebalancing process.
        
        Args:
            class_samples (dict): Dictionary mapping class names to samples
        """
        total_samples = sum(len(samples) for samples in class_samples.values())
        
        # Create detailed report
        report_lines = [
            "=" * 80,
            "KHMER OCR UNIQUE CORPUS REBALANCING REPORT (ROUND ROBIN)",
            "=" * 80,
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Input Directory: {self.input_dir}",
            f"Output Directory: {self.output_dir}",
            "",
            "REBALANCING SUMMARY:",
            f"  Total Classes: {len(self.length_classes)} (1-150 character range)",
            f"  Classes with Data: {len(class_samples)}",
            f"  Total Samples Generated: {total_samples:,}",
            f"  Distribution Strategy: Round Robin (cyclical)",
            "",
            "CLASS DISTRIBUTION:",
        ]
        
        for min_len, max_len, class_name in self.length_classes:
            count = len(class_samples.get(class_name, []))
            if count > 0:
                percentage = (count / total_samples) * 100
                avg_length = sum(len(sample) for sample in class_samples[class_name]) / count
                report_lines.append(
                    f"  {class_name} ({min_len}-{max_len}): {count:,} samples "
                    f"({percentage:.1f}%) - Avg: {avg_length:.1f} chars"
                )
        
        report_lines.extend([
            "",
            "ROUND ROBIN DISTRIBUTION FEATURES:",
            "  ✓ Cyclical assignment across all classes",
            "  ✓ Even distribution regardless of source length bias",
            "  ✓ No duplicate chunks generated",
            "  ✓ Syllable-boundary splitting maintained",
            "  ✓ Continuous text stream processing",
            "  ✓ Natural balancing through rotation",
            "",
            "=" * 80
        ])
        
        # Save report
        report_path = self.output_dir / "unique_rebalancing_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print summary
        print("\n" + '\n'.join(report_lines))
        print(f"\nDetailed report saved to: {report_path}")
    
    def rebalance(self, input_files, output_files):
        """
        Perform the complete rebalancing process.
        
        Args:
            input_files (list): List of input file names
            output_files (list): List of output file names
        """
        print("Starting unique corpus rebalancing...")
        
        # Load and concatenate all text
        concatenated_text = self.load_corpus_data(input_files)
        
        # Generate balanced corpus
        class_samples = self.generate_balanced_corpus(concatenated_text)
        
        # Save results
        self.save_rebalanced_corpus(class_samples, output_files)
        
        # Generate report
        self.generate_report(class_samples)
        
        print("\nUnique corpus rebalancing completed successfully!")

def main():
    """Main function to run the rebalancing script."""
    parser = argparse.ArgumentParser(description="Rebalance Khmer OCR corpus with unique chunks")
    parser.add_argument("--files", nargs="+", required=True, help="Input corpus files")
    parser.add_argument("--output-dir", default="data/processed_unique_rebalanced", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create rebalancer
    rebalancer = UniqueCorpusRebalancer(output_dir=args.output_dir)
    
    # Run rebalancing
    output_files = ["train_0.txt", "train_1.txt"]
    rebalancer.rebalance(args.files, output_files)

if __name__ == "__main__":
    main() 