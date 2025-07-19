#!/usr/bin/env python3
"""
Khmer OCR Corpus Length Rebalancing Script

This script rebalances the training corpus by splitting long texts into smaller chunks
while respecting Khmer syllable boundaries using the split_syllables_advanced function.
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import datetime
import json

# Add src to path to import khtext modules
sys.path.append(str(Path(__file__).parent / "src"))
from khtext.subword_cluster import split_syllables_advanced

class CorpusRebalancer:
    """Rebalances corpus data by splitting texts into equal length classes."""
    
    def __init__(self, 
                 input_dir="data/processed", 
                 output_dir="data/processed_rebalanced",
                 backup_dir="data/processed_backup"):
        """
        Initialize the rebalancer.
        
        Args:
            input_dir (str): Directory containing original corpus files
            output_dir (str): Directory to save rebalanced corpus files
            backup_dir (str): Directory to backup original files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.backup_dir = Path(backup_dir)
        
        # Length classes with 5-character intervals (extended to 200)
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
            (151, 155, "class_151_155"),
            (156, 160, "class_156_160"),
            (161, 165, "class_161_165"),
            (166, 170, "class_166_170"),
            (171, 175, "class_171_175"),
            (176, 180, "class_176_180"),
            (181, 185, "class_181_185"),
            (186, 190, "class_186_190"),
            (191, 195, "class_191_195"),
            (196, 200, "class_196_200")
        ]
        
        # Target samples per class (will be calculated based on available data)
        self.target_samples_per_class = None
        
        # Data storage
        self.original_data = {}
        self.rebalanced_data = defaultdict(list)
        self.stats = {}
        
    def create_backup(self):
        """Create backup of original corpus files."""
        print("Creating backup of original corpus files...")
        
        # Create timestamped backup directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files to backup
        for file_path in self.input_dir.glob("*.txt"):
            if file_path.is_file():
                shutil.copy2(file_path, backup_path)
                print(f"  Backed up: {file_path.name}")
        
        print(f"Backup created at: {backup_path}")
        return backup_path
        
    def load_corpus_data(self):
        """Load corpus data from input files."""
        print("Loading corpus data...")
        
        for file_name in ["train_0.txt", "train_1.txt", "test.txt", "val.txt"]:
            file_path = self.input_dir / file_name
            if file_path.exists():
                print(f"Loading {file_name}...")
                
                lines = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            lines.append(line)
                            
                        if line_num % 10000 == 0:
                            print(f"  Loaded {line_num:,} lines from {file_name}")
                
                self.original_data[file_name] = lines
                print(f"  Total lines in {file_name}: {len(lines):,}")
        
        print(f"Total files loaded: {len(self.original_data)}")
        
    def split_text_by_syllables(self, text, target_length):
        """
        Split text into chunks of approximately target_length while respecting syllable boundaries.
        
        Args:
            text (str): Text to split
            target_length (int): Target length for each chunk
            
        Returns:
            list: List of text chunks
        """
        if len(text) <= target_length:
            return [text]
        
        # Get syllables using the advanced splitter
        syllables = split_syllables_advanced(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for syllable in syllables:
            syllable_length = len(syllable)
            
            # If adding this syllable would exceed target length and we have content
            if current_length + syllable_length > target_length and current_chunk:
                # Save current chunk
                chunk_text = ''.join(current_chunk)
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text)
                
                # Start new chunk
                current_chunk = [syllable]
                current_length = syllable_length
            else:
                # Add to current chunk
                current_chunk.append(syllable)
                current_length += syllable_length
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def split_syllables_advanced(self, text):
        """Helper method to import and use the advanced syllable splitter."""
        return split_syllables_advanced(text)
    
    def categorize_text_by_length(self, text):
        """
        Categorize text by its length into appropriate class.
        
        Args:
            text (str): Text to categorize
            
        Returns:
            str or None: Class name or None if text is too long
        """
        length = len(text)
        
        for min_len, max_len, class_name in self.length_classes:
            if min_len <= length <= max_len:
                return class_name
        
        # Text is longer than our defined classes
        return None
    
    def calculate_target_distribution(self):
        """Calculate target number of samples per class for balanced distribution."""
        print("Calculating target distribution...")
        
        # Sample a subset of data to estimate distribution (for efficiency)
        sample_size = 1000  # Sample 1000 lines from each file for estimation
        class_sample_counts = defaultdict(int)
        total_sampled = 0
        
        for file_name, lines in self.original_data.items():
            print(f"Sampling from {file_name}...")
            
            # Take a sample of lines for analysis
            import random
            sample_lines = random.sample(lines, min(sample_size, len(lines)))
            
            for line in sample_lines:
                # Try a few representative target lengths (extended range)
                for target_length in [7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 
                                     102, 107, 112, 117, 122, 127, 132, 137, 142, 147, 152, 157, 162, 167, 
                                     172, 177, 182, 187, 192, 197]:
                    chunks = self.split_text_by_syllables(line, target_length)
                    
                    for chunk in chunks:
                        class_name = self.categorize_text_by_length(chunk)
                        if class_name:
                            class_sample_counts[class_name] += 1
                            total_sampled += 1
        
        # Estimate total potential samples
        scaling_factor = sum(len(lines) for lines in self.original_data.values()) / (sample_size * len(self.original_data))
        estimated_total = int(total_sampled * scaling_factor)
        
        # Calculate target as reasonable distribution
        num_classes = len(self.length_classes)
        
        # Aim for at least 7500 samples per class to maintain training volume, but adjust based on available data
        min_samples_per_class = 7500  # Increased from 2500
        max_samples_per_class = estimated_total // num_classes
        
        self.target_samples_per_class = min(min_samples_per_class, max_samples_per_class)
        
        print(f"Estimated total potential samples: {estimated_total:,}")
        print(f"Target samples per class: {self.target_samples_per_class:,}")
        print(f"Sample class distribution (estimated):")
        for class_name in [cls[2] for cls in self.length_classes[:15]]:  # Show first 15
            count = int(class_sample_counts.get(class_name, 0) * scaling_factor)
            print(f"  {class_name}: ~{count:,} potential samples")
    
    def rebalance_dataset(self, file_name):
        """
        Rebalance a single dataset file.
        
        Args:
            file_name (str): Name of the file to rebalance
        """
        if file_name not in self.original_data:
            print(f"File {file_name} not found in loaded data")
            return
        
        print(f"Rebalancing {file_name}...")
        
        lines = self.original_data[file_name]
        class_samples = defaultdict(list)
        
        # Process each line multiple times with different strategies
        for line_idx, line in enumerate(lines):
            if line_idx % 1000 == 0 and line_idx > 0:
                print(f"  Processed {line_idx:,} lines...")
            
            # Strategy 1: Generate chunks for each class individually
            for min_len, max_len, class_name in self.length_classes:
                # Skip if we already have enough samples for this class
                if len(class_samples[class_name]) >= self.target_samples_per_class:
                    continue
                
                # Try multiple target lengths within the class range
                target_lengths = []
                step = max(1, (max_len - min_len) // 3)  # Try 3-4 different lengths per class
                for target in range(min_len, max_len + 1, step):
                    target_lengths.append(target)
                if max_len not in target_lengths:
                    target_lengths.append(max_len)
                
                for target_length in target_lengths:
                    if len(class_samples[class_name]) >= self.target_samples_per_class:
                        break
                        
                    # Generate chunks with this target length
                    chunks = self.split_text_by_syllables(line, target_length)
                    
                    # Add chunks that fit this class
                    for chunk in chunks:
                        if len(class_samples[class_name]) >= self.target_samples_per_class:
                            break
                        chunk_class = self.categorize_text_by_length(chunk)
                        if chunk_class == class_name:
                            class_samples[class_name].append(chunk)
            
            # Strategy 2: Overlapping windows for longer texts (if text > 100 chars)
            if len(line) > 100:
                for min_len, max_len, class_name in self.length_classes:
                    if len(class_samples[class_name]) >= self.target_samples_per_class:
                        continue
                    
                    # Create overlapping windows
                    target_length = (min_len + max_len) // 2
                    step_size = target_length // 2  # 50% overlap
                    
                    # Generate overlapping chunks
                    syllables = self.split_syllables_advanced(line)
                    if len(syllables) > 3:  # Only for texts with multiple syllables
                        current_pos = 0
                        current_length = 0
                        current_chunk = []
                        
                        for i, syllable in enumerate(syllables):
                            current_chunk.append(syllable)
                            current_length += len(syllable)
                            
                            # If we've reached target length, create a chunk
                            if current_length >= target_length - 5:  # Allow small tolerance
                                chunk_text = ''.join(current_chunk)
                                chunk_class = self.categorize_text_by_length(chunk_text)
                                
                                if chunk_class == class_name and len(class_samples[class_name]) < self.target_samples_per_class:
                                    class_samples[class_name].append(chunk_text)
                                
                                # Move window forward by step_size
                                chars_to_remove = 0
                                syllables_to_remove = 0
                                for j, syl in enumerate(current_chunk):
                                    chars_to_remove += len(syl)
                                    syllables_to_remove += 1
                                    if chars_to_remove >= step_size:
                                        break
                                
                                # Remove syllables from start
                                if syllables_to_remove < len(current_chunk):
                                    current_chunk = current_chunk[syllables_to_remove:]
                                    current_length = sum(len(s) for s in current_chunk)
                                else:
                                    current_chunk = []
                                    current_length = 0
        
        # Store rebalanced data for this file
        self.rebalanced_data[file_name] = class_samples
        
        # Print statistics
        print(f"  Rebalancing complete for {file_name}:")
        total_samples = sum(len(samples) for samples in class_samples.values())
        print(f"    Total samples generated: {total_samples:,}")
        print(f"    Samples per class (showing first 10):")
        for i, (min_len, max_len, class_name) in enumerate(self.length_classes[:10]):
            count = len(class_samples.get(class_name, []))
            if count > 0:
                print(f"      {class_name}: {count:,}")
        
        # Show classes that didn't reach target
        insufficient_classes = [name for name in [cls[2] for cls in self.length_classes] 
                              if len(class_samples.get(name, [])) < self.target_samples_per_class]
        if insufficient_classes:
            print(f"    Classes below target ({len(insufficient_classes)}): {', '.join(insufficient_classes[:5])}{'...' if len(insufficient_classes) > 5 else ''}")
    
    def save_rebalanced_data(self):
        """Save rebalanced data to output files."""
        print("Saving rebalanced data...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for file_name, class_samples in self.rebalanced_data.items():
            output_file = self.output_dir / file_name
            
            print(f"Saving {file_name}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write samples from all classes
                total_written = 0
                for class_name in [cls[2] for cls in self.length_classes]:
                    samples = class_samples.get(class_name, [])
                    for sample in samples:
                        f.write(sample + '\n')
                        total_written += 1
            
            print(f"  Saved {total_written:,} samples to {output_file}")
    
    def generate_analysis_report(self):
        """Generate analysis report of the rebalancing process."""
        print("Generating analysis report...")
        
        report_file = self.output_dir / "rebalancing_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("KHMER OCR CORPUS REBALANCING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Rebalancing Date: {datetime.datetime.now()}\n")
            f.write(f"Target Samples Per Class: {self.target_samples_per_class:,}\n\n")
            
            for file_name, class_samples in self.rebalanced_data.items():
                f.write(f"\n{file_name.upper()}:\n")
                f.write("-" * 40 + "\n")
                
                total_samples = sum(len(samples) for samples in class_samples.values())
                f.write(f"Total Samples: {total_samples:,}\n\n")
                
                f.write(f"{'Class':<15} {'Range':<10} {'Count':<10} {'%':<8}\n")
                f.write("-" * 50 + "\n")
                
                for min_len, max_len, class_name in self.length_classes:
                    count = len(class_samples.get(class_name, []))
                    percentage = (count / total_samples * 100) if total_samples > 0 else 0
                    range_str = f"{min_len}-{max_len}"
                    
                    if count > 0:
                        f.write(f"{class_name:<15} {range_str:<10} {count:<10,} {percentage:<7.1f}%\n")
        
        print(f"Report saved to: {report_file}")
    
    def run_rebalancing(self, files_to_process=None):
        """
        Run the complete rebalancing process.
        
        Args:
            files_to_process (list): List of file names to process, or None for all
        """
        print("Starting Khmer OCR Corpus Rebalancing...")
        
        # Create backup
        backup_path = self.create_backup()
        
        # Load data
        self.load_corpus_data()
        
        if not self.original_data:
            print("No data found! Please check that corpus files exist.")
            return
        
        # Calculate target distribution
        self.calculate_target_distribution()
        
        # Process files
        files_to_process = files_to_process or list(self.original_data.keys())
        
        for file_name in files_to_process:
            self.rebalance_dataset(file_name)
        
        # Save results
        self.save_rebalanced_data()
        
        # Generate report
        self.generate_analysis_report()
        
        print(f"\nRebalancing complete!")
        print(f"Original files backed up to: {backup_path}")
        print(f"Rebalanced files saved to: {self.output_dir}")
        print(f"Check '{self.output_dir}/rebalancing_report.txt' for detailed analysis.")


def main():
    """Main function to run the corpus rebalancing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rebalance Khmer OCR corpus data")
    parser.add_argument("--input-dir", default="data/processed", 
                       help="Input directory containing corpus files")
    parser.add_argument("--output-dir", default="data/processed_rebalanced", 
                       help="Output directory for rebalanced files")
    parser.add_argument("--backup-dir", default="data/processed_backup", 
                       help="Backup directory for original files")
    parser.add_argument("--files", nargs="*", 
                       help="Specific files to process (default: all .txt files)")
    parser.add_argument("--target-samples", type=int, 
                       help="Override target samples per class")
    
    args = parser.parse_args()
    
    rebalancer = CorpusRebalancer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backup_dir=args.backup_dir
    )
    
    if args.target_samples:
        rebalancer.target_samples_per_class = args.target_samples
    
    rebalancer.run_rebalancing(files_to_process=args.files)


if __name__ == "__main__":
    main() 