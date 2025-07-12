"""
Khmer Corpus Processing Pipeline

This module processes the corpus data from multiple sources (Gemini-generated and Wikipedia)
to create clean, formatted text lines for OCR training.
"""

import os
import re
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from src.khtext.khnormal_fast import khnormal
from src.khtext.subword_cluster import split_syllables_advanced

logger = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    """Statistics about processed corpus data"""
    total_files: int
    total_lines: int
    total_characters: int
    unique_characters: set
    avg_line_length: float
    max_line_length: int
    min_line_length: int
    sources: Dict[str, int]  # source -> line count


class KhmerCorpusProcessor:
    """
    Processes Khmer corpus data from multiple sources for OCR training.
    
    Handles text extraction, cleaning, filtering, and dataset preparation
    from Gemini-generated and Wikipedia corpus data.
    """
    
    def __init__(
        self,
        corpus_root: str = "corpus",
        min_line_length: int = 5,
        max_line_length: int = 150,
        test_split: float = 0.1,
        val_split: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize corpus processor
        
        Args:
            corpus_root: Root directory containing corpus folders
            min_line_length: Minimum characters per line
            max_line_length: Maximum characters per line
            test_split: Fraction for test set
            val_split: Fraction for validation set
            random_seed: Random seed for reproducibility
        """
        self.corpus_root = Path(corpus_root)
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length
        self.test_split = test_split
        self.val_split = val_split
        
        # Set random seed
        random.seed(random_seed)
        
        # Regex patterns for text cleaning
        self.cleanup_patterns = {
            'bullet_points': re.compile(r'^[•\-\*]\s*'),  # Remove bullet points
            'multiple_spaces': re.compile(r'\s+'),  # Multiple spaces -> single space
            'trailing_spaces': re.compile(r'\s+$'),  # Trailing spaces
            'leading_spaces': re.compile(r'^\s+'),  # Leading spaces
            'empty_parentheses': re.compile(r'\(\s*\)'),  # Empty parentheses
            'standalone_punctuation': re.compile(r'^\s*[។៖។]\s*$'),  # Lines with only punctuation
        }
        
        # Statistics tracking
        self.stats = None
        self.long_lines = 0

    def process_corpus(self, output_dir: str = "data/processed") -> CorpusStats:
        """
        Process all corpus data and create train/val/test splits
        
        Args:
            output_dir: Directory to save processed data
            
        Returns:
            CorpusStats: Statistics about processed corpus
        """
        logger.info("Starting corpus processing...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process all sources
        all_lines = []
        source_stats = {}
        
        # Process Gemini-generated data
        gemini_lines = self._process_gemini_data()
        all_lines.extend([(line, "gemini") for line in gemini_lines])
        source_stats["gemini"] = len(gemini_lines)
        logger.info(f"Processed Gemini data: {len(gemini_lines)} lines")
        
        # Process Wikipedia data
        wiki_lines = self._process_wikipedia_data()
        all_lines.extend([(line, "wikipedia") for line in wiki_lines])
        source_stats["wikipedia"] = len(wiki_lines)
        logger.info(f"Processed Wikipedia data: {len(wiki_lines)} lines")
        
        # Shuffle all lines
        random.shuffle(all_lines)
        
        # Create splits
        train_lines, val_lines, test_lines = self._create_splits([line for line, _ in all_lines])
        
        # Save splits
        self._save_lines(train_lines, output_path / "train.txt")
        self._save_lines(val_lines, output_path / "val.txt")
        self._save_lines(test_lines, output_path / "test.txt")
        
        # Calculate statistics
        all_text_lines = [line for line, _ in all_lines]
        self.stats = self._calculate_stats(all_text_lines, source_stats)
        
        # Save statistics
        self._save_stats(output_path / "corpus_stats.txt")
        
        logger.info(f"Corpus processing complete. Total lines: {len(all_text_lines)}")
        logger.info(f"Train: {len(train_lines)}, Val: {len(val_lines)}, Test: {len(test_lines)}")
        
        return self.stats
    
    def _process_gemini_data(self) -> List[str]:
        """Process Gemini-generated corpus data"""
        gemini_path = self.corpus_root / "gemini_generated"
        lines = []
        
        if not gemini_path.exists():
            logger.warning(f"Gemini data path not found: {gemini_path}")
            return lines
        
        # Process all subdirectories
        for domain_dir in gemini_path.iterdir():
            if domain_dir.is_dir():
                for file_path in domain_dir.glob("*.txt"):
                    file_lines = self._extract_lines_from_file(file_path)
                    lines.extend(file_lines)
                    logger.debug(f"Processed {file_path}: {len(file_lines)} lines")
        
        return lines
    
    def _process_wikipedia_data(self) -> List[str]:
        """Process Wikipedia corpus data"""
        wiki_path = self.corpus_root / "kmwiki_data"
        lines = []
        
        if not wiki_path.exists():
            logger.warning(f"Wikipedia data path not found: {wiki_path}")
            return lines
        
        # Process all text files
        for file_path in wiki_path.glob("*.txt"):
            file_lines = self._extract_lines_from_file(file_path)
            lines.extend(file_lines)
            logger.debug(f"Processed {file_path}: {len(file_lines)} lines")
        
        return lines
    
    def _split_lines(self, text: str, max_length: int) -> List[str]:
        """Split text into lines of maximum length"""
        lines = []
        # Split by various delimiters that indicate sentence/line boundaries
        raw_lines = re.split(r'[\n]', text)
        short_lines = [line for line in raw_lines if len(line) < max_length]
        long_lines = [line for line in raw_lines if len(line) >= max_length]
        
        # Split long lines into multiple lines
        for line in long_lines:
            # Split by spaces
            words = line.split(' ')
            new_line = ''
            
            for word in words:
                if len(new_line) + len(word) <= max_length:
                    new_line += word + ' '
                elif len(word) > max_length:
                    lines.append(new_line.strip())
                    new_line = ''
                    # split by syllables
                    syllables = split_syllables_advanced(word)
                    for syllable in syllables:
                        if len(new_line) + len(syllable) <= max_length:
                            new_line += syllable
                        else:
                            lines.append(new_line.strip())
                            new_line = syllable
                else:
                    lines.append(new_line.strip())
                    new_line = word + ' '
            
            # add the last line if it's not empty
            new_line = new_line.strip()
            if new_line and len(new_line) > 0:
                lines.append(new_line)
        
        # Add short lines
        lines.extend(short_lines)
        
        return lines
                
        

    def _extract_lines_from_file(self, file_path: Path) -> List[str]:
        """Extract and clean lines from a text file"""
        lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Normalize khmer text
                content = khnormal(content)

                # Replace double spaces with <SPACE>
                content = content.replace('  ', '<SPACE>')
                content = content.replace(' ', '')

                # Replace <SPACE> with space
                content = content.replace('<SPACE>', ' ')
                content = content.replace(' ។', '។')
                content = content.replace(' ៖', '៖')

                # Split by various delimiters that indicate sentence/line boundaries
                raw_lines = self._split_lines(content, self.max_line_length)
                
                for line in raw_lines:
                    cleaned_line = self._clean_line(line)
                    if self._is_valid_line(cleaned_line):
                        lines.append(cleaned_line)
                        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        
        return lines
    
    def _clean_line(self, line: str) -> str:
        """Clean and normalize a text line"""
        # Basic cleanup
        line = line.strip()
        
        # Apply cleanup patterns
        for pattern_name, pattern in self.cleanup_patterns.items():
            if pattern_name == 'bullet_points':
                line = pattern.sub('', line)
            elif pattern_name == 'multiple_spaces':
                line = pattern.sub(' ', line)
            elif pattern_name in ['trailing_spaces', 'leading_spaces']:
                line = pattern.sub('', line)
            elif pattern_name == 'empty_parentheses':
                line = pattern.sub('', line)
        
        # Remove extra whitespace
        line = ' '.join(line.split())
        
        return line
    
    def _is_valid_line(self, line: str) -> bool:
        """Check if a line is valid for training"""
        if not line:
            return False
        # check and record number of lines with longer than 150 characters
        if len(line) > 150:
            # logger.warning(f"Long line: {line}")
            self.long_lines += 1

        # Check length constraints
        if len(line) < self.min_line_length or len(line) > self.max_line_length:
            return False
        
        # Check for lines with only punctuation
        if self.cleanup_patterns['standalone_punctuation'].match(line):
            return False
        
        # Check for mostly non-Khmer content (basic heuristic)
        khmer_chars = sum(1 for char in line if '\u1780' <= char <= '\u17FF')
        if khmer_chars < len(line) * 0.5:  # At least 50% Khmer characters
            return False
        
        return True
    
    def _create_splits(self, lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Create train/validation/test splits"""
        total_lines = len(lines)
        
        # Calculate split indices
        test_size = int(total_lines * self.test_split)
        val_size = int(total_lines * self.val_split)
        train_size = total_lines - test_size - val_size
        
        # Create splits
        train_lines = lines[:train_size]
        val_lines = lines[train_size:train_size + val_size]
        test_lines = lines[train_size + val_size:]
        
        return train_lines, val_lines, test_lines
    
    def _save_lines(self, lines: List[str], file_path: Path):
        """Save lines to a text file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        
        logger.info(f"Saved {len(lines)} lines to {file_path}")
    
    def _calculate_stats(self, lines: List[str], source_stats: Dict[str, int]) -> CorpusStats:
        """Calculate corpus statistics"""
        if not lines:
            return CorpusStats(0, 0, 0, set(), 0.0, 0, 0, source_stats)
        
        total_characters = sum(len(line) for line in lines)
        unique_characters = set(''.join(lines))
        line_lengths = [len(line) for line in lines]
        
        return CorpusStats(
            total_files=len(source_stats),
            total_lines=len(lines),
            total_characters=total_characters,
            unique_characters=unique_characters,
            avg_line_length=total_characters / len(lines),
            max_line_length=max(line_lengths),
            min_line_length=min(line_lengths),
            sources=source_stats
        )
    
    def _save_stats(self, file_path: Path):
        """Save corpus statistics to file"""
        if not self.stats:
            return
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== Khmer Corpus Statistics ===\n\n")
            f.write(f"Total files processed: {self.stats.total_files}\n")
            f.write(f"Total lines: {self.stats.total_lines:,}\n")
            f.write(f"Total characters: {self.stats.total_characters:,}\n")
            f.write(f"Unique characters: {len(self.stats.unique_characters)}\n")
            f.write(f"Average line length: {self.stats.avg_line_length:.1f}\n")
            f.write(f"Max line length: {self.stats.max_line_length}\n")
            f.write(f"Min line length: {self.stats.min_line_length}\n\n")
            f.write(f"Long lines: {self.long_lines:,}\n")

            f.write("=== Source Distribution ===\n")
            for source, count in self.stats.sources.items():
                percentage = (count / self.stats.total_lines) * 100
                f.write(f"{source}: {count:,} lines ({percentage:.1f}%)\n")
            
            f.write(f"\n=== Character Set Sample ===\n")
            # Sample of unique characters
            char_sample = sorted(list(self.stats.unique_characters))
            f.write(''.join(char_sample))
    
    def load_processed_data(self, data_dir: str = "data/processed") -> Dict[str, List[str]]:
        """Load previously processed corpus data"""
        data_path = Path(data_dir)
        
        splits = {}
        
        # Handle training split - check for multiple files
        if (data_path / "train.txt").exists():
            # Single train.txt file
            with open(data_path / "train.txt", 'r', encoding='utf-8') as f:
                splits["train"] = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded train: {len(splits['train'])} lines")
        else:
            # Multiple training files (train_0.txt, train_1.txt, etc.)
            train_files = sorted(data_path.glob("train_*.txt"))
            if train_files:
                train_lines = []
                logger.info(f"Found {len(train_files)} training files: {[f.name for f in train_files]}")
                
                for train_file in train_files:
                    try:
                        with open(train_file, 'r', encoding='utf-8') as f:
                            file_lines = [line.strip() for line in f.readlines()]
                            train_lines.extend(file_lines)
                            logger.info(f"Loaded {len(file_lines)} lines from {train_file.name}")
                    except Exception as e:
                        logger.warning(f"Error loading {train_file}: {e}")
                        continue
                
                splits["train"] = train_lines
                logger.info(f"Total train lines: {len(splits['train'])}")
            else:
                logger.warning("No training files found (train.txt or train_*.txt)")
                splits["train"] = []
        
        # Handle val/test splits (single files)
        for split_name in ["val", "test"]:
            file_path = data_path / f"{split_name}.txt"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    splits[split_name] = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {split_name}: {len(splits[split_name])} lines")
            else:
                logger.warning(f"Split file not found: {file_path}")
                splits[split_name] = []
        
        return splits


def main():
    """Example usage of corpus processor"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process corpus
    processor = KhmerCorpusProcessor()
    stats = processor.process_corpus()
    
    print(f"\n=== Processing Complete ===")
    print(f"Total lines: {stats.total_lines:,}")
    print(f"Average line length: {stats.avg_line_length:.1f} characters")
    print(f"Character set size: {len(stats.unique_characters)}")
    
    for source, count in stats.sources.items():
        percentage = (count / stats.total_lines) * 100
        print(f"{source}: {count:,} lines ({percentage:.1f}%)")


if __name__ == "__main__":
    main() 