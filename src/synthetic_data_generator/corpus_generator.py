"""
Corpus-based data generator for Khmer OCR that uses real text corpus for authentic generation.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import yaml
from tqdm import tqdm

from .base_generator import BaseDataGenerator
from .utils import load_khmer_corpus, generate_corpus_based_text


class CorpusGenerator(BaseDataGenerator):
    """
    Generates training data for Khmer OCR using corpus-based text generation.
    """
    
    def __init__(self, 
                 config_path: str,
                 fonts_dir: str,
                 output_dir: str,
                 mode: str = "full_text"):
        """
        Initialize the corpus data generator.
        
        Args:
            config_path: Path to model configuration file
            fonts_dir: Directory containing Khmer fonts
            output_dir: Directory to save generated data
            mode: Generation mode ('digits', 'full_text', 'mixed')
        """
        super().__init__(config_path, fonts_dir, output_dir, mode)
        
        # Load corpus for authentic text generation
        self.corpus_lines = None
        if mode in ["full_text", "mixed"]:
            self.corpus_lines = load_khmer_corpus()
            if self.corpus_lines:
                print(f"✅ Corpus loaded: {len(self.corpus_lines)} lines for authentic text generation")
            else:
                raise ValueError("⚠️ Corpus not available. Cannot initialize CorpusGenerator without a corpus.")
        else:
            print("✅ Corpus generator initialized - mode is 'digits', no corpus needed")
        
        print("✅ Corpus generator initialized - using authentic text from corpus")
    
    def _generate_text_content(self, 
                              content_type: str = "auto",
                              length_range: Tuple[int, int] = (1, 15),
                              allowed_characters: Optional[List[str]] = None) -> str:
        """
        Generate corpus-based text content.
        
        Args:
            content_type: Type of content to generate
            length_range: Range of text lengths
            allowed_characters: Allowed characters for curriculum learning
            
        Returns:
            Generated text content
        """
        target_length = random.randint(*length_range)
        
        if self.mode == "digits" or content_type == "digits":
            return self.khmer_generator.generate_digit_sequence(1, min(8, target_length))
        
        if not self.corpus_lines:
            # Fallback to synthetic generation if corpus is not available
            print("⚠️ Corpus not available, falling back to synthetic generation")
            return self.khmer_generator.generate_content_by_type(
                content_type=content_type,
                length=target_length,
                allowed_characters=allowed_characters
            )
        
        # Use corpus-based generation with retries for validation
        num_retries = 50
        for i in range(num_retries):
            corpus_text = generate_corpus_based_text(
                corpus_lines=self.corpus_lines,
                target_length=target_length,
                content_type=content_type,
                allowed_characters=allowed_characters
            )
            
            # Validate the corpus text meets our requirements and linguistic rules
            if corpus_text and len(corpus_text) >= 1:
                is_valid, error = self.khmer_generator.validate_khmer_text_structure(corpus_text)
                if is_valid:
                    return corpus_text
                else:
                    if i < 5:  # Only print first few errors to avoid spam
                        print(f"⚠️ Extracted corpus text invalid: {error}. Retrying ({i+1}/{num_retries})...")
                    continue
        
        # If corpus generation fails, fallback to synthetic
        print(f"⚠️ Failed to generate valid corpus text after {num_retries} retries. Using synthetic generation.")
        return self.khmer_generator.generate_content_by_type(
            content_type=content_type,
            length=target_length,
            allowed_characters=allowed_characters
        )
    
    def generate_corpus_comparison_dataset(self,
                                         num_samples: int = 1000,
                                         train_split: float = 0.8,
                                         save_images: bool = True,
                                         show_progress: bool = True) -> Dict:
        """
        Generate dataset comparing corpus-based vs synthetic text generation.
        
        Args:
            num_samples: Total number of samples to generate (split between corpus and synthetic)
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with dataset information
        """
        print("Generating corpus comparison dataset (50% corpus, 50% synthetic)")
        
        # Split samples between corpus and synthetic
        num_corpus = num_samples // 2
        num_synthetic = num_samples - num_corpus
        
        # Calculate train/val splits
        num_train_corpus = int(num_corpus * train_split)
        num_val_corpus = num_corpus - num_train_corpus
        num_train_synthetic = int(num_synthetic * train_split)
        num_val_synthetic = num_synthetic - num_train_synthetic
        
        # Create directories
        corpus_dir = Path(self.output_dir) / 'corpus_based'
        synthetic_dir = Path(self.output_dir) / 'synthetic'
        
        if save_images:
            for base_dir in [corpus_dir, synthetic_dir]:
                (base_dir / 'train').mkdir(parents=True, exist_ok=True)
                (base_dir / 'val').mkdir(parents=True, exist_ok=True)
        
        all_metadata = {
            'corpus_based': {'train': {'samples': []}, 'val': {'samples': []}},
            'synthetic': {'train': {'samples': []}, 'val': {'samples': []}}
        }
        
        # Progress bar setup
        total_progress = tqdm(total=num_samples, desc="Generating comparison dataset") if show_progress else None
        
        # Generate corpus-based samples
        self._generate_comparison_split(
            'corpus_based', num_train_corpus, num_val_corpus, 
            corpus_dir, all_metadata['corpus_based'], total_progress, save_images, True
        )
        
        # Generate synthetic samples (fallback to synthetic generation)
        self._generate_comparison_split(
            'synthetic', num_train_synthetic, num_val_synthetic,
            synthetic_dir, all_metadata['synthetic'], total_progress, save_images, False
        )
        
        if total_progress:
            total_progress.close()
        
        # Add dataset-level metadata
        dataset_info = {
            'comparison_type': 'corpus_vs_synthetic',
            'corpus_samples': num_corpus,
            'synthetic_samples': num_synthetic,
            'total_samples': num_samples,
            'train_samples': num_train_corpus + num_train_synthetic,
            'val_samples': num_val_corpus + num_val_synthetic,
            'image_size': list(self.image_size),
            'fonts_used': list(self.working_fonts.keys()),
            'generator_type': 'CorpusGenerator',
            'generated_by': 'CorpusGenerator v2.0 - Comparison Dataset'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata for each type
        if save_images:
            for data_type, data_dir in [('corpus_based', corpus_dir), ('synthetic', synthetic_dir)]:
                metadata_path = data_dir / 'metadata.yaml'
                type_metadata = {
                    'train': all_metadata[data_type]['train'],
                    'val': all_metadata[data_type]['val'],
                    'dataset_info': {**dataset_info, 'data_type': data_type}
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    yaml.dump(type_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nComparison dataset generated successfully!")
        print(f"Corpus-based samples: {num_corpus}")
        print(f"Synthetic samples: {num_synthetic}")
        print(f"Total samples: {num_samples}")
        
        return all_metadata
    
    def _generate_comparison_split(self, data_type: str, num_train: int, num_val: int, 
                                 output_dir: Path, metadata_dict: Dict, 
                                 progress_bar, save_images: bool, use_corpus: bool):
        """Generate train/val split for comparison dataset."""
        
        # Temporarily modify corpus usage for synthetic comparison
        original_corpus = self.corpus_lines
        if not use_corpus:
            self.corpus_lines = None
        
        try:
            # Generate training samples
            for i in range(num_train):
                image, metadata = self.generate_single_image()
                
                if save_images:
                    image_filename = f"train_{i:06d}.png"
                    image_path = output_dir / 'train' / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['data_type'] = data_type
                metadata_dict['train']['samples'].append(metadata)
                
                if progress_bar:
                    progress_bar.update(1)
            
            # Generate validation samples
            for i in range(num_val):
                image, metadata = self.generate_single_image()
                
                if save_images:
                    image_filename = f"val_{i:06d}.png"
                    image_path = output_dir / 'val' / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['data_type'] = data_type
                metadata_dict['val']['samples'].append(metadata)
                
                if progress_bar:
                    progress_bar.update(1)
        
        finally:
            # Restore original corpus
            self.corpus_lines = original_corpus
    
    def generate_corpus_curriculum_dataset(self,
                                         stage: str = "stage1",
                                         num_samples: int = 1000,
                                         train_split: float = 0.8,
                                         save_images: bool = True,
                                         show_progress: bool = True) -> Dict:
        """
        Generate corpus-based curriculum learning dataset.
        
        Args:
            stage: Curriculum stage ('stage1', 'stage2', 'stage3')
            num_samples: Total number of samples to generate
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with dataset information
        """
        if not self.corpus_lines:
            raise ValueError("Corpus not available for curriculum generation")
        
        # Define corpus-based curriculum stages
        curriculum_stages = {
            'stage1': {
                'description': 'Short corpus segments (1-5 characters)',
                'length_range': (1, 5),
                'content_preference': ['characters', 'syllables']
            },
            'stage2': {
                'description': 'Medium corpus segments (6-12 characters)',
                'length_range': (6, 12),
                'content_preference': ['syllables', 'words']
            },
            'stage3': {
                'description': 'Long corpus segments (13-20 characters)',
                'length_range': (13, 20),
                'content_preference': ['words', 'phrases']
            }
        }
        
        if stage not in curriculum_stages:
            raise ValueError(f"Unknown curriculum stage: {stage}. Available: {list(curriculum_stages.keys())}")
        
        stage_config = curriculum_stages[stage]
        print(f"Generating corpus curriculum dataset - {stage}: {stage_config['description']}")
        
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        stage_dir = Path(self.output_dir) / f'corpus_curriculum_{stage}'
        train_dir = stage_dir / 'train'
        val_dir = stage_dir / 'val'
        
        if save_images:
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []},
            'curriculum_stage': stage,
            'stage_config': stage_config
        }
        
        # Progress bar setup
        total_progress = tqdm(total=num_samples, desc=f"Generating corpus {stage} dataset") if show_progress else None
        
        # Generate training samples
        for i in range(num_train):
            content_type = random.choice(stage_config['content_preference'])
            
            text = self._generate_text_content(
                content_type=content_type,
                length_range=stage_config['length_range']
            )
            
            image, metadata = self.generate_single_image(
                text=text,
                content_type=content_type,
                apply_augmentation=True
            )
            
            if save_images:
                image_filename = f"train_{i:06d}.png"
                image_path = train_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            metadata['curriculum_stage'] = stage
            metadata['corpus_based'] = True
            all_metadata['train']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        # Generate validation samples
        for i in range(num_val):
            content_type = random.choice(stage_config['content_preference'])
            
            text = self._generate_text_content(
                content_type=content_type,
                length_range=stage_config['length_range']
            )
            
            image, metadata = self.generate_single_image(
                text=text,
                content_type=content_type,
                apply_augmentation=False
            )
            
            if save_images:
                image_filename = f"val_{i:06d}.png"
                image_path = val_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            metadata['curriculum_stage'] = stage
            metadata['corpus_based'] = True
            all_metadata['val']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        if total_progress:
            total_progress.close()
        
        # Add dataset-level metadata
        dataset_info = {
            'curriculum_stage': stage,
            'stage_description': stage_config['description'],
            'corpus_based': True,
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'image_size': list(self.image_size),
            'length_range': stage_config['length_range'],
            'content_preference': stage_config['content_preference'],
            'fonts_used': list(self.working_fonts.keys()),
            'generator_type': 'CorpusGenerator',
            'generated_by': f'CorpusGenerator v2.0 - Corpus Curriculum Learning'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = stage_dir / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nCorpus curriculum dataset ({stage}) generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_val}")
        print(f"Length range: {stage_config['length_range']}")
        
        return all_metadata
    
    def get_corpus_statistics(self) -> Dict:
        """
        Get statistics about the loaded corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        if not self.corpus_lines:
            return {'corpus_loaded': False, 'error': 'No corpus available'}
        
        # Analyze corpus
        total_lines = len(self.corpus_lines)
        total_chars = sum(len(line) for line in self.corpus_lines)
        avg_line_length = total_chars / total_lines if total_lines > 0 else 0
        
        # Character frequency analysis
        char_freq = {}
        for line in self.corpus_lines:
            for char in line:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Length distribution
        length_distribution = {}
        for line in self.corpus_lines:
            length = len(line)
            length_distribution[length] = length_distribution.get(length, 0) + 1
        
        return {
            'corpus_loaded': True,
            'total_lines': total_lines,
            'total_characters': total_chars,
            'average_line_length': avg_line_length,
            'unique_characters': len(char_freq),
            'most_common_chars': sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:20],
            'length_distribution': dict(sorted(length_distribution.items())),
            'min_length': min(len(line) for line in self.corpus_lines) if self.corpus_lines else 0,
            'max_length': max(len(line) for line in self.corpus_lines) if self.corpus_lines else 0
        }