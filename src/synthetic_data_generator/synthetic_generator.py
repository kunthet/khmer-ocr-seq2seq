"""
Synthetic data generator for Khmer OCR that focuses on rule-based synthetic text generation.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import yaml
from tqdm import tqdm

from .base_generator import BaseDataGenerator


class SyntheticGenerator(BaseDataGenerator):
    """
    Generates synthetic training data for Khmer OCR using rule-based text generation.
    """
    
    def __init__(self, 
                 config_path: str,
                 fonts_dir: str,
                 output_dir: str,
                 mode: str = "full_text"):
        """
        Initialize the synthetic data generator.
        
        Args:
            config_path: Path to model configuration file
            fonts_dir: Directory containing Khmer fonts
            output_dir: Directory to save generated data
            mode: Generation mode ('digits', 'full_text', 'mixed')
        """
        super().__init__(config_path, fonts_dir, output_dir, mode)
        print("✅ Synthetic generator initialized - using rule-based text generation")
    
    def _generate_text_content(self, 
                              content_type: str = "auto",
                              length_range: Tuple[int, int] = (1, 15),
                              allowed_characters: Optional[List[str]] = None) -> str:
        """
        Generate synthetic text content based on linguistic rules.
        
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
        
        # Use rule-compliant synthetic generation
        return self.khmer_generator.generate_content_by_type(
            content_type=content_type,
            length=target_length,
            allowed_characters=allowed_characters
        )
    
    def generate_curriculum_dataset(self,
                                  stage: str = "stage1",
                                  num_samples: int = 1000,
                                  train_split: float = 0.8,
                                  save_images: bool = True,
                                  show_progress: bool = True) -> Dict:
        """
        Generate dataset for curriculum learning stages.
        
        Args:
            stage: Curriculum stage ('stage1', 'stage2', 'stage3', 'mixed')
            num_samples: Total number of samples to generate
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with dataset information
        """
        # Define curriculum stages based on character frequency analysis
        curriculum_stages = {
            'stage1': {
                'description': 'High-frequency characters (top 30)',
                'content_types': ['characters', 'syllables'],
                'content_weights': [0.6, 0.4],
                'length_range': (1, 5),
                'character_limit': 30  # Top 30 most frequent characters
            },
            'stage2': {
                'description': 'Medium-frequency characters (top 60)',
                'content_types': ['characters', 'syllables', 'words'],
                'content_weights': [0.4, 0.4, 0.2],
                'length_range': (1, 10),
                'character_limit': 60  # Top 60 most frequent characters
            },
            'stage3': {
                'description': 'All characters with complex structures',
                'content_types': ['syllables', 'words', 'phrases'],
                'content_weights': [0.3, 0.5, 0.2],
                'length_range': (1, 20),
                'character_limit': None  # All characters
            },
            'mixed': {
                'description': 'Mixed content including digits',
                'content_types': ['digits', 'characters', 'syllables', 'words'],
                'content_weights': [0.2, 0.3, 0.3, 0.2],
                'length_range': (1, 15),
                'character_limit': None
            }
        }
        
        if stage not in curriculum_stages:
            raise ValueError(f"Unknown curriculum stage: {stage}. Available: {list(curriculum_stages.keys())}")
        
        stage_config = curriculum_stages[stage]
        print(f"Generating curriculum dataset - {stage}: {stage_config['description']}")
        
        # Get character subset for this stage
        character_subset = self._get_character_subset(stage_config['character_limit'])
        
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        stage_dir = Path(self.output_dir) / f'curriculum_{stage}'
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
        total_progress = tqdm(total=num_samples, desc=f"Generating {stage} dataset") if show_progress else None
        
        # Generate training samples
        for i in range(num_train):
            content_type = np.random.choice(
                stage_config['content_types'], 
                p=stage_config['content_weights']
            )
            
            # Generate with specific complexity and character subset
            text = self._generate_stage_appropriate_text(content_type, stage_config, character_subset)
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
            all_metadata['train']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        # Generate validation samples
        for i in range(num_val):
            content_type = np.random.choice(
                stage_config['content_types'], 
                p=stage_config['content_weights']
            )
            
            # Generate with specific complexity and character subset
            text = self._generate_stage_appropriate_text(content_type, stage_config, character_subset)
            image, metadata = self.generate_single_image(
                text=text,
                content_type=content_type,
                apply_augmentation=False  # No augmentation for validation
            )
            
            if save_images:
                image_filename = f"val_{i:06d}.png"
                image_path = val_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            metadata['curriculum_stage'] = stage
            all_metadata['val']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        if total_progress:
            total_progress.close()
        
        # Add dataset-level metadata
        dataset_info = {
            'curriculum_stage': stage,
            'stage_description': stage_config['description'],
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'image_size': list(self.image_size),
            'content_types': stage_config['content_types'],
            'length_range': stage_config['length_range'],
            'character_subset_size': len(character_subset) if character_subset else len(self.character_frequencies),
            'fonts_used': list(self.working_fonts.keys()),
            'generator_type': 'SyntheticGenerator',
            'generated_by': f'SyntheticGenerator v2.0 - Curriculum Learning'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = stage_dir / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nCurriculum dataset ({stage}) generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_val}")
        print(f"Character subset size: {len(character_subset) if character_subset else 'All characters'}")
        
        return all_metadata
    
    def _get_character_subset(self, limit: Optional[int]) -> Optional[List[str]]:
        """Get top N characters by frequency."""
        if limit is None:
            return None
        
        # Sort characters by frequency
        sorted_chars = sorted(
            self.character_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top N characters
        return [char for char, freq in sorted_chars[:limit]]
    
    def _text_uses_allowed_characters(self, text: str, allowed_chars: List[str]) -> bool:
        """Check if text only uses allowed characters."""
        allowed_set = set(allowed_chars)
        text_chars = set(text)
        return text_chars.issubset(allowed_set)
    
    def _generate_stage_appropriate_text(self, 
                                       content_type: str, 
                                       stage_config: Dict, 
                                       character_subset: List[str]) -> str:
        """Generate text appropriate for curriculum stage."""
        length_range = stage_config['length_range']
        target_length = random.randint(*length_range)
        
        # Use the rule-compliant text generation method
        return self.khmer_generator.generate_content_by_type(
            content_type=content_type,
            length=target_length,
            allowed_characters=character_subset
        )
    
    def generate_frequency_balanced_dataset(self,
                                          num_samples: int = 1000,
                                          balance_factor: float = 0.5,
                                          train_split: float = 0.8,
                                          save_images: bool = True) -> Dict:
        """
        Generate dataset with frequency-balanced character distribution.
        
        Args:
            num_samples: Total number of samples
            balance_factor: 0.0 = pure frequency weighting, 1.0 = uniform distribution
            train_split: Fraction for training
            save_images: Whether to save images
            
        Returns:
            Dataset metadata
        """
        print(f"Generating frequency-balanced dataset (balance_factor: {balance_factor})")
        
        # Create balanced character frequencies
        balanced_frequencies = {}
        uniform_weight = 1.0 / len(self.character_frequencies)
        
        for char, freq in self.character_frequencies.items():
            # Interpolate between original frequency and uniform distribution
            balanced_freq = (1 - balance_factor) * freq + balance_factor * uniform_weight
            balanced_frequencies[char] = balanced_freq
        
        # Temporarily replace character frequencies
        original_frequencies = self.character_frequencies
        self.character_frequencies = balanced_frequencies
        
        try:
            # Generate dataset with balanced frequencies
            dataset = self.generate_dataset(
                num_samples=num_samples,
                train_split=train_split,
                save_images=save_images
            )
            
            # Add balance information to metadata
            dataset['dataset_info']['balance_factor'] = balance_factor
            dataset['dataset_info']['generation_type'] = 'frequency_balanced'
            
            return dataset
            
        finally:
            # Restore original frequencies
            self.character_frequencies = original_frequencies
    
    def generate_mixed_complexity_dataset(self,
                                        num_samples: int = 1000,
                                        train_split: float = 0.8,
                                        save_images: bool = True) -> Dict:
        """
        Generate dataset with mixed complexity levels.
        
        Args:
            num_samples: Total number of samples
            train_split: Fraction for training
            save_images: Whether to save images
            
        Returns:
            Dataset metadata
        """
        print("Generating mixed complexity dataset")
        
        # Define complexity distribution
        complexity_levels = {
            'simple': {'weight': 0.4, 'content_types': ['digits', 'characters'], 'length_range': (1, 3)},
            'medium': {'weight': 0.4, 'content_types': ['syllables', 'words'], 'length_range': (4, 10)},
            'complex': {'weight': 0.2, 'content_types': ['words', 'phrases'], 'length_range': (11, 20)}
        }
        
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        output_dir = Path(self.output_dir) / 'mixed_complexity'
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        
        if save_images:
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []},
            'complexity_levels': complexity_levels
        }
        
        # Generate samples with complexity distribution
        for split_name, num_split, split_dir in [('train', num_train, train_dir), ('val', num_val, val_dir)]:
            for i in tqdm(range(num_split), desc=f"Generating {split_name} samples"):
                # Choose complexity level
                complexity = np.random.choice(
                    list(complexity_levels.keys()),
                    p=[level['weight'] for level in complexity_levels.values()]
                )
                
                level_config = complexity_levels[complexity]
                content_type = random.choice(level_config['content_types'])
                
                # Generate with specific complexity
                image, metadata = self.generate_single_image(
                    content_type=content_type,
                    apply_augmentation=(split_name == 'train')
                )
                
                if save_images:
                    image_filename = f"{split_name}_{i:06d}.png"
                    image_path = split_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['complexity_level'] = complexity
                all_metadata[split_name]['samples'].append(metadata)
        
        # Add dataset info
        dataset_info = {
            'generation_type': 'mixed_complexity',
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'complexity_distribution': complexity_levels,
            'image_size': list(self.image_size),
            'fonts_used': list(self.working_fonts.keys()),
            'generator_type': 'SyntheticGenerator',
            'generated_by': 'SyntheticGenerator v2.0 - Mixed Complexity'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = output_dir / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Mixed complexity dataset generated: {num_samples} samples")
        return all_metadata
    
    def generate_samples_by_length(self, 
                                  samples_per_length: int = 100,
                                  save_images: bool = True) -> Dict:
        """
        Generate samples with balanced sequence lengths.
        
        Args:
            samples_per_length: Number of samples per sequence length
            save_images: Whether to save images to disk
            
        Returns:
            Dictionary with dataset information
        """
        all_metadata = {'samples': []}
        
        # Create output directory
        if save_images:
            output_dir = Path(self.output_dir) / 'balanced'
            output_dir.mkdir(exist_ok=True)
        
        sample_count = 0
        
        for length in range(1, self.max_sequence_length + 1):
            print(f"Generating {samples_per_length} samples with {length} character(s)...")
            
            for i in tqdm(range(samples_per_length), desc=f"Length {length}"):
                # Generate text with specific length
                if self.mode == "digits":
                    text = self.khmer_generator.generate_digit_sequence(length, length)
                else:
                    text = self.khmer_generator.generate_content_by_type(
                        content_type="auto",
                        length=length
                    )
                
                image, metadata = self.generate_single_image(text=text)
                
                if save_images:
                    image_filename = f"sample_{sample_count:06d}_len{length}.png"
                    image_path = output_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                all_metadata['samples'].append(metadata)
                sample_count += 1
        
        # Add dataset info
        dataset_info = {
            'total_samples': sample_count,
            'samples_per_length': samples_per_length,
            'sequence_lengths': list(range(1, self.max_sequence_length + 1)),
            'image_size': list(self.image_size),
            'fonts_used': list(self.working_fonts.keys()),
            'character_set': list(self.char_to_idx.keys()),
            'generator_type': 'SyntheticGenerator',
            'generated_by': 'SyntheticGenerator v2.0 (balanced)'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = Path(self.output_dir) / 'balanced' / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nBalanced dataset generated successfully!")
        print(f"Total samples: {sample_count}")
        print(f"Samples per length: {samples_per_length}")
        
        return all_metadata 