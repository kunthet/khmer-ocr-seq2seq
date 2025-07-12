"""
Curriculum Dataset Generator for Khmer OCR - Advanced Progressive Learning Dataset Creation.

This module provides a sophisticated curriculum learning dataset generator that orchestrates
the creation of progressive difficulty datasets using both synthetic and corpus-based approaches.
"""

import os
import json
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
from collections import defaultdict

from .synthetic_generator import SyntheticGenerator
from .corpus_generator import CorpusGenerator
from .base_generator import BaseDataGenerator


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage."""
    name: str
    description: str
    difficulty_level: int  # 1-10 scale
    content_types: List[str]
    content_weights: List[float]
    length_range: Tuple[int, int]
    character_limit: Optional[int] = None
    min_accuracy_threshold: float = 0.85
    samples_per_stage: int = 1000
    use_corpus: bool = True
    corpus_ratio: float = 0.6  # 60% corpus, 40% synthetic


@dataclass
class CurriculumConfig:
    """Complete curriculum configuration."""
    name: str
    description: str
    stages: List[CurriculumStage]
    progression_strategy: str = "accuracy_based"  # accuracy_based, fixed_epochs, manual
    global_settings: Dict = None


class CurriculumDatasetGenerator:
    """
    Advanced curriculum dataset generator for progressive Khmer OCR training.
    
    Features:
    - Multi-stage curriculum design with configurable difficulty progression
    - Hybrid synthetic + corpus-based generation strategies
    - Automatic stage progression based on performance metrics
    - Comprehensive curriculum analytics and validation
    - Flexible curriculum configuration with predefined and custom curricula
    """
    
    def __init__(self, 
                 config_path: str,
                 fonts_dir: str,
                 output_dir: str,
                 mode: str = "full_text"):
        """
        Initialize the curriculum dataset generator.
        
        Args:
            config_path: Path to model configuration file
            fonts_dir: Directory containing Khmer fonts
            output_dir: Directory to save generated curriculum datasets
            mode: Generation mode ('digits', 'full_text', 'mixed')
        """
        self.config_path = config_path
        self.fonts_dir = fonts_dir
        self.output_dir = output_dir
        self.mode = mode
        
        # Initialize generators
        self.synthetic_generator = SyntheticGenerator(config_path, fonts_dir, output_dir, mode)
        try:
            self.corpus_generator = CorpusGenerator(config_path, fonts_dir, output_dir, mode)
            self.corpus_available = True
        except ValueError:
            self.corpus_generator = None
            self.corpus_available = False
            print("⚠️ Corpus not available - using synthetic generation only")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Curriculum dataset generator initialized")
        print(f"📚 Corpus available: {'Yes' if self.corpus_available else 'No'}")
        print(f"🎯 Mode: {mode}")
    
    def get_predefined_curricula(self) -> Dict[str, CurriculumConfig]:
        """Get predefined curriculum configurations."""
        return {
            'basic_khmer': self._create_basic_khmer_curriculum(),
            'advanced_khmer': self._create_advanced_khmer_curriculum(),
            'comprehensive': self._create_comprehensive_curriculum(),
            'digits_only': self._create_digits_curriculum(),
            'corpus_intensive': self._create_corpus_intensive_curriculum()
        }
    
    def _create_basic_khmer_curriculum(self) -> CurriculumConfig:
        """Create basic Khmer curriculum for beginners."""
        stages = [
            CurriculumStage(
                name="foundations",
                description="Single characters and basic combinations",
                difficulty_level=1,
                content_types=['characters', 'syllables'],
                content_weights=[0.7, 0.3],
                length_range=(1, 3),
                character_limit=20,
                min_accuracy_threshold=0.90,
                samples_per_stage=800,
                corpus_ratio=0.5
            ),
            CurriculumStage(
                name="building_blocks",
                description="Simple syllables and short words",
                difficulty_level=3,
                content_types=['syllables', 'words'],
                content_weights=[0.6, 0.4],
                length_range=(2, 6),
                character_limit=40,
                min_accuracy_threshold=0.85,
                samples_per_stage=1000,
                corpus_ratio=0.6
            ),
            CurriculumStage(
                name="text_basics",
                description="Words and simple phrases",
                difficulty_level=5,
                content_types=['words', 'phrases'],
                content_weights=[0.5, 0.5],
                length_range=(4, 12),
                character_limit=60,
                min_accuracy_threshold=0.80,
                samples_per_stage=1200,
                corpus_ratio=0.7
            ),
            CurriculumStage(
                name="fluency",
                description="Complete text with all characters",
                difficulty_level=7,
                content_types=['words', 'phrases'],
                content_weights=[0.3, 0.7],
                length_range=(8, 20),
                character_limit=None,
                min_accuracy_threshold=0.75,
                samples_per_stage=1500,
                corpus_ratio=0.8
            )
        ]
        
        return CurriculumConfig(
            name="basic_khmer",
            description="Progressive curriculum for basic Khmer OCR learning",
            stages=stages,
            progression_strategy="accuracy_based",
            global_settings={
                'train_split': 0.8,
                'save_images': True,
                'show_progress': True
            }
        )
    
    def _create_advanced_khmer_curriculum(self) -> CurriculumConfig:
        """Create advanced Khmer curriculum with complex patterns."""
        stages = [
            CurriculumStage(
                name="characters_mastery",
                description="High-frequency character mastery",
                difficulty_level=2,
                content_types=['characters', 'syllables'],
                content_weights=[0.5, 0.5],
                length_range=(1, 4),
                character_limit=30,
                min_accuracy_threshold=0.95,
                samples_per_stage=1000,
                corpus_ratio=0.4
            ),
            CurriculumStage(
                name="complex_combinations",
                description="Complex character combinations and modifiers",
                difficulty_level=4,
                content_types=['syllables', 'words'],
                content_weights=[0.4, 0.6],
                length_range=(3, 8),
                character_limit=50,
                min_accuracy_threshold=0.88,
                samples_per_stage=1200,
                corpus_ratio=0.6
            ),
            CurriculumStage(
                name="linguistic_patterns",
                description="Real linguistic patterns and structures",
                difficulty_level=6,
                content_types=['words', 'phrases'],
                content_weights=[0.4, 0.6],
                length_range=(6, 15),
                character_limit=None,
                min_accuracy_threshold=0.82,
                samples_per_stage=1500,
                corpus_ratio=0.8
            ),
            CurriculumStage(
                name="expert_level",
                description="Complete vocabulary with rare characters",
                difficulty_level=8,
                content_types=['phrases', 'mixed'],
                content_weights=[0.6, 0.4],
                length_range=(10, 25),
                character_limit=None,
                min_accuracy_threshold=0.78,
                samples_per_stage=2000,
                corpus_ratio=0.9
            )
        ]
        
        return CurriculumConfig(
            name="advanced_khmer",
            description="Advanced curriculum for expert Khmer OCR performance",
            stages=stages,
            progression_strategy="accuracy_based"
        )
    
    def _create_comprehensive_curriculum(self) -> CurriculumConfig:
        """Create comprehensive curriculum covering all aspects."""
        stages = [
            CurriculumStage(
                name="digits_foundation",
                description="Khmer digits recognition foundation",
                difficulty_level=1,
                content_types=['digits'],
                content_weights=[1.0],
                length_range=(1, 4),
                character_limit=None,
                min_accuracy_threshold=0.95,
                samples_per_stage=600,
                corpus_ratio=0.2
            ),
            CurriculumStage(
                name="character_introduction",
                description="Basic character recognition",
                difficulty_level=2,
                content_types=['characters', 'digits'],
                content_weights=[0.8, 0.2],
                length_range=(1, 3),
                character_limit=25,
                min_accuracy_threshold=0.90,
                samples_per_stage=800,
                corpus_ratio=0.4
            ),
            CurriculumStage(
                name="syllable_formation",
                description="Syllable formation and recognition",
                difficulty_level=3,
                content_types=['syllables', 'characters'],
                content_weights=[0.7, 0.3],
                length_range=(2, 6),
                character_limit=40,
                min_accuracy_threshold=0.85,
                samples_per_stage=1000,
                corpus_ratio=0.6
            ),
            CurriculumStage(
                name="word_building",
                description="Word construction and recognition",
                difficulty_level=5,
                content_types=['words', 'syllables'],
                content_weights=[0.6, 0.4],
                length_range=(4, 12),
                character_limit=60,
                min_accuracy_threshold=0.80,
                samples_per_stage=1200,
                corpus_ratio=0.7
            ),
            CurriculumStage(
                name="phrase_mastery",
                description="Phrase and sentence recognition",
                difficulty_level=7,
                content_types=['phrases', 'words'],
                content_weights=[0.7, 0.3],
                length_range=(8, 20),
                character_limit=None,
                min_accuracy_threshold=0.75,
                samples_per_stage=1500,
                corpus_ratio=0.8
            ),
            CurriculumStage(
                name="mixed_mastery",
                description="Mixed content with digits and text",
                difficulty_level=9,
                content_types=['mixed', 'phrases'],
                content_weights=[0.4, 0.6],
                length_range=(6, 25),
                character_limit=None,
                min_accuracy_threshold=0.70,
                samples_per_stage=2000,
                corpus_ratio=0.9
            )
        ]
        
        return CurriculumConfig(
            name="comprehensive",
            description="Comprehensive curriculum covering digits, characters, and full text",
            stages=stages,
            progression_strategy="accuracy_based"
        )
    
    def _create_digits_curriculum(self) -> CurriculumConfig:
        """Create curriculum focused on digits only."""
        stages = [
            CurriculumStage(
                name="single_digits",
                description="Individual digit recognition",
                difficulty_level=1,
                content_types=['digits'],
                content_weights=[1.0],
                length_range=(1, 1),
                min_accuracy_threshold=0.98,
                samples_per_stage=500,
                corpus_ratio=0.0
            ),
            CurriculumStage(
                name="digit_pairs",
                description="Two-digit number recognition",
                difficulty_level=3,
                content_types=['digits'],
                content_weights=[1.0],
                length_range=(2, 2),
                min_accuracy_threshold=0.95,
                samples_per_stage=800,
                corpus_ratio=0.0
            ),
            CurriculumStage(
                name="short_numbers",
                description="Short number sequences",
                difficulty_level=5,
                content_types=['digits'],
                content_weights=[1.0],
                length_range=(3, 5),
                min_accuracy_threshold=0.90,
                samples_per_stage=1000,
                corpus_ratio=0.0
            ),
            CurriculumStage(
                name="long_numbers",
                description="Long number sequences",
                difficulty_level=7,
                content_types=['digits'],
                content_weights=[1.0],
                length_range=(6, 8),
                min_accuracy_threshold=0.85,
                samples_per_stage=1200,
                corpus_ratio=0.0
            )
        ]
        
        return CurriculumConfig(
            name="digits_only",
            description="Specialized curriculum for Khmer digits recognition",
            stages=stages,
            progression_strategy="accuracy_based"
        )
    
    def _create_corpus_intensive_curriculum(self) -> CurriculumConfig:
        """Create corpus-intensive curriculum for authentic text patterns."""
        if not self.corpus_available:
            # Fallback to synthetic if corpus not available
            return self._create_basic_khmer_curriculum()
        
        stages = [
            CurriculumStage(
                name="corpus_basics",
                description="Short authentic text segments",
                difficulty_level=2,
                content_types=['characters', 'syllables'],
                content_weights=[0.4, 0.6],
                length_range=(1, 5),
                character_limit=30,
                min_accuracy_threshold=0.88,
                samples_per_stage=1000,
                corpus_ratio=0.9
            ),
            CurriculumStage(
                name="corpus_words",
                description="Authentic word patterns",
                difficulty_level=4,
                content_types=['syllables', 'words'],
                content_weights=[0.3, 0.7],
                length_range=(3, 10),
                character_limit=50,
                min_accuracy_threshold=0.83,
                samples_per_stage=1200,
                corpus_ratio=0.95
            ),
            CurriculumStage(
                name="corpus_phrases",
                description="Real phrase structures",
                difficulty_level=6,
                content_types=['words', 'phrases'],
                content_weights=[0.2, 0.8],
                length_range=(8, 18),
                character_limit=None,
                min_accuracy_threshold=0.78,
                samples_per_stage=1500,
                corpus_ratio=0.98
            ),
            CurriculumStage(
                name="corpus_mastery",
                description="Complex authentic text patterns",
                difficulty_level=8,
                content_types=['phrases'],
                content_weights=[1.0],
                length_range=(12, 25),
                character_limit=None,
                min_accuracy_threshold=0.73,
                samples_per_stage=2000,
                corpus_ratio=1.0
            )
        ]
        
        return CurriculumConfig(
            name="corpus_intensive",
            description="Corpus-intensive curriculum for authentic text mastery",
            stages=stages,
            progression_strategy="accuracy_based"
        )
    
    def generate_curriculum_dataset(self,
                                  curriculum_name: str = "basic_khmer",
                                  custom_curriculum: Optional[CurriculumConfig] = None,
                                  train_split: float = 0.8,
                                  save_images: bool = True,
                                  show_progress: bool = True) -> Dict:
        """
        Generate a complete curriculum dataset.
        
        Args:
            curriculum_name: Name of predefined curriculum or 'custom'
            custom_curriculum: Custom curriculum configuration if using 'custom'
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with complete curriculum dataset information
        """
        # Get curriculum configuration
        if custom_curriculum:
            curriculum = custom_curriculum
        else:
            curricula = self.get_predefined_curricula()
            if curriculum_name not in curricula:
                raise ValueError(f"Unknown curriculum: {curriculum_name}. Available: {list(curricula.keys())}")
            curriculum = curricula[curriculum_name]
        
        print(f"🎓 Generating curriculum dataset: {curriculum.name}")
        print(f"📖 {curriculum.description}")
        print(f"📚 Stages: {len(curriculum.stages)}")
        
        # Create curriculum output directory with simpler structure
        curriculum_dir = Path(self.output_dir) / curriculum.name
        curriculum_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize curriculum metadata
        curriculum_metadata = {
            'curriculum_config': asdict(curriculum),
            'stages': {},
            'generation_summary': {},
            'statistics': {}
        }
        
        total_samples = sum(stage.samples_per_stage for stage in curriculum.stages)
        overall_progress = tqdm(total=total_samples, desc="Overall curriculum progress") if show_progress else None
        
        # Generate each stage
        for stage_idx, stage in enumerate(curriculum.stages):
            print(f"\n🎯 Stage {stage_idx + 1}/{len(curriculum.stages)}: {stage.name}")
            print(f"   {stage.description}")
            print(f"   Difficulty: {stage.difficulty_level}/10")
            print(f"   Samples: {stage.samples_per_stage}")
            
            stage_metadata = self._generate_curriculum_stage(
                stage=stage,
                stage_idx=stage_idx,
                curriculum_dir=curriculum_dir,
                train_split=train_split,
                save_images=save_images,
                show_progress=show_progress,
                overall_progress=overall_progress
            )
            
            curriculum_metadata['stages'][stage.name] = stage_metadata
        
        if overall_progress:
            overall_progress.close()
        
        # Generate curriculum statistics
        curriculum_metadata['statistics'] = self._analyze_curriculum_dataset(curriculum_metadata)
        
        # Save curriculum metadata
        if save_images:
            curriculum_metadata_path = curriculum_dir / 'curriculum_metadata.yaml'
            with open(curriculum_metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(curriculum_metadata, f, default_flow_style=False, allow_unicode=True)
            
            # Save curriculum configuration separately for easy reference
            config_path = curriculum_dir / 'curriculum_config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(curriculum), f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n✅ Curriculum dataset generated successfully!")
        print(f"📊 Total samples: {total_samples}")
        print(f"📁 Output directory: {curriculum_dir}")
        print(f"🎓 Curriculum: {curriculum.name}")
        
        return curriculum_metadata
    
    def _generate_curriculum_stage(self,
                                 stage: CurriculumStage,
                                 stage_idx: int,
                                 curriculum_dir: Path,
                                 train_split: float,
                                 save_images: bool,
                                 show_progress: bool,
                                 overall_progress) -> Dict:
        """Generate a single curriculum stage."""
        
        stage_dir = curriculum_dir / f's{stage_idx+1}_{stage.name}'
        
        # For digits mode, use synthetic generator directly
        if self.mode == "digits" or not self.corpus_available:
            print(f"   🔧 Synthetic generation: {stage.samples_per_stage} samples")
            
            stage_metadata = self._generate_synthetic_stage(
                stage=stage,
                stage_dir=stage_dir,
                train_split=train_split,
                save_images=save_images,
                show_progress=show_progress,
                overall_progress=overall_progress
            )
        else:
            # Mixed generation for full text mode
            corpus_samples = int(stage.samples_per_stage * stage.corpus_ratio) if stage.use_corpus else 0
            synthetic_samples = stage.samples_per_stage - corpus_samples
            
            print(f"   📊 Mixed generation: {corpus_samples} corpus + {synthetic_samples} synthetic")
            
            stage_metadata = self._generate_mixed_stage(
                stage=stage,
                stage_dir=stage_dir,
                corpus_samples=corpus_samples,
                synthetic_samples=synthetic_samples,
                train_split=train_split,
                save_images=save_images,
                show_progress=show_progress,
                overall_progress=overall_progress
            )
        
        return stage_metadata
    
    def _generate_synthetic_stage(self,
                                stage: CurriculumStage,
                                stage_dir: Path,
                                train_split: float,
                                save_images: bool,
                                show_progress: bool,
                                overall_progress) -> Dict:
        """Generate a pure synthetic stage."""
        
        # Create stage directory structure
        if save_images:
            stage_dir.mkdir(parents=True, exist_ok=True)
            train_dir = stage_dir / 'train'
            val_dir = stage_dir / 'val'
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate train/val splits
        num_train = int(stage.samples_per_stage * train_split)
        num_val = stage.samples_per_stage - num_train
        
        # Generate samples
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []},
            'stage_config': asdict(stage),
            'generation_type': 'synthetic'
        }
        
        # Generate training samples
        for i in range(num_train):
            content_type = np.random.choice(stage.content_types, p=stage.content_weights)
            
            image, metadata = self.synthetic_generator.generate_single_image(
                content_type=content_type,
                apply_augmentation=True
            )
            
            if save_images:
                image_filename = f"train_{i:06d}.png"
                image_path = train_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            metadata['curriculum_stage'] = stage.name
            all_metadata['train']['samples'].append(metadata)
            
            if overall_progress:
                overall_progress.update(1)
        
        # Generate validation samples
        for i in range(num_val):
            content_type = np.random.choice(stage.content_types, p=stage.content_weights)
            
            image, metadata = self.synthetic_generator.generate_single_image(
                content_type=content_type,
                apply_augmentation=False
            )
            
            if save_images:
                image_filename = f"val_{i:06d}.png"
                image_path = val_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            metadata['curriculum_stage'] = stage.name
            all_metadata['val']['samples'].append(metadata)
            
            if overall_progress:
                overall_progress.update(1)
        
        # Save stage metadata
        if save_images:
            metadata_path = stage_dir / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        return all_metadata
    
    def _generate_mixed_stage(self,
                            stage: CurriculumStage,
                            stage_dir: Path,
                            corpus_samples: int,
                            synthetic_samples: int,
                            train_split: float,
                            save_images: bool,
                            show_progress: bool,
                            overall_progress) -> Dict:
        """Generate a mixed corpus + synthetic stage."""
        
        # Create stage directory structure with simpler paths
        if save_images:
            stage_dir.mkdir(parents=True, exist_ok=True)
            train_dir = stage_dir / 'train'
            val_dir = stage_dir / 'val'
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []},
            'stage_config': asdict(stage),
            'generation_type': 'mixed',
            'corpus_samples': corpus_samples,
            'synthetic_samples': synthetic_samples,
            'total_samples': corpus_samples + synthetic_samples
        }
        
        # Calculate train/val splits for each type
        corpus_train = int(corpus_samples * train_split) if corpus_samples > 0 else 0
        corpus_val = corpus_samples - corpus_train if corpus_samples > 0 else 0
        synthetic_train = int(synthetic_samples * train_split) if synthetic_samples > 0 else 0
        synthetic_val = synthetic_samples - synthetic_train if synthetic_samples > 0 else 0
        
        sample_count = 0
        
        # Generate corpus samples directly if needed
        if corpus_samples > 0 and self.corpus_generator:
            for i in range(corpus_train):
                image, metadata = self.corpus_generator.generate_single_image(apply_augmentation=True)
                
                if save_images:
                    image_filename = f"train_{sample_count:06d}.png"
                    image_path = train_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['generation_source'] = 'corpus'
                metadata['curriculum_stage'] = stage.name
                all_metadata['train']['samples'].append(metadata)
                sample_count += 1
                
                if overall_progress:
                    overall_progress.update(1)
            
            for i in range(corpus_val):
                image, metadata = self.corpus_generator.generate_single_image(apply_augmentation=False)
                
                if save_images:
                    image_filename = f"val_{i:06d}.png"
                    image_path = val_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['generation_source'] = 'corpus'
                metadata['curriculum_stage'] = stage.name
                all_metadata['val']['samples'].append(metadata)
                
                if overall_progress:
                    overall_progress.update(1)
        
        # Generate synthetic samples directly if needed
        if synthetic_samples > 0:
            for i in range(synthetic_train):
                content_type = np.random.choice(stage.content_types, p=stage.content_weights)
                image, metadata = self.synthetic_generator.generate_single_image(
                    content_type=content_type,
                    apply_augmentation=True
                )
                
                if save_images:
                    image_filename = f"train_{sample_count:06d}.png"
                    image_path = train_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['generation_source'] = 'synthetic'
                metadata['curriculum_stage'] = stage.name
                all_metadata['train']['samples'].append(metadata)
                sample_count += 1
                
                if overall_progress:
                    overall_progress.update(1)
            
            val_count = 0
            for i in range(synthetic_val):
                content_type = np.random.choice(stage.content_types, p=stage.content_weights)
                image, metadata = self.synthetic_generator.generate_single_image(
                    content_type=content_type,
                    apply_augmentation=False
                )
                
                if save_images:
                    image_filename = f"val_{val_count:06d}.png"
                    image_path = val_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['generation_source'] = 'synthetic'
                metadata['curriculum_stage'] = stage.name
                all_metadata['val']['samples'].append(metadata)
                val_count += 1
                
                if overall_progress:
                    overall_progress.update(1)
        
        # Save metadata
        if save_images:
            metadata_path = stage_dir / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        return all_metadata
    
    def _analyze_curriculum_dataset(self, curriculum_metadata: Dict) -> Dict:
        """Analyze the generated curriculum dataset and provide statistics."""
        
        stats = {
            'total_stages': len(curriculum_metadata['stages']),
            'total_samples': 0,
            'total_train_samples': 0,
            'total_val_samples': 0,
            'difficulty_progression': [],
            'content_type_distribution': defaultdict(int),
            'generation_type_distribution': defaultdict(int),
            'stage_summary': []
        }
        
        for stage_name, stage_data in curriculum_metadata['stages'].items():
            stage_config = stage_data.get('stage_config', {})
            
            # Basic counts
            train_samples = len(stage_data.get('train', {}).get('samples', []))
            val_samples = len(stage_data.get('val', {}).get('samples', []))
            total_samples = train_samples + val_samples
            
            stats['total_samples'] += total_samples
            stats['total_train_samples'] += train_samples
            stats['total_val_samples'] += val_samples
            
            # Difficulty progression
            stats['difficulty_progression'].append({
                'stage': stage_name,
                'difficulty': stage_config.get('difficulty_level', 0),
                'samples': total_samples
            })
            
            # Content type distribution
            content_types = stage_config.get('content_types', [])
            for content_type in content_types:
                stats['content_type_distribution'][content_type] += total_samples
            
            # Generation type distribution
            gen_type = stage_data.get('generation_type', 'unknown')
            stats['generation_type_distribution'][gen_type] += total_samples
            
            # Stage summary
            stats['stage_summary'].append({
                'name': stage_name,
                'difficulty': stage_config.get('difficulty_level', 0),
                'samples': total_samples,
                'train_samples': train_samples,
                'val_samples': val_samples,
                'content_types': content_types,
                'generation_type': gen_type
            })
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['content_type_distribution'] = dict(stats['content_type_distribution'])
        stats['generation_type_distribution'] = dict(stats['generation_type_distribution'])
        
        return stats
    
    def get_curriculum_info(self, curriculum_name: str) -> Dict:
        """Get information about a curriculum without generating it."""
        curricula = self.get_predefined_curricula()
        if curriculum_name not in curricula:
            raise ValueError(f"Unknown curriculum: {curriculum_name}")
        
        curriculum = curricula[curriculum_name]
        
        info = {
            'name': curriculum.name,
            'description': curriculum.description,
            'total_stages': len(curriculum.stages),
            'total_samples': sum(stage.samples_per_stage for stage in curriculum.stages),
            'difficulty_range': (
                min(stage.difficulty_level for stage in curriculum.stages),
                max(stage.difficulty_level for stage in curriculum.stages)
            ),
            'stages': []
        }
        
        for i, stage in enumerate(curriculum.stages):
            stage_info = {
                'index': i + 1,
                'name': stage.name,
                'description': stage.description,
                'difficulty': stage.difficulty_level,
                'samples': stage.samples_per_stage,
                'content_types': stage.content_types,
                'length_range': stage.length_range,
                'corpus_ratio': stage.corpus_ratio if stage.use_corpus else 0.0
            }
            info['stages'].append(stage_info)
        
        return info
    
    def list_available_curricula(self) -> List[str]:
        """List all available predefined curricula."""
        return list(self.get_predefined_curricula().keys())
    
    def validate_curriculum_config(self, curriculum: CurriculumConfig) -> Tuple[bool, List[str]]:
        """Validate a curriculum configuration."""
        errors = []
        
        # Basic validation
        if not curriculum.name:
            errors.append("Curriculum name is required")
        
        if not curriculum.stages:
            errors.append("At least one stage is required")
        
        # Stage validation
        for i, stage in enumerate(curriculum.stages):
            if not stage.name:
                errors.append(f"Stage {i+1}: name is required")
            
            if not stage.content_types:
                errors.append(f"Stage {i+1}: at least one content type is required")
            
            if len(stage.content_types) != len(stage.content_weights):
                errors.append(f"Stage {i+1}: content_types and content_weights must have same length")
            
            if abs(sum(stage.content_weights) - 1.0) > 0.001:
                errors.append(f"Stage {i+1}: content_weights must sum to 1.0")
            
            if stage.difficulty_level < 1 or stage.difficulty_level > 10:
                errors.append(f"Stage {i+1}: difficulty_level must be between 1 and 10")
            
            if stage.samples_per_stage <= 0:
                errors.append(f"Stage {i+1}: samples_per_stage must be positive")
        
        return len(errors) == 0, errors 