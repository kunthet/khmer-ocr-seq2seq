"""
Synthetic data generation module for Khmer OCR.

This module provides specialized generators for creating training data:
- KhmerOCRSyntheticGenerator: Integrated generator for the Khmer OCR project
- SyntheticGenerator: Rule-based synthetic text generation
- CorpusGenerator: Corpus-based authentic text generation  
- SyntheticDataGenerator: Legacy wrapper with auto-selection
"""

from .khmer_ocr_generator import KhmerOCRSyntheticGenerator
from .synthetic_generator import SyntheticGenerator
from .corpus_generator import CorpusGenerator
from .base_generator import BaseDataGenerator
from .curriculum_dataset_generator import CurriculumDatasetGenerator, CurriculumStage, CurriculumConfig
from .generator import (
    SyntheticDataGenerator, 
    create_synthetic_generator,
    create_corpus_generator,
    create_generator
)
from .backgrounds import BackgroundGenerator
from .augmentation import ImageAugmentor
from .utils import (
    load_khmer_fonts,
    validate_font_collection,
    normalize_khmer_text,
    create_character_mapping,
    get_full_khmer_characters,
    load_character_frequencies,
    generate_weighted_character_sequence,
    generate_khmer_syllable,
    generate_khmer_word,
    generate_khmer_phrase,
    generate_mixed_content,
    load_khmer_corpus,
    generate_corpus_based_text,
    segment_corpus_text,
    KhmerTextGenerator
)

__all__ = [
    # Main generator classes
    'KhmerOCRSyntheticGenerator',
    'SyntheticGenerator',
    'CorpusGenerator', 
    'BaseDataGenerator',
    'CurriculumDatasetGenerator',
    'SyntheticDataGenerator',  # Legacy
    
    # Configuration classes
    'CurriculumStage',
    'CurriculumConfig',
    
    # Factory functions
    'create_synthetic_generator',
    'create_corpus_generator',
    'create_generator',
    
    # Component classes
    'BackgroundGenerator',
    'ImageAugmentor',
    'KhmerTextGenerator',
    
    # Utility functions
    'load_khmer_fonts',
    'validate_font_collection',
    'normalize_khmer_text',
    'create_character_mapping',
    'get_full_khmer_characters',
    'load_character_frequencies',
    'generate_weighted_character_sequence',
    'generate_khmer_syllable',
    'generate_khmer_word',
    'generate_khmer_phrase',
    'generate_mixed_content',
    'load_khmer_corpus',
    'generate_corpus_based_text',
    'segment_corpus_text',
]

__version__ = '1.0.0' 