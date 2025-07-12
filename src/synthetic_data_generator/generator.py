"""
Factory and legacy support for Khmer OCR data generators.
This file maintains backward compatibility while providing access to the new specialized generators.
"""

# Import the new specialized generators
from .synthetic_generator import SyntheticGenerator
from .corpus_generator import CorpusGenerator
from .base_generator import BaseDataGenerator

# Legacy class for backward compatibility
class SyntheticDataGenerator:
    """
    Legacy wrapper that delegates to the appropriate specialized generator.
    Maintains backward compatibility while using the new architecture.
    """
    
    def __init__(self, 
                 config_path: str,
                 fonts_dir: str,
                 output_dir: str,
                 mode: str = "full_text",
                 use_corpus: bool = True):
        """
        Initialize the data generator using the appropriate specialized generator.
        
        Args:
            config_path: Path to model configuration file
            fonts_dir: Directory containing Khmer fonts
            output_dir: Directory to save generated data
            mode: Generation mode ('digits', 'full_text', 'mixed')
            use_corpus: Whether to use real corpus text for generation
        """
        # Choose the appropriate generator based on use_corpus preference
        if use_corpus and mode in ["full_text", "mixed"]:
            try:
                self._generator = CorpusGenerator(config_path, fonts_dir, output_dir, mode)
                print("🎯 Using CorpusGenerator for authentic text generation")
            except ValueError as e:
                print(f"⚠️ Corpus generator failed: {e}")
                print("🔄 Falling back to SyntheticGenerator")
                self._generator = SyntheticGenerator(config_path, fonts_dir, output_dir, mode)
        else:
            self._generator = SyntheticGenerator(config_path, fonts_dir, output_dir, mode)
            print("🎯 Using SyntheticGenerator for rule-based text generation")
        
        # Expose the underlying generator's attributes for backward compatibility
        self.config_path = self._generator.config_path
        self.fonts_dir = self._generator.fonts_dir
        self.output_dir = self._generator.output_dir
        self.mode = self._generator.mode
        self.use_corpus = use_corpus
        self.config = self._generator.config
        self.image_size = self._generator.image_size
        self.max_sequence_length = self._generator.max_sequence_length
        self.background_generator = self._generator.background_generator
        self.augmentor = self._generator.augmentor
        self.fonts = self._generator.fonts
        self.font_validation = self._generator.font_validation
        self.working_fonts = self._generator.working_fonts
        self.char_to_idx = self._generator.char_to_idx
        self.idx_to_char = self._generator.idx_to_char
        self.khmer_generator = self._generator.khmer_generator
        self.character_frequencies = self._generator.character_frequencies
        self.khmer_characters = self._generator.khmer_characters
        
        # Corpus-specific attributes
        if hasattr(self._generator, 'corpus_lines'):
            self.corpus_lines = self._generator.corpus_lines
        else:
            self.corpus_lines = None
    
    # Delegate all methods to the underlying generator
    def __getattr__(self, name):
        """Delegate method calls to the underlying generator."""
        return getattr(self._generator, name)


# Factory functions for easier access to specific generators
def create_synthetic_generator(config_path: str, fonts_dir: str, output_dir: str, mode: str = "full_text") -> SyntheticGenerator:
    """Create a SyntheticGenerator instance."""
    return SyntheticGenerator(config_path, fonts_dir, output_dir, mode)


def create_corpus_generator(config_path: str, fonts_dir: str, output_dir: str, mode: str = "full_text") -> CorpusGenerator:
    """Create a CorpusGenerator instance."""
    return CorpusGenerator(config_path, fonts_dir, output_dir, mode)


def create_generator(config_path: str, fonts_dir: str, output_dir: str, mode: str = "full_text", 
                     use_corpus: bool = True, generator_type: str = "auto"):
    """
    Factory function to create the appropriate generator.
    
    Args:
        config_path: Path to model configuration file
        fonts_dir: Directory containing Khmer fonts
        output_dir: Directory to save generated data
        mode: Generation mode ('digits', 'full_text', 'mixed')
        use_corpus: Whether to prefer corpus-based generation
        generator_type: Specific generator type ('synthetic', 'corpus', 'auto')
        
    Returns:
        Appropriate generator instance
    """
    if generator_type == "synthetic":
        return SyntheticGenerator(config_path, fonts_dir, output_dir, mode)
    elif generator_type == "corpus":
        return CorpusGenerator(config_path, fonts_dir, output_dir, mode)
    else:  # auto
        return SyntheticDataGenerator(config_path, fonts_dir, output_dir, mode, use_corpus) 