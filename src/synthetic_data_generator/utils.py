"""
Utilities for synthetic data generation including font management,
text generation, and validation functions.
"""

import os
import yaml
import random
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import sys

# Add khtext module to path for syllable segmentation
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / 'src' / 'modules'))

try:
    from khtext.khnormal_fast import khnormal
    KHNORMAL_AVAILABLE = True
except ImportError:
    print("Warning: Khmer normalization not available")
    KHNORMAL_AVAILABLE = False

try:
    from khtext.subword_cluster import split_syllables_advanced, restore_whitespace_tags
    SYLLABLE_SEGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: Khmer syllable segmentation not available")
    SYLLABLE_SEGMENTATION_AVAILABLE = False

# Khmer Character Unicode Constants
COENG_SIGN = '\u17D2'  # ្
REPETITION_SIGN = '\u17D7'  # ៗ
BANTOC_SIGN = '\u17CB'  # ់

# Dependent vowels that cannot be placed next to each other
DEPENDENT_VOWELS_NO_DUPLICATE = ['ា', 'ិ', 'ី', 'ឹ', 'ឺ', 'ុ', 'ូ', 'ួ', 'ើ', 'ៀ', 'េ', 'ែ', 'ៃ', 'ោ', 'ៅ']

class KhmerTextGenerator:
    """
    A comprehensive Khmer text generator that follows proper linguistic rules.
    
    Rules implemented:
    1. Must start with consonant, independent vowel, or repetition sign (ៗ)
    2. Coeng sign (្) must be between consonants
    3. Never placing 2 or more Coeng signs next to each other
    4. Never placing specific dependent vowels next to each other
    5. Never end with Coeng sign (្)
    6. Bantoc sign (់) must be after consonants and usually at end of syllable
    """
    
    def __init__(self, character_frequencies: Optional[Dict[str, float]] = None):
        """
        Initialize the Khmer text generator.
        
        Args:
            character_frequencies: Optional character frequency dictionary
        """
        self.character_frequencies = character_frequencies or self.load_character_frequencies()
        self.khmer_chars = self.get_full_khmer_characters()
        
        # Character sets for easy access
        self.consonants = self.khmer_chars['consonants']
        self.vowels = self.khmer_chars['vowels']
        self.independents = self.khmer_chars['independents']
        self.signs = self.khmer_chars['signs']
        self.digits = self.khmer_chars['digits']
    
    @staticmethod
    def is_khmer_consonant(char: str) -> bool:
        """Check if character is a Khmer consonant."""
        return '\u1780' <= char <= '\u17A2'
    
    @staticmethod
    def is_khmer_vowel(char: str) -> bool:
        """Check if character is a Khmer dependent vowel."""
        return '\u17B6' <= char <= '\u17C5'
    
    @staticmethod
    def is_khmer_independent_vowel(char: str) -> bool:
        """Check if character is a Khmer independent vowel."""
        return '\u17A5' <= char <= '\u17B5'
    
    @staticmethod
    def is_khmer_sign(char: str) -> bool:
        """Check if character is a Khmer sign/diacritic."""
        return '\u17C6' <= char <= '\u17D3'
    
    @staticmethod
    def is_valid_khmer_start_character(char: str) -> bool:
        """
        Check if character can legally start a Khmer text.
        Rule 1: Must start with consonant, independent vowel, or repetition sign.
        """
        return (KhmerTextGenerator.is_khmer_consonant(char) or 
                KhmerTextGenerator.is_khmer_independent_vowel(char) or
                char == REPETITION_SIGN)
    
    def _split_into_syllables(self, text: str) -> List[str]:
        """
        Split Khmer text into syllable units for validation.
        
        This is a simplified syllable boundary detection that identifies
        syllable breaks based on consonant patterns and independent vowels.
        
        Args:
            text: Input Khmer text
            
        Returns:
            List of syllable strings
        """
        if not text:
            return []
        return split_syllables_advanced(text)
        
    
    def validate_khmer_text_structure(self, text: str) -> Tuple[bool, str]:
        """
        Validate Khmer text against linguistic rules.
        
        Rules:
        1. Must start with consonant, independent vowel, or repetition sign
        2. Coeng sign (្) must be between consonants
        3. Never place 2+ Coeng signs next to each other
        4. Never place 2+ specific dependent vowels next to each other
        5. Never end with Coeng sign (្)
        6. Bantoc sign (់) must be after consonants
        7. In same syllable: at most 2 Coeng signs and not adjacent
        
        Args:
            text: Khmer text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text:
            return False, "Empty text"
        
        # Rule 1: Must start with consonant, independent vowel, or repetition sign
        if not self.is_valid_khmer_start_character(text[0]):
            return False, f"Text must start with consonant, independent vowel, or repetition sign, not '{text[0]}'"
        
        # Rule 5: Never end with Coeng sign
        if text.endswith(COENG_SIGN):
            return False, "Text cannot end with Coeng sign (្)"
        
        # Rule 7: Syllable-level Coeng constraints
        syllables = self._split_into_syllables(text)
        for syll_idx, syllable in enumerate(syllables):
            coeng_count = syllable.count(COENG_SIGN)
            if coeng_count > 2:
                return False, f"Syllable '{syllable}' has {coeng_count} Coeng signs (max 2 allowed per syllable)"
            
            # Check for adjacent Coeng signs within syllable
            if COENG_SIGN + COENG_SIGN in syllable:
                return False, f"Syllable '{syllable}' has adjacent Coeng signs (not allowed)"
        
        # Check each character position
        for i, char in enumerate(text):
            # Rule 2: Coeng sign must be between consonants
            if char == COENG_SIGN:
                # Check if previous character is consonant
                if i == 0 or not self.is_khmer_consonant(text[i-1]):
                    return False, f"Coeng sign at position {i} must follow a consonant"
                
                # Check if next character is consonant
                if i == len(text) - 1 or not self.is_khmer_consonant(text[i+1]):
                    return False, f"Coeng sign at position {i} must be followed by a consonant"
            
            # Rule 3: Never place 2+ Coeng signs next to each other
            if i > 0 and char == COENG_SIGN and text[i-1] == COENG_SIGN:
                return False, f"Multiple Coeng signs found at positions {i-1}-{i}"
            
            # Rule 4: Never place specific dependent vowels next to each other
            if (i > 0 and char in DEPENDENT_VOWELS_NO_DUPLICATE and 
                text[i-1] in DEPENDENT_VOWELS_NO_DUPLICATE):
                return False, f"Dependent vowels '{text[i-1]}' and '{char}' cannot be adjacent at positions {i-1}-{i}"
            
            # Rule 6: Bantoc sign (់) must be after consonants and usually at end of syllable
            if char == BANTOC_SIGN:
                if i == 0 or not self.is_khmer_consonant(text[i-1]):
                    return False, f"Bantoc sign (់) at position {i} must follow a consonant"
        
        return True, "Valid"
    
    def generate_syllable(self) -> str:
        """
        Generate a linguistically valid Khmer syllable following all rules.
        
        Returns:
            Generated valid Khmer syllable
        """
        syllable = ""
        
        # Rule 1: Start with consonant, independent vowel, or repetition sign
        start_options = self.consonants + self.independents + [REPETITION_SIGN]
        if start_options:
            start_weights = [self.character_frequencies.get(c, 0.01) for c in start_options]
            total_weight = sum(start_weights)
            if total_weight > 0:
                start_weights = [w / total_weight for w in start_weights]
                start_char = np.random.choice(start_options, p=start_weights)
                syllable += start_char
            else:
                return "ក"  # Fallback
        else:
            return "ក"  # Fallback
        
        # If repetition sign, that's the complete syllable
        if syllable == REPETITION_SIGN:
            return normalize_khmer_text(syllable)
        
        # If started with consonant, may add Coeng+consonant cluster (increased probability)
        # Rule 7: At most 2 Coeng signs per syllable, not adjacent
        if (self.is_khmer_consonant(syllable[0]) and random.random() < 0.25):  # 25% chance for Coeng cluster
            consonant_weights = [self.character_frequencies.get(c, 0.01) for c in self.consonants]
            total_weight = sum(consonant_weights)
            if total_weight > 0:
                consonant_weights = [w / total_weight for w in consonant_weights]
                next_consonant = np.random.choice(self.consonants, p=consonant_weights)
                syllable += COENG_SIGN + next_consonant
                
                # Possibly add a second Coeng cluster (lower probability) with non-adjacent constraint
                if (random.random() < 0.15 and  # 15% chance for second Coeng
                    self.is_khmer_consonant(syllable[-1]) and  # Last char is consonant
                    not syllable.endswith(COENG_SIGN)):  # Not ending with Coeng already
                    
                    # Add vowel or other character to separate Coeng signs
                    if self.vowels and random.random() < 0.7:
                        vowel_weights = [self.character_frequencies.get(v, 0.01) for v in self.vowels]
                        vowel_total = sum(vowel_weights)
                        if vowel_total > 0:
                            vowel_weights = [w / vowel_total for w in vowel_weights]
                            vowel = np.random.choice(self.vowels, p=vowel_weights)
                            syllable += vowel
                    
                    # Add second Coeng+consonant cluster
                    second_consonant = np.random.choice(self.consonants, p=consonant_weights)
                    syllable += COENG_SIGN + second_consonant
        
        # If started with consonant (and no Coeng added), may add dependent vowel
        if (self.is_khmer_consonant(syllable[0]) and COENG_SIGN not in syllable and 
            self.vowels and random.random() < 0.6):
            vowel_weights = [self.character_frequencies.get(v, 0.01) for v in self.vowels]
            total_weight = sum(vowel_weights)
            if total_weight > 0:
                vowel_weights = [w / total_weight for w in vowel_weights]
                vowel = np.random.choice(self.vowels, p=vowel_weights)
                syllable += vowel
        
        # If Coeng cluster exists, may add vowel after the second consonant
        elif COENG_SIGN in syllable and self.vowels and random.random() < 0.4:
            vowel_weights = [self.character_frequencies.get(v, 0.01) for v in self.vowels]
            total_weight = sum(vowel_weights)
            if total_weight > 0:
                vowel_weights = [w / total_weight for w in vowel_weights]
                vowel = np.random.choice(self.vowels, p=vowel_weights)
                syllable += vowel
        
        # Add Bantoc sign (់) after consonants - higher probability at end
        if (self.is_khmer_consonant(syllable[-1]) and random.random() < 0.15):
            syllable += BANTOC_SIGN
        
        # Add other signs/diacritics (but not Coeng, repetition, or Bantoc)
        filtered_signs = [s for s in self.signs if s not in [COENG_SIGN, REPETITION_SIGN, BANTOC_SIGN]]
        if filtered_signs and random.random() < 0.15:
            sign_weights = [self.character_frequencies.get(s, 0.01) for s in filtered_signs]
            total_weight = sum(sign_weights)
            if total_weight > 0:
                sign_weights = [w / total_weight for w in sign_weights]
                sign = np.random.choice(filtered_signs, p=sign_weights)
                syllable += sign
        
        # Validate and return
        is_valid, error = self.validate_khmer_text_structure(syllable)
        if is_valid:
            return normalize_khmer_text(syllable)
        else:
            # Fallback to simple consonant
            return "ក"
    
    def generate_word(self, min_syllables: int = 1, max_syllables: int = 4) -> str:
        """
        Generate a linguistically valid Khmer word following all rules.
        
        Args:
            min_syllables: Minimum number of syllables
            max_syllables: Maximum number of syllables
            
        Returns:
            Generated valid Khmer word
        """
        num_syllables = random.randint(min_syllables, max_syllables)
        word = ""
        
        for i in range(num_syllables):
            if i == 0:
                # First syllable
                syllable = self.generate_syllable()
                word += syllable
            else:
                # Subsequent syllables - enhanced Coeng generation strategy
                use_coeng = False
                
                # Higher probability of Coeng if:
                # 1. Previous character is consonant (75% chance)
                # 2. Word is getting longer (increases Coeng usage)
                # 3. We want to maintain realistic Coeng frequency
                if word and self.is_khmer_consonant(word[-1]):
                    base_coeng_prob = 0.75  # Increased from 60% to 75%
                    length_bonus = min(0.15, len(word) * 0.03)  # Bonus for longer words
                    # Additional bonus for multi-syllable words
                    syllable_bonus = 0.1 if len(word) > 2 else 0
                    coeng_probability = base_coeng_prob + length_bonus + syllable_bonus
                    use_coeng = random.random() < coeng_probability
                
                if use_coeng and word and self.is_khmer_consonant(word[-1]):
                    # Rule 2: Coeng must be between consonants
                    # Generate consonant to follow Coeng
                    consonant_weights = [self.character_frequencies.get(c, 0.01) for c in self.consonants]
                    total_weight = sum(consonant_weights)
                    if total_weight > 0:
                        consonant_weights = [w / total_weight for w in consonant_weights]
                        next_consonant = np.random.choice(self.consonants, p=consonant_weights)
                        
                        # Rule 3: Ensure no double Coeng
                        if not word.endswith(COENG_SIGN):
                            word += COENG_SIGN + next_consonant
                            
                            # After Coeng+consonant, may add vowel or signs
                            if random.random() < 0.4:  # 40% chance to add vowel
                                vowel_weights = [self.character_frequencies.get(v, 0.01) for v in self.vowels]
                                vowel_total = sum(vowel_weights)
                                if vowel_total > 0:
                                    vowel_weights = [w / vowel_total for w in vowel_weights]
                                    vowel = np.random.choice(self.vowels, p=vowel_weights)
                                    word += vowel
                        else:
                            # Skip Coeng if previous char is already Coeng, just add consonant
                            word += next_consonant
                    else:
                        # Fallback - add normal syllable
                        syllable = self.generate_syllable()
                        word += syllable
                else:
                    # Add normal syllable (no Coeng)
                    syllable = self.generate_syllable()
                    word += syllable
        
        # Rule 5: Ensure doesn't end with Coeng
        if word.endswith(COENG_SIGN):
            # Add a consonant after the trailing Coeng
            word += random.choice(self.consonants)
        
        # Final validation
        is_valid, error = self.validate_khmer_text_structure(word)
        if not is_valid:
            # Fallback to simple syllable
            return self.generate_syllable()
        
        return normalize_khmer_text(word)
    
    def generate_phrase(self, min_words: int = 1, max_words: int = 5) -> str:
        """
        Generate a Khmer phrase with multiple words.
        
        Args:
            min_words: Minimum number of words
            max_words: Maximum number of words
            
        Returns:
            Generated Khmer phrase
        """
        num_words = random.randint(min_words, max_words)
        words = []
        
        for _ in range(num_words):
            # Generate longer words to increase Coeng opportunities
            word = self.generate_word(min_syllables=1, max_syllables=5)  # Extended max syllables
            words.append(word)
        
        # Join words with space (though Khmer traditionally doesn't use spaces)
        phrase = " ".join(words) if random.random() < 0.3 else "".join(words)
        return normalize_khmer_text(phrase)
    
    def generate_character_sequence(self, length: int, 
                                  character_set: Optional[List[str]] = None) -> str:
        """
        Generate a valid character sequence following Khmer linguistic rules.
        
        Args:
            length: Length of sequence to generate
            character_set: List of characters to choose from
            
        Returns:
            Generated valid character sequence
        """
        if character_set is None:
            character_set = list(self.character_frequencies.keys())
        
        # Filter to valid starting characters for first position
        valid_start_chars = [char for char in character_set 
                            if self.is_valid_khmer_start_character(char)]
        
        if not valid_start_chars:
            # Fallback to consonants + independents + repetition sign
            valid_start_chars = self.consonants + self.independents + [REPETITION_SIGN]
        
        sequence = ""
        
        for i in range(length):
            if i == 0:
                # Rule 1: Start with consonant or independent vowel
                start_weights = [self.character_frequencies.get(char, 0.01) for char in valid_start_chars]
                total_weight = sum(start_weights)
                if total_weight > 0:
                    start_weights = [w / total_weight for w in start_weights]
                    char = np.random.choice(valid_start_chars, p=start_weights)
                    sequence += char
                else:
                    sequence += random.choice(valid_start_chars)
            else:
                # Subsequent characters - apply all rules
                available_chars = []
                
                # Filter characters based on context
                for char in character_set:
                    # Rule 2: Coeng must be between consonants
                    if char == COENG_SIGN:
                        if (i > 0 and self.is_khmer_consonant(sequence[-1]) and 
                            i < length - 1):  # Not at end (Rule 5)
                            
                            # Rule 7: Check syllable-level Coeng constraints
                            current_syllables = self._split_into_syllables(sequence)
                            if current_syllables:
                                current_syllable = current_syllables[-1]  # Last syllable being built
                                coeng_count_in_syllable = current_syllable.count(COENG_SIGN)
                                
                                # Don't add Coeng if syllable already has 2 or would create adjacent Coeng
                                if (coeng_count_in_syllable >= 2 or 
                                    current_syllable.endswith(COENG_SIGN)):
                                    continue
                            
                            # Significantly increase Coeng probability to maintain frequency
                            available_chars.extend([char] * 8)  # 8x weight instead of 3x
                        continue
                    
                    # Rule 3: No double Coeng
                    if i > 0 and sequence[-1] == COENG_SIGN and char == COENG_SIGN:
                        continue
                    
                    # Rule 4: No duplicate specific dependent vowels
                    if (i > 0 and char in DEPENDENT_VOWELS_NO_DUPLICATE and 
                        sequence[-1] in DEPENDENT_VOWELS_NO_DUPLICATE):
                        continue
                    
                    # Rule 5: If this is last position, don't use Coeng
                    if i == length - 1 and char == COENG_SIGN:
                        continue
                    
                    # Rule 2: If previous is Coeng, this must be consonant
                    if i > 0 and sequence[-1] == COENG_SIGN and not self.is_khmer_consonant(char):
                        continue
                    
                    # Rule 6: Bantoc (់) must be after consonants
                    if char == BANTOC_SIGN:
                        if i > 0 and self.is_khmer_consonant(sequence[-1]):
                            available_chars.append(char)
                        continue
                    
                    available_chars.append(char)
                
                if not available_chars:
                    # Fallback - use consonants (safe choice)
                    available_chars = [char for char in self.consonants 
                                     if char in character_set]
                    if not available_chars:
                        available_chars = self.consonants
                
                # Choose next character
                char_weights = [self.character_frequencies.get(char, 0.01) for char in available_chars]
                total_weight = sum(char_weights)
                if total_weight > 0:
                    char_weights = [w / total_weight for w in char_weights]
                    char = np.random.choice(available_chars, p=char_weights)
                else:
                    char = random.choice(available_chars)
                
                sequence += char
        
        # Final validation and correction
        is_valid, error = self.validate_khmer_text_structure(sequence)
        if not is_valid:
            # Try to fix common issues
            if sequence.endswith(COENG_SIGN):
                # Rule 5: Add consonant after trailing Coeng
                sequence += random.choice(self.consonants)
            elif not self.is_valid_khmer_start_character(sequence[0]):
                # Rule 1: Fix invalid start
                sequence = random.choice(self.consonants) + sequence[1:]
        
        return normalize_khmer_text(sequence)
    
    def generate_content_by_type(self, content_type: str = "auto", 
                               length: int = None,
                               allowed_characters: Optional[List[str]] = None) -> str:
        """
        Generate content based on specified type.
        
        Args:
            content_type: Type of content ('auto', 'digits', 'characters', 'syllables', 'words', 'phrases', 'mixed')
            length: Target length
            allowed_characters: Allowed characters for curriculum learning
            
        Returns:
            Generated content
        """
        if length is None:
            length = random.randint(1, 15)
        
        if content_type == "digits":
            return self.generate_digit_sequence(1, min(8, length))
        elif content_type == "characters":
            character_set = allowed_characters if allowed_characters else list(self.character_frequencies.keys())
            return self.generate_character_sequence(length, character_set)
        elif content_type == "syllables":
            num_syllables = max(1, length // 3)
            syllables = []
            for i in range(num_syllables):
                syllable = self.generate_syllable()
                syllables.append(syllable)
                
                # Add inter-syllable Coeng connections occasionally for authenticity
                if (i < num_syllables - 1 and random.random() < 0.15 and 
                    self.is_khmer_consonant(syllable[-1])):
                    # Generate a Coeng+consonant to connect syllables
                    consonant_weights = [self.character_frequencies.get(c, 0.01) for c in self.consonants]
                    total_weight = sum(consonant_weights)
                    if total_weight > 0:
                        consonant_weights = [w / total_weight for w in consonant_weights]
                        next_consonant = np.random.choice(self.consonants, p=consonant_weights)
                        syllables.append(COENG_SIGN + next_consonant)
            
            return "".join(syllables)
        elif content_type == "words":
            if length <= 5:
                return self.generate_word(1, 2)
            else:
                return self.generate_phrase(1, max(1, length // 10))
        elif content_type == "phrases":
            return self.generate_phrase(1, max(1, length // 8))
        elif content_type == "mixed":
            content_types = ["characters", "syllables", "words"]
            weights = [0.4, 0.4, 0.2]
            chosen_type = np.random.choice(content_types, p=weights)
            return self.generate_content_by_type(chosen_type, length, allowed_characters)
        else:  # auto
            # Intelligent content type selection based on length
            if length <= 3:
                return self.generate_content_by_type("characters", length, allowed_characters)
            elif length <= 8:
                return self.generate_content_by_type("syllables", length, allowed_characters)
            elif length <= 15:
                return self.generate_content_by_type("words", length, allowed_characters)
            else:
                return self.generate_content_by_type("phrases", length, allowed_characters)
    
    @staticmethod
    def get_full_khmer_characters() -> Dict[str, List[str]]:
        """
        Get the complete Khmer character set organized by categories.
        
        Returns:
            Dictionary with character categories
        """
        try:
            # Import from khchar if available
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'khtext'))
            from khchar import CONSONANTS, VOWELS, INDEPENDENTS, SIGN_CHARS, DIGITS, LEK_ATTAK
            
            return {
                'consonants': [chr(c) for c in CONSONANTS],
                'vowels': [chr(c) for c in VOWELS],
                'independents': [chr(c) for c in INDEPENDENTS],
                'signs': [chr(c) for c in SIGN_CHARS],
                'digits': [chr(c) for c in DIGITS],
                'lek_attak': [chr(c) for c in LEK_ATTAK]
            }
        except ImportError:
            # Fallback to basic character set
            return {
                'consonants': [chr(i) for i in range(0x1780, 0x17A3)],  # 33 consonants
                'vowels': [chr(i) for i in range(0x17B6, 0x17C6)],      # 16 vowels
                'independents': [chr(i) for i in range(0x17A5, 0x17B6)], # 14 independent vowels
                'signs': [chr(i) for i in range(0x17C6, 0x17D4)],       # 13 signs
                'digits': [chr(i) for i in range(0x17E0, 0x17EA)],      # 10 digits
                'lek_attak': [chr(i) for i in range(0x17F0, 0x17FA)]    # 10 lek attak
            }
    
    @staticmethod
    def load_character_frequencies(analysis_file: str = "khmer_text_analysis_results.json") -> Dict[str, float]:
        """
        Load character frequencies from analysis results.
        
        Args:
            analysis_file: Path to analysis results JSON file
            
        Returns:
            Dictionary mapping characters to normalized frequencies
        """
        frequencies = {}
        
        try:
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'top_frequencies' in data:
                    total_freq = sum(freq for char_code, freq in data['top_frequencies'])
                    for char_code, freq in data['top_frequencies']:
                        char = chr(char_code)
                        frequencies[char] = freq / total_freq
        except Exception as e:
            print(f"Warning: Could not load character frequencies: {e}")
        
        # Fallback to uniform distribution if no frequencies available
        if not frequencies:
            khmer_chars = KhmerTextGenerator.get_full_khmer_characters()
            all_chars = []
            for category in khmer_chars.values():
                all_chars.extend(category)
            
            uniform_freq = 1.0 / len(all_chars)
            frequencies = {char: uniform_freq for char in all_chars}
        
        return frequencies
    
    @staticmethod
    def generate_digit_sequence(min_length: int = 1, max_length: int = 8) -> str:
        """
        Generate a random sequence of Khmer digits.
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            
        Returns:
            Generated digit sequence
        """
        digits = ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]
        length = random.randint(min_length, max_length)
        sequence = ''.join(random.choices(digits, k=length))
        return normalize_khmer_text(sequence)

# Global instance for backward compatibility
_khmer_generator = None

def get_khmer_generator() -> KhmerTextGenerator:
    """Get the global Khmer text generator instance."""
    global _khmer_generator
    if _khmer_generator is None:
        _khmer_generator = KhmerTextGenerator()
    return _khmer_generator

# Legacy function compatibility - now using the class
def is_khmer_consonant(char: str) -> bool:
    """Check if character is a Khmer consonant."""
    return KhmerTextGenerator.is_khmer_consonant(char)

def is_khmer_vowel(char: str) -> bool:
    """Check if character is a Khmer dependent vowel."""
    return KhmerTextGenerator.is_khmer_vowel(char)

def is_khmer_independent_vowel(char: str) -> bool:
    """Check if character is a Khmer independent vowel."""
    return KhmerTextGenerator.is_khmer_independent_vowel(char)

def is_khmer_sign(char: str) -> bool:
    """Check if character is a Khmer sign/diacritic."""
    return KhmerTextGenerator.is_khmer_sign(char)

def is_valid_khmer_start_character(char: str) -> bool:
    """Check if character can legally start a Khmer text."""
    return KhmerTextGenerator.is_valid_khmer_start_character(char)

def validate_khmer_text_structure(text: str) -> Tuple[bool, str]:
    """Validate Khmer text against linguistic rules."""
    return get_khmer_generator().validate_khmer_text_structure(text)

def generate_valid_khmer_syllable() -> str:
    """Generate a linguistically valid Khmer syllable following all rules."""
    return get_khmer_generator().generate_syllable()

def generate_valid_khmer_word(min_syllables: int = 1, max_syllables: int = 4) -> str:
    """Generate a linguistically valid Khmer word following all rules."""
    return get_khmer_generator().generate_word(min_syllables, max_syllables)

def generate_valid_khmer_character_sequence(length: int, 
                                          character_frequencies: Optional[Dict[str, float]] = None,
                                          character_set: Optional[List[str]] = None) -> str:
    """Generate a valid character sequence following Khmer linguistic rules."""
    generator = get_khmer_generator()
    if character_frequencies:
        # Create temporary generator with custom frequencies
        temp_generator = KhmerTextGenerator(character_frequencies)
        return temp_generator.generate_character_sequence(length, character_set)
    return generator.generate_character_sequence(length, character_set)

def normalize_khmer_text(text: str) -> str:
    """
    Normalize Khmer text using Unicode NFC normalization.
    
    Args:
        text: Input Khmer text
        
    Returns:
        Normalized text
    """
    if KHNORMAL_AVAILABLE:
        return khnormal(text)
    return unicodedata.normalize('NFC', text)

def load_khmer_fonts(fonts_dir: str) -> Dict[str, str]:
    """
    Load all TTF fonts from the specified directory.
    
    Args:
        fonts_dir: Path to fonts directory
        
    Returns:
        Dictionary mapping font names to font file paths
    """
    fonts = {}
    fonts_path = Path(fonts_dir)
    
    if not fonts_path.exists():
        raise FileNotFoundError(f"Fonts directory not found: {fonts_dir}")
    
    for font_file in fonts_path.glob("*.ttf"):
        font_name = font_file.stem
        fonts[font_name] = str(font_file)
    
    if not fonts:
        raise ValueError(f"No TTF fonts found in {fonts_dir}")
    
    return fonts


def get_khmer_digits() -> List[str]:
    """
    Get the list of Khmer digits.
    
    Returns:
        List of Khmer digit characters
    """
    return ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]


def get_full_khmer_characters() -> Dict[str, List[str]]:
    """
    Get the complete Khmer character set organized by categories.
    
    Returns:
        Dictionary with character categories
    """
    try:
        # Import from khchar if available
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'khtext'))
        from khchar import CONSONANTS, VOWELS, INDEPENDENTS, SIGN_CHARS, DIGITS, LEK_ATTAK
        
        return {
            'consonants': [chr(c) for c in CONSONANTS],
            'vowels': [chr(c) for c in VOWELS],
            'independents': [chr(c) for c in INDEPENDENTS],
            'signs': [chr(c) for c in SIGN_CHARS],
            'digits': [chr(c) for c in DIGITS],
            'lek_attak': [chr(c) for c in LEK_ATTAK]
        }
    except ImportError:
        # Fallback to basic character set
        return {
            'consonants': [chr(i) for i in range(0x1780, 0x17A3)],  # 33 consonants
            'vowels': [chr(i) for i in range(0x17B6, 0x17C6)],      # 16 vowels
            'independents': [chr(i) for i in range(0x17A5, 0x17B6)], # 14 independent vowels
            'signs': [chr(i) for i in range(0x17C6, 0x17D4)],       # 13 signs
            'digits': [chr(i) for i in range(0x17E0, 0x17EA)],      # 10 digits
            'lek_attak': [chr(i) for i in range(0x17F0, 0x17FA)]    # 10 lek attak
        }


def load_character_frequencies(analysis_file: str = "khmer_text_analysis_results.json") -> Dict[str, float]:
    """
    Load character frequencies from analysis results.
    
    Args:
        analysis_file: Path to analysis results JSON file
        
    Returns:
        Dictionary mapping characters to normalized frequencies
    """
    frequencies = {}
    
    try:
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'top_frequencies' in data:
                total_freq = sum(freq for char_code, freq in data['top_frequencies'])
                for char_code, freq in data['top_frequencies']:
                    char = chr(char_code)
                    frequencies[char] = freq / total_freq
    except Exception as e:
        print(f"Warning: Could not load character frequencies: {e}")
    
    # Fallback to uniform distribution if no frequencies available
    if not frequencies:
        khmer_chars = get_full_khmer_characters()
        all_chars = []
        for category in khmer_chars.values():
            all_chars.extend(category)
        
        uniform_freq = 1.0 / len(all_chars)
        frequencies = {char: uniform_freq for char in all_chars}
    
    return frequencies


def get_special_tokens() -> List[str]:
    """
    Get the list of special tokens used in the model.
    
    Returns:
        List of special token strings
    """
    return ["<EOS>", "<PAD>", "<BLANK>"]


def create_character_mapping(use_full_khmer: bool = True) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character to index and index to character mappings.
    
    Args:
        use_full_khmer: If True, use full Khmer character set, else just digits
    
    Returns:
        Tuple of (char_to_idx, idx_to_char) dictionaries
    """
    if use_full_khmer:
        khmer_chars = get_full_khmer_characters()
        chars = []
        for category in khmer_chars.values():
            chars.extend(category)
        chars.extend(get_special_tokens())
    else:
        chars = get_khmer_digits() + get_special_tokens()
    
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    return char_to_idx, idx_to_char


def generate_digit_sequence(min_length: int = 1, max_length: int = 8) -> str:
    """
    Generate a random sequence of Khmer digits.
    
    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        Random Khmer digit sequence
    """
    digits = get_khmer_digits()
    length = np.random.randint(min_length, max_length + 1)
    sequence = ''.join(np.random.choice(digits, size=length))
    return normalize_khmer_text(sequence)


def generate_weighted_character_sequence(length: int, 
                                       character_frequencies: Optional[Dict[str, float]] = None,
                                       character_set: Optional[List[str]] = None) -> str:
    """
    Generate a character sequence using frequency weighting with proper Khmer linguistic rules.
    
    Args:
        length: Length of sequence to generate
        character_frequencies: Dictionary of character frequencies
        character_set: List of characters to choose from
        
    Returns:
        Generated character sequence following Khmer rules
    """
    # Use the rule-compliant generator
    return generate_valid_khmer_character_sequence(length, character_frequencies, character_set)


def generate_khmer_syllable() -> str:
    """
    Generate a linguistically valid Khmer syllable following all rules.
    
    Returns:
        Generated Khmer syllable
    """
    # Use the rule-compliant generator
    return generate_valid_khmer_syllable()


def generate_khmer_word(min_syllables: int = 1, max_syllables: int = 4) -> str:
    """
    Generate a linguistically valid Khmer word following all rules.
    
    Args:
        min_syllables: Minimum number of syllables
        max_syllables: Maximum number of syllables
        
    Returns:
        Generated Khmer word
    """
    # Use the rule-compliant generator
    return generate_valid_khmer_word(min_syllables, max_syllables)


def generate_khmer_phrase(min_words: int = 1, max_words: int = 5) -> str:
    """
    Generate a Khmer phrase with multiple words.
    
    Args:
        min_words: Minimum number of words
        max_words: Maximum number of words
        
    Returns:
        Generated Khmer phrase
    """
    num_words = random.randint(min_words, max_words)
    words = []
    
    for _ in range(num_words):
        word = generate_khmer_word()
        words.append(word)
    
    # Join words with space (though Khmer traditionally doesn't use spaces)
    phrase = " ".join(words) if random.random() < 0.3 else "".join(words)
    return normalize_khmer_text(phrase)


def generate_mixed_content(length: int = None, content_type: str = "mixed") -> str:
    """
    Generate mixed content including digits, text, and special characters.
    
    Args:
        length: Target length (if None, random length chosen)
        content_type: Type of content ('digits', 'characters', 'syllables', 'words', 'mixed')
        
    Returns:
        Generated content
    """
    if length is None:
        length = random.randint(1, 20)
    
    if content_type == "digits":
        return generate_digit_sequence(min(1, length), min(8, length))
    elif content_type == "characters":
        return generate_weighted_character_sequence(length)
    elif content_type == "syllables":
        num_syllables = max(1, length // 3)
        return "".join([generate_khmer_syllable() for _ in range(num_syllables)])
    elif content_type == "words":
        if length <= 5:
            return generate_khmer_word(1, 2)
        else:
            return generate_khmer_phrase(1, max(1, length // 10))
    else:  # mixed
        content_types = ["digits", "characters", "syllables", "words"]
        weights = [0.2, 0.3, 0.3, 0.2]  # Balanced distribution
        chosen_type = np.random.choice(content_types, p=weights)
        return generate_mixed_content(length, chosen_type)


def test_font_rendering(font_path: str, test_text: str = "០១២៣៤កខគ") -> bool:
    """
    Test if a font can properly render Khmer text.
    
    Args:
        font_path: Path to font file
        test_text: Test text to render
        
    Returns:
        True if font renders properly, False otherwise
    """
    try:
        font = ImageFont.truetype(font_path, size=48)
        # Try to get text bounding box - this will fail if characters are not supported
        bbox = font.getbbox(test_text)
        return bbox[2] > bbox[0] and bbox[3] > bbox[1]  # width > 0 and height > 0
    except Exception:
        return False


def validate_font_collection(fonts_dir: str) -> Dict[str, bool]:
    """
    Validate all fonts in the collection for Khmer text support.
    
    Args:
        fonts_dir: Path to fonts directory
        
    Returns:
        Dictionary mapping font names to validation status
    """
    fonts = load_khmer_fonts(fonts_dir)
    validation_results = {}
    
    # Test with a mix of digits and characters
    test_text = "០១២៣៤កខគឃង"  # Mix of digits and consonants
    
    for font_name, font_path in fonts.items():
        validation_results[font_name] = test_font_rendering(font_path, test_text)
    
    return validation_results


def validate_dataset(dataset_path: str, expected_size: int) -> Dict[str, any]:
    """
    Validate a generated dataset.
    
    Args:
        dataset_path: Path to dataset directory
        expected_size: Expected number of samples
        
    Returns:
        Dictionary with validation results
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        return {"valid": False, "error": "Dataset directory does not exist"}
    
    # Count image files
    image_files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
    
    # Check for metadata file
    metadata_file = dataset_dir / "metadata.yaml"
    has_metadata = metadata_file.exists()
    
    # Load and validate metadata if it exists
    metadata = None
    if has_metadata:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            return {"valid": False, "error": f"Failed to load metadata: {e}"}
    
    return {
        "valid": True,
        "num_images": len(image_files),
        "expected_size": expected_size,
        "size_match": len(image_files) == expected_size,
        "has_metadata": has_metadata,
        "metadata": metadata
    }


def calculate_dataset_statistics(dataset_path: str) -> Dict[str, any]:
    """
    Calculate statistics for a generated dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with dataset statistics
    """
    metadata_file = Path(dataset_path) / "metadata.yaml"
    
    if not metadata_file.exists():
        return {"error": "No metadata file found"}
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        return {"error": f"Failed to load metadata: {e}"}
    
    # Calculate statistics by combining train and val samples
    all_samples = []
    if 'train' in metadata and 'samples' in metadata['train']:
        all_samples.extend(metadata['train']['samples'])
    if 'val' in metadata and 'samples' in metadata['val']:
        all_samples.extend(metadata['val']['samples'])
    
    # If no train/val structure, try direct samples
    if not all_samples and 'samples' in metadata:
        all_samples = metadata['samples']
    
    sequence_lengths = [len(item['label']) for item in all_samples]
    fonts_used = [item['font'] for item in all_samples]
    
    stats = {
        "total_samples": len(all_samples),
        "sequence_length_distribution": {
            "min": min(sequence_lengths) if sequence_lengths else 0,
            "max": max(sequence_lengths) if sequence_lengths else 0,
            "mean": np.mean(sequence_lengths) if sequence_lengths else 0,
            "std": np.std(sequence_lengths) if sequence_lengths else 0
        },
        "font_distribution": {font: fonts_used.count(font) for font in set(fonts_used)},
        "character_frequency": {}
    }
    
    # Calculate character frequencies from all labels
    all_chars = ''.join([item['label'] for item in all_samples])
    khmer_chars = get_full_khmer_characters()
    for category_chars in khmer_chars.values():
        for char in category_chars:
            stats["character_frequency"][char] = all_chars.count(char)
    
    return stats


def load_khmer_corpus(corpus_file: str = "data/khmer_clean_text.txt") -> List[str]:
    """
    Load and prepare Khmer corpus text for segmentation.
    
    Args:
        corpus_file: Path to the corpus text file
        
    Returns:
        List of text lines from the corpus
    """
    if not os.path.exists(corpus_file):
        print(f"Warning: Corpus file not found: {corpus_file}")
        return []
    
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded corpus: {len(lines)} lines, total ~{sum(len(line) for line in lines):,} characters")
        return lines
        
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return []


def segment_corpus_text(corpus_lines: List[str], 
                       target_length: int,
                       min_length: int = 1,
                       max_length: int = 50,
                       allowed_characters: Optional[List[str]] = None,
                       use_syllable_boundaries: bool = True) -> str:
    """
    Extract a random text segment from the corpus with specified length constraints.
    Uses advanced Khmer syllable segmentation for proper text boundaries.
    
    Args:
        corpus_lines: List of corpus text lines
        target_length: Desired segment length (approximate)
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        allowed_characters: List of allowed characters for curriculum filtering
        use_syllable_boundaries: Whether to use syllable-aware boundary detection
        
    Returns:
        Text segment from corpus
    """
    if not corpus_lines:
        return generate_weighted_character_sequence(target_length)
    
    # Choose random line
    line = random.choice(corpus_lines)
    
    # Find suitable segment
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts:
        # For very short target lengths, use simple character-based extraction
        if target_length <= 3 or not use_syllable_boundaries or not SYLLABLE_SEGMENTATION_AVAILABLE:
            segment = _extract_simple_segment(line, target_length, min_length, max_length)
        else:
            # Use syllable-aware extraction for longer segments
            segment = _extract_syllable_aware_segment(line, target_length, min_length, max_length)
        
        if not segment:
            attempts += 1
            continue
            
        # Clean and normalize
        segment = normalize_khmer_text(segment.strip())
        
        # Check length constraints
        if min_length <= len(segment) <= max_length:
            # Check character constraints for curriculum learning
            if allowed_characters is None or _text_uses_allowed_characters(segment, allowed_characters):
                return segment
        
        attempts += 1
    
    # Fallback: generate synthetic text if no suitable segment found
    print(f"Warning: Could not find suitable corpus segment, generating synthetic text")
    if allowed_characters:
        return generate_weighted_character_sequence(
            target_length, 
            character_frequencies={char: load_character_frequencies().get(char, 0.001) 
                                 for char in allowed_characters},
            character_set=allowed_characters
        )
    else:
        return generate_weighted_character_sequence(target_length)


def _extract_simple_segment(line: str, target_length: int, min_length: int, max_length: int) -> str:
    """Extract segment using simple character-based approach."""
    if len(line) <= target_length:
        return line
    
    # Random starting position
    start_pos = random.randint(0, len(line) - target_length)
    end_pos = start_pos + target_length
    
    # Adjust to word boundaries if possible
    segment = line[start_pos:end_pos]
    
    # Try to end at word boundary (space or punctuation)
    word_boundaries = [' ', '។', '៕', '៖', 'ៗ', '\n']
    for i in range(len(segment) - 1, max(0, len(segment) - 10), -1):
        if segment[i] in word_boundaries:
            segment = segment[:i].strip()
            break
    
    return segment


def _extract_syllable_aware_segment(line: str, target_length: int, min_length: int, max_length: int) -> str:
    """Extract segment using syllable-aware boundary detection."""
    try:
        # Segment the entire line into syllables
        syllables = split_syllables_advanced(line)
        
        if not syllables:
            return _extract_simple_segment(line, target_length, min_length, max_length)
        
        # Calculate approximate syllables needed
        avg_syllable_length = len(line) / len(syllables) if syllables else 3
        target_syllables = max(1, int(target_length / avg_syllable_length))
        
        # Choose random starting position in syllables
        if len(syllables) <= target_syllables:
            selected_syllables = syllables
        else:
            start_idx = random.randint(0, len(syllables) - target_syllables)
            
            # Try different segment lengths around target
            best_segment = None
            best_length_diff = float('inf')
            
            for length_adjustment in range(-2, 3):  # Try ±2 syllables
                adjusted_length = max(1, target_syllables + length_adjustment)
                end_idx = min(len(syllables), start_idx + adjusted_length)
                
                segment_syllables = syllables[start_idx:end_idx]
                segment_text = restore_whitespace_tags(''.join(segment_syllables))
                
                if min_length <= len(segment_text) <= max_length:
                    length_diff = abs(len(segment_text) - target_length)
                    if length_diff < best_length_diff:
                        best_length_diff = length_diff
                        best_segment = segment_text
            
            if best_segment:
                return best_segment
            
            # Fallback to target length
            selected_syllables = syllables[start_idx:start_idx + target_syllables]
        
        # Join syllables and restore whitespace
        segment = restore_whitespace_tags(''.join(selected_syllables))
        return segment
        
    except Exception as e:
        print(f"Warning: Syllable segmentation failed: {e}, falling back to simple extraction")
        return _extract_simple_segment(line, target_length, min_length, max_length)


def _text_uses_allowed_characters(text: str, allowed_chars: List[str]) -> bool:
    """Check if text only uses allowed characters."""
    allowed_set = set(allowed_chars)
    text_chars = set(text)
    khmer_chars = {char for char in text_chars if '\u1780' <= char <= '\u17FF'}
    return khmer_chars.issubset(allowed_set)


def extract_corpus_segments_by_complexity(corpus_lines: List[str],
                                         complexity_level: str = "medium",
                                         num_segments: int = 100,
                                         use_syllable_boundaries: bool = True) -> List[str]:
    """
    Extract corpus segments categorized by complexity level using syllable-aware boundaries.
    
    Args:
        corpus_lines: List of corpus text lines
        complexity_level: 'simple', 'medium', or 'complex'
        num_segments: Number of segments to extract
        use_syllable_boundaries: Whether to use syllable-aware boundary detection
        
    Returns:
        List of extracted text segments
    """
    complexity_configs = {
        'simple': {'min_length': 1, 'max_length': 5, 'target_length': 3},
        'medium': {'min_length': 6, 'max_length': 15, 'target_length': 10},
        'complex': {'min_length': 16, 'max_length': 50, 'target_length': 25}
    }
    
    if complexity_level not in complexity_configs:
        complexity_level = 'medium'
    
    config = complexity_configs[complexity_level]
    segments = []
    
    for _ in range(num_segments * 3):  # Try more to get enough good segments
        segment = segment_corpus_text(
            corpus_lines,
            target_length=random.randint(config['min_length'], config['max_length']),
            min_length=config['min_length'],
            max_length=config['max_length'],
            use_syllable_boundaries=use_syllable_boundaries
        )
        
        if segment and len(segment) >= config['min_length']:
            segments.append(segment)
        
        if len(segments) >= num_segments:
            break
    
    return segments[:num_segments]


def generate_corpus_based_text(corpus_lines: Optional[List[str]] = None,
                              target_length: int = None,
                              content_type: str = "auto",
                              allowed_characters: Optional[List[str]] = None) -> str:
    """
    Generate text using corpus segmentation with fallback to synthetic generation.
    
    Args:
        corpus_lines: Pre-loaded corpus lines (if None, will load automatically)
        target_length: Target text length
        content_type: Type of content to generate
        allowed_characters: Allowed characters for curriculum learning
        
    Returns:
        Generated or extracted text
    """
    if corpus_lines is None:
        corpus_lines = load_khmer_corpus()
    
    if target_length is None:
        target_length = random.randint(1, 20)
    
    # For certain content types, prefer corpus extraction
    if content_type in ["auto", "characters", "syllables", "words", "phrases", "mixed"] and corpus_lines:
        # Try corpus extraction first
        retries = 50
        for i in range(retries):
            corpus_text = segment_corpus_text(
                corpus_lines,
                target_length=target_length,
                min_length=max(1, target_length - 5),
                max_length=target_length + 10,
                allowed_characters=allowed_characters
            )
            
            if corpus_text and len(corpus_text) >= 1:
                return corpus_text
            else:
                print(f"⚠️ Got empty text. retrying ({i}/{retries})...")
                continue
        print(f"⚠️ Failed to generate valid corpus text after {retries} retries. Using synthetic generation.")
    
    # Fallback to synthetic generation
    if content_type == "digits":
        return generate_digit_sequence(1, min(8, target_length))
    elif content_type == "characters":
        return generate_weighted_character_sequence(
            target_length, 
            character_frequencies={char: load_character_frequencies().get(char, 0.001) 
                                 for char in allowed_characters} if allowed_characters else None,
            character_set=allowed_characters
        )
    elif content_type == "syllables":
        num_syllables = max(1, target_length // 3)
        return "".join([generate_khmer_syllable() for _ in range(num_syllables)])
    else:
        return generate_mixed_content(target_length, "mixed")


def analyze_corpus_segments(corpus_file: str = "data/khmer_clean_text.txt", 
                           num_samples: int = 100) -> Dict:
    """
    Analyze corpus text segments to understand their characteristics.
    
    Args:
        corpus_file: Path to corpus file
        num_samples: Number of segments to analyze
        
    Returns:
        Analysis results dictionary
    """
    corpus_lines = load_khmer_corpus(corpus_file)
    if not corpus_lines:
        return {"error": "Could not load corpus"}
    
    segments = {
        'simple': extract_corpus_segments_by_complexity(corpus_lines, 'simple', num_samples // 3),
        'medium': extract_corpus_segments_by_complexity(corpus_lines, 'medium', num_samples // 3),
        'complex': extract_corpus_segments_by_complexity(corpus_lines, 'complex', num_samples // 3)
    }
    
    analysis = {
        'corpus_stats': {
            'total_lines': len(corpus_lines),
            'avg_line_length': sum(len(line) for line in corpus_lines) / len(corpus_lines),
            'total_characters': sum(len(line) for line in corpus_lines)
        },
        'segment_analysis': {}
    }
    
    for complexity, segs in segments.items():
        if segs:
            lengths = [len(seg) for seg in segs]
            char_counts = {}
            for seg in segs:
                for char in seg:
                    if '\u1780' <= char <= '\u17FF':  # Khmer range
                        char_counts[char] = char_counts.get(char, 0) + 1
            
            analysis['segment_analysis'][complexity] = {
                'count': len(segs),
                'avg_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'unique_characters': len(char_counts),
                'sample_segments': segs[:5],
                'top_characters': sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
    
    return analysis 