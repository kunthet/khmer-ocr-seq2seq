"""
Unit tests for Khmer vocabulary system.
Tests encoding, decoding, and vocabulary properties.
"""
import unittest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from utils.config import KhmerVocab


class TestKhmerVocab(unittest.TestCase):
    """Test cases for KhmerVocab class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab = KhmerVocab()
    
    def test_vocab_size(self):
        """Test that vocabulary has exactly 117 tokens."""
        self.assertEqual(len(self.vocab), 117)
        self.assertEqual(len(self.vocab.vocab), 117)
    
    def test_special_tokens(self):
        """Test special token properties."""
        # Check special tokens exist
        special_tokens = ["SOS", "EOS", "PAD", "UNK"]
        for token in special_tokens:
            self.assertIn(token, self.vocab.vocab)
        
        # Check special token indices
        self.assertEqual(self.vocab.SOS_IDX, 0)
        self.assertEqual(self.vocab.EOS_IDX, 1)
        self.assertEqual(self.vocab.PAD_IDX, 2)
        self.assertEqual(self.vocab.UNK_IDX, 3)
    
    def test_character_categories(self):
        """Test that all character categories are present."""
        # Numbers
        khmer_numbers = ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]
        arabic_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        for num in khmer_numbers + arabic_numbers:
            self.assertIn(num, self.vocab.vocab)
        
        # Common consonants
        common_consonants = ["ក", "ខ", "គ", "ង", "ច", "ញ", "ត", "ន", "ប", "ម", "យ", "រ", "ល", "ស", "ហ", "អ"]
        for consonant in common_consonants:
            self.assertIn(consonant, self.vocab.vocab)
        
        # Common vowels
        common_vowels = ["ា", "ិ", "ី", "ុ", "ូ", "េ", "ោ"]
        for vowel in common_vowels:
            self.assertIn(vowel, self.vocab.vocab)
        
        # Symbols
        symbols = ["(", ")", ",", ".", "។", " "]
        for symbol in symbols:
            self.assertIn(symbol, self.vocab.vocab)
    
    def test_encoding_decoding(self):
        """Test text encoding and decoding."""
        # Test simple Khmer word
        text = "កម្ពុជា"
        encoded = self.vocab.encode(text)
        decoded = self.vocab.decode(encoded)
        
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertEqual(decoded, text)
    
    def test_unknown_character_handling(self):
        """Test handling of unknown characters."""
        # Test with unknown character
        text = "កា🙂"  # Contains emoji
        encoded = self.vocab.encode(text)
        
        # Should contain UNK token
        self.assertIn(self.vocab.UNK_IDX, encoded)
    
    def test_empty_string(self):
        """Test encoding/decoding empty string."""
        text = ""
        encoded = self.vocab.encode(text)
        decoded = self.vocab.decode(encoded)
        
        self.assertEqual(encoded, [])
        self.assertEqual(decoded, "")
    
    def test_mixed_script(self):
        """Test mixed Khmer and English text."""
        text = "កម្ពុជា123"
        encoded = self.vocab.encode(text)
        decoded = self.vocab.decode(encoded)
        
        self.assertEqual(decoded, text)
    
    def test_char_to_idx_mapping(self):
        """Test character to index mapping consistency."""
        for char, idx in self.vocab.char_to_idx.items():
            self.assertEqual(self.vocab.idx_to_char[idx], char)
    
    def test_idx_to_char_mapping(self):
        """Test index to character mapping consistency."""
        for idx, char in self.vocab.idx_to_char.items():
            self.assertEqual(self.vocab.char_to_idx[char], idx)
    
    def test_roundtrip_encoding(self):
        """Test that encoding then decoding preserves text."""
        test_texts = [
            "កម្ពុជា",
            "សួស្តី",
            "អរគុណ",
            "ភាសាខ្មែរ",
            "123456",
            "កា១២៣",
            "។",
            " ",
            "កម្ពុជា។ សួស្តី។"
        ]
        
        for text in test_texts:
            with self.subTest(text=text):
                encoded = self.vocab.encode(text)
                decoded = self.vocab.decode(encoded)
                self.assertEqual(decoded, text)
    
    def test_vocab_completeness(self):
        """Test that vocabulary contains expected number of each category."""
        # Count by category
        special_count = len(self.vocab.special_tokens)
        khmer_num_count = len(self.vocab.khmer_numbers)
        arabic_num_count = len(self.vocab.arabic_numbers)
        consonant_count = len(self.vocab.consonants)
        independent_vowel_count = len(self.vocab.independent_vowels)
        dependent_vowel_count = len(self.vocab.dependent_vowels)
        subscript_count = len(self.vocab.subscript)
        diacritic_count = len(self.vocab.diacritics)
        symbol_count = len(self.vocab.symbols)
        
        total = (special_count + khmer_num_count + arabic_num_count + 
                consonant_count + independent_vowel_count + dependent_vowel_count +
                subscript_count + diacritic_count + symbol_count)
        
        # Should equal total vocabulary size
        self.assertEqual(total, 117)
        
        # Verify expected counts
        self.assertEqual(special_count, 4)
        self.assertEqual(khmer_num_count, 10)
        self.assertEqual(arabic_num_count, 10)
        self.assertEqual(consonant_count, 33)


if __name__ == '__main__':
    unittest.main() 