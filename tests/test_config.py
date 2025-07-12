"""
Test configuration management and vocabulary functionality.
"""
import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config import ConfigManager, KhmerVocab, ModelConfig, TrainingConfig, DataConfig


class TestKhmerVocab:
    """Test Khmer vocabulary functionality."""
    
    def test_vocab_initialization(self):
        """Test vocabulary initializes correctly."""
        vocab = KhmerVocab()
        
        # Test vocabulary size
        assert len(vocab) == 117, f"Expected 117 tokens, got {len(vocab)}"
        
        # Test special tokens exist
        assert "SOS" in vocab.vocab
        assert "EOS" in vocab.vocab
        assert "PAD" in vocab.vocab
        assert "UNK" in vocab.vocab
        
        # Test special token indices
        assert vocab.SOS_IDX == 0
        assert vocab.EOS_IDX == 1
        assert vocab.PAD_IDX == 2
        assert vocab.UNK_IDX == 3
    
    def test_char_mappings(self):
        """Test character to index mappings."""
        vocab = KhmerVocab()
        
        # Test basic character encoding/decoding
        test_char = "ក"  # First Khmer consonant
        if test_char in vocab.char_to_idx:
            idx = vocab.char_to_idx[test_char]
            decoded_char = vocab.idx_to_char[idx]
            assert decoded_char == test_char
    
    def test_encode_decode(self):
        """Test text encoding and decoding."""
        vocab = KhmerVocab()
        
        # Test simple text
        test_text = "ក។"  # Khmer consonant and period
        encoded = vocab.encode(test_text)
        decoded = vocab.decode(encoded)
        
        # Check that known characters are preserved
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
        assert len(encoded) == len(test_text)
    
    def test_unknown_character_handling(self):
        """Test handling of unknown characters."""
        vocab = KhmerVocab()
        
        # Test with unknown character
        unknown_char = "X"  # Should not be in Khmer vocab
        encoded = vocab.encode(unknown_char)
        assert encoded[0] == vocab.UNK_IDX


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def test_config_initialization(self):
        """Test config manager initializes with defaults."""
        # Test without config file
        config = ConfigManager(config_path="nonexistent.yaml")
        
        assert isinstance(config.model_config, ModelConfig)
        assert isinstance(config.training_config, TrainingConfig)
        assert isinstance(config.data_config, DataConfig)
        assert isinstance(config.vocab, KhmerVocab)
    
    def test_device_detection(self):
        """Test device detection works."""
        config = ConfigManager()
        device = config.get_device()
        
        # Should return a valid device
        assert str(device) in ['cpu', 'cuda', 'mps']
    
    def test_model_config_defaults(self):
        """Test model configuration defaults."""
        config = ConfigManager()
        
        # Test encoder defaults
        assert config.model_config.encoder_input_channels == 1
        assert config.model_config.encoder_gru_hidden_size == 256
        assert config.model_config.encoder_bidirectional == True
        
        # Test decoder defaults
        assert config.model_config.decoder_vocab_size == 117
        assert config.model_config.decoder_embedding_dim == 256
    
    def test_training_config_defaults(self):
        """Test training configuration defaults."""
        config = ConfigManager()
        
        assert config.training_config.epochs == 150
        assert config.training_config.batch_size == 64
        # Handle both string and float learning rate values
        lr = config.training_config.learning_rate
        expected_lr = 1e-6
        if isinstance(lr, str):
            lr = float(lr)
        assert abs(lr - expected_lr) < 1e-10
        assert config.training_config.teacher_forcing_ratio == 1.0


if __name__ == "__main__":
    # Run basic tests
    test_vocab = TestKhmerVocab()
    test_vocab.test_vocab_initialization()
    test_vocab.test_char_mappings()
    test_vocab.test_encode_decode()
    test_vocab.test_unknown_character_handling()
    
    test_config = TestConfigManager()
    test_config.test_config_initialization()
    test_config.test_device_detection()
    test_config.test_model_config_defaults()
    test_config.test_training_config_defaults()
    
    print("✅ All configuration tests passed!") 