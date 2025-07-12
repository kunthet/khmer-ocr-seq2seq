"""
Unit tests for model components.
Tests encoder, decoder, attention, and complete seq2seq model.
"""
import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from models.encoder import CRNNEncoder, EncoderConfig
from models.attention import BahdanauAttention, MultiHeadBahdanauAttention
from models.decoder import AttentionGRUDecoder, DecoderConfig
from models.seq2seq import KhmerOCRSeq2Seq


class TestCRNNEncoder(unittest.TestCase):
    """Test cases for CRNN Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EncoderConfig()
        self.encoder = CRNNEncoder(self.config)
    
    def test_encoder_creation(self):
        """Test encoder creation and basic properties."""
        self.assertIsInstance(self.encoder, CRNNEncoder)
        self.assertEqual(self.encoder.get_output_dim(), 512)
    
    def test_forward_pass(self):
        """Test encoder forward pass with various input sizes."""
        batch_size = 2
        
        # Test different input widths
        widths = [64, 128, 256]
        
        for width in widths:
            with self.subTest(width=width):
                # Input: [B, 1, 32, W]
                x = torch.randn(batch_size, 1, 32, width)
                
                # Forward pass
                encoder_outputs, encoder_hidden = self.encoder(x)
                
                # Check output shapes
                self.assertEqual(len(encoder_outputs.shape), 3)  # [B, seq_len, hidden_size]
                self.assertEqual(encoder_outputs.shape[0], batch_size)
                self.assertEqual(encoder_outputs.shape[2], 512)
                
                # Check hidden state
                self.assertEqual(len(encoder_hidden.shape), 3)  # [num_layers, B, hidden_size]
                self.assertEqual(encoder_hidden.shape[1], batch_size)
    
    def test_output_sequence_length(self):
        """Test that output sequence length is correct."""
        x = torch.randn(1, 1, 32, 128)
        encoder_outputs, _ = self.encoder(x)
        
        # Based on the architecture, width 128 should result in sequence length 30
        # (128 / 4 - 2 = 30)
        expected_seq_len = 128 // 4 - 2
        self.assertEqual(encoder_outputs.shape[1], expected_seq_len)
    
    def test_parameter_count(self):
        """Test encoder parameter count."""
        total_params = sum(p.numel() for p in self.encoder.parameters())
        
        # Should be around 14-15M parameters
        self.assertGreater(total_params, 10_000_000)
        self.assertLess(total_params, 20_000_000)


class TestBahdanauAttention(unittest.TestCase):
    """Test cases for Bahdanau Attention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder_hidden_size = 512
        self.decoder_hidden_size = 256
        self.attention_hidden_size = 256
        
        self.attention = BahdanauAttention(
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            attention_hidden_size=self.attention_hidden_size
        )
    
    def test_attention_creation(self):
        """Test attention mechanism creation."""
        self.assertIsInstance(self.attention, BahdanauAttention)
    
    def test_forward_pass(self):
        """Test attention forward pass."""
        batch_size = 2
        seq_len = 30
        
        # Create dummy inputs
        encoder_outputs = torch.randn(batch_size, seq_len, self.encoder_hidden_size)
        decoder_hidden = torch.randn(batch_size, self.decoder_hidden_size)
        
        # Forward pass
        context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden)
        
        # Check output shapes
        self.assertEqual(context_vector.shape, (batch_size, self.encoder_hidden_size))
        self.assertEqual(attention_weights.shape, (batch_size, seq_len))
        
        # Check attention weights sum to 1
        weights_sum = attention_weights.sum(dim=1)
        torch.testing.assert_close(weights_sum, torch.ones(batch_size), atol=1e-5, rtol=1e-5)
    
    def test_attention_with_mask(self):
        """Test attention with encoder mask."""
        batch_size = 2
        seq_len = 30
        
        encoder_outputs = torch.randn(batch_size, seq_len, self.encoder_hidden_size)
        decoder_hidden = torch.randn(batch_size, self.decoder_hidden_size)
        
        # Create mask (mask out last 10 positions for first sample)
        encoder_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        encoder_mask[0, -10:] = False
        
        context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden, encoder_mask)
        
        # Check that masked positions have zero attention
        self.assertTrue(torch.all(attention_weights[0, -10:] == 0))
        
        # Check that weights still sum to 1
        weights_sum = attention_weights.sum(dim=1)
        torch.testing.assert_close(weights_sum, torch.ones(batch_size), atol=1e-5, rtol=1e-5)


class TestMultiHeadBahdanauAttention(unittest.TestCase):
    """Test cases for Multi-Head Bahdanau Attention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.attention = MultiHeadBahdanauAttention(
            encoder_hidden_size=512,
            decoder_hidden_size=256,
            attention_hidden_size=256,
            num_heads=4
        )
    
    def test_multihead_forward_pass(self):
        """Test multi-head attention forward pass."""
        batch_size = 2
        seq_len = 30
        
        encoder_outputs = torch.randn(batch_size, seq_len, 512)
        decoder_hidden = torch.randn(batch_size, 256)
        
        context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden)
        
        # Check output shapes
        self.assertEqual(context_vector.shape, (batch_size, 512))
        self.assertEqual(attention_weights.shape, (batch_size, seq_len))


class TestAttentionGRUDecoder(unittest.TestCase):
    """Test cases for Attention GRU Decoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DecoderConfig(
            vocab_size=117,
            encoder_hidden_size=512,
            decoder_hidden_size=256,
            embedding_dim=256
        )
        self.decoder = self.config.create_decoder()
    
    def test_decoder_creation(self):
        """Test decoder creation."""
        self.assertIsInstance(self.decoder, AttentionGRUDecoder)
        self.assertEqual(self.decoder.vocab_size, 117)
    
    def test_init_hidden(self):
        """Test hidden state initialization."""
        batch_size = 2
        device = torch.device('cpu')
        
        hidden = self.decoder.init_hidden(batch_size, device)
        
        # Check shape: [num_layers, batch_size, hidden_size]
        self.assertEqual(hidden.shape, (self.config.num_layers, batch_size, self.config.decoder_hidden_size))
    
    def test_forward_step(self):
        """Test single decoder step."""
        batch_size = 2
        seq_len = 30
        
        # Create inputs
        input_token = torch.randint(0, 117, (batch_size,))
        hidden_state = self.decoder.init_hidden(batch_size, torch.device('cpu'))
        encoder_outputs = torch.randn(batch_size, seq_len, 512)
        
        # Forward step
        logits, new_hidden, attention_weights, coverage = self.decoder.forward_step(
            input_token, hidden_state, encoder_outputs
        )
        
        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, 117))
        self.assertEqual(new_hidden.shape, hidden_state.shape)
        self.assertEqual(attention_weights.shape, (batch_size, seq_len))
    
    def test_generation(self):
        """Test text generation."""
        batch_size = 1
        seq_len = 30
        
        encoder_outputs = torch.randn(batch_size, seq_len, 512)
        
        # Generate text
        result = self.decoder.generate(
            encoder_outputs=encoder_outputs,
            max_length=10,
            sos_token=0,
            eos_token=1
        )
        
        # Check result structure
        self.assertIn('sequences', result)
        self.assertIn('attention_weights', result)
        self.assertIn('lengths', result)
        
        sequences = result['sequences']
        self.assertEqual(sequences.shape[0], batch_size)
        self.assertLessEqual(sequences.shape[1], 10)  # Max length


class TestKhmerOCRSeq2Seq(unittest.TestCase):
    """Test cases for complete Seq2Seq model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = KhmerOCRSeq2Seq(vocab_size=117)
    
    def test_model_creation(self):
        """Test model creation and basic properties."""
        self.assertIsInstance(self.model, KhmerOCRSeq2Seq)
        self.assertEqual(self.model.vocab_size, 117)
        
        # Check model info
        info = self.model.get_model_info()
        self.assertIn('total_params', info)
        self.assertIn('encoder_params', info)
        self.assertIn('decoder_params', info)
    
    def test_forward_pass_training(self):
        """Test forward pass in training mode."""
        batch_size = 2
        image_width = 128
        seq_len = 10
        
        # Create inputs
        images = torch.randn(batch_size, 1, 32, image_width)
        targets = torch.randint(0, 117, (batch_size, seq_len))
        
        # Forward pass
        result = self.model.forward(images, targets)
        
        # Check outputs
        self.assertIn('encoder_outputs', result)
        self.assertIn('logits', result)
        self.assertIn('attention_weights', result)
        self.assertIn('loss', result)
        
        # Check shapes
        logits = result['logits']
        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[2], 117)  # Vocab size
    
    def test_forward_pass_inference(self):
        """Test forward pass in inference mode."""
        batch_size = 2
        image_width = 128
        
        images = torch.randn(batch_size, 1, 32, image_width)
        
        # Forward pass without targets (inference mode)
        result = self.model.forward(images)
        
        # Should only have encoder outputs
        self.assertIn('encoder_outputs', result)
        self.assertNotIn('logits', result)
        self.assertNotIn('loss', result)
    
    def test_generation(self):
        """Test text generation."""
        batch_size = 1
        image_width = 128
        
        images = torch.randn(batch_size, 1, 32, image_width)
        
        # Generate with greedy decoding
        result = self.model.generate(images, method='greedy', max_length=20)
        
        self.assertIn('sequences', result)
        self.assertIn('attention_weights', result)
        
        sequences = result['sequences']
        self.assertEqual(sequences.shape[0], batch_size)
    
    def test_beam_search_generation(self):
        """Test beam search generation."""
        batch_size = 1
        image_width = 128
        
        images = torch.randn(batch_size, 1, 32, image_width)
        
        # Generate with beam search
        result = self.model.generate(images, method='beam_search', beam_size=3, max_length=20)
        
        self.assertIn('sequences', result)
        self.assertIn('scores', result)
        
        sequences = result['sequences']
        scores = result['scores']
        self.assertEqual(sequences.shape[0], 3)  # Beam size
        self.assertEqual(len(scores), 3)
    
    def test_parameter_count(self):
        """Test total parameter count."""
        total_params = self.model.count_parameters()
        
        # Should be around 16M parameters
        self.assertGreater(total_params, 15_000_000)
        self.assertLess(total_params, 18_000_000)
    
    def test_model_modes(self):
        """Test training and evaluation modes."""
        # Training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_convenience_methods(self):
        """Test encode and decode_step convenience methods."""
        batch_size = 1
        image_width = 128
        
        images = torch.randn(batch_size, 1, 32, image_width)
        
        # Test encode method
        encoder_outputs, decoder_hidden = self.model.encode(images)
        
        self.assertEqual(encoder_outputs.shape[0], batch_size)
        self.assertEqual(decoder_hidden.shape[1], batch_size)
        
        # Test decode_step method
        input_token = torch.randint(0, 117, (batch_size, 1))
        logits, new_hidden, attention_weights = self.model.decode_step(
            input_token, decoder_hidden, encoder_outputs
        )
        
        self.assertEqual(logits.shape, (batch_size, 117))
        self.assertEqual(new_hidden.shape, decoder_hidden.shape)


if __name__ == '__main__':
    unittest.main() 