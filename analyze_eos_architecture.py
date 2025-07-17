#!/usr/bin/env python3
"""
Deep architectural analysis for EOS generation failure in Khmer OCR Seq2Seq model.
Investigates decoder structure, attention mechanisms, vocabulary handling, and data preprocessing.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.append('src')

from src.utils.config import ConfigManager
from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.onthefly_dataset import OnTheFlyDataset
# from src.data.corpus_dataset import KhmerCorpusDataset  # Not needed for this analysis

class EOSArchitecturalAnalyzer:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def analyze_vocabulary_structure(self):
        """Analyze vocabulary configuration and EOS token setup."""
        print("\n" + "="*60)
        print("VOCABULARY STRUCTURE ANALYSIS")
        print("="*60)
        
        # Load vocabulary config
        vocab = self.config_manager.vocab
        print(f"Vocabulary size: {len(vocab)}")
        print(f"PAD token index: {vocab.PAD_IDX}")
        print(f"EOS token index: {vocab.EOS_IDX}")
        print(f"UNK token index: {vocab.UNK_IDX}")
        print(f"SOS token index: {vocab.SOS_IDX}")
        
        # Check if EOS is properly defined
        print(f"\nSpecial tokens in vocabulary:")
        for token in vocab.special_tokens:
            idx = vocab.char_to_idx[token]
            print(f"  {token}: {idx}")
        
        # Verify EOS token position
        eos_idx = vocab.EOS_IDX
        if eos_idx < len(vocab):
            print(f"\n✅ EOS token properly positioned at index {eos_idx}")
        else:
            print(f"\n❌ EOS token index {eos_idx} exceeds vocabulary size!")
            
        return eos_idx
    
    def analyze_model_architecture(self):
        """Analyze the seq2seq model architecture for EOS generation capability."""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE ANALYSIS")
        print("="*60)
        
        # Create model
        model = KhmerOCRSeq2Seq(self.config_manager)
        model.to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Analyze decoder structure
        decoder = model.decoder
        print(f"\nDecoder architecture:")
        print(f"  Embedding dim: {decoder.embedding_dim}")
        print(f"  Decoder hidden size: {decoder.decoder_hidden_size}")
        print(f"  Encoder hidden size: {decoder.encoder_hidden_size}")
        print(f"  Vocab size: {decoder.vocab_size}")
        print(f"  GRU layers: {decoder.num_layers}")
        print(f"  Max length: {decoder.max_length}")
        print(f"  Use coverage: {decoder.use_coverage}")
        
        # Check output projection layer
        if hasattr(decoder, 'out'):
            out_layer = decoder.out
            print(f"  Output projection: {out_layer.in_features} -> {out_layer.out_features}")
            print(f"  Bias enabled: {out_layer.bias is not None}")
            
            # Analyze initial weights for EOS position
            with torch.no_grad():
                vocab = self.config_manager.vocab
                eos_idx = vocab.EOS_IDX
                eos_weights = out_layer.weight[eos_idx]
                eos_bias = out_layer.bias[eos_idx] if out_layer.bias is not None else 0
                
                print(f"\n  EOS token weights analysis:")
                print(f"    Weight mean: {eos_weights.mean().item():.6f}")
                print(f"    Weight std: {eos_weights.std().item():.6f}")
                print(f"    Weight range: [{eos_weights.min().item():.6f}, {eos_weights.max().item():.6f}]")
                print(f"    Bias: {eos_bias:.6f}")
        
        return model
    
    def analyze_attention_mechanism(self, model):
        """Analyze attention mechanism for EOS token visibility."""
        print("\n" + "="*60)
        print("ATTENTION MECHANISM ANALYSIS")
        print("="*60)
        
        decoder = model.decoder
        if hasattr(decoder, 'attention'):
            attention = decoder.attention
            print(f"Attention type: {type(attention).__name__}")
            print(f"Attention hidden size: {attention.attention_hidden_size}")
            print(f"Encoder hidden size: {attention.encoder_hidden_size}")
            print(f"Decoder hidden size: {attention.decoder_hidden_size}")
            print(f"Coverage enabled: {attention.coverage}")
            
            # Check attention weight initialization
            if hasattr(attention, 'encoder_projection'):
                with torch.no_grad():
                    enc_proj_stats = attention.encoder_projection.weight
                    print(f"Encoder projection weights - mean: {enc_proj_stats.mean().item():.6f}, std: {enc_proj_stats.std().item():.6f}")
            
            if hasattr(attention, 'decoder_projection'):
                with torch.no_grad():
                    dec_proj_stats = attention.decoder_projection.weight
                    print(f"Decoder projection weights - mean: {dec_proj_stats.mean().item():.6f}, std: {dec_proj_stats.std().item():.6f}")
        else:
            print("❌ No attention mechanism found in decoder!")
    
    def analyze_data_preprocessing(self):
        """Analyze data preprocessing pipeline for EOS token handling."""
        print("\n" + "="*60)
        print("DATA PREPROCESSING ANALYSIS")
        print("="*60)
        
        try:
            # Create dataset to check EOS handling
            dataset = OnTheFlyDataset(
                config_manager=self.config_manager,
                mode='train'
            )
            
            print(f"Dataset size: {len(dataset)}")
            
            # Sample a few examples
            eos_count = 0
            total_sequences = 0
            sequence_lengths = []
            
            for i in range(min(10, len(dataset))):
                sample = dataset[i]
                
                if isinstance(sample, dict):
                    target_seq = sample.get('targets', sample.get('target_sequence', []))
                else:
                    target_seq = sample[1] if len(sample) > 1 else []
                
                if isinstance(target_seq, torch.Tensor):
                    target_seq = target_seq.tolist()
                
                # Count EOS tokens
                vocab = self.config_manager.vocab
                eos_idx = vocab.EOS_IDX
                eos_in_seq = target_seq.count(eos_idx)
                eos_count += eos_in_seq
                total_sequences += 1
                sequence_lengths.append(len(target_seq))
                
                if i < 3:  # Show first 3 sequences in detail
                    print(f"\nSample {i}:")
                    print(f"  Target length: {len(target_seq)}")
                    print(f"  EOS tokens: {eos_in_seq}")
                    print(f"  Last 5 tokens: {target_seq[-5:] if len(target_seq) >= 5 else target_seq}")
                    print(f"  Contains EOS: {eos_idx in target_seq}")
            
            print(f"\nEOS Statistics:")
            print(f"  Total EOS tokens found: {eos_count}")
            print(f"  Average EOS per sequence: {eos_count / total_sequences:.2f}")
            print(f"  Average sequence length: {np.mean(sequence_lengths):.1f}")
            print(f"  Sequence length range: {min(sequence_lengths)} - {max(sequence_lengths)}")
            
            return eos_count > 0
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
            return False
    
    def analyze_loss_computation(self):
        """Analyze how EOS tokens are handled in loss computation."""
        print("\n" + "="*60)
        print("LOSS COMPUTATION ANALYSIS")
        print("="*60)
        
        # Create sample tensors to test loss behavior
        vocab = self.config_manager.vocab
        vocab_size = len(vocab)
        eos_idx = vocab.EOS_IDX
        pad_idx = vocab.PAD_IDX
        
        # Sample prediction logits (before softmax)
        batch_size, seq_len = 2, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Sample targets with EOS
        targets = torch.randint(0, vocab_size-1, (batch_size, seq_len))
        targets[0, 5] = eos_idx  # Place EOS in middle of first sequence
        targets[1, 8] = eos_idx  # Place EOS near end of second sequence
        targets[:, 6:] = pad_idx  # Pad after EOS
        
        print(f"Sample targets shape: {targets.shape}")
        print(f"EOS positions: {torch.where(targets == eos_idx)}")
        
        # Test standard CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        print(f"Standard CE loss: {loss.item():.4f}")
        
        # Test loss for EOS positions specifically
        eos_mask = (targets == eos_idx)
        if eos_mask.any():
            eos_logits = logits[eos_mask]
            eos_targets = targets[eos_mask]
            eos_loss = criterion(eos_logits, eos_targets)
            print(f"EOS-specific loss: {eos_loss.item():.4f}")
        else:
            print("No EOS tokens in sample")
    
    def test_model_eos_generation(self, model):
        """Test model's ability to generate EOS tokens."""
        print("\n" + "="*60)
        print("EOS GENERATION TEST")
        print("="*60)
        
        model.eval()
        vocab = self.config_manager.vocab
        vocab_size = len(vocab)
        eos_idx = vocab.EOS_IDX
        
        # Create dummy encoder output
        batch_size = 1
        encoder_seq_len = 20
        hidden_size = model.encoder.gru_hidden_size * 2  # Bidirectional
        
        encoder_outputs = torch.randn(batch_size, encoder_seq_len, hidden_size).to(self.device)
        encoder_hidden = torch.randn(1, batch_size, hidden_size).to(self.device)
        
        # Test decoder step by step
        with torch.no_grad():
            decoder_input = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)  # Start token
            decoder_hidden = encoder_hidden
            
            eos_probabilities = []
            max_steps = 20
            
            for step in range(max_steps):
                # Forward pass through decoder
                if hasattr(model, 'decode_step'):
                    output, decoder_hidden, attention_weights = model.decode_step(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                else:
                    # Manual decode step
                    decoder_output, decoder_hidden = model.decoder.gru(decoder_input, decoder_hidden)
                    
                    if hasattr(model.decoder, 'attention'):
                        context, attention_weights = model.decoder.attention(decoder_hidden, encoder_outputs)
                        decoder_output = torch.cat([decoder_output, context], dim=2)
                    
                    output = model.decoder.out(decoder_output)
                
                # Get probabilities
                probs = torch.softmax(output.squeeze(1), dim=-1)
                eos_prob = probs[0, eos_idx].item()
                eos_probabilities.append(eos_prob)
                
                # Get next token
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                decoder_input = next_token.unsqueeze(1)
                
                print(f"Step {step}: EOS prob = {eos_prob:.6f}, Next token = {next_token.item()}")
                
                if next_token.item() == eos_idx:
                    print(f"✅ EOS generated at step {step}!")
                    break
            else:
                print(f"❌ No EOS generated in {max_steps} steps")
            
            print(f"\nEOS probability statistics:")
            print(f"  Mean: {np.mean(eos_probabilities):.6f}")
            print(f"  Max: {np.max(eos_probabilities):.6f}")
            print(f"  Min: {np.min(eos_probabilities):.6f}")
            print(f"  Final: {eos_probabilities[-1]:.6f}")
    
    def run_full_analysis(self):
        """Run complete architectural analysis."""
        print("🔍 KHMER OCR EOS ARCHITECTURAL ANALYSIS")
        print("="*80)
        
        # Step 1: Vocabulary analysis
        eos_idx = self.analyze_vocabulary_structure()
        
        # Step 2: Model architecture analysis
        model = self.analyze_model_architecture()
        
        # Step 3: Attention mechanism analysis
        self.analyze_attention_mechanism(model)
        
        # Step 4: Data preprocessing analysis
        data_has_eos = self.analyze_data_preprocessing()
        
        # Step 5: Loss computation analysis
        self.analyze_loss_computation()
        
        # Step 6: EOS generation test
        self.test_model_eos_generation(model)
        
        # Summary
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        issues_found = []
        if not data_has_eos:
            issues_found.append("❌ EOS tokens missing in training data")
        
        if len(issues_found) == 0:
            print("✅ No obvious architectural issues found")
            print("💡 EOS generation failure likely due to:")
            print("   - Insufficient EOS loss weighting")
            print("   - Teacher forcing schedule preventing EOS learning")
            print("   - Model overfitting to repetitive patterns")
            print("   - Need for curriculum learning approach")
        else:
            print("Issues found:")
            for issue in issues_found:
                print(f"  {issue}")

if __name__ == "__main__":
    analyzer = EOSArchitecturalAnalyzer()
    analyzer.run_full_analysis() 