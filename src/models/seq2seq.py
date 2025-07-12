"""
Sequence-to-Sequence Model with Attention

This module implements the complete Seq2Seq architecture for Khmer OCR,
combining CRNN encoder, attention mechanism, and GRU decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
import json

from .encoder import CRNNEncoder, EncoderConfig
from .decoder import AttentionGRUDecoder, DecoderConfig


class KhmerOCRSeq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence model for Khmer OCR.
    
    Combines CRNN encoder with attention-based GRU decoder for
    end-to-end text recognition from text-line images.
    """
    
    def __init__(
        self,
        config_or_vocab_size = None,
        encoder_config: Optional[EncoderConfig] = None,
        decoder_config: Optional[DecoderConfig] = None,
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        vocab_size: int = None  # For backward compatibility
    ):
        """
        Initialize Seq2Seq model.
        
        Args:
            config_or_vocab_size: Either ConfigManager or vocab_size int
            encoder_config: Configuration for encoder
            decoder_config: Configuration for decoder
            pad_token_id: Padding token ID
            sos_token_id: Start of sequence token ID
            eos_token_id: End of sequence token ID
            unk_token_id: Unknown token ID
            vocab_size: Vocab size (for backward compatibility)
        """
        super(KhmerOCRSeq2Seq, self).__init__()
        
        # Handle different initialization patterns
        if config_or_vocab_size is not None:
            if hasattr(config_or_vocab_size, 'vocab'):  # ConfigManager
                config_manager = config_or_vocab_size
                self.vocab_size = len(config_manager.vocab)
                self.pad_token_id = config_manager.vocab.PAD_IDX
                self.sos_token_id = config_manager.vocab.SOS_IDX
                self.eos_token_id = config_manager.vocab.EOS_IDX
                self.unk_token_id = config_manager.vocab.UNK_IDX
                
                # Create encoder config from ConfigManager
                if encoder_config is None:
                    encoder_config = EncoderConfig(
                        conv_channels=config_manager.model_config.encoder_conv_channels,
                        gru_hidden_size=config_manager.model_config.encoder_gru_hidden_size,
                        gru_num_layers=config_manager.model_config.encoder_gru_num_layers,
                        bidirectional=config_manager.model_config.encoder_bidirectional
                    )
                
                # Create decoder config from ConfigManager  
                if decoder_config is None:
                    # Calculate encoder output dimension
                    encoder_output_dim = encoder_config.gru_hidden_size
                    if encoder_config.bidirectional:
                        encoder_output_dim *= 2
                        
                    decoder_config = DecoderConfig(
                        vocab_size=self.vocab_size,
                        encoder_hidden_size=encoder_output_dim,
                        decoder_hidden_size=config_manager.model_config.decoder_hidden_size,
                        embedding_dim=config_manager.model_config.decoder_embedding_dim,
                        attention_hidden_size=config_manager.model_config.attention_hidden_size,
                        max_length=config_manager.model_config.decoder_max_length
                    )
            else:  # int vocab_size
                self.vocab_size = config_or_vocab_size
                self.pad_token_id = pad_token_id
                self.sos_token_id = sos_token_id
                self.eos_token_id = eos_token_id
                self.unk_token_id = unk_token_id
        elif vocab_size is not None:  # backward compatibility
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
            self.sos_token_id = sos_token_id
            self.eos_token_id = eos_token_id
            self.unk_token_id = unk_token_id
        else:
            # Default values
            self.vocab_size = 117
            self.pad_token_id = pad_token_id
            self.sos_token_id = sos_token_id
            self.eos_token_id = eos_token_id
            self.unk_token_id = unk_token_id
        
        # Default configurations
        if encoder_config is None:
            encoder_config = EncoderConfig()
        if decoder_config is None:
            decoder_config = DecoderConfig(vocab_size=self.vocab_size)
        
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        
        # Build model components
        self.encoder = encoder_config.create_encoder()
        self.decoder = decoder_config.create_decoder()
        
        # Verify dimension compatibility
        assert self.encoder.get_output_dim() == decoder_config.encoder_hidden_size, \
            f"Encoder output dim ({self.encoder.get_output_dim()}) must match decoder encoder_hidden_size ({decoder_config.encoder_hidden_size})"
        
        # Model info
        self._calculate_model_info()
    
    def _calculate_model_info(self):
        """Calculate model parameter information."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        self.model_info = {
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'total_params': total_params,
            'vocab_size': self.vocab_size
        }
    
    def forward(
        self,
        images: torch.Tensor,
        target_sequences: Optional[torch.Tensor] = None,
        image_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: Input images (B, 1, 32, W)
            target_sequences: Target sequences for training (B, target_len)
            image_lengths: Actual lengths of images (B,) for masking
            teacher_forcing_ratio: Teacher forcing ratio (1.0 = always use targets)
            
        Returns:
            Dictionary containing:
            - encoder_outputs: Encoder outputs (B, seq_len, hidden_size)
            - logits: Decoder logits (B, target_len, vocab_size) if targets provided
            - attention_weights: Attention weights (B, target_len, seq_len) if targets provided
            - loss: Cross-entropy loss if targets provided
        """
        batch_size = images.size(0)
        device = images.device
        
        # Encode images
        encoder_outputs, encoder_hidden = self.encoder(images)  # (B, seq_len, hidden_size)
        
        # Create encoder mask from image lengths
        encoder_mask = None
        if image_lengths is not None:
            seq_len = encoder_outputs.size(1)
            encoder_mask = self._create_sequence_mask(image_lengths, seq_len, device)
        
        result = {
            'encoder_outputs': encoder_outputs,
            'encoder_hidden': encoder_hidden
        }
        
        # Decode sequences if targets provided (training mode)
        if target_sequences is not None:
            decoder_results = self.decoder.forward(
                encoder_outputs=encoder_outputs,
                target_sequences=target_sequences,
                encoder_mask=encoder_mask,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            result.update(decoder_results)
            
            # Compute loss
            if 'logits' in decoder_results:
                loss = self.compute_loss(
                    logits=decoder_results['logits'],
                    targets=target_sequences,
                    ignore_index=self.pad_token_id
                )
                result['loss'] = loss
        
        return result
    
    def generate(
        self,
        images: torch.Tensor,
        image_lengths: Optional[torch.Tensor] = None,
        max_length: int = 100,
        method: str = 'greedy',
        beam_size: int = 5,
        length_penalty: float = 0.6,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text sequences from images.
        
        Args:
            images: Input images (B, 1, 32, W)
            image_lengths: Actual lengths of images (B,)
            max_length: Maximum sequence length
            method: Generation method ('greedy' or 'beam_search')
            beam_size: Beam size for beam search
            length_penalty: Length penalty for beam search
            temperature: Temperature for sampling
            
        Returns:
            Dictionary containing:
            - sequences: Generated sequences (B, gen_len) or (beam_size, gen_len) for beam search
            - scores: Generation scores
            - attention_weights: Attention weights
            - lengths: Actual sequence lengths
        """
        self.eval()
        
        with torch.no_grad():
            # Encode images
            encoder_outputs, _ = self.encoder(images)
            
            # Create encoder mask
            encoder_mask = None
            if image_lengths is not None:
                seq_len = encoder_outputs.size(1)
                encoder_mask = self._create_sequence_mask(image_lengths, seq_len, images.device)
            
            if method == 'greedy':
                return self.decoder.generate(
                    encoder_outputs=encoder_outputs,
                    encoder_mask=encoder_mask,
                    max_length=max_length,
                    sos_token=self.sos_token_id,
                    eos_token=self.eos_token_id,
                    temperature=temperature
                )
            elif method == 'beam_search':
                if images.size(0) != 1:
                    raise ValueError("Beam search only supports batch size 1")
                
                return self.decoder.beam_search(
                    encoder_outputs=encoder_outputs,
                    encoder_mask=encoder_mask,
                    beam_size=beam_size,
                    max_length=max_length,
                    sos_token=self.sos_token_id,
                    eos_token=self.eos_token_id,
                    length_penalty=length_penalty
                )
            else:
                raise ValueError(f"Unknown generation method: {method}")
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = 0,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model logits (B, seq_len, vocab_size)
            targets: Target sequences (B, seq_len)
            ignore_index: Index to ignore in loss computation
            label_smoothing: Label smoothing factor
            
        Returns:
            Cross-entropy loss
        """
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # (B*seq_len, vocab_size)
        targets_flat = targets.view(-1)  # (B*seq_len,)
        
        if label_smoothing > 0:
            loss = self._label_smoothing_loss(
                logits_flat, targets_flat, ignore_index, label_smoothing
            )
        else:
            loss = F.cross_entropy(
                logits_flat, targets_flat, ignore_index=ignore_index
            )
        
        return loss
    
    def _label_smoothing_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        smoothing: float
    ) -> torch.Tensor:
        """Compute label smoothing loss."""
        vocab_size = logits.size(-1)
        
        # Create smoothed targets
        confidence = 1.0 - smoothing
        smooth_target = smoothing / (vocab_size - 1)
        
        # One-hot encode targets
        one_hot = torch.zeros_like(logits).fill_(smooth_target)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        
        # Mask ignored indices
        mask = (targets != ignore_index).unsqueeze(1).float()
        one_hot = one_hot * mask
        
        # Compute loss
        log_prob = F.log_softmax(logits, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1)
        
        # Average over valid positions
        valid_positions = mask.squeeze(1).sum()
        if valid_positions > 0:
            loss = loss.sum() / valid_positions
        else:
            loss = loss.sum()
        
        return loss
    
    def _create_sequence_mask(
        self,
        lengths: torch.Tensor,
        max_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create sequence mask for variable length sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_length, device=device).expand(
            batch_size, max_length
        ) < lengths.unsqueeze(1)
        return mask.float()
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return self.model_info.copy()
    
    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: int = 0,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        best_score: Optional[float] = None,
        additional_info: Optional[Dict] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            best_score: Best validation score
            additional_info: Additional information to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'encoder_config': self.encoder_config.__dict__,
            'decoder_config': self.decoder_config.__dict__,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'pad_token_id': self.pad_token_id,
                'sos_token_id': self.sos_token_id,
                'eos_token_id': self.eos_token_id,
                'unk_token_id': self.unk_token_id
            },
            'model_info': self.model_info
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        if best_score is not None:
            checkpoint['best_score'] = best_score
        if additional_info is not None:
            checkpoint['additional_info'] = additional_info
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, Path],
        device: Optional[torch.device] = None,
        map_location: Optional[str] = None
    ) -> Tuple['KhmerOCRSeq2Seq', Dict]:
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            map_location: Map location for torch.load
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        if map_location is None and device is not None:
            map_location = str(device)
        
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        
        # Handle different checkpoint formats
        if 'encoder_config' in checkpoint and 'decoder_config' in checkpoint:
            # New format with separate encoder/decoder configs
            encoder_config = EncoderConfig(**checkpoint['encoder_config'])
            decoder_config = DecoderConfig(**checkpoint['decoder_config'])
            vocab_size = checkpoint['vocab_size']
            special_tokens = checkpoint['special_tokens']
            
        elif 'config' in checkpoint:
            # Old format with unified config
            config_dict = checkpoint['config']
            
            # Extract encoder config
            encoder_config = EncoderConfig(
                conv_channels=config_dict.get('encoder_conv_channels', [64, 128, 256, 256, 512, 512, 512]),
                gru_hidden_size=config_dict.get('encoder_gru_hidden_size', 256),
                gru_num_layers=config_dict.get('encoder_gru_num_layers', 2),
                bidirectional=config_dict.get('encoder_bidirectional', True),
                dropout=config_dict.get('dropout', 0.1)
            )
            
            # Extract decoder config
            encoder_output_size = encoder_config.gru_hidden_size * (2 if encoder_config.bidirectional else 1)
            decoder_config = DecoderConfig(
                vocab_size=config_dict.get('decoder_vocab_size', 117),
                encoder_hidden_size=encoder_output_size,
                decoder_hidden_size=config_dict.get('decoder_hidden_size', 256),  # Use original 256
                embedding_dim=config_dict.get('decoder_embedding_dim', 256),
                attention_hidden_size=config_dict.get('attention_hidden_size', 256),  # Use original 256
                num_layers=config_dict.get('decoder_num_layers', 2),
                dropout=config_dict.get('dropout', 0.1),
                use_coverage=config_dict.get('use_coverage', False),
                max_length=config_dict.get('decoder_max_length', 256)
            )
            
            vocab_size = config_dict.get('decoder_vocab_size', 117)
            special_tokens = {
                'pad_token_id': 2,  # PAD token
                'sos_token_id': 0,  # SOS token  
                'eos_token_id': 1,  # EOS token
                'unk_token_id': 3   # UNK token
            }
            
        else:
            raise ValueError("Checkpoint format not recognized. Missing 'encoder_config' or 'config' key.")
        
        # Create model
        model = cls(
            vocab_size=vocab_size,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            **special_tokens
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        # Return checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'best_score': checkpoint.get('best_score') or checkpoint.get('best_cer'),
            'additional_info': checkpoint.get('additional_info'),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
            'metrics': checkpoint.get('metrics')
        }
        
        return model, checkpoint_info
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to encode images and get initial decoder state.
        
        Args:
            images: Input images (B, 1, 32, W)
            
        Returns:
            Tuple of (encoder_outputs, decoder_initial_hidden)
        """
        encoder_outputs, encoder_hidden = self.encoder(images)
        # Initialize decoder hidden state from encoder
        decoder_hidden = self.decoder.init_hidden(images.size(0), images.device)
        return encoder_outputs, decoder_hidden
    
    def decode_step(
        self,
        input_token: torch.Tensor,
        hidden_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method for single decoder step.
        
        Args:
            input_token: Input token (B, 1)
            hidden_state: Decoder hidden state
            encoder_outputs: Encoder outputs (B, seq_len, hidden_size)
            encoder_mask: Encoder mask (B, seq_len)
            
        Returns:
            Tuple of (log_probs, new_hidden_state, attention_weights)
        """
        log_probs, new_hidden, attention_weights, _ = self.decoder.forward_step(
            input_token=input_token.squeeze(-1),  # Remove last dim for forward_step
            hidden_state=hidden_state,
            encoder_outputs=encoder_outputs,
            encoder_mask=encoder_mask,
            coverage_vector=None
        )
        return log_probs, new_hidden, attention_weights


class Seq2SeqConfig:
    """Configuration for complete Seq2Seq model."""
    
    def __init__(
        self,
        vocab_size: int = 117,
        # Encoder config
        encoder_conv_channels: List[int] = None,
        encoder_gru_hidden_size: int = 256,
        encoder_gru_layers: int = 2,
        # Decoder config
        decoder_hidden_size: int = 256,
        decoder_embedding_dim: int = 256,
        decoder_num_layers: int = 2,
        # Attention config
        attention_hidden_size: int = 256,
        attention_type: str = "bahdanau",
        use_coverage: bool = False,
        # Training config
        dropout: float = 0.1,
        max_length: int = 100,
        # Special tokens
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3
    ):
        self.vocab_size = vocab_size
        
        # Create component configs
        self.encoder_config = EncoderConfig(
            conv_channels=encoder_conv_channels,
            gru_hidden_size=encoder_gru_hidden_size,
            gru_num_layers=encoder_gru_layers,
            dropout=dropout
        )
        
        self.decoder_config = DecoderConfig(
            vocab_size=vocab_size,
            encoder_hidden_size=encoder_gru_hidden_size * 2,  # Bidirectional
            decoder_hidden_size=decoder_hidden_size,
            embedding_dim=decoder_embedding_dim,
            attention_hidden_size=attention_hidden_size,
            num_layers=decoder_num_layers,
            dropout=dropout,
            use_coverage=use_coverage,
            max_length=max_length
        )
        
        # Special tokens
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
    
    def create_model(self) -> KhmerOCRSeq2Seq:
        """Create Seq2Seq model from config."""
        return KhmerOCRSeq2Seq(
            vocab_size=self.vocab_size,
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            pad_token_id=self.pad_token_id,
            sos_token_id=self.sos_token_id,
            eos_token_id=self.eos_token_id,
            unk_token_id=self.unk_token_id
        )
    
    def save_config(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        config_dict = {
            'vocab_size': self.vocab_size,
            'encoder_config': self.encoder_config.__dict__,
            'decoder_config': self.decoder_config.__dict__,
            'special_tokens': {
                'pad_token_id': self.pad_token_id,
                'sos_token_id': self.sos_token_id,
                'eos_token_id': self.eos_token_id,
                'unk_token_id': self.unk_token_id
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> 'Seq2SeqConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = cls(vocab_size=config_dict['vocab_size'])
        
        # Update with loaded values
        config.encoder_config = EncoderConfig(**config_dict['encoder_config'])
        config.decoder_config = DecoderConfig(**config_dict['decoder_config'])
        
        special_tokens = config_dict['special_tokens']
        config.pad_token_id = special_tokens['pad_token_id']
        config.sos_token_id = special_tokens['sos_token_id']
        config.eos_token_id = special_tokens['eos_token_id']
        config.unk_token_id = special_tokens['unk_token_id']
        
        return config 