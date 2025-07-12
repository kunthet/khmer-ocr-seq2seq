"""
GRU Decoder with attention for Khmer OCR Seq2Seq model.
Implements attention-based decoder with GRU cells.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Union
import random

from .attention import BahdanauAttention


class AttentionGRUDecoder(nn.Module):
    """
    GRU-based decoder with Bahdanau attention mechanism.
    
    This decoder generates output sequences token by token, using attention
    to focus on relevant parts of the encoder output at each step.
    """
    
    def __init__(
        self,
        vocab_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int = 256,
        embedding_dim: int = 256,
        attention_hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_coverage: bool = False,
        max_length: int = 100
    ):
        """
        Initialize attention-based GRU decoder.
        
        Args:
            vocab_size: Size of vocabulary (117 for Khmer)
            encoder_hidden_size: Hidden size of encoder outputs (512)
            decoder_hidden_size: Hidden size of decoder GRU
            embedding_dim: Dimension of token embeddings
            attention_hidden_size: Hidden size of attention computation
            num_layers: Number of GRU layers
            dropout: Dropout rate
            use_coverage: Whether to use coverage mechanism
            max_length: Maximum sequence length for generation
        """
        super(AttentionGRUDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.max_length = max_length
        self.use_coverage = use_coverage
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Attention mechanism
        self.attention = BahdanauAttention(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            attention_hidden_size=attention_hidden_size,
            coverage=use_coverage,
            dropout=dropout
        )
        
        # Input projection: combine embedding + context
        self.input_projection = nn.Linear(
            embedding_dim + encoder_hidden_size, decoder_hidden_size
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=decoder_hidden_size,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection: combine GRU output + context + embedding
        self.output_projection = nn.Linear(
            decoder_hidden_size + encoder_hidden_size + embedding_dim,
            vocab_size
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights."""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        nn.init.constant_(self.embedding.weight[0], 0)  # Padding token
        
        # Linear layer initialization
        for layer in [self.input_projection, self.output_projection]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        
        # GRU initialization
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize decoder hidden state."""
        return torch.zeros(
            self.num_layers, batch_size, self.decoder_hidden_size,
            device=device, dtype=torch.float
        )
    
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        coverage_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward step of the decoder.
        
        Args:
            input_token: Current input token (B,)
            hidden_state: Current hidden state (num_layers, B, hidden_size)
            encoder_outputs: Encoder outputs (B, seq_len, encoder_hidden_size)
            encoder_mask: Encoder mask (B, seq_len)
            coverage_vector: Coverage vector (B, seq_len)
            
        Returns:
            Tuple of:
            - output_logits: Logits for next token (B, vocab_size)
            - new_hidden: Updated hidden state (num_layers, B, hidden_size)
            - attention_weights: Attention weights (B, seq_len)
            - new_coverage: Updated coverage vector (B, seq_len)
        """
        batch_size = input_token.size(0)
        
        # Embed input token
        embedded = self.embedding(input_token)  # (B, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Get current decoder state (top layer)
        current_hidden = hidden_state[-1]  # (B, decoder_hidden_size)
        
        # Compute attention
        context_vector, attention_weights, new_coverage = self.attention(
            encoder_outputs=encoder_outputs,
            decoder_hidden=current_hidden,
            encoder_mask=encoder_mask,
            coverage_vector=coverage_vector
        )
        
        # Combine embedding and context for GRU input
        gru_input = torch.cat([embedded, context_vector], dim=1)  # (B, embedding_dim + encoder_hidden_size)
        gru_input = self.input_projection(gru_input).unsqueeze(1)  # (B, 1, decoder_hidden_size)
        
        # GRU forward pass
        gru_output, new_hidden = self.gru(gru_input, hidden_state)  # (B, 1, decoder_hidden_size)
        gru_output = gru_output.squeeze(1)  # (B, decoder_hidden_size)
        
        # Combine GRU output, context, and embedding for final prediction
        combined_output = torch.cat([
            gru_output,
            context_vector,
            embedded
        ], dim=1)  # (B, decoder_hidden_size + encoder_hidden_size + embedding_dim)
        
        # Project to vocabulary and apply LogSoftmax as per PRD specifications
        output_logits = self.output_projection(combined_output)  # (B, vocab_size)
        output_log_probs = F.log_softmax(output_logits, dim=-1)  # (B, vocab_size)
        
        return output_log_probs, new_hidden, attention_weights, new_coverage
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        target_sequences: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing for training.
        
        Args:
            encoder_outputs: Encoder outputs (B, seq_len, encoder_hidden_size)
            target_sequences: Target sequences (B, target_len)
            encoder_mask: Encoder mask (B, seq_len)
            teacher_forcing_ratio: Ratio of teacher forcing (1.0 = always use targets)
            
        Returns:
            Dictionary containing:
            - logits: Output logits (B, target_len, vocab_size)
            - attention_weights: Attention weights (B, target_len, seq_len)
            - coverage_vectors: Coverage vectors if enabled (B, target_len, seq_len)
        """
        batch_size, target_len = target_sequences.size()
        device = encoder_outputs.device
        
        # Initialize hidden state
        hidden_state = self.init_hidden(batch_size, device)
        
        # Initialize coverage if enabled
        coverage_vector = None
        coverage_vectors = []
        
        # Storage for outputs
        all_logits = []
        all_attention_weights = []
        
        # Start with SOS token (assuming index 1)
        current_input = torch.ones(batch_size, device=device, dtype=torch.long)
        
        for t in range(target_len):
            # Forward step
            log_probs, hidden_state, attention_weights, coverage_vector = self.forward_step(
                input_token=current_input,
                hidden_state=hidden_state,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                coverage_vector=coverage_vector
            )
            
            # Store outputs
            all_logits.append(log_probs)
            all_attention_weights.append(attention_weights)
            if self.use_coverage:
                coverage_vectors.append(coverage_vector)
            
            # Teacher forcing: decide whether to use target or prediction
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing and t < target_len - 1:
                current_input = target_sequences[:, t]
            else:
                current_input = log_probs.argmax(dim=1)
        
        # Stack outputs
        result = {
            'logits': torch.stack(all_logits, dim=1),  # (B, target_len, vocab_size) - Actually log probabilities
            'attention_weights': torch.stack(all_attention_weights, dim=1)  # (B, target_len, seq_len)
        }
        
        if self.use_coverage:
            result['coverage_vectors'] = torch.stack(coverage_vectors, dim=1)  # (B, target_len, seq_len)
        
        return result
    
    def generate(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        sos_token: int = 1,
        eos_token: int = 2,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using greedy decoding.
        
        Args:
            encoder_outputs: Encoder outputs (B, seq_len, encoder_hidden_size)
            encoder_mask: Encoder mask (B, seq_len)
            max_length: Maximum generation length
            sos_token: Start of sequence token ID
            eos_token: End of sequence token ID
            temperature: Temperature for sampling (1.0 = no change)
            
        Returns:
            Dictionary containing:
            - sequences: Generated sequences (B, gen_len)
            - attention_weights: Attention weights (B, gen_len, seq_len)
            - lengths: Actual sequence lengths (B,)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        max_len = max_length or self.max_length
        
        # Initialize
        hidden_state = self.init_hidden(batch_size, device)
        coverage_vector = None
        
        # Storage
        generated_sequences = []
        all_attention_weights = []
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
        
        # Start with SOS token
        current_input = torch.full((batch_size,), sos_token, device=device, dtype=torch.long)
        
        # Store SOS token as first token of each sequence
        generated_sequences.append(current_input)
        
        for t in range(max_len):
            # Forward step
            log_probs, hidden_state, attention_weights, coverage_vector = self.forward_step(
                input_token=current_input,
                hidden_state=hidden_state,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                coverage_vector=coverage_vector
            )
            
            # Store attention weights
            all_attention_weights.append(attention_weights)
            
            # Apply temperature and get next token
            if temperature != 1.0:
                log_probs = log_probs / temperature
            
            next_tokens = log_probs.argmax(dim=1)
            
            # Store next tokens
            generated_sequences.append(next_tokens)
            
            # Update finished sequences
            finished = finished | (next_tokens == eos_token)
            
            # Break if all sequences are finished
            if finished.all():
                break
            
            # Update input for next step
            current_input = next_tokens
        
        # Convert to tensors
        sequences = torch.stack(generated_sequences, dim=1)  # (B, gen_len)
        attention_weights = torch.stack(all_attention_weights, dim=1)  # (B, gen_len, seq_len)
        
        # Calculate actual lengths (find first EOS token)
        lengths = torch.full((batch_size,), sequences.size(1), device=device, dtype=torch.long)
        for b in range(batch_size):
            eos_positions = (sequences[b] == eos_token).nonzero(as_tuple=False)
            if len(eos_positions) > 0:
                lengths[b] = eos_positions[0].item() + 1
        
        return {
            'sequences': sequences,
            'attention_weights': attention_weights,
            'lengths': lengths
        }
    
    def beam_search(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        max_length: Optional[int] = None,
        sos_token: int = 1,
        eos_token: int = 2,
        length_penalty: float = 0.6
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using beam search.
        
        Args:
            encoder_outputs: Encoder outputs (1, seq_len, encoder_hidden_size) - single sample
            encoder_mask: Encoder mask (1, seq_len)
            beam_size: Number of beams
            max_length: Maximum generation length
            sos_token: Start of sequence token ID
            eos_token: End of sequence token ID
            length_penalty: Length penalty factor
            
        Returns:
            Dictionary containing:
            - sequences: Top beam sequences (beam_size, gen_len)
            - scores: Beam scores (beam_size,)
            - attention_weights: Attention weights for top beam (gen_len, seq_len)
        """
        if encoder_outputs.size(0) != 1:
            raise ValueError("Beam search only supports batch size 1")
        
        device = encoder_outputs.device
        max_len = max_length or self.max_length
        
        # Expand encoder outputs for beam search
        encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)  # (beam_size, seq_len, hidden_size)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.expand(beam_size, -1)  # (beam_size, seq_len)
        
        # Initialize beams
        beam_scores = torch.zeros(beam_size, device=device)
        beam_sequences = torch.full((beam_size, 1), sos_token, device=device, dtype=torch.long)
        beam_hidden_states = [self.init_hidden(beam_size, device)]
        beam_finished = torch.zeros(beam_size, device=device, dtype=torch.bool)
        
        # Coverage vectors if enabled
        beam_coverage = None
        
        for t in range(max_len):
            if beam_finished.all():
                break
            
            # Get current inputs
            current_inputs = beam_sequences[:, -1]  # (beam_size,)
            current_hidden = beam_hidden_states[-1]  # (num_layers, beam_size, hidden_size)
            
            # Forward step for all beams
            log_probs, new_hidden, attention_weights, new_coverage = self.forward_step(
                input_token=current_inputs,
                hidden_state=current_hidden,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                coverage_vector=beam_coverage
            )
            
            # The output is already log probabilities, no need to apply log_softmax again
            # log_probs = F.log_softmax(logits, dim=1)  # (beam_size, vocab_size)
            
            # Add to beam scores
            vocab_size = log_probs.size(1)
            expanded_scores = beam_scores.unsqueeze(1) + log_probs  # (beam_size, vocab_size)
            
            # Mask finished sequences (only allow EOS continuation)
            if t > 0:
                for b in range(beam_size):
                    if beam_finished[b]:
                        expanded_scores[b, :] = -float('inf')
                        expanded_scores[b, eos_token] = beam_scores[b]
            
            # Get top candidates
            flat_scores = expanded_scores.view(-1)  # (beam_size * vocab_size,)
            top_scores, top_indices = flat_scores.topk(beam_size)
            
            # Convert flat indices back to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams
            new_sequences = []
            new_hidden_states = []
            new_finished = []
            
            for i, (score, beam_idx, token_idx) in enumerate(zip(top_scores, beam_indices, token_indices)):
                # Copy sequence from selected beam and append new token
                new_seq = torch.cat([
                    beam_sequences[beam_idx],
                    token_idx.unsqueeze(0)
                ], dim=0)
                new_sequences.append(new_seq)
                
                # Copy hidden state from selected beam
                new_hidden_states.append(new_hidden[:, beam_idx:beam_idx+1, :].clone())
                
                # Check if finished
                is_finished = beam_finished[beam_idx] or (token_idx == eos_token)
                new_finished.append(is_finished)
            
            # Update beam states
            beam_sequences = torch.stack(new_sequences, dim=0)  # (beam_size, seq_len)
            beam_hidden_states.append(torch.cat(new_hidden_states, dim=1))  # (num_layers, beam_size, hidden_size)
            beam_scores = top_scores
            beam_finished = torch.tensor(new_finished, device=device)
            
            # Update coverage
            if self.use_coverage:
                beam_coverage = new_coverage[beam_indices]
        
        # Apply length penalty to final scores
        if length_penalty > 0:
            lengths = beam_sequences.size(1)
            length_penalties = ((5 + lengths) / 6) ** length_penalty
            beam_scores = beam_scores / length_penalties
        
        # Sort beams by score
        sorted_indices = beam_scores.argsort(descending=True)
        
        return {
            'sequences': beam_sequences[sorted_indices],
            'scores': beam_scores[sorted_indices],
            'attention_weights': attention_weights[sorted_indices[0]]  # Top beam attention
        }


class DecoderConfig:
    """Configuration for GRU decoder."""
    
    def __init__(
        self,
        vocab_size: int = 117,
        encoder_hidden_size: int = 512,
        decoder_hidden_size: int = 256,
        embedding_dim: int = 256,
        attention_hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_coverage: bool = False,
        max_length: int = 100
    ):
        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.embedding_dim = embedding_dim
        self.attention_hidden_size = attention_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_coverage = use_coverage
        self.max_length = max_length
    
    def create_decoder(self) -> AttentionGRUDecoder:
        """Create decoder from config."""
        return AttentionGRUDecoder(
            vocab_size=self.vocab_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            embedding_dim=self.embedding_dim,
            attention_hidden_size=self.attention_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_coverage=self.use_coverage,
            max_length=self.max_length
        ) 