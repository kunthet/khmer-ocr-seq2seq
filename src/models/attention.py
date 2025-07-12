"""
Bahdanau Attention mechanism for Khmer OCR Seq2Seq model.
Implements additive attention as described in Bahdanau et al. (2014).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    
    This attention mechanism computes a context vector as a weighted sum of encoder outputs,
    where the weights are determined by the compatibility between the current decoder state
    and each encoder output.
    
    Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
    by Bahdanau et al. (2014)
    """
    
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_hidden_size: int = 256,
        coverage: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize Bahdanau attention.
        
        Args:
            encoder_hidden_size: Hidden size of encoder outputs
            decoder_hidden_size: Hidden size of decoder state
            attention_hidden_size: Hidden size of attention computation
            coverage: Whether to use coverage mechanism
            dropout: Dropout rate
        """
        super(BahdanauAttention, self).__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.coverage = coverage
        
        # Linear transformations for attention computation
        self.encoder_projection = nn.Linear(
            encoder_hidden_size, attention_hidden_size, bias=False
        )
        self.decoder_projection = nn.Linear(
            decoder_hidden_size, attention_hidden_size, bias=False
        )
        
        # Coverage mechanism (optional)
        if coverage:
            self.coverage_projection = nn.Linear(
                1, attention_hidden_size, bias=False
            )
        
        # Attention scoring
        self.attention_scorer = nn.Linear(attention_hidden_size, 1, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize attention weights."""
        # Xavier initialization for linear layers
        for layer in [self.encoder_projection, self.decoder_projection, self.attention_scorer]:
            nn.init.xavier_uniform_(layer.weight)
        
        if self.coverage:
            nn.init.xavier_uniform_(self.coverage_projection.weight)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        coverage_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute attention context vector.
        
        Args:
            encoder_outputs: Encoder outputs (B, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (B, decoder_hidden_size)
            encoder_mask: Mask for encoder outputs (B, seq_len)
            coverage_vector: Coverage vector (B, seq_len) for coverage mechanism
            
        Returns:
            Tuple of:
            - context_vector: Weighted sum of encoder outputs (B, encoder_hidden_size)
            - attention_weights: Attention weights (B, seq_len)
            - updated_coverage: Updated coverage vector (B, seq_len) if coverage enabled
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Project encoder outputs: (B, seq_len, attention_hidden_size)
        encoder_features = self.encoder_projection(encoder_outputs)
        
        # Project decoder hidden state: (B, attention_hidden_size)
        decoder_features = self.decoder_projection(decoder_hidden)
        
        # Expand decoder features to match sequence length: (B, seq_len, attention_hidden_size)
        decoder_features = decoder_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine encoder and decoder features
        combined_features = encoder_features + decoder_features
        
        # Add coverage if enabled
        if self.coverage and coverage_vector is not None:
            # coverage_vector: (B, seq_len) -> (B, seq_len, 1)
            coverage_features = self.coverage_projection(coverage_vector.unsqueeze(-1))
            combined_features = combined_features + coverage_features
        
        # Apply tanh activation and compute attention scores
        attention_scores = self.attention_scorer(
            torch.tanh(combined_features)
        ).squeeze(-1)  # (B, seq_len)
        
        # Apply mask if provided (set masked positions to large negative value)
        if encoder_mask is not None:
            # encoder_mask should be 1 for valid positions, 0 for padding
            attention_scores = attention_scores.masked_fill(
                encoder_mask == 0, -float('inf')
            )
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, seq_len)
        
        # Note: We don't apply dropout to attention weights as it would break the sum=1 property
        
        # Compute context vector as weighted sum of encoder outputs
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, seq_len)
            encoder_outputs  # (B, seq_len, encoder_hidden_size)
        ).squeeze(1)  # (B, encoder_hidden_size)
        
        # Update coverage vector if enabled
        updated_coverage = None
        if self.coverage:
            if coverage_vector is None:
                updated_coverage = attention_weights
            else:
                updated_coverage = coverage_vector + attention_weights
        
        return context_vector, attention_weights, updated_coverage
    
    def get_attention_scores(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get raw attention scores without softmax (useful for analysis).
        
        Args:
            encoder_outputs: Encoder outputs (B, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (B, decoder_hidden_size)
            encoder_mask: Mask for encoder outputs (B, seq_len)
            
        Returns:
            attention_scores: Raw attention scores (B, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Project features
        encoder_features = self.encoder_projection(encoder_outputs)
        decoder_features = self.decoder_projection(decoder_hidden)
        decoder_features = decoder_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute scores
        combined_features = encoder_features + decoder_features
        attention_scores = self.attention_scorer(
            torch.tanh(combined_features)
        ).squeeze(-1)
        
        # Apply mask
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(
                encoder_mask == 0, -float('inf')
            )
        
        return attention_scores


class MultiHeadBahdanauAttention(nn.Module):
    """
    Multi-head version of Bahdanau attention for improved representation learning.
    """
    
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_hidden_size: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head Bahdanau attention.
        
        Args:
            encoder_hidden_size: Hidden size of encoder outputs
            decoder_hidden_size: Hidden size of decoder state
            attention_hidden_size: Hidden size of attention computation
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadBahdanauAttention, self).__init__()
        
        assert attention_hidden_size % num_heads == 0, \
            "attention_hidden_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = attention_hidden_size // num_heads
        self.attention_hidden_size = attention_hidden_size
        
        # Multi-head attention modules
        self.attention_heads = nn.ModuleList([
            BahdanauAttention(
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                attention_hidden_size=self.head_dim,
                dropout=dropout
            )
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(
            encoder_hidden_size * num_heads, encoder_hidden_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass.
        
        Args:
            encoder_outputs: Encoder outputs (B, seq_len, encoder_hidden_size)
            decoder_hidden: Current decoder hidden state (B, decoder_hidden_size)
            encoder_mask: Mask for encoder outputs (B, seq_len)
            
        Returns:
            Tuple of:
            - context_vector: Combined context from all heads (B, encoder_hidden_size)
            - attention_weights: Average attention weights (B, seq_len)
        """
        head_contexts = []
        head_weights = []
        
        # Compute attention for each head
        for head in self.attention_heads:
            context, weights, _ = head(encoder_outputs, decoder_hidden, encoder_mask)
            head_contexts.append(context)
            head_weights.append(weights)
        
        # Concatenate contexts from all heads
        combined_context = torch.cat(head_contexts, dim=-1)  # (B, encoder_hidden_size * num_heads)
        
        # Project to original size
        context_vector = self.output_projection(combined_context)  # (B, encoder_hidden_size)
        context_vector = self.dropout(context_vector)
        
        # Average attention weights across heads
        attention_weights = torch.stack(head_weights, dim=0).mean(dim=0)  # (B, seq_len)
        
        return context_vector, attention_weights


class AttentionConfig:
    """Configuration for attention mechanisms."""
    
    def __init__(
        self,
        encoder_hidden_size: int = 512,
        decoder_hidden_size: int = 256,
        attention_hidden_size: int = 256,
        attention_type: str = "bahdanau",  # "bahdanau" or "multi_head"
        num_heads: int = 8,
        coverage: bool = False,
        dropout: float = 0.1
    ):
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.coverage = coverage
        self.dropout = dropout
    
    def create_attention(self) -> nn.Module:
        """Create attention mechanism from config."""
        if self.attention_type == "bahdanau":
            return BahdanauAttention(
                encoder_hidden_size=self.encoder_hidden_size,
                decoder_hidden_size=self.decoder_hidden_size,
                attention_hidden_size=self.attention_hidden_size,
                coverage=self.coverage,
                dropout=self.dropout
            )
        elif self.attention_type == "multi_head":
            return MultiHeadBahdanauAttention(
                encoder_hidden_size=self.encoder_hidden_size,
                decoder_hidden_size=self.decoder_hidden_size,
                attention_hidden_size=self.attention_hidden_size,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}") 