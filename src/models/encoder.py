"""
CRNN Encoder for Khmer OCR Seq2Seq model.
Implements the 13-layer architecture as specified in the PRD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CRNNEncoder(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) encoder.
    
    Architecture (13 layers total):
    - 7 Convolutional layers with channels: [64, 128, 256, 256, 512, 512, 512]
    - All conv layers use 3x3 kernels
    - 4 MaxPool layers with sizes: [2,2], [2,2], [2,1], [2,1]
    - 2 Bidirectional GRU layers with 256 hidden units each
    
    Input: (B, 1, 32, W) - Batch of grayscale images
    Output: (B, W//4, 512) - Sequence of feature vectors
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: list = None,
        kernel_size: int = 3,
        pool_sizes: list = None,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize CRNN encoder.
        
        Args:
            input_channels: Input image channels (1 for grayscale)
            conv_channels: List of channel sizes for conv layers
            kernel_size: Kernel size for all conv layers
            pool_sizes: List of pooling sizes [(H, W), ...]
            gru_hidden_size: Hidden size for GRU layers
            gru_num_layers: Number of GRU layers
            bidirectional: Whether to use bidirectional GRU
            dropout: Dropout rate
        """
        super(CRNNEncoder, self).__init__()
        
        # Default architecture from PRD
        if conv_channels is None:
            conv_channels = [64, 128, 256, 256, 512, 512, 512]
        
        if pool_sizes is None:
            pool_sizes = [[2, 2], [2, 2], [2, 1], [2, 1]]
        
        self.conv_channels = conv_channels
        self.pool_sizes = pool_sizes
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.bidirectional = bidirectional
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        in_channels = input_channels
        pool_idx = 0
        
        for i, out_channels in enumerate(conv_channels):
            # Convolutional layer
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding
                bias=False  # Using batch norm
            )
            self.conv_layers.append(conv)
            
            # Batch normalization
            bn = nn.BatchNorm2d(out_channels)
            self.batch_norms.append(bn)
            
            # Add pooling after specific conv layers
            # Apply only first two pools to avoid over-reduction
            if i in [0, 1] and pool_idx < 2:
                pool = nn.MaxPool2d(
                    kernel_size=pool_sizes[pool_idx],
                    stride=pool_sizes[pool_idx]
                )
                self.pool_layers.append(pool)
                pool_idx += 1
            else:
                self.pool_layers.append(None)
            
            in_channels = out_channels
        
        # Calculate the number of features after conv layers
        # After pooling: height goes from 32 -> 16 -> 8 = 8
        # Width is reduced by factor of 4: W -> W//4
        self.feature_height = 8
        self.conv_output_size = conv_channels[-1] * self.feature_height
        
        # RNN layers
        self.rnn = nn.GRU(
            input_size=self.conv_output_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        rnn_output_size = gru_hidden_size * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(rnn_output_size, gru_hidden_size * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CRNN encoder.
        
        Args:
            x: Input tensor of shape (B, 1, 32, W)
            
        Returns:
            Tuple of:
            - encoded_features: (B, W//4, hidden_size*2) - Sequence features
            - final_hidden: (num_layers*directions, B, hidden_size) - Final RNN hidden state
        """
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        for i, (conv, bn, pool) in enumerate(zip(self.conv_layers, self.batch_norms, self.pool_layers)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x, inplace=True)
            
            if pool is not None:
                x = pool(x)
            
            # Apply dropout after some layers
            if i in [2, 4, 6]:  # Add dropout at certain depths
                x = self.dropout(x)
        
        # Reshape for RNN: (B, C, H, W) -> (B, W, C*H)
        batch_size, channels, height, width = x.size()
        
        # Verify expected dimensions
        assert height == self.feature_height, f"Expected height {self.feature_height}, got {height}"
        
        # Reshape to sequence format
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(batch_size, width, channels * height)  # (B, W, C*H)
        
        # RNN processing
        rnn_output, final_hidden = self.rnn(x)  # rnn_output: (B, W, hidden_size*directions)
        
        # Apply output projection
        encoded_features = self.output_projection(rnn_output)  # (B, W, hidden_size*2)
        
        return encoded_features, final_hidden
    
    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.gru_hidden_size * 2
    
    def get_conv_output_shape(self, input_width: int) -> Tuple[int, int]:
        """
        Calculate output shape after conv layers.
        
        Args:
            input_width: Width of input image
            
        Returns:
            Tuple of (output_height, output_width)
        """
        height, width = 32, input_width
        
        # Apply only the first 2 pooling operations (as implemented)
        for pool_size in self.pool_sizes[:2]:
            height = height // pool_size[0]
            width = width // pool_size[1]
        
        return height, width


class EncoderConfig:
    """Configuration for CRNN Encoder."""
    
    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: list = None,
        kernel_size: int = 3,
        pool_sizes: list = None,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        self.input_channels = input_channels
        self.conv_channels = conv_channels or [64, 128, 256, 256, 512, 512, 512]
        self.kernel_size = kernel_size
        self.pool_sizes = pool_sizes or [[2, 2], [2, 2], [2, 1], [2, 1]]
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
    
    def create_encoder(self) -> CRNNEncoder:
        """Create encoder from config."""
        return CRNNEncoder(
            input_channels=self.input_channels,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            pool_sizes=self.pool_sizes,
            gru_hidden_size=self.gru_hidden_size,
            gru_num_layers=self.gru_num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        ) 