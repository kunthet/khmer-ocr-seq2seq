"""
Configuration management for Khmer OCR Seq2Seq project.
"""
import yaml
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Encoder settings
    encoder_input_channels: int = 1
    encoder_conv_channels: List[int] = None
    encoder_gru_hidden_size: int = 256
    encoder_gru_num_layers: int = 2
    encoder_bidirectional: bool = True
    
    # Attention settings
    attention_type: str = "bahdanau"
    attention_hidden_size: int = 512
    
    # Decoder settings
    decoder_embedding_dim: int = 256
    decoder_hidden_size: int = 512
    decoder_vocab_size: int = 117
    decoder_max_length: int = 256
    
    def __post_init__(self):
        if self.encoder_conv_channels is None:
            self.encoder_conv_channels = [64, 128, 256, 256, 512, 512, 512]


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 1e-6
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    teacher_forcing_ratio: float = 1.0
    gradient_clip: float = 5.0


@dataclass
class DataConfig:
    """Data processing configuration."""
    image_height: int = 32
    image_channels: int = 1
    vocab_size: int = 117
    max_sequence_length: int = 256
    augmentation_prob: float = 0.5
    concatenation_prob: float = 0.5


class KhmerVocab:
    """Khmer vocabulary with 117 tokens as specified in PRD."""
    
    def __init__(self):
        # Special tokens (4)
        self.special_tokens = ["SOS", "EOS", "PAD", "UNK"]
        
        # Numbers (20): Khmer (10) + Arabic (10)
        self.khmer_numbers = ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]
        self.arabic_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        # Consonants (33)
        self.consonants = [
            "ក", "ខ", "គ", "ឃ", "ង", "ច", "ឆ", "ជ", "ឈ", "ញ",
            "ដ", "ឋ", "ឌ", "ឍ", "ណ", "ត", "ថ", "ទ", "ធ", "ន",
            "ប", "ផ", "ព", "ភ", "ម", "យ", "រ", "ល", "វ", "ស",
            "ហ", "ឡ", "អ"
        ]
        
        # Independent vowels (12)
        self.independent_vowels = [
            "ឥ", "ឦ", "ឧ", "ឩ", "ឪ", "ឫ", "ឬ", "ឭ", "ឮ", "ឯ", "ឰ", "ឱ"
        ]
        
        # Dependent vowels (23) - Added common combinations
        self.dependent_vowels = [
            "ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ", "ើ", "ឿ",
            "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ", "ាំ", "ះ", "ៈ", "ំ",
            "ុំ", "ាៈ", "េៈ"  # Additional common vowel combinations
        ]
        
        # Subscript diacritic (1)
        self.subscript = ["្"]
        
        # Other diacritics (10) - Added missing diacritics
        self.diacritics = ["់", "៉", "៊", "័", "៏", "៌", "៍", "៎", "៑", "៓"]
        
        # Symbols (14) - Added common punctuation  
        self.symbols = ["(", ")", ",", ".", "៕", "។", "ៗ", "?", " ", ";", ":", "-", "!", "\""]
        
        # Build complete vocabulary
        self.vocab = (
            self.special_tokens + 
            self.khmer_numbers + 
            self.arabic_numbers + 
            self.consonants + 
            self.independent_vowels + 
            self.dependent_vowels + 
            self.subscript + 
            self.diacritics + 
            self.symbols
        )
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # Special token indices
        self.SOS_IDX = self.char_to_idx["SOS"]
        self.EOS_IDX = self.char_to_idx["EOS"] 
        self.PAD_IDX = self.char_to_idx["PAD"]
        self.UNK_IDX = self.char_to_idx["UNK"]
        
    def __len__(self):
        return len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to sequence of indices."""
        return [self.char_to_idx.get(char, self.UNK_IDX) for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Decode sequence of indices to text."""
        return "".join([self.idx_to_char.get(idx, "UNK") for idx in indices])


class ConfigManager:
    """Central configuration manager for the project."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "configs/train_config.yaml"
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        self.vocab = KhmerVocab()
        
        if os.path.exists(self.config_path):
            self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Update model config
        if 'model' in config_dict:
            model_dict = config_dict['model']
            if 'encoder' in model_dict:
                encoder = model_dict['encoder']
                self.model_config.encoder_input_channels = encoder.get('input_channels', 1)
                self.model_config.encoder_conv_channels = encoder.get('conv_channels', [64, 128, 256, 256, 512, 512, 512])
                self.model_config.encoder_gru_hidden_size = encoder.get('gru_hidden_size', 256)
                self.model_config.encoder_gru_num_layers = encoder.get('gru_num_layers', 2)
                self.model_config.encoder_bidirectional = encoder.get('bidirectional', True)
            
            if 'attention' in model_dict:
                attention = model_dict['attention']
                self.model_config.attention_type = attention.get('type', 'bahdanau')
                self.model_config.attention_hidden_size = attention.get('hidden_size', 512)
            
            if 'decoder' in model_dict:
                decoder = model_dict['decoder']
                self.model_config.decoder_embedding_dim = decoder.get('embedding_dim', 256)
                self.model_config.decoder_hidden_size = decoder.get('hidden_size', 512)
                self.model_config.decoder_vocab_size = decoder.get('vocab_size', 117)
                self.model_config.decoder_max_length = decoder.get('max_length', 256)
        
        # Update training config
        if 'training' in config_dict:
            training = config_dict['training']
            self.training_config.epochs = training.get('epochs', 150)
            self.training_config.batch_size = training.get('batch_size', 64)
            lr = training.get('learning_rate', 1e-6)
            self.training_config.learning_rate = float(lr) if isinstance(lr, str) else lr
            self.training_config.optimizer = training.get('optimizer', 'adam')
            self.training_config.loss_function = training.get('loss_function', 'cross_entropy')
            self.training_config.teacher_forcing_ratio = training.get('teacher_forcing_ratio', 1.0)
            self.training_config.gradient_clip = training.get('gradient_clip', 5.0)
        
        # Update data config
        if 'data' in config_dict:
            data = config_dict['data']
            self.data_config.image_height = data.get('image_height', 32)
            self.data_config.vocab_size = data.get('vocab_size', 117)
            if 'augmentation' in data:
                aug = data['augmentation']
                self.data_config.augmentation_prob = aug.get('probability', 0.5)
                if 'concatenation' in aug:
                    self.data_config.concatenation_prob = aug['concatenation'].get('probability', 0.5)
    
    def save_config(self, path: str = None):
        """Save current configuration to YAML file."""
        save_path = path or self.config_path
        
        config_dict = {
            'model': {
                'encoder': {
                    'input_channels': self.model_config.encoder_input_channels,
                    'conv_channels': self.model_config.encoder_conv_channels,
                    'gru_hidden_size': self.model_config.encoder_gru_hidden_size,
                    'gru_num_layers': self.model_config.encoder_gru_num_layers,
                    'bidirectional': self.model_config.encoder_bidirectional
                },
                'attention': {
                    'type': self.model_config.attention_type,
                    'hidden_size': self.model_config.attention_hidden_size
                },
                'decoder': {
                    'embedding_dim': self.model_config.decoder_embedding_dim,
                    'hidden_size': self.model_config.decoder_hidden_size,
                    'vocab_size': self.model_config.decoder_vocab_size,
                    'max_length': self.model_config.decoder_max_length
                }
            },
            'training': {
                'epochs': self.training_config.epochs,
                'batch_size': self.training_config.batch_size,
                'learning_rate': self.training_config.learning_rate,
                'optimizer': self.training_config.optimizer,
                'loss_function': self.training_config.loss_function,
                'teacher_forcing_ratio': self.training_config.teacher_forcing_ratio,
                'gradient_clip': self.training_config.gradient_clip
            },
            'data': {
                'image_height': self.data_config.image_height,
                'vocab_size': self.data_config.vocab_size,
                'augmentation': {
                    'probability': self.data_config.augmentation_prob,
                    'concatenation': {
                        'probability': self.data_config.concatenation_prob
                    }
                }
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_device(self):
        """Get the appropriate device for training."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        except ImportError:
            return 'cpu'  # Fallback if torch not available 