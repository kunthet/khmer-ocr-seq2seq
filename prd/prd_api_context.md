# API Context

## Classes
- `CheckpointManager` - Enhanced with Google Drive backup functionality
- `KhmerCorpusProcessor` - Enhanced to support multiple training files
- `OnTheFlyDataset` - Enhanced to support multiple training files and integrated with KhmerOCRSyntheticGenerator
- `KhmerCorpusDataset` - Enhanced to support multiple training files
- `KhmerOCRSyntheticGenerator` - Advanced synthetic data generator with variable-width images and Khmer text rendering
- `Trainer` - Enhanced to use Google Drive backup
- `ConfigManager` - Verified and fixed configuration loading with complete 117-token Khmer vocabulary
- `KhmerVocab` - Complete vocabulary class with all 117 tokens properly defined
- `KhmerOCRSeq2Seq` - Model architecture verified with 16.3M parameters and proper CRNN encoder
- `UserService`
- `OrderController`

## Methods
- `CheckpointManager.save_checkpoint(checkpoint_data, epoch, is_best, filename, backup_to_gdrive) -> str` - Enhanced with Google Drive backup
- `CheckpointManager.load_checkpoint(checkpoint_path, fallback_to_gdrive) -> Dict[str, Any]` - Enhanced with Google Drive fallback
- `CheckpointManager.sync_from_gdrive() -> bool` - New method to sync checkpoints from Google Drive
- `CheckpointManager.get_gdrive_info() -> Dict[str, Any]` - New method to get Google Drive backup status
- `CheckpointManager.list_gdrive_checkpoints() -> List[Dict[str, Any]]` - New method to list Google Drive checkpoints
- `OnTheFlyDataset.__init__(split, config_manager, corpus_dir, samples_per_epoch, augment_prob, fonts, shuffle_texts, random_seed)` - Enhanced to use KhmerOCRSyntheticGenerator
- `OnTheFlyDataset._load_text_lines() -> List[str]` - Enhanced to detect and load multiple training files (train_*.txt)
- `OnTheFlyDataset.__getitem__(idx) -> Tuple[torch.Tensor, torch.Tensor, str]` - Enhanced to use KhmerOCRSyntheticGenerator for text rendering
- `KhmerOCRSyntheticGenerator._render_text_image(text: str, font_path: str, image_width: int) -> Image.Image` - Advanced Khmer text rendering with backgrounds
- `KhmerOCRSyntheticGenerator._calculate_optimal_width(text: str, font: ImageFont.FreeTypeFont) -> int` - Dynamic width calculation
- `KhmerOCRSyntheticGenerator._select_font(text: str, split: str) -> str` - Smart font selection for training/validation
- `KhmerOCRSyntheticGenerator._apply_augmentation(image: Image.Image) -> Image.Image` - Integrated augmentation pipeline
- `KhmerCorpusDataset._load_text_lines() -> List[str]` - Enhanced to handle multiple training files with fallback
- `KhmerCorpusProcessor.load_processed_data(data_dir: str) -> Dict[str, List[str]]` - Enhanced to load multiple training files
- `ConfigManager.__init__(config_path: str) -> None` - Verified configuration loading with complete vocabulary
- `ConfigManager.load_config() -> None` - Fixed and verified YAML configuration loading
- `KhmerVocab.__init__() -> None` - Complete 117-token vocabulary initialization
- `KhmerVocab.encode(text: str) -> List[int]` - Text to token indices encoding
- `KhmerVocab.decode(indices: List[int]) -> str` - Token indices to text decoding
- `UserService.create_user(user_name:str, ...) -> bool`
- `OrderController.get_order_status(order_id:str) -> Order`

## Google Drive Backup Features
- **Automatic Backup**: Best models and checkpoints (every 10 epochs) automatically backed up to Google Drive
- **Smart Fallback**: Automatically loads from Google Drive if local checkpoints are missing
- **Training History**: Complete training history and metrics backed up for session recovery
- **Colab Integration**: Seamless Google Colab environment setup with `setup_colab_gdrive.py`

## Recent Enhancements
- **KhmerOCRSyntheticGenerator Integration**: OnTheFlyDataset now uses advanced synthetic data generator for improved Khmer text rendering with variable-width images, advanced backgrounds, and better font handling
- **Google Drive Backup**: Complete checkpoint backup system with automatic recovery
- **Multiple Training Files Support**: All data loading systems now handle split training files (train_0.txt, train_1.txt, etc.)
- **Font Size Consistency**: Standardized font size calculations across all text rendering methods to prevent inconsistencies
- **Configuration System Verification**: Complete verification and fix of all configuration files
  - Fixed truncated YAML files (config.yaml, model_config.yaml, vocab_config.yaml, vocabulary.yaml)
  - Verified complete 117-token Khmer vocabulary with proper special tokens
  - Optimized batch size from 64 to 32 for better convergence and reduced overfitting
  - Validated all training data (105MB corpus), fonts (8 TTF files), and validation set
  - Confirmed model architecture with 16.3M parameters and proper CRNN encoder
  - Tested configuration loading and model creation successfully 