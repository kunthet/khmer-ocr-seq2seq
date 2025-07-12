# API Context

## Classes
- `CheckpointManager` - Enhanced with Google Drive backup functionality
- `KhmerCorpusProcessor` - Enhanced to support multiple training files
- `OnTheFlyDataset` - Enhanced to support multiple training files  
- `KhmerCorpusDataset` - Enhanced to support multiple training files
- `Trainer` - Enhanced to use Google Drive backup
- `UserService`
- `OrderController`

## Methods
- `CheckpointManager.save_checkpoint(checkpoint_data, epoch, is_best, filename, backup_to_gdrive) -> str` - Enhanced with Google Drive backup
- `CheckpointManager.load_checkpoint(checkpoint_path, fallback_to_gdrive) -> Dict[str, Any]` - Enhanced with Google Drive fallback
- `CheckpointManager.sync_from_gdrive() -> bool` - New method to sync checkpoints from Google Drive
- `CheckpointManager.get_gdrive_info() -> Dict[str, Any]` - New method to get Google Drive backup status
- `CheckpointManager.list_gdrive_checkpoints() -> List[Dict[str, Any]]` - New method to list Google Drive checkpoints
- `OnTheFlyDataset._load_text_lines() -> List[str]` - Enhanced to detect and load multiple training files (train_*.txt)
- `KhmerCorpusDataset._load_text_lines() -> List[str]` - Enhanced to handle multiple training files with fallback
- `KhmerCorpusProcessor.load_processed_data(data_dir: str) -> Dict[str, List[str]]` - Enhanced to load multiple training files
- `UserService.create_user(user_name:str, ...) -> bool`
- `OrderController.get_order_status(order_id:str) -> Order`

## Google Drive Backup Features
- **Automatic Backup**: Best models and checkpoints (every 10 epochs) automatically backed up to Google Drive
- **Smart Fallback**: Automatically loads from Google Drive if local checkpoints are missing
- **Training History**: Complete training history and metrics backed up for session recovery
- **Colab Integration**: Seamless Google Colab environment setup with `setup_colab_gdrive.py`

## Recent Enhancements
- **Google Drive Backup**: Complete checkpoint backup system with automatic recovery
- **Multiple Training Files Support**: All data loading systems now handle split training files (train_0.txt, train_1.txt, etc.) 