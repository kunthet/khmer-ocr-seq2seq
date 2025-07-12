# API Context

## Classes
- `KhmerCorpusProcessor` - Enhanced to support multiple training files
- `OnTheFlyDataset` - Enhanced to support multiple training files  
- `KhmerCorpusDataset` - Enhanced to support multiple training files
- `UserService`
- `OrderController`

## Methods
- `OnTheFlyDataset._load_text_lines() -> List[str]` - Enhanced to detect and load multiple training files (train_*.txt)
- `KhmerCorpusDataset._load_text_lines() -> List[str]` - Enhanced to handle multiple training files with fallback
- `KhmerCorpusProcessor.load_processed_data(data_dir: str) -> Dict[str, List[str]]` - Enhanced to load multiple training files
- `UserService.create_user(user_name:str, ...) -> bool`
- `OrderController.get_order_status(order_id:str) -> Order`

## Recent Enhancements
- **Multiple Training Files Support**: All data loading components now support loading from multiple training files (train_0.txt, train_1.txt, etc.) while maintaining backward compatibility with single train.txt files
- **Automatic Detection**: System automatically detects available training file patterns and loads accordingly
- **Aggregated Statistics**: Proper aggregation of line counts and statistics across multiple training files 