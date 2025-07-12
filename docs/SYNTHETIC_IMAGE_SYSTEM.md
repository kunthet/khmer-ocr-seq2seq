# Synthetic Image Generation System Documentation

## Overview

The Khmer OCR Seq2Seq project uses a pre-generation approach for synthetic images, where all training images are generated upfront and organized into train/val/test splits in the `data/synthetic/` folder. This approach provides:

- **Consistent Training Data**: All images are pre-generated with consistent quality
- **Faster Training**: No image generation overhead during training
- **Reproducible Results**: Same images used across training runs
- **Easy Management**: Simple directory structure with organized splits

## Architecture

### Directory Structure

```
data/synthetic/
├── generation_metadata.json    # Overall generation metadata
├── train/                      # Training split
│   ├── images/                 # Training images
│   │   ├── train_000000.png
│   │   ├── train_000001.png
│   │   └── ...
│   ├── labels/                 # Training labels
│   │   ├── train_000000.txt
│   │   ├── train_000001.txt
│   │   └── ...
│   └── metadata.json          # Training split metadata
├── val/                        # Validation split
│   ├── images/
│   ├── labels/
│   └── metadata.json
└── test/                       # Test split
    ├── images/
    ├── labels/
    └── metadata.json
```

### File Naming Convention

- **Images**: `{split}_{index:06d}.png` (e.g., `train_000001.png`)
- **Labels**: `{split}_{index:06d}.txt` (e.g., `train_000001.txt`)
- **Consistent indexing** across image-label pairs

## Usage

### 1. Generate Synthetic Images

Use the `generate_synthetic_images.py` script to create all training data:

```bash
# Generate all splits with full corpus
python generate_synthetic_images.py

# Generate specific splits
python generate_synthetic_images.py --splits train val

# Generate with limited samples (for testing)
python generate_synthetic_images.py --max-lines 1000

# Generate without augmentation
python generate_synthetic_images.py --no-augment

# Show statistics of existing images
python generate_synthetic_images.py --stats
```

### 2. Train with Synthetic Images

Use the updated training script:

```bash
# Train with pre-generated images
python src/training/train_production.py

# Train with limited samples (for testing)
python src/training/train_production.py --max-samples 1000

# Specify custom synthetic directory
python src/training/train_production.py --synthetic-dir data/my_synthetic
```

### 3. Test the System

Verify everything is working:

```bash
# Test the synthetic dataset implementation
python test_synthetic_dataset.py
```

## Image Generation Process

### Font Selection Strategy

- **Training Split**: Random font selection for data augmentation
- **Validation/Test Splits**: Consistent font selection based on text hash for reproducible results

### Augmentation Strategy

- **Training Images**: Augmentation applied during generation (blur, morphology, noise)
- **Validation/Test Images**: No augmentation for consistent evaluation

### Text Processing

1. Load corpus text from `data/processed/` splits
2. Render each text line with selected font
3. Apply augmentation (training only)
4. Save as PNG image with corresponding text label

## Configuration

### Generation Parameters

```python
class SyntheticImageGenerator:
    def __init__(
        self,
        corpus_dir: str = "data/processed",      # Source corpus text
        output_dir: str = "data/synthetic",      # Output directory
        config_manager: ConfigManager = None,    # Configuration
        fonts: List[str] = None,                 # Font paths
        augment_train: bool = True,              # Enable training augmentation
        random_seed: int = 42                    # Reproducibility seed
    )
```

### Dataset Parameters

```python
class SyntheticImageDataset:
    def __init__(
        self,
        split: str = "train",                    # Dataset split
        synthetic_dir: str = "data/synthetic",   # Synthetic images directory
        config_manager: ConfigManager = None,    # Configuration
        max_samples: int = None,                 # Limit samples (testing)
        transform = None                         # Optional transforms
    )
```

## Metadata Format

### Generation Metadata

Each split includes comprehensive metadata:

```json
{
  "split": "train",
  "total_samples": 192005,
  "augmentation_applied": true,
  "fonts_used": ["KhmerOS", "KhmerOSsiemreap", "..."],
  "image_height": 32,
  "samples": [
    {
      "filename": "train_000000",
      "text": "ភាសាខ្មែរ",
      "font": "KhmerOS",
      "image_path": "data/synthetic/train/images/train_000000.png",
      "label_path": "data/synthetic/train/labels/train_000000.txt",
      "image_width": 128,
      "image_height": 32
    }
  ]
}
```

### Overall Metadata

```json
{
  "generation_config": {
    "corpus_dir": "data/processed",
    "output_dir": "data/synthetic",
    "image_height": 32,
    "augment_train": true,
    "fonts": ["KhmerOS.ttf", "KhmerOSsiemreap.ttf", "..."]
  },
  "splits": {
    "train": { /* split metadata */ },
    "val": { /* split metadata */ },
    "test": { /* split metadata */ }
  },
  "total_images": 288015
}
```

## Performance Benefits

### Generation Time

- **One-time cost**: Images generated once, used many times
- **Parallel processing**: Can be optimized with multiprocessing
- **Progress tracking**: Real-time progress bars with tqdm

### Training Speed

- **No rendering overhead**: Images loaded directly from disk
- **Consistent I/O**: Predictable loading times
- **Memory efficient**: Images loaded on-demand

### Storage Efficiency

- **PNG compression**: Efficient storage for grayscale images
- **Organized structure**: Easy to manage and backup
- **Metadata tracking**: Complete provenance information

## Commands Reference

### Image Generation

```bash
# Basic generation
python generate_synthetic_images.py

# Advanced options
python generate_synthetic_images.py \
  --corpus-dir data/processed \
  --output-dir data/synthetic \
  --max-lines 10000 \
  --splits train val test \
  --no-augment \
  --log-level INFO

# Statistics
python generate_synthetic_images.py --stats
```

### Training

```bash
# Basic training
python src/training/train_production.py

# Advanced options
python src/training/train_production.py \
  --synthetic-dir data/synthetic \
  --batch-size 32 \
  --num-epochs 150 \
  --max-samples 10000 \
  --resume models/checkpoints/checkpoint_epoch_050.pth \
  --log-dir logs/production \
  --checkpoint-dir models/checkpoints
```

### Testing

```bash
# Test dataset functionality
python test_synthetic_dataset.py

# Test with specific directory
PYTHONPATH=. python -c "
from src.data.synthetic_dataset import SyntheticImageDataset
from src.utils.config import ConfigManager
config = ConfigManager()
dataset = SyntheticImageDataset('train', 'data/synthetic', config, 10)
print(f'Dataset loaded: {len(dataset)} samples')
"
```

## Workflow

### Development Workflow

1. **Setup**: Ensure corpus is processed (`python process_corpus.py`)
2. **Generate**: Create synthetic images (`python generate_synthetic_images.py --max-lines 100`)
3. **Test**: Verify dataset (`python test_synthetic_dataset.py`)
4. **Train**: Start training (`python src/training/train_production.py --max-samples 100`)

### Production Workflow

1. **Full Generation**: `python generate_synthetic_images.py`
2. **Validation**: `python test_synthetic_dataset.py`
3. **Production Training**: `python src/training/train_production.py`

### Experimentation Workflow

1. **Generate Subset**: `python generate_synthetic_images.py --max-lines 1000`
2. **Quick Training**: `python src/training/train_production.py --max-samples 1000 --num-epochs 5`
3. **Iterate**: Modify parameters and repeat

## Troubleshooting

### Common Issues

1. **Missing Corpus Data**
   ```
   Error: Split file not found: data/processed/train.txt
   Solution: Run python process_corpus.py first
   ```

2. **Missing Fonts**
   ```
   Warning: No Khmer fonts found in fonts/ directory
   Solution: Add Khmer font files to fonts/ directory
   ```

3. **Insufficient Disk Space**
   ```
   Error: No space left on device
   Solution: Free up disk space or use different output directory
   ```

4. **Memory Issues During Generation**
   ```
   Solution: Process splits separately or use --max-lines to limit
   ```

### Validation Commands

```bash
# Check directory structure
ls -la data/synthetic/*/

# Verify image count matches labels
find data/synthetic/train/images -name "*.png" | wc -l
find data/synthetic/train/labels -name "*.txt" | wc -l

# Check metadata integrity
python -c "
import json
with open('data/synthetic/train/metadata.json') as f:
    meta = json.load(f)
    print(f'Metadata samples: {len(meta[\"samples\"])}')
"
```

## Integration with Training Pipeline

### Dataset Loading

The `SyntheticImageDataset` class provides:

- **Automatic metadata loading**: Reads generation metadata for efficient loading
- **Fallback scanning**: Scans directories if metadata is missing
- **Error handling**: Graceful handling of missing or corrupted files
- **Statistics**: Built-in dataset statistics and sample inspection

### Data Loaders

The `create_synthetic_dataloaders` function creates:

- **Optimized batching**: Variable-width image padding
- **Consistent collation**: Proper tensor formatting for training
- **Configurable workers**: Adjustable parallel loading
- **Memory management**: Efficient memory usage with pin_memory

### Training Integration

The training pipeline automatically:

- **Validates structure**: Checks for required splits and directories
- **Loads efficiently**: Uses optimized data loaders
- **Handles errors**: Graceful error handling and reporting
- **Tracks progress**: Comprehensive logging and monitoring

---

## Summary

The synthetic image generation system provides a robust, efficient, and reproducible approach to training data preparation for the Khmer OCR Seq2Seq model. Key advantages:

- ✅ **Pre-generation**: One-time image creation for multiple training runs
- ✅ **Organized structure**: Clean directory organization with metadata
- ✅ **Reproducible**: Consistent results across different training sessions
- ✅ **Efficient**: Fast training with no image generation overhead
- ✅ **Scalable**: Easy to generate full corpus or subsets for testing
- ✅ **Maintainable**: Clear separation between data generation and training

This system is designed for production use while maintaining flexibility for development and experimentation. 