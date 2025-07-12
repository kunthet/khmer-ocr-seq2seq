# Khmer OCR Integrated Synthetic Generator

## Overview

The Khmer OCR Integrated Synthetic Generator combines the best features from the imported `src/synthetic_data_generator/` modules with the existing Khmer OCR project architecture. It generates **variable-width images** with **32px height** from **corpus data** in `data/processed/` folder and outputs to the `data/synthetic/` structure for training.

## Key Features

### ✅ **Corpus-Based Text Generation**
- Uses real Khmer text from `data/processed/train.txt`, `val.txt`, `test.txt`
- Authentic linguistic patterns and structures
- Better training data quality compared to synthetic text

### ✅ **Variable-Width Images**
- Images automatically sized to fit text content
- 32px height as defined in project configuration
- Optimal width calculated per text sample
- Maintains aspect ratio for better OCR performance

### ✅ **Advanced Text Rendering**
- Proper Khmer text normalization using `khnormal()`
- Fixes rendering issues with complex Khmer characters
- Multiple font support with intelligent selection
- Correct character positioning and combinations

### ✅ **Enhanced Backgrounds and Augmentation**
- Advanced background generation (gradients, textures, patterns)
- Training data augmentation (rotation, scaling, noise)
- Realistic document-like backgrounds
- Optional simple backgrounds for testing

### ✅ **Project Integration**
- Compatible with existing training pipeline
- Uses project configuration (`configs/config.yaml`)
- Outputs to `data/synthetic/` structure
- Works with existing `SyntheticImageDataset`

## Architecture

### Directory Structure

```
data/synthetic/
├── generation_metadata.json    # Overall generation metadata
├── train/                      # Training split
│   ├── images/                 # Variable-width PNG images
│   │   ├── train_000000.png    # 32px height, auto width
│   │   ├── train_000001.png
│   │   └── ...
│   ├── labels/                 # Text labels
│   │   ├── train_000000.txt
│   │   ├── train_000001.txt
│   │   └── ...
│   └── metadata.json          # Split metadata
├── val/                        # Validation split
│   ├── images/
│   ├── labels/
│   └── metadata.json
└── test/                       # Test split
    ├── images/
    ├── labels/
    └── metadata.json
```

### Integration Components

1. **`KhmerOCRSyntheticGenerator`**: Main generator class
2. **`generate_khmer_synthetic_images.py`**: Generation script
3. **`test_khmer_synthetic_generator.py`**: Test suite
4. **Advanced modules**: Backgrounds, augmentation, text processing

## Usage

### 1. Basic Generation

```bash
# Generate all splits from corpus data
python generate_khmer_synthetic_images.py

# Generate with limited samples for testing
python generate_khmer_synthetic_images.py --max-samples 100

# Generate specific splits
python generate_khmer_synthetic_images.py --splits train val
```

### 2. Advanced Options

```bash
# Full generation with all features
python generate_khmer_synthetic_images.py \
  --corpus-dir data/processed \
  --output-dir data/synthetic \
  --fonts-dir fonts \
  --log-level INFO

# Generation without augmentation
python generate_khmer_synthetic_images.py \
  --no-augment \
  --max-samples 1000

# Generation with simple backgrounds
python generate_khmer_synthetic_images.py \
  --no-advanced-backgrounds \
  --max-samples 1000
```

### 3. Statistics and Monitoring

```bash
# Show statistics of generated images
python generate_khmer_synthetic_images.py --stats

# Test the generator
python test_khmer_synthetic_generator.py
```

## Configuration

### Generator Parameters

```python
generator = KhmerOCRSyntheticGenerator(
    corpus_dir="data/processed",           # Corpus text files
    output_dir="data/synthetic",           # Output directory
    config_manager=ConfigManager(),        # Project configuration
    fonts_dir="fonts",                     # Khmer fonts directory
    augment_train=True,                    # Enable training augmentation
    use_advanced_backgrounds=True,         # Enable advanced backgrounds
    random_seed=42                         # Reproducibility seed
)
```

### Image Properties

- **Height**: 32px (from `configs/config.yaml`)
- **Width**: Variable (calculated per text)
- **Channels**: 1 (grayscale)
- **Format**: PNG
- **Background**: Advanced or simple
- **Text**: Khmer normalized

## Font Selection Strategy

### Training Split
- **Random font selection** for each sample
- Provides data augmentation
- Improves model generalization

### Validation/Test Splits
- **Hash-based consistent selection**
- Same text always gets same font
- Reproducible evaluation results

## Text Processing Pipeline

1. **Load corpus text** from `data/processed/{split}.txt`
2. **Normalize Khmer text** using `khnormal()`
3. **Calculate optimal width** based on text and font
4. **Select appropriate font** based on split strategy
5. **Render text to image** with proper positioning
6. **Apply augmentation** (training only)
7. **Save image and label** files

## Integration with Training Pipeline

### Dataset Loading

```python
from src.data.synthetic_dataset import SyntheticImageDataset

# Load generated synthetic images
dataset = SyntheticImageDataset(
    split="train",
    synthetic_dir="data/synthetic",
    config_manager=ConfigManager(),
    max_samples=None  # Use all samples
)
```

### Training

```python
# Use existing training script
python src/training/train_production.py \
  --synthetic-dir data/synthetic \
  --batch-size 32 \
  --num-epochs 150
```

## Advanced Features

### Background Generation

The generator can create various background types:
- **Solid colors**: White, light gray, paper-like
- **Gradients**: Horizontal, vertical, diagonal
- **Textures**: Paper, fabric, noise patterns
- **Patterns**: Subtle dots, lines, grids

### Augmentation Pipeline

For training images:
- **Rotation**: ±2 degrees
- **Scaling**: 0.9-1.1x
- **Noise**: Gaussian noise
- **Brightness**: 0.8-1.2x
- **Contrast**: 0.8-1.2x

### Text Normalization

Uses `khnormal()` to fix:
- Dotted circles (្◌)
- Character positioning
- Improper combinations
- Rendering artifacts

## Performance Optimization

### Image Generation
- **Optimal width calculation** prevents unnecessary padding
- **Font caching** reduces loading overhead
- **Batch processing** with progress tracking
- **Memory efficient** image handling

### Storage Efficiency
- **PNG compression** for grayscale images
- **Metadata caching** for fast dataset loading
- **Organized structure** for easy management

## Testing and Validation

### Test Suite

```bash
# Run comprehensive tests
python test_khmer_synthetic_generator.py
```

Tests include:
- Basic image generation
- Dataset integration
- Advanced features
- Statistics calculation
- Image dimension validation

### Validation Workflow

1. **Test with small samples** first
2. **Verify image dimensions** (32px height)
3. **Check dataset compatibility**
4. **Validate text rendering** quality
5. **Run full generation**

## Production Workflow

### 1. Development Testing

```bash
# Test with small samples
python generate_khmer_synthetic_images.py --max-samples 100
python test_khmer_synthetic_generator.py
```

### 2. Full Generation

```bash
# Generate all training data
python generate_khmer_synthetic_images.py
```

### 3. Training

```bash
# Start training with generated data
python src/training/train_production.py
```

## Troubleshooting

### Common Issues

1. **Missing corpus files**
   ```
   Error: Corpus file not found: data/processed/train.txt
   Solution: Run corpus processing first
   ```

2. **No fonts found**
   ```
   Error: No working Khmer fonts found in fonts
   Solution: Add Khmer font files to fonts/ directory
   ```

3. **Import errors**
   ```
   Error: Cannot import KhmerOCRSyntheticGenerator
   Solution: Check Python path and dependencies
   ```

### Debug Commands

```bash
# Check corpus files
ls -la data/processed/

# Check fonts
ls -la fonts/

# Test basic functionality
python -c "from src.synthetic_data_generator import KhmerOCRSyntheticGenerator; print('Import successful')"
```

## Comparison with Original System

### Original System
- Fixed-width images
- Uses existing text renderer
- Simple background generation
- Basic augmentation

### Integrated System
- ✅ **Variable-width images** (optimal sizing)
- ✅ **Corpus-based text** (authentic Khmer)
- ✅ **Advanced backgrounds** (realistic textures)
- ✅ **Proper text normalization** (correct rendering)
- ✅ **Enhanced augmentation** (better generalization)
- ✅ **Comprehensive testing** (quality assurance)

## Migration Guide

### From Original System

1. **Keep existing code** (for compatibility)
2. **Test new generator** with small samples
3. **Compare results** with original system
4. **Gradually migrate** to new system
5. **Update documentation** and workflows

### Commands Comparison

```bash
# Original system
python generate_synthetic_images.py

# Integrated system
python generate_khmer_synthetic_images.py
```

Both systems:
- Use same corpus data (`data/processed/`)
- Output to same structure (`data/synthetic/`)
- Compatible with existing training pipeline

## Future Enhancements

### Planned Features
- **Multi-threaded generation** for faster processing
- **GPU acceleration** for background generation
- **Advanced text effects** (shadows, outlines)
- **Curriculum learning** support
- **Quality metrics** and validation

### Extensibility
- **Plugin system** for custom backgrounds
- **Configurable augmentation** pipelines
- **Custom font loading** strategies
- **Export formats** (besides PNG)

## Summary

The Khmer OCR Integrated Synthetic Generator provides a comprehensive solution for generating high-quality synthetic training data. It combines:

- **Authentic corpus text** for realistic training data
- **Variable-width images** for optimal OCR performance
- **Advanced text rendering** with proper Khmer normalization
- **Enhanced backgrounds** and augmentation
- **Seamless integration** with existing training pipeline

The system is ready for production use and provides significant improvements over the original synthetic data generation approach while maintaining full compatibility with the existing codebase.

---

**Ready for Production**: The integrated system is thoroughly tested and ready for full-scale training data generation. 