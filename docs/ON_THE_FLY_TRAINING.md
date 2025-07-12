# On-the-Fly Training System for Khmer OCR

## Overview

The Khmer OCR Seq2Seq project now features an **On-the-Fly Training System** that generates synthetic images dynamically during training instead of pre-generating all images to disk. This approach provides significant benefits in terms of storage requirements, augmentation variety, and system flexibility.

## Key Features

### ✅ **Dynamic Image Generation**
- Images generated in real-time during training batches
- No pre-generation of massive training datasets
- Unlimited augmentation variety per epoch
- Reduces storage requirements by 90%+

### ✅ **Fixed Validation Set**
- Exactly **6,400 validation images** generated once and kept fixed
- Ensures consistent and reproducible evaluation across training runs
- Hash-based font selection for deterministic validation results
- Pre-generated validation images stored in `data/validation_fixed/`

### ✅ **Memory Efficient**
- Low memory footprint during training
- No large pre-generated datasets to load
- Dynamic text selection from corpus
- Efficient batch processing

### ✅ **Reproducible Validation**
- Fixed validation set with seed=42 for reproducibility
- Consistent evaluation metrics across different training sessions
- Validation images generated once and reused

## Architecture

### Training Data Flow
```
Corpus Text Files → OnTheFlyDataset → Dynamic Image Generation → Training Batches
    ↓                      ↓                      ↓                    ↓
data/processed/     Real-time rendering    Augmentation       Model Training
  - train.txt       Font selection         Variety            Unlimited epochs
  - 192K lines      Background generation  No storage
```

### Validation Data Flow
```
Corpus Text Files → Fixed Generation → Pre-saved Images → Validation Batches
    ↓                     ↓                   ↓                   ↓
data/processed/     One-time generation  data/validation_fixed/  Consistent
  - val.txt         6,400 images         - images/               Evaluation
  - 24K lines       Seed=42              - labels/
                                         - metadata.json
```

## Directory Structure

```
data/
├── processed/                    # Corpus text files
│   ├── train.txt                # 192K training lines
│   ├── val.txt                  # 24K validation lines (source)
│   └── test.txt                 # 24K test lines
├── validation_fixed/            # Fixed validation set (6,400 images)
│   ├── images/                  # PNG image files
│   │   ├── val_000000.png
│   │   ├── val_000001.png
│   │   └── ... (6,400 total)
│   ├── labels/                  # Text label files
│   │   ├── val_000000.txt
│   │   ├── val_000001.txt
│   │   └── ... (6,400 total)
│   ├── metadata.json            # Validation set metadata
│   └── validation_texts.txt     # Reference text list
└── synthetic/                   # Legacy pre-generated (optional)
    └── ... (can be removed)
```

## Usage

### 1. Setup Corpus Data
First, ensure you have processed corpus data:
```bash
# If not already done, process corpus
python process_corpus.py
```

### 2. Generate Fixed Validation Set
Generate the fixed validation set once:
```bash
# Generate 6,400 fixed validation images
python generate_fixed_validation_set.py

# Or customize the generation
python generate_fixed_validation_set.py \
  --num-samples 6400 \
  --output-dir data/validation_fixed \
  --random-seed 42
```

### 3. Test the System
Verify everything works correctly:
```bash
# Run comprehensive tests
python test_onthefly_system.py
```

### 4. Start Training
Use the new on-the-fly training script:
```bash
# Basic training with default parameters
python src/training/train_onthefly.py

# Customized training
python src/training/train_onthefly.py \
  --num-epochs 150 \
  --train-samples-per-epoch 10000 \
  --batch-size 32 \
  --validation-dir data/validation_fixed
```

## Configuration

### Training Parameters
- **Epochs**: 150 (target ≤1.0% CER)
- **Batch Size**: Auto-calculated based on GPU memory (32-64 typical)
- **Samples per Epoch**: 10,000 (configurable)
- **Learning Rate**: 1e-6
- **Teacher Forcing**: 1.0

### Validation Parameters
- **Fixed Size**: 6,400 images exactly
- **No Augmentation**: Clean validation for consistent evaluation
- **Reproducible**: Same images every training run
- **Font Selection**: Hash-based for deterministic results

### Augmentation Strategy
- **Training**: Random augmentation per sample (blur, morphology, noise)
- **Validation**: No augmentation for consistent evaluation
- **Font Selection**: Random for training, deterministic for validation

## Benefits

### 🚀 **Performance**
- **90%+ Storage Reduction**: No pre-generated training images
- **Faster Startup**: No large dataset loading
- **Memory Efficient**: Low memory footprint
- **Unlimited Variety**: Different augmentations every epoch

### 🔄 **Flexibility**
- **Dynamic Corpus**: Easy to update training text
- **Configurable Epochs**: Any number of training samples per epoch
- **Real-time Augmentation**: Fresh augmentations every batch
- **Easy Experimentation**: Quick parameter changes

### 📊 **Reproducibility**
- **Fixed Validation**: Consistent evaluation across runs
- **Deterministic Results**: Reproducible validation metrics
- **Version Control Friendly**: No large binary datasets
- **Easy Backup**: Small corpus files instead of massive images

## Comparison: Old vs New System

| Aspect | Pre-Generation System | On-the-Fly System |
|--------|----------------------|-------------------|
| **Training Images** | 192K+ pre-generated PNG files | Generated dynamically per batch |
| **Storage Required** | ~50-100GB for full dataset | ~100MB corpus text files |
| **Augmentation** | Fixed augmentations per image | Unlimited variety per epoch |
| **Startup Time** | 5-10 minutes dataset loading | Instant startup |
| **Memory Usage** | High (loads all images) | Low (generates on demand) |
| **Validation** | Variable size (~24K images) | Fixed 6,400 images |
| **Reproducibility** | Different images each generation | Fixed validation set |
| **Experimentation** | Slow (re-generate dataset) | Fast (change parameters) |

## Advanced Usage

### Custom Validation Set
Generate a custom validation set with different parameters:
```bash
python generate_fixed_validation_set.py \
  --num-samples 3200 \
  --random-seed 123 \
  --output-dir data/validation_custom
```

### Memory Optimization
For limited memory systems:
```bash
python src/training/train_onthefly.py \
  --train-samples-per-epoch 5000 \
  --batch-size 16
```

### High-Performance Training
For powerful GPUs:
```bash
python src/training/train_onthefly.py \
  --train-samples-per-epoch 20000 \
  --batch-size 64
```

## Troubleshooting

### Common Issues

**1. "Fixed validation set not found"**
```bash
# Generate the validation set first
python generate_fixed_validation_set.py
```

**2. "Corpus files not found"**
```bash
# Process corpus data first
python process_corpus.py
```

**3. "Out of GPU memory"**
```bash
# Reduce batch size
python src/training/train_onthefly.py --batch-size 16
```

**4. "Font rendering errors"**
```bash
# Check font directory exists
ls fonts/
# Should contain .ttf font files
```

### Validation Set Issues

**Validate existing validation set:**
```bash
python generate_fixed_validation_set.py --validate-only
```

**Regenerate if corrupted:**
```bash
rm -rf data/validation_fixed
python generate_fixed_validation_set.py
```

## Performance Monitoring

### Training Metrics
- **Loss**: Cross-entropy loss with teacher forcing
- **CER**: Character Error Rate on fixed validation set
- **Speed**: Samples/second generation rate
- **Memory**: GPU memory usage monitoring

### Validation Consistency
- **Fixed Images**: Same 6,400 images every run
- **Reproducible CER**: Consistent evaluation metrics
- **Progress Tracking**: Comparable results across training sessions

## Migration from Pre-Generation

### For Existing Projects
1. **Keep existing validation**: Use pre-generated validation if available
2. **Generate fixed set**: Create new 6,400 image validation set
3. **Update training script**: Switch to `train_onthefly.py`
4. **Remove old data**: Clean up pre-generated training images (optional)

### Cleanup Commands
```bash
# Remove pre-generated training data (save space)
rm -rf data/synthetic/train/
rm -rf data/synthetic/test/

# Keep validation if it's good, or replace with fixed set
# rm -rf data/synthetic/val/
```

## Next Steps

1. **Generate Validation Set**: `python generate_fixed_validation_set.py`
2. **Test System**: `python test_onthefly_system.py`
3. **Start Training**: `python src/training/train_onthefly.py`
4. **Monitor Progress**: Check logs and TensorBoard
5. **Achieve Target**: Train until ≤1.0% CER

---

## Summary

The On-the-Fly Training System provides a modern, efficient, and flexible approach to training the Khmer OCR model with:

- ✅ **90%+ storage reduction** through dynamic generation
- ✅ **Fixed 6,400 validation images** for reproducible evaluation  
- ✅ **Unlimited augmentation variety** for better generalization
- ✅ **Easy experimentation** and parameter tuning
- ✅ **Consistent validation results** across training runs

This system is now the **recommended approach** for training the Khmer OCR Seq2Seq model. 