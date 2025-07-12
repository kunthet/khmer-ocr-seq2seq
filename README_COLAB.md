# 🇰🇭 Khmer OCR Training on Google Colab

This guide helps you train the Khmer OCR Seq2Seq model on Google Colab with resumable training and Google Drive checkpoint saving.

## 🚀 Quick Start

### 1. Upload to Google Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload Notebook**: Click `File → Upload notebook` and select `khmer_ocr_colab_training.ipynb`
3. **Set Runtime**: Go to `Runtime → Change runtime type`
   - **Hardware accelerator**: GPU
   - **GPU type**: T4, V100, or A100 (if available)
   - **Runtime shape**: High-RAM (recommended)

### 2. Pre-Training Setup

Before running the notebook, ensure you have:

#### **Required Data**
- **Corpus files** in `data/processed/`:
  - `train.txt` (100K+ lines recommended)
  - `val.txt` (10K+ lines recommended)  
  - `test.txt` (5K+ lines recommended)
- **Khmer fonts** in `fonts/` directory:
  - KhmerOS.ttf
  - KhmerOSBattambang.ttf
  - KhmerOSSiemreap.ttf
  - KhmerOSMuol.ttf
  - etc.

#### **Google Drive Setup**
- Minimum 50GB free space for checkpoints
- The notebook will create `/KhmerOCR_Checkpoints/` directory automatically

### 3. Training Process

The notebook implements:

#### **📊 Data Strategy**
- **Training**: On-the-fly image generation (10K samples/epoch)
- **Validation**: Fixed 6,400 images for reproducible evaluation
- **Storage**: 90% reduction vs pre-generated approach

#### **🎯 Training Configuration**
```python
# Auto-adjusted based on GPU memory
TRAINING_CONFIG = {
    "batch_size": 16-64,      # Based on GPU memory
    "learning_rate": 1e-6,    # As per PRD
    "num_epochs": 150,        # Target epochs
    "samples_per_epoch": 10000,
    "target_cer": 0.01,       # 1% Character Error Rate
}
```

#### **🔄 Resumable Training**
- Automatic checkpoint saving every 5 epochs
- Resume from latest or best checkpoint
- All checkpoints backed up to Google Drive
- Training history and metrics saved

## 📁 Google Drive Structure

After training, your Google Drive will contain:

```
/KhmerOCR_Checkpoints/
├── models/
│   ├── best_model.pth              # Best performing model
│   ├── latest_checkpoint.pth       # Latest checkpoint
│   ├── checkpoint_epoch_XXX.pth    # Epoch checkpoints
│   ├── khmer_ocr_deployment.pth    # Ready for deployment
│   └── training_history.json       # Training metrics
├── logs/
│   └── training_results.png        # Performance plots
└── validation_set/
    ├── images/                     # 6,400 validation images
    ├── labels/                     # Corresponding labels
    └── metadata.json               # Validation set info
```

## 🎯 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **CER** | ≤ 1.0% | Character Error Rate |
| **Word Accuracy** | ≥ 95% | Complete word matches |
| **Training Time** | ~15-20 hours | On V100/A100 GPUs |

## 🔧 GPU Requirements

| GPU | Memory | Batch Size | Training Time |
|-----|---------|------------|---------------|
| **T4** | 16GB | 16-24 | ~25 hours |
| **V100** | 32GB | 32-48 | ~15 hours |
| **A100** | 40GB | 48-64 | ~10 hours |

## 🚀 Running the Training

### Step 1: Setup
```python
# Run cells 1-4 to:
# - Check system specs
# - Mount Google Drive  
# - Clone repository
# - Install dependencies
```

### Step 2: Data Preparation
```python
# Run cells 5-6 to:
# - Generate/load validation set
# - Configure training parameters
```

### Step 3: Training
```python
# Run cell 7 to start training:
# - Automatic batch size adjustment
# - Resumable training with checkpoints
# - Real-time monitoring
```

### Step 4: Results
```python
# Run cells 8-9 to:
# - Visualize training progress
# - Export best model
# - Generate summary report
```

## 🛠️ Advanced Configuration

### Custom Training Parameters
```python
# Modify TRAINING_CONFIG before training
TRAINING_CONFIG.update({
    "batch_size": 32,           # Custom batch size
    "learning_rate": 5e-7,      # Lower learning rate
    "samples_per_epoch": 15000, # More samples per epoch
    "num_epochs": 200,          # Extended training
})
```

### Resume Training
```python
# The notebook automatically detects existing checkpoints
# Choose 'y' when prompted to resume from checkpoint
# Or specify a specific checkpoint to load
```

### Mixed Precision Training
```python
# Automatically enabled for faster training
# Reduces memory usage by ~50%
# Maintains model accuracy
```

## 📊 Monitoring Training

### Real-time Metrics
- Training loss per batch
- Validation CER per epoch
- Word accuracy tracking
- GPU memory usage

### Checkpoints
- Best model (lowest CER)
- Latest checkpoint (for resuming)
- Epoch checkpoints (every 5 epochs)

### Early Stopping
- Automatically stops when target CER ≤ 1% is reached
- Saves final model for deployment

## 🚀 Deployment

### Export Model
```python
# Best model is automatically exported as:
# khmer_ocr_deployment.pth
# Contains:
# - Model weights
# - Configuration  
# - Vocabulary
# - Performance metrics
```

### Load for Inference
```python
import torch

# Load deployment package
deployment = torch.load('khmer_ocr_deployment.pth')

# Model ready for inference
model.load_state_dict(deployment['model_state_dict'])
vocab = deployment['vocab']
config = deployment['config']
```

## 🔍 Troubleshooting

### Common Issues

#### **Out of Memory**
```python
# Reduce batch size
TRAINING_CONFIG["batch_size"] = 8

# Enable gradient checkpointing
TRAINING_CONFIG["gradient_checkpointing"] = True
```

#### **Training Slow**
```python
# Reduce samples per epoch
TRAINING_CONFIG["samples_per_epoch"] = 5000

# Use mixed precision (already enabled)
TRAINING_CONFIG["mixed_precision"] = True
```

#### **Poor Convergence**
```python
# Increase learning rate
TRAINING_CONFIG["learning_rate"] = 1e-5

# More samples per epoch
TRAINING_CONFIG["samples_per_epoch"] = 20000
```

### Google Drive Issues
- Ensure sufficient space (50GB+)
- Check Drive mounting permissions
- Verify checkpoint directory creation

### Data Issues
- Verify corpus files exist and are UTF-8 encoded
- Check font files are valid TTF/OTF format
- Ensure validation set generation completed

## 📈 Performance Optimization

### For Faster Training
1. **Use A100 GPU** (if available)
2. **High-RAM runtime** for larger batches
3. **Mixed precision** training (enabled by default)
4. **Optimal batch size** (auto-adjusted)

### For Better Accuracy
1. **More training data** (larger corpus)
2. **Extended training** (more epochs)
3. **Data augmentation** (enabled by default)
4. **Multiple font variety** (8+ fonts recommended)

## 🔗 Resources

- **Repository**: https://github.com/kunthet/khmer-ocr-seq2seq
- **Documentation**: `docs/ON_THE_FLY_TRAINING.md`
- **PRD**: `docs/prd.md`
- **Changes**: `docs/changes.md`

## 🎉 Success Metrics

Your training is successful when:
- ✅ CER ≤ 1.0% achieved
- ✅ Word accuracy ≥ 95%
- ✅ Model converges smoothly
- ✅ Checkpoints saved to Google Drive
- ✅ Deployment package ready

## 💡 Tips for Success

1. **Start Small**: Test with 1-2 epochs first
2. **Monitor Progress**: Watch CER decrease over epochs
3. **Be Patient**: Training takes 15-25 hours
4. **Save Frequently**: Checkpoints every 5 epochs
5. **Use Resume**: Continue training if interrupted

---

**Happy Training! 🚀**

Your Khmer OCR model will be ready for production deployment once training completes successfully! 