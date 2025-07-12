# Khmer OCR Seq2Seq

An attention-based Sequence-to-Sequence neural network for printed Khmer text recognition, implementing the architecture described in "Khmer printed character recognition using attention-based Seq2Seq network" (Buoy et al., 2022).

## 🎯 Project Overview

This project implements an end-to-end OCR system for Khmer text that:
- Takes raw text-line images as input (32px height, variable width)
- Generates character sequences as output without explicit segmentation
- Achieves target performance of ≤1.0% Character Error Rate (CER)
- Uses synthetic data generation for training

## 🏗️ Architecture

### Model Components
- **CRNN Encoder**: 13-layer convolutional-recurrent network
- **Attention Mechanism**: Bahdanau-style additive attention
- **GRU Decoder**: Recurrent decoder with teacher forcing
- **Vocabulary**: 117 tokens (4 special + 113 Khmer characters)

### Key Features
- Synthetic data generation using text2image
- Dynamic data augmentation (blur, morphology, noise, concatenation)
- Variable-width image processing
- Comprehensive evaluation metrics
- End-to-end training and inference pipeline
- Checkpoint saving/loading for resumable training
- Model freezing for transfer learning

## 📁 Project Structure

```
khmer_ocr_seq2seq/
├── src/                    # Source code
│   ├── data/              # Data pipeline components
│   ├── models/            # Neural network models
│   ├── training/          # Training infrastructure
│   ├── inference/         # Inference and evaluation
│   └── utils/             # Utilities and configuration
├── data/                   # Data storage
│   ├── raw/               # Raw corpus data
│   ├── processed/         # Processed datasets
│   └── synthetic/         # Generated synthetic data
├── models/                 # Model artifacts
│   ├── checkpoints/       # Training checkpoints
│   └── configs/           # Model configurations
├── configs/               # Configuration files
├── tests/                 # Unit and integration tests
├── logs/                  # Training logs and metrics
├── docs/                  # Documentation
└── prd/                   # Project requirements and tracking
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The project uses YAML configuration files in `configs/`:
- `train_config.yaml`: Training hyperparameters and model architecture
- Configuration is managed by `src/utils/config.py`

### 3. Vocabulary

The complete Khmer vocabulary (117 tokens) includes:
- **Special tokens** (4): SOS, EOS, PAD, UNK
- **Numbers** (20): Khmer digits (០-៩) + Arabic digits (0-9)
- **Consonants** (33): Complete Khmer consonant set
- **Vowels** (32): Independent (12) + Dependent (20)
- **Diacritics** (9): Subscript (1) + Other diacritics (8)
- **Symbols** (9): Punctuation and space

## 📋 Development Phases

### ✅ Phase 1: Project Setup [COMPLETED]
- [x] Repository structure
- [x] Dependencies (requirements.txt)
- [x] Configuration system
- [x] Vocabulary definition

### ✅ Phase 2: Data Pipeline [COMPLETED]
- [x] Character vocabulary class
- [x] Text-to-image renderer
- [x] Data augmentation pipeline
- [x] PyTorch Dataset implementation
- [x] Custom collate functions

### ✅ Phase 3: Model Architecture [COMPLETED]
- [x] CRNN encoder (13 layers)
- [x] Bahdanau-style attention
- [x] GRU decoder with attention
- [x] Complete Seq2Seq model integration
- [x] Model utilities (checkpointing, etc.)

### ✅ Phase 4: Training Infrastructure [COMPLETED]
- [x] Training loop with teacher forcing
- [x] Validation pipeline with CER calculation
- [x] Checkpoint management system
- [x] Logging and monitoring (TensorBoard)

### ✅ Phase 5: Inference & Evaluation [COMPLETED]
- [x] Greedy decoding for inference
- [x] Beam search decoding option
- [x] Character Error Rate (CER) and comprehensive metrics
- [x] Inference API with preprocessing

### ✅ Phase 6: Testing & Validation [COMPLETED]
- [x] Unit tests for data pipeline, model components, and utilities
- [x] Integration tests for end-to-end pipeline validation
- [x] Small-scale training test (50-100 samples) with performance monitoring
- [x] Memory profiling and performance optimization analysis

### 📅 Upcoming Phases
- **Phase 7**: Production Training (full-scale training with corpus data)

## 🔧 Development Guidelines

### Code Standards
- Follow CLEAN code principles
- Use type hints for all functions
- Comprehensive docstrings
- Modular, testable components
- Single Responsibility Principle

### Configuration Management
```python
from src.utils import ConfigManager

# Load configuration
config = ConfigManager()
vocab = config.vocab

# Access model parameters
model_config = config.model_config
training_config = config.training_config
```

## 📊 Target Performance

- **Character Error Rate**: ≤ 1.0% (target), ≤ 0.7% (stretch goal)
- **Training Duration**: 150 epochs
- **Hardware Requirements**: GPU with ≥16GB VRAM

## 🔗 References

- Buoy et al. (2022): "Khmer printed character recognition using attention-based Seq2Seq network"
- Implementation based on PRD specifications in `docs/prd.md`

## 📈 Progress Tracking

See `prd/PRD_ToDo.md` for detailed progress tracking and `docs/changes.md` for implementation history.

---

**Estimated Development Time**: 26-36 days
**Current Status**: Phase 6 Complete - Ready for Production Training 