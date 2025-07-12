# Khmer OCR Inference Guide

This guide explains how to use the inference system to test your trained Khmer OCR model.

## Overview

The inference system provides several ways to test your trained model:

1. **Quick Test** - Simple validation on synthetic data
2. **Comprehensive Testing** - Advanced testing with multiple modes
3. **Demo Mode** - Interactive demonstrations
4. **API Usage** - Direct use of the OCR engine

## Files Overview

- `quick_test.py` - Simple test script for immediate validation
- `inference_test.py` - Comprehensive testing with multiple modes
- `src/inference/demo.py` - Interactive demo with examples
- `src/inference/ocr_engine.py` - Core OCR engine class
- `src/inference/metrics.py` - Evaluation metrics

## Quick Start

### 1. Quick Test (Recommended First Step)

Test your model quickly on synthetic data:

```bash
python quick_test.py
```

This will:
- Load your trained model from `models/checkpoints/best_model.pth`
- Test on 5 synthetic samples
- Show predictions, accuracy, and timing
- Work even if no checkpoint exists (uses untrained model)

### 2. Comprehensive Testing

Use the comprehensive test script for detailed evaluation:

```bash
# Test on synthetic dataset
python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode synthetic

# Test a single image
python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode single --image path/to/image.png

# Batch process a directory
python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode batch --input-dir path/to/images/

# Interactive mode
python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode interactive

# Performance benchmark
python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode benchmark
```

### 3. Demo Mode

Run the interactive demo:

```bash
python src/inference/demo.py --checkpoint models/checkpoints/best_model.pth
```

## Testing Modes

### Synthetic Dataset Testing

Tests your model on the synthetic test dataset:

```bash
python inference_test.py --mode synthetic --max-samples 100
```

**Output:**
- Character Error Rate (CER)
- Word Error Rate (WER)
- Sequence Accuracy
- BLEU Score
- Processing time statistics
- Sample predictions

### Single Image Testing

Test on a specific image:

```bash
python inference_test.py --mode single --image test_image.png --ground-truth "កម្ពុជា"
```

**Features:**
- Tests both greedy and beam search
- Shows confidence scores
- Compares with ground truth if provided
- Displays processing time

### Batch Processing

Process all images in a directory:

```bash
python inference_test.py --mode batch --input-dir images/ --output-file results.json
```

**Features:**
- Processes all images (PNG, JPG, JPEG, BMP, TIFF)
- Saves results to JSON file
- Shows processing statistics
- Handles errors gracefully

### Interactive Mode

Test images interactively:

```bash
python inference_test.py --mode interactive
```

**Usage:**
1. Enter image path when prompted
2. Optionally provide ground truth
3. See results for both greedy and beam search
4. Type 'quit' to exit

### Performance Benchmark

Measure processing speed:

```bash
python inference_test.py --mode benchmark --max-samples 100
```

**Output:**
- Samples per second
- Average processing time
- Time statistics (min, max, std)
- Comparison between greedy and beam search

## API Usage

### Basic Usage

```python
from src.inference.ocr_engine import KhmerOCREngine
from src.utils.config import ConfigManager
from PIL import Image

# Load model
config = ConfigManager()
engine = KhmerOCREngine.from_checkpoint(
    "models/checkpoints/best_model.pth",
    config_manager=config
)

# Load image
image = Image.open("test_image.png")

# Recognize text
result = engine.recognize(
    image,
    method='greedy',
    return_confidence=True,
    preprocess=True
)

print(f"Recognized text: '{result['text']}'")
print(f"Confidence: {result['confidence']:.3f}")
```

### Advanced Usage

```python
# Beam search with custom parameters
result = engine.recognize(
    image,
    method='beam_search',
    beam_size=5,
    length_penalty=0.6,
    return_confidence=True
)

# Batch processing
images = [Image.open(f"image_{i}.png") for i in range(10)]
results = engine.recognize_batch(images, method='greedy')

# Custom preprocessing
result = engine.recognize(
    image,
    method='greedy',
    preprocess=True  # Enable preprocessing
)
```

### Preprocessing Options

The OCR engine supports various preprocessing options:

```python
# Manual preprocessing
processed_image = engine.preprocess_image(
    image,
    enhance_contrast=True,
    denoise=True,
    binarize=False
)

# Use preprocessed image
result = engine.recognize(processed_image, preprocess=False)
```

## Evaluation Metrics

### Character Error Rate (CER)

Measures character-level accuracy:
- **Good**: < 5%
- **Acceptable**: 5-15%
- **Poor**: > 15%

### Word Error Rate (WER)

Measures word-level accuracy:
- **Good**: < 10%
- **Acceptable**: 10-25%
- **Poor**: > 25%

### Sequence Accuracy

Percentage of perfectly predicted sequences:
- **Good**: > 80%
- **Acceptable**: 50-80%
- **Poor**: < 50%

### BLEU Score

Measures sequence similarity (0-1):
- **Good**: > 0.8
- **Acceptable**: 0.5-0.8
- **Poor**: < 0.5

## Expected Performance

Based on your training log (CER: 120.66%), the model currently has poor performance. This is expected for:

1. **Untrained model** - Random weights
2. **Insufficient training** - Only 1 epoch
3. **Small dataset** - Only 100 samples
4. **Learning rate issues** - May need adjustment

### Improving Performance

1. **Train longer**: Use more epochs (10-50)
2. **Larger dataset**: Add more training samples
3. **Adjust hyperparameters**: Learning rate, batch size
4. **Better preprocessing**: Enhance image quality
5. **Data augmentation**: Increase training diversity

## Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Solution: Check checkpoint path, train model first
   ```

2. **CUDA out of memory**
   ```
   Solution: Use --device cpu or reduce batch size
   ```

3. **Poor accuracy**
   ```
   Solution: Train longer, check data quality, adjust hyperparameters
   ```

4. **Slow inference**
   ```
   Solution: Use greedy instead of beam search, reduce image size
   ```

### Debug Mode

Enable debug information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Test Current Model

```bash
# Quick test with current model
python quick_test.py

# Detailed evaluation
python inference_test.py --mode synthetic --max-samples 50

# Test specific image
python inference_test.py --mode single --image data/synthetic/test/images/test_000001.png
```

### Production Use

```python
# Production inference example
from src.inference.ocr_engine import KhmerOCREngine
from src.utils.config import ConfigManager
from PIL import Image
import time

# Initialize once
config = ConfigManager()
engine = KhmerOCREngine.from_checkpoint(
    "models/checkpoints/best_model.pth",
    config_manager=config
)

# Process multiple images
for image_path in image_paths:
    image = Image.open(image_path)
    
    start_time = time.time()
    result = engine.recognize(image, method='greedy')
    processing_time = time.time() - start_time
    
    print(f"Image: {image_path}")
    print(f"Text: '{result['text']}'")
    print(f"Time: {processing_time:.3f}s")
    print()
```

## Performance Tips

1. **Use greedy decoding** for speed
2. **Batch process** multiple images
3. **Preprocess once** if processing similar images
4. **Use GPU** if available
5. **Optimize image size** (height=32px recommended)

## Next Steps

1. **Improve model**: Train with more data and epochs
2. **Add real data**: Test on actual Khmer text images
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Deploy**: Create web service or API
5. **Monitor**: Track performance over time 