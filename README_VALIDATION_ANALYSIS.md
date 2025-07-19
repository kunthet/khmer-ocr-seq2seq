# Comprehensive Validation Analysis for Khmer OCR

This document describes the `full_validation_analysis.py` script for comprehensive evaluation of trained Khmer OCR models.

## Overview

The validation analysis script provides detailed performance evaluation on the full validation dataset (`data/validation_fixed/`) with comprehensive metrics, error analysis, and performance profiling.

## Features

### 📊 **Comprehensive Metrics**
- **Character Error Rate (CER)** - Primary OCR accuracy metric
- **Word Error Rate (WER)** - Word-level accuracy
- **Sequence Accuracy** - Exact match percentage
- **BLEU Score** - Sequence similarity metric
- **Confidence Calibration** - Model confidence vs actual accuracy

### 🔍 **Detailed Analysis**
- **Performance by Text Length** - How accuracy varies with text complexity
- **Error Distribution** - Breakdown of error rates across samples
- **Speed Profiling** - Inference time analysis and throughput metrics
- **Confidence Analysis** - Confidence score distribution and calibration

### 📈 **Visualizations**
- CER distribution histograms
- Error rate by text length charts
- Confidence calibration curves
- Performance correlation plots
- Speed analysis charts

### 💾 **Output Formats**
- **JSON Results** - Complete analysis data (`validation_results.json`)
- **CSV Data** - Per-sample detailed results (`detailed_sample_results.csv`)
- **Summary Report** - Human-readable analysis (`validation_summary_report.txt`)
- **Visualizations** - Charts and graphs (`validation_analysis_visualizations.png`)
- **Error Samples** - High-error and perfect samples for inspection

## Usage

### Basic Usage

```bash
python full_validation_analysis.py --checkpoint models/checkpoints/best_model.pth
```

### Advanced Usage

```bash
python full_validation_analysis.py \
    --checkpoint models/checkpoints/epoch_50_cer_0.015.pth \
    --validation-dir data/validation_fixed \
    --output-dir validation_analysis_results \
    --batch-size 64 \
    --config configs/config.yaml \
    --log-level INFO
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | **Required.** Path to model checkpoint file | - |
| `--validation-dir` | Path to validation dataset directory | `data/validation_fixed` |
| `--output-dir` | Output directory for results | `validation_analysis_results` |
| `--batch-size` | Batch size for validation (higher = faster, more memory) | `32` |
| `--config` | Path to configuration file | `configs/config.yaml` |
| `--log-level` | Logging verbosity level | `INFO` |

## Example Output

### Console Output
```
================================================================================
KHMER OCR FULL VALIDATION ANALYSIS
================================================================================
Using device: cuda
GPU: NVIDIA RTX 3090
Loading model from checkpoint: models/checkpoints/best_model.pth
Model loaded successfully
Loading validation dataset from: data/validation_fixed
Loaded 6,400 validation samples in 200 batches
Starting comprehensive validation analysis...
Processing batch 0/200 (0.0%)
Processing batch 20/200 (10.0%)
...
================================================================================
VALIDATION ANALYSIS COMPLETED SUCCESSFULLY!
================================================================================
📊 Results Summary:
   • Total Samples: 6,400
   • Mean CER: 0.023 (2.3%)
   • Sequence Accuracy: 0.847 (84.7%)
   • Exact Match: 0.823 (82.3%)
   • Mean Confidence: 0.891
⚡ Performance: 45.2 samples/second
✅ Perfect Predictions: 67.8%
🎯 Low Error (≤10%): 89.3%
```

### Generated Files

After completion, the following files are created in the output directory:

```
validation_analysis_results/
├── validation_results.json                    # Complete analysis data (JSON)
├── detailed_sample_results.csv               # Per-sample results (CSV)
├── validation_summary_report.txt             # Human-readable summary
├── validation_analysis_visualizations.png    # Charts and graphs
├── high_error_samples.csv                    # Worst performing samples
├── perfect_samples.csv                       # Perfect predictions
└── validation_analysis.log                   # Detailed execution log
```

## Understanding the Results

### Key Metrics

- **CER (Character Error Rate)**: Lower is better. Values below 0.05 (5%) are excellent for OCR.
- **Sequence Accuracy**: Percentage of completely correct predictions.
- **Confidence Calibration**: How well model confidence correlates with actual accuracy.

### Performance Analysis

The script analyzes performance across different text lengths:
- **0-10 characters**: Short text performance
- **11-20 characters**: Medium text performance  
- **21-50 characters**: Long text performance
- **51+ characters**: Very long text performance

### Error Analysis

- **Perfect (0% CER)**: Completely correct predictions
- **Low Error (≤10% CER)**: Mostly correct with minor errors
- **Medium Error (10-30% CER)**: Partially correct
- **High Error (>30% CER)**: Poor predictions requiring investigation

## Performance Expectations

For a well-trained model on the validation_fixed dataset, expect:

- **Mean CER**: 0.01-0.03 (1-3%)
- **Sequence Accuracy**: 80-95%
- **Exact Match**: 75-90%
- **Inference Speed**: 20-100 samples/second (depending on GPU)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```
   Solution: Reduce --batch-size to 16 or 8
   ```

2. **Checkpoint Not Found**
   ```
   Error: Checkpoint not found: models/checkpoints/best_model.pth
   Solution: Verify checkpoint path exists
   ```

3. **Validation Dataset Missing**
   ```
   Error: Split directory not found: data/validation_fixed/val
   Solution: Generate validation set with generate_fixed_validation_set.py
   ```

### Performance Optimization

- **Faster Analysis**: Increase `--batch-size` (requires more GPU memory)
- **Memory Efficient**: Decrease `--batch-size` or use CPU device
- **Detailed Logging**: Use `--log-level DEBUG` for verbose output

## Integration with Training

### During Training
Use this script to evaluate checkpoints periodically:

```bash
# Evaluate best checkpoint
python full_validation_analysis.py --checkpoint models/checkpoints/best_model.pth

# Evaluate specific epoch
python full_validation_analysis.py --checkpoint models/checkpoints/epoch_25.pth
```

### Model Comparison
Compare multiple models by running analysis on different checkpoints:

```bash
for checkpoint in models/checkpoints/*.pth; do
    echo "Analyzing $checkpoint"
    python full_validation_analysis.py --checkpoint "$checkpoint" --output-dir "analysis_$(basename $checkpoint .pth)"
done
```

## Technical Notes

### Memory Requirements
- **Minimum**: 4GB GPU memory with batch_size=8
- **Recommended**: 8GB+ GPU memory with batch_size=32
- **High Performance**: 16GB+ GPU memory with batch_size=64

### Validation Dataset
The script expects the `data/validation_fixed/` directory structure:
```
data/validation_fixed/
├── val/
│   ├── images/           # 6,400 validation images
│   └── labels/           # Corresponding text labels
└── metadata.json         # Dataset metadata
```

### Dependencies
- PyTorch with CUDA support
- PIL/Pillow for image processing
- pandas for data analysis
- matplotlib/seaborn for visualizations
- numpy for numerical computations

## Contributing

To extend the validation analysis:

1. **Add New Metrics**: Modify `_analyze_all_samples()` method
2. **Custom Visualizations**: Extend `generate_visualizations()` method  
3. **Additional Analysis**: Add methods to `FullValidationAnalyzer` class
4. **Output Formats**: Modify `save_results()` for new export formats

## License

This validation analysis script is part of the Khmer OCR Seq2Seq project and follows the same license terms. 