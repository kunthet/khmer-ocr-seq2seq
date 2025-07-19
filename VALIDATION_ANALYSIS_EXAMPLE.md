# Quick Start: Validation Analysis

This guide shows how to quickly run comprehensive validation analysis on your trained Khmer OCR model.

## Prerequisites

1. **Trained model checkpoint** - e.g., `models/checkpoints/best_model.pth`
2. **Validation dataset** - `data/validation_fixed/` directory with 6,400 images
3. **Dependencies installed** - All packages in `requirements.txt`

## Basic Usage

### 1. Simple Analysis
Run analysis with default settings:
```bash
python full_validation_analysis.py --checkpoint models/checkpoints/best_model.pth
```

### 2. Custom Output Directory
Save results to a specific directory:
```bash
python full_validation_analysis.py \
    --checkpoint models/checkpoints/best_model.pth \
    --output-dir my_validation_results
```

### 3. High-Performance Analysis
Use larger batch size for faster processing (requires more GPU memory):
```bash
python full_validation_analysis.py \
    --checkpoint models/checkpoints/best_model.pth \
    --batch-size 64 \
    --output-dir fast_validation_results
```

### 4. Memory-Efficient Analysis
Use smaller batch size for limited GPU memory:
```bash
python full_validation_analysis.py \
    --checkpoint models/checkpoints/best_model.pth \
    --batch-size 8 \
    --output-dir memory_efficient_results
```

## Expected Runtime

| GPU | Batch Size | Expected Time | Memory Usage |
|-----|------------|---------------|--------------|
| RTX 3090 | 32 | ~3-5 minutes | ~8GB |
| RTX 4090 | 64 | ~2-3 minutes | ~12GB |
| RTX 3060 | 16 | ~8-10 minutes | ~6GB |
| CPU Only | 8 | ~30-45 minutes | ~4GB RAM |

## Sample Output Files

After running the analysis, you'll get these files in your output directory:

```
validation_analysis_results/
├── validation_results.json                 # Complete metrics and analysis
├── detailed_sample_results.csv            # Per-sample results (6,400 rows)
├── validation_summary_report.txt          # Human-readable summary
├── validation_analysis_visualizations.png # 12 charts and graphs
├── high_error_samples.csv                 # Worst 20 predictions
├── perfect_samples.csv                    # Best 20 predictions
└── validation_analysis.log                # Detailed execution log
```

## Key Metrics to Look For

### Excellent Performance (Production Ready)
- **Mean CER**: < 0.02 (2%)
- **Sequence Accuracy**: > 90%
- **Perfect Predictions**: > 70%

### Good Performance (Acceptable)
- **Mean CER**: 0.02-0.05 (2-5%)
- **Sequence Accuracy**: 80-90%
- **Perfect Predictions**: 50-70%

### Needs Improvement
- **Mean CER**: > 0.05 (5%)
- **Sequence Accuracy**: < 80%
- **Perfect Predictions**: < 50%

## Troubleshooting

### Error: "CUDA out of memory"
```bash
# Solution: Reduce batch size
python full_validation_analysis.py --checkpoint your_model.pth --batch-size 8
```

### Error: "Checkpoint not found"
```bash
# Check available checkpoints
ls models/checkpoints/*.pth

# Use correct path
python full_validation_analysis.py --checkpoint models/checkpoints/epoch_25.pth
```

### Error: "Validation dataset not found"
```bash
# Generate validation set first
python generate_fixed_validation_set.py

# Then run analysis
python full_validation_analysis.py --checkpoint your_model.pth
```

## Quick Performance Check

To quickly check if your model is performing well:

1. **Run the analysis**:
   ```bash
   python full_validation_analysis.py --checkpoint models/checkpoints/best_model.pth
   ```

2. **Check the console output** for the summary:
   ```
   📊 Results Summary:
      • Mean CER: 0.023 (2.3%)          ← Should be < 5%
      • Sequence Accuracy: 0.847 (84.7%) ← Should be > 80%
      • Exact Match: 0.823 (82.3%)       ← Should be > 70%
   ```

3. **Review the summary report**:
   ```bash
   cat validation_analysis_results/validation_summary_report.txt
   ```

4. **Check error distribution** in the visualizations:
   - Open `validation_analysis_visualizations.png`
   - Look at the "Error Rate Distribution" pie chart
   - Green sections (Perfect/Low Error) should dominate

## Next Steps

- **Good results**: Deploy model to production
- **Mixed results**: Analyze error samples and retrain specific areas
- **Poor results**: Review training data quality and model architecture

For detailed analysis guidance, see `README_VALIDATION_ANALYSIS.md`. 