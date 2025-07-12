# Image Cache System Documentation

## Overview

The Khmer OCR Seq2Seq project includes a sophisticated image caching system that stores generated synthetic images in the `data/synthetic/` folder. This system provides:

- **Consistent Results**: Same text + font combinations always produce identical images
- **Faster Training**: Cached images are loaded instantly instead of being re-rendered
- **Reproducible Experiments**: Enables consistent comparison between training runs
- **Storage Efficiency**: Optimized storage with metadata tracking

## Architecture

### Directory Structure

```
data/synthetic/
├── cache_index.json          # Master index of all cached images
├── images/                   # Rendered PNG images
│   ├── abc123def.png         # Cached image files (MD5 hash names)
│   └── ...
└── metadata/                 # Detailed metadata for each image
    ├── abc123def.json        # Metadata files
    └── ...
```

### Cache Key Generation

Each cached image is identified by an MD5 hash of:
- Text content
- Font file path
- Image height

This ensures unique identification while handling:
- Different fonts rendering the same text
- Same text at different image heights
- Consistent retrieval across sessions

## Usage

### Automatic Caching

The system automatically caches images during training:

```python
# Caching is enabled by default
dataset = KhmerCorpusDataset(
    split="train",
    use_cache=True,           # Enable caching (default)
    cache_dir="data/synthetic" # Cache directory (default)
)
```

### Manual Cache Management

Use the `manage_cache.py` utility for cache operations:

```bash
# Show cache statistics
python manage_cache.py stats

# Pre-generate cache for faster training
python manage_cache.py pregenerate --max-lines 1000

# Clear cache
python manage_cache.py clear

# Validate cache integrity
python manage_cache.py validate

# Export cache information
python manage_cache.py export --output cache_info.json
```

### Training with Cache

The training script automatically uses caching:

```bash
# Training automatically uses cached images
python src/training/train_production.py --max-lines 1000 --num-epochs 5
```

## Cache Behavior

### Training vs Validation/Test

- **Training**: Uses random font selection for data augmentation
- **Validation/Test**: Uses consistent font selection (hash-based) for reproducible results

### Cache Hit/Miss

1. **Cache Hit**: Image loaded instantly from disk
2. **Cache Miss**: Image rendered, then cached for future use

### Augmentation Handling

- Base images are cached before augmentation
- Augmentation is applied dynamically during training
- Validation/test sets use unaugmented cached images

## Performance Benefits

### Speed Improvements

- **First Run**: Normal rendering speed + caching overhead
- **Subsequent Runs**: ~10-100x faster image loading
- **Large Datasets**: Significant time savings for repeated experiments

### Storage Efficiency

- Average image size: ~4KB per image
- Metadata overhead: ~1KB per image
- 100K images ≈ 500MB total storage

## Configuration

### Dataset Parameters

```python
KhmerCorpusDataset(
    split="train",
    use_cache=True,                    # Enable/disable caching
    cache_dir="data/synthetic",        # Cache directory
    # ... other parameters
)
```

### Cache Directory Structure

The cache automatically creates:
- `images/`: PNG files with MD5 hash names
- `metadata/`: JSON files with rendering details
- `cache_index.json`: Master index for fast lookup

## Metadata Format

Each cached image includes metadata:

```json
{
  "text": "ភាសាខ្មែរ",
  "font_path": "fonts/KhmerOS.ttf",
  "image_height": 32,
  "image_width": 128,
  "cache_key": "abc123def456...",
  "created_at": "1641234567.89"
}
```

## Cache Management Commands

### Statistics
```bash
python manage_cache.py stats
```
Shows:
- Total cached images
- Storage usage
- Directory structure

### Pre-generation
```bash
# Generate cache for specific splits
python manage_cache.py pregenerate --splits train val

# Limit number of samples
python manage_cache.py pregenerate --max-lines 5000

# Generate for all splits
python manage_cache.py pregenerate
```

### Validation
```bash
python manage_cache.py validate
```
Checks:
- Index consistency
- File existence
- Metadata integrity

### Cleanup
```bash
# Clear all cached data
python manage_cache.py clear

# Clear with confirmation
python manage_cache.py clear --yes
```

## Best Practices

### Cache Management

1. **Pre-generate for Large Datasets**: Use `pregenerate` command before training
2. **Monitor Storage**: Check cache size regularly with `stats` command
3. **Validate Integrity**: Run `validate` after system changes
4. **Backup Important Caches**: Consider backing up cache for critical experiments

### Development Workflow

1. **Development**: Use small cache (`--max-lines 100`) for quick iteration
2. **Experimentation**: Pre-generate full cache for consistent results
3. **Production**: Use full cache for final training runs

### Storage Considerations

- **SSD Recommended**: Faster I/O for cache access
- **Network Storage**: Avoid network drives for cache directory
- **Disk Space**: Plan for ~5KB per unique text+font combination

## Troubleshooting

### Common Issues

1. **Cache Miss on Expected Hit**
   - Check font path consistency
   - Verify image height matches
   - Ensure text encoding is identical

2. **Slow Cache Performance**
   - Use SSD storage
   - Avoid network drives
   - Check disk space availability

3. **Cache Corruption**
   - Run `validate` command
   - Clear and regenerate if needed
   - Check file permissions

### Error Messages

- `"Error loading cached image"`: File corruption, will re-render
- `"Error caching image"`: Disk space or permission issue
- `"Cache validation failed"`: Run integrity check

## Integration Examples

### Custom Training Script

```python
from src.data.corpus_dataset import KhmerCorpusDataset, create_corpus_dataloaders

# Create datasets with caching
train_loader, val_loader, test_loader = create_corpus_dataloaders(
    config_manager=config,
    batch_size=32,
    use_cache=True,
    cache_dir="data/synthetic"
)

# Cache statistics
dataset = train_loader.dataset
cache_info = dataset.get_cache_info()
print(f"Cache: {cache_info['total_images']} images, {cache_info['total_size_mb']:.1f}MB")
```

### Programmatic Cache Management

```python
from src.data.corpus_dataset import ImageCache

# Initialize cache
cache = ImageCache("data/synthetic")

# Get statistics
stats = cache.get_cache_stats()
print(f"Cached images: {stats['total_images']}")

# Clear cache
cache.clear_cache()
```

## Future Enhancements

### Planned Features

1. **Distributed Caching**: Share cache across multiple machines
2. **Compression**: Reduce storage footprint
3. **Version Control**: Track cache versions with model changes
4. **Analytics**: Detailed cache hit/miss statistics

### Optimization Opportunities

1. **Parallel Pre-generation**: Multi-threaded cache building
2. **Smart Eviction**: LRU-based cache management
3. **Delta Caching**: Store only differences for similar images
4. **Memory Caching**: In-memory cache for frequently accessed images

---

## Summary

The image caching system provides significant performance improvements for the Khmer OCR training pipeline while ensuring reproducible results. By caching rendered images with their metadata, the system eliminates redundant text rendering operations and enables consistent experimental comparisons.

Key benefits:
- ✅ **10-100x faster** image loading for repeated training runs
- ✅ **Consistent results** across different training sessions
- ✅ **Efficient storage** with metadata tracking
- ✅ **Easy management** through command-line utilities
- ✅ **Automatic operation** with minimal configuration required

The system is designed to be transparent to users while providing powerful optimization and reproducibility features for serious machine learning workflows. 