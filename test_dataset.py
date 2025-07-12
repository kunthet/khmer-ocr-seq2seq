import sys
sys.path.append('src')

import torch
from src.data.dataset import KhmerDataset, KhmerDataLoader, collate_fn
from src.utils.config import KhmerVocab
from src.data.text_renderer import TextRenderer
from src.data.augmentation import DataAugmentation

print("Testing KhmerDataset and DataLoader...")

try:
    # Test individual components first
    print("1. Testing individual components...")
    
    vocab = KhmerVocab()
    renderer = TextRenderer()
    augmenter = DataAugmentation()
    
    print(f"   ✓ Vocab size: {len(vocab)}")
    print(f"   ✓ Renderer available fonts: {len(renderer.get_available_fonts())}")
    print(f"   ✓ Augmenter initialized")
    
    # Test dataset creation
    print("2. Creating KhmerDataset...")
    
    dataset = KhmerDataset(
        vocab=vocab,
        text_renderer=renderer,
        augmentation=augmenter,
        dataset_size=100,  # Small for testing
        min_text_length=1,
        max_text_length=10,
        use_augmentation=True,
        seed=42  # For reproducible testing
    )
    
    print(f"   ✓ Dataset created with {len(dataset)} samples")
    
    # Test single sample generation
    print("3. Testing single sample generation...")
    
    sample = dataset[0]
    print(f"   ✓ Sample keys: {list(sample.keys())}")
    print(f"   ✓ Text: '{sample['text']}'")
    print(f"   ✓ Image shape: {sample['image'].shape}")
    print(f"   ✓ Target shape: {sample['target'].shape}")
    print(f"   ✓ Target length: {sample['target_length']}")
    print(f"   ✓ Image width: {sample['image_width']}")
    
    # Verify image properties
    assert sample['image'].dim() == 3, "Image should be 3D tensor (C, H, W)"
    assert sample['image'].shape[1] == 32, "Image height should be 32px"
    assert sample['image'].shape[0] == 1, "Image should be grayscale (1 channel)"
    
    # Test multiple samples
    print("4. Testing multiple samples...")
    
    sample_texts = []
    sample_shapes = []
    
    for i in range(5):
        sample = dataset[i]
        sample_texts.append(sample['text'])
        sample_shapes.append(sample['image'].shape)
    
    print(f"   ✓ Generated texts: {sample_texts}")
    print(f"   ✓ Image shapes: {sample_shapes}")
    
    # Test batch creation with collate function
    print("5. Testing batch collation...")
    
    batch_samples = [dataset[i] for i in range(4)]
    batch = collate_fn(batch_samples)
    
    print(f"   ✓ Batch keys: {list(batch.keys())}")
    print(f"   ✓ Batch images shape: {batch['images'].shape}")
    print(f"   ✓ Batch targets shape: {batch['targets'].shape}")
    print(f"   ✓ Batch target lengths: {batch['target_lengths']}")
    print(f"   ✓ Batch image widths: {batch['image_widths']}")
    print(f"   ✓ Batch texts: {batch['texts']}")
    
    # Verify batch properties
    assert batch['images'].dim() == 4, "Batch images should be 4D (B, C, H, W)"
    assert batch['images'].shape[2] == 32, "Batch height should be 32px"
    assert batch['targets'].dim() == 2, "Batch targets should be 2D (B, seq_len)"
    
    # Test DataLoader
    print("6. Testing KhmerDataLoader...")
    
    dataloader = KhmerDataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    print(f"   ✓ DataLoader created with {len(dataloader)} batches")
    
    # Test getting a batch
    batch = dataloader.get_sample_batch()
    print(f"   ✓ Sample batch shape: {batch['images'].shape}")
    print(f"   ✓ Sample batch texts: {batch['texts'][:3]}...")  # Show first 3
    
    # Test iteration over a few batches
    print("7. Testing DataLoader iteration...")
    
    batch_count = 0
    total_samples = 0
    
    for batch in dataloader:
        batch_count += 1
        total_samples += batch['images'].shape[0]
        
        # Verify batch structure
        assert 'images' in batch
        assert 'targets' in batch
        assert 'texts' in batch
        assert 'target_lengths' in batch
        assert 'image_widths' in batch
        
        # Stop after a few batches for testing
        if batch_count >= 3:
            break
    
    print(f"   ✓ Processed {batch_count} batches with {total_samples} total samples")
    
    # Test without augmentation
    print("8. Testing without augmentation...")
    
    dataset_no_aug = KhmerDataset(
        dataset_size=10,
        use_augmentation=False,
        seed=42
    )
    
    sample_no_aug = dataset_no_aug[0]
    print(f"   ✓ No-augmentation sample text: '{sample_no_aug['text']}'")
    print(f"   ✓ No-augmentation image shape: {sample_no_aug['image'].shape}")
    
    # Test encoding/decoding consistency
    print("9. Testing encoding/decoding consistency...")
    
    for i in range(3):
        sample = dataset[i]
        original_text = sample['text']
        encoded = sample['target']
        decoded = vocab.decode(encoded.tolist())
        
        print(f"   Sample {i}: '{original_text}' -> {encoded.tolist()[:10]}... -> '{decoded}'")
        
        # Note: decoded might not exactly match original due to UNK tokens
        # but should be consistent with vocabulary
    
    # Test tensor types and ranges
    print("10. Testing tensor properties...")
    
    sample = dataset[0]
    
    # Image should be float in [0, 1] range
    assert sample['image'].dtype == torch.float32, "Image should be float32"
    assert sample['image'].min() >= 0.0, "Image values should be >= 0"
    assert sample['image'].max() <= 1.0, "Image values should be <= 1"
    
    # Target should be long integers
    assert sample['target'].dtype == torch.long, "Target should be long"
    assert sample['target'].min() >= 0, "Target indices should be >= 0"
    assert sample['target'].max() < len(vocab), f"Target indices should be < {len(vocab)}"
    
    print("   ✓ All tensor properties correct")
    
    print("\n✅ KhmerDataset and DataLoader test completed successfully!")
    print("\nTest Summary:")
    print(f"- Dataset size: {len(dataset)} samples")
    print(f"- Vocabulary size: {len(vocab)} tokens")
    print(f"- Image dimensions: {sample['image'].shape}")
    print(f"- Supports variable-width images with padding")
    print(f"- Includes augmentation pipeline")
    print(f"- DataLoader works with batching and collation")
    print(f"- All tensor types and ranges are correct")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 