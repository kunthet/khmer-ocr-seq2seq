import sys
sys.path.append('src')

from src.data.text_renderer import TextRenderer
from src.data.augmentation import DataAugmentation
from PIL import Image
import os

print("Testing DataAugmentation...")

try:
    # Create test images using TextRenderer
    print("1. Creating test images...")
    renderer = TextRenderer()
    
    # Create several test images
    test_texts = ["ក", "ខ", "គ", "ឃ", "ង"]
    test_images = renderer.batch_render(test_texts)
    print(f"   ✓ Created {len(test_images)} test images")
    
    # Initialize augmentation pipeline
    print("2. Initializing DataAugmentation...")
    augmenter = DataAugmentation(
        blur_prob=1.0,  # Force blur for testing
        morph_prob=1.0,  # Force morphological ops
        noise_prob=1.0,  # Force noise
        concat_prob=0.5   # 50% chance of concatenation
    )
    print("   ✓ DataAugmentation initialized")
    print(f"   Parameters: {augmenter.get_augmentation_stats()}")
    
    # Test individual augmentations
    print("3. Testing individual augmentations...")
    
    # Test Gaussian blur
    original_image = test_images[0]
    blurred = augmenter.apply_gaussian_blur(original_image)
    print(f"   ✓ Gaussian blur: {original_image.size} -> {blurred.size}")
    
    # Test morphological operations
    morphed = augmenter.apply_morphological_operations(original_image)
    print(f"   ✓ Morphological ops: {original_image.size} -> {morphed.size}")
    
    # Test noise injection
    noisy = augmenter.apply_noise_injection(original_image)
    print(f"   ✓ Noise injection: {original_image.size} -> {noisy.size}")
    
    # Test concatenation
    concat_images = test_images[:3]
    concatenated = augmenter.concatenate_images(concat_images)
    expected_width = sum(img.width for img in concat_images)
    print(f"   ✓ Concatenation: {[img.size for img in concat_images]} -> {concatenated.size}")
    print(f"   Expected width: {expected_width}, actual: {concatenated.width}")
    
    # Test full augmentation pipeline
    print("4. Testing full augmentation pipeline...")
    fully_augmented = augmenter.apply_augmentations(original_image)
    print(f"   ✓ Full augmentation: {original_image.size} -> {fully_augmented.size}")
    
    # Test batch augmentations
    print("5. Testing batch augmentations...")
    augmented_images, augmented_texts = augmenter.apply_batch_augmentations(
        test_images, test_texts
    )
    print(f"   ✓ Batch augmentation: {len(test_images)} -> {len(augmented_images)} images")
    print(f"   Original texts: {test_texts}")
    print(f"   Augmented texts: {augmented_texts}")
    
    # Save sample outputs for visual inspection
    print("6. Saving sample outputs...")
    
    # Save original
    original_image.save("sample_original.png")
    
    # Save augmented versions
    blurred.save("sample_blurred.png")
    morphed.save("sample_morphed.png") 
    noisy.save("sample_noisy.png")
    concatenated.save("sample_concatenated.png")
    fully_augmented.save("sample_fully_augmented.png")
    
    print("   ✓ Saved sample images:")
    print("     - sample_original.png")
    print("     - sample_blurred.png")
    print("     - sample_morphed.png")
    print("     - sample_noisy.png")
    print("     - sample_concatenated.png")
    print("     - sample_fully_augmented.png")
    
    # Test with different probabilities
    print("7. Testing with different probabilities...")
    augmenter_low_prob = DataAugmentation(
        blur_prob=0.1,
        morph_prob=0.1,
        noise_prob=0.1,
        concat_prob=0.1
    )
    
    # Set seed for reproducible results
    augmenter_low_prob.set_seed(42)
    
    low_prob_result = augmenter_low_prob.apply_augmentations(original_image)
    print(f"   ✓ Low probability augmentation: {original_image.size} -> {low_prob_result.size}")
    
    print("\n✅ DataAugmentation test completed successfully!")
    print("\nTest Summary:")
    print(f"- Created {len(test_images)} base images")
    print(f"- Tested all individual augmentation techniques")
    print(f"- Batch processing: {len(test_images)} -> {len(augmented_images)} images")
    print(f"- All images maintain 32px height: {all(img.height == 32 for img in augmented_images)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 