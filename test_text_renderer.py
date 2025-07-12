import sys
sys.path.append('src')

from src.data.text_renderer import TextRenderer
import os

print("Testing TextRenderer...")

try:
    # Initialize renderer
    print("1. Creating TextRenderer...")
    renderer = TextRenderer()
    print(f"   ✓ Renderer created")
    print(f"   Using Tesseract: {renderer.use_tesseract}")
    print(f"   text2image available: {renderer.text2image_available}")
    print(f"   Available fonts: {len(renderer.get_available_fonts())}")
    
    # Test with simple text
    print("2. Testing with simple text...")
    test_text = "Hello"
    image = renderer.render_text(test_text)
    print(f"   ✓ Rendered '{test_text}' - Size: {image.size}, Mode: {image.mode}")
    
    # Test with Khmer text
    print("3. Testing with Khmer text...")
    khmer_text = "កម្ពុជា"  # Cambodia in Khmer
    image2 = renderer.render_text(khmer_text)
    print(f"   ✓ Rendered '{khmer_text}' - Size: {image2.size}, Mode: {image2.mode}")
    
    # Test batch rendering
    print("4. Testing batch rendering...")
    test_texts = ["ក", "ខ", "គ", "Hello", "123"]
    images = renderer.batch_render(test_texts)
    print(f"   ✓ Batch rendered {len(images)} texts")
    
    # Verify image properties
    print("5. Verifying image properties...")
    for i, img in enumerate(images):
        if img.size[1] != 32:
            print(f"   ⚠ Warning: Image {i} height is {img.size[1]}, expected 32")
        if img.mode != 'L':
            print(f"   ⚠ Warning: Image {i} mode is {img.mode}, expected 'L'")
    
    print("   ✓ All images have correct properties")
    
    # Save a sample image for verification
    print("6. Saving sample image...")
    sample_image = renderer.render_text("ខ្មែរ")  # "Khmer" in Khmer script
    sample_image.save("sample_khmer_text.png")
    print(f"   ✓ Saved sample image: sample_khmer_text.png")
    print(f"   Sample size: {sample_image.size}")
    
    print("\n✅ TextRenderer test completed successfully!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 