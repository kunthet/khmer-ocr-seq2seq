#!/usr/bin/env python3
"""
Debug script to test Khmer text rendering and identify issues.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.text_renderer import TextRenderer
from data.khmer_text_renderer import KhmerTextRenderer

def debug_khmer_rendering():
    """Debug Khmer text rendering issues."""
    
    # Test cases with problematic subscripts
    test_cases = [
        "កន្ទុំ",    # ka + n subscript + tum (tail)
        "ជ្រុក",     # cho + r subscript + uk (pig)
        "ក្រុម",     # ka + r subscript + um (group)
        "ស្ទួន",     # sa + t subscript + uon (deer)
        "ក្រមុំ",     # ka + r subscript + mum (crab)
        "ដ្ឋាន",     # da + th subscript + aan (place)
        "ស្ត្រី",     # sa + t subscript + r subscript + ii (woman)
        "ពិន្ទុ",     # pa + n subscript + tu (dot)
        "ឈ្មោះ",     # cha + m subscript + uoh (name)
        "ខ្ញុំ",      # kha + nh subscript + um (I/me)
    ]
    
    print("🔍 Debugging Khmer Text Rendering Issues")
    print("=" * 60)
    
    # Initialize renderers
    text_renderer = TextRenderer()
    khmer_renderer = KhmerTextRenderer()
    
    # Check available fonts
    print(f"Available fonts: {len(text_renderer.fonts)}")
    for i, font_path in enumerate(text_renderer.fonts):
        print(f"  {i}: {font_path}")
    
    if not text_renderer.fonts:
        print("❌ No fonts available!")
        return
    
    # Test with first available font
    font_path = text_renderer.fonts[0]
    font_size = 32
    
    print(f"\n📝 Testing with font: {font_path}")
    print(f"📏 Font size: {font_size}")
    
    # Test each problematic case
    for i, test_text in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: '{test_text}' ---")
        
        # Analyze the text
        print(f"Original text: {test_text}")
        print(f"Text length: {len(test_text)}")
        print(f"Unicode points: {[hex(ord(c)) for c in test_text]}")
        
        # Check for Coeng sequences
        coeng_count = test_text.count('\u17D2')  # Coeng sign
        print(f"Coeng (subscript) count: {coeng_count}")
        
        # Test normalization
        normalized = khmer_renderer.normalize_khmer_text(test_text)
        print(f"Normalized text: {normalized}")
        print(f"Normalized length: {len(normalized)}")
        
        # Test rendering with different methods
        try:
            # Method 1: Current TextRenderer
            image_size = (200, 100)
            image1 = text_renderer.render_text(test_text, font_path)
            
            # Method 2: Advanced Khmer renderer
            image2, metadata = khmer_renderer.render_text(test_text, font_path, font_size, image_size)
            
            # Save test images
            output_dir = Path("test_output/debug_khmer")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image1.save(output_dir / f"test_{i+1}_original.png")
            image2.save(output_dir / f"test_{i+1}_advanced.png")
            
            print(f"✅ Images saved: test_{i+1}_original.png, test_{i+1}_advanced.png")
            print(f"Advanced renderer metadata: {metadata}")
            
        except Exception as e:
            print(f"❌ Error rendering '{test_text}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Test images saved in: test_output/debug_khmer/")
    print("Check the images to see the rendering differences!")

if __name__ == "__main__":
    debug_khmer_rendering() 