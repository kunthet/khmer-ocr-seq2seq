#!/usr/bin/env python3
"""
Comprehensive test to verify Khmer subscript rendering fixes.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.text_renderer import TextRenderer
from data.khmer_text_renderer import KhmerTextRenderer

def test_subscript_rendering():
    """Test that Khmer subscript rendering issues are resolved."""
    
    print("🧪 Testing Khmer Subscript Rendering Fix")
    print("=" * 60)
    
    # Test cases that previously had issues
    problematic_cases = [
        ("កន្ទុំ", "ka + n subscript + tum (tail)"),
        ("ជ្រុក", "cho + r subscript + uk (pig)"),
        ("ក្រុម", "ka + r subscript + um (group)"),
        ("ស្ទួន", "sa + t subscript + uon (deer)"),
        ("ក្រមុំ", "ka + r subscript + mum (crab)"),
        ("ដ្ឋាន", "da + th subscript + aan (place)"),
        ("ស្ត្រី", "sa + t subscript + r subscript + ii (woman)"),
        ("ពិន្ទុ", "pa + n subscript + tu (dot)"),
        ("ឈ្មោះ", "cha + m subscript + uoh (name)"),
        ("ខ្ញុំ", "kha + nh subscript + um (I/me)"),
    ]
    
    # Initialize renderer
    renderer = TextRenderer()
    
    print(f"✅ TextRenderer initialized with {len(renderer.fonts)} fonts")
    print(f"✅ Advanced Khmer renderer available: {renderer.khmer_renderer is not None}")
    
    # Test each problematic case
    results = []
    for i, (text, description) in enumerate(problematic_cases):
        print(f"\n--- Test {i+1}: {text} ({description}) ---")
        
        try:
            # Generate image
            image = renderer.render_text(text)
            
            # Check image properties
            width, height = image.size
            
            # Save test image
            output_dir = Path("test_output/subscript_fix_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_path = output_dir / f"subscript_test_{i+1:02d}.png"
            image.save(image_path)
            
            # Analyze image content
            image_array = image.convert('L')
            pixels = list(image_array.getdata())
            
            # Check for content (not all white/empty)
            non_white_pixels = sum(1 for p in pixels if p < 250)
            total_pixels = len(pixels)
            content_ratio = non_white_pixels / total_pixels
            
            # Results
            success = content_ratio > 0.01  # At least 1% content
            
            result = {
                'text': text,
                'description': description,
                'success': success,
                'image_size': (width, height),
                'content_ratio': content_ratio,
                'file_path': str(image_path)
            }
            
            results.append(result)
            
            if success:
                print(f"✅ SUCCESS: Image {width}x{height}, {content_ratio:.1%} content")
            else:
                print(f"❌ FAILED: Image {width}x{height}, {content_ratio:.1%} content")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'text': text,
                'description': description,
                'success': False,
                'error': str(e)
            })
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r.get('success', False))
    total_tests = len(results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Khmer subscript rendering issues have been RESOLVED!")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests still failing")
        print("❌ Some subscript rendering issues remain")
    
    # List failed tests
    failed_tests = [r for r in results if not r.get('success', False)]
    if failed_tests:
        print("\n❌ Failed tests:")
        for test in failed_tests:
            print(f"  - {test['text']} ({test['description']})")
            if 'error' in test:
                print(f"    Error: {test['error']}")
    
    print(f"\n🎯 Test images saved in: test_output/subscript_fix_test/")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = test_subscript_rendering()
    sys.exit(0 if success else 1) 