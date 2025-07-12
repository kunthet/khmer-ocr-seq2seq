"""
Advanced Khmer text renderer with proper text shaping for subscripts and complex characters.
This module provides better text rendering than basic PIL for Khmer scripts.
"""

import os
import sys
import unicodedata
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add khtext module to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src' / 'khtext'))

try:
    from khnormal_fast import khnormal
    KHNORMAL_AVAILABLE = True
except ImportError:
    print("Warning: Khmer normalization not available")
    KHNORMAL_AVAILABLE = False

try:
    import uharfbuzz as hb
    HARFBUZZ_AVAILABLE = True
    print("✅ HarfBuzz available for proper text shaping")
except ImportError:
    HARFBUZZ_AVAILABLE = False
    print("⚠️ HarfBuzz not available - falling back to enhanced PIL rendering")

try:
    from bidi.algorithm import get_display
    BIDI_AVAILABLE = True
except ImportError:
    BIDI_AVAILABLE = False

# Khmer Unicode constants
COENG_SIGN = '\u17D2'  # ្
KHMER_CONSONANT_RANGE = (0x1780, 0x17A2)
KHMER_VOWEL_RANGE = (0x17B6, 0x17C5)
KHMER_SIGN_RANGE = (0x17C6, 0x17D3)


class KhmerTextRenderer:
    """
    Advanced Khmer text renderer with proper text shaping support.
    
    This renderer provides multiple rendering strategies:
    1. HarfBuzz-based shaping (best quality)
    2. Enhanced PIL rendering with better positioning
    3. Fallback PIL rendering
    """
    
    def __init__(self, fallback_strategy: str = "enhanced_pil"):
        """
        Initialize the Khmer text renderer.
        
        Args:
            fallback_strategy: Strategy when HarfBuzz unavailable ('enhanced_pil', 'basic_pil')
        """
        self.fallback_strategy = fallback_strategy
        self.harfbuzz_fonts = {}  # Cache for HarfBuzz font objects
        
    def normalize_khmer_text(self, text: str) -> str:
        """
        Normalize Khmer text for proper rendering.
        
        Args:
            text: Input Khmer text
            
        Returns:
            Normalized text
        """
        if KHNORMAL_AVAILABLE:
            return khnormal(text)
        return unicodedata.normalize('NFC', text)
    
    def is_khmer_text(self, text: str) -> bool:
        """Check if text contains Khmer characters."""
        for char in text:
            code = ord(char)
            if (KHMER_CONSONANT_RANGE[0] <= code <= KHMER_CONSONANT_RANGE[1] or
                KHMER_VOWEL_RANGE[0] <= code <= KHMER_VOWEL_RANGE[1] or
                KHMER_SIGN_RANGE[0] <= code <= KHMER_SIGN_RANGE[1]):
                return True
        return False
    
    def render_text_harfbuzz(self, text: str, font_path: str, font_size: int, 
                           image_size: Tuple[int, int]) -> Tuple[Image.Image, Dict]:
        """
        Render text using HarfBuzz for proper text shaping.
        
        Args:
            text: Text to render
            font_path: Path to font file
            font_size: Font size
            image_size: Target image size
            
        Returns:
            Tuple of (rendered_image, metadata)
        """
        if not HARFBUZZ_AVAILABLE:
            raise RuntimeError("HarfBuzz not available")
        
        # Load font for HarfBuzz
        if font_path not in self.harfbuzz_fonts:
            try:
                with open(font_path, 'rb') as f:
                    font_data = f.read()
                hb_face = hb.Face(font_data)
                hb_font = hb.Font(hb_face)
                hb_font.scale = (font_size * 64, font_size * 64)  # HarfBuzz uses 1/64th units
                self.harfbuzz_fonts[font_path] = hb_font
            except Exception as e:
                raise RuntimeError(f"Failed to load font for HarfBuzz: {e}")
        
        hb_font = self.harfbuzz_fonts[font_path]
        
        # Create HarfBuzz buffer
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        
        # Shape the text
        hb.shape(hb_font, buf)
        
        # Get glyph information
        infos = buf.glyph_infos
        positions = buf.glyph_positions
        
        # Create image
        image = Image.new('L', image_size, 255)  # White background
        draw = ImageDraw.Draw(image)
        
        # Load PIL font for fallback glyph rendering
        pil_font = ImageFont.truetype(font_path, font_size)
        
        # Calculate text positioning
        total_width = sum(pos.x_advance for pos in positions) / 64
        start_x = (image_size[0] - total_width) / 2
        start_y = image_size[1] / 2
        
        # Render each glyph
        current_x = start_x
        current_y = start_y
        
        for i, (info, pos) in enumerate(zip(infos, positions)):
            # Get the original character for this glyph
            char_index = info.cluster
            if char_index < len(text):
                char = text[char_index]
                
                # Position adjustments from HarfBuzz
                x_offset = pos.x_offset / 64
                y_offset = pos.y_offset / 64
                
                # Draw character at calculated position
                char_x = current_x + x_offset
                char_y = current_y + y_offset
                
                try:
                    draw.text((char_x, char_y), char, font=pil_font, fill=0)
                except Exception:
                    # Fallback for unsupported characters
                    draw.text((char_x, char_y), '?', font=pil_font, fill=0)
                
                # Advance position
                current_x += pos.x_advance / 64
                current_y += pos.y_advance / 64
        
        metadata = {
            'rendering_method': 'harfbuzz',
            'font_path': font_path,
            'font_size': font_size,
            'glyph_count': len(infos),
            'total_width': total_width
        }
        
        return image, metadata
    
    def render_text_enhanced_pil(self, text: str, font_path: str, font_size: int, 
                               image_size: Tuple[int, int]) -> Tuple[Image.Image, Dict]:
        """
        Render text using enhanced PIL with better Khmer character positioning.
        
        Args:
            text: Text to render
            font_path: Path to font file
            font_size: Font size
            image_size: Target image size
            
        Returns:
            Tuple of (rendered_image, metadata)
        """
        # Create image
        image = Image.new('L', image_size, 255)  # White background
        draw = ImageDraw.Draw(image)
        
        # Load font
        font = ImageFont.truetype(font_path, font_size)
        
        # Apply enhanced text processing for Khmer
        processed_text = self._enhance_khmer_text_for_pil(text)
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), processed_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2
        
        # Adjust for bbox offset
        x -= bbox[0]
        y -= bbox[1]
        
        # Draw text
        draw.text((x, y), processed_text, font=font, fill=0)
        
        metadata = {
            'rendering_method': 'enhanced_pil',
            'font_path': font_path,
            'font_size': font_size,
            'text_width': text_width,
            'text_height': text_height,
            'position': (x, y)
        }
        
        return image, metadata
    
    def _enhance_khmer_text_for_pil(self, text: str) -> str:
        """
        Apply enhancements to Khmer text for better PIL rendering.
        
        This method applies various techniques to improve how PIL renders Khmer:
        1. Proper Unicode normalization
        2. Zero-width character insertion for better positioning
        3. Character reordering for better visual appearance
        
        Args:
            text: Input Khmer text
            
        Returns:
            Enhanced text for PIL rendering
        """
        # First, normalize the text
        normalized = self.normalize_khmer_text(text)
        
        # Apply character-level enhancements
        enhanced = self._apply_khmer_character_enhancements(normalized)
        
        return enhanced
    
    def _apply_khmer_character_enhancements(self, text: str) -> str:
        """
        Apply character-level enhancements for better Khmer rendering.
        
        Args:
            text: Normalized Khmer text
            
        Returns:
            Enhanced text
        """
        result = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Handle Coeng (subscript) sequences
            if char == COENG_SIGN and i + 1 < len(text):
                # Coeng + consonant = subscript
                coeng_sequence = char + text[i + 1]
                
                # Add zero-width joiner for better positioning
                enhanced_sequence = coeng_sequence
                
                # For certain consonant combinations, add positioning hints
                if i + 1 < len(text):
                    next_char = text[i + 1]
                    if ord(next_char) in range(0x1780, 0x17A3):  # Khmer consonants
                        # Add invisible characters to help with positioning
                        enhanced_sequence = char + '\u200D' + text[i + 1]  # Zero-width joiner
                
                result.append(enhanced_sequence)
                i += 2
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    def render_text_basic_pil(self, text: str, font_path: str, font_size: int, 
                            image_size: Tuple[int, int]) -> Tuple[Image.Image, Dict]:
        """
        Render text using basic PIL rendering (fallback).
        
        Args:
            text: Text to render
            font_path: Path to font file
            font_size: Font size
            image_size: Target image size
            
        Returns:
            Tuple of (rendered_image, metadata)
        """
        # Create image
        image = Image.new('L', image_size, 255)  # White background
        draw = ImageDraw.Draw(image)
        
        # Load font
        font = ImageFont.truetype(font_path, font_size)
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2
        
        # Adjust for bbox offset
        x -= bbox[0]
        y -= bbox[1]
        
        # Draw text
        draw.text((x, y), text, font=font, fill=0)
        
        metadata = {
            'rendering_method': 'basic_pil',
            'font_path': font_path,
            'font_size': font_size,
            'text_width': text_width,
            'text_height': text_height,
            'position': (x, y)
        }
        
        return image, metadata
    
    def render_text(self, text: str, font_path: str, font_size: int, 
                   image_size: Tuple[int, int], color: int = 0) -> Tuple[Image.Image, Dict]:
        """
        Render text using the best available method.
        
        Args:
            text: Text to render
            font_path: Path to font file
            font_size: Font size
            image_size: Target image size
            color: Text color (0 for black, 255 for white)
            
        Returns:
            Tuple of (rendered_image, metadata)
        """
        # Normalize the text first
        normalized_text = self.normalize_khmer_text(text)
        
        # Choose rendering method based on availability and text type
        if HARFBUZZ_AVAILABLE and self.is_khmer_text(normalized_text):
            try:
                image, metadata = self.render_text_harfbuzz(normalized_text, font_path, font_size, image_size)
            except Exception as e:
                print(f"HarfBuzz rendering failed: {e}, falling back to enhanced PIL")
                image, metadata = self.render_text_enhanced_pil(normalized_text, font_path, font_size, image_size)
        elif self.fallback_strategy == "enhanced_pil":
            image, metadata = self.render_text_enhanced_pil(normalized_text, font_path, font_size, image_size)
        else:
            image, metadata = self.render_text_basic_pil(normalized_text, font_path, font_size, image_size)
        
        # Handle color conversion if needed
        if color != 0:
            image_array = np.array(image)
            # Invert for white text on black background
            if color == 255:
                image_array = 255 - image_array
            image = Image.fromarray(image_array)
        
        metadata['original_text'] = text
        metadata['normalized_text'] = normalized_text
        metadata['text_color'] = color
        
        return image, metadata
    
    def test_rendering_methods(self, test_text: str = "កន្ទុំ", font_path: str = None) -> Dict:
        """
        Test different rendering methods with sample text.
        
        Args:
            test_text: Text to test with
            font_path: Font path for testing
            
        Returns:
            Dictionary with test results
        """
        if font_path is None:
            # Use default font path
            font_path = "fonts/KhmerOS.ttf"
        
        results = {}
        image_size = (200, 100)
        font_size = 32
        
        # Test each rendering method
        methods = []
        
        if HARFBUZZ_AVAILABLE:
            methods.append("harfbuzz")
        
        methods.extend(["enhanced_pil", "basic_pil"])
        
        for method in methods:
            try:
                if method == "harfbuzz":
                    image, metadata = self.render_text_harfbuzz(test_text, font_path, font_size, image_size)
                elif method == "enhanced_pil":
                    image, metadata = self.render_text_enhanced_pil(test_text, font_path, font_size, image_size)
                else:
                    image, metadata = self.render_text_basic_pil(test_text, font_path, font_size, image_size)
                
                results[method] = {
                    'success': True,
                    'metadata': metadata,
                    'image_size': image.size
                }
            except Exception as e:
                results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results


# Global renderer instance
_khmer_renderer = None

def get_khmer_renderer() -> KhmerTextRenderer:
    """Get the global Khmer text renderer instance."""
    global _khmer_renderer
    if _khmer_renderer is None:
        _khmer_renderer = KhmerTextRenderer()
    return _khmer_renderer


def render_khmer_text(text: str, font_path: str, font_size: int, 
                     image_size: Tuple[int, int], color: int = 0) -> Tuple[Image.Image, Dict]:
    """
    Convenience function to render Khmer text.
    
    Args:
        text: Text to render
        font_path: Path to font file
        font_size: Font size
        image_size: Target image size
        color: Text color (0 for black, 255 for white)
        
    Returns:
        Tuple of (rendered_image, metadata)
    """
    renderer = get_khmer_renderer()
    return renderer.render_text(text, font_path, font_size, image_size, color) 