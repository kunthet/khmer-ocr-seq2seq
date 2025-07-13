"""
Text to image renderer for Khmer OCR synthetic data generation.
Uses Tesseract's text2image utility to render Khmer text.
"""
import os
import subprocess
import tempfile
from typing import List, Tuple, Optional
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import random

# Import Khmer text processing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'khtext'))
from khnormal_fast import khnormal

# Import background generation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'synthetic_data_generator'))
from backgrounds import BackgroundGenerator

# Import advanced Khmer text renderer
try:
    # Try importing from the same data directory
    from .khmer_text_renderer import KhmerTextRenderer, render_khmer_text
    ADVANCED_KHMER_RENDERER_AVAILABLE = True
    print("✅ Advanced Khmer text renderer available")
except ImportError:
    try:
        # Try importing from current directory
        sys.path.append(os.path.dirname(__file__))
        from khmer_text_renderer import KhmerTextRenderer, render_khmer_text
        ADVANCED_KHMER_RENDERER_AVAILABLE = True
        print("✅ Advanced Khmer text renderer available")
    except ImportError:
        ADVANCED_KHMER_RENDERER_AVAILABLE = False
        print("⚠️ Advanced Khmer text renderer not available - using basic PIL rendering")


class TextRenderer:
    """
    Renders Khmer text to images using multiple approaches.
    
    Primary method uses advanced Khmer text renderer with proper subscript handling.
    Fallback methods include text2image from Tesseract and PIL rendering.
    """
    
    def __init__(
        self, 
        fonts: List[str] = None, 
        image_height: int = 32,
        use_tesseract: bool = True
    ):
        """
        Initialize text renderer.
        
        Args:
            fonts: List of font paths or font names for Khmer text
            image_height: Fixed height for output images (32px as per PRD)
            use_tesseract: Whether to use text2image or PIL fallback
        """
        self.image_height = image_height
        self.use_tesseract = use_tesseract
        self.font_size_factor = 0.5
        
        # Default Khmer fonts from fonts/ directory
        if fonts is None:
            fonts = self._get_default_font_paths()
        
        self.fonts = fonts
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize advanced Khmer text renderer
        if ADVANCED_KHMER_RENDERER_AVAILABLE:
            self.khmer_renderer = KhmerTextRenderer()
            print("✅ Advanced Khmer text renderer initialized")
        else:
            self.khmer_renderer = None
        
        # Check text2image availability
        self._check_text2image_availability()
        
        # For PIL fallback, try to load fonts
        self._load_pil_fonts()
        
        # Initialize background generator
        self.background_generator = BackgroundGenerator((256, 32))  # Variable width, fixed height
    
    def _get_default_font_paths(self) -> List[str]:
        """Get default Khmer font paths from fonts/ directory."""
        font_dir = "fonts"
        default_fonts = [
            "KhmerOS.ttf",
            "KhmerOSsiemreap.ttf",
            "KhmerOSbattambang.ttf",
            "KhmerOSbokor.ttf",
            "KhmerOSmuol.ttf",
            "KhmerOSmuollight.ttf",
            "KhmerOSmetalchrieng.ttf",
            "KhmerOSfasthand.ttf"
        ]
        
        # Get full paths and filter to existing fonts
        font_paths = []
        for font_name in default_fonts:
            font_path = os.path.join(font_dir, font_name)
            if os.path.exists(font_path):
                font_paths.append(font_path)
        
        if not font_paths:
            print(f"Warning: No Khmer fonts found in {font_dir}/ directory")
            print("Using system default fonts - text rendering quality may be poor for Khmer text")
            # Fallback to system font names
            font_paths = [
                "Khmer OS",
                "Khmer OS Siemreap",
                "Khmer OS Battambang", 
                "Khmer OS Muol Light",
                "Hanuman",
                "Nokora"
            ]
        
        return font_paths
    
    def _check_text2image_availability(self):
        """Check if text2image utility is available."""
        try:
            result = subprocess.run(
                ["text2image", "--help"], 
                capture_output=True, 
                text=True
            )
            self.text2image_available = result.returncode == 0
        except FileNotFoundError:
            self.text2image_available = False
        
        if not self.text2image_available:
            print("Warning: text2image not found. Using PIL fallback.")
            self.use_tesseract = False
    
    def _load_pil_fonts(self):
        """Load available PIL fonts for fallback rendering."""
        self.pil_fonts = []
        
        for font_path in self.fonts:
            font_loaded = False
            
            # If it's a file path, try to load it directly
            if os.path.exists(font_path):
                try:
                    # Calculate font size for target height
                    font_size = int(self.image_height * self.font_size_factor)  # 50% of height
                    font = ImageFont.truetype(font_path, font_size)
                    font_name = os.path.basename(font_path).replace('.ttf', '')
                    self.pil_fonts.append((font_name, font))
                    font_loaded = True
                    print(f"Loaded font: {font_name} from {font_path}")
                except Exception as e:
                    print(f"Warning: Could not load font {font_path}: {e}")
                    continue
            
            # If not a file path, try system font locations
            if not font_loaded:
                font_name = font_path
                system_paths = [
                    "/System/Library/Fonts/",  # macOS
                    "/usr/share/fonts/",       # Linux
                    "C:/Windows/Fonts/",       # Windows
                    "/Library/Fonts/",         # macOS user fonts
                ]
                
                for base_path in system_paths:
                    if os.path.exists(base_path):
                        for ext in ['.ttf', '.otf', '.ttc']:
                            font_file = os.path.join(base_path, font_name + ext)
                            if os.path.exists(font_file):
                                try:
                                    font_size = int(self.image_height * self.font_size_factor)
                                    font = ImageFont.truetype(font_file, font_size)
                                    self.pil_fonts.append((font_name, font))
                                    font_loaded = True
                                    print(f"Loaded system font: {font_name} from {font_file}")
                                    break
                                except Exception as e:
                                    print(f"Warning: Could not load system font {font_file}: {e}")
                                    continue
                    
                    if font_loaded:
                        break
        
        # Fallback to default font if no fonts loaded
        if not self.pil_fonts:
            try:
                default_font = ImageFont.load_default()
                self.pil_fonts.append(("default", default_font))
                print("Using default system font as fallback")
            except:
                self.pil_fonts.append(("default", None))
                print("Warning: Could not load any fonts, including default")
    
    def render_text_tesseract(self, text: str, font: str = None) -> Image.Image:
        """
        Render text using Tesseract's text2image utility.
        
        Args:
            text: Khmer text to render
            font: Font name to use (optional)
            
        Returns:
            PIL Image object
        """
        if not self.text2image_available:
            raise RuntimeError("text2image utility not available")
        
        # Create temporary files
        text_file = os.path.join(self.temp_dir, "input.txt")
        output_file = os.path.join(self.temp_dir, "output.tif")
        
        # Write text to file
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Build text2image command
        cmd = [
            "text2image",
            "--text", text_file,
            "--outputbase", output_file.replace('.tif', ''),
            "--fontsize", str(int(self.image_height * 0.8)),
            "--xsize", "2000",  # Max width, will be cropped
            "--ysize", str(self.image_height + 10),  # Slightly larger for padding
            "--margin", "5"
        ]
        
        # Add font if specified
        if font:
            # If font is a file path, use just the font name
            if os.path.exists(font):
                font_name = os.path.basename(font).replace('.ttf', '')
                cmd.extend(["--font", font_name])
            else:
                cmd.extend(["--font", font])
        
        try:
            # Run text2image command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: text2image failed: {result.stderr}")
                raise RuntimeError(f"text2image failed: {result.stderr}")
            
            # Load generated image
            output_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.tif')]
            if not output_files:
                raise RuntimeError("No output file generated by text2image")
            
            image_path = os.path.join(self.temp_dir, output_files[0])
            image = Image.open(image_path).convert('L')
            
            # Resize to target height and crop to content
            image = self._resize_to_height(image, self.image_height)
            image = self._crop_to_content_preserve_height(image)
            
            return image
            
        except Exception as e:
            print(f"Warning: text2image rendering failed: {e}")
            # Fall back to PIL rendering
            return self.render_text_pil(text, font)
    
    def render_text_pil(self, text: str, font_name: str = None) -> Image.Image:
        """
        Render text using PIL as fallback.
        
        Args:
            text: Text to render
            font_name: Font name to use (optional)
            
        Returns:
            PIL Image object
        """
        if not self.pil_fonts:
            raise RuntimeError("No fonts available for PIL rendering")
        
        # Select font
        font = None
        if font_name:
            # Extract font name from path if it's a path
            if os.path.exists(font_name):
                # It's a file path, extract the font name
                base_name = os.path.basename(font_name).replace('.ttf', '').replace('.otf', '')
            else:
                # It's already a font name
                base_name = font_name
            
            # Try to match font by name
            for name, f in self.pil_fonts:
                if name == base_name:
                    font = f
                    break
        
        if font is None and self.pil_fonts:
            font = self.pil_fonts[0][1]  # Use first available font
        
        # Calculate text size
        if font is None:
            # Use default font size estimation
            text_width = len(text) * 12
            text_height = 16
        else:
            try:
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # Fallback if getbbox fails
                text_width = len(text) * 12
                text_height = 16
        
        # Create image with some padding
        img_width = max(text_width + 20, 50)
        img_height = self.image_height
        
        # Create white background
        image = Image.new('L', (img_width, img_height), color=255)
        draw = ImageDraw.Draw(image)
        
        # Calculate position to center text vertically
        y_offset = (img_height - text_height) // 2
        
        # Draw text in black
        draw.text((10, y_offset), text, fill=0, font=font)
        
        # Crop to content horizontally but maintain full height
        image = self._crop_to_content_preserve_height(image)
        
        return image
    
    def render_text_advanced_khmer(self, text: str, font_path: str = None) -> Image.Image:
        """
        Render text using advanced Khmer text renderer with proper subscript handling.
        
        Args:
            text: Khmer text to render
            font_path: Path to font file (optional)
            
        Returns:
            PIL Image object with properly rendered Khmer text
        """
        if not ADVANCED_KHMER_RENDERER_AVAILABLE:
            raise RuntimeError("Advanced Khmer text renderer not available")
        
        # Normalize the text
        normalized_text = khnormal(text)
        
        # Choose font
        if font_path is None:
            font_path = self.fonts[0] if self.fonts else "fonts/KhmerOS.ttf"
        
        # Calculate appropriate font size for target height
        font_size = int(self.image_height * self.font_size_factor)  # 50% of height
        
        # Create a temporary image to measure text dimensions
        temp_image = Image.new('L', (1000, self.image_height), 255)
        temp_draw = ImageDraw.Draw(temp_image)
        
        # Try to load font for measuring
        try:
            temp_font = ImageFont.truetype(font_path, font_size)
            bbox = temp_draw.textbbox((0, 0), normalized_text, font=temp_font)
            # Add padding to prevent text cropping
            estimated_width = bbox[2] - bbox[0] + 50
        except:
            # Fallback width estimation
            estimated_width = len(normalized_text) * font_size // 2
        
        # Ensure minimum width
        estimated_width = max(estimated_width, 50)
        
        # Render using advanced Khmer renderer
        rendered_image, metadata = self.khmer_renderer.render_text(
            normalized_text, 
            font_path, 
            font_size, 
            (estimated_width, self.image_height)
        )
        
        # Crop to content while preserving height
        final_image = self._crop_to_content_preserve_height(rendered_image)
        
        return final_image
    
    def render_text(self, text: str, font: str = None) -> Image.Image:
        """
        Render text using the best available method.
        
        Priority order:
        1. Advanced Khmer text renderer (handles subscripts properly)
        2. Tesseract text2image (if available)
        3. PIL fallback
        
        Args:
            text: Text to render
            font: Font name or path to use
            
        Returns:
            PIL Image object
        """
        # Normalize Khmer text
        normalized_text = khnormal(text)
        
        # Try advanced Khmer renderer first for Khmer text
        if ADVANCED_KHMER_RENDERER_AVAILABLE and self._contains_khmer_text(normalized_text):
            try:
                return self.render_text_advanced_khmer(normalized_text, font)
            except Exception as e:
                print(f"Advanced Khmer renderer failed: {e}, falling back to next method")
        
        # Fallback to text2image if available
        if self.use_tesseract and self.text2image_available:
            try:
                return self.render_text_tesseract(normalized_text, font)
            except Exception as e:
                print(f"Tesseract rendering failed: {e}, falling back to PIL")
        
        # Final fallback to PIL
        return self.render_text_pil(normalized_text, font)
    
    def _contains_khmer_text(self, text: str) -> bool:
        """Check if text contains Khmer characters."""
        for char in text:
            code = ord(char)
            # Check for Khmer Unicode ranges
            if (0x1780 <= code <= 0x17FF):  # Khmer script range
                return True
        return False
    
    def render_text_with_background(self, text: str, font: str = None, 
                                   use_background: bool = True) -> Image.Image:
        """
        Render text with optional background generation.
        
        Args:
            text: Text to render
            font: Font name or path to use
            use_background: Whether to use background generation
            
        Returns:
            PIL Image object
        """
        # Normalize Khmer text
        normalized_text = khnormal(text)
        
        if use_background and hasattr(self, 'background_generator'):
            # Use advanced Khmer renderer with background if available
            if ADVANCED_KHMER_RENDERER_AVAILABLE and self._contains_khmer_text(normalized_text):
                try:
                    return self._render_text_with_background_advanced(normalized_text, font)
                except Exception as e:
                    print(f"Advanced Khmer renderer with background failed: {e}, falling back")
            
            # Fallback to original background method
            return self._render_text_with_background_original(normalized_text, font)
        else:
            # No background, use standard rendering
            return self.render_text(normalized_text, font)
    
    def _render_text_with_background_advanced(self, text: str, font_path: str = None) -> Image.Image:
        """
        Render text with background using advanced Khmer renderer.
        
        Args:
            text: Normalized Khmer text
            font_path: Path to font file
            
        Returns:
            PIL Image with text and background
        """
        # Choose font
        if font_path is None:
            font_path = self.fonts[0] if self.fonts else "fonts/KhmerOS.ttf"
        
        # Calculate appropriate font size
        font_size = int(self.image_height * self.font_size_factor)
        
        # Estimate width needed
        try:
            temp_font = ImageFont.truetype(font_path, font_size)
            temp_image = Image.new('L', (1000, self.image_height), 255)
            temp_draw = ImageDraw.Draw(temp_image)
            bbox = temp_draw.textbbox((0, 0), text, font=temp_font)
            # Add extra padding for background variant to prevent text cropping
            estimated_width = bbox[2] - bbox[0] + 60
        except:
            estimated_width = len(text) * font_size // 2
        
        estimated_width = max(estimated_width, 100)  # Minimum width
        
        # Generate background
        background = self.background_generator.generate_random_background()
        background = background.resize((estimated_width, self.image_height))
        
        # Get optimal text color
        text_color = self.background_generator.get_optimal_text_color(background)
        text_color_value = text_color[0] if isinstance(text_color, tuple) else text_color
        
        # Render text using advanced renderer
        text_image, metadata = self.khmer_renderer.render_text(
            text, 
            font_path, 
            font_size, 
            (estimated_width, self.image_height),
            color=255 - text_color_value  # Invert color if needed
        )
        
        # Convert background to grayscale if needed
        if background.mode != 'L':
            background = background.convert('L')
        
        # Blend text with background
        if text_color_value < 128:  # Dark text
            # Use text as mask (black text on white background)
            text_array = np.array(text_image)
            background_array = np.array(background)
            
            # Create mask: text pixels are dark (low values)
            mask = text_array < 128
            result_array = background_array.copy()
            result_array[mask] = text_color_value
            
            final_image = Image.fromarray(result_array)
        else:  # Light text
            # Use inverted text as mask
            text_array = np.array(text_image)
            background_array = np.array(background)
            
            # Create mask: text pixels are light (high values)
            mask = text_array > 128
            result_array = background_array.copy()
            result_array[mask] = text_color_value
            
            final_image = Image.fromarray(result_array)
        
        # Crop to content while preserving height
        final_image = self._crop_to_content_preserve_height(final_image)
        
        return final_image
    
    def _render_text_with_background_original(self, text: str, font: str = None) -> Image.Image:
        """
        Original background rendering method as fallback.
        
        Args:
            text: Text to render
            font: Font name or path
            
        Returns:
            PIL Image with text and background
        """
        # First render the text normally
        text_image = self.render_text(text, font)
        
        # Get image dimensions
        width, height = text_image.size
        
        # Generate background that matches text dimensions
        background = self.background_generator.generate_random_background()
        background = background.resize((width, height))
        
        # Get optimal text color for this background
        text_color = self.background_generator.get_optimal_text_color(background)
        
        # Convert everything to grayscale arrays
        text_array = np.array(text_image.convert('L'))
        background_array = np.array(background.convert('L'))
        
        # Create the final image
        # Text pixels are dark (low values) in the text_image
        # Background pixels are light (high values) in the text_image
        
        # Determine text color value
        if isinstance(text_color, tuple):
            text_color_value = text_color[0]  # Use first channel for grayscale
        else:
            text_color_value = text_color
        
        # Create mask where text should be drawn
        text_mask = text_array < 128  # Dark areas are text
        
        # Apply text to background
        result_array = background_array.copy()
        result_array[text_mask] = text_color_value
        
        # Convert back to PIL Image
        final_image = Image.fromarray(result_array, mode='L')
        
        return final_image
    
    def _resize_to_height(self, image: Image.Image, target_height: int) -> Image.Image:
        """Resize image to target height while maintaining aspect ratio."""
        width, height = image.size
        if height == target_height:
            return image
        
        # Calculate new width maintaining aspect ratio
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        return image.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    def _crop_to_content_preserve_height(self, image: Image.Image) -> Image.Image:
        """Crop image horizontally but preserve the target height."""
        # Convert to numpy for easier processing
        img_array = np.array(image)
        
        # Find bounding box of non-white pixels
        non_white_coords = np.where(img_array < 255)
        
        if len(non_white_coords[0]) == 0:
            # No content, return minimal image with target height
            return Image.new('L', (10, self.image_height), color=255)
        
        min_x, max_x = non_white_coords[1].min(), non_white_coords[1].max()
        
        # Add small padding horizontally
        padding = 5
        min_x = max(0, min_x - padding)
        max_x = min(img_array.shape[1], max_x + padding)
        
        # Crop horizontally but keep full height
        cropped_array = img_array[:, min_x:max_x+1]
        
        # Ensure the height is exactly what we want
        if cropped_array.shape[0] != self.image_height:
            cropped_image = Image.fromarray(cropped_array)
            cropped_image = self._resize_to_height(cropped_image, self.image_height)
            return cropped_image
        
        return Image.fromarray(cropped_array)
    
    def batch_render(self, texts: List[str], fonts: List[str] = None) -> List[Image.Image]:
        """
        Render multiple texts efficiently.
        
        Args:
            texts: List of texts to render
            fonts: List of fonts to use (cycled through texts)
            
        Returns:
            List of PIL Image objects
        """
        images = []
        
        for i, text in enumerate(texts):
            # Cycle through fonts if provided
            font = None
            if fonts:
                font = fonts[i % len(fonts)]
            
            image = self.render_text(text, font)
            images.append(image)
        
        return images
    
    def get_available_fonts(self) -> List[str]:
        """Get list of available fonts."""
        if self.use_tesseract:
            return self.fonts
        else:
            return [name for name, _ in self.pil_fonts]
    
    def __del__(self):
        """Clean up temporary directory."""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass 