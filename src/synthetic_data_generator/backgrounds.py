"""
Background generation utilities for synthetic data.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, List, Optional
import random
import math


class BackgroundGenerator:
    """
    Generates diverse backgrounds for synthetic text images.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (128, 64)):
        """
        Initialize the background generator.
        
        Args:
            image_size: Target image size as (width, height)
        """
        self.image_size = image_size
        self.width, self.height = image_size
    
    def generate_solid_color(self, color: Optional[Tuple[int, int, int]] = None) -> Image.Image:
        """
        Generate a solid color background.
        
        Args:
            color: RGB color tuple, if None will use random color
            
        Returns:
            PIL Image with solid color background
        """
        if color is None:
            # Generate random light colors that work well with dark text
            color = tuple(np.random.randint(200, 256, 3))
        
        return Image.new('RGB', self.image_size, color)
    
    def generate_gradient(self, 
                         start_color: Optional[Tuple[int, int, int]] = None,
                         end_color: Optional[Tuple[int, int, int]] = None,
                         direction: str = 'horizontal') -> Image.Image:
        """
        Generate a gradient background.
        
        Args:
            start_color: Starting RGB color
            end_color: Ending RGB color
            direction: 'horizontal', 'vertical', or 'diagonal'
            
        Returns:
            PIL Image with gradient background
        """
        if start_color is None:
            start_color = tuple(np.random.randint(180, 255, 3))
        if end_color is None:
            end_color = tuple(np.random.randint(180, 255, 3))
        
        image = Image.new('RGB', self.image_size)
        draw = ImageDraw.Draw(image)
        
        if direction == 'horizontal':
            for x in range(self.width):
                ratio = x / self.width
                color = tuple(int(start_color[i] * (1 - ratio) + end_color[i] * ratio) for i in range(3))
                draw.line([(x, 0), (x, self.height)], fill=color)
        
        elif direction == 'vertical':
            for y in range(self.height):
                ratio = y / self.height
                color = tuple(int(start_color[i] * (1 - ratio) + end_color[i] * ratio) for i in range(3))
                draw.line([(0, y), (self.width, y)], fill=color)
        
        elif direction == 'diagonal':
            for x in range(self.width):
                for y in range(self.height):
                    ratio = (x + y) / (self.width + self.height)
                    color = tuple(int(start_color[i] * (1 - ratio) + end_color[i] * ratio) for i in range(3))
                    draw.point((x, y), fill=color)
        
        return image
    
    def generate_noise_texture(self, 
                              base_color: Optional[Tuple[int, int, int]] = None,
                              noise_intensity: float = 0.1) -> Image.Image:
        """
        Generate a textured background with noise.
        
        Args:
            base_color: Base RGB color
            noise_intensity: Intensity of noise (0.0 to 1.0)
            
        Returns:
            PIL Image with noise texture
        """
        if base_color is None:
            base_color = tuple(np.random.randint(200, 240, 3))
        
        # Create base image
        image = Image.new('RGB', self.image_size, base_color)
        
        # Add noise
        noise = np.random.normal(0, noise_intensity * 255, (self.height, self.width, 3))
        image_array = np.array(image, dtype=np.float32)
        
        # Apply noise and clamp values
        noisy_array = image_array + noise
        noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)
    
    def generate_paper_texture(self) -> Image.Image:
        """
        Generate a paper-like texture background.
        
        Returns:
            PIL Image with paper texture
        """
        # Start with off-white base
        base_colors = [
            (248, 248, 246),  # Off-white
            (245, 245, 240),  # Cream
            (250, 248, 240),  # Ivory
            (255, 253, 240),  # Beige
        ]
        base_color = random.choice(base_colors)
        
        image = Image.new('RGB', self.image_size, base_color)
        
        # Add subtle fiber-like patterns
        draw = ImageDraw.Draw(image)
        
        # Random small spots and lines to simulate paper fibers
        num_fibers = random.randint(20, 50)
        for _ in range(num_fibers):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            
            # Random fiber color (slightly darker than base)
            fiber_color = tuple(max(0, c - random.randint(5, 20)) for c in base_color)
            
            # Draw small spots or short lines
            if random.random() < 0.7:
                # Small spot
                draw.point((x, y), fill=fiber_color)
            else:
                # Short line
                length = random.randint(2, 5)
                angle = random.uniform(0, 2 * math.pi)
                end_x = x + int(length * math.cos(angle))
                end_y = y + int(length * math.sin(angle))
                draw.line([(x, y), (end_x, end_y)], fill=fiber_color)
        
        # Apply subtle blur to make it more realistic
        image = image.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        return image
    
    def generate_subtle_pattern(self, pattern_type: str = 'dots') -> Image.Image:
        """
        Generate a background with subtle patterns.
        
        Args:
            pattern_type: 'dots', 'lines', or 'grid'
            
        Returns:
            PIL Image with pattern background
        """
        # Light background color
        bg_color = tuple(np.random.randint(230, 250, 3))
        image = Image.new('RGB', self.image_size, bg_color)
        draw = ImageDraw.Draw(image)
        
        # Pattern color (slightly darker)
        pattern_color = tuple(max(0, c - random.randint(10, 30)) for c in bg_color)
        
        if pattern_type == 'dots':
            spacing = random.randint(10, 20)
            for x in range(0, self.width, spacing):
                for y in range(0, self.height, spacing):
                    if random.random() < 0.3:  # Sparse dots
                        draw.point((x, y), fill=pattern_color)
        
        elif pattern_type == 'lines':
            spacing = random.randint(15, 25)
            if random.random() < 0.5:  # Horizontal lines
                for y in range(0, self.height, spacing):
                    draw.line([(0, y), (self.width, y)], fill=pattern_color)
            else:  # Vertical lines
                for x in range(0, self.width, spacing):
                    draw.line([(x, 0), (x, self.height)], fill=pattern_color)
        
        elif pattern_type == 'grid':
            spacing = random.randint(20, 30)
            for x in range(0, self.width, spacing):
                draw.line([(x, 0), (x, self.height)], fill=pattern_color)
            for y in range(0, self.height, spacing):
                draw.line([(0, y), (self.width, y)], fill=pattern_color)
        
        return image
    
    def generate_random_background(self) -> Image.Image:
        """
        Generate a random background using one of the available methods.
        
        Returns:
            PIL Image with random background
        """
        background_types = [
            'solid',
            'gradient_horizontal',
            'gradient_vertical', 
            'gradient_diagonal',
            'noise',
            'paper',
            'dots',
            'lines',
            'grid'
        ]
        
        bg_type = random.choice(background_types)
        
        if bg_type == 'solid':
            return self.generate_solid_color()
        elif bg_type.startswith('gradient'):
            direction = bg_type.split('_')[1] if '_' in bg_type else 'horizontal'
            return self.generate_gradient(direction=direction)
        elif bg_type == 'noise':
            return self.generate_noise_texture()
        elif bg_type == 'paper':
            return self.generate_paper_texture()
        else:  # pattern types
            return self.generate_subtle_pattern(bg_type)
    
    def get_optimal_text_color(self, background: Image.Image) -> Tuple[int, int, int]:
        """
        Determine optimal text color based on background brightness.
        
        Args:
            background: Background image
            
        Returns:
            RGB color tuple for text
        """
        # Calculate average brightness
        grayscale = background.convert('L')
        avg_brightness = np.array(grayscale).mean()
        
        # If background is light, use dark text; if dark, use light text
        if avg_brightness > 127:
            # Dark text colors for light backgrounds
            colors = [
                (0, 0, 0),        # Black
                (64, 64, 64),     # Dark gray
                (32, 32, 32),     # Very dark gray
                (96, 96, 96),     # Medium dark gray
            ]
        else:
            # Light text colors for dark backgrounds
            colors = [
                (255, 255, 255),  # White
                (240, 240, 240),  # Light gray
                (220, 220, 220),  # Medium light gray
                (200, 200, 200),  # Light gray
            ]
        
        return random.choice(colors) 