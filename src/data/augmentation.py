"""
Data augmentation pipeline for Khmer OCR synthetic data.
Implements augmentations as specified in the PRD:
- Gaussian blur
- Morphological operations  
- Noise injection
- Text concatenation
"""
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter
from typing import List, Tuple, Union, Optional
import math


class DataAugmentation:
    """
    Data augmentation pipeline for synthetic Khmer text images.
    
    Implements the augmentation strategies defined in the PRD:
    - Gaussian blur with sigma range 0-1.5
    - Morphological operations (dilation/erosion) with kernel sizes 2-3
    - Noise injection with blob radius 1-3 and background intensity 0.1
    - Text concatenation with probability 0.5
    """
    
    def __init__(
        self,
        blur_prob: float = 0.5,
        morph_prob: float = 0.5,
        noise_prob: float = 0.5,
        concat_prob: float = 0.5,
        blur_sigma_range: Tuple[float, float] = (0.0, 1.5),
        morph_kernel_sizes: List[int] = [2, 3],
        noise_blob_radius_range: Tuple[int, int] = (1, 3),
        noise_background_intensity: float = 0.1
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            blur_prob: Probability of applying Gaussian blur
            morph_prob: Probability of applying morphological operations
            noise_prob: Probability of applying noise injection
            concat_prob: Probability of concatenating multiple texts
            blur_sigma_range: Range for Gaussian blur sigma values
            morph_kernel_sizes: Available kernel sizes for morphological operations
            noise_blob_radius_range: Range for noise blob radius
            noise_background_intensity: Background intensity for noise
        """
        self.blur_prob = blur_prob
        self.morph_prob = morph_prob
        self.noise_prob = noise_prob
        self.concat_prob = concat_prob
        
        self.blur_sigma_range = blur_sigma_range
        self.morph_kernel_sizes = morph_kernel_sizes
        self.noise_blob_radius_range = noise_blob_radius_range
        self.noise_background_intensity = noise_background_intensity
        
        # Morphological operations
        self.morph_operations = ['dilation', 'erosion']
    
    def apply_gaussian_blur(self, image: Image.Image) -> Image.Image:
        """
        Apply Gaussian blur with random sigma from specified range.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Blurred PIL Image
        """
        if random.random() > self.blur_prob:
            return image
        
        # Random sigma from specified range
        sigma = random.uniform(*self.blur_sigma_range)
        
        if sigma == 0:
            return image
        
        # Apply Gaussian blur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return blurred
    
    def apply_morphological_operations(self, image: Image.Image) -> Image.Image:
        """
        Apply morphological operations (dilation or erosion).
        
        Args:
            image: Input PIL Image
            
        Returns:
            Processed PIL Image
        """
        if random.random() > self.morph_prob:
            return image
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Random kernel size and operation
        kernel_size = random.choice(self.morph_kernel_sizes)
        operation = random.choice(self.morph_operations)
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply operation
        if operation == 'dilation':
            result = cv2.dilate(img_array, kernel, iterations=1)
        else:  # erosion
            result = cv2.erode(img_array, kernel, iterations=1)
        
        return Image.fromarray(result)
    
    def apply_noise_injection(self, image: Image.Image) -> Image.Image:
        """
        Apply noise injection with random blob patterns.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Noisy PIL Image
        """
        if random.random() > self.noise_prob:
            return image
        
        img_array = np.array(image).astype(np.float32)
        height, width = img_array.shape
        
        # Random blob parameters
        blob_radius = random.randint(*self.noise_blob_radius_range)
        num_blobs = random.randint(1, 5)  # Random number of noise blobs
        
        # Create noise mask
        noise_mask = np.zeros((height, width), dtype=np.float32)
        
        for _ in range(num_blobs):
            # Random position for blob with bounds checking
            min_x = max(blob_radius, 0)
            max_x = max(min_x + 1, width - blob_radius)
            min_y = max(blob_radius, 0)
            max_y = max(min_y + 1, height - blob_radius)
            
            if max_x <= min_x:
                center_x = width // 2
            else:
                center_x = random.randint(min_x, max_x - 1)
                
            if max_y <= min_y:
                center_y = height // 2
            else:
                center_y = random.randint(min_y, max_y - 1)
            
            # Create circular blob
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= blob_radius ** 2
            
            # Random intensity for this blob
            blob_intensity = random.uniform(0.1, 0.8)
            noise_mask[mask] = blob_intensity
        
        # Apply background noise
        background_noise = np.random.normal(0, self.noise_background_intensity, (height, width))
        background_noise = np.clip(background_noise, -0.2, 0.2)
        
        # Combine original image with noise
        noisy_image = img_array + (noise_mask * 255) + (background_noise * 255)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_image)
    
    def concatenate_images(self, images: List[Image.Image]) -> Image.Image:
        """
        Concatenate multiple images horizontally.
        
        Args:
            images: List of PIL Images to concatenate
            
        Returns:
            Concatenated PIL Image
        """
        if len(images) <= 1:
            return images[0] if images else Image.new('L', (10, 32), color=255)
        
        # Get consistent height (should be 32 for all images)
        target_height = images[0].height
        
        # Resize all images to same height if needed
        resized_images = []
        for img in images:
            if img.height != target_height:
                aspect_ratio = img.width / img.height
                new_width = int(target_height * aspect_ratio)
                img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            resized_images.append(img)
        
        # Calculate total width
        total_width = sum(img.width for img in resized_images)
        
        # Create new image
        concatenated = Image.new('L', (total_width, target_height), color=255)
        
        # Paste images
        x_offset = 0
        for img in resized_images:
            concatenated.paste(img, (x_offset, 0))
            x_offset += img.width
        
        return concatenated
    
    def should_concatenate(self) -> bool:
        """Check if concatenation should be applied."""
        return random.random() < self.concat_prob
    
    def apply_augmentations(self, image: Image.Image) -> Image.Image:
        """
        Apply all augmentations to a single image in sequence.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Augmented PIL Image
        """
        # Apply augmentations in sequence
        augmented = image
        
        # 1. Gaussian blur
        augmented = self.apply_gaussian_blur(augmented)
        
        # 2. Morphological operations
        augmented = self.apply_morphological_operations(augmented)
        
        # 3. Noise injection
        augmented = self.apply_noise_injection(augmented)
        
        return augmented
    
    def apply_batch_augmentations(
        self, 
        images: List[Image.Image], 
        texts: List[str] = None
    ) -> Tuple[List[Image.Image], List[str]]:
        """
        Apply augmentations to a batch of images, including concatenation.
        
        Args:
            images: List of PIL Images
            texts: Corresponding text labels (optional)
            
        Returns:
            Tuple of (augmented_images, updated_texts)
        """
        if not images:
            return [], []
        
        augmented_images = []
        augmented_texts = texts[:] if texts else [f"text_{i}" for i in range(len(images))]
        
        i = 0
        while i < len(images):
            current_image = images[i]
            current_text = augmented_texts[i] if texts else f"text_{i}"
            
            # Check if we should concatenate
            if self.should_concatenate() and i < len(images) - 1:
                # Concatenate 2-3 images
                num_to_concat = random.randint(2, min(3, len(images) - i))
                images_to_concat = images[i:i + num_to_concat]
                texts_to_concat = augmented_texts[i:i + num_to_concat] if texts else [f"text_{j}" for j in range(i, i + num_to_concat)]
                
                # Concatenate images
                concatenated_image = self.concatenate_images(images_to_concat)
                
                # Apply augmentations
                final_image = self.apply_augmentations(concatenated_image)
                
                # Combine texts
                combined_text = "".join(texts_to_concat) if texts else f"concatenated_{i}"
                
                augmented_images.append(final_image)
                if texts:
                    augmented_texts = augmented_texts[:i] + [combined_text] + augmented_texts[i + num_to_concat:]
                
                i += num_to_concat
            else:
                # Apply augmentations to single image
                final_image = self.apply_augmentations(current_image)
                augmented_images.append(final_image)
                i += 1
        
        # Ensure we return the correct number of texts
        final_texts = augmented_texts[:len(augmented_images)]
        
        return augmented_images, final_texts
    
    def get_augmentation_stats(self) -> dict:
        """Get current augmentation parameters."""
        return {
            'blur_prob': self.blur_prob,
            'morph_prob': self.morph_prob,
            'noise_prob': self.noise_prob,
            'concat_prob': self.concat_prob,
            'blur_sigma_range': self.blur_sigma_range,
            'morph_kernel_sizes': self.morph_kernel_sizes,
            'noise_blob_radius_range': self.noise_blob_radius_range,
            'noise_background_intensity': self.noise_background_intensity
        }
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible augmentations."""
        random.seed(seed)
        np.random.seed(seed) 