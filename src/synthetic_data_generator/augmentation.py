"""
Image augmentation utilities for synthetic data generation.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Optional, List
import random
import cv2


class ImageAugmentor:
    """
    Provides various image augmentation techniques for synthetic data.
    """
    
    def __init__(self, 
                 rotation_range: Tuple[float, float] = (-15, 15),
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.01,
                 brightness_range: Tuple[float, float] = (-0.2, 0.2),
                 contrast_range: Tuple[float, float] = (-0.2, 0.2)):
        """
        Initialize the image augmentor.
        
        Args:
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scaling factors
            noise_std: Standard deviation for Gaussian noise
            brightness_range: Range of brightness adjustments
            contrast_range: Range of contrast adjustments
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def rotate_image(self, image: Image.Image, angle: Optional[float] = None) -> Image.Image:
        """
        Rotate the image by a random or specified angle, expanding canvas to fit entire rotated content.
        
        Args:
            image: Input PIL image
            angle: Rotation angle in degrees, if None uses random angle
            
        Returns:
            Rotated PIL image with expanded canvas to preserve all content
        """
        if angle is None:
            angle = random.uniform(*self.rotation_range)
        
        # Determine fill color based on image mode
        if image.mode == 'L':
            fillcolor = 255  # Grayscale white
        elif image.mode == 'RGBA':
            fillcolor = (255, 255, 255, 0) # Transparent white for RGBA
        else:
            fillcolor = (255, 255, 255) # RGB white

        # Rotate with expanding canvas to preserve entire rotated content
        # This prevents cropping and allows the content to be shrunk to fit target dimensions
        rotated = image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=fillcolor)
        return rotated
    
    def scale_image(self, image: Image.Image, 
                   scale_factor: Optional[float] = None,
                   target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Scale the image by a random or specified factor.
        
        Args:
            image: Input PIL image
            scale_factor: Scaling factor, if None uses random factor
            target_size: Target size to fit scaled image into
            
        Returns:
            Scaled PIL image
        """
        if scale_factor is None:
            scale_factor = random.uniform(*self.scale_range)
        
        original_size = image.size
        new_size = (int(original_size[0] * scale_factor), 
                   int(original_size[1] * scale_factor))
        
        scaled = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # If target size is specified, pad or crop to fit
        if target_size:
            scaled = self._fit_to_size(scaled, target_size)
        
        return scaled
    
    def _fit_to_size(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Fit image to target size by padding or cropping.
        
        Args:
            image: Input PIL image
            target_size: Target (width, height)
            
        Returns:
            Fitted PIL image
        """
        target_width, target_height = target_size
        current_width, current_height = image.size
        
        # Create new image with target size and adaptive background color
        if image.mode == 'L':
            fill_color = 255
        elif image.mode == 'RGBA':
            fill_color = (255, 255, 255, 0)
        else:
            fill_color = (255, 255, 255)
            
        new_image = Image.new(image.mode, target_size, fill_color)
        
        # Calculate position to center the image
        x_offset = (target_width - current_width) // 2
        y_offset = (target_height - current_height) // 2
        
        # If image is larger than target, crop it
        if current_width > target_width or current_height > target_height:
            crop_x = max(0, (current_width - target_width) // 2)
            crop_y = max(0, (current_height - target_height) // 2)
            image = image.crop((crop_x, crop_y, 
                              crop_x + min(target_width, current_width),
                              crop_y + min(target_height, current_height)))
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)
        
        # Paste the image onto the new background
        new_image.paste(image, (x_offset, y_offset))
        return new_image
    
    def add_gaussian_noise(self, image: Image.Image, 
                          std: Optional[float] = None) -> Image.Image:
        """
        Add Gaussian noise to the image.
        
        Args:
            image: Input PIL image
            std: Standard deviation of noise, if None uses default
            
        Returns:
            Noisy PIL image
        """
        if std is None:
            std = self.noise_std
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Generate noise
        noise = np.random.normal(0, std * 255, img_array.shape)
        
        # Add noise and clamp values
        noisy_array = img_array + noise
        noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)
    
    def adjust_brightness(self, image: Image.Image, 
                         factor: Optional[float] = None) -> Image.Image:
        """
        Adjust image brightness.
        
        Args:
            image: Input PIL image
            factor: Brightness adjustment factor, if None uses random factor
            
        Returns:
            Brightness-adjusted PIL image
        """
        if factor is None:
            factor = 1.0 + random.uniform(*self.brightness_range)
        else:
            factor = 1.0 + factor
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image: Image.Image, 
                       factor: Optional[float] = None) -> Image.Image:
        """
        Adjust image contrast.
        
        Args:
            image: Input PIL image
            factor: Contrast adjustment factor, if None uses random factor
            
        Returns:
            Contrast-adjusted PIL image
        """
        if factor is None:
            factor = 1.0 + random.uniform(*self.contrast_range)
        else:
            factor = 1.0 + factor
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def apply_blur(self, image: Image.Image, 
                   radius: Optional[float] = None) -> Image.Image:
        """
        Apply Gaussian blur to the image.
        
        Args:
            image: Input PIL image
            radius: Blur radius, if None uses random radius
            
        Returns:
            Blurred PIL image
        """
        if radius is None:
            radius = random.uniform(0.1, 1.0)
        
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def apply_motion_blur(self, image: Image.Image, 
                         size: Optional[int] = None,
                         angle: Optional[float] = None) -> Image.Image:
        """
        Apply motion blur effect.
        
        Args:
            image: Input PIL image
            size: Size of motion blur kernel
            angle: Angle of motion blur in degrees
            
        Returns:
            Motion-blurred PIL image
        """
        if size is None:
            size = random.randint(3, 7)
        if angle is None:
            angle = random.uniform(0, 360)
        
        # Convert to OpenCV format, handling grayscale
        if image.mode == 'L':
            img_array = np.array(image)
        else:
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create motion blur kernel
        kernel = self._get_motion_blur_kernel(size, angle)
        
        # Apply blur
        blurred = cv2.filter2D(img_array, -1, kernel)
        
        # Convert back to PIL, preserving original mode
        if image.mode == 'L':
            return Image.fromarray(blurred)
        else:
            blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
            return Image.fromarray(blurred_rgb)
    
    def _get_motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        """
        Generate motion blur kernel.
        
        Args:
            size: Kernel size
            angle: Blur angle in degrees
            
        Returns:
            Motion blur kernel
        """
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate the line coordinates
        for i in range(size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
        return kernel
    
    def add_speckle_noise(self, image: Image.Image, 
                         intensity: Optional[float] = None) -> Image.Image:
        """
        Add speckle noise to simulate paper texture or printing artifacts.
        
        Args:
            image: Input PIL image
            intensity: Noise intensity (0.0 to 1.0)
            
        Returns:
            PIL image with speckle noise
        """
        if intensity is None:
            intensity = random.uniform(0.01, 0.05)
        
        img_array = np.array(image, dtype=np.float32)
        
        # Generate speckle noise (multiplicative)
        speckle = np.random.normal(1.0, intensity, img_array.shape)
        
        # Apply speckle noise
        noisy_array = img_array * speckle
        noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)
    
    def perspective_transform(self, image: Image.Image, 
                            intensity: Optional[float] = None) -> Image.Image:
        """
        Apply perspective transformation.
        
        Args:
            image: Input PIL image
            intensity: Transformation intensity (0.0 to 1.0)
            
        Returns:
            Transformed PIL image
        """
        if intensity is None:
            intensity = random.uniform(0.01, 0.05)
        
        # Convert to OpenCV format, handling grayscale
        if image.mode == 'L':
            img_array = np.array(image)
            border_value = 255
        else:
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            border_value = (255, 255, 255)

        # Get transformation matrix
        M = self._get_perspective_transform_matrix(image.size, intensity)
        
        # Apply transformation
        transformed = cv2.warpPerspective(img_array, M, (image.width, image.height),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        
        # Convert back to PIL, preserving original mode
        if image.mode == 'L':
            return Image.fromarray(transformed)
        else:
            transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
            return Image.fromarray(transformed_rgb)

    def _get_perspective_transform_matrix(self, size: Tuple[int, int], 
                                          intensity: float) -> np.ndarray:
        """
        Generate the perspective transformation matrix.
        
        Args:
            size: Original image size (width, height)
            intensity: Transformation intensity (0.0 to 1.0)
            
        Returns:
            Perspective transformation matrix
        """
        width, height = size
        variation = int(min(width, height) * intensity)
        
        # Define corner points with slight variations
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        dst_points = np.float32([
            [random.randint(-variation, variation), 
             random.randint(-variation, variation)],
            [width - random.randint(-variation, variation), 
             random.randint(-variation, variation)],
            [width - random.randint(-variation, variation), 
             height - random.randint(-variation, variation)],
            [random.randint(-variation, variation), 
             height - random.randint(-variation, variation)]
        ])
        
        return cv2.getPerspectiveTransform(src_points, dst_points)
    
    def augment_image(self, image: Image.Image, 
                     augmentations: Optional[List[str]] = None) -> Image.Image:
        """
        Apply a random set of augmentations to the image.
        
        Args:
            image: Input PIL image
            augmentations: List of augmentation names to apply, if None uses random selection
            
        Returns:
            Augmented PIL image
        """
        available_augs = [
            'rotate', 'brightness', 'contrast', 'noise', 'blur', 
            'speckle', 'perspective'
        ]
        
        if augmentations is None:
            # Randomly select 2-4 augmentations
            num_augs = random.randint(2, 4)
            augmentations = random.sample(available_augs, num_augs)
        
        result = image.copy()
        
        for aug in augmentations:
            if aug == 'rotate':
                result = self.rotate_image(result)
            elif aug == 'brightness':
                result = self.adjust_brightness(result)
            elif aug == 'contrast':
                result = self.adjust_contrast(result)
            elif aug == 'noise':
                result = self.add_gaussian_noise(result)
            elif aug == 'blur':
                if random.random() < 0.7:
                    result = self.apply_blur(result)
                else:
                    result = self.apply_motion_blur(result)
            elif aug == 'speckle':
                result = self.add_speckle_noise(result)
            elif aug == 'perspective':
                result = self.perspective_transform(result)
        
        return result
    
    def apply_random_augmentation(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """
        Apply random augmentation and return both the result and parameters used.
        
        Args:
            image: Input PIL image
            
        Returns:
            Tuple of (augmented_image, augmentation_parameters)
        """
        # Available augmentations with their probability of being applied
        available_augs = {
            'rotate': 0.5,
            'brightness': 0.7,
            'contrast': 0.7,
            'noise': 0.4,
            'blur': 0.3,
            'speckle': 0.2,
            'perspective': 0.3
        }
        
        result = image.copy()
        applied_augmentations = {}
        
        for aug_name, probability in available_augs.items():
            if random.random() < probability:
                if aug_name == 'rotate':
                    angle = random.uniform(*self.rotation_range)
                    result = self.rotate_image(result, angle)
                    applied_augmentations['rotation_angle'] = angle
                    
                elif aug_name == 'brightness':
                    factor = random.uniform(*self.brightness_range)
                    result = self.adjust_brightness(result, factor)
                    applied_augmentations['brightness_factor'] = factor
                    
                elif aug_name == 'contrast':
                    factor = random.uniform(*self.contrast_range)
                    result = self.adjust_contrast(result, factor)
                    applied_augmentations['contrast_factor'] = factor
                    
                elif aug_name == 'noise':
                    std = random.uniform(0.005, self.noise_std)
                    result = self.add_gaussian_noise(result, std)
                    applied_augmentations['noise_std'] = std
                    
                elif aug_name == 'blur':
                    if random.random() < 0.7:
                        radius = random.uniform(0.1, 1.0)
                        result = self.apply_blur(result, radius)
                        applied_augmentations['blur_radius'] = radius
                        applied_augmentations['blur_type'] = 'gaussian'
                    else:
                        size = random.randint(3, 7)
                        angle = random.uniform(0, 360)
                        result = self.apply_motion_blur(result, size, angle)
                        applied_augmentations['motion_blur_size'] = size
                        applied_augmentations['motion_blur_angle'] = angle
                        applied_augmentations['blur_type'] = 'motion'
                        
                elif aug_name == 'speckle':
                    intensity = random.uniform(0.01, 0.05)
                    result = self.add_speckle_noise(result, intensity)
                    applied_augmentations['speckle_intensity'] = intensity
                    
                elif aug_name == 'perspective':
                    intensity = random.uniform(0.05, 0.15)
                    result = self.perspective_transform(result, intensity)
                    applied_augmentations['perspective_intensity'] = intensity
        
        # Add metadata about what was applied
        augmentation_metadata = {
            'applied_augmentations': list(applied_augmentations.keys()),
            'num_augmentations': len(applied_augmentations),
            'parameters': applied_augmentations
        }
        
        return result, augmentation_metadata 