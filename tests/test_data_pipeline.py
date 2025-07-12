"""
Unit tests for data pipeline components.
Tests text rendering, augmentation, and dataset functionality.
"""
import unittest
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from data.text_renderer import TextRenderer
from data.augmentation import DataAugmentation
from data.dataset import KhmerDataset, collate_fn
from utils.config import KhmerVocab


class TestTextRenderer(unittest.TestCase):
    """Test cases for TextRenderer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = TextRenderer(image_height=32)
    
    def test_render_text_basic(self):
        """Test basic text rendering."""
        text = "កម្ពុជា"
        image = self.renderer.render_text(text)
        
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.height, 32)
        self.assertGreater(image.width, 0)
        self.assertEqual(image.mode, 'L')  # Grayscale
    
    def test_render_empty_text(self):
        """Test rendering empty text."""
        text = ""
        image = self.renderer.render_text(text)
        
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.height, 32)
        self.assertGreater(image.width, 0)  # Should still have some width
    
    def test_render_numbers(self):
        """Test rendering numbers."""
        texts = ["123", "៤៥៦", "១2៣"]
        
        for text in texts:
            with self.subTest(text=text):
                image = self.renderer.render_text(text)
                self.assertIsInstance(image, Image.Image)
                self.assertEqual(image.height, 32)
    
    def test_render_punctuation(self):
        """Test rendering punctuation."""
        texts = ["។", ",", "!", "?", "()"]
        
        for text in texts:
            with self.subTest(text=text):
                image = self.renderer.render_text(text)
                self.assertIsInstance(image, Image.Image)
                self.assertEqual(image.height, 32)
    
    def test_image_properties(self):
        """Test consistent image properties."""
        texts = ["កម្ពុជា", "hello", "១២៣", "។"]
        
        for text in texts:
            with self.subTest(text=text):
                image = self.renderer.render_text(text)
                
                # Check image properties
                self.assertEqual(image.height, 32)
                self.assertEqual(image.mode, 'L')
                
                # Check image has content (not all black)
                img_array = np.array(image)
                self.assertTrue(np.any(img_array > 0))


class TestDataAugmentation(unittest.TestCase):
    """Test cases for DataAugmentation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.augmentation = DataAugmentation(blur_prob=0.5, morphology_prob=0.5, 
                                           noise_prob=0.5, concatenation_prob=0.5)
        # Create a test image
        self.test_image = Image.new('L', (64, 32), color=255)
        # Add some content to the image
        from PIL import ImageDraw
        draw = ImageDraw.Draw(self.test_image)
        draw.text((10, 10), "Test", fill=0)
    
    def test_gaussian_blur(self):
        """Test Gaussian blur augmentation."""
        blurred = self.augmentation.gaussian_blur(self.test_image)
        
        self.assertIsInstance(blurred, Image.Image)
        self.assertEqual(blurred.size, self.test_image.size)
        self.assertEqual(blurred.mode, self.test_image.mode)
    
    def test_morphological_operations(self):
        """Test morphological operations."""
        # Test dilation
        dilated = self.augmentation.morphological_operations(self.test_image, operation='dilation')
        self.assertIsInstance(dilated, Image.Image)
        self.assertEqual(dilated.size, self.test_image.size)
        
        # Test erosion
        eroded = self.augmentation.morphological_operations(self.test_image, operation='erosion')
        self.assertIsInstance(eroded, Image.Image)
        self.assertEqual(eroded.size, self.test_image.size)
    
    def test_noise_injection(self):
        """Test noise injection."""
        noisy = self.augmentation.noise_injection(self.test_image)
        
        self.assertIsInstance(noisy, Image.Image)
        self.assertEqual(noisy.size, self.test_image.size)
        self.assertEqual(noisy.mode, self.test_image.mode)
    
    def test_apply_augmentations(self):
        """Test applying multiple augmentations."""
        augmented = self.augmentation.apply_augmentations(self.test_image)
        
        self.assertIsInstance(augmented, Image.Image)
        self.assertEqual(augmented.size[1], self.test_image.size[1])  # Height should be preserved
        # Width might change due to concatenation
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        self.augmentation.set_seed(42)
        
        # Apply augmentation twice with same seed
        result1 = self.augmentation.apply_augmentations(self.test_image)
        
        self.augmentation.set_seed(42)
        result2 = self.augmentation.apply_augmentations(self.test_image)
        
        # Results should be identical
        self.assertEqual(result1.size, result2.size)


class TestKhmerDataset(unittest.TestCase):
    """Test cases for KhmerDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab = KhmerVocab()
        self.dataset = KhmerDataset(
            vocab=self.vocab,
            dataset_size=10,
            min_text_length=1,
            max_text_length=5,
            use_augmentation=False,  # Disable for consistent testing
            seed=42
        )
    
    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), 10)
    
    def test_dataset_item_structure(self):
        """Test structure of dataset items."""
        item = self.dataset[0]
        
        # Check required keys
        required_keys = ['image', 'text', 'target', 'target_length', 'image_width']
        for key in required_keys:
            self.assertIn(key, item)
        
        # Check types
        self.assertIsInstance(item['image'], torch.Tensor)
        self.assertIsInstance(item['text'], str)
        self.assertIsInstance(item['target'], torch.Tensor)
        self.assertIsInstance(item['target_length'], int)
        self.assertIsInstance(item['image_width'], int)
    
    def test_image_dimensions(self):
        """Test image tensor dimensions."""
        item = self.dataset[0]
        image = item['image']
        
        # Should be [channels, height, width]
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[0], 1)  # Grayscale
        self.assertEqual(image.shape[1], 32)  # Height
        self.assertGreater(image.shape[2], 0)  # Width > 0
    
    def test_target_encoding(self):
        """Test target sequence encoding."""
        item = self.dataset[0]
        text = item['text']
        target = item['target']
        target_length = item['target_length']
        
        # Decode target back to text
        decoded_text = self.vocab.decode(target.tolist())
        self.assertEqual(decoded_text, text)
        self.assertEqual(len(target), target_length)
    
    def test_text_length_constraints(self):
        """Test that generated text respects length constraints."""
        for i in range(5):
            item = self.dataset[i]
            text = item['text']
            text_length = len(text)
            
            self.assertGreaterEqual(text_length, self.dataset.min_text_length)
            self.assertLessEqual(text_length, self.dataset.max_text_length)
    
    def test_reproducibility(self):
        """Test that dataset is reproducible with same seed."""
        # Create two identical datasets
        dataset1 = KhmerDataset(vocab=self.vocab, dataset_size=5, seed=42)
        dataset2 = KhmerDataset(vocab=self.vocab, dataset_size=5, seed=42)
        
        # First items should be identical
        item1 = dataset1[0]
        item2 = dataset2[0]
        
        self.assertEqual(item1['text'], item2['text'])
        self.assertTrue(torch.equal(item1['target'], item2['target']))


class TestCollateFunction(unittest.TestCase):
    """Test cases for collate function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab = KhmerVocab()
        self.dataset = KhmerDataset(
            vocab=self.vocab,
            dataset_size=5,
            use_augmentation=False,
            seed=42
        )
    
    def test_collate_basic(self):
        """Test basic collate functionality."""
        # Get a small batch
        batch = [self.dataset[i] for i in range(3)]
        collated = collate_fn(batch)
        
        # Check required keys
        required_keys = ['images', 'texts', 'targets', 'target_lengths', 'image_widths']
        for key in required_keys:
            self.assertIn(key, collated)
    
    def test_batch_dimensions(self):
        """Test batch tensor dimensions."""
        batch = [self.dataset[i] for i in range(3)]
        collated = collate_fn(batch)
        
        images = collated['images']
        targets = collated['targets']
        target_lengths = collated['target_lengths']
        
        # Check batch dimensions
        self.assertEqual(images.shape[0], 3)  # Batch size
        self.assertEqual(targets.shape[0], 3)  # Batch size
        self.assertEqual(len(target_lengths), 3)  # Batch size
        
        # Images should be padded to same width
        self.assertEqual(len(set([images[i].shape[2] for i in range(3)])), 1)
    
    def test_sequence_padding(self):
        """Test that sequences are padded correctly."""
        batch = [self.dataset[i] for i in range(3)]
        collated = collate_fn(batch)
        
        targets = collated['targets']
        target_lengths = collated['target_lengths']
        
        # All sequences should have same length (padded)
        max_len = targets.shape[1]
        
        # Check padding
        for i in range(3):
            actual_length = target_lengths[i].item()
            # Positions beyond actual length should be PAD tokens
            if actual_length < max_len:
                for j in range(actual_length, max_len):
                    self.assertEqual(targets[i, j].item(), self.vocab.PAD_IDX)
    
    def test_dataloader_integration(self):
        """Test integration with PyTorch DataLoader."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        # Should have correct structure
        self.assertIn('images', batch)
        self.assertIn('targets', batch)
        self.assertEqual(batch['images'].shape[0], 3)
        self.assertEqual(batch['targets'].shape[0], 3)


if __name__ == '__main__':
    unittest.main() 