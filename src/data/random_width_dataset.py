import sys
sys.path.append('src')

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from collections import defaultdict
from src.khtext.subword_cluster import split_syllables_advanced

class RandomWidthDataset:
    """Dataset that generates short texts with exponential probability distribution favoring shorter lengths."""
    
    def __init__(self, base_dataset, min_length=1, max_length=150, config_manager=None, alpha=0.02):
        self.base_dataset = base_dataset
        self.min_length = min_length
        self.max_length = max_length
        self.config_manager = config_manager
        self.vocab = config_manager.vocab if config_manager else None
        self.alpha = alpha  # Controls exponential decay rate
        
        # Import syllable segmentation function
        self.split_syllables = self._split_syllables_preserve_spaces
        
        # Create exponential probability distribution for text lengths
        self.length_probabilities = self._create_length_distribution()
        
        # Cache texts by length and shuffle them
        print(f"🔄 Caching and organizing texts by length for {len(base_dataset)} samples...")
        self.texts_by_length = self._cache_and_organize_texts()
        
        print(f"✅ Text caching complete! Generated {sum(len(texts) for texts in self.texts_by_length.values())} texts across {len(self.texts_by_length)} length categories.")
        
    def _create_length_distribution(self):
        """Create probability distribution heavily favoring very short texts (1-5 chars)."""
        lengths = np.arange(self.min_length, self.max_length + 1)
        
        # Create custom distribution with heavy bias toward 1-5 character texts
        raw_probs = np.zeros_like(lengths, dtype=float)
        
        for i, length in enumerate(lengths):
            if 1 <= length <= 5:
                # Very high probability for 1-5 characters (40% total)
                raw_probs[i] = 8.0  # Base weight for very short texts
            elif 6 <= length <= 10:
                # Moderate probability for 6-10 characters (25% total)
                raw_probs[i] = 5.0
            elif 11 <= length <= 20:
                # Lower probability for 11-20 characters (20% total)
                raw_probs[i] = 2.0
            elif 21 <= length <= 50:
                # Even lower for 21-50 characters (10% total)
                raw_probs[i] = 0.5
            else:
                # Very low probability for longer texts (5% total)
                raw_probs[i] = np.exp(-self.alpha * (length - 50)) * 0.1
        
        # Additional boost for single characters (1-3 chars get extra weight)
        for i, length in enumerate(lengths):
            if 1 <= length <= 3:
                raw_probs[i] *= 2.0  # Double the weight for 1-3 characters
        
        # Normalize to create proper probability distribution
        probabilities = raw_probs / np.sum(raw_probs)
        
        print(f"📊 Length distribution created (optimized for short texts):")
        print(f"   Length 1-5: {np.sum(probabilities[:5]):.3f}")
        print(f"   Length 6-10: {np.sum(probabilities[5:10]):.3f}")
        print(f"   Length 11-20: {np.sum(probabilities[10:20]):.3f}")
        print(f"   Length 21-50: {np.sum(probabilities[20:50]):.3f}")
        print(f"   Length 51+: {np.sum(probabilities[50:]):.3f}")
        
        return probabilities
    
    def _split_syllables_preserve_spaces(self, text):
        """Split syllables while preserving actual space characters."""
        
        
        syllables = split_syllables_advanced(text)
        
        # Convert special tokens back to actual characters
        preserved_syllables = []
        for syllable in syllables:
            if syllable == '<SPACE>':
                preserved_syllables.append(' ')
            elif syllable == '<SPACES>':
                preserved_syllables.append('  ')
            elif syllable == '<TAB>':
                preserved_syllables.append('\t')
            elif syllable == '<NEWLINE>':
                preserved_syllables.append('\n')
            elif syllable == '<CRLF>':
                preserved_syllables.append('\r\n')
            else:
                preserved_syllables.append(syllable)
        
        return preserved_syllables
    
    def _cache_and_organize_texts(self):
        """Cache texts and organize them by character length, then shuffle."""
        texts_by_length = defaultdict(list)
        
        # Access text_lines directly for efficiency
        if hasattr(self.base_dataset, 'text_lines'):
            text_source = self.base_dataset.text_lines
        else:
            print("⚠️  Warning: Base dataset doesn't have text_lines, using slower method")
            text_source = [self.base_dataset[i][2] for i in range(len(self.base_dataset))]
        
        # Process each text to create variations of different lengths
        for original_text in text_source:
            # Generate multiple length variations from each source text
            self._generate_length_variations(original_text, texts_by_length)
        
        # Shuffle texts within each length category
        for length in texts_by_length:
            random.shuffle(texts_by_length[length])
            
        return dict(texts_by_length)
    
    def _generate_length_variations(self, original_text, texts_by_length):
        """Generate multiple text variations of different target lengths from a source text."""
        syllables = self.split_syllables(original_text)
        
        # Create texts of various lengths by progressive truncation
        current_text = ""
        current_length = 0
        
        for syllable in syllables:
            current_text += syllable
            current_length = len(current_text)
            
            # If this length is within our target range, add it
            if self.min_length <= current_length <= self.max_length:
                texts_by_length[current_length].append(current_text)
            
            # Stop if we exceed max length
            if current_length > self.max_length:
                break
        
        # Also generate some shorter texts by word-level truncation for variety
        words = original_text.split()
        if len(words) > 1:
            current_text = ""
            for word in words:
                if current_text:
                    test_text = current_text + " " + word
                else:
                    test_text = word
                    
                test_length = len(test_text)
                if test_length > self.max_length:
                    break
                    
                current_text = test_text
                if self.min_length <= test_length <= self.max_length:
                    texts_by_length[test_length].append(current_text)
    
    def _sample_target_length(self):
        """Sample a target length based on exponential probability distribution."""
        lengths = np.arange(self.min_length, self.max_length + 1)
        return np.random.choice(lengths, p=self.length_probabilities)
    
    def _get_text_for_length(self, target_length):
        """Get a text of approximately the target length."""
        # Try to find exact length first
        if target_length in self.texts_by_length and self.texts_by_length[target_length]:
            return random.choice(self.texts_by_length[target_length])
        
        # Find closest available length
        available_lengths = [length for length in self.texts_by_length.keys() 
                           if self.texts_by_length[length]]
        
        if not available_lengths:
            # Fallback: create a minimal text
            return "តេស្ត" if target_length >= 4 else "ត"
        
        # Find closest length
        closest_length = min(available_lengths, key=lambda x: abs(x - target_length))
        selected_text = random.choice(self.texts_by_length[closest_length])
        
        # Truncate if necessary
        if len(selected_text) > target_length:
            # Intelligent truncation using syllables
            syllables = self.split_syllables(selected_text)
            current_text = ""
            for syllable in syllables:
                test_text = current_text + syllable
                if len(test_text) <= target_length:
                    current_text = test_text
                else:
                    break
            return current_text if current_text else selected_text[:target_length]
        
        return selected_text
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Sample target length based on exponential distribution
        target_length = self._sample_target_length()
        
        # Get text of approximately target length
        text = self._get_text_for_length(target_length)
        
        # Ensure text is not empty
        if not text or len(text) == 0:
            text = "តេស្ត"  # Fallback text
        
        # Encode the text with SOS/EOS if vocab is available
        if self.vocab:
            target_indices = [self.vocab.SOS_IDX] + self.vocab.encode(text) + [self.vocab.EOS_IDX]
            target_tensor = torch.tensor(target_indices, dtype=torch.long)
        else:
            # Create dummy tensor if no vocab
            target_tensor = torch.tensor([0] + list(range(len(text))) + [1], dtype=torch.long)
        
        # Generate synthetic image for the text using the base dataset's text generator
        try:
            if hasattr(self.base_dataset, 'text_generator'):
                image = self._generate_image_for_text(text)
            else:
                # Fallback: create a dummy image 
                image = torch.zeros((3, 64, max(256, len(text) * 8)))
                
        except Exception as e:
            print(f"⚠️  Error generating image: {e}")
            # Fallback on any error
            image = torch.zeros((3, 64, max(256, len(text) * 8)))
        
        return {
            'image': image,
            'targets': target_tensor,
            'text': text,
            'target_length': target_length,
            'actual_length': len(text)
        }
    
    def _generate_image_for_text(self, text):
        """Generate synthetic image for the given text."""
        text_generator = self.base_dataset.text_generator
        
        # Generate image for the text with minimal augmentation for curriculum learning
        from PIL import ImageFont, Image
        import torchvision.transforms as transforms
        
        # Select appropriate font
        font_path = text_generator._select_font(text, self.base_dataset.split)
        temp_font = ImageFont.truetype(font_path, text_generator.font_size)
        image_width = text_generator._calculate_optimal_width(text, temp_font)
        
        # Ensure minimum width for very short texts
        image_width = max(image_width, len(text) * 16)
        
        # Render the text
        pil_image = text_generator._render_text_image(text, font_path, image_width)
        
        # Apply light augmentation for short texts
        if self.base_dataset.use_augmentation and text_generator.augmentor and len(text) > 10:
            # Reduce augmentation for very short texts to maintain clarity
            original_aug_prob = getattr(self.base_dataset, 'augment_prob', 0.5)
            light_aug_prob = 0.3 if len(text) < 20 else 0.5
            self.base_dataset.augment_prob = light_aug_prob
            pil_image = text_generator._apply_augmentation(pil_image)
            self.base_dataset.augment_prob = original_aug_prob
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = transform(pil_image)
        
        return image
    
    def get_length_statistics(self):
        """Get statistics about the cached texts by length."""
        stats = {}
        for length, texts in self.texts_by_length.items():
            stats[length] = len(texts)
        return stats
    
    def sample_batch_by_length_distribution(self, batch_size):
        """Sample a batch following the exponential length distribution."""
        batch = []
        for _ in range(batch_size):
            batch.append(self[random.randint(0, len(self) - 1)])
        return batch
