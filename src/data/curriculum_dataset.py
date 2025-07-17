import sys
sys.path.append('src')

import torch
from torch.nn.utils.rnn import pad_sequence


def curriculum_collate_fn(batch):
    """Custom collate function for variable-width images and sequences."""
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['targets'])
    
    # Pad images to max width in batch
    max_width = max(img.shape[-1] for img in images)
    padded_images = []
    
    for img in images:
        if len(img.shape) == 3:  # (C, H, W)
            c, h, w = img.shape
            padded = torch.zeros(c, h, max_width, dtype=img.dtype, device=img.device)
            padded[:, :, :w] = img
        else:  # (H, W) or other
            h, w = img.shape[-2:]
            padded = torch.zeros(*img.shape[:-1], max_width, dtype=img.dtype, device=img.device)
            padded[..., :w] = img
        padded_images.append(padded)
    
    # Stack images
    images_tensor = torch.stack(padded_images, dim=0)
    
    # Pad target sequences
    targets_tensor = pad_sequence(targets, batch_first=True, padding_value=0)  # 0 is PAD_IDX
    
    return {
        'image': images_tensor,
        'targets': targets_tensor
    }

class CurriculumDataset:
    """Wrapper for OnTheFlyDataset that enforces max sequence length using syllable-based truncation."""
    
    def __init__(self, base_dataset, max_length, config_manager):
        self.base_dataset = base_dataset
        self.max_length = max_length
        self.config_manager = config_manager
        self.vocab = config_manager.vocab
        
        # Import syllable segmentation function
        from src.khtext.subword_cluster import split_syllables_advanced
        self.split_syllables = self._split_syllables_preserve_spaces
        
        # Cache only text strings to avoid OnTheFlyDataset randomness (much faster than full samples)
        print(f"🔄 Caching text strings for {len(base_dataset)} samples...")
        self.cached_texts = []
        
        # Access the text_lines directly from OnTheFlyDataset to avoid slow __getitem__ calls
        if hasattr(base_dataset, 'text_lines'):
            # For datasets smaller than text_lines, cycle through them deterministically
            for i in range(len(base_dataset)):
                text_idx = i % len(base_dataset.text_lines)
                self.cached_texts.append(base_dataset.text_lines[text_idx])
        else:
            # Fallback: this might be slow for other dataset types
            print("⚠️  Warning: Base dataset doesn't have text_lines, falling back to slower method")
            for i in range(len(base_dataset)):
                _, _, text = base_dataset[i]
                self.cached_texts.append(text)
        
        print(f"✅ Text caching complete! {len(self.cached_texts)} texts cached.")
    
    def _split_syllables_preserve_spaces(self, text):
        """
        Split syllables while preserving actual space characters instead of converting to <SPACE> tags.
        This is needed for curriculum learning to maintain proper vocabulary encoding.
        """
        from src.khtext.subword_cluster import split_syllables_advanced
        
        # Get syllables using the original function
        syllables = split_syllables_advanced(text)
        
        # Convert <SPACE> tags back to actual space characters
        preserved_syllables = []
        for syllable in syllables:
            if syllable == '<SPACE>':
                preserved_syllables.append(' ')  # Convert back to actual space
            elif syllable == '<SPACES>':
                preserved_syllables.append('  ')  # Convert back to multiple spaces
            elif syllable == '<TAB>':
                preserved_syllables.append('\t')  # Convert back to tab
            elif syllable == '<NEWLINE>':
                preserved_syllables.append('\n')  # Convert back to newline
            elif syllable == '<CRLF>':
                preserved_syllables.append('\r\n')  # Convert back to CRLF
            else:
                preserved_syllables.append(syllable)  # Keep as-is
        
        return preserved_syllables
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get cached text to ensure consistency
        original_text = self.cached_texts[idx]
        
        # Split into syllables for intelligent truncation
        syllables = self.split_syllables(original_text)
        
        # Calculate current length with SOS/EOS tokens (reserve 2 tokens for SOS+EOS)
        max_content_tokens = self.max_length - 2  # Reserve 2 tokens for SOS and EOS
        current_length = 0  # Start counting content tokens only
        selected_syllables = []
        
        for syllable in syllables:
            syllable_tokens = len(self.vocab.encode(syllable))
            if current_length + syllable_tokens <= max_content_tokens:
                selected_syllables.append(syllable)
                current_length += syllable_tokens
            else:
                break
        
        # Join selected syllables
        truncated_text = ''.join(selected_syllables)
        
        # Encode the truncated text with SOS/EOS
        target_indices = [self.vocab.SOS_IDX] + self.vocab.encode(truncated_text) + [self.vocab.EOS_IDX]
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        
        # Verify final length doesn't exceed max_length (safety check)
        if len(target_indices) > self.max_length:
            # Emergency truncation if still too long (should not happen with correct logic)
            truncated_indices = target_indices[:self.max_length-1] + [self.vocab.EOS_IDX]
            target_tensor = torch.tensor(truncated_indices, dtype=torch.long)
            # Decode back to get the actual truncated text
            truncated_text = self.vocab.decode(truncated_indices[1:-1])  # Remove SOS/EOS for decoding
        
        # Generate synthetic image for truncated text using the base dataset's text generator
        try:
            if hasattr(self.base_dataset, 'text_generator'):
                # Use the same text generator as the base dataset
                text_generator = self.base_dataset.text_generator
                
                # Generate image for the truncated text with minimal augmentation for curriculum learning
                from PIL import ImageFont, Image
                import torchvision.transforms as transforms
                
                # Select appropriate font
                font_path = text_generator._select_font(truncated_text, self.base_dataset.split)
                temp_font = ImageFont.truetype(font_path, text_generator.font_size)
                image_width = text_generator._calculate_optimal_width(truncated_text, temp_font)
                
                # Render the text
                pil_image = text_generator._render_text_image(truncated_text, font_path, image_width)
                
                # Apply light augmentation (reduce for curriculum learning clarity)
                if self.base_dataset.use_augmentation and text_generator.augmentor:
                    # Temporarily reduce augmentation
                    original_aug_prob = self.base_dataset.augment_prob
                    self.base_dataset.augment_prob = 0.3  # Very light augmentation
                    pil_image = text_generator._apply_augmentation(pil_image)
                    self.base_dataset.augment_prob = original_aug_prob
                
                # Convert to tensor
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                image = transform(pil_image)
                
            else:
                # Fallback: create a dummy image 
                image = torch.zeros((3, 64, 256))
                
        except Exception as e:
            # Fallback on any error
            image = torch.zeros((3, 64, 256))
        
        return {
            'image': image,
            'targets': target_tensor,
            'text': truncated_text
        }
