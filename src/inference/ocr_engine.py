"""
Khmer OCR Engine for inference.
Provides easy-to-use API for single image OCR with preprocessing and postprocessing.
"""
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import cv2

from ..models.seq2seq import KhmerOCRSeq2Seq
from ..utils.config import ConfigManager, KhmerVocab
from ..data.augmentation import DataAugmentation


class KhmerOCREngine:
    """
    Main OCR engine for Khmer text recognition.
    
    Features:
    - Single image OCR with automatic preprocessing
    - Greedy decoding and beam search options
    - Confidence scoring and post-processing
    - Batch processing support
    """
    
    def __init__(
        self,
        model: KhmerOCRSeq2Seq,
        vocab: KhmerVocab,
        device: torch.device = None,
        image_height: int = 32,
        max_width: int = 1600  # Updated from 800 to 1600 to match new config default
    ):
        """
        Initialize OCR engine.
        
        Args:
            model: Trained Seq2Seq model
            vocab: Vocabulary for encoding/decoding
            device: Device to run inference on
            image_height: Fixed height for input images
            max_width: Maximum width for input images (should match training config)
        """
        self.model = model
        self.vocab = vocab
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_height = image_height
        self.max_width = max_width
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing setup
        self.setup_preprocessing()
    
    def setup_preprocessing(self):
        """Setup image preprocessing pipeline."""
        # Basic transforms
        import torchvision.transforms as transforms
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
        ])
    
    def preprocess_image(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        enhance_contrast: bool = True,
        denoise: bool = True,
        binarize: bool = False
    ) -> torch.Tensor:
        """
        Preprocess image for OCR.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            enhance_contrast: Whether to enhance contrast
            denoise: Whether to apply denoising
            binarize: Whether to binarize the image
            
        Returns:
            Preprocessed image tensor (1, 1, H, W)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 4:
                image = image.squeeze(0)  # Remove batch dimension if present
            if image.dim() == 3 and image.size(0) == 1:
                image = image.squeeze(0)  # Remove channel dimension if grayscale
            # Convert to numpy and then PIL
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(img_np)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply preprocessing steps
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Slight contrast boost
        
        if denoise:
            # Convert to numpy for OpenCV operations
            img_np = np.array(image)
            img_np = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
            image = Image.fromarray(img_np)
        
        if binarize:
            # Simple thresholding
            img_np = np.array(image)
            _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = Image.fromarray(img_np)
        
        # Resize to target height while maintaining aspect ratio
        if image.height != self.image_height:
            aspect_ratio = image.width / image.height
            new_width = int(self.image_height * aspect_ratio)
            image = image.resize((new_width, self.image_height), Image.Resampling.LANCZOS)
        
        # Clip width if too large
        if image.width > self.max_width:
            image = image.crop((0, 0, self.max_width, self.image_height))
        
        # Convert to tensor and add batch dimension
        image_tensor = self.image_transform(image)  # (1, H, W)
        image_tensor = image_tensor.unsqueeze(0)    # (1, 1, H, W)
        
        return image_tensor.to(self.device)
    
    def recognize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        method: str = 'greedy',
        beam_size: int = 5,
        max_length: int = 256,
        length_penalty: float = 0.6,
        temperature: float = 1.0,
        return_confidence: bool = True,
        preprocess: bool = True
    ) -> Dict[str, any]:
        """
        Recognize text from image.
        
        Args:
            image: Input image
            method: Decoding method ('greedy' or 'beam_search')
            beam_size: Beam size for beam search
            max_length: Maximum sequence length
            length_penalty: Length penalty for beam search
            temperature: Temperature for sampling
            return_confidence: Whether to return confidence scores
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary containing:
            - text: Recognized text string
            - confidence: Confidence score (if requested)
            - attention_weights: Attention weights
            - raw_output: Raw model output for debugging
        """
        with torch.no_grad():
            # Preprocess image
            if preprocess:
                image_tensor = self.preprocess_image(image)
            else:
                # Minimal processing for tensor input
                if isinstance(image, torch.Tensor):
                    image_tensor = image.to(self.device)
                else:
                    image_tensor = self.preprocess_image(image, 
                                                       enhance_contrast=False, 
                                                       denoise=False, 
                                                       binarize=False)
            
            # Generate text
            if method == 'greedy':
                result = self.model.generate(
                    images=image_tensor,
                    max_length=max_length,
                    method='greedy',
                    temperature=temperature
                )
            elif method == 'beam_search':
                result = self.model.generate(
                    images=image_tensor,
                    max_length=max_length,
                    method='beam_search',
                    beam_size=beam_size,
                    length_penalty=length_penalty
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Decode sequences to text
            sequences = result['sequences']
            if method == 'beam_search':
                # Take top beam
                sequence = sequences[0].cpu().tolist()
            else:
                # Take first (and only) sequence
                sequence = sequences[0].cpu().tolist()
            
            # Remove special tokens and decode
            text = self._decode_sequence(sequence)
            
            # Calculate confidence if requested
            confidence = None
            if return_confidence and 'scores' in result:
                if method == 'beam_search':
                    confidence = torch.softmax(result['scores'], dim=0)[0].item()
                else:
                    # For greedy, use average log probability
                    confidence = 0.5  # Placeholder - would need to track probabilities
            
            return {
                'text': text,
                'confidence': confidence,
                'attention_weights': result.get('attention_weights'),
                'raw_output': result
            }
    
    def recognize_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        method: str = 'greedy',
        max_length: int = 256,
        preprocess: bool = True
    ) -> List[Dict[str, any]]:
        """
        Recognize text from multiple images.
        
        Args:
            images: List of input images
            method: Decoding method (only 'greedy' supported for batch)
            max_length: Maximum sequence length
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of recognition results
        """
        if method != 'greedy':
            # Process individually for beam search
            return [self.recognize(img, method=method, preprocess=preprocess) 
                   for img in images]
        
        results = []
        batch_size = min(len(images), 8)  # Process in smaller batches
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            image_tensors = []
            for img in batch_images:
                if preprocess:
                    img_tensor = self.preprocess_image(img)
                else:
                    img_tensor = self.preprocess_image(img, 
                                                     enhance_contrast=False, 
                                                     denoise=False, 
                                                     binarize=False)
                image_tensors.append(img_tensor.squeeze(0))  # Remove batch dim
            
            # Pad images to same width for batching
            max_width = max(tensor.size(-1) for tensor in image_tensors)
            padded_tensors = []
            for tensor in image_tensors:
                # Pad width to max_width
                pad_width = max_width - tensor.size(-1)
                if pad_width > 0:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_width), value=0)
                padded_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(padded_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                # Generate for batch
                result = self.model.generate(
                    images=batch_tensor,
                    max_length=max_length,
                    method='greedy'
                )
                
                # Decode each sequence
                sequences = result['sequences']
                for j, sequence in enumerate(sequences):
                    text = self._decode_sequence(sequence.cpu().tolist())
                    results.append({
                        'text': text,
                        'confidence': None,
                        'attention_weights': result.get('attention_weights')[j] if result.get('attention_weights') is not None else None,
                        'raw_output': None
                    })
        
        return results
    
    def _decode_sequence(self, sequence: List[int]) -> str:
        """
        Decode sequence of token IDs to text string.
        
        Args:
            sequence: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Remove special tokens
        cleaned_sequence = []
        for token_id in sequence:
            if token_id == self.vocab.EOS_IDX:
                break
            if token_id not in [self.vocab.SOS_IDX, self.vocab.PAD_IDX, self.vocab.UNK_IDX]:
                cleaned_sequence.append(token_id)
        
        # Decode to text
        text = self.vocab.decode(cleaned_sequence)
        
        # Post-process text
        text = self._postprocess_text(text)
        
        return text
    
    def _postprocess_text(self, text: str) -> str:
        """
        Apply post-processing to recognized text.
        
        Args:
            text: Raw recognized text
            
        Returns:
            Post-processed text
        """
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Basic text cleanup
        text = text.strip()
        
        # Additional Khmer-specific post-processing could go here
        # - Fix common recognition errors
        # - Apply language model corrections
        # - Handle diacritic positioning
        
        return text
    
    def get_attention_visualization(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        text: str = None
    ) -> Dict[str, any]:
        """
        Get attention visualization for an image.
        
        Args:
            image: Input image
            text: Target text (optional, for analysis)
            
        Returns:
            Dictionary with attention weights and visualization data
        """
        result = self.recognize(image, return_confidence=True)
        attention_weights = result['attention_weights']
        
        if attention_weights is None:
            return {'error': 'No attention weights available'}
        
        # Convert to numpy for visualization
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        return {
            'attention_weights': attention_weights,
            'recognized_text': result['text'],
            'confidence': result['confidence'],
            'image_shape': self.preprocess_image(image, preprocess=False).shape
        }
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_manager: ConfigManager = None,
        device: torch.device = None
    ) -> 'KhmerOCREngine':
        """
        Create OCR engine from saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_manager: Configuration manager (optional)
            device: Device to run on (optional)
            
        Returns:
            Initialized OCR engine
        """
        if config_manager is None:
            config_manager = ConfigManager()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with config manager to ensure correct architecture
        try:
            # Try loading with proper config first
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Create model with correct config and then load state
            model = KhmerOCRSeq2Seq(config_or_vocab_size=config_manager)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            checkpoint_info = {
                'epoch': checkpoint.get('epoch', 0),
                'best_score': checkpoint.get('best_score') or checkpoint.get('best_cer'),
                'additional_info': checkpoint.get('additional_info'),
            }
        except Exception:
            # Fallback to original method
            model, checkpoint_info = KhmerOCRSeq2Seq.load_checkpoint(
                checkpoint_path, 
                device=device
            )
        
        # Get max_width from config (use image_width if available, otherwise default to 800)
        max_width = getattr(config_manager.data_config, 'image_width', 800)
        
        # Create engine with configured max_width
        engine = cls(
            model=model,
            vocab=config_manager.vocab,
            device=device,
            image_height=config_manager.data_config.image_height,
            max_width=max_width  # Now uses configured width
        )
        
        return engine
    
    def save_model(self, filepath: str, additional_info: Dict = None):
        """
        Save the current model state.
        
        Args:
            filepath: Path to save model
            additional_info: Additional information to save
        """
        self.model.save_checkpoint(
            filepath=filepath,
            additional_info=additional_info or {}
        ) 