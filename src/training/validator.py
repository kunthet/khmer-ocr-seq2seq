"""
Validation pipeline for Khmer OCR Seq2Seq model.
Implements Character Error Rate (CER) calculation and model performance monitoring.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, List, Tuple, Optional
import Levenshtein

from ..models.seq2seq import KhmerOCRSeq2Seq
from ..utils.config import ConfigManager


class Validator:
    """
    Validator class for model evaluation and performance monitoring.
    
    Implements:
    - Character Error Rate (CER) calculation using Levenshtein distance
    - Validation loss computation
    - Model performance monitoring and reporting
    """
    
    def __init__(
        self,
        model: KhmerOCRSeq2Seq,
        config_manager: ConfigManager,
        device: torch.device
    ):
        self.model = model
        self.config = config_manager
        self.device = device
        
        # Loss function for validation (same as training)
        self.criterion = nn.NLLLoss(ignore_index=self.config.vocab.PAD_IDX)
    
    def calculate_cer(self, predictions: List[str], targets: List[str]) -> float:
        """
        Calculate Character Error Rate (CER) using Levenshtein distance.
        
        CER = (Substitutions + Deletions + Insertions) / Total_Characters
        
        Args:
            predictions: List of predicted text strings
            targets: List of ground truth text strings
            
        Returns:
            Character Error Rate as a float between 0 and 1
        """
        if len(predictions) != len(targets):
            raise ValueError("Number of predictions must match number of targets")
        
        total_chars = 0
        total_errors = 0
        
        for pred, target in zip(predictions, targets):
            # Remove special tokens for CER calculation
            pred_clean = self._clean_text(pred)
            target_clean = self._clean_text(target)
            
            # Calculate Levenshtein distance (edit distance)
            edit_distance = Levenshtein.distance(pred_clean, target_clean)
            
            # Total characters is the length of the target
            target_length = len(target_clean)
            
            if target_length > 0:
                total_chars += target_length
                total_errors += edit_distance
        
        # Calculate CER
        cer = total_errors / total_chars if total_chars > 0 else 0.0
        return cer
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing special tokens for CER calculation.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Remove special tokens
        special_tokens = ["SOS", "EOS", "PAD", "UNK"]
        cleaned = text
        
        for token in special_tokens:
            cleaned = cleaned.replace(token, "")
        
        # Remove extra whitespaces
        cleaned = " ".join(cleaned.split())
        
        return cleaned
    
    def decode_predictions(self, outputs: torch.Tensor) -> List[str]:
        """
        Decode model outputs to text strings.
        
        Args:
            outputs: Model outputs tensor [batch_size, seq_len, vocab_size]
            
        Returns:
            List of decoded text strings
        """
        # Get predicted indices (greedy decoding)
        predicted_indices = outputs.argmax(dim=-1)  # [batch_size, seq_len]
        
        predictions = []
        for batch_idx in range(predicted_indices.size(0)):
            indices = predicted_indices[batch_idx].cpu().tolist()
            
            # Stop at EOS token
            if self.config.vocab.EOS_IDX in indices:
                eos_pos = indices.index(self.config.vocab.EOS_IDX)
                indices = indices[:eos_pos]
            
            # Decode to text
            text = self.config.vocab.decode(indices)
            predictions.append(text)
        
        return predictions
    
    def decode_targets(self, targets: torch.Tensor) -> List[str]:
        """
        Decode target sequences to text strings.
        
        Args:
            targets: Target tensor [batch_size, seq_len]
            
        Returns:
            List of decoded text strings
        """
        target_texts = []
        for batch_idx in range(targets.size(0)):
            indices = targets[batch_idx].cpu().tolist()
            
            # Remove PAD tokens and stop at EOS
            cleaned_indices = []
            for idx in indices:
                if idx == self.config.vocab.PAD_IDX:
                    continue
                if idx == self.config.vocab.EOS_IDX:
                    break
                cleaned_indices.append(idx)
            
            # Decode to text (skip SOS token if present)
            if cleaned_indices and cleaned_indices[0] == self.config.vocab.SOS_IDX:
                cleaned_indices = cleaned_indices[1:]
            
            text = self.config.vocab.decode(cleaned_indices)
            target_texts.append(text)
        
        return target_texts
    
    def validate_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, List[str], List[str]]:
        """
        Validate a single batch and return loss and predictions.
        
        Args:
            batch: Batch dictionary containing images and targets
            
        Returns:
            Tuple of (batch_loss, predictions, targets)
        """
        # Move batch to device
        images = batch['images'].to(self.device)  # [B, 1, H, W]
        targets = batch['targets'].to(self.device)  # [B, max_seq_len]
        target_lengths = batch['target_lengths'].to(self.device)  # [B]
        
        batch_size = images.size(0)
        max_seq_len = targets.size(1)
        
        with torch.no_grad():
            # Encode images
            encoder_outputs, decoder_hidden = self.model.encode(images)
            
            # Initialize decoder input with SOS tokens
            decoder_input = torch.full(
                (batch_size, 1),
                self.config.vocab.SOS_IDX,
                dtype=torch.long,
                device=self.device
            )
            
            # Store predictions and compute loss
            all_outputs = []
            batch_loss = 0.0
            total_predictions = 0
            
            # Greedy decoding for validation (no teacher forcing)
            for t in range(1, max_seq_len):
                # Decode next token
                decoder_output, decoder_hidden, attention_weights = self.model.decode_step(
                    decoder_input[:, -1:],  # Last predicted token
                    decoder_hidden,
                    encoder_outputs
                )
                
                all_outputs.append(decoder_output.unsqueeze(1))  # [B, 1, vocab_size]
                
                # Calculate loss for this step
                target_t = targets[:, t]  # [B]
                mask = target_t != self.config.vocab.PAD_IDX
                valid_predictions = mask.sum().item()
                
                if valid_predictions > 0:
                    step_loss = self.criterion(decoder_output[mask], target_t[mask])
                    batch_loss += step_loss
                    total_predictions += valid_predictions
                
                # Use predicted token as next input (greedy decoding)
                predicted_token = decoder_output.argmax(dim=-1, keepdim=True)  # [B, 1]
                decoder_input = torch.cat([decoder_input, predicted_token], dim=1)
                
                # Early stopping if all sequences predicted EOS
                if torch.all(predicted_token.squeeze() == self.config.vocab.EOS_IDX):
                    break
            
            # Concatenate all outputs
            if all_outputs:
                outputs = torch.cat(all_outputs, dim=1)  # [B, seq_len, vocab_size]
            else:
                outputs = torch.zeros(batch_size, 1, len(self.config.vocab)).to(self.device)
            
            # Average loss over sequence length
            if total_predictions > 0:
                batch_loss = batch_loss.item() / max_seq_len
            else:
                batch_loss = 0.0
        
        # Decode predictions and targets
        predictions = self.decode_predictions(outputs)
        target_texts = self.decode_targets(targets)
        
        return batch_loss, predictions, target_texts
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run full validation on the provided dataloader.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = len(dataloader)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Validate batch
                batch_loss, predictions, targets = self.validate_batch(batch)
                
                # Accumulate metrics
                total_loss += batch_loss
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # Log progress every 50 batches
                if batch_idx % 50 == 0:
                    progress = 100.0 * batch_idx / num_batches
                    print(f"Validation progress: {progress:.1f}%")
        
        # Calculate final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        cer = self.calculate_cer(all_predictions, all_targets)
        validation_time = time.time() - start_time
        
        # Calculate additional metrics
        total_chars = sum(len(self._clean_text(target)) for target in all_targets)
        avg_pred_length = sum(len(self._clean_text(pred)) for pred in all_predictions) / len(all_predictions) if all_predictions else 0
        avg_target_length = sum(len(self._clean_text(target)) for target in all_targets) / len(all_targets) if all_targets else 0
        
        return {
            'val_loss': avg_loss,
            'cer': cer,
            'validation_time': validation_time,
            'num_samples': len(all_predictions),
            'total_characters': total_chars,
            'avg_pred_length': avg_pred_length,
            'avg_target_length': avg_target_length
        }
    
    def validate_samples(
        self, 
        dataloader: DataLoader, 
        num_samples: int = 5
    ) -> List[Dict[str, str]]:
        """
        Validate a few samples and return detailed results for inspection.
        
        Args:
            dataloader: Validation dataloader
            num_samples: Number of samples to validate in detail
            
        Returns:
            List of dictionaries containing sample validation results
        """
        self.model.eval()
        
        sample_results = []
        samples_collected = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= num_samples:
                    break
                
                batch_loss, predictions, targets = self.validate_batch(batch)
                
                # Collect samples from this batch
                batch_size = len(predictions)
                for i in range(min(batch_size, num_samples - samples_collected)):
                    pred_clean = self._clean_text(predictions[i])
                    target_clean = self._clean_text(targets[i])
                    
                    sample_cer = Levenshtein.distance(pred_clean, target_clean) / len(target_clean) if len(target_clean) > 0 else 0.0
                    
                    sample_results.append({
                        'prediction': predictions[i],
                        'target': targets[i],
                        'prediction_clean': pred_clean,
                        'target_clean': target_clean,
                        'cer': sample_cer,
                        'correct': pred_clean == target_clean
                    })
                    
                    samples_collected += 1
        
        return sample_results 