"""
Evaluation metrics for Khmer OCR model assessment.
Implements Character Error Rate (CER) and other metrics for model evaluation.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import Levenshtein
from collections import Counter
import re


class OCRMetrics:
    """
    Comprehensive metrics for OCR model evaluation.
    
    Includes:
    - Character Error Rate (CER)
    - Word Error Rate (WER) 
    - Sequence Accuracy
    - BLEU score for sequence similarity
    - Character-level precision, recall, F1
    """
    
    def __init__(self, ignore_case: bool = False, ignore_punctuation: bool = False):
        """
        Initialize metrics calculator.
        
        Args:
            ignore_case: Whether to ignore case differences
            ignore_punctuation: Whether to ignore punctuation
        """
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        
        # Punctuation patterns for Khmer and English
        self.punctuation_pattern = re.compile(r'[។៕៖ៗ៘៙៚\',\.\?\!;:"\(\)\[\]\{\}]')
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for metric calculation.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text
        """
        if self.ignore_case:
            text = text.lower()
        
        if self.ignore_punctuation:
            text = self.punctuation_pattern.sub('', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def character_error_rate(
        self, 
        predictions: Union[str, List[str]], 
        targets: Union[str, List[str]]
    ) -> float:
        """
        Calculate Character Error Rate (CER).
        
        CER = (Substitutions + Deletions + Insertions) / Total_Target_Characters
        
        Args:
            predictions: Predicted text(s)
            targets: Ground truth text(s)
            
        Returns:
            Character Error Rate as float between 0 and 1
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(targets, str):
            targets = [targets]
        
        if len(predictions) != len(targets):
            raise ValueError("Number of predictions must match number of targets")
        
        total_chars = 0
        total_errors = 0
        
        for pred, target in zip(predictions, targets):
            pred_clean = self.preprocess_text(pred)
            target_clean = self.preprocess_text(target)
            
            # Calculate edit distance
            edit_distance = Levenshtein.distance(pred_clean, target_clean)
            target_length = len(target_clean)
            
            if target_length > 0:
                total_chars += target_length
                total_errors += edit_distance
        
        return total_errors / total_chars if total_chars > 0 else 0.0
    
    def word_error_rate(
        self, 
        predictions: Union[str, List[str]], 
        targets: Union[str, List[str]]
    ) -> float:
        """
        Calculate Word Error Rate (WER).
        
        WER = (Word_Substitutions + Word_Deletions + Word_Insertions) / Total_Target_Words
        
        Args:
            predictions: Predicted text(s)
            targets: Ground truth text(s)
            
        Returns:
            Word Error Rate as float between 0 and 1
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(targets, str):
            targets = [targets]
        
        total_words = 0
        total_errors = 0
        
        for pred, target in zip(predictions, targets):
            pred_clean = self.preprocess_text(pred)
            target_clean = self.preprocess_text(target)
            
            # Split into words (handle Khmer text which may not have spaces)
            pred_words = self._split_words(pred_clean)
            target_words = self._split_words(target_clean)
            
            # Calculate word-level edit distance
            edit_distance = Levenshtein.distance(
                ' '.join(pred_words), 
                ' '.join(target_words)
            )
            
            if len(target_words) > 0:
                total_words += len(target_words)
                total_errors += edit_distance
        
        return total_errors / total_words if total_words > 0 else 0.0
    
    def sequence_accuracy(
        self, 
        predictions: Union[str, List[str]], 
        targets: Union[str, List[str]]
    ) -> float:
        """
        Calculate exact sequence match accuracy.
        
        Args:
            predictions: Predicted text(s)
            targets: Ground truth text(s)
            
        Returns:
            Sequence accuracy as float between 0 and 1
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(targets, str):
            targets = [targets]
        
        correct = 0
        total = len(predictions)
        
        for pred, target in zip(predictions, targets):
            pred_clean = self.preprocess_text(pred)
            target_clean = self.preprocess_text(target)
            
            if pred_clean == target_clean:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def character_level_metrics(
        self, 
        predictions: Union[str, List[str]], 
        targets: Union[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate character-level precision, recall, and F1 score.
        
        Args:
            predictions: Predicted text(s)
            targets: Ground truth text(s)
            
        Returns:
            Dictionary with precision, recall, f1 scores
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(targets, str):
            targets = [targets]
        
        all_pred_chars = []
        all_target_chars = []
        
        for pred, target in zip(predictions, targets):
            pred_clean = self.preprocess_text(pred)
            target_clean = self.preprocess_text(target)
            
            all_pred_chars.extend(list(pred_clean))
            all_target_chars.extend(list(target_clean))
        
        # Count character frequencies
        pred_counter = Counter(all_pred_chars)
        target_counter = Counter(all_target_chars)
        
        # Calculate metrics
        all_chars = set(all_pred_chars + all_target_chars)
        
        tp = sum(min(pred_counter[char], target_counter[char]) for char in all_chars)
        fp = sum(max(0, pred_counter[char] - target_counter[char]) for char in all_chars)
        fn = sum(max(0, target_counter[char] - pred_counter[char]) for char in all_chars)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def bleu_score(
        self, 
        predictions: Union[str, List[str]], 
        targets: Union[str, List[str]],
        n_grams: int = 4
    ) -> float:
        """
        Calculate BLEU score for sequence similarity.
        
        Args:
            predictions: Predicted text(s)
            targets: Ground truth text(s)
            n_grams: Maximum n-gram length
            
        Returns:
            BLEU score as float between 0 and 1
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(targets, str):
            targets = [targets]
        
        # Calculate for each n-gram level
        precisions = []
        
        for n in range(1, n_grams + 1):
            total_matches = 0
            total_pred_ngrams = 0
            
            for pred, target in zip(predictions, targets):
                pred_clean = self.preprocess_text(pred)
                target_clean = self.preprocess_text(target)
                
                pred_ngrams = self._get_ngrams(pred_clean, n)
                target_ngrams = self._get_ngrams(target_clean, n)
                
                # Count matches
                matches = 0
                for ngram in pred_ngrams:
                    if ngram in target_ngrams:
                        matches += min(pred_ngrams[ngram], target_ngrams[ngram])
                
                total_matches += matches
                total_pred_ngrams += sum(pred_ngrams.values())
            
            precision = total_matches / total_pred_ngrams if total_pred_ngrams > 0 else 0.0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0
        
        # Brevity penalty (simplified)
        total_pred_len = sum(len(self.preprocess_text(pred)) for pred in predictions)
        total_target_len = sum(len(self.preprocess_text(target)) for target in targets)
        
        if total_pred_len < total_target_len and total_target_len > 0 and total_pred_len > 0:
            bp = np.exp(1 - total_target_len / total_pred_len)
            bleu *= bp
        elif total_pred_len == 0:
            bleu = 0.0  # No prediction, no BLEU
        
        return bleu
    
    def comprehensive_evaluation(
        self, 
        predictions: List[str], 
        targets: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all metrics in one go.
        
        Args:
            predictions: List of predicted texts
            targets: List of ground truth texts
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['cer'] = self.character_error_rate(predictions, targets)
        metrics['wer'] = self.word_error_rate(predictions, targets)
        metrics['sequence_accuracy'] = self.sequence_accuracy(predictions, targets)
        
        # Character-level metrics
        char_metrics = self.character_level_metrics(predictions, targets)
        metrics.update({f'char_{k}': v for k, v in char_metrics.items()})
        
        # BLEU score
        metrics['bleu'] = self.bleu_score(predictions, targets)
        
        # Additional statistics
        metrics['avg_pred_length'] = np.mean([len(self.preprocess_text(p)) for p in predictions])
        metrics['avg_target_length'] = np.mean([len(self.preprocess_text(t)) for t in targets])
        metrics['total_samples'] = len(predictions)
        
        return metrics
    
    def _split_words(self, text: str) -> List[str]:
        """
        Split text into words (handles both spaced and non-spaced text).
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # For Khmer text, spaces might not separate all words
        # This is a simplified approach - could be enhanced with proper tokenization
        
        if ' ' in text:
            # Text has spaces, use them
            return text.split()
        else:
            # No spaces, treat as character sequence (for CJK-like languages)
            # Or could implement more sophisticated word segmentation
            return list(text)
    
    def _get_ngrams(self, text: str, n: int) -> Counter:
        """
        Get n-grams from text.
        
        Args:
            text: Input text
            n: N-gram length
            
        Returns:
            Counter of n-grams
        """
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return Counter(ngrams)


class ConfidenceMetrics:
    """
    Metrics for evaluating model confidence and calibration.
    """
    
    def __init__(self):
        pass
    
    def confidence_accuracy_correlation(
        self,
        predictions: List[str],
        targets: List[str],
        confidences: List[float]
    ) -> float:
        """
        Calculate correlation between confidence and accuracy.
        
        Args:
            predictions: Predicted texts
            targets: Ground truth texts
            confidences: Model confidence scores
            
        Returns:
            Pearson correlation coefficient
        """
        # Calculate accuracy for each sample
        accuracies = []
        for pred, target in zip(predictions, targets):
            accuracy = 1.0 if pred.strip() == target.strip() else 0.0
            accuracies.append(accuracy)
        
        # Calculate correlation
        if len(set(confidences)) == 1 or len(set(accuracies)) == 1:
            return 0.0  # No variation in one of the variables
        
        correlation = np.corrcoef(confidences, accuracies)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def calibration_curve(
        self,
        predictions: List[str],
        targets: List[str],
        confidences: List[float],
        n_bins: int = 10
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate calibration curve for confidence scores.
        
        Args:
            predictions: Predicted texts
            targets: Ground truth texts
            confidences: Model confidence scores
            n_bins: Number of bins for calibration curve
            
        Returns:
            Tuple of (bin_boundaries, bin_accuracies)
        """
        # Calculate accuracy for each sample
        accuracies = []
        for pred, target in zip(predictions, targets):
            accuracy = 1.0 if pred.strip() == target.strip() else 0.0
            accuracies.append(accuracy)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies = []
        
        for i in range(n_bins):
            # Find samples in this bin
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if i == n_bins - 1:  # Include right boundary for last bin
                mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if np.sum(mask) > 0:
                bin_accuracy = np.mean([accuracies[j] for j, m in enumerate(mask) if m])
                bin_accuracies.append(bin_accuracy)
            else:
                bin_accuracies.append(0.0)
        
        return bin_centers.tolist(), bin_accuracies


def evaluate_model_predictions(
    predictions: List[str],
    targets: List[str],
    confidences: Optional[List[float]] = None,
    detailed: bool = True
) -> Dict[str, any]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        predictions: List of predicted texts
        targets: List of ground truth texts
        confidences: Optional confidence scores
        detailed: Whether to include detailed metrics
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize metrics
    ocr_metrics = OCRMetrics()
    
    # Calculate basic metrics
    results = ocr_metrics.comprehensive_evaluation(predictions, targets)
    
    if detailed:
        # Add per-sample analysis
        sample_results = []
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            sample_cer = ocr_metrics.character_error_rate(pred, target)
            sample_acc = ocr_metrics.sequence_accuracy(pred, target)
            
            sample_result = {
                'index': i,
                'prediction': pred,
                'target': target,
                'cer': sample_cer,
                'accuracy': sample_acc,
                'pred_length': len(pred),
                'target_length': len(target)
            }
            
            if confidences and i < len(confidences):
                sample_result['confidence'] = confidences[i]
            
            sample_results.append(sample_result)
        
        results['samples'] = sample_results
    
    # Confidence analysis if available
    if confidences:
        conf_metrics = ConfidenceMetrics()
        
        results['confidence_accuracy_correlation'] = conf_metrics.confidence_accuracy_correlation(
            predictions, targets, confidences
        )
        
        bin_centers, bin_accuracies = conf_metrics.calibration_curve(
            predictions, targets, confidences
        )
        results['calibration'] = {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies
        }
    
    return results 