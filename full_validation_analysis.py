#!/usr/bin/env python3
"""
Full Validation Analysis Script for Khmer OCR Seq2Seq Model

This script performs comprehensive validation analysis on the full validation_fixed dataset:
- Loads model from checkpoint
- Runs inference on all validation samples (6,400 images)
- Calculates detailed metrics and performance statistics
- Provides error analysis and sample inspections
- Generates comprehensive reports and visualizations
- Saves results for further analysis

Usage:
    python full_validation_analysis.py --checkpoint models/checkpoints/best_model.pth --output validation_results/
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import json
import time
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from PIL import Image

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.synthetic_dataset import SyntheticImageDataset, SyntheticCollateFunction
from src.utils.config import ConfigManager
from src.inference.ocr_engine import KhmerOCREngine
from src.inference.metrics import OCRMetrics, evaluate_model_predictions
from src.training.validator import Validator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup comprehensive logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class FullValidationAnalyzer:
    """
    Comprehensive validation analyzer for Khmer OCR model.
    
    Provides detailed analysis including:
    - Standard metrics (CER, BLEU, etc.)
    - Performance statistics
    - Error analysis by text length, complexity
    - Confidence calibration
    - Sample-level detailed results
    """
    
    def __init__(
        self,
        model: KhmerOCRSeq2Seq,
        config_manager: ConfigManager,
        device: torch.device,
        output_dir: str
    ):
        self.model = model
        self.config = config_manager
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ocr_engine = KhmerOCREngine(
            model=model,
            vocab=config_manager.vocab,
            device=device,
            max_width=config_manager.data_config.image_width
        )
        self.metrics = OCRMetrics()
        self.validator = Validator(model, config_manager, device)
        
        # Results storage
        self.results = {}
        self.sample_results = []
        
        self.logger = logging.getLogger("FullValidationAnalyzer")
    
    def load_validation_dataset(
        self,
        validation_dir: str = "data/validation_fixed",
        batch_size: int = 64,  # Increased from 32 to 64 for better GPU utilization
        num_workers: int = 4
    ) -> DataLoader:
        """Load the full validation dataset."""
        self.logger.info(f"Loading validation dataset from: {validation_dir}")
        
        # Load full validation dataset
        val_dataset = SyntheticImageDataset(
            split="val",
            synthetic_dir=validation_dir,
            config_manager=self.config,
            max_samples=None  # Load all samples
        )
        
        # Create dataloader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep deterministic order for analysis
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=SyntheticCollateFunction(self.config.vocab)
        )
        
        self.logger.info(f"Loaded {len(val_dataset)} validation samples in {len(val_dataloader)} batches")
        return val_dataloader
    
    def load_combined_validation_datasets(
        self,
        validation_dirs: List[str] = ["data/validation_fixed", "data/validation_short_text"],
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[Dict[str, DataLoader], DataLoader]:
        """
        Load multiple validation datasets and return both individual and combined dataloaders.
        
        Returns:
            Tuple of (individual_dataloaders_dict, combined_dataloader)
        """
        self.logger.info(f"Loading validation datasets from: {validation_dirs}")
        
        individual_dataloaders = {}
        all_datasets = []
        dataset_labels = []
        
        for val_dir in validation_dirs:
            if not Path(val_dir).exists():
                self.logger.warning(f"Validation directory not found: {val_dir}, skipping...")
                continue
            
            # Determine dataset name and split
            dataset_name = Path(val_dir).name
            split = "val" if "validation_fixed" in val_dir else ""  # Use "val" for validation_fixed, "" for validation_short_text
            
            self.logger.info(f"Loading {dataset_name} dataset with split='{split}'")
            
            # Load individual dataset
            try:
                val_dataset = SyntheticImageDataset(
                    split=split,
                    synthetic_dir=val_dir,
                    config_manager=self.config,
                    max_samples=None  # Load all samples
                )
                
                # Create individual dataloader
                individual_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=SyntheticCollateFunction(self.config.vocab)
                )
                
                individual_dataloaders[dataset_name] = individual_dataloader
                
                # Add to combined dataset
                all_datasets.append(val_dataset)
                dataset_labels.extend([dataset_name] * len(val_dataset))
                
                self.logger.info(f"Loaded {len(val_dataset)} samples from {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset from {val_dir}: {e}")
                continue
        
        if not all_datasets:
            raise RuntimeError("No validation datasets could be loaded")
        
        # Create combined dataset
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(all_datasets)
        
        # Create combined dataloader
        combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=SyntheticCollateFunction(self.config.vocab)
        )
        
        # Store dataset labels for analysis
        self.dataset_labels = dataset_labels
        
        total_samples = sum(len(ds) for ds in all_datasets)
        self.logger.info(f"Created combined validation dataset with {total_samples} total samples")
        
        return individual_dataloaders, combined_dataloader
    
    def run_comprehensive_validation(self, dataloader: DataLoader, individual_dataloaders: Optional[Dict[str, DataLoader]] = None) -> Dict:
        """Run comprehensive validation analysis."""
        self.logger.info("Starting comprehensive validation analysis...")
        
        start_time = time.time()
        
        # Standard validation metrics on combined dataset
        self.logger.info("Computing standard validation metrics on combined dataset...")
        standard_metrics = self.validator.validate(dataloader)
        
        # Individual dataset analysis if provided
        individual_results = {}
        if individual_dataloaders:
            self.logger.info("Running individual dataset analysis...")
            for dataset_name, individual_dataloader in individual_dataloaders.items():
                self.logger.info(f"Analyzing {dataset_name} dataset...")
                individual_metrics = self.validator.validate(individual_dataloader)
                individual_detailed = self._analyze_all_samples(individual_dataloader, dataset_label=dataset_name)
                
                individual_results[dataset_name] = {
                    'standard_metrics': individual_metrics,
                    'sample_results': individual_detailed,
                    'performance_analysis': self._analyze_performance(individual_detailed),
                    'error_analysis': self._analyze_errors(individual_detailed)
                }
        
        # Detailed sample-by-sample analysis on combined dataset
        self.logger.info("Running detailed sample analysis on combined dataset...")
        detailed_results = self._analyze_all_samples(dataloader)
        
        # Performance analysis
        self.logger.info("Analyzing performance characteristics...")
        performance_analysis = self._analyze_performance(detailed_results)
        
        # Error analysis
        self.logger.info("Conducting error analysis...")
        error_analysis = self._analyze_errors(detailed_results)
        
        # Confidence analysis (if available)
        confidence_analysis = self._analyze_confidence(detailed_results)
        
        # Dataset comparison analysis
        dataset_comparison = {}
        if hasattr(self, 'dataset_labels') and len(set(self.dataset_labels)) > 1:
            self.logger.info("Performing dataset comparison analysis...")
            dataset_comparison = self._analyze_dataset_comparison(detailed_results)
        
        total_time = time.time() - start_time
        
        # Compile results
        self.results = {
            'metadata': {
                'total_samples': len(detailed_results),
                'analysis_time': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'datasets_analyzed': list(individual_dataloaders.keys()) if individual_dataloaders else ['combined'],
                'model_config': {
                    'vocab_size': len(self.config.vocab),
                    'max_width': self.config.data_config.image_width,
                    'image_height': self.config.data_config.image_height
                }
            },
            'combined_analysis': {
                'standard_metrics': standard_metrics,
                'performance_analysis': performance_analysis,
                'error_analysis': error_analysis,
                'confidence_analysis': confidence_analysis
            },
            'individual_datasets': individual_results,
            'dataset_comparison': dataset_comparison,
            'sample_results': detailed_results[:100]  # Store first 100 for detailed inspection
        }
        
        self.sample_results = detailed_results
        
        self.logger.info(f"Validation analysis completed in {total_time:.2f} seconds")
        return self.results
    
    def _analyze_all_samples(self, dataloader: DataLoader, dataset_label: Optional[str] = None) -> List[Dict]:
        """Analyze all samples individually with detailed metrics."""
        sample_results = []
        self.model.eval()
        
        # Track dataset assignment for combined analysis
        dataset_labels = getattr(self, 'dataset_labels', None)
        current_sample_idx = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    progress = 100.0 * batch_idx / len(dataloader)
                    dataset_info = f" ({dataset_label})" if dataset_label else ""
                    self.logger.info(f"Processing batch {batch_idx}/{len(dataloader)} ({progress:.1f}%){dataset_info}")
                
                # Get batch data
                images = batch['images'].to(self.device)
                targets = batch['targets']
                texts = batch.get('texts', [])
                
                batch_size = images.size(0)
                
                # **OPTIMIZED: Process entire batch at once**
                batch_inference_start = time.time()
                
                try:
                    # Direct batch inference with the model (much faster)
                    result = self.model.generate(
                        images=images,  # Process entire batch
                        max_length=256,
                        method='greedy'
                    )
                    
                    predictions = result['sequences']  # (batch_size, seq_len)
                    batch_inference_time = time.time() - batch_inference_start
                    avg_inference_time = batch_inference_time / batch_size
                    
                    # Process batch results
                    for i in range(batch_size):
                        sample_target = targets[i]
                        sample_text = texts[i] if i < len(texts) else ""
                        
                        # Determine dataset source for this sample
                        if dataset_label:
                            sample_dataset = dataset_label
                        elif dataset_labels and current_sample_idx < len(dataset_labels):
                            sample_dataset = dataset_labels[current_sample_idx]
                        else:
                            sample_dataset = "unknown"
                        
                        # Decode prediction (FIXED: Stop at first EOS like engine.recognize())
                        pred_sequence = predictions[i].cpu().tolist()
                        
                        # Use the same logic as OCR engine - stop at first EOS
                        clean_tokens = []
                        for token_id in pred_sequence:
                            if token_id == self.config.vocab.EOS_IDX:
                                break  # Stop at first EOS
                            if token_id not in [self.config.vocab.SOS_IDX, self.config.vocab.PAD_IDX, self.config.vocab.UNK_IDX]:
                                clean_tokens.append(token_id)
                        
                        prediction = self.config.vocab.decode(clean_tokens)
                        
                        # Decode target (also stop at first EOS for consistency)
                        target_sequence = sample_target.numpy().tolist()
                        clean_target_tokens = []
                        for token_id in target_sequence:
                            if token_id == self.config.vocab.EOS_IDX:
                                break  # Stop at first EOS
                            if token_id not in [self.config.vocab.SOS_IDX, self.config.vocab.PAD_IDX, self.config.vocab.UNK_IDX]:
                                clean_target_tokens.append(token_id)
                        
                        target_text = self.config.vocab.decode(clean_target_tokens)
                        
                        # Calculate metrics
                        sample_cer = self.metrics.character_error_rate(prediction, target_text)
                        sample_wer = self.metrics.word_error_rate(prediction, target_text)
                        sequence_acc = self.metrics.sequence_accuracy(prediction, target_text)
                        
                        # Categorize text length for analysis
                        text_length_category = self._get_length_category(len(target_text))
                        
                        # Store detailed result
                        sample_result = {
                            'sample_id': len(sample_results),
                            'dataset': sample_dataset,
                            'batch_idx': batch_idx,
                            'batch_position': i,
                            'prediction': prediction,
                            'target': target_text,
                            'original_text': sample_text,
                            'cer': sample_cer,
                            'wer': sample_wer,
                            'sequence_accuracy': sequence_acc,
                            'confidence': 0.5,  # Placeholder for batch processing
                            'inference_time': avg_inference_time,
                            'pred_length': len(prediction),
                            'target_length': len(target_text),
                            'text_length_category': text_length_category,
                            'image_width': images[i].shape[-1],
                            'correct_prediction': prediction.strip() == target_text.strip()
                        }
                        
                        sample_results.append(sample_result)
                        current_sample_idx += 1
                
                except Exception as e:
                    self.logger.warning(f"Error during batch inference for batch {batch_idx}: {e}")
                    # Fallback to individual processing for this batch
                    avg_inference_time = batch_inference_time / batch_size if 'batch_inference_time' in locals() else 0.1
                    
                    for i in range(batch_size):
                        sample_target = targets[i]
                        sample_text = texts[i] if i < len(texts) else ""
                        
                        # Determine dataset source for this sample
                        if dataset_label:
                            sample_dataset = dataset_label
                        elif dataset_labels and current_sample_idx < len(dataset_labels):
                            sample_dataset = dataset_labels[current_sample_idx]
                        else:
                            sample_dataset = "unknown"
                        
                        # Decode target (fallback - also stop at first EOS for consistency)
                        target_sequence = sample_target.numpy().tolist()
                        clean_target_tokens = []
                        for token_id in target_sequence:
                            if token_id == self.config.vocab.EOS_IDX:
                                break  # Stop at first EOS
                            if token_id not in [self.config.vocab.SOS_IDX, self.config.vocab.PAD_IDX, self.config.vocab.UNK_IDX]:
                                clean_target_tokens.append(token_id)
                        
                        target_text = self.config.vocab.decode(clean_target_tokens)
                        text_length_category = self._get_length_category(len(target_text))
                        
                        sample_result = {
                            'sample_id': len(sample_results),
                            'dataset': sample_dataset,
                            'batch_idx': batch_idx,
                            'batch_position': i,
                            'prediction': "",
                            'target': target_text,
                            'original_text': sample_text,
                            'cer': 1.0,
                            'wer': 1.0,
                            'sequence_accuracy': 0.0,
                            'confidence': 0.0,
                            'inference_time': avg_inference_time,
                            'pred_length': 0,
                            'target_length': len(target_text),
                            'text_length_category': text_length_category,
                            'image_width': images[i].shape[-1],
                            'correct_prediction': False
                        }
                        
                        sample_results.append(sample_result)
                        current_sample_idx += 1
        
        return sample_results
    
    def _get_length_category(self, length: int) -> str:
        """Categorize text length for analysis."""
        if length <= 5:
            return "very_short (1-5)"
        elif length <= 10:
            return "short (6-10)"
        elif length <= 20:
            return "medium (11-20)"
        elif length <= 50:
            return "long (21-50)"
        else:
            return "very_long (51+)"
    
    def _analyze_performance(self, sample_results: List[Dict]) -> Dict:
        """Analyze performance characteristics."""
        df = pd.DataFrame(sample_results)
        
        analysis = {
            'overall_stats': {
                'mean_cer': df['cer'].mean(),
                'std_cer': df['cer'].std(),
                'median_cer': df['cer'].median(),
                'mean_confidence': df['confidence'].mean(),
                'mean_inference_time': df['inference_time'].mean(),
                'sequence_accuracy': df['sequence_accuracy'].mean(),
                'exact_match_accuracy': df['correct_prediction'].mean()
            },
            'length_analysis': self._analyze_by_length(df),
            'speed_analysis': {
                'mean_inference_time_ms': df['inference_time'].mean() * 1000,
                'std_inference_time_ms': df['inference_time'].std() * 1000,
                'samples_per_second': 1.0 / df['inference_time'].mean(),
                'fastest_sample_ms': df['inference_time'].min() * 1000,
                'slowest_sample_ms': df['inference_time'].max() * 1000
            }
        }
        
        return analysis
    
    def _analyze_by_length(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by text length."""
        length_bins = [0, 10, 20, 50, 100, float('inf')]
        length_labels = ['0-10', '11-20', '21-50', '51-100', '100+']
        
        df['length_bin'] = pd.cut(df['target_length'], bins=length_bins, labels=length_labels, right=True)
        
        length_analysis = {}
        for label in length_labels:
            subset = df[df['length_bin'] == label]
            if len(subset) > 0:
                length_analysis[label] = {
                    'count': len(subset),
                    'mean_cer': subset['cer'].mean(),
                    'mean_confidence': subset['confidence'].mean(),
                    'sequence_accuracy': subset['sequence_accuracy'].mean(),
                    'mean_inference_time_ms': subset['inference_time'].mean() * 1000
                }
        
        return length_analysis
    
    def _analyze_errors(self, sample_results: List[Dict]) -> Dict:
        """Analyze error patterns and characteristics."""
        df = pd.DataFrame(sample_results)
        
        # Find high-error samples
        high_error_threshold = 0.3  # 30% CER
        high_error_samples = df[df['cer'] > high_error_threshold]
        
        # Find perfect predictions
        perfect_predictions = df[df['cer'] == 0.0]
        
        error_analysis = {
            'error_distribution': {
                'zero_error_count': len(df[df['cer'] == 0.0]),
                'low_error_count': len(df[df['cer'] <= 0.1]),  # <= 10%
                'medium_error_count': len(df[(df['cer'] > 0.1) & (df['cer'] <= 0.3)]),  # 10-30%
                'high_error_count': len(df[df['cer'] > 0.3]),  # > 30%
                'zero_error_percentage': len(df[df['cer'] == 0.0]) / len(df) * 100,
                'low_error_percentage': len(df[df['cer'] <= 0.1]) / len(df) * 100
            },
            'worst_samples': df.nlargest(10, 'cer')[['sample_id', 'prediction', 'target', 'cer']].to_dict('records'),
            'best_samples': df.nsmallest(10, 'cer')[['sample_id', 'prediction', 'target', 'cer']].to_dict('records'),
            'error_by_length': self._error_by_length_analysis(df)
        }
        
        return error_analysis
    
    def _error_by_length_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze how errors correlate with text length."""
        length_ranges = [(0, 10), (11, 20), (21, 50), (51, 100), (101, 1000)]
        error_by_length = {}
        
        for start, end in length_ranges:
            subset = df[(df['target_length'] >= start) & (df['target_length'] <= end)]
            if len(subset) > 0:
                error_by_length[f'{start}-{end}'] = {
                    'count': len(subset),
                    'mean_cer': subset['cer'].mean(),
                    'std_cer': subset['cer'].std(),
                    'zero_error_rate': (subset['cer'] == 0.0).mean()
                }
        
        return error_by_length
    
    def _analyze_confidence(self, sample_results: List[Dict]) -> Dict:
        """Analyze confidence scores and calibration."""
        df = pd.DataFrame(sample_results)
        
        # Confidence calibration analysis
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (df['confidence'] >= confidence_bins[i]) & (df['confidence'] < confidence_bins[i + 1])
            bin_data = df[bin_mask]
            
            if len(bin_data) > 0:
                bin_accuracy = bin_data['correct_prediction'].mean()
                bin_confidence = bin_data['confidence'].mean()
                bin_count = len(bin_data)
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(bin_count)
        
        confidence_analysis = {
            'confidence_stats': {
                'mean_confidence': df['confidence'].mean(),
                'std_confidence': df['confidence'].std(),
                'min_confidence': df['confidence'].min(),
                'max_confidence': df['confidence'].max()
            },
            'calibration': {
                'bin_confidences': bin_confidences,
                'bin_accuracies': bin_accuracies,
                'bin_counts': bin_counts
            },
            'confidence_accuracy_correlation': df['confidence'].corr(df['correct_prediction'].astype(float))
        }
        
        return confidence_analysis
    
    def _analyze_dataset_comparison(self, sample_results: List[Dict]) -> Dict:
        """Analyze performance differences between datasets."""
        df = pd.DataFrame(sample_results)
        
        dataset_comparison = {}
        unique_datasets = df['dataset'].unique()
        
        for dataset in unique_datasets:
            dataset_df = df[df['dataset'] == dataset]
            
            dataset_comparison[dataset] = {
                'sample_count': len(dataset_df),
                'mean_cer': dataset_df['cer'].mean(),
                'median_cer': dataset_df['cer'].median(),
                'std_cer': dataset_df['cer'].std(),
                'sequence_accuracy': dataset_df['sequence_accuracy'].mean(),
                'exact_match_accuracy': dataset_df['correct_prediction'].mean(),
                'mean_target_length': dataset_df['target_length'].mean(),
                'mean_inference_time_ms': dataset_df['inference_time'].mean() * 1000,
                'length_distribution': self._get_length_distribution(dataset_df),
                'error_distribution': self._get_error_distribution(dataset_df)
            }
        
        # Add comparison statistics
        if len(unique_datasets) >= 2:
            dataset_comparison['comparison_summary'] = self._compute_dataset_comparison_stats(df)
        
        return dataset_comparison
    
    def _get_length_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution of text lengths for a dataset."""
        length_dist = {}
        for category in ["very_short (1-5)", "short (6-10)", "medium (11-20)", "long (21-50)", "very_long (51+)"]:
            count = len(df[df['text_length_category'] == category])
            percentage = count / len(df) * 100 if len(df) > 0 else 0
            length_dist[category] = {
                'count': count,
                'percentage': percentage,
                'mean_cer': df[df['text_length_category'] == category]['cer'].mean() if count > 0 else 0
            }
        return length_dist
    
    def _get_error_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution of error rates for a dataset."""
        return {
            'perfect_predictions': len(df[df['cer'] == 0.0]),
            'low_error_le_10pct': len(df[df['cer'] <= 0.1]),
            'medium_error_10_30pct': len(df[(df['cer'] > 0.1) & (df['cer'] <= 0.3)]),
            'high_error_gt_30pct': len(df[df['cer'] > 0.3]),
            'perfect_percentage': len(df[df['cer'] == 0.0]) / len(df) * 100 if len(df) > 0 else 0,
            'low_error_percentage': len(df[df['cer'] <= 0.1]) / len(df) * 100 if len(df) > 0 else 0
        }
    
    def _compute_dataset_comparison_stats(self, df: pd.DataFrame) -> Dict:
        """Compute comparison statistics between datasets."""
        comparison = {}
        
        # Get datasets
        datasets = df['dataset'].unique()
        if len(datasets) >= 2:
            # Assume first dataset is general validation, second is short text
            general_dataset = "validation_fixed" if "validation_fixed" in datasets else datasets[0]
            short_text_dataset = "validation_short_text" if "validation_short_text" in datasets else datasets[1]
            
            general_df = df[df['dataset'] == general_dataset]
            short_text_df = df[df['dataset'] == short_text_dataset]
            
            comparison = {
                'general_vs_short_text': {
                    'general_dataset': general_dataset,
                    'short_text_dataset': short_text_dataset,
                    'cer_difference': short_text_df['cer'].mean() - general_df['cer'].mean(),
                    'performance_gap_percentage': (short_text_df['cer'].mean() - general_df['cer'].mean()) * 100,
                    'general_cer': general_df['cer'].mean(),
                    'short_text_cer': short_text_df['cer'].mean(),
                    'general_sequence_acc': general_df['sequence_accuracy'].mean(),
                    'short_text_sequence_acc': short_text_df['sequence_accuracy'].mean(),
                    'improvement_needed': {
                        'absolute_cer_reduction_needed': short_text_df['cer'].mean() - general_df['cer'].mean(),
                        'relative_improvement_factor': short_text_df['cer'].mean() / general_df['cer'].mean() if general_df['cer'].mean() > 0 else float('inf')
                    }
                }
            }
        
        return comparison
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        self.logger.info("Generating visualizations...")
        
        df = pd.DataFrame(self.sample_results)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots - expanded for dataset comparison
        fig = plt.figure(figsize=(24, 28))
        
        # 1. CER Distribution
        plt.subplot(5, 3, 1)
        plt.hist(df['cer'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Character Error Rate')
        plt.ylabel('Frequency')
        plt.title('Overall CER Distribution')
        plt.axvline(df['cer'].mean(), color='red', linestyle='--', label=f'Mean: {df["cer"].mean():.3f}')
        plt.legend()
        
        # 2. CER by Dataset (if multiple datasets)
        if 'dataset' in df.columns and len(df['dataset'].unique()) > 1:
            plt.subplot(5, 3, 2)
            datasets = df['dataset'].unique()
            for dataset in datasets:
                dataset_df = df[df['dataset'] == dataset]
                plt.hist(dataset_df['cer'], bins=30, alpha=0.6, label=f'{dataset} (n={len(dataset_df)})', edgecolor='black')
            plt.xlabel('Character Error Rate')
            plt.ylabel('Frequency')
            plt.title('CER Distribution by Dataset')
            plt.legend()
        else:
            # 2. CER vs Text Length
            plt.subplot(5, 3, 2)
            plt.scatter(df['target_length'], df['cer'], alpha=0.5, s=1)
            plt.xlabel('Target Text Length')
            plt.ylabel('Character Error Rate')
            plt.title('CER vs Text Length')
        
        # 3. Dataset Performance Comparison (if multiple datasets)
        if 'dataset' in df.columns and len(df['dataset'].unique()) > 1:
            plt.subplot(5, 3, 3)
            datasets = df['dataset'].unique()
            cer_means = [df[df['dataset'] == dataset]['cer'].mean() for dataset in datasets]
            bars = plt.bar(datasets, cer_means, color=['lightblue', 'lightcoral'][:len(datasets)])
            plt.ylabel('Mean CER')
            plt.title('Mean CER by Dataset')
            plt.xticks(rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars, cer_means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        else:
            # 3. Confidence vs Accuracy
            plt.subplot(5, 3, 3)
            plt.scatter(df['confidence'], df['correct_prediction'].astype(float), alpha=0.5, s=1)
            plt.xlabel('Confidence Score')
            plt.ylabel('Correct Prediction (0/1)')
            plt.title('Confidence vs Accuracy')
        
        # 4. Text Length Category Performance
        plt.subplot(5, 3, 4)
        if 'text_length_category' in df.columns:
            categories = ["very_short (1-5)", "short (6-10)", "medium (11-20)", "long (21-50)", "very_long (51+)"]
            category_data = []
            category_counts = []
            
            for cat in categories:
                cat_df = df[df['text_length_category'] == cat]
                if len(cat_df) > 0:
                    category_data.append(cat_df['cer'].mean())
                    category_counts.append(len(cat_df))
                else:
                    category_data.append(0)
                    category_counts.append(0)
            
            bars = plt.bar(range(len(categories)), category_data)
            plt.xlabel('Text Length Category')
            plt.ylabel('Mean CER')
            plt.title('Mean CER by Text Length')
            plt.xticks(range(len(categories)), [cat.split()[0] for cat in categories], rotation=45)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, category_counts)):
                if count > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 5. Dataset Length Distribution Comparison (if multiple datasets)
        if 'dataset' in df.columns and len(df['dataset'].unique()) > 1:
            plt.subplot(5, 3, 5)
            datasets = df['dataset'].unique()
            width = 0.35
            categories = ["very_short (1-5)", "short (6-10)", "medium (11-20)", "long (21-50)", "very_long (51+)"]
            
            for i, dataset in enumerate(datasets):
                dataset_df = df[df['dataset'] == dataset]
                counts = [len(dataset_df[dataset_df['text_length_category'] == cat]) for cat in categories]
                percentages = [c/len(dataset_df)*100 if len(dataset_df) > 0 else 0 for c in counts]
                
                x_pos = np.arange(len(categories)) + i * width
                plt.bar(x_pos, percentages, width, label=dataset, alpha=0.7)
            
            plt.xlabel('Text Length Category')
            plt.ylabel('Percentage of Samples')
            plt.title('Length Distribution by Dataset')
            plt.xticks(np.arange(len(categories)) + width/2, [cat.split()[0] for cat in categories], rotation=45)
            plt.legend()
        else:
            # 5. Inference Time Distribution
            plt.subplot(5, 3, 5)
            plt.hist(df['inference_time'] * 1000, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('Frequency')
            plt.title('Inference Time Distribution')
        
        # 6. Error Distribution
        error_dist = self.results['combined_analysis']['error_analysis']['error_distribution']
        plt.subplot(5, 3, 6)
        categories = ['Perfect\n(0%)', 'Low\n(≤10%)', 'Medium\n(10-30%)', 'High\n(>30%)']
        counts = [error_dist['zero_error_count'], 
                 error_dist['low_error_count'] - error_dist['zero_error_count'],
                 error_dist['medium_error_count'], 
                 error_dist['high_error_count']]
        colors = ['green', 'lightgreen', 'orange', 'red']
        
        plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Error Rate Distribution')
        
        # 7. Confidence Calibration
        calib = self.results['combined_analysis']['confidence_analysis']['calibration']
        if calib['bin_confidences'] and calib['bin_accuracies']:
            plt.subplot(5, 3, 7)
            plt.plot(calib['bin_confidences'], calib['bin_accuracies'], 'bo-', label='Model')
            plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            plt.xlabel('Mean Confidence')
            plt.ylabel('Accuracy')
            plt.title('Confidence Calibration')
            plt.legend()
        
        # 8. CER vs Confidence
        plt.subplot(5, 3, 8)
        plt.scatter(df['confidence'], df['cer'], alpha=0.5, s=1)
        plt.xlabel('Confidence Score')
        plt.ylabel('Character Error Rate')
        plt.title('CER vs Confidence')
        
        # 9. Performance Gap Visualization (if dataset comparison available)
        if 'dataset_comparison' in self.results and self.results['dataset_comparison']:
            plt.subplot(5, 3, 9)
            comparison = self.results['dataset_comparison']
            if 'comparison_summary' in comparison and 'general_vs_short_text' in comparison['comparison_summary']:
                gap_data = comparison['comparison_summary']['general_vs_short_text']
                
                metrics = ['CER', 'Sequence Accuracy']
                general_values = [gap_data['general_cer'], gap_data['general_sequence_acc']]
                short_text_values = [gap_data['short_text_cer'], gap_data['short_text_sequence_acc']]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                plt.bar(x - width/2, general_values, width, label='General Validation', alpha=0.7, color='lightblue')
                plt.bar(x + width/2, short_text_values, width, label='Short Text Validation', alpha=0.7, color='lightcoral')
                
                plt.xlabel('Metrics')
                plt.ylabel('Performance')
                plt.title('Performance Gap Analysis')
                plt.xticks(x, metrics)
                plt.legend()
                
                # Add gap annotation
                cer_gap = gap_data['performance_gap_percentage']
                plt.text(0, max(general_values[0], short_text_values[0]) * 1.1,
                        f'CER Gap: {cer_gap:.1f}%', ha='center', fontweight='bold')
        else:
            # 9. Sequence Accuracy by Length
            plt.subplot(5, 3, 9)
            length_analysis = self.results['combined_analysis']['performance_analysis']['length_analysis']
            if length_analysis:
                lengths = list(length_analysis.keys())
                seq_accs = [length_analysis[l]['sequence_accuracy'] for l in lengths]
                plt.bar(lengths, seq_accs)
                plt.xlabel('Text Length Range')
                plt.ylabel('Sequence Accuracy')
                plt.title('Sequence Accuracy by Length')
                plt.xticks(rotation=45)
        
        # 10. Overall Statistics
        plt.subplot(5, 3, 10)
        plt.axis('off')
        stats = self.results['combined_analysis']['performance_analysis']['overall_stats']
        
        # Add dataset-specific stats if available
        stats_text = f"""
        Overall Statistics:
        
        Mean CER: {stats['mean_cer']:.3f}
        Median CER: {stats['median_cer']:.3f}
        Sequence Accuracy: {stats['sequence_accuracy']:.3f}
        Exact Match: {stats['exact_match_accuracy']:.3f}
        
        Mean Confidence: {stats['mean_confidence']:.3f}
        Inference Speed: {1000/stats['mean_inference_time']:.1f} samples/s
        
        Total Samples: {len(df):,}
        """
        
        # Add dataset breakdown if available
        if 'individual_datasets' in self.results and self.results['individual_datasets']:
            stats_text += "\nDataset Breakdown:\n"
            for dataset_name, dataset_data in self.results['individual_datasets'].items():
                dataset_stats = dataset_data['performance_analysis']['overall_stats']
                stats_text += f"• {dataset_name}: {dataset_stats['mean_cer']:.3f} CER\n"
        
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # 11. Image Width Distribution
        plt.subplot(5, 3, 11)
        plt.hist(df['image_width'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Image Width (pixels)')
        plt.ylabel('Frequency')
        plt.title('Image Width Distribution')
        
        # 12. Prediction Length vs Target Length
        plt.subplot(5, 3, 12)
        plt.scatter(df['target_length'], df['pred_length'], alpha=0.5, s=1)
        plt.plot([0, df['target_length'].max()], [0, df['target_length'].max()], 'r--', label='Perfect Match')
        plt.xlabel('Target Length')
        plt.ylabel('Prediction Length')
        plt.title('Prediction vs Target Length')
        plt.legend()
        
        # 13. Short Text Performance Breakdown (if available)
        if 'dataset' in df.columns and 'validation_short_text' in df['dataset'].unique():
            plt.subplot(5, 3, 13)
            short_text_df = df[df['dataset'] == 'validation_short_text']
            
            length_categories = short_text_df['text_length_category'].value_counts()
            plt.pie(length_categories.values, labels=length_categories.index, autopct='%1.1f%%', startangle=90)
            plt.title('Short Text Dataset\nLength Distribution')
        
        # 14. Training Progress Indicator (if available)
        if 'dataset_comparison' in self.results and self.results['dataset_comparison']:
            plt.subplot(5, 3, 14)
            comparison = self.results['dataset_comparison']
            if 'comparison_summary' in comparison and 'general_vs_short_text' in comparison['comparison_summary']:
                gap_data = comparison['comparison_summary']['general_vs_short_text']
                
                # Create a visual showing the improvement target
                current_gap = gap_data['performance_gap_percentage']
                target_gap = 2.0  # Target: within 2% of general performance
                
                plt.barh(['Current Gap', 'Target Gap'], [current_gap, target_gap], 
                        color=['red' if current_gap > target_gap else 'green', 'lightgreen'])
                plt.xlabel('Performance Gap (%)')
                plt.title('Short Text Training Progress')
                
                if current_gap > target_gap:
                    plt.text(current_gap/2, 0, f'{current_gap:.1f}%', ha='center', va='center', fontweight='bold')
                    plt.text(target_gap/2, 1, f'{target_gap:.1f}%', ha='center', va='center', fontweight='bold')
        
        # 15. Dataset Sample Distribution
        if 'dataset' in df.columns and len(df['dataset'].unique()) > 1:
            plt.subplot(5, 3, 15)
            dataset_counts = df['dataset'].value_counts()
            plt.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Sample Distribution\nAcross Datasets')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'validation_analysis_visualizations.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to: {viz_path}")
    
    def save_results(self):
        """Save comprehensive results to files."""
        self.logger.info("Saving results...")
        
        # Save main results as JSON
        results_path = self.output_dir / 'validation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save detailed sample results as CSV
        samples_path = self.output_dir / 'detailed_sample_results.csv'
        df = pd.DataFrame(self.sample_results)
        df.to_csv(samples_path, index=False, encoding='utf-8')
        
        # Save summary report
        self._save_summary_report()
        
        # Save error samples for inspection
        self._save_error_samples()
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def _save_summary_report(self):
        """Save a human-readable summary report."""
        report_path = self.output_dir / 'validation_summary_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("KHMER OCR VALIDATION ANALYSIS SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Metadata
            meta = self.results['metadata']
            f.write(f"Analysis Date: {meta['timestamp']}\n")
            f.write(f"Total Samples: {meta['total_samples']:,}\n")
            f.write(f"Analysis Time: {meta['analysis_time']:.2f} seconds\n")
            f.write(f"Model Config: Max Width {meta['model_config']['max_width']}px, "
                   f"Vocab Size {meta['model_config']['vocab_size']}\n\n")
            
            # Overall Performance
            stats = self.results['combined_analysis']['performance_analysis']['overall_stats']
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Character Error Rate (CER): {stats['mean_cer']:.3f} ({stats['mean_cer']*100:.1f}%)\n")
            f.write(f"Median CER: {stats['median_cer']:.3f} ({stats['median_cer']*100:.1f}%)\n")
            f.write(f"Sequence Accuracy: {stats['sequence_accuracy']:.3f} ({stats['sequence_accuracy']*100:.1f}%)\n")
            f.write(f"Exact Match Accuracy: {stats['exact_match_accuracy']:.3f} ({stats['exact_match_accuracy']*100:.1f}%)\n")
            f.write(f"Mean Confidence: {stats['mean_confidence']:.3f}\n\n")
            
            # Speed Performance
            speed = self.results['combined_analysis']['performance_analysis']['speed_analysis']
            f.write("SPEED PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Inference Time: {speed['mean_inference_time_ms']:.1f} ms\n")
            f.write(f"Inference Speed: {speed['samples_per_second']:.1f} samples/second\n")
            f.write(f"Fastest Sample: {speed['fastest_sample_ms']:.1f} ms\n")
            f.write(f"Slowest Sample: {speed['slowest_sample_ms']:.1f} ms\n\n")
            
            # Error Analysis
            error_dist = self.results['combined_analysis']['error_analysis']['error_distribution']
            f.write("ERROR DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Perfect Predictions (0% CER): {error_dist['zero_error_count']:,} ({error_dist['zero_error_percentage']:.1f}%)\n")
            f.write(f"Low Error (≤10% CER): {error_dist['low_error_count']:,} ({error_dist['low_error_percentage']:.1f}%)\n")
            f.write(f"Medium Error (10-30% CER): {error_dist['medium_error_count']:,}\n")
            f.write(f"High Error (>30% CER): {error_dist['high_error_count']:,}\n\n")
            
            # Length Analysis
            f.write("PERFORMANCE BY TEXT LENGTH:\n")
            f.write("-" * 40 + "\n")
            length_analysis = self.results['combined_analysis']['performance_analysis']['length_analysis']
            for length_range, data in length_analysis.items():
                f.write(f"{length_range} chars: {data['count']:,} samples, "
                       f"Mean CER: {data['mean_cer']:.3f}, "
                       f"Seq Acc: {data['sequence_accuracy']:.3f}\n")
            
            # Dataset Comparison Summary
            if 'dataset_comparison' in self.results and self.results['dataset_comparison']:
                f.write("\nDATASET COMPARISON:\n")
                f.write("-" * 40 + "\n")
                comparison = self.results['dataset_comparison']
                if 'comparison_summary' in comparison and 'general_vs_short_text' in comparison['comparison_summary']:
                    gap_data = comparison['comparison_summary']['general_vs_short_text']
                    f.write(f"General Validation vs Short Text:\n")
                    f.write(f"  CER Difference: {gap_data['cer_difference']:.3f}\n")
                    f.write(f"  Performance Gap: {gap_data['performance_gap_percentage']:.1f}%\n")
                    f.write(f"  General CER: {gap_data['general_cer']:.3f}\n")
                    f.write(f"  Short Text CER: {gap_data['short_text_cer']:.3f}\n")
                    f.write(f"  General Sequence Accuracy: {gap_data['general_sequence_acc']:.3f}\n")
                    f.write(f"  Short Text Sequence Accuracy: {gap_data['short_text_sequence_acc']:.3f}\n")
                    f.write(f"  Improvement Needed: {gap_data['improvement_needed']['absolute_cer_reduction_needed']:.3f}\n")
                    f.write(f"  Relative Improvement Factor: {gap_data['improvement_needed']['relative_improvement_factor']:.2f}\n")
            
            # Individual Dataset Results
            if 'individual_datasets' in self.results and self.results['individual_datasets']:
                f.write("\nINDIVIDUAL DATASET RESULTS:\n")
                f.write("-" * 40 + "\n")
                for dataset_name, dataset_data in self.results['individual_datasets'].items():
                    f.write(f"\n{dataset_name.upper()}:\n")
                    dataset_stats = dataset_data['performance_analysis']['overall_stats']
                    f.write(f"  Samples: {len(dataset_data['sample_results']):,}\n")
                    f.write(f"  Mean CER: {dataset_stats['mean_cer']:.3f} ({dataset_stats['mean_cer']*100:.1f}%)\n")
                    f.write(f"  Sequence Accuracy: {dataset_stats['sequence_accuracy']:.3f}\n")
                    f.write(f"  Exact Match: {dataset_stats['exact_match_accuracy']:.3f}\n")
                    
                    # Length distribution for this dataset
                    if 'dataset_comparison' in self.results and dataset_name in self.results['dataset_comparison']:
                        length_dist = self.results['dataset_comparison'][dataset_name]['length_distribution']
                        f.write(f"  Length Distribution:\n")
                        for length_cat, data in length_dist.items():
                            if data['count'] > 0:
                                f.write(f"    {length_cat}: {data['count']} samples ({data['percentage']:.1f}%), CER: {data['mean_cer']:.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    def _save_error_samples(self):
        """Save samples with high errors for manual inspection."""
        df = pd.DataFrame(self.sample_results)
        
        # High error samples
        high_error = df[df['cer'] > 0.3].nlargest(20, 'cer')
        high_error_path = self.output_dir / 'high_error_samples.csv'
        high_error.to_csv(high_error_path, index=False, encoding='utf-8')
        
        # Perfect samples
        perfect = df[df['cer'] == 0.0].head(20)
        perfect_path = self.output_dir / 'perfect_samples.csv'
        perfect.to_csv(perfect_path, index=False, encoding='utf-8')


def main():
    """Main function for full validation analysis."""
    parser = argparse.ArgumentParser(description="Full Validation Analysis for Khmer OCR")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--validation-dirs",
        type=str,
        nargs="+",
        default=["data/validation_fixed", "data/validation_short_text"],
        help="Paths to validation dataset directories (supports multiple)"
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        help="Single validation directory (legacy support, will be added to validation-dirs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_analysis_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,  # Increased from 32 for better performance
        help="Batch size for validation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--single-dataset",
        action="store_true",
        help="Use only the first validation directory instead of combining datasets"
    )
    
    args = parser.parse_args()
    
    # Handle legacy validation-dir argument
    validation_dirs = args.validation_dirs.copy()
    if args.validation_dir and args.validation_dir not in validation_dirs:
        validation_dirs.insert(0, args.validation_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'validation_analysis.log'
    setup_logging(args.log_level, str(log_file))
    logger = logging.getLogger("FullValidationAnalysis")
    
    logger.info("="*80)
    logger.info("KHMER OCR FULL VALIDATION ANALYSIS")
    logger.info("="*80)
    
    # Setup device with better detection and warnings
    device = get_device()
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("⚠️  Running on CPU - this will be slower!")
        logger.warning("   To enable GPU: Install PyTorch with CUDA support:")
        logger.warning("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # Check if CUDA is available but PyTorch doesn't support it
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.warning("   ✓ NVIDIA GPU detected - PyTorch just needs CUDA support")
            else:
                logger.info("   No NVIDIA GPU detected")
        except:
            logger.info("   Could not detect GPU status")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = ConfigManager(args.config)
        
        # Load model from checkpoint
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        engine = KhmerOCREngine.from_checkpoint(
            args.checkpoint,
            config_manager=config,
            device=device
        )
        
        logger.info("Model loaded successfully")
        
        # Initialize analyzer
        analyzer = FullValidationAnalyzer(
            model=engine.model,
            config_manager=config,
            device=device,
            output_dir=args.output_dir
        )
        
        # Adjust batch size based on device
        batch_size = args.batch_size
        if device.type == "cpu" and batch_size > 32:
            batch_size = 16  # Smaller batch size for CPU
            logger.info(f"Reduced batch size to {batch_size} for CPU processing")
        elif device.type == "cuda" and batch_size < 64:
            batch_size = 64  # Larger batch size for GPU
            logger.info(f"Increased batch size to {batch_size} for GPU processing")

        # Load validation datasets
        if args.single_dataset:
            logger.info(f"Using single dataset mode with: {validation_dirs[0]}")
            dataloader = analyzer.load_validation_dataset(
                validation_dir=validation_dirs[0],
                batch_size=batch_size
            )
            results = analyzer.run_comprehensive_validation(dataloader)
        else:
            logger.info(f"Using combined dataset mode with: {validation_dirs}")
            individual_dataloaders, combined_dataloader = analyzer.load_combined_validation_datasets(
                validation_dirs=validation_dirs,
                batch_size=batch_size
            )
            
            # Run comprehensive analysis with both individual and combined datasets
            results = analyzer.run_comprehensive_validation(
                dataloader=combined_dataloader,
                individual_dataloaders=individual_dataloaders
            )
        
        # Generate visualizations
        analyzer.generate_visualizations()
        
        # Save results
        analyzer.save_results()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        stats = results['combined_analysis']['performance_analysis']['overall_stats']
        logger.info(f"Results Summary:")
        logger.info(f"   • Total Samples: {results['metadata']['total_samples']:,}")
        logger.info(f"   • Datasets Analyzed: {', '.join(results['metadata']['datasets_analyzed'])}")
        logger.info(f"   • Mean CER: {stats['mean_cer']:.3f} ({stats['mean_cer']*100:.1f}%)")
        logger.info(f"   • Sequence Accuracy: {stats['sequence_accuracy']:.3f} ({stats['sequence_accuracy']*100:.1f}%)")
        logger.info(f"   • Exact Match: {stats['exact_match_accuracy']:.3f} ({stats['exact_match_accuracy']*100:.1f}%)")
        logger.info(f"   • Mean Confidence: {stats['mean_confidence']:.3f}")
        
        speed = results['combined_analysis']['performance_analysis']['speed_analysis']
        logger.info(f"Performance: {speed['samples_per_second']:.1f} samples/second")
        
        error_dist = results['combined_analysis']['error_analysis']['error_distribution']
        logger.info(f"Perfect Predictions: {error_dist['zero_error_percentage']:.1f}%")
        logger.info(f"Low Error (<=10%): {error_dist['low_error_percentage']:.1f}%")
        
        # Dataset-specific summary
        if 'individual_datasets' in results and results['individual_datasets']:
            logger.info(f"\nIndividual Dataset Performance:")
            for dataset_name, dataset_data in results['individual_datasets'].items():
                dataset_stats = dataset_data['performance_analysis']['overall_stats']
                logger.info(f"   • {dataset_name}: {dataset_stats['mean_cer']:.3f} CER ({dataset_stats['mean_cer']*100:.1f}%)")
        
        # Performance gap analysis
        if 'dataset_comparison' in results and results['dataset_comparison']:
            comparison = results['dataset_comparison']
            if 'comparison_summary' in comparison and 'general_vs_short_text' in comparison['comparison_summary']:
                gap_data = comparison['comparison_summary']['general_vs_short_text']
                logger.info(f"\nPerformance Gap Analysis:")
                logger.info(f"   • General Validation CER: {gap_data['general_cer']:.3f} ({gap_data['general_cer']*100:.1f}%)")
                logger.info(f"   • Short Text Validation CER: {gap_data['short_text_cer']:.3f} ({gap_data['short_text_cer']*100:.1f}%)")
                logger.info(f"   • Performance Gap: {gap_data['performance_gap_percentage']:.1f} percentage points")
                logger.info(f"   • Improvement Factor Needed: {gap_data['improvement_needed']['relative_improvement_factor']:.2f}x")
                
                if gap_data['performance_gap_percentage'] > 5.0:
                    logger.warning(f"   ⚠️  Large performance gap detected! Short text training is needed.")
                elif gap_data['performance_gap_percentage'] > 2.0:
                    logger.info(f"   📈 Moderate performance gap - short text training recommended.")
                else:
                    logger.info(f"   ✅ Good performance balance across datasets.")
        
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"   • validation_results.json - Complete analysis")
        logger.info(f"   • detailed_sample_results.csv - Per-sample data")
        logger.info(f"   • validation_summary_report.txt - Human-readable summary")
        logger.info(f"   • validation_analysis_visualizations.png - Charts and graphs")
        
    except Exception as e:
        logger.error(f"Validation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 