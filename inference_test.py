#!/usr/bin/env python3
"""
Comprehensive Khmer OCR Inference Testing Script

This script provides multiple ways to test your trained Khmer OCR model:
1. Test on synthetic dataset
2. Test on custom images
3. Interactive mode for single image testing
4. Batch processing of image directories
5. Performance benchmarking

Usage:
    python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode synthetic
    python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode single --image path/to/image.png
    python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode batch --input-dir path/to/images/
    python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode interactive
    python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode benchmark
"""

import torch
import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
from PIL import Image
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.inference.ocr_engine import KhmerOCREngine
from src.inference.metrics import evaluate_model_predictions
from src.utils.config import ConfigManager
from src.data.synthetic_dataset import SyntheticImageDataset


class KhmerOCRTester:
    """
    Comprehensive testing interface for Khmer OCR model.
    """
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Initialize the OCR tester.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.checkpoint_path = checkpoint_path
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.config = ConfigManager()
        self.engine = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        print(f"Loading model from: {self.checkpoint_path}")
        
        try:
            if os.path.exists(self.checkpoint_path):
                try:
                    self.engine = KhmerOCREngine.from_checkpoint(
                        self.checkpoint_path,
                        config_manager=self.config,
                        device=self.device
                    )
                    print("✓ Model loaded successfully")
                except Exception as e:
                    print(f"⚠ Checkpoint loading failed: {e}")
                    print("Creating model with random weights for testing...")
                    
                    from src.models.seq2seq import KhmerOCRSeq2Seq
                    model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
                    self.engine = KhmerOCREngine(
                        model=model,
                        vocab=self.config.vocab,
                        device=self.device
                    )
                    print("⚠ Using untrained model - results will be poor")
            else:
                print(f"✗ Checkpoint not found: {self.checkpoint_path}")
                print("Creating model with random weights for testing...")
                
                from src.models.seq2seq import KhmerOCRSeq2Seq
                model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
                self.engine = KhmerOCREngine(
                    model=model,
                    vocab=self.config.vocab,
                    device=self.device
                )
                print("⚠ Using untrained model - results will be poor")
        
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            sys.exit(1)
    
    def test_synthetic_dataset(self, max_samples: int = 50) -> Dict:
        """
        Test on synthetic dataset.
        
        Args:
            max_samples: Maximum number of samples to test
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print("SYNTHETIC DATASET TESTING")
        print(f"{'='*60}")
        
        try:
            # Load synthetic dataset
            dataset = SyntheticImageDataset(
                split="test",
                synthetic_dir="data/synthetic",
                config_manager=self.config,
                max_samples=max_samples
            )
            
            print(f"Loaded {len(dataset)} test samples")
            
            # Get predictions
            predictions = []
            targets = []
            confidences = []
            processing_times = []
            
            print("Processing samples...")
            for i, (image, target_text) in enumerate(dataset):
                if i >= max_samples:
                    break
                
                # Convert tensor to PIL Image
                if isinstance(image, torch.Tensor):
                    # Denormalize and convert to PIL
                    image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image_np)
                else:
                    image_pil = image
                
                # Time the inference
                start_time = time.time()
                
                # Get prediction
                result = self.engine.recognize(
                    image_pil,
                    method='greedy',
                    return_confidence=True,
                    preprocess=True
                )
                
                processing_time = time.time() - start_time
                
                predictions.append(result['text'])
                targets.append(target_text)
                confidences.append(result['confidence'] or 0.5)
                processing_times.append(processing_time)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{min(max_samples, len(dataset))} samples")
                    
                    # Show a sample result
                    exact_match = result['text'].strip() == target_text.strip()
                    print(f"    Sample {i+1}: '{result['text']}' | GT: '{target_text}' | Match: {exact_match}")
            
            # Calculate metrics
            print("\nCalculating metrics...")
            metrics = evaluate_model_predictions(
                predictions=predictions,
                targets=targets,
                confidences=confidences,
                detailed=True
            )
            
            # Display results
            print(f"\n{'='*60}")
            print("SYNTHETIC DATASET RESULTS")
            print(f"{'='*60}")
            
            print(f"Test samples:           {len(predictions)}")
            print(f"Character Error Rate:   {metrics['cer']:.2%}")
            print(f"Word Error Rate:        {metrics['wer']:.2%}")
            print(f"Sequence Accuracy:      {metrics['sequence_accuracy']:.2%}")
            print(f"BLEU Score:             {metrics['bleu']:.3f}")
            print(f"Avg processing time:    {np.mean(processing_times):.3f}s")
            print(f"Total processing time:  {sum(processing_times):.1f}s")
            
            # Character-level metrics
            print(f"\nCharacter-level metrics:")
            print(f"  Precision: {metrics['char_precision']:.3f}")
            print(f"  Recall:    {metrics['char_recall']:.3f}")
            print(f"  F1 Score:  {metrics['char_f1']:.3f}")
            
            # Show some examples
            print(f"\nSample predictions:")
            for i in range(min(5, len(predictions))):
                exact_match = predictions[i].strip() == targets[i].strip()
                print(f"  {i+1}. Pred: '{predictions[i]}' | GT: '{targets[i]}' | Match: {exact_match}")
            
            return metrics
            
        except Exception as e:
            print(f"✗ Synthetic dataset testing failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def test_single_image(self, image_path: str, ground_truth: str = None) -> Dict:
        """
        Test on a single image.
        
        Args:
            image_path: Path to image file
            ground_truth: Optional ground truth text
            
        Returns:
            Dictionary with prediction results
        """
        print(f"\n{'='*60}")
        print("SINGLE IMAGE TESTING")
        print(f"{'='*60}")
        
        try:
            # Load image
            image = Image.open(image_path)
            print(f"Image: {image_path}")
            print(f"Size: {image.size}")
            
            if ground_truth:
                print(f"Ground truth: '{ground_truth}'")
            
            # Test different methods
            methods = ['greedy', 'beam_search']
            results = {}
            
            for method in methods:
                print(f"\nTesting with {method}...")
                
                start_time = time.time()
                result = self.engine.recognize(
                    image,
                    method=method,
                    beam_size=5 if method == 'beam_search' else None,
                    return_confidence=True,
                    preprocess=True
                )
                processing_time = time.time() - start_time
                
                results[method] = result
                results[method]['processing_time'] = processing_time
                
                print(f"  Prediction: '{result['text']}'")
                print(f"  Confidence: {result['confidence']}")
                print(f"  Time: {processing_time:.3f}s")
                
                if ground_truth:
                    exact_match = result['text'].strip() == ground_truth.strip()
                    print(f"  Exact match: {exact_match}")
            
            return results
            
        except Exception as e:
            print(f"✗ Single image testing failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def test_batch_directory(self, input_dir: str, output_file: str = None) -> Dict:
        """
        Test on all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_file: Optional file to save results
            
        Returns:
            Dictionary with batch results
        """
        print(f"\n{'='*60}")
        print("BATCH DIRECTORY TESTING")
        print(f"{'='*60}")
        
        try:
            # Find all images
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(input_dir).glob(f'*{ext}'))
                image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
            
            print(f"Found {len(image_files)} images in {input_dir}")
            
            if not image_files:
                print("No images found!")
                return {}
            
            # Process images
            results = []
            processing_times = []
            
            for i, image_path in enumerate(image_files):
                print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
                
                try:
                    start_time = time.time()
                    result = self.engine.recognize(
                        str(image_path),
                        method='greedy',
                        return_confidence=True,
                        preprocess=True
                    )
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    result_data = {
                        'filename': image_path.name,
                        'prediction': result['text'],
                        'confidence': result['confidence'],
                        'processing_time': processing_time
                    }
                    
                    results.append(result_data)
                    print(f"  Result: '{result['text']}' (confidence: {result['confidence']:.3f})")
                
                except Exception as e:
                    print(f"  ✗ Failed to process {image_path.name}: {e}")
                    results.append({
                        'filename': image_path.name,
                        'prediction': '',
                        'confidence': 0.0,
                        'processing_time': 0.0,
                        'error': str(e)
                    })
            
            # Summary
            successful_results = [r for r in results if 'error' not in r]
            print(f"\n{'='*60}")
            print("BATCH PROCESSING RESULTS")
            print(f"{'='*60}")
            print(f"Total images:           {len(image_files)}")
            print(f"Successfully processed: {len(successful_results)}")
            print(f"Failed:                 {len(results) - len(successful_results)}")
            print(f"Average processing time: {np.mean(processing_times):.3f}s")
            print(f"Total processing time:   {sum(processing_times):.1f}s")
            
            # Save results if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {output_file}")
            
            return {
                'results': results,
                'summary': {
                    'total_images': len(image_files),
                    'successful': len(successful_results),
                    'failed': len(results) - len(successful_results),
                    'avg_processing_time': np.mean(processing_times),
                    'total_processing_time': sum(processing_times)
                }
            }
            
        except Exception as e:
            print(f"✗ Batch directory testing failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def interactive_mode(self):
        """
        Interactive mode for testing single images.
        """
        print(f"\n{'='*60}")
        print("INTERACTIVE MODE")
        print(f"{'='*60}")
        print("Enter image paths to test (type 'quit' to exit)")
        
        while True:
            try:
                image_path = input("\nImage path: ").strip()
                
                if image_path.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not image_path:
                    continue
                
                if not os.path.exists(image_path):
                    print(f"✗ File not found: {image_path}")
                    continue
                
                # Optional ground truth
                ground_truth = input("Ground truth (optional): ").strip()
                ground_truth = ground_truth if ground_truth else None
                
                # Test the image
                results = self.test_single_image(image_path, ground_truth)
                
                print(f"\nResults:")
                for method, result in results.items():
                    if isinstance(result, dict) and 'text' in result:
                        print(f"  {method}: '{result['text']}' (confidence: {result.get('confidence', 0):.3f})")
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
    
    def benchmark_mode(self, num_samples: int = 100):
        """
        Benchmark performance on synthetic data.
        
        Args:
            num_samples: Number of samples to benchmark
        """
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        
        try:
            # Load synthetic dataset
            dataset = SyntheticImageDataset(
                split="test",
                synthetic_dir="data/synthetic",
                config_manager=self.config,
                max_samples=num_samples
            )
            
            print(f"Benchmarking on {len(dataset)} samples")
            
            # Test different methods
            methods = ['greedy', 'beam_search']
            benchmark_results = {}
            
            for method in methods:
                print(f"\nBenchmarking {method}...")
                
                processing_times = []
                predictions = []
                
                for i, (image, target_text) in enumerate(dataset):
                    if i >= num_samples:
                        break
                    
                    # Convert tensor to PIL Image
                    if isinstance(image, torch.Tensor):
                        image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                        image_pil = Image.fromarray(image_np)
                    else:
                        image_pil = image
                    
                    # Time the inference
                    start_time = time.time()
                    
                    result = self.engine.recognize(
                        image_pil,
                        method=method,
                        beam_size=5 if method == 'beam_search' else None,
                        return_confidence=False,  # Skip confidence for speed
                        preprocess=True
                    )
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    predictions.append(result['text'])
                    
                    if (i + 1) % 25 == 0:
                        print(f"  Processed {i + 1}/{num_samples} samples")
                
                # Calculate statistics
                benchmark_results[method] = {
                    'avg_time': np.mean(processing_times),
                    'std_time': np.std(processing_times),
                    'min_time': np.min(processing_times),
                    'max_time': np.max(processing_times),
                    'total_time': sum(processing_times),
                    'samples_per_second': len(processing_times) / sum(processing_times)
                }
                
                print(f"  Results for {method}:")
                print(f"    Average time: {benchmark_results[method]['avg_time']:.3f}s")
                print(f"    Std deviation: {benchmark_results[method]['std_time']:.3f}s")
                print(f"    Min time: {benchmark_results[method]['min_time']:.3f}s")
                print(f"    Max time: {benchmark_results[method]['max_time']:.3f}s")
                print(f"    Samples/second: {benchmark_results[method]['samples_per_second']:.1f}")
            
            # Summary
            print(f"\n{'='*60}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*60}")
            
            for method, stats in benchmark_results.items():
                print(f"{method}:")
                print(f"  {stats['samples_per_second']:.1f} samples/second")
                print(f"  {stats['avg_time']:.3f}s average per sample")
            
            return benchmark_results
            
        except Exception as e:
            print(f"✗ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Khmer OCR Inference Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on synthetic dataset
  python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode synthetic

  # Test single image
  python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode single --image test_image.png

  # Batch process directory
  python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode batch --input-dir test_images/

  # Interactive mode
  python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode interactive

  # Performance benchmark
  python inference_test.py --checkpoint models/checkpoints/best_model.pth --mode benchmark
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pth",
        help="Path to model checkpoint (default: models/checkpoints/best_model.pth)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "single", "batch", "interactive", "benchmark"],
        default="synthetic",
        help="Testing mode (default: synthetic)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (default: auto)"
    )
    
    # Mode-specific arguments
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image (for single mode)"
    )
    
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Ground truth text for single image (optional)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing images (for batch mode)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for batch results (optional)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to test (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Validate mode-specific arguments
    if args.mode == "single" and not args.image:
        parser.error("--image is required for single mode")
    
    if args.mode == "batch" and not args.input_dir:
        parser.error("--input-dir is required for batch mode")
    
    print("Khmer OCR Inference Testing")
    print("="*60)
    
    try:
        # Initialize tester
        tester = KhmerOCRTester(args.checkpoint, args.device)
        
        # Run appropriate test mode
        if args.mode == "synthetic":
            tester.test_synthetic_dataset(args.max_samples)
        
        elif args.mode == "single":
            tester.test_single_image(args.image, args.ground_truth)
        
        elif args.mode == "batch":
            tester.test_batch_directory(args.input_dir, args.output_file)
        
        elif args.mode == "interactive":
            tester.interactive_mode()
        
        elif args.mode == "benchmark":
            tester.benchmark_mode(args.max_samples)
        
        print(f"\n{'='*60}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 