#!/usr/bin/env python3
"""
Training Sample Inspector for Khmer OCR Seq2Seq Model

This script generates sample images from the same pipeline used in train_onthefly.py
and saves them to files for manual inspection. This helps verify that the image
generation and data processing are working correctly.

Now includes inference results using the best trained model.
"""

import torch
import argparse
import os
from pathlib import Path
import logging
import sys
from PIL import Image
import numpy as np
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.onthefly_dataset import OnTheFlyDataset
from src.data.curriculum_dataset import CurriculumDataset
from src.utils.config import ConfigManager
from src.inference.ocr_engine import KhmerOCREngine


def setup_logging(log_level: str = "INFO") -> None:
    """Setup global logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert image tensor to PIL Image for saving.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) or (H, W)
        
    Returns:
        PIL Image
    """
    # Handle different tensor shapes
    if len(image_tensor.shape) == 3:
        # (C, H, W) -> (H, W, C) or (H, W) if single channel
        if image_tensor.shape[0] == 1:
            image_np = image_tensor.squeeze(0).cpu().numpy()
        else:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        # (H, W)
        image_np = image_tensor.cpu().numpy()
    
    # Denormalize if needed (from [-1, 1] to [0, 255])
    if image_np.min() < 0:
        image_np = (image_np + 1) / 2  # [-1, 1] -> [0, 1]
    
    # Convert to uint8
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(image_np.shape) == 2:
        return Image.fromarray(image_np, mode='L')
    else:
        return Image.fromarray(image_np)


def calculate_accuracy_metrics(predictions, ground_truths):
    """
    Calculate various accuracy metrics.
    
    Args:
        predictions: List of predicted text strings
        ground_truths: List of ground truth text strings
        
    Returns:
        Dict with accuracy metrics
    """
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return {
            'exact_match': 0.0,
            'character_accuracy': 0.0,
            'total_samples': 0,
            'exact_matches': 0
        }
    
    exact_matches = 0
    total_chars = 0
    correct_chars = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Exact match
        if pred == gt:
            exact_matches += 1
        
        # Character-level accuracy
        # Use simple character-by-character comparison
        max_len = max(len(pred), len(gt))
        total_chars += max_len
        
        for i in range(max_len):
            pred_char = pred[i] if i < len(pred) else ''
            gt_char = gt[i] if i < len(gt) else ''
            if pred_char == gt_char:
                correct_chars += 1
    
    return {
        'exact_match': exact_matches / len(predictions) * 100,
        'character_accuracy': correct_chars / total_chars * 100 if total_chars > 0 else 0,
        'total_samples': len(predictions),
        'exact_matches': exact_matches
    }


def generate_and_save_samples(
    output_dir: str,
    num_samples: int = 50,
    config_path: str = "configs/train_config.yaml",
    corpus_dir: str = "data/processed",
    samples_per_epoch: int = 1000,
    max_length: int = 150,
    split: str = "train",
    model_checkpoint: str = "models/checkpoints/full/best_model.pth"
):
    """
    Generate and save sample images from the training pipeline with inference results.
    
    Args:
        output_dir: Directory to save sample images
        num_samples: Number of samples to generate
        config_path: Path to configuration file
        corpus_dir: Directory with processed corpus data
        samples_per_epoch: Samples per epoch for the dataset
        max_length: Maximum text length for curriculum dataset
        split: Dataset split to use
        model_checkpoint: Path to model checkpoint for inference
    """
    logger = logging.getLogger("SampleInspector")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generating {num_samples} samples with inference...")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Model checkpoint: {model_checkpoint}")
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    
    # Load OCR engine for inference
    ocr_engine = None
    if os.path.exists(model_checkpoint):
        try:
            logger.info("Loading OCR engine...")
            ocr_engine = KhmerOCREngine.from_checkpoint(model_checkpoint)
            logger.info("✅ OCR engine loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load OCR engine: {e}")
            logger.warning("Continuing without inference results")
    else:
        logger.warning(f"Model checkpoint not found: {model_checkpoint}")
        logger.warning("Continuing without inference results")
    
    # Create base dataset (same as train_onthefly.py)
    logger.info("Creating OnTheFlyDataset...")
    base_dataset = OnTheFlyDataset(
        split=split,
        config_manager=config_manager,
        corpus_dir=corpus_dir,
        samples_per_epoch=samples_per_epoch,
        augment_prob=0.8,  # High augmentation probability to see variety
        shuffle_texts=True,
        random_seed=None  # No seed for variety
    )
    
    # Wrap with CurriculumDataset (same as train_onthefly.py)
    logger.info(f"Wrapping with CurriculumDataset (max_length={max_length})...")
    dataset = CurriculumDataset(base_dataset, max_length=max_length, config_manager=config_manager)
    
    logger.info(f"Dataset created successfully")
    logger.info(f"Base dataset size: {len(base_dataset)}")
    logger.info(f"Curriculum dataset size: {len(dataset)}")
    
    # Generate samples
    logger.info("Generating samples...")
    sample_info = []
    predictions = []
    ground_truths = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            # Get sample from dataset
            sample = dataset[i]
            
            # Extract data
            image_tensor = sample['image']
            targets = sample['targets']
            text = sample.get('text', '')
            
            # Convert image tensor to PIL
            pil_image = tensor_to_pil(image_tensor)
            
            # Run inference if OCR engine is available
            prediction = ""
            if ocr_engine is not None:
                try:
                    result = ocr_engine.recognize(pil_image, method='greedy', preprocess=True)
                    prediction = result['text']
                    predictions.append(prediction)
                    ground_truths.append(text)
                except Exception as e:
                    logger.warning(f"Inference failed for sample {i}: {e}")
                    prediction = f"[INFERENCE_ERROR: {str(e)[:50]}]"
            else:
                prediction = "[NO_MODEL_LOADED]"
            
            # Create filenames
            sample_id = f"sample_{i:04d}"
            image_filename = f"{sample_id}.png"
            label_filename = f"{sample_id}.txt"
            
            # Save image
            image_path = images_dir / image_filename
            pil_image.save(image_path)
            
            # Save label
            label_path = labels_dir / label_filename
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Record sample info
            info = {
                'id': sample_id,
                'text': text,
                'prediction': prediction,
                'text_length': len(text),
                'image_width': pil_image.width,
                'image_height': pil_image.height,
                'targets_length': len(targets),
                'image_file': image_filename,
                'label_file': label_filename,
                'is_correct': text == prediction if ocr_engine else None
            }
            sample_info.append(info)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples} samples")
                
        except Exception as e:
            logger.error(f"Error generating sample {i}: {e}")
            continue
    
    # Calculate accuracy metrics if we have predictions
    accuracy_metrics = None
    if predictions and ground_truths:
        accuracy_metrics = calculate_accuracy_metrics(predictions, ground_truths)
        logger.info("")
        logger.info("🎯 ACCURACY RESULTS:")
        logger.info(f"  Exact Match Accuracy: {accuracy_metrics['exact_match']:.2f}% ({accuracy_metrics['exact_matches']}/{accuracy_metrics['total_samples']})")
        logger.info(f"  Character-level Accuracy: {accuracy_metrics['character_accuracy']:.2f}%")
        logger.info("")
    
    # Save metadata
    metadata_path = output_path / "metadata.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Sample Inspection Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {config_path}\n")
        f.write(f"Corpus directory: {corpus_dir}\n")
        f.write(f"Model checkpoint: {model_checkpoint}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Max length: {max_length}\n")
        f.write(f"Samples generated: {len(sample_info)}\n")
        
        if accuracy_metrics:
            f.write(f"\nACCURACY METRICS:\n")
            f.write(f"Exact Match: {accuracy_metrics['exact_match']:.2f}% ({accuracy_metrics['exact_matches']}/{accuracy_metrics['total_samples']})\n")
            f.write(f"Character-level: {accuracy_metrics['character_accuracy']:.2f}%\n")
        
        f.write("="*60 + "\n\n")
        
        # Statistics
        if sample_info:
            text_lengths = [info['text_length'] for info in sample_info]
            image_widths = [info['image_width'] for info in sample_info]
            
            f.write("STATISTICS:\n")
            f.write(f"Text length - Min: {min(text_lengths)}, Max: {max(text_lengths)}, Avg: {sum(text_lengths)/len(text_lengths):.1f}\n")
            f.write(f"Image width - Min: {min(image_widths)}, Max: {max(image_widths)}, Avg: {sum(image_widths)/len(image_widths):.1f}\n")
            f.write(f"Image height: {sample_info[0]['image_height']}\n")
            f.write("\n")
        
        # Sample details
        f.write("SAMPLE DETAILS:\n")
        f.write("-" * 60 + "\n")
        for info in sample_info:
            f.write(f"ID: {info['id']}\n")
            f.write(f"Ground Truth: '{info['text']}'\n")
            f.write(f"Prediction: '{info['prediction']}'\n")
            f.write(f"Correct: {info['is_correct']}\n")
            f.write(f"Length: {info['text_length']} chars, {info['targets_length']} tokens\n")
            f.write(f"Image: {info['image_width']}x{info['image_height']} px\n")
            f.write(f"Files: {info['image_file']}, {info['label_file']}\n")
            f.write("-" * 60 + "\n")
    
    # Create HTML viewer for easy inspection
    html_path = output_path / "viewer.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Training Samples Viewer with Inference</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .sample { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }
        .sample img { max-width: 100%; border: 1px solid #ddd; }
        .text { font-family: 'Khmer OS', Arial, sans-serif; font-size: 18px; margin: 5px 0; }
        .ground-truth { background-color: #e8f5e8; padding: 5px; border-radius: 3px; }
        .prediction { background-color: #f0f8ff; padding: 5px; border-radius: 3px; margin-top: 5px; }
        .correct { border-left: 4px solid #4CAF50; }
        .incorrect { border-left: 4px solid #f44336; }
        .info { font-size: 12px; color: #666; }
        .header { background: #f0f0f0; padding: 10px; margin-bottom: 20px; }
        .accuracy { background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .metrics { display: flex; gap: 20px; }
        .metric { text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .metric-label { font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Khmer OCR Training Samples with Inference Results</h1>
        <p>Generated samples from the training pipeline with model predictions</p>
        <p><strong>Total samples:</strong> """ + str(len(sample_info)) + """</p>
        <p><strong>Model:</strong> """ + model_checkpoint + """</p>
    </div>
""")
        
        # Add accuracy metrics if available
        if accuracy_metrics:
            f.write(f"""
    <div class="accuracy">
        <h2>🎯 Accuracy Results</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{accuracy_metrics['exact_match']:.1f}%</div>
                <div class="metric-label">Exact Match<br>({accuracy_metrics['exact_matches']}/{accuracy_metrics['total_samples']})</div>
            </div>
            <div class="metric">
                <div class="metric-value">{accuracy_metrics['character_accuracy']:.1f}%</div>
                <div class="metric-label">Character-level<br>Accuracy</div>
            </div>
        </div>
    </div>
""")
        
        for info in sample_info:
            correct_class = "correct" if info['is_correct'] else "incorrect" if info['is_correct'] is not None else ""
            f.write(f"""
    <div class="sample {correct_class}">
        <h3>{info['id']}</h3>
        <img src="images/{info['image_file']}" alt="{info['id']}">
        <div class="text ground-truth"><strong>Ground Truth:</strong> {info['text']}</div>
        <div class="text prediction"><strong>Prediction:</strong> {info['prediction']}</div>
        <div class="info">
            Length: {info['text_length']} chars, {info['targets_length']} tokens | 
            Size: {info['image_width']}x{info['image_height']} px |
            Match: {'✅ Correct' if info['is_correct'] else '❌ Incorrect' if info['is_correct'] is not None else '⚠️ No inference'}
        </div>
    </div>
""")
        
        f.write("""
</body>
</html>""")
    
    logger.info(f"✅ Sample generation completed!")
    logger.info(f"Generated {len(sample_info)} samples")
    if accuracy_metrics:
        logger.info(f"🎯 Exact Match Accuracy: {accuracy_metrics['exact_match']:.2f}%")
        logger.info(f"🎯 Character Accuracy: {accuracy_metrics['character_accuracy']:.2f}%")
    logger.info(f"Files saved to: {output_path}")
    logger.info(f"📁 Images: {images_dir}")
    logger.info(f"📄 Labels: {labels_dir}")
    logger.info(f"📊 Metadata: {metadata_path}")
    logger.info(f"🌐 HTML Viewer: {html_path}")
    logger.info("")
    logger.info(f"To view samples:")
    logger.info(f"  1. Open {html_path} in a web browser")
    logger.info(f"  2. Or browse images manually in {images_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate and save training samples for inspection with inference results")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_samples_inspection",
        help="Directory to save sample images and labels"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/processed",
        help="Directory with processed corpus data"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=150,
        help="Maximum text length for curriculum dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="models/checkpoints/full/best_model.pth",
        help="Path to model checkpoint for inference"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("SampleInspector")
    
    logger.info("="*70)
    logger.info("Khmer OCR Training Sample Inspector with Inference")
    logger.info("="*70)
    
    try:
        generate_and_save_samples(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            config_path=args.config,
            corpus_dir=args.corpus_dir,
            max_length=args.max_length,
            split=args.split,
            model_checkpoint=args.model_checkpoint
        )
        
    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Sample generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 