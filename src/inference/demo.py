"""
Demo script for Khmer OCR inference.
Showcases the OCR engine capabilities with examples.
"""
import torch
import argparse
from pathlib import Path
import sys
import json
from PIL import Image
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.inference.ocr_engine import KhmerOCREngine
from src.inference.metrics import evaluate_model_predictions
from src.utils.config import ConfigManager
from src.data.text_renderer import TextRenderer


def create_sample_images(vocab, num_samples: int = 5) -> list:
    """Create sample images for demo."""
    print("Creating sample images for demo...")
    
    # Sample Khmer texts
    sample_texts = [
        "កម្ពុជា",           # Cambodia
        "សួស្តី",            # Hello  
        "អរគុណ",             # Thank you
        "ភាសាខ្មែរ",          # Khmer language
        "រាជធានីភ្នំពេញ"      # Capital Phnom Penh
    ]
    
    # Create text renderer
    renderer = TextRenderer(image_height=32)
    
    samples = []
    for i, text in enumerate(sample_texts[:num_samples]):
        try:
            # Render text to image
            image = renderer.render_text(text)
            
            samples.append({
                'image': image,
                'text': text,
                'id': f"sample_{i+1}"
            })
            print(f"  Created sample {i+1}: '{text}'")
        except Exception as e:
            print(f"  Failed to create sample {i+1}: {e}")
    
    return samples


def demo_basic_inference(engine: KhmerOCREngine, samples: list):
    """Demo basic OCR inference."""
    print("\n" + "="*60)
    print("BASIC INFERENCE DEMO")
    print("="*60)
    
    for sample in samples:
        print(f"\nProcessing {sample['id']}...")
        print(f"Ground truth: '{sample['text']}'")
        
        # Time the inference
        start_time = time.time()
        
        # Recognize with greedy decoding
        result = engine.recognize(
            sample['image'],
            method='greedy',
            return_confidence=True,
            preprocess=True
        )
        
        inference_time = time.time() - start_time
        
        print(f"Prediction:   '{result['text']}'")
        print(f"Confidence:   {result['confidence']}")
        print(f"Time:         {inference_time:.3f}s")
        
        # Check accuracy
        exact_match = result['text'].strip() == sample['text'].strip()
        print(f"Exact match:  {exact_match}")


def demo_beam_search(engine: KhmerOCREngine, samples: list):
    """Demo beam search inference."""
    print("\n" + "="*60)
    print("BEAM SEARCH DEMO")
    print("="*60)
    
    sample = samples[0]  # Use first sample
    print(f"\nProcessing {sample['id']} with beam search...")
    print(f"Ground truth: '{sample['text']}'")
    
    # Time the inference
    start_time = time.time()
    
    # Recognize with beam search
    result = engine.recognize(
        sample['image'],
        method='beam_search',
        beam_size=5,
        length_penalty=0.6,
        return_confidence=True,
        preprocess=True
    )
    
    inference_time = time.time() - start_time
    
    print(f"Prediction:   '{result['text']}'")
    print(f"Confidence:   {result['confidence']}")
    print(f"Time:         {inference_time:.3f}s")
    
    # Show raw beam search results if available
    if 'raw_output' in result and 'sequences' in result['raw_output']:
        sequences = result['raw_output']['sequences']
        scores = result['raw_output'].get('scores')
        
        print(f"\nTop {min(3, len(sequences))} beam search results:")
        for i, seq in enumerate(sequences[:3]):
            decoded = engine._decode_sequence(seq.cpu().tolist())
            score = scores[i].item() if scores is not None else "N/A"
            print(f"  Beam {i+1}: '{decoded}' (score: {score})")


def demo_batch_processing(engine: KhmerOCREngine, samples: list):
    """Demo batch processing."""
    print("\n" + "="*60)
    print("BATCH PROCESSING DEMO")
    print("="*60)
    
    images = [sample['image'] for sample in samples]
    ground_truths = [sample['text'] for sample in samples]
    
    print(f"Processing batch of {len(images)} images...")
    
    # Time the batch inference
    start_time = time.time()
    
    # Batch recognition
    results = engine.recognize_batch(
        images,
        method='greedy',
        preprocess=True
    )
    
    batch_time = time.time() - start_time
    
    print(f"Batch processing time: {batch_time:.3f}s")
    print(f"Average per image: {batch_time/len(images):.3f}s")
    
    # Show results
    print("\nBatch results:")
    for i, (result, gt) in enumerate(zip(results, ground_truths)):
        exact_match = result['text'].strip() == gt.strip()
        print(f"  Sample {i+1}: '{result['text']}' | GT: '{gt}' | Match: {exact_match}")


def demo_evaluation_metrics(engine: KhmerOCREngine, samples: list):
    """Demo evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION METRICS DEMO")
    print("="*60)
    
    # Get predictions for all samples
    predictions = []
    ground_truths = []
    confidences = []
    
    print("Getting predictions for evaluation...")
    for sample in samples:
        result = engine.recognize(
            sample['image'],
            method='greedy',
            return_confidence=True
        )
        
        predictions.append(result['text'])
        ground_truths.append(sample['text'])
        confidences.append(result['confidence'] or 0.5)
    
    # Calculate comprehensive metrics
    metrics = evaluate_model_predictions(
        predictions=predictions,
        targets=ground_truths,
        confidences=confidences,
        detailed=True
    )
    
    # Display results
    print("\nOverall Metrics:")
    print(f"  Character Error Rate (CER): {metrics['cer']:.2%}")
    print(f"  Word Error Rate (WER):       {metrics['wer']:.2%}")
    print(f"  Sequence Accuracy:           {metrics['sequence_accuracy']:.2%}")
    print(f"  BLEU Score:                  {metrics['bleu']:.3f}")
    
    print(f"\nCharacter-level Metrics:")
    print(f"  Precision: {metrics['char_precision']:.3f}")
    print(f"  Recall:    {metrics['char_recall']:.3f}")
    print(f"  F1 Score:  {metrics['char_f1']:.3f}")
    
    print(f"\nStatistics:")
    print(f"  Total samples:      {metrics['total_samples']}")
    print(f"  Avg pred length:    {metrics['avg_pred_length']:.1f}")
    print(f"  Avg target length:  {metrics['avg_target_length']:.1f}")
    
    if 'confidence_accuracy_correlation' in metrics:
        print(f"  Confidence correlation: {metrics['confidence_accuracy_correlation']:.3f}")
    
    # Show per-sample details
    if 'samples' in metrics:
        print(f"\nPer-sample Results:")
        for sample in metrics['samples']:
            print(f"  Sample {sample['index']+1}: CER={sample['cer']:.2%}, "
                  f"Acc={sample['accuracy']:.0%}, Pred='{sample['prediction'][:20]}...'")


def demo_preprocessing_effects(engine: KhmerOCREngine, samples: list):
    """Demo different preprocessing options."""
    print("\n" + "="*60)
    print("PREPROCESSING EFFECTS DEMO")
    print("="*60)
    
    sample = samples[0]  # Use first sample
    print(f"\nTesting preprocessing on {sample['id']}")
    print(f"Ground truth: '{sample['text']}'")
    
    # Test different preprocessing options
    preprocess_options = [
        {"preprocess": False, "name": "No preprocessing"},
        {"preprocess": True, "name": "Full preprocessing"},
    ]
    
    for options in preprocess_options:
        result = engine.recognize(
            sample['image'],
            method='greedy',
            **{k: v for k, v in options.items() if k != 'name'}
        )
        
        exact_match = result['text'].strip() == sample['text'].strip()
        print(f"  {options['name']}: '{result['text']}' | Match: {exact_match}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Khmer OCR Inference Demo")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to model checkpoint (optional, uses default model if not provided)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=5,
        help="Number of sample images to create"
    )
    
    args = parser.parse_args()
    
    print("Khmer OCR Inference Demo")
    print("="*60)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # Initialize components
        config = ConfigManager()
        
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            engine = KhmerOCREngine.from_checkpoint(
                args.checkpoint,
                config_manager=config,
                device=device
            )
        else:
            print("Creating model with random weights (for demo purposes)")
            from src.models.seq2seq import KhmerOCRSeq2Seq
            model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
            engine = KhmerOCREngine(
                model=model,
                vocab=config.vocab,
                device=device
            )
        
        # Create sample images
        samples = create_sample_images(config.vocab, args.samples)
        
        if not samples:
            print("Failed to create sample images. Exiting.")
            return
        
        # Run demos
        demo_basic_inference(engine, samples)
        demo_beam_search(engine, samples)
        demo_batch_processing(engine, samples)
        demo_evaluation_metrics(engine, samples)
        demo_preprocessing_effects(engine, samples)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe OCR engine supports:")
        print("  ✓ Greedy decoding for fast inference")
        print("  ✓ Beam search for improved accuracy")
        print("  ✓ Batch processing for efficiency")
        print("  ✓ Comprehensive evaluation metrics")
        print("  ✓ Flexible preprocessing options")
        print("  ✓ Confidence scoring and calibration")
    
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 