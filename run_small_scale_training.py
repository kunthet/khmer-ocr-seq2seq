"""
Small-scale training validation script.
Tests the entire pipeline with 50-100 samples to validate end-to-end functionality.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import time
import tempfile
import shutil

# Add src to path
sys.path.append('src')

from src.models.seq2seq import KhmerOCRSeq2Seq
from src.data.dataset import KhmerDataset, collate_fn
from src.utils.config import ConfigManager
from src.training.trainer import Trainer
from src.inference.ocr_engine import KhmerOCREngine
from src.inference.metrics import evaluate_model_predictions


def run_small_scale_training(
    num_train_samples: int = 100,
    num_val_samples: int = 20,
    num_epochs: int = 3,
    batch_size: int = 8,
    max_text_length: int = 8
):
    """
    Run small-scale training to validate the entire pipeline.
    
    Args:
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        max_text_length: Maximum text length for generation
    """
    print("="*70)
    print("SMALL-SCALE TRAINING VALIDATION")
    print("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"
        checkpoint_dir.mkdir()
        log_dir.mkdir()
        
        try:
            print(f"\n1. Creating datasets...")
            print(f"   Training samples: {num_train_samples}")
            print(f"   Validation samples: {num_val_samples}")
            
            # Initialize configuration
            config = ConfigManager()
            
            # Override training settings for small-scale test
            config.training_config.epochs = num_epochs
            config.training_config.batch_size = batch_size
            config.training_config.learning_rate = 1e-4  # Slightly higher for faster learning
            
            # Create datasets
            train_dataset = KhmerDataset(
                vocab=config.vocab,
                dataset_size=num_train_samples,
                min_text_length=1,
                max_text_length=max_text_length,
                use_augmentation=True,
                seed=42
            )
            
            val_dataset = KhmerDataset(
                vocab=config.vocab,
                dataset_size=num_val_samples,
                min_text_length=1,
                max_text_length=max_text_length,
                use_augmentation=False,
                seed=123
            )
            
            print(f"   ✓ Datasets created successfully")
            
            print(f"\n2. Creating data loaders...")
            
            # Create data loaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues in test
                collate_fn=collate_fn
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            print(f"   ✓ Train batches: {len(train_dataloader)}")
            print(f"   ✓ Val batches: {len(val_dataloader)}")
            
            print(f"\n3. Creating model...")
            
            # Create model
            model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   ✓ Model created with {total_params:,} parameters")
            
            print(f"\n4. Testing data flow...")
            
            # Test a single batch to ensure everything works
            test_batch = next(iter(train_dataloader))
            test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in test_batch.items()}
            
            # Forward pass test
            with torch.no_grad():
                result = model(test_batch['images'], test_batch['targets'])
                print(f"   ✓ Forward pass successful")
                print(f"   ✓ Loss value: {result['loss'].item():.4f}")
            
            print(f"\n5. Starting training...")
            print(f"   Epochs: {num_epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Learning rate: {config.training_config.learning_rate}")
            
            # Create trainer
            trainer = Trainer(
                model=model,
                config_manager=config,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                device=device,
                log_dir=str(log_dir),
                checkpoint_dir=str(checkpoint_dir)
            )
            
            # Record initial metrics
            print(f"\n   Getting initial validation metrics...")
            initial_metrics = trainer.validator.validate(val_dataloader)
            print(f"   Initial CER: {initial_metrics['cer']:.2%}")
            print(f"   Initial Val Loss: {initial_metrics['val_loss']:.4f}")
            
            # Run training
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            print(f"   ✓ Training completed in {training_time:.2f} seconds")
            
            print(f"\n6. Evaluating trained model...")
            
            # Get final metrics
            final_metrics = trainer.validator.validate(val_dataloader)
            print(f"   Final CER: {final_metrics['cer']:.2%}")
            print(f"   Final Val Loss: {final_metrics['val_loss']:.4f}")
            
            # Check for improvement
            cer_improvement = initial_metrics['cer'] - final_metrics['cer']
            loss_improvement = initial_metrics['val_loss'] - final_metrics['val_loss']
            
            print(f"   CER improvement: {cer_improvement:.2%}")
            print(f"   Loss improvement: {loss_improvement:.4f}")
            
            print(f"\n7. Testing inference pipeline...")
            
            # Create OCR engine
            engine = KhmerOCREngine(
                model=model,
                vocab=config.vocab,
                device=device
            )
            
            # Test on a few validation samples
            sample_predictions = []
            sample_targets = []
            
            for i in range(min(5, num_val_samples)):
                sample = val_dataset[i]
                
                # Recognize text
                result = engine.recognize(
                    sample['image'],
                    method='greedy',
                    preprocess=True
                )
                
                sample_predictions.append(result['text'])
                sample_targets.append(sample['text'])
                
                print(f"   Sample {i+1}:")
                print(f"     Target:     '{sample['text']}'")
                print(f"     Prediction: '{result['text']}'")
                print(f"     Match:      {sample['text'] == result['text']}")
            
            print(f"\n8. Computing comprehensive metrics...")
            
            # Get predictions for all validation samples
            all_predictions = []
            all_targets = []
            
            for i in range(num_val_samples):
                sample = val_dataset[i]
                result = engine.recognize(sample['image'], method='greedy')
                
                all_predictions.append(result['text'])
                all_targets.append(sample['text'])
            
            # Calculate comprehensive metrics
            metrics = evaluate_model_predictions(all_predictions, all_targets)
            
            print(f"   Character Error Rate (CER): {metrics['cer']:.2%}")
            print(f"   Word Error Rate (WER):       {metrics['wer']:.2%}")
            print(f"   Sequence Accuracy:           {metrics['sequence_accuracy']:.2%}")
            print(f"   BLEU Score:                  {metrics['bleu']:.3f}")
            
            print(f"\n9. Testing checkpoint functionality...")
            
            # List created checkpoints
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            print(f"   Created {len(checkpoints)} checkpoint(s)")
            
            if checkpoints:
                # Test loading from checkpoint
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                print(f"   Testing checkpoint loading: {latest_checkpoint.name}")
                
                # Create new engine from checkpoint
                engine_from_checkpoint = KhmerOCREngine.from_checkpoint(
                    str(latest_checkpoint),
                    config_manager=config,
                    device=device
                )
                
                # Test inference
                test_sample = val_dataset[0]
                result1 = engine.recognize(test_sample['image'])
                result2 = engine_from_checkpoint.recognize(test_sample['image'])
                
                print(f"   ✓ Checkpoint loading successful")
                print(f"   Predictions match: {result1['text'] == result2['text']}")
            
            print(f"\n10. Memory and performance analysis...")
            
            if device.type == 'cuda':
                print(f"    GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
                print(f"    GPU Memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f} MB")
            
            # Calculate inference speed
            inference_times = []
            for i in range(10):
                sample = val_dataset[i % num_val_samples]
                
                start_time = time.time()
                result = engine.recognize(sample['image'], method='greedy')
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference_time = sum(inference_times) / len(inference_times)
            print(f"    Average inference time: {avg_inference_time:.3f} seconds")
            print(f"    Inference speed: {1/avg_inference_time:.1f} samples/second")
            
            print(f"\n" + "="*70)
            print("VALIDATION RESULTS")
            print("="*70)
            
            # Summary
            success_criteria = []
            
            # Check if training reduced loss
            if loss_improvement > 0:
                success_criteria.append("✓ Training reduced validation loss")
            else:
                success_criteria.append("✗ Training did not reduce validation loss")
            
            # Check if any samples are recognized correctly
            correct_samples = sum(1 for p, t in zip(sample_predictions, sample_targets) if p == t)
            if correct_samples > 0:
                success_criteria.append(f"✓ {correct_samples}/{len(sample_predictions)} samples recognized correctly")
            else:
                success_criteria.append("✗ No samples recognized correctly")
            
            # Check if inference works
            if all(isinstance(p, str) for p in all_predictions):
                success_criteria.append("✓ Inference pipeline functional")
            else:
                success_criteria.append("✗ Inference pipeline has issues")
            
            # Check if checkpoints were created
            if len(checkpoints) > 0:
                success_criteria.append("✓ Checkpoints created successfully")
            else:
                success_criteria.append("✗ No checkpoints created")
            
            for criterion in success_criteria:
                print(f"  {criterion}")
            
            # Overall assessment
            passed_criteria = sum(1 for c in success_criteria if c.startswith("✓"))
            total_criteria = len(success_criteria)
            
            print(f"\nPassed: {passed_criteria}/{total_criteria} criteria")
            
            if passed_criteria >= 3:
                print("🎉 SMALL-SCALE TRAINING VALIDATION PASSED!")
                print("   The pipeline is ready for full-scale training.")
            else:
                print("⚠️  VALIDATION NEEDS ATTENTION")
                print("   Some issues detected - review before full-scale training.")
            
            return {
                'training_successful': loss_improvement > 0,
                'inference_functional': len(all_predictions) == num_val_samples,
                'checkpoints_created': len(checkpoints) > 0,
                'final_cer': final_metrics['cer'],
                'final_loss': final_metrics['val_loss'],
                'avg_inference_time': avg_inference_time,
                'passed_criteria': passed_criteria,
                'total_criteria': total_criteria
            }
            
        except Exception as e:
            print(f"\n❌ VALIDATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Small-scale training validation")
    parser.add_argument("--train-samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-length", type=int, default=8, help="Maximum text length")
    
    args = parser.parse_args()
    
    results = run_small_scale_training(
        num_train_samples=args.train_samples,
        num_val_samples=args.val_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_text_length=args.max_length
    ) 