"""
Integration tests for end-to-end pipeline.
Tests data → model → training → inference flow.
"""
import unittest
import torch
from torch.utils.data import DataLoader
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from models.seq2seq import KhmerOCRSeq2Seq
from data.dataset import KhmerDataset, collate_fn
from utils.config import ConfigManager
from training.trainer import Trainer
from training.validator import Validator
from training.checkpoint_manager import CheckpointManager
from inference.ocr_engine import KhmerOCREngine
from inference.metrics import evaluate_model_predictions


class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager()
        self.device = torch.device('cpu')  # Use CPU for tests
        
        # Create small datasets for testing
        self.train_dataset = KhmerDataset(
            vocab=self.config.vocab,
            dataset_size=20,
            min_text_length=1,
            max_text_length=3,
            use_augmentation=False,
            seed=42
        )
        
        self.val_dataset = KhmerDataset(
            vocab=self.config.vocab,
            dataset_size=8,
            min_text_length=1,
            max_text_length=3,
            use_augmentation=False,
            seed=123
        )
    
    def test_data_to_model_flow(self):
        """Test data pipeline to model forward pass."""
        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Create model
        model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
        
        # Forward pass
        result = model.forward(batch['images'], batch['targets'])
        
        # Check that forward pass works
        self.assertIn('loss', result)
        self.assertIn('logits', result)
        self.assertIsInstance(result['loss'], torch.Tensor)
        self.assertEqual(result['logits'].shape[0], 4)  # Batch size
    
    def test_training_pipeline(self):
        """Test complete training pipeline with mini training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            log_dir = os.path.join(temp_dir, "logs")
            
            # Create dataloaders
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=4,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            # Create model
            model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
            
            # Override training config for quick test
            self.config.training_config.epochs = 1
            self.config.training_config.batch_size = 4
            
            # Create trainer
            trainer = Trainer(
                model=model,
                config_manager=self.config,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                device=self.device,
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir
            )
            
            # Run training
            trainer.train()
            
            # Check that checkpoint was created
            checkpoint_files = os.listdir(checkpoint_dir)
            self.assertTrue(any(f.endswith('.pth') for f in checkpoint_files))
    
    def test_inference_pipeline(self):
        """Test complete inference pipeline."""
        # Create model
        model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
        
        # Create OCR engine
        engine = KhmerOCREngine(
            model=model,
            vocab=self.config.vocab,
            device=self.device
        )
        
        # Get test sample
        sample = self.val_dataset[0]
        
        # Run inference
        result = engine.recognize(
            sample['image'],
            method='greedy',
            preprocess=False  # Skip preprocessing for test
        )
        
        # Check result structure
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['text'], str)
    
    def test_checkpoint_save_load_cycle(self):
        """Test saving and loading model checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_model.pth")
            
            # Create and train model briefly
            model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
            original_params = {name: param.clone() for name, param in model.named_parameters()}
            
            # Save checkpoint
            model.save_checkpoint(checkpoint_path, epoch=0)
            
            # Modify model parameters
            for param in model.parameters():
                param.data.fill_(0.5)
            
            # Load checkpoint
            loaded_model, checkpoint_info = KhmerOCRSeq2Seq.load_checkpoint(
                checkpoint_path, device=self.device
            )
            
            # Check that parameters were restored
            for name, param in loaded_model.named_parameters():
                torch.testing.assert_close(param, original_params[name])
    
    def test_evaluation_metrics_pipeline(self):
        """Test evaluation metrics on predictions."""
        # Create model and engine
        model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
        engine = KhmerOCREngine(model=model, vocab=self.config.vocab, device=self.device)
        
        # Get predictions for small dataset
        predictions = []
        targets = []
        
        for i in range(3):
            sample = self.val_dataset[i]
            result = engine.recognize(sample['image'], preprocess=False)
            
            predictions.append(result['text'])
            targets.append(sample['text'])
        
        # Calculate metrics
        metrics = evaluate_model_predictions(predictions, targets)
        
        # Check metric structure
        self.assertIn('cer', metrics)
        self.assertIn('wer', metrics)
        self.assertIn('sequence_accuracy', metrics)
        self.assertIn('bleu', metrics)
        
        # All should be numbers
        for key in ['cer', 'wer', 'sequence_accuracy', 'bleu']:
            self.assertIsInstance(metrics[key], (int, float))
    
    def test_batch_processing_pipeline(self):
        """Test batch processing through inference pipeline."""
        model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
        engine = KhmerOCREngine(model=model, vocab=self.config.vocab, device=self.device)
        
        # Get multiple samples
        images = [self.val_dataset[i]['image'] for i in range(3)]
        
        # Batch recognition
        results = engine.recognize_batch(images, preprocess=False)
        
        # Check results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('text', result)
            self.assertIsInstance(result['text'], str)
    
    def test_validator_integration(self):
        """Test validator integration with model and data."""
        model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
        validator = Validator(model, self.config, self.device)
        
        # Create validation dataloader
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Run validation
        metrics = validator.validate(val_dataloader)
        
        # Check metrics
        self.assertIn('val_loss', metrics)
        self.assertIn('cer', metrics)
        self.assertIn('validation_time', metrics)
        self.assertIn('num_samples', metrics)
        
        # Check that CER is reasonable (should be very high for untrained model)
        self.assertGreaterEqual(metrics['cer'], 0.0)
        self.assertLessEqual(metrics['cer'], 2.0)  # Can be > 1.0 for very bad predictions
    
    def test_checkpoint_manager_integration(self):
        """Test checkpoint manager with training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            
            manager = CheckpointManager(checkpoint_dir)
            
            # Create dummy checkpoint data
            model = KhmerOCRSeq2Seq(vocab_size=len(self.config.vocab))
            optimizer = torch.optim.Adam(model.parameters())
            
            checkpoint_data = {
                'epoch': 5,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'cer': 0.15, 'val_loss': 2.5},
                'best_cer': 0.15
            }
            
            # Save checkpoint
            saved_path = manager.save_checkpoint(checkpoint_data, epoch=5, is_best=True)
            self.assertTrue(os.path.exists(saved_path))
            
            # List checkpoints
            checkpoints = manager.list_checkpoints()
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0]['epoch'], 5)
            
            # Load checkpoint
            loaded_data = manager.load_checkpoint(saved_path)
            self.assertEqual(loaded_data['epoch'], 5)
            self.assertEqual(loaded_data['metrics']['cer'], 0.15)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration."""
    
    def test_config_to_model_integration(self):
        """Test that configuration properly initializes model."""
        config = ConfigManager()
        
        # Create model with config-derived parameters
        model = KhmerOCRSeq2Seq(
            vocab_size=len(config.vocab),
            pad_token_id=config.vocab.PAD_IDX,
            sos_token_id=config.vocab.SOS_IDX,
            eos_token_id=config.vocab.EOS_IDX,
            unk_token_id=config.vocab.UNK_IDX
        )
        
        # Check that model uses correct token IDs
        self.assertEqual(model.pad_token_id, config.vocab.PAD_IDX)
        self.assertEqual(model.sos_token_id, config.vocab.SOS_IDX)
        self.assertEqual(model.eos_token_id, config.vocab.EOS_IDX)
        self.assertEqual(model.unk_token_id, config.vocab.UNK_IDX)
    
    def test_config_to_dataset_integration(self):
        """Test configuration to dataset integration."""
        config = ConfigManager()
        
        dataset = KhmerDataset(
            vocab=config.vocab,
            dataset_size=10,
            image_height=config.data_config.image_height,
            use_augmentation=False
        )
        
        # Check that dataset uses correct configurations
        sample = dataset[0]
        self.assertEqual(sample['image'].shape[1], config.data_config.image_height)
        
        # Check that vocabulary encoding works
        encoded = config.vocab.encode(sample['text'])
        decoded = config.vocab.decode(encoded)
        self.assertEqual(decoded, sample['text'])


class TestErrorHandling(unittest.TestCase):
    """Test error handling in pipeline."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        config = ConfigManager()
        model = KhmerOCRSeq2Seq(vocab_size=len(config.vocab))
        engine = KhmerOCREngine(model=model, vocab=config.vocab)
        
        # Test with invalid image dimensions
        with self.assertRaises((RuntimeError, ValueError, TypeError)):
            invalid_image = torch.randn(10, 10)  # Wrong dimensions
            engine.recognize(invalid_image, preprocess=False)
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        config = ConfigManager()
        
        # Create empty dataset
        empty_dataset = KhmerDataset(vocab=config.vocab, dataset_size=0)
        
        # Should handle empty dataset gracefully
        self.assertEqual(len(empty_dataset), 0)


if __name__ == '__main__':
    unittest.main() 