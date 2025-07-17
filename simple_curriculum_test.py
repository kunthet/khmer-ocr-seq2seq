#!/usr/bin/env python3
"""
Simple test script to verify the fixed curriculum dataset implementation.
This directly uses the CurriculumDataset from train_curriculum_eos_v1.py
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append('src')

# Import from training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_curriculum", "train_curriculum_eos_v1.py")
train_curriculum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_curriculum)

from src.utils.config import ConfigManager
from src.data.onthefly_dataset import OnTheFlyDataset

def test_curriculum_fixes():
    """Test the fixed curriculum dataset implementation."""
    print("🧪 Testing Fixed Curriculum Dataset Implementation")
    print("=" * 60)
    
    # Initialize config
    config_manager = ConfigManager()
    print("✅ ConfigManager initialized")
    
    # Create base dataset with controlled parameters
    base_dataset = OnTheFlyDataset(
        split="train",
        config_manager=config_manager,
        corpus_dir="data/processed",
        samples_per_epoch=10000,
        augment_prob=0.3,  # Disable augmentation for testing
        shuffle_texts=False,  # Disable shuffling for consistency
        random_seed=42  # Fixed seed for reproducibility
    )
    print(f"✅ Base dataset created: {len(base_dataset)} samples")
    
    # Test different curriculum phases
    phases = [
        {"phase": 1, "max_length": 10, "description": "Phase 1: Very Short"},
        {"phase": 2, "max_length": 20, "description": "Phase 2: Short"},
        {"phase": 3, "max_length": 30, "description": "Phase 3: Medium"},
        {"phase": 4, "max_length": 40, "description": "Phase 4: Long"},
        {"phase": 5, "max_length": 50, "description": "Phase 5: Very Long"}
    ]
    
    # Create output directory for sample images
    output_dir = Path("curriculum_test_samples")
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for phase_info in phases:
        print(f"\n📋 {phase_info['description']} (max_length={phase_info['max_length']})")
        print("-" * 40)
        
        try:
            # Create curriculum dataset
            curriculum_dataset = train_curriculum.CurriculumDataset(
                base_dataset=base_dataset,
                max_length=phase_info['max_length'],
                config_manager=config_manager
            )
            print(f"✅ Created curriculum dataset: {len(curriculum_dataset)} samples")
            
            # Test 3 samples
            phase_results = []
            for i in range(3):
                sample = curriculum_dataset[i]
                image = sample['image']
                targets = sample['targets']
                
                # Decode targets
                target_text = config_manager.vocab.decode(targets.tolist())
                token_count = len(targets.tolist())  # Actual token count
                                
                # Clean up text (remove special tokens for length calculation)
                clean_text = target_text.replace('SOS', '').replace('EOS', '').replace('UNK', '')
                
                print(f"  Sample {i+1}:")
                print(f"    Target: '{target_text[:60]}{'...' if len(target_text) > 60 else ''}'")
                print(f"    Length: {token_count} tokens")  # Use actual token count
                print(f"    Clean length: {len(clean_text)} chars")
                print(f"    Image shape: {tuple(image.shape)}")
                print(f"    Compliant: {'✅' if token_count <= phase_info['max_length'] else '❌'}")  # Use token count for compliance
                
                # Save sample image
                if isinstance(image, torch.Tensor):
                    if image.dim() == 3 and image.size(0) == 1:
                        img_np = image.squeeze(0).cpu().numpy()
                    else:
                        img_np = image.cpu().numpy()
                else:
                    img_np = image
                
                # convert img_np to uint8
                img_np = (img_np * 255).astype(np.uint8)
                
                # Save image
                filename = f"phase{phase_info['phase']}_sample{i+1}_len{token_count}.png"  # Use token count
                text_filename = filename.replace(".png", ".txt")
                filepath = output_dir / filename
                
                # save image to file

                Image.fromarray(img_np).save(filepath)

                # save target text to file
                with open(output_dir / text_filename, "w", encoding="utf-8") as f:
                    f.write(target_text)
               
                print(f"    💾 Saved: {filename}")
                
                phase_results.append({
                    'target_text': target_text,
                    'target_length': token_count,  # Use token count
                    'clean_length': len(clean_text),
                    'compliant': token_count <= phase_info['max_length'],  # Use token count
                    'image_shape': tuple(image.shape)
                })
            
            # Summary for this phase
            compliant_count = sum(1 for r in phase_results if r['compliant'])
            print(f"\n  📊 Phase Summary:")
            print(f"    Compliant samples: {compliant_count}/3")
            print(f"    Average length: {np.mean([r['target_length'] for r in phase_results]):.1f}")
            print(f"    Max length found: {max(r['target_length'] for r in phase_results)}")
            
            results[phase_info['phase']] = {
                'phase_info': phase_info,
                'samples': phase_results,
                'compliant_rate': compliant_count / len(phase_results)
            }
            
        except Exception as e:
            print(f"❌ Error testing phase {phase_info['phase']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print(f"\n🎯 Overall Test Results")
    print("=" * 60)
    for phase, result in results.items():
        print(f"Phase {phase}: {result['compliant_rate']*100:.1f}% compliant "
              f"(max_length={result['phase_info']['max_length']})")
    
    print(f"\n💾 Sample images saved to: {output_dir}")
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_curriculum_fixes() 