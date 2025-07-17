#!/usr/bin/env python3
"""
Test script to inspect curriculum dataset images and verify fixes.
This script generates sample images from all curriculum phases to visually
inspect the quality and alignment between images and text labels.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import ConfigManager
from src.khtext.subword_cluster import split_syllables_advanced

# Import CurriculumDataset from the training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_curriculum", "train_curriculum_eos_v1.py")
train_curriculum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_curriculum)
CurriculumDataset = train_curriculum.CurriculumDataset

def create_curriculum_dataset(config_manager, phase_info):
    """Create curriculum dataset for a specific phase."""
    # Import OnTheFlyDataset
    from src.data.onthefly_dataset import OnTheFlyDataset
    
    # Create base dataset
    base_dataset = OnTheFlyDataset(config_manager=config_manager)
    
    # Create curriculum dataset
    return CurriculumDataset(
        base_dataset=base_dataset,
        max_length=phase_info['max_length'],
        config_manager=config_manager
    )

def inspect_curriculum_phases():
    """Generate and display sample images from all curriculum phases."""
    print("🔍 Inspecting Curriculum Dataset Images...")
    
    # Initialize config
    config_manager = ConfigManager()
    
    # Define curriculum phases
    curriculum_phases = [
        {"phase": 1, "max_length": 10, "description": "Phase 1: Very Short (≤10 chars)"},
        {"phase": 2, "max_length": 20, "description": "Phase 2: Short (≤20 chars)"},
        {"phase": 3, "max_length": 30, "description": "Phase 3: Medium (≤30 chars)"},
        {"phase": 4, "max_length": 40, "description": "Phase 4: Long (≤40 chars)"},
        {"phase": 5, "max_length": 50, "description": "Phase 5: Very Long (≤50 chars)"}
    ]
    
    # Create output directory
    output_dir = Path("curriculum_inspection_samples")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(len(curriculum_phases), 3, figsize=(15, 20))
    fig.suptitle("Curriculum Dataset Image Inspection", fontsize=16, fontweight='bold')
    
    for phase_idx, phase_info in enumerate(curriculum_phases):
        print(f"\n📋 {phase_info['description']}")
        print(f"   Max Length: {phase_info['max_length']} characters")
        
        try:
            # Create dataset for this phase
            dataset = create_curriculum_dataset(config_manager, phase_info)
            print(f"   Dataset size: {len(dataset)} samples")
            
            # Sample 3 items from this phase
            sample_indices = [0, len(dataset)//2, len(dataset)-1] if len(dataset) >= 3 else [0]
            
            for sample_idx, data_idx in enumerate(sample_indices):
                if sample_idx >= 3:
                    break
                    
                try:
                    # Get sample
                    sample = dataset[data_idx]
                    image = sample['image']
                    target = sample['targets']
                    
                    # Convert tensor to numpy for display
                    if isinstance(image, torch.Tensor):
                        if image.dim() == 3 and image.size(0) == 1:
                            img_np = image.squeeze(0).cpu().numpy()
                        else:
                            img_np = image.cpu().numpy()
                    else:
                        img_np = image
                    
                    # Decode target to text
                    target_text = dataset.vocab.decode(target.tolist())
                    
                    # Get original text for comparison (from base dataset)
                    original_line = dataset.base_dataset.texts[data_idx] if hasattr(dataset.base_dataset, 'texts') else "N/A"
                    
                    # Calculate syllable truncation
                    syllables = split_syllables_advanced(original_line)
                    truncated_syllables = []
                    char_count = 0
                    for syllable in syllables:
                        if char_count + len(syllable) <= phase_info['max_length']:
                            truncated_syllables.append(syllable)
                            char_count += len(syllable)
                        else:
                            break
                    expected_text = ''.join(truncated_syllables)
                    
                    # Plot
                    ax = axes[phase_idx, sample_idx]
                    ax.imshow(img_np, cmap='gray')
                    ax.set_title(f"Phase {phase_info['phase']} - Sample {sample_idx + 1}\n"
                               f"Target: {target_text[:30]}{'...' if len(target_text) > 30 else ''}\n"
                               f"Length: {len(target_text)} chars", 
                               fontsize=8)
                    ax.axis('off')
                    
                    # Add colored border based on length compliance
                    is_compliant = len(target_text) <= phase_info['max_length']
                    border_color = 'green' if is_compliant else 'red'
                    border_width = 3
                    
                    rect = patches.Rectangle((0, 0), img_np.shape[1]-1, img_np.shape[0]-1,
                                           linewidth=border_width, edgecolor=border_color, 
                                           facecolor='none')
                    ax.add_patch(rect)
                    
                    print(f"     Sample {sample_idx + 1}:")
                    print(f"       Original: {original_line[:50]}{'...' if len(original_line) > 50 else ''}")
                    print(f"       Expected: {expected_text}")
                    print(f"       Target:   {target_text}")
                    print(f"       Lengths:  Orig={len(original_line)}, Expected={len(expected_text)}, Target={len(target_text)}")
                    print(f"       Compliant: {'✅' if is_compliant else '❌'}")
                    
                except Exception as e:
                    print(f"     ❌ Error processing sample {sample_idx}: {e}")
                    if sample_idx < 3:
                        ax = axes[phase_idx, sample_idx]
                        ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
            
            # Fill empty subplots if less than 3 samples
            for empty_idx in range(len(sample_indices), 3):
                ax = axes[phase_idx, empty_idx]
                ax.text(0.5, 0.5, 'No Sample', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                
        except Exception as e:
            print(f"   ❌ Error creating dataset for phase {phase_info['phase']}: {e}")
            for col_idx in range(3):
                ax = axes[phase_idx, col_idx]
                ax.text(0.5, 0.5, f"Phase {phase_info['phase']}\nError:\n{str(e)[:30]}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    plt.tight_layout()
    output_file = output_dir / "curriculum_phases_inspection.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved inspection plot to: {output_file}")
    plt.show()

def test_syllable_truncation():
    """Test syllable-based truncation functionality."""
    print("\n🔤 Testing Syllable-Based Truncation...")
    
    test_texts = [
        "កម្ពុជា",
        "ខ្ញុំសូមចូលរួម",
        "ការអប់រំនេះមានសារៈសំខាន់ណាស់",
        "ព្រះរាជាណាចក្រកម្ពុជាជាប្រទេសមួយក្នុងតំបន់អាស៊ីអាគ្នេយ៍",
        "វិទ្យាស្ថានបច្ចេកវិទ្យាកម្ពុជាប្រទេសយើងមានកម្មវិធីសិក្សាច្រើនជាង"
    ]
    
    max_lengths = [10, 20, 30, 40, 50]
    
    for text in test_texts:
        print(f"\nOriginal: {text} (length: {len(text)})")
        syllables = split_syllables_advanced(text)
        print(f"Syllables: {syllables}")
        
        for max_len in max_lengths:
            truncated_syllables = []
            char_count = 0
            for syllable in syllables:
                if char_count + len(syllable) <= max_len:
                    truncated_syllables.append(syllable)
                    char_count += len(syllable)
                else:
                    break
            
            truncated_text = ''.join(truncated_syllables)
            print(f"  Max {max_len:2d}: {truncated_text} (length: {len(truncated_text)})")

def verify_fixes():
    """Verify that all the fixes are working properly."""
    print("\n🔧 Verifying Curriculum Dataset Fixes...")
    
    try:
        config_manager = ConfigManager()
        print("✅ ConfigManager created successfully")
        
        # Test Phase 4 (where issues were most apparent)
        phase_info = {"phase": 4, "max_length": 30, "description": "Phase 4: Long (≤30 chars)"}
        print(f"✅ Phase info created: {phase_info}")
        
        dataset = create_curriculum_dataset(config_manager, phase_info)
        print(f"✅ Successfully created Phase 4 dataset with {len(dataset)} samples")
        
        # Test a few samples
        for i in range(min(3, len(dataset))):
            print(f"   Testing sample {i}...")
            sample = dataset[i]
            print(f"     Sample keys: {list(sample.keys())}")
            
            # Check what's in the sample
            if 'image' in sample:
                image = sample['image']
            elif 'images' in sample:
                image = sample['images']
            else:
                print(f"     ❌ No image found in sample")
                continue
            
            if 'target' in sample:
                target = sample['target']
            elif 'targets' in sample:
                target = sample['targets']
            else:
                print(f"     ❌ No target found in sample")
                continue
            
            # Verify image format
            print(f"     Image type: {type(image)}, shape: {getattr(image, 'shape', 'No shape attr')}")
            assert isinstance(image, torch.Tensor), f"Image should be tensor, got {type(image)}"
            assert image.dim() in [2, 3], f"Image should be 2D or 3D, got {image.dim()}D"
            
            # Verify target length
            target_text = dataset.vocab.decode(target.tolist())
            print(f"     Target text: '{target_text}', length: {len(target_text)}")
            assert len(target_text) <= phase_info['max_length'], \
                f"Target length {len(target_text)} exceeds max_length {phase_info['max_length']}"
            
            print(f"  ✅ Sample {i}: Image shape {tuple(image.shape)}, Target length {len(target_text)}")
        
        print("✅ All verification tests passed!")
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Verification failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Curriculum Dataset Inspection...")
    
    # Verify fixes first
    if verify_fixes():
        # Test syllable truncation
        test_syllable_truncation()
        
        # Inspect curriculum phases
        inspect_curriculum_phases()
        
        print("\n✅ Curriculum inspection completed successfully!")
    else:
        print("\n❌ Fix verification failed. Please check the implementation.")