#!/usr/bin/env python3
"""
Production Training Launcher for Khmer OCR Seq2Seq Model

This script provides a comprehensive interface to start production training
with different strategies based on the completed EOS fixes and system testing.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import torch

def check_system_requirements():
    """Check if system is ready for training"""
    print("🔍 Checking System Requirements...")
    
    # Check PyTorch and CUDA
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.device_count()} GPU(s)")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
        device_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = device_props.total_memory / (1024**3)
        print(f"✅ GPU Memory: {gpu_memory_gb:.1f}GB")
        
        # Provide platform-specific guidance
        gpu_name_lower = gpu_name.lower()
        if 'a100' in gpu_name_lower:
            if gpu_memory_gb >= 40:
                print("🚀 Excellent! A100 GPU detected - optimal for large-scale training")
                print("   💡 Tip: A100 supports very large batch sizes (64-96) for faster training")
            else:
                print("🚀 A100 GPU detected - excellent for training")
        elif 't4' in gpu_name_lower:
            print("📱 Google Colab T4 detected - good for smaller experiments")
            print("   💡 Tip: Consider upgrading to Colab Pro for A100 access")
        elif 'v100' in gpu_name_lower:
            print("🔬 V100 GPU detected - solid for research training")
        
        if gpu_memory_gb < 8:
            print("⚠️  Warning: Less than 8GB GPU memory. Consider reducing batch size.")
        elif gpu_memory_gb >= 40:
            print("💪 High memory GPU - can handle very large batch sizes for fast training")
    else:
        print("❌ CUDA not available. Training will be very slow on CPU.")
        return False
    
    # Check validation set
    validation_dir = Path("data/validation_fixed")
    if validation_dir.exists() and (validation_dir / "metadata.json").exists():
        print("✅ Fixed validation set ready")
    else:
        print("❌ Fixed validation set not found")
        print("   Run: python generate_fixed_validation_set.py")
        return False
    
    # Check corpus data
    corpus_dir = Path("data/processed")
    if corpus_dir.exists() and (corpus_dir / "train.txt").exists():
        print("✅ Corpus data ready")
    else:
        print("❌ Corpus data not found")
        print("   Run: python process_corpus.py")
        return False
    
    # Check fonts
    fonts_dir = Path("fonts")
    font_files = list(fonts_dir.glob("*.ttf"))
    if len(font_files) >= 8:
        print(f"✅ Fonts ready: {len(font_files)} TTF files")
    else:
        print(f"❌ Insufficient fonts: {len(font_files)} found, need 8+")
        return False
    
    print("✅ System requirements check passed!")
    return True

def get_recommended_batch_size():
    """Get recommended batch size based on GPU memory"""
    if not torch.cuda.is_available():
        return 8
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    # Check for specific GPU models first
    if 'a100' in gpu_name:
        if gpu_memory_gb >= 80:  # A100 80GB (enterprise)
            return 96
        elif gpu_memory_gb >= 40:  # A100 40GB (Colab Pro/Pro+)
            return 64
        else:  # A100 variants
            return 48
    elif 'v100' in gpu_name:  # V100 (older Colab)
        return 20
    elif 't4' in gpu_name:  # T4 (free Colab)
        return 12
    
    # General memory-based recommendations
    if gpu_memory_gb >= 80:    # A100 80GB, H100
        return 96
    elif gpu_memory_gb >= 40:  # A100 40GB, RTX 6000 Ada
        return 64
    elif gpu_memory_gb >= 23:  # RTX 3090/4090 (23.99GB), RTX 6000
        return 32
    elif gpu_memory_gb >= 16:  # RTX 4070Ti/3080, V100 16GB
        return 24
    elif gpu_memory_gb >= 12:  # RTX 3060Ti/4060Ti, T4
        return 16
    elif gpu_memory_gb >= 8:   # RTX 3060/4060
        return 12
    else:
        return 8

def launch_curriculum_training(args):
    """Launch curriculum EOS training (recommended)"""
    print("🚀 Launching Curriculum EOS Training...")
    print("📚 Strategy: Progressive sequence length (10→50 chars)")
    print("🎯 Goal: Fix EOS generation through curriculum learning")
    
    cmd = [
        "python", "train_curriculum_eos_v1.py",
        "--num-epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--log-dir", "logs/curriculum",
        "--checkpoint-dir", "models/checkpoints/curriculum_eos_v1"
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"📝 Command: {' '.join(cmd)}")
    return subprocess.run(cmd)

def launch_onthefly_training(args):
    """Launch on-the-fly production training"""
    print("🚀 Launching On-the-Fly Production Training...")
    print("📊 Strategy: Dynamic image generation with 313K corpus")
    print("🎯 Goal: Full production training with unlimited variety")
    
    cmd = [
        "python", "src/training/train_onthefly.py",
        "--num-epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--train-samples-per-epoch", str(args.samples_per_epoch),
        "--log-dir", "logs/onthefly",
        "--checkpoint-dir", "models/checkpoints/onthefly"
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"📝 Command: {' '.join(cmd)}")
    return subprocess.run(cmd)

def launch_eos_focused_training(args):
    """Launch EOS-focused training"""
    print("🚀 Launching EOS-Focused Training...")
    print("🔚 Strategy: Enhanced EOS loss weighting (10x)")
    print("🎯 Goal: Force stronger EOS token generation")
    
    cmd = [
        "python", "train_eos_focused_v1.py",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--log-dir", "logs/eos_focused"
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"📝 Command: {' '.join(cmd)}")
    return subprocess.run(cmd)

def launch_pattern_fix_training(args):
    """Launch pattern-fix training"""
    print("🚀 Launching Pattern-Fix Training...")
    print("🔄 Strategy: Anti-repetition loss with pattern detection")
    print("🎯 Goal: Prevent repetitive pattern generation")
    
    cmd = [
        "python", "train_pattern_fix_v2.py",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size)
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"📝 Command: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Khmer OCR Production Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Strategies Available:

1. CURRICULUM (Recommended)
   - Progressive sequence length training (10→50 chars)
   - Forces EOS learning through curriculum approach
   - Based on successful testing results
   - Command: python launch_production_training.py curriculum

2. ONTHEFLY (Full Production)  
   - Dynamic image generation with 313K corpus
   - Unlimited training variety, fixed validation
   - Memory efficient, storage friendly
   - Command: python launch_production_training.py onthefly

3. EOS_FOCUSED (Specialized)
   - Enhanced EOS loss weighting (10x normal)
   - Direct approach to EOS generation problem
   - Command: python launch_production_training.py eos-focused

4. PATTERN_FIX (Specialized)
   - Anti-repetition loss with pattern detection
   - Prevents repetitive character sequences
   - Command: python launch_production_training.py pattern-fix

Recommended Flow:
1. Start with curriculum (20-50 epochs) to fix EOS
2. Continue with onthefly (100+ epochs) for production model

GPU Platform Recommendations:
- Local RTX 3090/4090: Full training with batch size 32
- Google Colab Pro A100: Excellent with batch size 64-96
- Google Colab T4: Limited to batch size 12, good for testing
- Enterprise A100 80GB: Maximum performance with batch size 96
        """
    )
    
    # Training strategy
    parser.add_argument(
        "strategy",
        choices=["curriculum", "onthefly", "eos-focused", "pattern-fix"],
        help="Training strategy to use"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs (default: 50 for curriculum, 150 for others)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size (auto-detected based on GPU if not specified)"
    )
    
    parser.add_argument(
        "--samples-per-epoch", "-s",
        type=int,
        default=10000,
        help="Training samples per epoch for on-the-fly training (default: 10000)"
    )
    
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Utility options
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system requirements, don't start training"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip system requirements check"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🇰🇭 Khmer OCR Production Training Launcher")
    print("="*60)
    
    # Check system requirements
    if not args.force:
        if not check_system_requirements():
            print("\n❌ System requirements check failed!")
            print("Fix the issues above or use --force to bypass checks.")
            sys.exit(1)
    
    if args.check_only:
        print("\n✅ System requirements check completed!")
        return
    
    # Auto-detect batch size if not specified
    if args.batch_size is None:
        args.batch_size = get_recommended_batch_size()
        print(f"\n🔧 Auto-selected batch size: {args.batch_size}")
    
    # Adjust default epochs for different strategies
    if args.epochs == 50:  # Default value
        if args.strategy == "onthefly":
            args.epochs = 150
            print(f"🔧 Auto-adjusted epochs for {args.strategy}: {args.epochs}")
    
    print(f"\n📋 Training Configuration:")
    print(f"   Strategy: {args.strategy}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    if args.strategy == "onthefly":
        print(f"   Samples/Epoch: {args.samples_per_epoch}")
    if args.resume:
        print(f"   Resume From: {args.resume}")
    
    # Confirm training start
    response = input(f"\n🚀 Start {args.strategy} training? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    print(f"\n🏁 Starting {args.strategy} training...")
    
    # Launch appropriate training
    if args.strategy == "curriculum":
        result = launch_curriculum_training(args)
    elif args.strategy == "onthefly":
        result = launch_onthefly_training(args)
    elif args.strategy == "eos-focused":
        result = launch_eos_focused_training(args)
    elif args.strategy == "pattern-fix":
        result = launch_pattern_fix_training(args)
    
    if result.returncode == 0:
        print(f"\n✅ {args.strategy} training completed successfully!")
    else:
        print(f"\n❌ {args.strategy} training failed!")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main() 