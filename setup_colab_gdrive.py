#!/usr/bin/env python3
"""
Google Colab Setup Script for Khmer OCR Training

This script sets up Google Drive mounting and checkpoint directories
for seamless training with automatic backup and resumption.
"""

import os
import sys
from pathlib import Path
import logging

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def mount_google_drive():
    """Mount Google Drive in Colab environment."""
    logger = logging.getLogger(__name__)
    
    try:
        # Check if we're in Colab
        import google.colab
        from google.colab import drive
        
        # Mount Google Drive
        logger.info("🔗 Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Verify mount
        gdrive_path = Path("/content/drive/MyDrive")
        if gdrive_path.exists():
            logger.info("✅ Google Drive mounted successfully")
            return True
        else:
            logger.error("❌ Google Drive mount failed")
            return False
            
    except ImportError:
        logger.warning("⚠️ Not running in Google Colab - Google Drive mounting skipped")
        return False
    except Exception as e:
        logger.error(f"❌ Google Drive mount failed: {e}")
        return False

def setup_checkpoint_directories():
    """Create necessary checkpoint directories."""
    logger = logging.getLogger(__name__)
    
    # Local checkpoint directory
    local_checkpoint_dir = Path("models/checkpoints")
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Local checkpoint directory: {local_checkpoint_dir}")
    
    # Google Drive checkpoint directory
    gdrive_checkpoint_dir = Path("/content/drive/MyDrive/KhmerOCR_Checkpoints")
    if gdrive_checkpoint_dir.parent.exists():  # Check if Google Drive is mounted
        gdrive_checkpoint_dir.mkdir(exist_ok=True)
        (gdrive_checkpoint_dir / "models").mkdir(exist_ok=True)
        (gdrive_checkpoint_dir / "logs").mkdir(exist_ok=True)
        (gdrive_checkpoint_dir / "validation_set").mkdir(exist_ok=True)
        logger.info(f"✅ Google Drive checkpoint directory: {gdrive_checkpoint_dir}")
        return True
    else:
        logger.warning("⚠️ Google Drive not available - local-only backup")
        return False

def check_existing_checkpoints():
    """Check for existing checkpoints in local and Google Drive."""
    logger = logging.getLogger(__name__)
    
    # Check local checkpoints
    local_checkpoint_dir = Path("models/checkpoints")
    local_checkpoints = list(local_checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    local_best = local_checkpoint_dir / "best_model.pth"
    
    logger.info(f"📁 Local checkpoints: {len(local_checkpoints)} found")
    if local_best.exists():
        logger.info("✅ Local best model found")
    
    # Check Google Drive checkpoints
    gdrive_checkpoint_dir = Path("/content/drive/MyDrive/KhmerOCR_Checkpoints/models")
    if gdrive_checkpoint_dir.exists():
        gdrive_checkpoints = list(gdrive_checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        gdrive_best = gdrive_checkpoint_dir / "best_model.pth"
        
        logger.info(f"☁️ Google Drive checkpoints: {len(gdrive_checkpoints)} found")
        if gdrive_best.exists():
            logger.info("✅ Google Drive best model found")
            
        # Check training history
        history_file = gdrive_checkpoint_dir / "training_history.json"
        if history_file.exists():
            logger.info("✅ Training history found in Google Drive")
            try:
                import json
                with open(history_file, 'r') as f:
                    history = json.load(f)
                logger.info(f"📊 Last training: Epoch {history.get('epoch', 'N/A')}, CER: {history.get('best_cer', 'N/A'):.2%}")
            except Exception as e:
                logger.warning(f"⚠️ Could not read training history: {e}")
    else:
        logger.info("☁️ No Google Drive checkpoints found")

def get_latest_checkpoint_path():
    """Get the path to the latest checkpoint for resuming training."""
    logger = logging.getLogger(__name__)
    
    # Check local checkpoints first
    local_checkpoint_dir = Path("models/checkpoints")
    local_checkpoints = list(local_checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    
    if local_checkpoints:
        # Extract epoch numbers and find latest
        def get_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return -1
        
        latest_local = max(local_checkpoints, key=get_epoch)
        logger.info(f"📁 Latest local checkpoint: {latest_local}")
        return str(latest_local)
    
    # Check Google Drive checkpoints
    gdrive_checkpoint_dir = Path("/content/drive/MyDrive/KhmerOCR_Checkpoints/models")
    if gdrive_checkpoint_dir.exists():
        gdrive_checkpoints = list(gdrive_checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if gdrive_checkpoints:
            def get_epoch(path):
                try:
                    return int(path.stem.split('_')[-1])
                except:
                    return -1
            
            latest_gdrive = max(gdrive_checkpoints, key=get_epoch)
            logger.info(f"☁️ Latest Google Drive checkpoint: {latest_gdrive}")
            return str(latest_gdrive)
    
    logger.info("📁 No checkpoints found for resuming")
    return None

def setup_colab_environment():
    """Complete Google Colab environment setup."""
    logger = setup_logging()
    
    logger.info("🚀 Setting up Google Colab environment for Khmer OCR training")
    logger.info("=" * 60)
    
    # Check if we're in Colab
    try:
        import google.colab
        logger.info("✅ Running in Google Colab")
    except ImportError:
        logger.warning("⚠️ Not running in Google Colab")
    
    # Mount Google Drive
    gdrive_success = mount_google_drive()
    
    # Setup directories
    checkpoint_success = setup_checkpoint_directories()
    
    # Check existing checkpoints
    check_existing_checkpoints()
    
    # Get resume checkpoint path
    resume_path = get_latest_checkpoint_path()
    
    # Summary
    logger.info("=" * 60)
    logger.info("🎯 Setup Summary:")
    logger.info(f"   Google Drive: {'✅ Ready' if gdrive_success else '❌ Failed'}")
    logger.info(f"   Checkpoints: {'✅ Ready' if checkpoint_success else '⚠️ Local only'}")
    logger.info(f"   Resume from: {resume_path if resume_path else 'Fresh start'}")
    
    if gdrive_success and checkpoint_success:
        logger.info("🟢 Environment ready for training with Google Drive backup!")
    elif checkpoint_success:
        logger.info("🟡 Environment ready for training (local backup only)")
    else:
        logger.error("🔴 Environment setup failed")
        return False
    
    return {
        'gdrive_success': gdrive_success,
        'checkpoint_success': checkpoint_success,
        'resume_checkpoint': resume_path,
        'gdrive_dir': "/content/drive/MyDrive/KhmerOCR_Checkpoints" if gdrive_success else None
    }

def show_gdrive_usage():
    """Show Google Drive usage for checkpoints."""
    logger = logging.getLogger(__name__)
    
    gdrive_dir = Path("/content/drive/MyDrive/KhmerOCR_Checkpoints")
    if not gdrive_dir.exists():
        logger.info("☁️ No Google Drive checkpoint directory found")
        return
    
    total_size = 0
    file_count = 0
    
    # Calculate size of all checkpoint files
    for file_path in gdrive_dir.rglob("*.pth"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1
    
    # Calculate size of other files (logs, history, etc.)
    for file_path in gdrive_dir.rglob("*"):
        if file_path.is_file() and not file_path.suffix == '.pth':
            total_size += file_path.stat().st_size
            file_count += 1
    
    size_mb = total_size / (1024 * 1024)
    logger.info(f"☁️ Google Drive Usage:")
    logger.info(f"   Total files: {file_count}")
    logger.info(f"   Total size: {size_mb:.1f} MB")
    
    if size_mb > 1000:  # 1GB
        logger.warning(f"⚠️ Large checkpoint directory ({size_mb:.1f} MB)")
        logger.info("   Consider cleaning old checkpoints if running low on Drive space")

if __name__ == "__main__":
    # Run setup when called directly
    result = setup_colab_environment()
    
    if result and result['gdrive_success']:
        show_gdrive_usage()
    
    # Print environment variables for use in training scripts
    if result and result['resume_checkpoint']:
        print(f"\nTo resume training, use:")
        print(f"python src/training/train_onthefly.py --resume {result['resume_checkpoint']}")
    else:
        print(f"\nTo start fresh training, use:")
        print(f"python src/training/train_onthefly.py") 