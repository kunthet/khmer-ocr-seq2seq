#!/usr/bin/env python3
"""
Google Colab Setup Check Script for Khmer OCR Training

This script verifies prerequisites and provides setup instructions for training
the Khmer OCR model on Google Colab.
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def check_system_requirements():
    """Check system requirements for training"""
    print("🔧 System Requirements Check")
    print("="*50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if running on Colab
    try:
        import google.colab
        print("✅ Running on Google Colab")
        is_colab = True
    except ImportError:
        print("❌ Not running on Google Colab")
        is_colab = False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Recommend batch size based on GPU memory
            if gpu_memory >= 40:
                batch_size = "48-64"
            elif gpu_memory >= 24:
                batch_size = "32-48"
            elif gpu_memory >= 16:
                batch_size = "16-32"
            else:
                batch_size = "8-16"
            
            print(f"   Recommended batch size: {batch_size}")
        else:
            print("❌ No GPU available")
            print("   ⚠️  GPU is required for training. Please enable GPU runtime.")
    except ImportError:
        print("❌ PyTorch not installed")
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3
        print(f"✅ RAM: {ram_gb:.1f}GB")
        
        if ram_gb < 12:
            print("   ⚠️  Consider using High-RAM runtime for better performance")
    except ImportError:
        print("❌ psutil not available")
    
    # Check disk space
    try:
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / 1024**3
        total_gb = disk_usage.total / 1024**3
        print(f"✅ Disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        
        if free_gb < 20:
            print("   ⚠️  Low disk space. Consider cleaning up temporary files.")
    except:
        print("❌ Cannot check disk space")
    
    return is_colab

def check_google_drive():
    """Check Google Drive mounting"""
    print("\n📁 Google Drive Check")
    print("="*50)
    
    drive_path = "/content/drive/MyDrive"
    
    if os.path.exists(drive_path):
        print("✅ Google Drive is mounted")
        
        # Check available space
        try:
            import shutil
            _, _, free_bytes = shutil.disk_usage(drive_path)
            free_gb = free_bytes / 1024**3
            print(f"✅ Google Drive free space: {free_gb:.1f}GB")
            
            if free_gb < 50:
                print("   ⚠️  Low Google Drive space. Training requires 50GB+ for checkpoints.")
        except:
            print("   ❌ Cannot check Google Drive space")
        
        # Check/create checkpoint directory
        checkpoint_dir = f"{drive_path}/KhmerOCR_Checkpoints"
        if not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                os.makedirs(f"{checkpoint_dir}/models", exist_ok=True)
                os.makedirs(f"{checkpoint_dir}/logs", exist_ok=True)
                os.makedirs(f"{checkpoint_dir}/validation_set", exist_ok=True)
                print("✅ Checkpoint directory created")
            except Exception as e:
                print(f"❌ Failed to create checkpoint directory: {e}")
        else:
            print("✅ Checkpoint directory exists")
        
        return True
    else:
        print("❌ Google Drive not mounted")
        print("   📝 Run: from google.colab import drive; drive.mount('/content/drive')")
        return False

def check_repository():
    """Check repository setup"""
    print("\n📦 Repository Check")
    print("="*50)
    
    repo_dir = "/content/khmer-ocr-seq2seq"
    
    if os.path.exists(repo_dir):
        print("✅ Repository directory exists")
        
        # Check if it's a git repository
        git_dir = os.path.join(repo_dir, ".git")
        if os.path.exists(git_dir):
            print("✅ Git repository detected")
            
            # Check current directory
            current_dir = os.getcwd()
            if current_dir != repo_dir:
                print(f"   ⚠️  Not in repository directory. Current: {current_dir}")
                print(f"   📝 Run: os.chdir('{repo_dir}')")
            else:
                print("✅ Working in repository directory")
        else:
            print("❌ Not a git repository")
        
        return True
    else:
        print("❌ Repository not found")
        print("   📝 Clone repository:")
        print("   git clone https://github.com/kunthet/khmer-ocr-seq2seq.git /content/khmer-ocr-seq2seq")
        return False

def check_data_files():
    """Check required data files"""
    print("\n📊 Data Files Check")
    print("="*50)
    
    # Check corpus files
    corpus_dir = "data/processed"
    
    if os.path.exists(corpus_dir):
        print("✅ Corpus directory exists")
        
        # Check for training files (single or multiple)
        train_files = []
        single_train = os.path.join(corpus_dir, "train.txt")
        
        if os.path.exists(single_train):
            # Single train.txt file
            train_files.append(single_train)
            try:
                with open(single_train, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"✅ train.txt: {line_count:,} lines")
            except Exception as e:
                print(f"❌ train.txt: Error reading file - {e}")
        else:
            # Check for multiple training files (train_0.txt, train_1.txt, etc.)
            import glob
            train_pattern = os.path.join(corpus_dir, "train_*.txt")
            train_files_found = sorted(glob.glob(train_pattern))
            
            if train_files_found:
                total_train_lines = 0
                print(f"✅ Found {len(train_files_found)} training files:")
                
                for train_file in train_files_found:
                    try:
                        with open(train_file, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f)
                            total_train_lines += line_count
                            file_name = os.path.basename(train_file)
                            print(f"   - {file_name}: {line_count:,} lines")
                    except Exception as e:
                        print(f"   ❌ {os.path.basename(train_file)}: Error reading file - {e}")
                        
                print(f"✅ Total training lines: {total_train_lines:,}")
            else:
                print("❌ No training files found (train.txt or train_*.txt)")
        
        # Check validation and test files
        for file in ["val.txt", "test.txt"]:
            file_path = os.path.join(corpus_dir, file)
            if os.path.exists(file_path):
                # Count lines
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    print(f"✅ {file}: {line_count:,} lines")
                except Exception as e:
                    print(f"❌ {file}: Error reading file - {e}")
            else:
                print(f"❌ {file}: Missing")
    else:
        print("❌ Corpus directory not found")
        print("   📝 Required structure:")
        print("   data/processed/")
        print("   ├── train.txt (or train_0.txt, train_1.txt, etc.)")
        print("   ├── val.txt")
        print("   └── test.txt")
    
    # Check fonts
    fonts_dir = "fonts"
    
    if os.path.exists(fonts_dir):
        font_files = [f for f in os.listdir(fonts_dir) if f.endswith(('.ttf', '.otf'))]
        if font_files:
            print(f"✅ Fonts directory: {len(font_files)} font files")
            for font in font_files:
                print(f"   - {font}")
        else:
            print("❌ No font files found")
    else:
        print("❌ Fonts directory not found")
        print("   📝 Required: fonts/ directory with Khmer .ttf/.otf files")

def check_validation_set():
    """Check validation set"""
    print("\n🔍 Validation Set Check")
    print("="*50)
    
    validation_dir = "data/validation_fixed"
    
    if os.path.exists(validation_dir):
        metadata_file = os.path.join(validation_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print("✅ Validation set exists")
                print(f"   Total samples: {metadata['total_samples']:,}")
                print(f"   Image height: {metadata['generation_config']['image_height']}px")
                print(f"   Fonts used: {len(metadata['fonts_used'])}")
                
                # Check image and label counts
                images_dir = os.path.join(validation_dir, "images")
                labels_dir = os.path.join(validation_dir, "labels")
                
                if os.path.exists(images_dir):
                    image_count = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
                    print(f"   Images: {image_count:,}")
                
                if os.path.exists(labels_dir):
                    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                    print(f"   Labels: {label_count:,}")
                
                return True
            except Exception as e:
                print(f"❌ Error reading validation metadata: {e}")
        else:
            print("❌ Validation metadata missing")
    else:
        print("❌ Validation set not found")
        print("   📝 Will generate during training setup")
    
    return False

def check_dependencies():
    """Check Python dependencies"""
    print("\n🐍 Dependencies Check")
    print("="*50)
    
    required_packages = [
        "torch", "torchvision", "torchaudio",
        "PIL", "numpy", "matplotlib", "tqdm",
        "cv2", "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "PIL":
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📝 Install missing packages:")
        if "torch" in missing_packages:
            print("   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        if "cv2" in missing_packages:
            print("   !pip install opencv-python-headless")
        if "PIL" in missing_packages:
            print("   !pip install pillow")
        other_packages = [p for p in missing_packages if p not in ["torch", "torchvision", "torchaudio", "cv2", "PIL"]]
        if other_packages:
            print(f"   !pip install {' '.join(other_packages)}")
    
    return len(missing_packages) == 0

def provide_setup_instructions():
    """Provide setup instructions"""
    print("\n📋 Setup Instructions")
    print("="*50)
    
    print("1. 📁 Mount Google Drive:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    
    print("\n2. 📥 Clone Repository:")
    print("   !git clone https://github.com/kunthet/khmer-ocr-seq2seq.git /content/khmer-ocr-seq2seq")
    print("   import os")
    print("   os.chdir('/content/khmer-ocr-seq2seq')")
    
    print("\n3. 📦 Install Dependencies:")
    print("   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   !pip install pillow numpy matplotlib tqdm opencv-python-headless")
    
    print("\n4. 📊 Prepare Data:")
    print("   - Upload corpus files to data/processed/")
    print("   - Upload Khmer fonts to fonts/")
    print("   - Or use the notebook's automatic setup")
    
    print("\n5. 🚀 Start Training:")
    print("   - Open khmer_ocr_colab_training.ipynb")
    print("   - Run all cells in sequence")
    print("   - Monitor training progress")

def main():
    """Main setup check"""
    print("🇰🇭 Khmer OCR Google Colab Setup Check")
    print("="*60)
    
    is_colab = check_system_requirements()
    
    if not is_colab:
        print("\n❌ This script is designed for Google Colab")
        print("   Please run this on Google Colab for training")
        return
    
    drive_ok = check_google_drive()
    repo_ok = check_repository()
    deps_ok = check_dependencies()
    
    if repo_ok:
        check_data_files()
        check_validation_set()
    
    print("\n🎯 Setup Status Summary")
    print("="*50)
    
    if drive_ok and repo_ok and deps_ok:
        print("✅ All prerequisites met!")
        print("🚀 Ready to start training!")
        print("\n📔 Open khmer_ocr_colab_training.ipynb and run all cells")
    else:
        print("❌ Some prerequisites missing")
        print("📝 Follow the setup instructions above")
        provide_setup_instructions()

if __name__ == "__main__":
    main() 