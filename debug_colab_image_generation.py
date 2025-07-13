"""
Debug script for testing synthetic image generation in a Colab environment.

This script will:
1. Initialize the necessary components from the khmer-ocr-seq2seq project.
2. Create an instance of the on-the-fly dataset generator.
3. Generate a single, fully augmented training sample.
4. Display the generated image and its corresponding text label.
"""

import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.font_manager as fm

# Suppress verbose logging from the project modules
import logging
logging.basicConfig(level=logging.WARNING)

# --- Check for necessary files and directories ---
import os
required_dirs = ['src', 'configs', 'fonts', 'data/processed']
if not all(os.path.exists(d) for d in required_dirs):
    print("❌ Error: You must run this script from the root of the 'khmer-ocr-seq2seq' repository.")
    print("Please ensure your Colab environment is set up correctly, e.g.:")
    print("!git clone https://github.com/kunthet/khmer-ocr-seq2seq.git")
    print("%cd khmer-ocr-seq2seq")
    # Stop execution if files are missing
    # In a real script, you might raise an exception. In Colab, we just print.
else:
    # --- Import project modules ---
    from src.utils.config import ConfigManager
    from src.data.onthefly_dataset import OnTheFlyDataset

    def generate_and_display_sample():
        """
        Initializes the dataset, generates one sample, and displays it.
        """
        print("🚀 Starting Khmer OCR Synthetic Image Generation Test...")
        print("="*60)

        try:
            # 1. Initialize ConfigManager to load all project configurations
            print("1. Loading configurations...")
            config_manager = ConfigManager(
                config_path='configs/train_config.yaml'
            )
            print("   ✅ Configurations loaded.")

            # 2. Initialize the On-the-fly Dataset for the 'train' split
            #    This will set up the text renderer and data augmentation pipeline.
            print("2. Initializing OnTheFlyDataset...")
            # We set a random seed for reproducibility of this test.
            # In real training, this might be omitted for more randomness.
            dataset = OnTheFlyDataset(
                split='train',
                config_manager=config_manager,
                corpus_dir='data/processed',
                samples_per_epoch=100, # Only need a few for testing
                augment_prob=1.0, # Force augmentation for the test
                random_seed=42
            )
            print("   ✅ Dataset initialized.")
            print(f"   - Found {len(dataset.text_lines)} text lines.")
            print(f"   - Using {len(dataset.text_renderer.fonts)} fonts.")


            # 3. Generate one sample
            print("3. Generating one augmented sample...")
            # The __getitem__ method returns (image_tensor, target_tensor, text_label)
            # We'll pick a random index to get a different image each time.
            random_index = random.randint(0, len(dataset) - 1)
            image_tensor, _, text_label = dataset[random_index]
            print("   ✅ Sample generated.")


            # 4. Convert the image tensor back to a displayable format
            print("4. Preparing image for display...")
            # The tensor is [C, H, W] and normalized. We need to:
            # - Convert to a NumPy array
            # - Transpose to [H, W, C] for plotting
            # - Squeeze the channel dimension if it's grayscale (C=1)
            # - Denormalize from [0, 1] to [0, 255]
            image_numpy = image_tensor.numpy()
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            if image_numpy.shape[2] == 1:
                image_numpy = image_numpy.squeeze(axis=2)

            # Denormalize (assuming the tensor was normalized to [0, 1])
            image_numpy = (image_numpy * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_numpy)
            print("   ✅ Image ready.")

            # 5. Display the image and its label
            print("5. Displaying results...")

            # Set a Khmer font for the plot title to avoid warnings
            font_path = 'fonts/KhmerOS.ttf'
            if os.path.exists(font_path):
                khmer_font = fm.FontProperties(fname=font_path)
            else:
                khmer_font = None # Fallback to default if not found

            plt.figure(figsize=(15, 4))
            plt.imshow(pil_image, cmap='gray')
            plt.title(f"Generated Text: \n{text_label}", fontproperties=khmer_font, fontsize=14, pad=20)
            plt.axis('off')
            plt.savefig('debug_sample_output.png', dpi=150, bbox_inches='tight')
            print("   ✅ Sample saved as 'debug_sample_output.png'")
            plt.close()

        except FileNotFoundError as e:
            print(f"❌ File Not Found Error: {e}")
            print("   Please ensure all required data and font files are present.")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")

    # Run the generation and display function
    generate_and_display_sample() 