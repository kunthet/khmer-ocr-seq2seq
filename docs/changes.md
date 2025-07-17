# Change Log

## Feature: Image Width Limitation Fix
**Purpose:**
Fixed critical inference limitation where images were being cropped to 512 pixels width during inference, causing predictions to be cut off for longer text sequences, despite the configuration specifying 800px maximum width.

**Implementation:**
- Updated `KhmerOCREngine` class in `src/inference/ocr_engine.py` to use configurable `max_width` instead of hardcoded 512px
- Changed default `max_width` parameter from 512 to 800 to match configuration
- Modified `from_checkpoint()` method to read `image_width` from config and pass it to engine initialization
- Added `image_width` field to `DataConfig` class in `src/utils/config.py` for proper configuration access
- Updated all inference scripts (`quick_test.py`, `src/inference/demo.py`) to use configured width
- Updated memory estimation function in `train_onthefly.py` to reflect configurable width

**History:**
- Created — Fixed 512px hardcoded limitation, implemented configurable width system using 800px default
- Updated — Doubled maximum width from 800px to 1600px to handle longer text sequences. Updated all components: config file, DataConfig class, KhmerOCREngine defaults, and memory estimation calculations.

## Feature: Model Configuration Loading Fix
**Purpose:**
Fixed critical model architecture mismatch where training scripts were creating models with default parameters instead of loading the correct configuration from YAML files, causing checkpoint loading failures.

**Implementation:**
- Updated `create_model()` functions in training scripts to pass `ConfigManager` object instead of just `vocab_size`
- Fixed `train_onthefly.py`, `train_production.py`, and `train_advanced_performance.py`
- Model now correctly loads decoder_hidden_size=512 and attention_hidden_size=512 from config files
- Enables proper checkpoint loading and resuming from previously trained models

**History:**
- Created — Fixed model initialization to use ConfigManager for proper architecture configuration instead of default values.
- Updated — Made checkpoint loading more robust by using .get() for optional fields (training_history, best_cer, global_step) to handle different checkpoint formats gracefully.
- Updated — Enhanced on-the-fly training to use CurriculumDataset with max_length=150 for consistent sequence length control and syllable-based truncation, improving training stability and text quality.
- Updated — Created CompatibleCollateFunction class in train_onthefly.py to handle batch collation directly (optimized single-pass processing instead of calling curriculum_collate_fn, provides required 'image_lengths'/'target_lengths' keys for trainer, picklable for multiprocessing).
- Updated — Fixed training_history structure in checkpoint loading to use dictionary format instead of list, preventing "list indices must be integers" TypeError during training.
- Updated — Fixed inference and testing scripts (inference_test.py, debug_inference.py, src/inference/demo.py, quick_test.py) to use ConfigManager for proper model architecture loading instead of default parameters.
- Updated — Provided corrected Colab training configuration with proper parameter naming that matches current ConfigManager structure.
- Updated — Fixed KhmerOCREngine.from_checkpoint() to use ConfigManager for model creation before loading state_dict, preventing size mismatch errors during checkpoint loading.
- Updated — Fixed benchmark mode in inference_test.py to properly unpack 3 values (image_tensor, target_tensor, text) from SyntheticImageDataset instead of expecting only 2 values.
- Created — Training sample inspection script (inspect_training_samples.py) to generate and save sample images from the training pipeline for manual verification of image generation quality.

## Feature: [Training Sample Inspector with Inference Results]
**Purpose:**  
Enhanced the existing training sample inspector to include model inference results and accuracy calculation. This allows for comprehensive evaluation of model performance on synthetically generated training data.

**Implementation:**  
- Modified `inspect_training_samples.py` to load the OCR engine from checkpoint
- Added inference functionality using `KhmerOCREngine.recognize()` method
- Implemented accuracy calculation with both exact match and character-level metrics
- Enhanced HTML viewer with:
  - Prominent accuracy metrics display (exact match % and character-level %)
  - Color-coded visual comparison (green for ground truth, blue for predictions)
  - Red/green borders indicating correct/incorrect predictions
  - Visual status indicators (✅❌) for each sample
- Added detailed accuracy reporting in metadata file and console output
- Used best model checkpoint at `models/checkpoints/full/best_model.pth`

**Testing Results:**
- Tested with 50 synthetic samples from training pipeline
- Achieved 6.00% exact match accuracy (3/50 samples)
- Achieved 28.51% character-level accuracy
- Successfully generated comprehensive HTML report with side-by-side comparisons

**History:**
- Created — Enhanced inspect_training_samples.py with OCR inference, accuracy calculation, and improved HTML visualization for training sample verification.

## Feature: [Core] EOS Generation Fix via Curriculum Learning
**Purpose:**
Resolved a critical failure where the model would generate endless repetitive character sequences instead of terminating with an End-Of-Sequence (EOS) token. This fix is essential for producing coherent and usable OCR output.

**Implementation:**
1.  **Architectural Analysis (`analyze_eos_architecture.py`)**: A deep analysis was conducted, confirming the model's architecture was correct and the issue was related to training dynamics, not a structural flaw. Findings were documented in `docs/EOS_ARCHITECTURAL_ANALYSIS.md`.
2.  **Curriculum Learning Trainer (`train_curriculum_eos_v1.py`)**: A new training script was developed to implement a progressive sequence length curriculum.
    -   **`CurriculumDataset`**: A wrapper dataset was created to truncate target sequences to a dynamically increasing maximum length.
    -   **`CurriculumEOSLoss`**: A custom loss function was designed with an adaptive weight for the EOS token, applying stronger pressure during early training phases with shorter sequences.
    -   **Progressive Length Schedule**: Training starts with very short sequences (10 tokens) and gradually increases the allowed length across several phases, forcing the model to learn termination before tackling complex sequences.
3.  **Custom Collation (`curriculum_collate_fn`)**: Implemented a custom collate function to handle the padding of variable-width images within a batch, which was necessary for the curriculum approach.

**History:**
-   **Created**: Implemented curriculum learning to fix the catastrophic EOS generation failure. The model now successfully learns to terminate sequences, achieving 100% EOS accuracy on shorter sequences and stopping early in the curriculum due to high success rates.
-   **Updated**: Enhanced curriculum learning with syllable-based text truncation to preserve Khmer linguistic integrity. Replaced arbitrary character truncation with intelligent syllable boundary detection using `split_syllables_advanced()` function. This ensures all truncated text remains linguistically valid and readable, improving training quality and model performance.

## Feature: Curriculum Training EOS Learning
**Purpose:**  
Implements curriculum learning strategy to teach the model proper EOS (End-of-Sequence) token generation through progressive sequence length training.

**Implementation:**  
Created `train_curriculum_eos_v1.py` with:
- 5-phase curriculum: 10→15→20→30→50 character sequences
- Adaptive EOS loss weighting (starts at 30x, reduces each phase)
- Syllable-based text truncation using `split_syllables_advanced`
- CurriculumDataset wrapper for length-limited training
- Progressive teacher forcing schedule

**History:**
- Created — Initial curriculum implementation with progressive length training and adaptive EOS loss weighting.
- Updated — Integrated syllable-based truncation system to preserve Khmer linguistic boundaries during curriculum training.
- Updated — Modified checkpoint saving to save every epoch and keep only latest 5 checkpoints with automatic cleanup.
- Updated — Added resume functionality with --resume and --auto-resume options for interrupted training recovery.
- Updated — Fixed argument compatibility with production launcher by adding --num-epochs, --batch-size, --log-dir, and --checkpoint-dir arguments.
- Updated — Enhanced resume functionality to properly restore curriculum phase progress, ensuring training continues from the correct phase and epoch within that phase instead of restarting from Phase 1.
- Updated — Fixed critical bug in phase calculation logic (`_calculate_phase_from_epoch`) that was incorrectly determining which curriculum phase to resume from, causing training to restart from wrong phases.
- Updated — Added comprehensive logging system with timestamped log files, epoch announcements before training starts, and detailed progress tracking to file and console for better monitoring and debugging.
- Updated — Fixed TypeError in checkpoint saving by ensuring checkpoint_dir remains a Path object when overridden by command line arguments.
- Updated — Fixed checkpoint directory mismatch between launcher (models/checkpoints/curriculum) and training script (models/checkpoints/curriculum_eos_v1) by updating launcher to use correct directory.
- Updated — **Critical Fix**: Images now properly match truncated text labels by regenerating images from syllable-truncated text instead of using original full-length images.
- Updated — **Quality Enhancement**: Reduced augmentation artifacts (blur_prob: 0.8→0.1, morph_prob: 0.8→0.1, noise_prob: 0.8→0.1, augment_prob: 0.8→0.2) for better text readability in curriculum training while preserving tensor format compatibility.

## Feature: Critical Loss Function Fix - LogSoftmax Implementation
**Purpose:**  
Fixed critical mismatch between decoder output and loss function that was causing negative losses and poor model performance. Implemented LogSoftmax output in decoder as specified in PRD to work correctly with NLLLoss.

**Implementation:**  
Resolved fundamental training issue where:
- Decoder was outputting raw logits instead of log probabilities
- Trainer was using NLLLoss which expects log probabilities
- This mismatch caused negative losses and prevented proper learning

Changes made:
- Added LogSoftmax to decoder forward_step method output
- Updated all decoder methods (forward, generate, beam_search) to handle log probabilities
- Updated seq2seq model decode_step to return log probabilities
- Updated trainer comments to reflect log probabilities
- Ensured proper PRD compliance with LogSoftmax + NLLLoss combination

**History:**
- Created — Fixed critical loss function mismatch causing negative losses (-3.8369, -4.2791, etc.) and 116.84% CER. Now properly implements LogSoftmax + NLLLoss as specified in PRD, enabling correct gradient flow and model learning.

## Feature: Google Drive Checkpoint Backup
**Purpose:**  
Automatic backup of training checkpoints to Google Drive for seamless training resumption in Google Colab environments, preventing loss of training progress due to session disconnections.

**Implementation:**  
Enhanced CheckpointManager with Google Drive integration:
- Automatic backup of best models and periodic checkpoints to Google Drive
- Smart fallback loading from Google Drive when local checkpoints are missing
- Training history and metrics backup for complete session recovery
- Colab environment setup script for automated Google Drive mounting and directory creation
- Configurable backup frequency and checkpoint cleanup

**History:**
- Created — Initial implementation with automatic Google Drive backup, fallback loading, training history preservation, and Colab setup automation.

## Feature: Multiple Training Files Support
**Purpose:**  
Support loading training data from multiple files (train_0.txt, train_1.txt, etc.) instead of requiring a single train.txt file, providing flexibility for large datasets and manual data splitting.

**Implementation:**  
Updated data loading mechanisms in OnTheFlyDataset, KhmerCorpusDataset, and KhmerCorpusProcessor to handle multiple training files. The system now:
- Automatically detects and loads multiple training files with pattern train_*.txt
- Falls back to single train.txt if multiple files are not found
- Maintains backward compatibility with existing single-file setups
- Aggregates statistics and line counts across all training files

**History:**
- Created — Initial implementation with automatic detection of multiple training files, fallback to single file, and comprehensive testing verification.

## Feature: [Project Initialization]
**Purpose:**  
Set up the basic project structure and development environment for the Khmer OCR Seq2Seq implementation.

**Implementation:**  
Created project directory structure, configuration files, and documentation framework. Established the foundation for modular development following CLEAN code principles.

**History:**
- Created — Initial project structure with directory organization, PRD tracking, and API context documentation.
- Updated — Complete Phase 1 implementation: directory structure (src/, data/, models/, configs/, tests/, logs/), requirements.txt with all dependencies, comprehensive configuration system with YAML files and Python ConfigManager class, complete Khmer vocabulary (117 tokens), Python package structure with __init__.py files, and detailed README.md with project documentation.

## Feature: [Model Architecture]
**Purpose:**
Implement the complete Seq2Seq model architecture, including the CRNN encoder, Bahdanau attention, and GRU decoder, as specified in the PRD.

**Implementation:**
- **`src/models/encoder.py`**: Created `CRNNEncoder` class with the 13-layer architecture.
- **`src/models/attention.py`**: Implemented `BahdanauAttention` and `MultiHeadBahdanauAttention`.
- **`src/models/decoder.py`**: Built `AttentionGRUDecoder` with teacher forcing, greedy search, and beam search capabilities.
- **`src/models/seq2seq.py`**: Integrated all components into the `KhmerOCRSeq2Seq` model, with methods for training, inference, and checkpointing.

**History:**
- Created — Implemented and tested all model components, ensuring they function correctly end-to-end. Total model size is ~16.3M parameters.

## Feature: [Training Infrastructure]
**Purpose:**
Implement complete training pipeline with teacher forcing, validation, checkpoint management, and monitoring for production-ready training of the Khmer OCR model.

**Implementation:**
- **`src/training/trainer.py`**: Created `Trainer` class with Adam optimizer, teacher forcing (ratio=1.0), gradient clipping, and TensorBoard logging.
- **`src/training/validator.py`**: Built `Validator` with CER calculation using Levenshtein distance, greedy decoding validation, and comprehensive metrics.
- **`src/training/checkpoint_manager.py`**: Implemented checkpoint persistence with best model tracking, automatic cleanup, and resume functionality.
- **`src/training/train.py`**: Main training script with argument parsing, device management, and end-to-end training pipeline.

**History:**
- Created — Complete training infrastructure supporting resume training, early stopping at 1.0% CER target, best model tracking, and comprehensive metrics logging. Successfully tested with mini training run.

## Feature: [Inference & Evaluation API]
**Purpose:**
Provide production-ready inference API with preprocessing, multiple decoding methods, and comprehensive evaluation metrics for deployed Khmer OCR system.

**Implementation:**
- **`src/inference/ocr_engine.py`**: Created `KhmerOCREngine` with image preprocessing, greedy/beam search decoding, batch processing, and confidence scoring.
- **`src/inference/metrics.py`**: Implemented comprehensive `OCRMetrics` with CER, WER, BLEU, character-level metrics, and confidence calibration.
- **`src/inference/demo.py`**: Demo script showcasing all inference capabilities with sample Khmer text generation and evaluation.

**History:**
- Created — Full inference API with support for greedy and beam search decoding, image preprocessing (contrast enhancement, denoising, binarization), batch processing, and comprehensive evaluation metrics including CER, WER, sequence accuracy, and confidence calibration analysis.

## Feature: [Testing & Validation]
**Purpose:**
Ensure system robustness and production readiness through comprehensive testing, validation, and performance optimization before full-scale training.

**Implementation:**
- **`tests/test_vocab.py`**: Unit tests for vocabulary class with encoding/decoding validation.
- **`tests/test_data_pipeline.py`**: Tests for text rendering, augmentation, and dataset functionality.
- **`tests/test_models.py`**: Comprehensive model component tests (encoder, attention, decoder, seq2seq).
- **`tests/test_integration.py`**: End-to-end pipeline integration tests with performance validation.
- **`run_small_scale_training.py`**: Small-scale training validation script (50-100 samples).
- **`profile_memory_usage.py`**: Memory profiling and performance optimization analysis.

**History:**
- Created — Complete testing infrastructure covering unit tests, integration tests, small-scale training validation, and memory profiling. All tests pass successfully, confirming system readiness for production training.

## Feature: [Corpus Data Integration & Production Training Setup]
**Purpose:**
Integrate real-world Khmer corpus data from multiple sources into the training pipeline and establish production-ready training infrastructure for achieving target ≤1.0% CER performance.

**Implementation:**
- **`src/data/corpus_processor.py`**: Comprehensive corpus processing pipeline that extracts, cleans, and filters text from Gemini-generated and Wikipedia sources. Handles line splitting, character validation, length filtering, and train/val/test splitting (80%/10%/10%).
- **`src/data/corpus_dataset.py`**: PyTorch dataset implementation for corpus-based training that renders real text to images with augmentation, handles variable-width images, and provides proper SOS/EOS token encoding.
- **`src/training/train_production.py`**: Production training script with automatic batch size calculation, environment validation, memory optimization, and comprehensive logging for 150-epoch training runs.
- **`process_corpus.py`**: Standalone utility script for corpus processing with detailed statistics reporting and data validation.
- **`test_corpus_integration.py`**: Integration test suite validating corpus dataset compatibility with existing training infrastructure.

**History:**
- Created — Successfully processed 240,007 text lines from dual sources: 82,338 Gemini-generated lines (34.3%) across 20+ domains with difficulty levels, and 157,669 Khmer Wikipedia lines (65.7%) from real-world articles. Character set contains 815 unique characters with average line length of 83.5 characters. Implemented production-ready dataset with real text rendering, dynamic augmentation, and proper vocabulary handling. Completed end-to-end integration testing confirming 16.3M parameter model compatibility with corpus data and successful forward/inference passes.
- Ready — System is now ready for full-scale production training targeting ≤1.0% Character Error Rate performance with comprehensive monitoring and checkpoint management. 

## Feature: [Synthetic Image Generation System]
**Purpose:**  
Implement a pre-generation approach for synthetic images where all training images are generated upfront and organized into train/val/test splits in `data/synthetic/` folder, replacing the previous caching system for better organization and efficiency.

**Implementation:**  
- **`generate_synthetic_images.py`**: Standalone script for pre-generating all synthetic images with organized directory structure, progress tracking, and comprehensive metadata
- **`SyntheticImageGenerator` Class**: Core generation functionality with font selection strategies, augmentation handling, and batch processing capabilities
- **`src/data/synthetic_dataset.py`**: PyTorch dataset implementation for loading pre-generated images with automatic metadata reading and fallback directory scanning
- **`SyntheticImageDataset` Class**: Efficient dataset loading with sample validation, statistics tracking, and error handling
- **Directory Organization**: Clean structure with separate images/ and labels/ folders per split, consistent naming convention, and comprehensive metadata files
- **Training Integration**: Updated production training pipeline to use synthetic datasets with automatic structure validation and optimized data loading
- **Testing Framework**: Complete test suite with `test_synthetic_dataset.py` for validation and debugging
- **Documentation**: Comprehensive documentation with usage examples, troubleshooting guides, and workflow recommendations

**History:**
- Created — Initial implementation replacing image caching system with pre-generation approach. Organized directory structure with train/val/test splits containing images/, labels/, and metadata.json. Implemented consistent font selection (hash-based for val/test, random for train), augmentation strategies, and comprehensive metadata tracking. Successfully tested with 150 sample images across all splits.
- Updated — Fixed critical font matching bug in `TextRenderer.render_text_pil()` method. The issue was that font paths like `fonts/KhmerOSmuollight.ttf` were not properly matched with loaded PIL font names. Added proper font name extraction and matching logic to ensure correct font selection for rendering.
- Enhanced — Integrated advanced Khmer text processing from imported `khtext` module. Added `khnormal_fast.py` for proper Khmer text normalization, fixing rendering issues with complex Khmer script. Updated `TextRenderer` to use corpus text from `data/processed/` files instead of random text generation. Integrated optional advanced background generation (gradient, noise, paper texture, patterns) while maintaining variable-width images as per project design. All generated images now use properly normalized Khmer text from the corpus with correct font rendering.

## Feature: [Khmer OCR Integrated Synthetic Generator]
**Purpose:**  
Integrate imported synthetic data generator modules from `src/synthetic_data_generator/` with the existing Khmer OCR project to generate variable-width images with 32px height using corpus data from `data/processed/` folder, providing advanced text rendering, backgrounds, and augmentation capabilities.

**Implementation:**  
- **`src/synthetic_data_generator/khmer_ocr_generator.py`**: Integrated generator class combining imported modules with project architecture, supporting variable-width image generation, corpus-based text, and advanced rendering
- **`generate_khmer_synthetic_images.py`**: New generation script using the integrated generator with comprehensive options for corpus-based image generation
- **`test_khmer_synthetic_generator.py`**: Validation script to test the integrated generator functionality

**History:**
- Created — Initial implementation integrating synthetic data generator modules with Khmer OCR project architecture, supporting variable-width images with 32px height
- Updated — Fixed text cutting issue by removing 1024px width limit in `_calculate_optimal_width()` method, allowing unlimited width to accommodate complete text sequences
- Updated — Fixed height consistency issue in rotated images by modifying `rotate_image()` method to use `expand=True` instead of `expand=False`, and added proper height validation in `_apply_augmentation()` method
- Updated — **ROTATION FIX**: Modified rotation augmentation to preserve content without cropping by using `expand=True` to allow canvas expansion, then shrinking the image to target height while maintaining aspect ratio. This prevents text content from being cut off during rotation augmentation. 

## Feature: [On-the-Fly Training System]
**Purpose:**  
Implement dynamic image generation during training to eliminate the need for pre-generating massive datasets, while maintaining a fixed validation set of exactly 6,400 images for consistent evaluation and reproducible results across training runs.

**Implementation:**  
- **`src/data/onthefly_dataset.py`**: Core `OnTheFlyDataset` class that generates synthetic images dynamically during training from corpus text files, providing unlimited augmentation variety and significant storage reduction
- **`OnTheFlyCollateFunction`**: Custom collate function for handling variable-width images in dynamic batches with proper padding and tensor formatting
- **`generate_fixed_validation_set.py`**: Script to generate exactly 6,400 fixed validation images that remain consistent across all training runs, ensuring reproducible evaluation metrics
- **`src/training/train_onthefly.py`**: New training script that uses on-the-fly generation for training data and pre-generated fixed validation set for evaluation
- **`test_onthefly_system.py`**: Comprehensive test suite validating on-the-fly dataset functionality, memory efficiency, reproducibility, and integration with training pipeline
- **`

## Feature: OnTheFlyDataset Integration with KhmerOCRSyntheticGenerator
**Purpose:**  
Replace TextRenderer with KhmerOCRSyntheticGenerator in OnTheFlyDataset to provide advanced Khmer text rendering, variable-width images, and improved background generation.

**Implementation:**  
- Modified `src/data/onthefly_dataset.py` to use KhmerOCRSyntheticGenerator instead of TextRenderer
- Replaced TextRenderer initialization with KhmerOCRSyntheticGenerator initialization
- Updated text rendering logic to use generator's methods:
  - `_select_font()` for font selection
  - `_calculate_optimal_width()` for dynamic width calculation
  - `_render_text_image()` for advanced text rendering
  - `_apply_augmentation()` for integrated augmentation
- Updated debug script to reflect the changes and show integration details
- Added comprehensive error handling and fallback mechanisms

**History:**
- Created — Integrated KhmerOCRSyntheticGenerator into OnTheFlyDataset, replacing TextRenderer for improved Khmer text rendering with variable-width images, advanced backgrounds, and better font handling. This provides more realistic training data with proper subscript support and dynamic image sizing.
- Updated — Fixed debug script attribute references to use correct `text_generator` property name, ensuring proper access to KhmerOCRSyntheticGenerator methods and properties.

## Feature: Font Size Consistency and Text Cropping Fix
**Purpose:**  
Fix inconsistent font size calculations across different text rendering methods and prevent text cropping in generated images.

**Implementation:**  
- Standardized font size calculation to 50% of image height (16px for 32px height) across all rendering methods
- Updated 4 locations in `src/data/text_renderer.py`:
  - `render_text_advanced_khmer()` method (line 359)
  - `_render_text_with_background_advanced()` method (line 481)
  - `_load_pil_fonts()` method (lines 151 and 177)
- Added explanatory comments for padding values (50px for basic rendering, 60px for background rendering)
- Fixed comment inconsistencies (updated "80% of height" to "50% of height")

**History:**
- Created — Fixed font size inconsistency where some methods used 80% of height while others used 50%, causing varying text sizes in generated images. Added proper padding comments to explain cropping prevention measures.

## Feature: Configuration System Verification and Fixes
**Purpose:**  
Complete verification and repair of the Khmer OCR configuration system to ensure proper training setup and resolve overfitting issues.

**Implementation:**  
Fixed truncated configuration files (config.yaml, model_config.yaml, vocab_config.yaml, vocabulary.yaml) that were incomplete. Verified complete 117-token Khmer vocabulary with proper special tokens (SOS, EOS, PAD, UNK). Optimized batch size from 64 to 32 for better convergence and reduced overfitting based on training analysis. Validated all training data (105MB corpus), fonts (8 TTF files), and validation set. Confirmed model architecture with 16.3M parameters and proper CRNN encoder functionality.

**History:**
- Created — Complete configuration system verification with fixes for truncated YAML files, vocabulary validation, and batch size optimization for improved training convergence.

## Feature: [Vocabulary Update to 132 tokens]
**Purpose:**  
Expanded Khmer vocabulary from 131 to 132 tokens to include additional symbol characters for complete text support.

**Implementation:**  
Updated KhmerVocab class in src/utils/config.py to include additional Khmer symbols. Updated all configuration files (config.yaml, vocab_config.yaml, vocabulary.yaml, model_config.yaml, train_config.yaml) to reflect the new vocabulary size of 132 tokens. The expansion includes additional punctuation and special characters found in Khmer text corpus.

**History:**
- Updated — Expanded vocabulary from 131 to 132 tokens by adding additional symbol characters. Updated all configuration files and model parameters to maintain consistency across the system.

## Feature: [Curriculum UNK Token Fix]
**Purpose:**  
Fixed excessive UNK tokens in curriculum training samples by replacing them with proper space characters and regenerating corresponding images to match labels.

**Implementation:**  
Identified that 11 out of 26 curriculum sample files had excessive UNK tokens (6-14 consecutive UNKs) representing word boundaries. Applied regex replacement of consecutive UNK tokens with single spaces. Regenerated all 26 sample images using KhmerTextRenderer with proper text content. All fixes validated with 100% perfect encoding/decoding success rate.

**History:**
- Fixed — Replaced UNK tokens with spaces in curriculum samples, regenerated images, validated encoding with 0 UNK tokens remaining and 100% success rate.

## Feature: Corpus Data Cleaning and UNK Token Resolution
**Purpose:**  
Remove all unsupported characters from processed corpus data to eliminate UNK token generation during training and curriculum learning. Ensures 100% vocabulary compliance across all 704,952 corpus lines.

**Implementation:**  
- Created `clean_processed_corpus.py` script to identify and clean unsupported characters
- Extracted 128 supported characters from KhmerVocab class (excluding SOS, EOS, PAD, UNK special tokens)
- Processed all files in `data/processed/` directory: train.txt, train_0.txt, train_1.txt, test.txt, val.txt
- Cleaned each line by keeping only vocabulary-supported characters, normalizing whitespace
- Created `replace_with_cleaned_corpus.py` script for safe backup and replacement
- Backed up original files to `data/processed_backup_[timestamp]` directory
- Replaced original corpus files with cleaned versions

**Results:**
- Cleaned 704,952 total lines across all corpus files
- Removed 1,376 unique unsupported characters, keeping only 132 vocabulary characters
- Achieved 84+ million clean characters total
- Reduced UNK tokens in curriculum datasets from dozens per sample to occasional occurrences
- Improved curriculum learning compliance rates (Phase 5: 33.3% → samples within length limits)
- All cleaned files verified to contain only vocabulary-supported characters

**History:**
- Created — Initial implementation with character filtering, corpus processing, and safe file replacement system
- Verified — Curriculum dataset testing shows dramatic UNK token reduction and proper length compliance

# Training and Development Change Log

## Feature: [Advanced Performance Training Script]
**Purpose:**  
Comprehensive training script with performance optimizations, Google Drive backup handling, and robust error management for both local and Colab environments.

**Implementation:**  
Created `train_advanced_performance.py` with advanced features: mixed precision training, curriculum learning, OneCycleLR scheduler, AdamW optimizer, dynamic teacher forcing, enhanced regularization, and intelligent Google Drive backup detection with fallback to local-only operation.

**History:**
- Created — Initial implementation with performance optimizations and argument parsing.
- Updated — Added robust Google Drive backup error handling with environment detection (Windows/Colab/Linux).
- Updated — Implemented safe fallback system for when Google Drive is not available or accessible.
- Updated — Added comprehensive logging with Unicode support and detailed validation result formatting.

## Feature: [Khmer OCR Synthetic Generator]
**Purpose:**  
Generates synthetic Khmer text images for training with various augmentations including morphological operations, Gaussian blur, background noise, and concatenation.

**Implementation:**  
Created KhmerOCRGenerator class in src/synthetic_data_generator/khmer_ocr_generator.py with comprehensive augmentation pipeline. Integrated with on-the-fly dataset generation for unlimited training variety.

**History:**
- Created — Initial implementation with morphological operations, Gaussian blur, noise generation, and concatenation augmentation. 
- Updated — Added advanced background generation and improved text rendering quality.
- Updated — Integrated with curriculum learning system for progressive difficulty.

## Feature: [On-the-Fly Training Dataset]
**Purpose:**  
Implements on-the-fly image generation during training to provide unlimited variety of training samples without requiring pre-generated datasets.

**Implementation:**  
Created OnTheFlyDataset class in src/data/onthefly_dataset.py that generates images in real-time during training. Supports text corpus loading, font management, and dynamic augmentation.

**History:**
- Created — Initial implementation with basic on-the-fly generation.
- Updated — Added support for multiple training files and improved memory efficiency.
- Updated — Integrated with advanced augmentation pipeline and curriculum learning.

## Feature: [Advanced Training Performance Optimizations]
**Purpose:**  
Comprehensive performance improvements to address slow convergence, overfitting, and suboptimal learning rates identified in training logs showing CER plateauing at 87-89%.

**Implementation:**  
1. Updated configs/train_config.yaml with optimized hyperparameters:
   - Increased learning rate from 1e-6 to 1e-4 (100x increase)
   - Added cosine annealing LR scheduler with warmup
   - Switched from Adam to AdamW optimizer with weight decay
   - Reduced teacher forcing from 1.0 to 0.7 with decay
   - Added dropout regularization (0.2-0.3) to all components
   - Enhanced data augmentation with rotation and scaling
   - Reduced gradient clipping from 5.0 to 1.0
   - Added early stopping with patience monitoring

2. Created train_advanced_performance.py implementing:
   - Mixed precision training (AMP) for faster convergence
   - Curriculum learning for progressive difficulty
   - Advanced LR scheduling (OneCycle, CosineAnnealing)
   - Dynamic teacher forcing decay
   - Enhanced performance tracking and logging

**History:**
- Created — Analyzed training logs showing severe overfitting (train loss decreasing while val loss increasing from 3.92→4.91) and extremely slow convergence due to 1e-6 learning rate.
- Implemented — Major configuration overhaul addressing learning rate, regularization, teacher forcing exposure bias, and augmentation improvements.
- Added — Advanced training script with mixed precision, curriculum learning, and sophisticated scheduling for production-ready performance optimization.

## Feature: Production Training Launcher
**Purpose:**  
Comprehensive production training launcher providing multiple training strategies with auto-configuration for optimal performance on RTX 3090 systems.

**Implementation:**  
Created `launch_production_training.py` with support for 4 training strategies (curriculum, on-the-fly, EOS-focused, pattern-fix), automatic system requirements checking, GPU-optimized batch size selection, and integrated checkpoint management. Includes `PRODUCTION_TRAINING_GUIDE.md` with complete setup and usage instructions.

**History:**
- Created — Production training launcher with auto-configuration for RTX 3090 (24GB VRAM), system requirements verification, multiple training strategy support, and comprehensive documentation. System verified ready for production with 313K corpus texts, 6,400 fixed validation images, and optimized training parameters.
- Enhanced — Extended launcher support for Google Colab platforms including A100 40GB/80GB (batch size 64/96), V100 (batch size 20), and T4 (batch size 12) with platform-specific optimizations, training time estimates, and Colab-specific setup recommendations.

## Feature: Complete Khmer Character Set and Space Handling Fixes
**Purpose:**  
Fix critical encoding issues where characters were being mapped to UNK tokens and `<SPACE>` tags were appearing in rendered images instead of actual spaces.

**Implementation:**  
1. **Vocabulary Expansion**: Updated KhmerVocab class from 117 to 131 tokens by adding missing characters:
   - Added 2 missing consonants: ឝ, ឞ (deprecated but still used)
   - Added 4 missing independent vowels: ឣ, ឤ, ឨ, ឲ  
   - Added 1 missing dependent vowel: ឳ
   - Added 6 missing symbols: ៖, ៘, ៙, ៚, ៛, ៜ, ៝
2. **Space Tag Processing**: Added restore_whitespace_tags() calls to all text rendering pipelines:
   - TextRenderer.render_text() and render_text_with_background()
   - KhmerTextRenderer.render_text()
   - KhmerOCRSyntheticGenerator._render_text_image()
   - CurriculumDataset image regeneration
3. **Configuration Updates**: Updated all config files (config.yaml, vocab_config.yaml, vocabulary.yaml, model_config.yaml) to reflect new vocabulary size of 131
4. **Comprehensive Testing**: Created test scripts to verify both fixes work correctly

**History:**
- Created — Identified two critical issues: UNK tokens for character 'ឲ' and `<SPACE>` tags appearing in images
- Analyzed — Failing examples showing character 'ឲ' (U+17B2) missing from vocabulary and unprocessed space tags
- **VOCABULARY FIX** — Added all 14 missing Khmer characters from khchar.py to vocabulary, expanding from 117 to 131 tokens
- **SPACE HANDLING FIX** — Added restore_whitespace_tags() calls to all text rendering pipelines to convert `<SPACE>` tags to actual spaces before rendering
- **CONFIGURATION FIX** — Updated all configuration files to reflect new vocabulary size
- **VERIFICATION** — Created comprehensive test confirming both fixes work correctly - no more UNK tokens or space tag display issues

## Feature: ConfigManager Dictionary Compatibility
**Purpose:**  
Fixed checkpoint loading error where ConfigManager object was saved in checkpoints but checkpoint loading code expected a dictionary with .get() method.

**Implementation:**  
- Added `.get(key, default)` method to ConfigManager class for backward compatibility
- Added `__getitem__`, `__contains__`, and `keys()` methods for full dictionary-like behavior
- Added `to_dict()` method to convert ConfigManager to dictionary format
- Updated curriculum training scripts to save config as dictionary using `config_manager.to_dict()`
- Maintains compatibility with both old checkpoints (ConfigManager objects) and new checkpoints (dictionaries)

**History:**
- Created — Fixed ConfigManager to support dictionary-like access for checkpoint loading compatibility. Updated curriculum training to save configs as dictionaries.