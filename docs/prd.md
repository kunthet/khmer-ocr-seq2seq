
## **Product Requirements Document: Attention-Based Khmer OCR Engine**

**Version:** 1.0
**Date:** October 26, 2023
**Author:** AI Assistant (based on Buoy et al., 2022)

### 1. Introduction & Overview

**1.1. Problem Statement:**
Traditional Optical Character Recognition (OCR) for complex scripts like Khmer is challenging due to stacked consonants (Coengs), various diacritics, and the need for explicit character segmentation, which is often error-prone.

**1.2. Proposed Solution:**
This document outlines the requirements for implementing an end-to-end, attention-based Sequence-to-Sequence (Seq2Seq) model for printed Khmer text recognition. The system treats OCR as an image-to-text translation task, taking a raw text-line image as input and generating a sequence of characters as output, thereby eliminating the need for a separate character segmentation module.

**1.3. Target User:**
A developer or Machine Learning Engineer tasked with building a high-accuracy OCR system for printed Khmer text.

### 2. Goals & Objectives

* **Primary Goal:** To implement a deep learning model that can accurately transcribe images of printed Khmer text lines into editable text.
* **Performance Target:** Achieve a Character Error Rate (CER) of **≤ 1.0%** on a heavily augmented validation set, with a stretch goal of matching the paper's reported **0.7% CER**.

### 3. Functional Requirements

#### 3.1. Data Pipeline

**3.1.1. Synthetic Data Generation**
The model will be trained exclusively on synthetically generated data.

* **Text Corpus:** A large text corpus of Khmer words, phrases, and sentences must be compiled.
  * **Source:** To be sourced from public domain text (e.g., Khmer Wikipedia, news articles, public books).
  * **Size:** Target a corpus of ~3 million unique text lines.
* **Image Rendering:**
  * **Tool:** Use the `text2image` utility from the open-source Tesseract OCR project.
  * **Fonts:** Use a variety of common Khmer fonts to ensure font-invariance. **(Assumption:** The paper does not list fonts. The following are recommended: `Khmer OS Siemreap`, `Khmer OS Battambang`, `Khmer OS Muol Light`, `Hanuman`, `Nokora`).
  * **Image Properties:**
    * Color: Grayscale.
    * Height: **Fixed at 32 pixels.**
    * Width: **Variable**, dependent on the length of the rendered text.

**3.1.2. Data Augmentation**
Augmentation must be applied dynamically during training and once for the validation set. Each technique has a 50% probability of being applied to an image.

* **Gaussian Blurring:** Apply a blur with a variable sigma. **(Assumption:** Sigma range `(0, 1.5)`).
* **Morphological Operations:**
  * **Dilation:** Use a `2x2` or `3x3` kernel for 1 iteration.
  * **Erosion:** Use a `2x2` or `3x3` kernel for 1 iteration.
* **Noise Injection:**
  * **Blob Noise:** Add random black and white circles of small radii to the image.
  * **Background Noise:** Add fibrous and multi-scale background noise. (**Implementation Note:** This can be simulated using Perlin noise or by overlaying textures).
* **Image Concatenation:** With a 50% probability, concatenate two different augmented images horizontally to simulate more complex lines.

**3.1.3. Character Set (Vocabulary)**
The model will predict characters from a predefined vocabulary of 117 unique tokens.

* **Special Tokens (4):** `<SOS>` (Start), `<EOS>` (End), `<PAD>` (Padding), `<UNK>` (Unknown).
* **Numbers (20):** `០១២៣៤៥៦៧៨៩` and `0123456789`.
* **Consonants (33):** `កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ`.
* **Independent Vowels (12):** `ឥឦឧឩឪឫឬឭឮឯឰ`. (Note: `អ` is in consonants, `អា` is `អ`+`ា`).
* **Dependent Vowels (20):** `ា ិ ី ឹ ឺ ុ ូ ួ ើ ឿ ៀ េ ែ ៃ ោ ៅ ាំ ះ`.
* **Subscript Diacritic (1):** `្` (Coeng).
* **Other Diacritics (8):** `់ ៉ ៊ ័ ៏ ៌ ៍ ៎`.
* **Symbols (9):** `( ) , . ៕ ។ ៗ ?` and the `space` character.
  * **(Note:** This cleaned list totals 107 characters + 4 special tokens = 111. The remaining ~6 characters from the paper's 117 are likely due to typos or non-standard characters in their set. This list forms a robust starting point.)

---

#### 3.2. Model Architecture: Seq2Seq with Attention

The model consists of an Encoder to process the image and a Decoder with an attention mechanism to generate text.

**3.2.1. Encoder: Convolutional Recurrent Neural Network (CRNN)**


| Layer # | Layer Type               | Specification                                                                    | Output Shape (Example)   |
| --------- | -------------------------- | ---------------------------------------------------------------------------------- | -------------------------- |
| -       | Input                    | Grayscale Image                                                                  | `1 x 32 x W`             |
| 1       | Convolution 2D           | 64 filters, kernel=(3,3), stride=1, padding=1                                    | `64 x 32 x W`            |
| 2       | Max Pooling 2D           | kernel=(2,2), stride=2                                                           | `64 x 16 x W/2`          |
| 3       | Convolution 2D           | 128 filters, kernel=(3,3), stride=1, padding=1                                   | `128 x 16 x W/2`         |
| 4       | Max Pooling 2D           | kernel=(2,2), stride=2                                                           | `128 x 8 x W/4`          |
| 5       | Convolution 2D           | 256 filters, kernel=(3,3), stride=1, padding=1                                   | `256 x 8 x W/4`          |
| 6       | Convolution 2D           | 256 filters, kernel=(3,3), stride=1, padding=1                                   | `256 x 8 x W/4`          |
| 7       | Max Pooling 2D           | kernel=(2,1), stride=1                                                           | `256 x 8 x W/4`          |
| 8       | Convolution 2D           | 512 filters, kernel=(3,3), stride=1, padding=1                                   | `512 x 8 x W/4`          |
| 9       | Batch Normalization      | -                                                                                | `512 x 8 x W/4`          |
| 10      | Convolution 2D           | 512 filters, kernel=(3,3), stride=1, padding=1                                   | `512 x 8 x W/4`          |
| 11      | Batch Normalization      | -                                                                                | `512 x 8 x W/4`          |
| 12      | Max Pooling 2D           | kernel=(2,1), stride=1                                                           | `512 x 8 x W/4`          |
| 13      | Convolution 2D           | 512 filters, kernel=(3,3), stride=1, padding=0                                   | `512 x 6 x (W/4-2)`      |
| 14      | **Map-to-Sequence**      | Reshape & Permute features to create a sequence. Merge height & channel dims.    | `(W/4-2) x 3072`         |
| 15      | Bi-directional GRU 1     | hidden_size=256.`(256 * 2 = 512)` output features.                               | `(W/4-2) x 512`          |
| 16      | Bi-directional GRU 2     | hidden_size=256.`(256 * 2 = 512)` output features.                               | `(W/4-2) x 512`          |
| 17      | **Final Encoder Output** | Returns**all hidden states** from GRU 2 and the **final combined hidden state**. | Seq length`T`, `512` dim |

**3.2.2. Attention Mechanism**

* **Type:** Bahdanau-style Additive Attention.
* **Function:** At each decoding step `i`, it computes a context vector `c_i` as a weighted sum of all encoder hidden states `h_j`. The weights `a_ij` are calculated based on the previous decoder hidden state `s_{i-1}` and each encoder state `h_j`.

**3.2.3. Decoder: Recurrent Network**

* **Architecture:**
  1. **Embedding Layer:** Converts the previously predicted character token into a dense vector. **(Assumption:** Embedding dimension of 256).
  2. **GRU Layer:** A single-layer, uni-directional GRU with **512 hidden units**.
     * **Input at step `t`:** Concatenation of the embedded previous character `d(y_{t-1})` and the attention context vector `c_{t-1}`.
     * **Initial State:** The final hidden state from the encoder, passed through a linear layer `Linear(512, 512)` with a `Tanh` activation.
  3. **Classifier:** A single Linear layer that takes the concatenation of the current GRU output `s_t`, the current context vector `c_t`, and the embedded previous character `d(y_{t-1})` to predict the next character.
     * **Output Dimension:** 117 (the size of the vocabulary).
     * **Activation:** Followed by a LogSoftmax function.

---

#### 3.3. Training Pipeline

* **Framework:** PyTorch
* **Hardware:** 1x GPU with ≥ 16GB VRAM (e.g., Tesla P100, V100, RTX 3090).
* **Optimizer:** Adam
* **Learning Rate:** `1e-6`
* **Loss Function:** Cross-Entropy Loss (`torch.nn.NLLLoss` with `LogSoftmax` output).
* **Batch Size:** 64 images.
* **Epochs:** 150.
* **Teacher Forcing Ratio:** 1.0 (The ground-truth character is always used as the input for the next decoding step during training).
* **Checkpointing:** The model state must be saved periodically to handle long training times and potential interruptions.

#### 3.4. Inference Pipeline

* **Input:** A single grayscale image of a text line, resized to 32px height.
* **Process:**
  1. The image is passed through the encoder to get the sequence of hidden states and the initial decoder state.
  2. The decoder starts with the `<SOS>` token.
  3. A greedy decoding loop begins:
     a. The decoder predicts the next character based on its current state and the attention context.
     b. The predicted character becomes the input for the next step.
  4. The loop terminates when the `<EOS>` token is predicted or a maximum sequence length is reached (e.g., 256 characters).
* **Output:** The sequence of predicted characters, joined into a string.

### 4. Out of Scope

* Handwritten text recognition.
* Document layout analysis (detecting text lines in a full page).
* Recognition of text in natural scenes (scene text).
* Training on real (non-synthetic) scanned document images.
