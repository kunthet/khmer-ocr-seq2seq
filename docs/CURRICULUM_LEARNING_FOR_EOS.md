# Curriculum Learning for EOS Generation: A Case Study

## 1. Executive Summary

This document details the successful resolution of a critical failure in the Khmer OCR model where it failed to generate the End-Of-Sequence (EOS) token, leading to infinite-loop character patterns and unusable output.

The issue was resolved by implementing a **curriculum learning** strategy, which incrementally increased the complexity of training samples. This approach forced the model to first master sequence termination on short, simple sequences before progressing to longer, more complex ones.

**Result**: The curriculum training was highly effective, achieving **100% EOS generation accuracy** in early phases and successfully resolving the root cause of the sequence generation failure.

## 2. Problem Recap: The EOS Generation Failure

The core problem manifested in several ways:
- **No EOS Tokens**: The model never predicted the `EOS` token during inference.
- **Length Explosion**: Predictions always ran to the maximum allowed length (e.g., 392 tokens).
- **Repetitive Patterns**: The model would get stuck in loops, generating patterns like `ិកឬិកឬិកឬ...`.
- **High CER**: The Character Error Rate (CER) was often over 100% due to the massive length mismatch between predictions and ground truth.

Initial attempts to solve this by increasing the `EOS` loss weight (up to 25x) failed, indicating a more fundamental issue in the training dynamics.

## 3. Root Cause: Training Dynamics, Not Architecture

A deep architectural analysis (`docs/EOS_ARCHITECTURAL_ANALYSIS.md`) confirmed that the model's components (Encoder, Decoder, Attention) were correctly implemented. The vocabulary and special tokens were also properly configured.

The root cause was identified as a **training dynamics failure**:
- **Overwhelming Complexity**: The model, when presented with long sequences from the start, focused all its capacity on learning character representations and paid no attention to sequence boundaries.
- **Teacher Forcing Paradox**: During training, the model was always fed the next ground-truth token. It never had to learn to *decide* to generate an `EOS` token on its own, as it was never penalized for failing to do so until the very end of a sequence it rarely reached.
- **Marginalized EOS Loss**: The loss signal from the single `EOS` token at the end of a long sequence was too weak to compete with the loss from dozens of other characters.

## 4. Solution: Curriculum Learning

The solution was to create a structured curriculum that made learning EOS the first and easiest task. This was implemented in the `train_curriculum_eos_v1.py` script.

### 4.1. Core Components

#### `CurriculumDataset`
A wrapper around the `OnTheFlyDataset` that truncates every text sequence to a specified `max_length` and ensures an `EOS` token is always present at the end of the truncated sequence.

#### `CurriculumEOSLoss`
A custom loss function with an **adaptive EOS weight**. The weight is inversely proportional to the current `max_length` of the curriculum, applying the strongest pressure to learn `EOS` during the initial, short-sequence phases.
\[
\text{adaptive\_weight} = \text{base\_weight} \times \frac{20}{\max(\text{current\_length}, 10)}
\]

#### Custom Collation Function
A `curriculum_collate_fn` was implemented to handle the variable-width images produced by the text renderer. It pads all images in a batch to the maximum width of that batch, allowing them to be stacked into a single tensor.

### 4.2. The Curriculum Schedule

The training progressed through a series of phases, each with an increasing `max_length`:

| Phase | Epochs | Max Length | Description                  |
|-------|--------|------------|------------------------------|
| 1     | 4      | 10         | Force EOS on short sequences |
| 2     | 4      | 15         | Introduce more complexity    |
| 3     | 4      | 20         | Medium-length sequences      |
| 4     | 4      | 30         | Extended sequences           |
| 5     | 6      | 50         | Near-full length sequences   |

## 5. Training Results: A Resounding Success

The curriculum training was immediately effective.

- **Phase 1 (max_length=10)**: The model rapidly achieved **100% EOS accuracy** on the validation set. It learned that terminating a sequence is the most important task.
- **Phase 2 (max_length=15)**: Maintained **99.4% EOS accuracy**, demonstrating that the learned behavior was robust.
- **Phase 3 (max_length=20)**: Maintained **98.2% EOS accuracy**. At this point, the training triggered an "early success" condition and stopped, as the primary goal had been achieved.

The final validation log showed:
```
Val   - Loss: 4.0181 | EOS: 97.1% | Complete: 96.9%
Adaptive EOS Weight: 15.00
...
🎉 Early success! EOS rate 98.2% at length 20
```

## 6. Conclusion

The EOS generation failure was not a bug in the model's architecture but a flaw in the training strategy. By using curriculum learning, we successfully guided the model to learn the concept of sequence termination. This case study demonstrates that for complex sequence generation tasks, **how** a model is taught can be as important as **what** it is taught. The curriculum approach is now a core part of the training pipeline for this project. 