# Continual Learning Experiment Specification

**Stack:** Python, PyTorch, Local CUDA GPU\
**Execution Environment:** Fully local (no cloud dependencies)\
**Objective:** Iterative, cumulative continual learning on MNIST with
model re-instantiation between iterations.

------------------------------------------------------------------------

# 1. Experimental Overview

We will implement an iterative training loop over MNIST with the
following structure:

1.  Instantiate a model.
2.  Train on the first 100 images.
3.  Save the model chekcpoint
4.  Predict the 101st image.
5.  Re-instantiate the model from last checkpoint.
6.  Train on image 101, using the prediction as label.
7.  Predict image 102.
8.  Repeat for `n` iterations.

This simulates cumulative incremental learning with explicit
re-instantiation of model state.

------------------------------------------------------------------------

# 2. High-Level Algorithm

For iteration `i` in `[100, 100+n)`:

1.  Instantiate fresh model.
2.  Move model to GPU.
3.  Instantiate optimizer and loss function.
4.  Train on dataset slice `[i:i+1]`.
5.  Switch to evaluation mode.
6.  Predict on image `i`.
7.  Log:
    -   True label
    -   Predicted label
    -   Softmax confidence
    -   Loss on prediction sample
    -   Iteration index
8.  Continue loop.

------------------------------------------------------------------------

# 3. Technical Constraints

## 3.1 Determinism

-   Fix random seed (Python, NumPy, PyTorch).
-   Enable CUDA deterministic behavior.
-   Disable cuDNN benchmarking.

## 3.2 GPU

-   Use CUDA if available.
-   No CPU fallback unless explicitly requested.
-   Move tensors explicitly to device.

## 3.3 Data

-   Use MNIST training dataset only.
-   Fixed seed shuffle at load time.
-   Normalize using standard MNIST normalization.

------------------------------------------------------------------------

# 4. Model Specification

## Architecture (Minimal CNN)

-   Conv2d(1, 32, kernel_size=3)
-   ReLU
-   MaxPool2d(2)
-   Conv2d(32, 64, kernel_size=3)
-   ReLU
-   MaxPool2d(2)
-   Flatten
-   Linear(1600, 128)
-   ReLU
-   Linear(128, 10)

No dropout. No batch normalization.

------------------------------------------------------------------------

# 5. Training Protocol

-   Optimizer: SGD
-   Learning rate: 0.01
-   No momentum
-   No weight decay
-   One epoch per iteration over cumulative slice
-   Batch size: 32

------------------------------------------------------------------------

# 6. Logging Schema

Per iteration store:

``` json
{
  "iteration": int,
  "train_size": int,
  "true_label": int,
  "predicted_label": int,
  "confidence": float,
  "loss": float
}
```

Persist to: - JSONL file - Optional CSV mirror

------------------------------------------------------------------------

# 7. CLI Interface

Arguments:

-   `--n` : number of continual iterations
-   `--batch_size`
-   `--learning_rate`
-   `--seed`
-   `--device`

------------------------------------------------------------------------

# 8. File Structure

    continual_learning/
    ├── main.py
    ├── model.py
    ├── train.py
    ├── data.py
    ├── utils.py
    └── logs/

No unnecessary abstraction layers.

------------------------------------------------------------------------

# 9. Milestones (Recursive CoPilot Prompts)

## Milestone 1 -- Data Pipeline

-   Scaffold/teraform the project
-   Load MNIST
-   Deterministic shuffle
-   GPU-ready tensors
-   setup seed and command line interface

Acceptance: - Print first 5 labels reproducibly.

------------------------------------------------------------------------

## Milestone 2 -- Model Definition

-   Implement CNN in `model.py`

Acceptance: - Forward pass returns `[batch_size, 10]`.

------------------------------------------------------------------------

## Milestone 3 -- Single Iteration Training

-   Train on first 100 samples
-   Predict sample 101

Acceptance: - Print prediction and loss.

------------------------------------------------------------------------

## Milestone 4 -- Iterative Re-Instantiation Loop

-   Implement outer loop over `n`
-   Recreate model + optimizer each iteration

Acceptance: - Log file grows with `n` entries.

------------------------------------------------------------------------

## Milestone 5 -- Structured Logging

-   Write JSONL output
-   Verify schema consistency

Acceptance: - File parses without error.

------------------------------------------------------------------------

## Milestone 6 -- Determinism Verification

-   Re-run with same seed
-   Verify identical predictions

Acceptance: - Hash of JSONL file matches.


## Milestone 7 -- Generate a report

-   Plot relevant data points
-   Aggregate into a MARP markdown slide deck report

## Milestone 8 -- Generate a deck

-   Use the artifacts from Milestone 7 and put results in a MARP markdown slide deck
-   Explain the setup, procedure, and results

------------------------------------------------------------------------

# 10. Extension Hooks (Future Work)

-   Add replay buffer
-   Compare persistent vs re-instantiated models
-   Integrate IRT-style ability estimation over prediction stream
-   Track weight drift norms between iterations

------------------------------------------------------------------------

# 11. Termination Condition

Program exits after `n` iterations and flushes logs.

------------------------------------------------------------------------

# End of Specification
