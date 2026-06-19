# BirdCLEF 2026 – Bird Species Classification Using CNN and Mel Spectrograms

## Project Overview

This notebook implements a deep learning pipeline for bird species classification using audio recordings from the BirdCLEF 2026 dataset (https://www.kaggle.com/competitions/birdclef-2026).

The approach converts raw bird audio recordings into Mel Spectrograms and uses a Convolutional Neural Network (CNN) with max pooling across temporal windows to classify recordings into one of 30 bird species.

---

## Dataset Preparation

The BirdCLEF 2026 training metadata (`train.csv`) is used to identify available recordings and species labels.

### Species Selection

To maintain class balance and manageable training time:

* Top 30 most represented bird species are selected.
* 405 recordings are sampled from each species.
* Total recordings used: 12,150.

### Train / Validation / Test Split

The balanced dataset is split using stratified sampling:

| Split      | Percentage | Recordings |
| ---------- | ---------- | ---------- |
| Train      | 70%        | 8,505      |
| Validation | 15%        | 1,822      |
| Test       | 15%        | 1,823      |

The validation set is used for model selection and early stopping, while the test set is reserved for final performance evaluation.

---

## Audio Preprocessing

Each audio recording undergoes the following preprocessing steps:

1. Load audio waveform.
2. Convert to mono if necessary.
3. Resample to 32 kHz.
4. Pad or truncate to 60 seconds.
5. Split into twelve 5-second windows.
6. Convert each window into a Mel Spectrogram.
7. Convert amplitudes to decibel scale.

### Mel Spectrogram Parameters

| Parameter         | Value     |
| ----------------- | --------- |
| Sample Rate       | 32,000 Hz |
| Duration          | 60 sec    |
| Window Length     | 5 sec     |
| Number of Windows | 12        |
| Mel Bins          | 128       |
| FFT Size          | 2048      |
| Hop Length        | 512       |

---

## Mel Spectrogram Storage

To avoid recomputing spectrograms during every training run, each spectrogram window is stored as a PyTorch `.pt` file.

Each file contains:

* Mel Spectrogram tensor
* Species label ID
* Species name
* Source audio path
* Window index

Example:

```python
{
    "mel": tensor(...),
    "label": 7,
    "primary_label": "amecro",
    "source_file": "...",
    "window_idx": 0
}
```

Approximately:

```text
12,150 recordings × 12 windows
= 145,800 .pt files
```

---

## Model Architecture

### CNN Encoder

Each spectrogram window is treated as a grayscale image.

The encoder consists of:

* Convolutional Layers
* Batch Normalization
* ReLU Activation
* Max Pooling
* Adaptive Average Pooling
* Dense Projection Layer

Output embedding size:

```text
512 dimensions
```

---

### Recording-Level Aggregation

Each recording contains 12 spectrogram windows.

The workflow is:

```text
Recording
    ↓
12 Mel Spectrogram Windows
    ↓
CNN Encoder
    ↓
12 Embeddings
    ↓
Temporal Max Pooling
    ↓
Recording Embedding
    ↓
Classification Layer
    ↓
Species Prediction
```

The maximum activation across windows is retained, allowing the model to focus on the most informative bird vocalizations within a recording.

---

## Training Configuration

| Parameter               | Value            |
| ----------------------- | ---------------- |
| Optimizer               | AdamW            |
| Learning Rate           | 3e-4             |
| Weight Decay            | 1e-4             |
| Batch Size              | 64               |
| Maximum Epochs          | 100              |
| Early Stopping Patience | 10               |
| Loss Function           | CrossEntropyLoss |

---

## Evaluation Metrics

Performance is monitored on the validation set during training and evaluated on the held-out test set after training.

Metrics include:

* Accuracy
* Macro F1 Score
* Weighted F1 Score
* Macro ROC-AUC
* Top-1 Accuracy
* Top-3 Accuracy
* Top-5 Accuracy

---

## Model Selection

The model with the lowest validation loss is automatically saved during training.

Training stops automatically if validation loss does not improve for 10 consecutive epochs.

---

## Final Workflow

```text
BirdCLEF Audio Recordings
            ↓
Species Selection (Top 30)
            ↓
Balanced Sampling (405 / Species)
            ↓
Train / Validation / Test Split
            ↓
Audio Preprocessing
            ↓
Mel Spectrogram Generation
            ↓
Storage as .pt Files
            ↓
CNN Encoder
            ↓
Max Pooling Across Windows
            ↓
Species Classification
            ↓
Performance Evaluation
```

---

## Expected Outputs

The notebook produces:

* Saved Mel Spectrogram tensors (`.pt`)
* Trained CNN-Max model
* Best model checkpoint
* Training history
* Validation metrics
* Final test metrics
* Classification report
* Spectrogram visualizations

This implementation serves as a complete end-to-end pipeline for bird species classification from audio recordings using deep learning and Mel Spectrogram representations.
