# Automated Wildlife Species Recognition Using Deep Learning on Bioacoustic Data

**Course:** Deep Learning — MSc in AI, Semester 2  
**Dataset:** [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026/data) (Kaggle)

---

## Abstract

This project addresses automatic bird species classification from raw audio recordings using the BirdCLEF 2026 competition dataset. To obtain a balanced training set, the 30 most frequently recorded species are selected, capped at 405 recordings each (the size of the smallest class), yielding 12,150 recordings in total. Each audio clip is resampled at 32 kHz and converted to a log-scale Mel spectrogram (n_fft=2048, hop_length=512, 128 Mel bins); only the first 15 seconds are used, split into three non-overlapping 5-second windows, producing a per-recording tensor of shape `[3, 1, 128, 313]`. Two custom architectures are evaluated: a pure CNN with Max Pooling over the three time windows, and a hybrid CNN-BiLSTM that treats the window sequence as a temporal input. Both are compared against a baseline of pretrained Perch bioacoustic embeddings combined with Logistic Regression. On the clean test set, Perch+LR achieves ~0.92 accuracy, the CNN ~0.79, and the CNN-BiLSTM ~0.66; a follow-up noise-robustness experiment (medium and heavy natural soundscape mixing) confirms that Perch embeddings degrade far less than the end-to-end CNN models.

---

## Repository Structure

```
mscai-dl-project/
│
├── datasets/
│   └── birdclef_2026/
│       └── about_dataset.md     # Dataset overview and download instructions
│
├── exams_2026/
│   ├── report.pdf               # Project report
│   └── presentation.pdf         # Project presentation slides
│
├── notebooks/
│   ├── 01_preprocessing.ipynb   # Species selection, mel spectrogram generation,
│   │                            #   train/val/test split, .pt file export
│   ├── 02_train_eval.rar        # Model definitions (CNN, CNN-BiLSTM, Perch+LR),
│   │                            #   training loops, and clean-data evaluation (compressed)
│   └── 03_eval_noise_data.ipynb # Noise-robustness evaluation: soundscape mixing
│                                #   at multiple SNR levels, CNN vs Perch comparison
│
├── .gitignore
└── LICENSE                      # MIT
```

> **Note:** `02_train_eval.rar` must be extracted before use. Extract it in the `notebooks/` directory to obtain the training/evaluation notebook.

---

## Setup and Usage

### Prerequisites

- A Google account with the BirdCLEF 2026 dataset already downloaded and placed in Google Drive (see [`datasets/birdclef_2026/README.md`](datasets/birdclef_2026/about_dataset.md) for download instructions).
- All notebooks are designed to run on **Google Colab**.

### 1. Open in Google Colab

Upload or clone the repository, then open each notebook directly in Colab. Enable a GPU runtime before running:

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4 or better)
```

### 2. Mount Google Drive

Each notebook begins with a Drive mount cell. Run it and authenticate when prompted:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then update the dataset path variable near the top of the notebook to match your own Drive folder layout, e.g.:

```python
DATA_ROOT = '/content/drive/MyDrive/birdclef-2026/'
```

### 3. Dependencies

The following non-standard packages are required. Standard Colab already ships with `torch`, `torchaudio`, `scikit-learn`, `pandas`, `numpy`, and `matplotlib`. Install the remaining ones at the top of any notebook that uses them:

```bash
pip install librosa soundfile
```

Notebook `02_train_eval` (inside the `.rar`) additionally uses the **Perch** bioacoustic embedding model. Follow any `pip install` cells present in that notebook for TensorFlow / TF Hub setup.

Full dependency list:

| Package | Purpose |
|---|---|
| `torch` / `torchaudio` | Model training and audio I/O |
| `librosa` | Audio loading and spectrogram utilities |
| `soundfile` | OGG audio decoding |
| `scikit-learn` | Data splitting, Logistic Regression, evaluation metrics |
| `joblib` | Saving / loading trained sklearn models |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` | Visualization |
| `google.colab` | Drive mounting and Colab utilities |

### 4. Run Order

Run notebooks in this order:

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01_preprocessing.ipynb` | Reads raw `.ogg` files from Drive, selects 30 species, generates Mel spectrograms, performs train/val/test split, and exports `.pt` tensor files back to Drive |
| 2 | `02_train_eval` *(extracted from `.rar`)* | Defines CNN and CNN-BiLSTM architectures and the Perch+LR baseline; trains each model on the preprocessed tensors; evaluates on the clean test split |
| 3 | `03_eval_noise_data.ipynb` | Loads the saved models, synthesises noisy test audio by mixing with unlabelled soundscapes at multiple SNR levels, and reports accuracy / F1 / top-k metrics for each model and noise level |

---

## Results Summary

### Clean Test Set

| Model | Accuracy |
|---|---|
| Perch + Logistic Regression | ~0.92 |
| CNN (Max Pooling) | ~0.79 |
| CNN-BiLSTM | ~0.66 |

### Noise Robustness (100 test samples) - Accuracy

| Model | Medium noise | Heavy noise |
|---|---|---|
| CNN (Max Pooling) | 0.54 | 0.46 |
| Perch + LR | 0.76 | 0.73 |

---

## Authors

**Elpida Gkouvra** (mtn2504) · **Stella Chantava** (mtn2528)
