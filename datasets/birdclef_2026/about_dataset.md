# BirdCLEF 2026 Dataset

## Dataset Overview

The BirdCLEF 2026 dataset contains field recordings of bird vocalizations collected from various ecosystems.

### Data Files

| File/Folder | Description |
|---|---|
| `train_audio/` | Labeled training recordings, organized by species (`train_audio/<species_code>/*.ogg`) |
| `train_soundscapes/` | Unlabeled soundscape recordings for self-supervised pre-training |
| `test_soundscapes/` | Unlabeled soundscape recordings used for evaluation |
| `train_metadata.csv` | Metadata for training audio (species, author, location, rating, etc.) |
| `sample_submission.csv` | Template showing the expected submission format |
| `taxonomy.csv` | Species taxonomy and common/scientific names |

### Audio Format

- Format: `.ogg` (Ogg Vorbis compressed audio)
- Sample rate: 32,000 Hz
- Soundscape recordings are split into 5-second chunks at inference time

### Labels

Each training clip is labeled with a `primary_label` (species code) and optionally `secondary_labels` for other species audible in the recording. The target is to predict the probability of each species being present in each 5-second window of a soundscape.

## How to Download

### Option 1 — Kaggle CLI (recommended)

```bash
# Install the Kaggle CLI
pip install kaggle

# Place your API token at ~/.kaggle/kaggle.json
# (Download from https://www.kaggle.com/settings > API > Create New Token)

# Download the competition data
kaggle competitions download -c birdclef-2026

# Unzip into this directory
unzip birdclef-2026.zip -d datasets/birdclef_2026/
```

### Option 2 — kagglehub (Python)

```python
import kagglehub

# Downloads and caches the competition files
path = kagglehub.competition_download('birdclef-2026')
print("Path to competition files:", path)
```

> Requires Kaggle account credentials. See [kagglehub docs](https://github.com/Kaggle/kagglehub) for setup.

### Option 3 — Manual Download

1. Go to [https://www.kaggle.com/competitions/birdclef-2026/data](https://www.kaggle.com/competitions/birdclef-2026/data)
2. Accept the competition rules
3. Click **Download All**

## Requirements

- A Kaggle account with competition rules accepted
- Sufficient disk space (~20–50 GB depending on the full audio set)
- Python packages: `kaggle` or `kagglehub`
