# Face Recognition System

A two-chapter machine learning project comparing classical and deep learning approaches to face recognition across two datasets — AT&T (controlled) and LFW (real-world).

## Contributors

| Student ID | Name | GitHub |
|---|---|---|
| 2221758042 | Muzahidur Rahman Saim | [@MuzahidurSaim](https://github.com/MuzahidurSaim) |
| 2231005042 | Ananya Sarkar | [@ananya0511sarkar-prog](https://github.com/ananya0511sarkar-prog) |
| 2231058642 | Nudrat Rahman Tushin | [@nudrat-nrt](https://github.com/nudrat-nrt) |
| 2311443042 | Safwan Ismaun Amin | [@aspiroo](https://github.com/aspiroo) |

---

## Project Overview

### Chapter 1 — AT&T Dataset (Controlled)
- 40 subjects, 10 images each (400 total)
- Controlled lighting, consistent pose, grayscale
- 80/20 train/test split
- Models: SVM, KNN, Random Forest

### Chapter 2 — LFW Dataset (Real-World)
- 19 subjects, 41–100 images each (1,227 total)
- Real-world variation in lighting, pose, expression
- 60/20/20 train/val/test split
- Models: SVM, KNN, Random Forest, CNN

### Results Summary

| Dataset | Model | Test Accuracy | CV Mean ± Std |
|---|---|---|---|
| AT&T | KNN | 98.75% | 82.24% ± 8.75% |
| AT&T | SVM | 96.25% | 97.50% ± 1.59% |
| AT&T | RF  | 92.50% | 94.06% ± 1.82% |
| LFW  | SVM | 55.28% | 57.29% ± 1.76% |
| LFW  | RF  | 40.24% | 41.08% ± 3.86% |
| LFW  | KNN | 39.02% | 37.78% ± 2.48% |
| LFW  | CNN |  8.13% | N/A |

The CNN failure on LFW is an intentional finding — CNNs require ~1,000+ images per class to train from scratch, compared to the 36–52 training images available per subject.

---

## Project Structure

```
Face-Recognition-System/
│
├── main.py                        # Terminal menu — run everything from here
├── compare_models.py              # (legacy) standalone compare script
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   ├── raw/
│   │   └── att/                   # AT&T dataset (s1/ to s40/)
│   └── processed/                 # Cached .npy arrays (auto-generated)
│
└── support/
    ├── compare_models.py          # Reads JSON results → comparison table + plots
    ├── comparison/                # Output folder for comparison plots
    │
    ├── att/                       # Chapter 1 — AT&T
    │   ├── load_normalized_data.py
    │   ├── preprocessing_att.py
    │   ├── model_svm_att.py
    │   ├── model_knn_att.py
    │   ├── model_rf_att.py
    │   ├── train_model.py         # Simple baseline SVM (Chapter 1 intro)
    │   ├── evaluate.py
    │   ├── recognize.py
    │   └── model/                 # Saved models + result JSONs
    │
    └── lfw/                       # Chapter 2 — LFW
        ├── load_data_lfw.py
        ├── preprocessing.py
        ├── model_svm.py
        ├── model_knn.py
        ├── model_rf.py
        ├── model_cnn.py
        └── model/                 # Saved models + result JSONs
```

---

## Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/aspiroo/Face-Recognition-System.git
cd Face-Recognition-System
```

### 2. Create and activate virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the AT&T dataset

Download the AT&T (ORL) Face Dataset and extract it to:
```
data/raw/att/
    s1/  1.pgm  2.pgm  ...  10.pgm
    s2/  1.pgm  2.pgm  ...  10.pgm
    ...
    s40/ 1.pgm  2.pgm  ...  10.pgm
```

The LFW dataset downloads automatically (~200MB) on first run via scikit-learn.

---

## Running the Project

### Main menu (recommended)

```bash
python main.py
```

This opens a terminal menu with all options:

```
  CHAPTER 1 — AT&T Dataset  (40 subjects, 80/20 split)
  ─────────────────────────────────────────────────────
  [1]  SVM   — train + evaluate (AT&T)
  [2]  KNN   — train + evaluate (AT&T)
  [3]  RF    — train + evaluate (AT&T)
  [4]  Run ALL Chapter 1 models

  CHAPTER 2 — LFW Dataset  (19 subjects, 60/20/20 split)
  ─────────────────────────────────────────────────────
  [5]  SVM   — train + evaluate (LFW)
  [6]  KNN   — train + evaluate (LFW)
  [7]  RF    — train + evaluate (LFW)
  [8]  CNN   — train + evaluate (LFW)
  [9]  Run ALL Chapter 2 models

  UTILITIES
  ─────────────────────────────────────────────────────
  [10] Recognize — predict a random face (AT&T SVM)
  [11] Evaluate  — evaluate saved AT&T SVM model
  [12] Compare   — compare all models from saved results
```

### Run individual models directly

```bash
# Chapter 1
python -m support.att.model_svm_att
python -m support.att.model_knn_att
python -m support.att.model_rf_att

# Chapter 2
python -m support.lfw.model_svm
python -m support.lfw.model_knn
python -m support.lfw.model_rf
python -m support.lfw.model_cnn
```

### Compare all models

Each model saves a JSON result file when it runs. After running all models, generate the comparison table and plots:

```bash
python -m support.compare_models
```

Output is saved to `support/comparison/`:
- `comparison_table.txt` — full metrics table
- `comparison_accuracy.png` — accuracy bar chart
- `comparison_metrics_att.png` — precision/recall/F1 (AT&T)
- `comparison_metrics_lfw.png` — precision/recall/F1 (LFW)
- `comparison_train_time.png` — training time comparison
- `comparison_cv_vs_test.png` — CV mean vs test accuracy

---

## Pipeline Overview

### AT&T Pipeline (Chapter 1)
```
load_normalized_data.py
    ↓ 400 images, 40 subjects, 10304 features
preprocessing_att.py
    ↓ 80/20 split → 320 train / 80 test
    ↓ PCA sweep (auto n_components via 5-fold CV)
    ↓ PCA fit on train only
model_svm/knn/rf_att.py
    ↓ Hyperparameter tuning via 5-fold CV
    ↓ Train final model on all 320 training images
    ↓ Evaluate on 80 test images (touched once)
    ↓ Save results to JSON
```

### LFW Pipeline (Chapter 2)
```
load_data_lfw.py
    ↓ 1,227 images, 19 subjects (min_faces=40, cap=100)
preprocessing.py
    ↓ 60/20/20 split → 736 train / 245 val / 246 test
    ↓ PCA sweep (auto n_components via val set)
    ↓ PCA fit on train only
model_svm/knn/rf.py
    ↓ Hyperparameter tuning on validation set
    ↓ 5-fold CV on train+val for stability
    ↓ Final model trained on train+val combined
    ↓ Evaluate on test set (touched once)
    ↓ Save results to JSON
model_cnn.py
    ↓ No PCA — raw 2D images
    ↓ Simple 3-block CNN with GlobalAveragePooling
    ↓ Early stopping on val_loss
    ↓ Save results to JSON
```

---

## Key Design Decisions

**No data leakage** — PCA is always fit on training data only and applied to val/test via `transform()`.

**Auto PCA selection** — all models sweep n_components and select the best automatically rather than using a hardcoded value.

**JSON result caching** — each model saves its metrics to a JSON file so `compare_models.py` can generate comparisons without re-running training.

**AT&T uses CV on training set instead of a validation set** — with only 8 training images per person, a three-way split would leave 1 image per person for validation, which is statistically meaningless. 5-fold CV on the training set is used instead.

**CNN failure is intentional** — the CNN result (8.13%) demonstrates the data requirements of deep learning. CNNs need ~1,000+ images per class trained from scratch; this dataset provides 40–100 training images per person.

---

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
opencv-python
joblib
tensorflow
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Dataset Information

### AT&T (ORL) Face Database
- 400 images, 40 subjects, 10 images each
- Resolution: 92×112 px, grayscale PGM
- Controlled lighting, slight pose variation
- Download: [AT&T Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

### Labeled Faces in the Wild (LFW)
- 13,233 images, 5,749 subjects (full dataset)
- Project subset: 1,227 images, 19 subjects (min_faces=40, cap=100)
- Resolution: native sklearn crop, resize=0.5
- Real-world variation: lighting, pose, expression
- Auto-downloaded via `sklearn.datasets.fetch_lfw_people`
- Citation: Huang et al., UMass Amherst Technical Report 07-49, 2007

<p align="right">(<a href="#top">back to top</a>)</p>
