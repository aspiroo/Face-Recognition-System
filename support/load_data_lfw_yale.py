# load_data.py
# Loads and merges Yale Face B + LFW datasets.
# AT&T is intentionally excluded from Chapter 2 — low image count per person.
# AT&T is still used in Chapter 1 (train_model.py) as the baseline.
#
# Sources:
#   Yale Face B  — downloaded manually to data/raw/yale/
#   LFW          — loaded via sklearn built-in (auto-downloads, no folder needed)
#
# All images are resized to 128x128, grayscale, normalized 0–1, flattened.
# Caches processed arrays to data/processed/ for fast reloading.
#
# ── Dataset combinations available ───────────────────────────────────────────
#   load_dataset("yale")           → Yale only
#   load_dataset("lfw")            → LFW only
#   load_dataset("yale_lfw")       → Yale + LFW merged  ← default for Chapter 2
#
# Run from repo root: python -m support.load_data
# -------------------------------------------------------

import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import LabelEncoder

# ── Standard size all images are resized to ───────────────────────────────────
# 128x128: compatible with Yale (192x168) and LFW (250x250) without heavy distortion
# Also used by preprocessing.py and model_cnn.py — do not change without updating those
IMG_WIDTH  = 128
IMG_HEIGHT = 128

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YALE_DIR  = os.path.join(BASE_DIR, "data", "raw", "yale")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")

# ── Yale balancing cap ────────────────────────────────────────────────────────
# Yale has ~64 images per person. Cap at 20 to stay balanced with LFW filtered subset.
YALE_MAX_PER_PERSON = 20

# ── LFW filter ────────────────────────────────────────────────────────────────
# Only keep people with at least this many images.
# 20 gives ~62 people — good balance of subjects and images per person.
LFW_MIN_FACES = 20

# ── LFW balancing cap ─────────────────────────────────────────────────────────
# LFW people with 20+ images can have up to 500+ each (very unbalanced).
# Cap at 20 to match Yale and prevent dominant subjects skewing the model.
LFW_MAX_PER_PERSON = 20


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read_image(img_path):
    """
    Read one image file → grayscale → resize to 128x128 → normalize → flatten.
    Returns np.ndarray of shape (16384,) or None if file is unreadable.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"  Warning: could not read {img_path}, skipping.")
        return None

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.flatten().astype(np.float32) / 255.0
    return img


def _load_yale(start_label=0):
    """
    Load Yale Face B from data/raw/yale/.
    Expects subfolders yaleB01/, yaleB02/, ... containing .pgm files.
    Labels start from start_label and increment per subject.

    Returns (images, labels, next_available_label)
    """
    images, labels = [], []

    if not os.path.exists(YALE_DIR):
        print(f"  Warning: Yale not found at {YALE_DIR} — skipping.")
        print(f"  Download from Kaggle and extract to data/raw/yale/")
        return images, labels, start_label

    subject_folders = sorted([
        d for d in os.listdir(YALE_DIR)
        if os.path.isdir(os.path.join(YALE_DIR, d))
    ])

    if not subject_folders:
        print(f"  Warning: No subfolders found in {YALE_DIR}")
        return images, labels, start_label

    print(f"  Yale Face B : {len(subject_folders)} subjects found")

    label_idx = start_label
    for folder in subject_folders:
        folder_path = os.path.join(YALE_DIR, folder)
        img_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".pgm", ".jpg", ".png"))
        ])

        # Cap per person to keep balance across datasets
        img_files = img_files[:YALE_MAX_PER_PERSON]

        loaded = 0
        for fname in img_files:
            img = _read_image(os.path.join(folder_path, fname))
            if img is not None:
                images.append(img)
                labels.append(label_idx)
                loaded += 1

        if loaded > 0:
            label_idx += 1   # only increment if subject had valid images

    print(f"  Yale Face B : {len(images)} images loaded ({label_idx - start_label} subjects)")
    return images, labels, label_idx


def _load_lfw(start_label=0):
    """
    Load LFW via sklearn built-in.
    Automatically downloads on first run (~200MB) to ~/scikit_learn_data/.
    Filters to people with at least LFW_MIN_FACES images.
    Resizes to 128x128 to match Yale.

    Returns (images, labels, next_available_label)
    """
    print(f"  LFW         : fetching via sklearn (min_faces={LFW_MIN_FACES})...")
    print(f"                First run downloads ~200MB — subsequent runs load from cache.")

    lfw = fetch_lfw_people(
        min_faces_per_person=LFW_MIN_FACES,
        resize=None,     # we handle resizing ourselves for consistency
        color=False,     # grayscale
        slice_=None      # full image, we resize ourselves
    )

    # lfw.images shape: (n_samples, h, w) — grayscale float32 0–255
    # lfw.target: integer labels 0..n_classes
    raw_images = lfw.images
    raw_labels = lfw.target

    print(f"  LFW         : {raw_images.shape[0]} images, {len(np.unique(raw_labels))} people raw")

    images, labels  = [], []
    lfw_label_map   = {}   # maps original LFW label → new sequential label
    per_person_count = {}  # tracks how many images loaded per person for cap

    for i in range(len(raw_images)):
        orig_label = raw_labels[i]

        # Assign new sequential label continuing from start_label
        if orig_label not in lfw_label_map:
            lfw_label_map[orig_label]    = start_label + len(lfw_label_map)
            per_person_count[orig_label] = 0

        # Skip if this person already hit the cap
        if per_person_count[orig_label] >= LFW_MAX_PER_PERSON:
            continue

        new_label = lfw_label_map[orig_label]

        # Resize to standard 128x128
        img = raw_images[i].astype(np.uint8)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.flatten().astype(np.float32) / 255.0

        images.append(img)
        labels.append(new_label)
        per_person_count[orig_label] += 1

    next_label = start_label + len(lfw_label_map)
    print(f"  LFW         : {len(images)} images loaded ({len(lfw_label_map)} subjects, capped at {LFW_MAX_PER_PERSON}/person)")
    return images, labels, next_label


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(mode="yale_lfw", force_reload=False):
    """
    Main entry point for Chapter 2 models (SVM, KNN, RF, CNN).

    Args:
        mode         : which datasets to include
                       "yale"      → Yale Face B only
                       "lfw"       → LFW only
                       "yale_lfw"  → Yale + LFW merged  (default, recommended)
        force_reload : ignore cache and re-read from scratch (default False)

    Returns:
        X : np.ndarray (n_samples, 16384)  — flattened normalized grayscale images
        y : np.ndarray (n_samples,)        — integer labels, contiguous from 0

    Usage:
        from support.load_data import load_dataset
        X, y = load_dataset()                    # Yale + LFW (default)
        X, y = load_dataset("yale")              # Yale only
        X, y = load_dataset(force_reload=True)   # ignore cache, rebuild
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    X_path = os.path.join(PROC_DIR, f"X_{mode}.npy")
    y_path = os.path.join(PROC_DIR, f"y_{mode}.npy")

    # ── Return cached version if available ───────────────────────────────────
    if not force_reload and os.path.exists(X_path) and os.path.exists(y_path):
        print(f"Loading cached dataset ({mode}) from data/processed/...")
        X = np.load(X_path)
        y = np.load(y_path)
        _print_summary(X, y, mode)
        return X, y

    # ── Build from scratch ────────────────────────────────────────────────────
    print(f"\nBuilding dataset: {mode}")
    print("-" * 40)

    all_images, all_labels = [], []
    next_label = 0

    if mode in ("yale", "yale_lfw"):
        yale_imgs, yale_lbls, next_label = _load_yale(start_label=next_label)
        all_images.extend(yale_imgs)
        all_labels.extend(yale_lbls)

    if mode in ("lfw", "yale_lfw"):
        lfw_imgs, lfw_lbls, next_label = _load_lfw(start_label=next_label)
        all_images.extend(lfw_imgs)
        all_labels.extend(lfw_lbls)

    if not all_images:
        raise RuntimeError(
            "No images loaded. Check that Yale is in data/raw/yale/ "
            "and that LFW downloaded successfully."
        )

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    # ── Ensure labels are contiguous 0-indexed (required for CNN output layer) ─
    unique_labels = np.unique(y)
    if len(unique_labels) != unique_labels.max() + 1:
        le = LabelEncoder()
        y  = le.fit_transform(y).astype(np.int32)

    # ── Cache ─────────────────────────────────────────────────────────────────
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"Cached to data/processed/")

    _print_summary(X, y, mode)
    return X, y


def _print_summary(X, y, mode):
    """Print a clean dataset summary."""
    n_subjects = len(np.unique(y))
    n_total    = X.shape[0]
    avg        = n_total / n_subjects if n_subjects > 0 else 0

    print(f"\n{'─'*40}")
    print(f"  Dataset      : {mode}")
    print(f"  Total images : {n_total}")
    print(f"  Subjects     : {n_subjects}")
    print(f"  Avg per subj : {avg:.1f}")
    print(f"  Feature size : {X.shape[1]}  ({IMG_WIDTH}x{IMG_HEIGHT} flattened)")
    print(f"{'─'*40}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Yale only")
    print("=" * 55)
    X, y = load_dataset("yale")

    print("=" * 55)
    print("  LFW only")
    print("=" * 55)
    X, y = load_dataset("lfw")

    print("=" * 55)
    print("  Yale + LFW (default)")
    print("=" * 55)
    X, y = load_dataset("yale_lfw")