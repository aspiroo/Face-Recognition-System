# load_data_lfw.py
# LFW-only dataset loader for Chapter 2 (final configuration).
#
# Why LFW-only:
#   The merged Yale+LFW (90 classes, 20 imgs/person) caused data starvation
#   across all models (SVM 27%, KNN 24%, CNN ~0%). Root causes were:
#     1. Only 12 training images per person after 60/20/20 split
#     2. Domain mismatch — Yale (controlled lab lighting) vs LFW (real-world)
#     3. 90-class softmax with <15 training examples per class is unsolvable
#
#   LFW-only with min_faces=70 gives ~20 subjects with 70-400+ images each.
#   After the 60/20/20 split that is 42+ training images per person minimum —
#   enough for SVM/KNN to generalise and for CNN augmentation to be effective.
#
# Dataset stats (min_faces=70, no per-person cap):
#   ~20 subjects | ~1140 total images | ~57 avg per subject
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   from support.load_data_lfw import load_dataset
#   X, y = load_dataset()               # default, recommended
#   X, y = load_dataset(force_reload=True)  # bust cache after code changes
#
# Run standalone: python -m support.load_data_lfw
# ─────────────────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import LabelEncoder

# ── Image dimensions ──────────────────────────────────────────────────────────
# Must match preprocessing.py, model_cnn.py
IMG_WIDTH  = 128
IMG_HEIGHT = 128

# ── LFW filter — only keep people with at least this many images ──────────────
# 70 gives ~20 people.
#   <50  → too many low-data subjects, starvation returns
#   >100 → very few subjects (~10), trivial classification problem
LFW_MIN_FACES = 40

# ── Per-person cap — set to None to use all available images ──────────────────
# Unlike the mixed dataset, we do NOT cap here. More data per person = better.
# If one subject dominates (e.g. 400+ images) the stratified split in
# preprocessing.py still keeps train/val/test proportional, so imbalance
# does not leak into evaluation.
LFW_MAX_PER_PERSON = 100   # None = no cap, use everything

# ── Cache paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_lfw():
    """
    Fetch LFW via sklearn, resize to 128x128, normalize to 0-1, flatten.

    sklearn's fetch_lfw_people returns float32 images in the 0-255 range.
    We keep them as float32 throughout and divide by 255 at the end —
    never cast to uint8 first, which would truncate decimal precision.

    Returns:
        images : list of np.ndarray (16384,)  float32 0-1
        labels : list of int  (contiguous from 0)
    """
    print(f"  Fetching LFW (min_faces_per_person={LFW_MIN_FACES})...")
    print(f"  First run downloads ~200MB to ~/scikit_learn_data/ — cached after that.")

    lfw = fetch_lfw_people(
        min_faces_per_person=LFW_MIN_FACES,
        resize=None,     # we resize ourselves for consistency with other loaders
        color=False,     # grayscale
        slice_=None      # full image, no pre-crop
    )

    raw_images = lfw.images    # (n_samples, h, w)  float32  0-255
    raw_labels = lfw.target    # (n_samples,)  int
    names      = lfw.target_names  # string name per class index

    n_raw      = raw_images.shape[0]
    n_subjects = len(np.unique(raw_labels))
    print(f"  Raw fetch   : {n_raw} images across {n_subjects} subjects")

    # ── Remap original LFW labels to contiguous 0-indexed integers ────────────
    # fetch_lfw_people labels are already 0-indexed but we remap anyway for
    # safety in case sklearn changes this behaviour across versions.
    label_map        = {}   # original label → new label
    per_person_count = {}   # original label → images loaded so far
    images, labels   = [], []

    for i in range(n_raw):
        orig = raw_labels[i]

        if orig not in label_map:
            label_map[orig]        = len(label_map)
            per_person_count[orig] = 0

        # Apply cap if set
        if LFW_MAX_PER_PERSON is not None and per_person_count[orig] >= LFW_MAX_PER_PERSON:
            continue

        # Resize and normalize — keep float32, never cast to uint8
        img = raw_images[i].astype(np.float32)          # already 0-255 float
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))   # → 128x128
        img = img.flatten() / 255.0                      # → (16384,) in 0-1

        images.append(img)
        labels.append(label_map[orig])
        per_person_count[orig] += 1

    # ── Print per-subject breakdown ───────────────────────────────────────────
    print(f"\n  Per-subject image counts:")
    for orig, new in sorted(label_map.items(), key=lambda x: x[1]):
        count = per_person_count[orig]
        bar   = "█" * (count // 5)
        print(f"    [{new:02d}] {names[orig]:<30} {count:>4} images  {bar}")

    return images, labels


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(force_reload=False):
    """
    Load the LFW dataset, cache to data/processed/, return X and y.

    Args:
        force_reload : bool — if True, ignore cache and re-fetch from sklearn.
                       Use this after changing LFW_MIN_FACES or LFW_MAX_PER_PERSON.

    Returns:
        X : np.ndarray  shape (n_samples, 16384)  float32  pixel values 0-1
        y : np.ndarray  shape (n_samples,)         int32   subject labels 0..n_classes-1
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    cache_tag = f"lfw_min{LFW_MIN_FACES}"
    if LFW_MAX_PER_PERSON is not None:
        cache_tag += f"_cap{LFW_MAX_PER_PERSON}"

    X_path = os.path.join(PROC_DIR, f"X_{cache_tag}.npy")
    y_path = os.path.join(PROC_DIR, f"y_{cache_tag}.npy")

    # ── Serve from cache if available ─────────────────────────────────────────
    if not force_reload and os.path.exists(X_path) and os.path.exists(y_path):
        print(f"Loading cached LFW dataset from data/processed/...")
        X = np.load(X_path)
        y = np.load(y_path)
        _print_summary(X, y)
        return X, y

    # ── Build from scratch ────────────────────────────────────────────────────
    print("\nBuilding LFW dataset")
    print("-" * 45)

    images, labels = _load_lfw()

    if not images:
        raise RuntimeError(
            "No LFW images loaded. Check your internet connection on first run, "
            "or verify ~/scikit_learn_data/lfw_home/ exists for cached runs."
        )

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Guarantee contiguous 0-indexed labels (required for CNN softmax output)
    unique = np.unique(y)
    if len(unique) != unique.max() + 1:
        le = LabelEncoder()
        y  = le.fit_transform(y).astype(np.int32)

    # ── Cache ─────────────────────────────────────────────────────────────────
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"\n  Cached → data/processed/{cache_tag}")

    _print_summary(X, y)
    return X, y


def _print_summary(X, y):
    n_subjects = len(np.unique(y))
    n_total    = X.shape[0]
    avg        = n_total / n_subjects if n_subjects > 0 else 0
    counts     = [np.sum(y == i) for i in range(n_subjects)]

    print(f"\n{'─'*45}")
    print(f"  Dataset      : LFW (min_faces={LFW_MIN_FACES})")
    print(f"  Total images : {n_total}")
    print(f"  Subjects     : {n_subjects}")
    print(f"  Avg per subj : {avg:.1f}")
    print(f"  Min per subj : {min(counts)}")
    print(f"  Max per subj : {max(counts)}")
    print(f"  Feature size : {X.shape[1]}  ({IMG_WIDTH}x{IMG_HEIGHT} flattened)")
    print(f"{'─'*45}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  LFW Dataset Loader — standalone test")
    print("=" * 55)

    X, y = load_dataset(force_reload=True)

    print(f"X shape : {X.shape}   dtype: {X.dtype}")
    print(f"y shape : {y.shape}   dtype: {y.dtype}")
    print(f"y range : {y.min()} – {y.max()}")
    print(f"Pixel range : {X.min():.4f} – {X.max():.4f}  (should be 0.0 – 1.0)")

    # Quick visual sanity check — show first image from each subject
    import matplotlib.pyplot as plt
    n_subjects = len(np.unique(y))
    fig, axes = plt.subplots(2, (n_subjects + 1) // 2, figsize=(18, 5))
    fig.suptitle(f"LFW — one sample per subject  (min_faces={LFW_MIN_FACES})", fontsize=13)

    for idx, ax in enumerate(axes.flat):
        if idx >= n_subjects:
            ax.axis("off")
            continue
        sample_idx = np.where(y == idx)[0][0]
        img = X[sample_idx].reshape(IMG_HEIGHT, IMG_WIDTH)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Class {idx}\n({np.sum(y==idx)} imgs)", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("lfw_sample_grid.png", dpi=120)
    print("\nSample grid saved → lfw_sample_grid.png")
    plt.show()