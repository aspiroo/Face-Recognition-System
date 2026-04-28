# load_data_lfw.py
# LFW-only dataset loader — Chapter 2.
# Location: support/lfw/load_data_lfw.py
#
# Changes from original:
#   - BASE_DIR uses 3x dirname (file is 2 levels deep: support/lfw/)
#   - PROC_DIR correctly points to repo_root/data/processed/
#   - No image resizing — LFW images kept at native sklearn crop size
#   - Standalone save path fixed to use MODEL_DIR not hardcoded string
# ─────────────────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import LabelEncoder

# ── File is at support/lfw/load_data_lfw.py ──────────────────────────────────
# dirname(__file__)        = support/lfw/
# dirname(dirname)         = support/
# dirname(dirname(dirname))= repo root  ← correct
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "support", "lfw", "model")

# ── LFW settings ──────────────────────────────────────────────────────────────
LFW_MIN_FACES      = 40
LFW_MAX_PER_PERSON = 100

# ── Image dimensions (native sklearn LFW crop — no resizing needed) ───────────
# fetch_lfw_people with slice_=None returns ~250x250 full images
# With default slice_ it returns 62x47. We use resize=0.5 → 125x94.
# All models flatten to 1D so exact size doesn't matter as long as it's consistent.
IMG_HEIGHT = 62
IMG_WIDTH  = 47


def _load_lfw():
    """
    Fetch LFW via sklearn. No manual resizing — use sklearn's built-in resize.
    Returns images as flat float32 arrays normalised to 0-1.
    """
    print(f"  Fetching LFW (min_faces_per_person={LFW_MIN_FACES})...")
    print(f"  First run downloads ~200MB to ~/scikit_learn_data/ — cached after that.")

    lfw = fetch_lfw_people(
        min_faces_per_person=LFW_MIN_FACES,
        resize=0.5,      # 125x94 → good balance of detail vs speed
        color=False,     # grayscale
    )

    raw_images = lfw.images       # (n_samples, h, w)  float32  0-255
    raw_labels = lfw.target       # (n_samples,)  int
    names      = lfw.target_names

    # Update actual dimensions from fetched data
    global IMG_HEIGHT, IMG_WIDTH
    IMG_HEIGHT = raw_images.shape[1]
    IMG_WIDTH  = raw_images.shape[2]

    n_raw      = raw_images.shape[0]
    n_subjects = len(np.unique(raw_labels))
    print(f"  Raw fetch   : {n_raw} images | {n_subjects} subjects | {IMG_HEIGHT}x{IMG_WIDTH}px")

    label_map        = {}
    per_person_count = {}
    images, labels   = [], []

    for i in range(n_raw):
        orig = raw_labels[i]

        if orig not in label_map:
            label_map[orig]        = len(label_map)
            per_person_count[orig] = 0

        if LFW_MAX_PER_PERSON is not None and per_person_count[orig] >= LFW_MAX_PER_PERSON:
            continue

        new_label = label_map[orig]

        # Keep float32, normalise to 0-1 — never cast to uint8
        img = raw_images[i].astype(np.float32).flatten() / 255.0

        images.append(img)
        labels.append(new_label)
        per_person_count[orig] += 1

    print(f"\n  Per-subject image counts:")
    for orig, new in sorted(label_map.items(), key=lambda x: x[1]):
        count = per_person_count[orig]
        bar   = "█" * (count // 5)
        print(f"    [{new:02d}] {names[orig]:<30} {count:>4} images  {bar}")

    return images, labels


def load_dataset(force_reload=False):
    """
    Load LFW, cache to data/processed/, return X and y.

    Returns:
        X : np.ndarray (n_samples, n_features)  float32  0-1
        y : np.ndarray (n_samples,)              int32    0-indexed labels
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    cache_tag = f"lfw_min{LFW_MIN_FACES}"
    if LFW_MAX_PER_PERSON is not None:
        cache_tag += f"_cap{LFW_MAX_PER_PERSON}"

    X_path = os.path.join(PROC_DIR, f"X_{cache_tag}.npy")
    y_path = os.path.join(PROC_DIR, f"y_{cache_tag}.npy")

    if not force_reload and os.path.exists(X_path) and os.path.exists(y_path):
        print(f"Loading cached LFW dataset from data/processed/...")
        X = np.load(X_path)
        y = np.load(y_path)
        _print_summary(X, y)
        return X, y

    print("\nBuilding LFW dataset")
    print("-" * 45)

    images, labels = _load_lfw()

    if not images:
        raise RuntimeError("No LFW images loaded.")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    unique = np.unique(y)
    if len(unique) != unique.max() + 1:
        le = LabelEncoder()
        y  = le.fit_transform(y).astype(np.int32)

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
    print(f"  Feature size : {X.shape[1]}")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("  LFW Dataset Loader — standalone test")
    print("=" * 55)

    X, y = load_dataset(force_reload=True)

    print(f"X shape     : {X.shape}   dtype: {X.dtype}")
    print(f"y shape     : {y.shape}   dtype: {y.dtype}")
    print(f"Pixel range : {X.min():.4f} – {X.max():.4f}")

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
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "lfw_sample_grid.png")
    plt.savefig(save_path, dpi=120)
    print(f"\nSample grid saved → {save_path}")
    plt.show()