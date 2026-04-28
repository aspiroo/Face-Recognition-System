# load_data_lfw_yale.py
# Loads and merges Yale Face B + LFW datasets.
#
# FIXES applied:
#   - LFW: removed .astype(np.uint8) cast before normalization.
#     fetch_lfw_people returns float32 in 0-255 range. Casting to uint8
#     truncates the decimal portion (e.g. 127.9 → 127), losing precision
#     that accumulates badly across thousands of pixels per image.
#   - LFW: pass color=False and a consistent slice_ to avoid double-resizing.
#     The raw fetch now returns consistent dimensions we resize ourselves.
#
# Sources:
#   Yale Face B  — downloaded manually to data/raw/yale/
#   LFW          — loaded via sklearn built-in (auto-downloads, no folder needed)
#
# All images are resized to 128x128, grayscale, normalized 0–1, flattened.
# ──────────────────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import LabelEncoder

IMG_WIDTH  = 128
IMG_HEIGHT = 128

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YALE_DIR  = os.path.join(BASE_DIR, "data", "raw", "yale")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")

YALE_MAX_PER_PERSON = 20
LFW_MIN_FACES       = 20
LFW_MAX_PER_PERSON  = 20


def _read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Warning: could not read {img_path}, skipping.")
        return None
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.flatten().astype(np.float32) / 255.0
    return img


def _load_yale(start_label=0):
    images, labels = [], []

    if not os.path.exists(YALE_DIR):
        print(f"  Warning: Yale not found at {YALE_DIR} — skipping.")
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
        img_files = img_files[:YALE_MAX_PER_PERSON]

        loaded = 0
        for fname in img_files:
            img = _read_image(os.path.join(folder_path, fname))
            if img is not None:
                images.append(img)
                labels.append(label_idx)
                loaded += 1

        if loaded > 0:
            label_idx += 1

    print(f"  Yale Face B : {len(images)} images loaded ({label_idx - start_label} subjects)")
    return images, labels, label_idx


def _load_lfw(start_label=0):
    """
    Load LFW via sklearn built-in.

    FIX: The original code did raw_images[i].astype(np.uint8) before dividing
    by 255. fetch_lfw_people returns float32 values in the 0–255 range (e.g.
    127.63). Casting to uint8 first truncates to 127, discarding the decimal
    portion. Over 16384 pixels this precision loss is significant and
    degrades model accuracy. We now keep float32 and divide directly.
    """
    print(f"  LFW         : fetching via sklearn (min_faces={LFW_MIN_FACES})...")
    print(f"                First run downloads ~200MB — subsequent runs load from cache.")

    lfw = fetch_lfw_people(
        min_faces_per_person=LFW_MIN_FACES,
        resize=None,
        color=False,
        slice_=None
    )

    raw_images = lfw.images   # shape: (n_samples, h, w), float32, range 0-255
    raw_labels = lfw.target

    print(f"  LFW         : {raw_images.shape[0]} images, {len(np.unique(raw_labels))} people raw")

    images, labels   = [], []
    lfw_label_map    = {}
    per_person_count = {}

    for i in range(len(raw_images)):
        orig_label = raw_labels[i]

        if orig_label not in lfw_label_map:
            lfw_label_map[orig_label]    = start_label + len(lfw_label_map)
            per_person_count[orig_label] = 0

        if per_person_count[orig_label] >= LFW_MAX_PER_PERSON:
            continue

        new_label = lfw_label_map[orig_label]

        # FIXED: was .astype(np.uint8) which truncated float precision.
        # Keep as float32 — cv2.resize handles float images correctly.
        img = raw_images[i].astype(np.float32)          # already 0-255 float
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.flatten() / 255.0                      # normalize to 0-1

        images.append(img)
        labels.append(new_label)
        per_person_count[orig_label] += 1

    next_label = start_label + len(lfw_label_map)
    print(f"  LFW         : {len(images)} images loaded ({len(lfw_label_map)} subjects, capped at {LFW_MAX_PER_PERSON}/person)")
    return images, labels, next_label


def load_dataset(mode="yale_lfw", force_reload=False):
    """
    Main entry point. See module docstring for usage.

    Args:
        mode         : "yale" | "lfw" | "yale_lfw"
        force_reload : ignore cache and re-read from scratch

    Returns:
        X : np.ndarray (n_samples, 16384)
        y : np.ndarray (n_samples,)
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    X_path = os.path.join(PROC_DIR, f"X_{mode}.npy")
    y_path = os.path.join(PROC_DIR, f"y_{mode}.npy")

    if not force_reload and os.path.exists(X_path) and os.path.exists(y_path):
        print(f"Loading cached dataset ({mode}) from data/processed/...")
        X = np.load(X_path)
        y = np.load(y_path)
        _print_summary(X, y, mode)
        return X, y

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

    unique_labels = np.unique(y)
    if len(unique_labels) != unique_labels.max() + 1:
        le = LabelEncoder()
        y  = le.fit_transform(y).astype(np.int32)

    np.save(X_path, X)
    np.save(y_path, y)
    print(f"Cached to data/processed/")

    _print_summary(X, y, mode)
    return X, y


def _print_summary(X, y, mode):
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


if __name__ == "__main__":
    # Pass force_reload=True to bust the cache after applying these fixes
    X, y = load_dataset("yale_lfw", force_reload=True)