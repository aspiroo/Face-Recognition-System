# generate_att_sample_grid.py
# Generates a grid showing one sample image per subject from the AT&T dataset.
# Saves to support/att/model/att_sample_grid.png
#
# Run from repo root: python generate_att_sample_grid.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATT_DIR  = os.path.join(ROOT, "data", "raw", "att")
SAVE_DIR = os.path.join(ROOT, "support", "att", "model")
os.makedirs(SAVE_DIR, exist_ok=True)


def generate_att_grid():
    # ── Collect one image per subject ─────────────────────────────────────────
    subjects = sorted([
        d for d in os.listdir(ATT_DIR)
        if os.path.isdir(os.path.join(ATT_DIR, d)) and d.startswith("s")
    ], key=lambda x: int(x[1:]))

    print(f"  Found {len(subjects)} subjects in {ATT_DIR}")

    images = []
    labels = []

    for subject in subjects:
        subject_path = os.path.join(ATT_DIR, subject)
        img_files = sorted([
            f for f in os.listdir(subject_path)
            if f.endswith(".pgm")
        ])

        if not img_files:
            continue

        # Take the first image from each subject
        img_path = os.path.join(subject_path, img_files[0])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"  Warning: could not read {img_path}")
            continue

        images.append(img)
        labels.append(int(subject[1:]))   # e.g. "s1" → 1

    n_subjects = len(images)
    print(f"  Loaded {n_subjects} sample images")

    # ── Plot grid ─────────────────────────────────────────────────────────────
    # 8 columns, enough rows to fit all 40 subjects
    n_cols = 8
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.2))
    fig.suptitle(
        f"AT\\&T Dataset — One Sample per Subject ({n_subjects} subjects, "
        f"10 images each, 92$\\times$112 px)",
        fontsize=13
    )

    for idx, ax in enumerate(axes.flat):
        if idx < n_subjects:
            ax.imshow(images[idx], cmap="gray")
            ax.set_title(f"s{labels[idx]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "att_sample_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
    return save_path


if __name__ == "__main__":
    print("=" * 55)
    print("  AT&T Sample Grid Generator")
    print("=" * 55)

    if not os.path.exists(ATT_DIR):
        print(f"\n  ERROR: AT&T dataset not found at {ATT_DIR}")
        print(f"  Please download and extract to data/raw/att/")
        sys.exit(1)

    path = generate_att_grid()
    print(f"\n  Done — grid saved to: {path}")
    print(f"  Copy to Overleaf as:  fig/att_sample_grid.png")