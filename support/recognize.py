# recognize.py
# Loads the saved PCA + SVM model and predicts who a face belongs to.
# Picks a random image from the test set, shows it, and prints the prediction.
# -------------------------------------------------------
# Run from repo root:  python -m support.recognize

import os
import sys
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split

# Import groupmate's data loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_normalized_data import load_dataset

# ── Settings (must match train_model.py exactly) ──────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
pca_path  = os.path.join(MODEL_DIR, "pca.pkl")
svm_path  = os.path.join(MODEL_DIR, "svm.pkl")


def recognize():

    # ── Step 1: Load saved model ──────────────────────────────────────────────
    print("=" * 55)
    print(" STEP 1 — Loading saved model")
    print("=" * 55)

    if not os.path.exists(pca_path) or not os.path.exists(svm_path):
        print("  ERROR: Model files not found in support/model/")
        print("  Please run train_model.py first!")
        return

    pca = joblib.load(pca_path)
    svm = joblib.load(svm_path)
    print("  Loaded → support/model/pca.pkl")
    print("  Loaded → support/model/svm.pkl")

    # ── Step 2: Recreate the same test set as train_model.py ──────────────────
    print("\n" + "=" * 55)
    print(" STEP 2 — Recreating test set")
    print("=" * 55)
    X, y = load_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Test images available : {X_test.shape[0]}")

    # ── Step 3: Pick a random test image ──────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 3 — Picking a random test image")
    print("=" * 55)
    random_idx   = random.randint(0, X_test.shape[0] - 1)
    test_image   = X_test[random_idx]       # flattened normalised vector
    actual_label = y_test[random_idx]       # true subject ID
    print(f"  Selected image index : {random_idx}")
    print(f"  Actual subject       : s{actual_label}")

    # ── Step 4: Apply PCA and predict ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 4 — Predicting subject")
    print("=" * 55)
    # Reshape to (1, 10304) — model expects a 2D array
    image_pca        = pca.transform(test_image.reshape(1, -1))
    predicted_label  = svm.predict(image_pca)[0]

    correct = predicted_label == actual_label
    result  = "CORRECT!" if correct else " WRONG!"

    print(f"  Actual    : s{actual_label}")
    print(f"  Predicted : s{predicted_label}")
    print(f"  Result    : {result}")

    # ── Step 5: Show the face image with prediction ───────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 5 — Displaying result")
    print("=" * 55)

    # Reshape back to image dimensions (112 height x 92 width) for display
    face_image = (test_image * 255).astype(np.uint8).reshape(112, 92)

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.imshow(face_image, cmap="gray")
    ax.set_title(
        f"Actual: s{actual_label}  |  Predicted: s{predicted_label}  {result}",
        fontsize=12,
        color="green" if correct else "red"
    )
    ax.axis("off")
    plt.tight_layout()

    # Save the result image
    result_path = os.path.join(MODEL_DIR, "recognition_result.png")
    plt.savefig(result_path, dpi=150)
    print(f"  Saved → support/model/recognition_result.png")
    plt.show()

    print("\n" + "=" * 55)
    print(f"  ALL DONE  —  Predicted s{predicted_label}, Actual s{actual_label}  {result}")
    print("=" * 55 + "\n")

    return predicted_label, actual_label, correct


if __name__ == "__main__":
    recognize()
