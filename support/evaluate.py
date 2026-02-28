# evaluate.py
# Loads the saved PCA + SVM model and evaluates it on the test set.
# Produces accuracy, classification report, and confusion matrix.
# Saves confusion matrix plot to support/model/confusion_matrix.png
# -------------------------------------------------------
# Run from repo root:  python -m support.evaluate

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Import groupmate's data loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_normalized_data import load_dataset

# ── Settings (must match train_model.py exactly) ─────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
pca_path  = os.path.join(MODEL_DIR, "pca.pkl")
svm_path  = os.path.join(MODEL_DIR, "svm.pkl")


def evaluate():

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

    # ── Step 2: Recreate the same test set as train_model.py ─────────────────
    print("\n" + "=" * 55)
    print(" STEP 2 — Recreating test set")
    print("=" * 55)
    # Using the exact same TEST_SIZE and RANDOM_STATE as train_model.py
    # guarantees we get the exact same 80 test images
    X, y = load_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Test images : {X_test.shape[0]}")
    print(f"  Subjects    : {len(np.unique(y_test))}")

    # ── Step 3: Apply PCA to test set ────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 3 — Applying PCA to test set")
    print("=" * 55)
    # Use transform (NOT fit_transform) — we apply the already-learned compression
    X_test_pca = pca.transform(X_test)
    print(f"  Compressed : {X_test.shape[1]} features → {X_test_pca.shape[1]} features")

    # ── Step 4: Predict ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 4 — Running predictions")
    print("=" * 55)
    y_pred   = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")

    # ── Step 5: Classification report ────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 5 — Classification report  (per subject)")
    print("=" * 55)
    # Shows precision, recall, f1-score for every subject
    # precision = of all times we said "subject X", how often were we right?
    # recall    = of all actual subject X images, how many did we catch?
    # f1-score  = balance between precision and recall
    labels      = sorted(np.unique(y_test))
    label_names = [f"s{l}" for l in labels]   # e.g. "s1", "s2", ..., "s40"
    report = classification_report(y_test, y_pred, labels=labels, target_names=label_names)
    print(report)

    # ── Step 6: Confusion matrix ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 6 — Saving confusion matrix plot")
    print("=" * 55)
    # Rows = actual subject, Columns = predicted subject
    # Diagonal = correct predictions (we want these to be high)
    # Off-diagonal = mistakes (we want these to be 0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix  —  Accuracy: {accuracy * 100:.2f}%", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved → support/model/confusion_matrix.png")

    print("\n" + "=" * 55)
    print(f"  ALL DONE  —  Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55 + "\n")

    return accuracy, report, cm


if __name__ == "__main__":
    evaluate()