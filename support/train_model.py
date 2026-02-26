# train_model.py
# Trains a face recognition model using PCA (Eigenfaces) + SVM classifier.
# Loads data via load_normalized_data.py, saves trained model to support/model/
# -------------------------------------------------------
# Run from repo root:  python -m support.train_model

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import groupmate's data loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_normalized_data import load_dataset

# ── Tuneable settings ────────────────────────────────────────────────────────
N_COMPONENTS = 100    # number of eigenfaces to keep (try 50, 100, 150)
TEST_SIZE    = 0.2    # 20% test → 80 images test, 320 images train
RANDOM_STATE = 42     # fixed seed for reproducibility

SVM_C        = 10.0   # SVM regularisation (try 1, 10, 100)
SVM_GAMMA    = "scale"

# ── Output directory (support/model/) ────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def train():

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("=" * 55)
    print(" STEP 1 — Loading dataset")
    print("=" * 55)
    # Calls groupmate's loader — reads from data/raw/ on first run,
    # then uses cached data/processed/ .npy files on every run after
    X, y = load_dataset()
    print(f"  X shape : {X.shape}")             # (400, 10304)
    print(f"  y shape : {y.shape}")             # (400,)
    print(f"  Subjects: {len(np.unique(y))}")   # 40

    # ── Step 2: Train / Test split ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 2 — Train / Test split  (80% / 20%)")
    print("=" * 55)
    # stratify=y guarantees every subject appears in BOTH train and test
    # so the model sees all 40 people during training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Training images : {X_train.shape[0]}")   # 320
    print(f"  Testing  images : {X_test.shape[0]}")    # 80

    # ── Step 3: PCA — Eigenfaces ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f" STEP 3 — PCA  (n_components={N_COMPONENTS}, whiten=True)")
    print("=" * 55)
    # PCA compresses 10,304 features down to 100 "eigenfaces"
    # whiten=True scales each component to unit variance → better SVM accuracy
    # fit on TRAINING data ONLY → then apply same transform to test
    # (fitting on test data too would be cheating — called "data leakage")
    pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)   # learn the eigenfaces from train
    X_test_pca  = pca.transform(X_test)        # apply same compression to test

    var_kept = pca.explained_variance_ratio_.sum() * 100
    print(f"  Reduced  : {X.shape[1]} features → {N_COMPONENTS} features")
    print(f"  Variance retained : {var_kept:.1f}%")

    # ── Step 4: Train SVM ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 4 — Training SVM  (kernel=rbf)")
    print("=" * 55)
    # SVM finds the best boundary between all 40 subjects
    # RBF kernel handles non-linear boundaries — best for face data
    print(f"  C={SVM_C}  |  gamma={SVM_GAMMA}")
    svm = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, random_state=RANDOM_STATE)
    svm.fit(X_train_pca, y_train)
    print("  Training complete ✓")

    # ── Step 5: Quick accuracy check ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 5 — Quick accuracy check")
    print("=" * 55)
    y_pred   = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")

    if accuracy >= 0.93:
        print("  → Excellent result!")
    elif accuracy >= 0.85:
        print("  → Good. Try tuning N_COMPONENTS or SVM_C.")
    else:
        print("  → Try N_COMPONENTS=150 or SVM_C=50.")

    # ── Step 6: Save model and PCA to support/model/ ─────────────────────────
    print("\n" + "=" * 55)
    print(" STEP 6 — Saving model")
    print("=" * 55)
    # Save both pca and svm — evaluate.py and recognize.py will load these
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    svm_path = os.path.join(MODEL_DIR, "svm.pkl")
    joblib.dump(pca, pca_path)
    joblib.dump(svm, svm_path)
    print(f"  Saved → support/model/pca.pkl")
    print(f"  Saved → support/model/svm.pkl")

    # ── Step 7: Eigenfaces plot (great for report / presentation) ────────────
    print("\n" + "=" * 55)
    print(" STEP 7 — Saving eigenfaces visualisation")
    print("=" * 55)
    # Each eigenface is a PCA component reshaped back into image dimensions
    # They look like ghostly faces — the "features" the model learned
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Top 10 Eigenfaces  (PCA Components)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        eigenface = pca.components_[i].reshape(112, 92)   # AT&T: height=112, width=92
        ax.imshow(eigenface, cmap="gray")
        ax.set_title(f"Eigenface {i + 1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "eigenfaces.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved → support/model/eigenfaces.png")

    print("\n" + "=" * 55)
    print(f"  ALL DONE  —  Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55 + "\n")

    return pca, svm, accuracy


if __name__ == "__main__":
    train()