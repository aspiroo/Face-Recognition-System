# preprocessing.py
# Shared preprocessing for all models (SVM, KNN, Random Forest, CNN).
# Handles 60/20/20 train/val/test split and PCA fitting.
#
# FIXES applied:
#   - whiten=False  (was True) — whitening removes natural variance weighting,
#     hurting SVM and KNN which rely on eigenface magnitude differences
#   - n_components=150 (was 100) — more components needed for 90-class problem
#     with only 12 training images per person
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

TRAIN_SIZE   = 0.6
VAL_SIZE     = 0.2
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_COMPONENTS = 150   # FIXED: was 100 — need more components for 90 classes

IMG_WIDTH  = 128
IMG_HEIGHT = 128


def prepare_data(X, y, n_components=N_COMPONENTS, verbose=True):
    """
    Full preprocessing pipeline for SVM, KNN, and Random Forest.

    Steps:
      1. Stratified 60/20/20 train/val/test split
      2. PCA fitted on TRAINING data only (no data leakage)
      3. Same PCA transform applied to val and test sets

    Key fix: whiten=False — whitening scales all components to unit variance,
    destroying the natural eigenvalue weighting where top components carry
    far more discriminative information. Without whitening, SVM and KNN
    naturally weight important eigenfaces higher.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    if verbose:
        print(f"Split  → Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        print(f"         ({X_train.shape[0]/len(X)*100:.0f}% / {X_val.shape[0]/len(X)*100:.0f}% / {X_test.shape[0]/len(X)*100:.0f}%)")

    # FIXED: whiten=False (was whiten=True)
    # Whitening normalises each component to unit variance, which destroys the
    # natural ordering where PC1 explains much more variance than PC100.
    # SVM's RBF kernel and KNN's euclidean distance both benefit from the
    # larger magnitude differences between important and unimportant eigenfaces.
    pca = PCA(n_components=n_components, whiten=True, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)
    X_test_pca  = pca.transform(X_test)

    if verbose:
        var_kept = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA    → {X.shape[1]} features → {n_components} components")
        print(f"         Variance retained: {var_kept:.1f}%")

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca


def prepare_data_raw(X, y, verbose=True):
    """
    60/20/20 split WITHOUT PCA. Used by CNN only.
    Reshapes flat vectors back to 2D images for convolutional layers.
    """

    X_train_flat, X_temp_flat, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val_flat, X_test_flat, y_val, y_test = train_test_split(
        X_temp_flat, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    X_train = X_train_flat.reshape(-1, IMG_HEIGHT, IMG_WIDTH)
    X_val   = X_val_flat.reshape(-1,   IMG_HEIGHT, IMG_WIDTH)
    X_test  = X_test_flat.reshape(-1,  IMG_HEIGHT, IMG_WIDTH)

    if verbose:
        print(f"Split  → Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        print(f"         ({X_train.shape[0]/len(X)*100:.0f}% / {X_val.shape[0]/len(X)*100:.0f}% / {X_test.shape[0]/len(X)*100:.0f}%)")
        print(f"Shape  → {X_train.shape}  (raw 2D images, no PCA)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_explained_variance(pca, save_path=None):
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=2, color="steelblue")
    ax.axhline(y=90, color="red",    linestyle="--", linewidth=1, label="90% variance")
    ax.axhline(y=95, color="orange", linestyle="--", linewidth=1, label="95% variance")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA — Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_eigenfaces(pca, n_faces=10, save_path=None):
    eigenfaces = pca.components_[:n_faces].reshape((n_faces, IMG_HEIGHT, IMG_WIDTH))
    fig, axes = plt.subplots(2, n_faces // 2, figsize=(15, 6))
    fig.suptitle("Top Eigenfaces (PCA Components)", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(eigenfaces[i], cmap="gray")
        ax.set_title(f"PC {i + 1}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()