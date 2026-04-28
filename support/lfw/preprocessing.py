# preprocessing.py
# Shared preprocessing for LFW models (SVM, KNN, RF, CNN).
# Location: support/lfw/preprocessing.py
# 60/20/20 train/val/test split + PCA.
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
N_COMPONENTS = 150

# ── Dimensions must match load_data_lfw.py ────────────────────────────────────
# Updated to match native LFW sklearn crop (resize=0.5 → ~62x47)
# These are only used for reshape in prepare_data_raw (CNN)
IMG_HEIGHT = 62
IMG_WIDTH  = 47


def prepare_data(X, y, n_components=N_COMPONENTS, verbose=True):
    """
    60/20/20 stratified split + PCA for LFW dataset.
    Used by SVM, KNN, RF.

    Returns:
        X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca
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

    pca         = PCA(n_components=n_components, whiten=True, random_state=RANDOM_STATE)
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
    Reshapes flat vectors back to 2D images.
    """
    # Infer image dimensions from feature size
    n_features = X.shape[1]
    h = IMG_HEIGHT
    w = IMG_WIDTH

    # Safety check — if feature size doesn't match, infer square-ish dims
    if h * w != n_features:
        import math
        h = int(math.sqrt(n_features))
        w = n_features // h

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

    X_train = X_train_flat.reshape(-1, h, w)
    X_val   = X_val_flat.reshape(-1,   h, w)
    X_test  = X_test_flat.reshape(-1,  h, w)

    if verbose:
        print(f"Split  → Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        print(f"         ({X_train.shape[0]/len(X)*100:.0f}% / {X_val.shape[0]/len(X)*100:.0f}% / {X_test.shape[0]/len(X)*100:.0f}%)")
        print(f"Shape  → {X_train.shape}  (raw 2D images, no PCA)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_explained_variance(pca, save_path=None):
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=2, color="steelblue")
    ax.axhline(y=90, color="red",    linestyle="--", linewidth=1, label="90%")
    ax.axhline(y=95, color="orange", linestyle="--", linewidth=1, label="95%")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA — Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_eigenfaces(pca, n_faces=10, save_path=None):
    eigenfaces = pca.components_[:n_faces].reshape((n_faces, IMG_HEIGHT, IMG_WIDTH))
    fig, axes  = plt.subplots(2, n_faces // 2, figsize=(15, 6))
    fig.suptitle("Top Eigenfaces (PCA Components)", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(eigenfaces[i], cmap="gray")
        ax.set_title(f"PC {i+1}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()