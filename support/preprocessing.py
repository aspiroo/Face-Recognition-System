# preprocessing.py
# Shared preprocessing for all models (SVM, KNN, Random Forest, CNN).
# Handles 60/20/20 train/val/test split and PCA fitting.
#
# Split rationale:
#   Train (60%)      — model learns from this
#   Validation (20%) — hyperparameter tuning, CNN early stopping
#   Test (20%)       — touched ONCE at the very end for final accuracy
#
# ── Usage ────────────────────────────────────────────────────────────────────
#
#   SVM / KNN / Random Forest:
#     from support.preprocessing import prepare_data
#     X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca = prepare_data(X, y)
#
#   CNN only (no PCA, raw 2D images):
#     from support.preprocessing import prepare_data_raw
#     X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_raw(X, y)
#
# Run standalone to verify:
#   python -m support.preprocessing
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ── Shared settings ───────────────────────────────────────────────────────────
# All models use the same split — changing here changes for everyone.
TRAIN_SIZE   = 0.6    # 60% train
VAL_SIZE     = 0.2    # 20% validation
TEST_SIZE    = 0.2    # 20% test
RANDOM_STATE = 42     # fixed seed — reproducible splits every run
N_COMPONENTS = 100    # PCA components to keep

IMG_WIDTH  = 128      # must match load_data.py
IMG_HEIGHT = 128


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(X, y, n_components=N_COMPONENTS, verbose=True):
    """
    Full preprocessing pipeline for SVM, KNN, and Random Forest.

    Steps:
      1. Stratified 60/20/20 train/val/test split
      2. PCA fitted on TRAINING data only (no data leakage)
      3. Same PCA transform applied to val and test sets

    Args:
        X            : np.ndarray (n_samples, n_features) — raw flattened images from load_data.py
        y            : np.ndarray (n_samples,)             — integer subject labels
        n_components : number of PCA components to keep (default 100)
        verbose      : print progress summary (default True)

    Returns:
        X_train_pca : np.ndarray (n_train, n_components)
        X_val_pca   : np.ndarray (n_val,   n_components)
        X_test_pca  : np.ndarray (n_test,  n_components)
        y_train     : np.ndarray (n_train,)
        y_val       : np.ndarray (n_val,)
        y_test      : np.ndarray (n_test,)
        pca         : fitted PCA object — pass this to evaluate/recognize scripts
    """

    # ── Step 1: First split — carve out 60% train, 40% temp ──────────────────
    # stratify=y ensures every subject appears in all three sets
    # With 20 images/person this gives: 12 train, 4 val, 4 test per subject
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),   # 0.4 → 40% goes to temp
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ── Step 2: Second split — split temp 50/50 into val and test ────────────
    # 50% of the 40% temp = 20% val, 20% test of total
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    if verbose:
        print(f"Split  → Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        print(f"         ({X_train.shape[0]/len(X)*100:.0f}% / {X_val.shape[0]/len(X)*100:.0f}% / {X_test.shape[0]/len(X)*100:.0f}%)")

    # ── Step 3: Fit PCA on training data only ────────────────────────────────
    # CRITICAL: fit() on X_train only — never on val or test.
    # Fitting on val/test = data leakage = inflated accuracy.
    # whiten=True: scales components to unit variance — improves SVM and KNN.
    pca = PCA(n_components=n_components, whiten=True, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)   # learns eigenfaces from train set

    # ── Step 4: Apply same PCA to val and test ────────────────────────────────
    # transform() uses the already-learned compression — does NOT refit
    X_val_pca  = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    if verbose:
        var_kept = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA    → {X.shape[1]} features → {n_components} components")
        print(f"         Variance retained: {var_kept:.1f}%")

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca


def prepare_data_raw(X, y, verbose=True):
    """
    60/20/20 split WITHOUT PCA. Used by CNN only.
    CNN learns its own features through convolution — does not need PCA.
    Reshapes flat vectors back to 2D images for convolutional layers.

    Args:
        X       : np.ndarray (n_samples, 16384) — flattened images from load_data.py
        y       : np.ndarray (n_samples,)        — integer subject labels
        verbose : print progress summary (default True)

    Returns:
        X_train : np.ndarray (n_train, IMG_HEIGHT, IMG_WIDTH) — 2D grayscale images
        X_val   : np.ndarray (n_val,   IMG_HEIGHT, IMG_WIDTH)
        X_test  : np.ndarray (n_test,  IMG_HEIGHT, IMG_WIDTH)
        y_train : np.ndarray (n_train,)
        y_val   : np.ndarray (n_val,)
        y_test  : np.ndarray (n_test,)
    """

    # ── Step 1: 60% train, 40% temp ──────────────────────────────────────────
    X_train_flat, X_temp_flat, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ── Step 2: 50/50 split of temp → 20% val, 20% test ─────────────────────
    X_val_flat, X_test_flat, y_val, y_test = train_test_split(
        X_temp_flat, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # ── Step 3: Reshape flat vectors to 2D images for CNN ────────────────────
    # CNN expects (n_samples, height, width) — channel dim added in model_cnn.py
    X_train = X_train_flat.reshape(-1, IMG_HEIGHT, IMG_WIDTH)
    X_val   = X_val_flat.reshape(-1,   IMG_HEIGHT, IMG_WIDTH)
    X_test  = X_test_flat.reshape(-1,  IMG_HEIGHT, IMG_WIDTH)

    if verbose:
        print(f"Split  → Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        print(f"         ({X_train.shape[0]/len(X)*100:.0f}% / {X_val.shape[0]/len(X)*100:.0f}% / {X_test.shape[0]/len(X)*100:.0f}%)")
        print(f"Shape  → {X_train.shape}  (raw 2D images, no PCA)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_explained_variance(pca, save_path=None):
    """
    Plot cumulative explained variance vs number of PCA components.
    The elbow shows the minimum components needed to retain most variance.
    """
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
    """
    Visualize the top-n eigenfaces (PCA components reshaped back to images).
    These ghostly face patterns are the features the model uses to distinguish people.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from load_data_lfw_yale import load_dataset

    print("=" * 55)
    print("  Preprocessing sanity check — 60/20/20 split")
    print("=" * 55)

    X, y = load_dataset("yale")
    print()

    # ── SVM/KNN/RF path ───────────────────────────────────────────────────────
    print("── PCA path (SVM / KNN / RF) ──")
    X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca = prepare_data(X, y)

    print(f"\nX_train_pca : {X_train_pca.shape}")
    print(f"X_val_pca   : {X_val_pca.shape}")
    print(f"X_test_pca  : {X_test_pca.shape}")
    print(f"y_train     : {y_train.shape}")
    print(f"y_val       : {y_val.shape}")
    print(f"y_test      : {y_test.shape}")

    print()

    # ── CNN path ──────────────────────────────────────────────────────────────
    print("── Raw path (CNN) ──")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_raw(X, y)

    print(f"\nX_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"X_test  : {X_test.shape}")

    # ── Save plots ───────────────────────────────────────────────────────────
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, "support")
    os.makedirs(output_dir, exist_ok=True)
 
    variance_path   = os.path.join(output_dir, "pca_variance.png")
    eigenfaces_path = os.path.join(output_dir, "eigenfaces.png")
 
    print()
    print("Saving plots...")
    plot_explained_variance(pca, save_path=variance_path)
    plot_eigenfaces(pca, save_path=eigenfaces_path)
    print(f"  Saved → support/model/pca_variance.png")
    print(f"  Saved → support/model/eigenfaces.png")