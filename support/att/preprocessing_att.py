# preprocessing_att.py
# Shared preprocessing for AT&T models (SVM, KNN, RF).
# 80/20 stratified train/test split — NO validation set.
#
# Why no validation set:
#   AT&T has only 10 images per person x 40 subjects = 400 total.
#   80/20 gives 8 train / 2 test per person.
#   A three-way split leaves only 1 image per person for validation
#   which is statistically meaningless. Hyperparameter tuning is done
#   via cross-validation on the training set instead.
#
# PCA sweep function is provided here so all three models (SVM, KNN, RF)
# can call it with their own proxy classifier to find the best n_components.
#
# Run standalone to verify split:
#   python -m support.att.preprocessing_att
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TRAIN_SIZE   = 0.8    # 80% train → 320 images (8 per person)
TEST_SIZE    = 0.2    # 20% test  →  80 images (2 per person)
RANDOM_STATE = 42
IMG_HEIGHT   = 112    # AT&T native dimensions
IMG_WIDTH    = 92


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SPLIT + PCA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(X, y, n_components, whiten=False, verbose=True):
    """
    80/20 stratified split + PCA for AT&T dataset.

    Args:
        X            : np.ndarray (400, 10304) — from load_normalized_data.py
        y            : np.ndarray (400,)        — subject IDs 1-40
        n_components : int — PCA components to keep
        whiten       : bool — whiten PCA components (default False)
        verbose      : bool — print summary (default True)

    Returns:
        X_train_pca : np.ndarray (320, n_components)
        X_test_pca  : np.ndarray (80,  n_components)
        y_train     : np.ndarray (320,)
        y_test      : np.ndarray (80,)
        pca         : fitted PCA object
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    if verbose:
        print(f"Split  → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        print(f"         ({X_train.shape[0]/len(X)*100:.0f}% / {X_test.shape[0]/len(X)*100:.0f}%)")

    pca         = PCA(n_components=n_components, whiten=whiten, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    if verbose:
        var_kept = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA    → {X.shape[1]} features → {n_components} components")
        print(f"         Variance retained: {var_kept:.1f}%")

    return X_train_pca, X_test_pca, y_train, y_test, pca


# ─────────────────────────────────────────────────────────────────────────────
# PCA SWEEP — shared by all three AT&T models
# ─────────────────────────────────────────────────────────────────────────────

def sweep_components(X, y, proxy_clf, label="proxy", whiten=False, step=5, max_n=150):
    """
    Sweep PCA components using cross-validation on the training set.
    Uses a proxy classifier (fast version of the real model) to score each n.

    Why CV instead of val set:
        With only 8 training images per person, a held-out val set would have
        1 image per person — too noisy to trust. 5-fold CV on the 320 training
        images gives more stable estimates.

    Args:
        X          : full dataset (400, 10304)
        y          : full labels  (400,)
        proxy_clf  : sklearn classifier — fast version for sweep
                     e.g. SVC(C=10), KNeighborsClassifier(n_neighbors=1),
                          RandomForestClassifier(n_estimators=100)
        label      : string label for print output
        whiten     : bool — passed through to prepare_data
        step       : int — component step size (default 5)
        max_n      : int — maximum components to try (default 150)

    Returns:
        best_n  : int — n_components with highest CV accuracy
        results : list of (n_comp, cv_acc, var_retained) tuples
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    # Split once to get training set for CV
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Sweeping PCA components ({label})...")
    print(f"  {'n_components':>14}  {'Variance':>10}  {'CV Acc (5-fold)':>16}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*16}")

    best_n   = 20
    best_acc = 0
    results  = []
    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for n_comp in range(5, max_n + 1, step):
        pca     = PCA(n_components=n_comp, whiten=whiten, random_state=RANDOM_STATE)
        X_pca   = pca.fit_transform(X_train)
        var     = pca.explained_variance_ratio_.sum() * 100
        scores  = cross_val_score(proxy_clf, X_pca, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        acc     = scores.mean()
        marker  = "  ←" if acc > best_acc else ""
        print(f"  {n_comp:>14}  {var:>9.1f}%  {acc*100:>15.1f}%{marker}")

        results.append((n_comp, acc, var))

        if acc > best_acc:
            best_acc = acc
            best_n   = n_comp

    print(f"\n  Auto-selected n_components={best_n}  (CV acc={best_acc*100:.1f}%)")
    return best_n, results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(results, best_n, title="PCA Component Sweep", save_path=None):
    """Plot CV accuracy vs n_components from sweep_components output."""
    ns   = [r[0] for r in results]
    accs = [r[1] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, accs, marker="o", linewidth=2, color="steelblue", markersize=5)
    ax.axvline(x=best_n, color="red", linestyle="--", linewidth=1.5,
               label=f"Best n={best_n}")
    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("CV Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.show()


def plot_explained_variance(pca, save_path=None):
    """Cumulative explained variance plot."""
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=2, color="steelblue")
    ax.axhline(y=90, color="red",    linestyle="--", linewidth=1, label="90%")
    ax.axhline(y=95, color="orange", linestyle="--", linewidth=1, label="95%")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("AT&T — PCA Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.show()


def plot_eigenfaces(pca, n_faces=10, save_path=None):
    """Visualise top eigenfaces."""
    eigenfaces = pca.components_[:n_faces].reshape((n_faces, IMG_HEIGHT, IMG_WIDTH))
    fig, axes  = plt.subplots(2, n_faces // 2, figsize=(14, 6))
    fig.suptitle("AT&T — Top Eigenfaces (PCA Components)", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(eigenfaces[i], cmap="gray")
        ax.set_title(f"PC {i+1}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from support.att.load_normalized_data import load_dataset

    print("=" * 55)
    print("  AT&T Preprocessing — sanity check")
    print("=" * 55)

    X, y = load_dataset()
    print(f"\nDataset: {X.shape[0]} images, {len(np.unique(y))} subjects\n")

    X_train_pca, X_test_pca, y_train, y_test, pca = prepare_data(X, y, n_components=50)

    print(f"\nX_train_pca : {X_train_pca.shape}")
    print(f"X_test_pca  : {X_test_pca.shape}")
    print(f"y_train     : {y_train.shape}")
    print(f"y_test      : {y_test.shape}")
