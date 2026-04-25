# model_knn.py
# Chapter 2 — KNN classifier for face recognition.
# Uses PCA-compressed features from preprocessing.py.
# Tunes K (1,3,5,7,9,11), adds cross-validation, plots K tuning curve.
# Follows the shared interface contract so compare_models.py can call it.
# -------------------------------------------------------
# Run from repo root:  python -m support.model_knn

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# ── Imports from groupmate's shared files ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.load_data_lfw import load_dataset
# from support.load_data_lfw_yale import load_dataset (when using mixed dataset)
# from support.load_data_lfw import load_dataset(if using LFW-only dataset)
from support.preprocessing      import prepare_data

# ── Settings ─────────────────────────────────────────────────────────────────
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]   # K values to try during tuning
CV_FOLDS     = 5                       # cross-validation folds
DATASET_MODE = "yale_lfw"             # yale / lfw / yale_lfw

# ── Output directory ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "support", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE CONTRACT FUNCTION
# Called by compare_models.py — must return exactly this dictionary
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test):
    """
    Trains KNN on PCA-compressed features and evaluates on test set.
    Uses the best K found during tuning on training data.

    Args:
        X_train_pca : np.ndarray (n_train, n_components) — from preprocessing.prepare_data()
        X_test_pca  : np.ndarray (n_test,  n_components)
        y_train     : np.ndarray (n_train,)
        y_test      : np.ndarray (n_test,)

    Returns dict with keys:
        model_name  : "KNN"
        accuracy    : float  — test set accuracy
        y_pred      : np.ndarray — predictions on test set
        train_time  : float  — seconds taken to train
        model       : fitted KNeighborsClassifier object
    """

    # ── Step 1: Find best K using cross-validation on training data ───────────
    print("=" * 55)
    print(" KNN — Tuning K with cross-validation")
    print("=" * 55)

    cv_scores = []
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        scores = cross_val_score(knn, X_train_pca, y_train, cv=CV_FOLDS, scoring="accuracy")
        mean_score = scores.mean()
        cv_scores.append(mean_score)
        print(f"  K={k:2d}  →  CV accuracy: {mean_score * 100:.2f}%  (±{scores.std() * 100:.2f}%)")

    best_k   = K_VALUES[np.argmax(cv_scores)]
    best_cv  = max(cv_scores)
    print(f"\n  Best K : {best_k}  (CV accuracy: {best_cv * 100:.2f}%)")

    # ── Step 2: Save K tuning plot ────────────────────────────────────────────
    _plot_k_curve(K_VALUES, cv_scores, best_k)

    # ── Step 3: Train final model with best K on full training set ────────────
    print("\n" + "=" * 55)
    print(f" KNN — Training final model  (K={best_k})")
    print("=" * 55)

    start_time = time.time()
    knn_final  = KNeighborsClassifier(n_neighbors=best_k, metric="cosine")
    knn_final.fit(X_train_pca, y_train)
    train_time = time.time() - start_time

    print(f"  Training complete ✓  ({train_time:.2f}s)")

    # ── Step 4: Evaluate on test set ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" KNN — Test set evaluation")
    print("=" * 55)

    y_pred   = knn_final.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Best K      : {best_k}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")

    print("\n" + "=" * 55)
    print(f"  KNN DONE  —  K={best_k}  Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55 + "\n")

    # ── Return standard dictionary (required by compare_models.py) ────────────
    return {
        "model_name" : "KNN",
        "accuracy"   : accuracy,
        "y_pred"     : y_pred,
        "train_time" : train_time,
        "model"      : knn_final,
        "best_k"     : best_k,          # KNN-specific extra info
        "cv_scores"  : cv_scores,       # KNN-specific extra info
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_k_curve(k_values, cv_scores, best_k):
    """
    Plot accuracy vs K — the KNN's unique visual contribution.
    Shows bias-variance tradeoff:
      Low K  → overfits (high variance)
      High K → underfits (high bias)
      Sweet spot → peak of the curve
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_values, [s * 100 for s in cv_scores],
            marker="o", linewidth=2, color="steelblue", label="CV Accuracy")

    # Highlight best K
    best_score = cv_scores[k_values.index(best_k)]
    ax.scatter([best_k], [best_score * 100],
               color="red", s=120, zorder=5, label=f"Best K={best_k}")

    ax.set_xlabel("K (Number of Neighbours)", fontsize=12)
    ax.set_ylabel("Cross-Validation Accuracy (%)", fontsize=12)
    ax.set_title("KNN — Accuracy vs K  (Bias-Variance Tradeoff)", fontsize=13)
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, "knn_k_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Saved → support/model/knn_k_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f" Loading dataset: {DATASET_MODE}")
    print("=" * 55)
    X, y = load_dataset()

    # ── Auto-select best n_components ─────────────────────────────────────────
    print("Sweeping PCA components...")
    best_n   = 50
    best_val = 0

    for n_comp in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
        X_tr, X_v, X_te, y_tr, y_v, y_te, _ = prepare_data(X, y, n_components=n_comp, verbose=False)
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(X_tr, y_tr)
        acc = accuracy_score(y_v, knn.predict(X_v))
        print(f"  n_components={n_comp:>4}  K=1  val_acc={acc*100:.1f}%")

        if acc > best_val:       # ← track the best
            best_val = acc
            best_n   = n_comp

    print(f"\n  Auto-selected n_components={best_n}  (val_acc={best_val*100:.1f}%)")

    # ── Preprocess with best n_components ─────────────────────────────────────
    print("=" * 55)
    print(f" Preprocessing  (60/20/20 split + PCA, n={best_n})")
    print("=" * 55)
    X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca = prepare_data(
        X, y, n_components=best_n   # ← uses auto-selected value
    )

    # ── Train and evaluate ────────────────────────────────────────────────────
    results = train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test)

    print(f"  Final result  →  K={results['best_k']}  |  Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"  Training time →  {results['train_time']:.2f}s")
    print(f"  Plot saved    →  support/model/knn_k_curve.png")