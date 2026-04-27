# model_knn_att.py
# KNN face recognition on AT&T (ORL) dataset — Chapter 1 baseline.
# 40 subjects, 10 images each, 80/20 split = 8 train / 2 test per person.
#
# Pipeline:
#   load AT&T → PCA sweep (CV on train) → K tuning (CV) → evaluate
#
# Metric: cosine (better than euclidean for PCA face features)
# All tuning via 5-fold stratified CV on training set.
#
# Run from repo root: python -m support.att.model_knn_att
# -------------------------------------------------------

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from support.att.load_normalized_data import load_dataset
from support.att.preprocessing_att    import (
    prepare_data, sweep_components, plot_sweep
)

from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "att", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15]


# ─────────────────────────────────────────────────────────────────────────────
# K TUNING VIA CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def tune_k(X_train_pca, y_train):
    """
    Find best K via 5-fold stratified CV on training set.
    Uses cosine distance — better than euclidean for PCA face vectors.

    Returns:
        best_k    : int
        cv_scores : list of mean CV accuracies per K
    """
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_acc  = 0
    best_k    = 1

    print("  Tuning K via 5-fold CV on training set...")
    print(f"  {'K':>4}  {'CV Accuracy':>13}  {'Std':>8}")
    print(f"  {'─'*4}  {'─'*13}  {'─'*8}")

    for k in K_VALUES:
        knn    = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        scores = cross_val_score(knn, X_train_pca, y_train,
                                 cv=skf, scoring="accuracy", n_jobs=-1)
        acc    = scores.mean()
        cv_scores.append(acc)
        marker = "  ←" if acc > best_acc else ""
        print(f"  {k:>4}  {acc*100:>12.2f}%  ±{scores.std()*100:>5.2f}%{marker}")

        if acc > best_acc:
            best_acc = acc
            best_k   = k

    print(f"\n  Best K={best_k}  (CV acc={best_acc*100:.2f}%)")
    return best_k, cv_scores


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_k_curve(k_values, cv_scores, best_k, save_path=None):
    """CV accuracy vs K — shows bias-variance tradeoff."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, [s * 100 for s in cv_scores],
            marker="o", linewidth=2, color="steelblue", label="CV Accuracy")

    best_score = cv_scores[k_values.index(best_k)]
    ax.scatter([best_k], [best_score * 100],
               color="red", s=120, zorder=5, label=f"Best K={best_k}")

    ax.set_xlabel("K (Number of Neighbours)", fontsize=12)
    ax.set_ylabel("5-Fold CV Accuracy (%)", fontsize=12)
    ax.set_title("AT&T KNN — Accuracy vs K  (Bias-Variance Tradeoff)", fontsize=13)
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "knn_att_k_curve.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm     = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("AT&T KNN — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "knn_att_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  KNN — AT&T Dataset  (Chapter 1 Baseline)")
    print("=" * 55)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n>>> STEP 1 — Loading AT&T dataset\n")
    X, y = load_dataset()
    print(f"  {X.shape[0]} images | {len(np.unique(y))} subjects | "
          f"{X.shape[1]} features")

    # ── Step 2: PCA sweep ─────────────────────────────────────────────────────
    print("\n>>> STEP 2 — PCA Component Sweep\n")
    proxy = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    best_n, sweep_results = sweep_components(
        X, y, proxy_clf=proxy, label="KNN K=1 cosine",
        whiten=False, step=5, max_n=150
    )

    # ── Step 3: Preprocess with best n ────────────────────────────────────────
    print("\n>>> STEP 3 — Preprocessing\n")
    X_train_pca, X_test_pca, y_train, y_test, pca = prepare_data(
        X, y, n_components=best_n, whiten=False
    )

    # ── Step 4: Tune K ────────────────────────────────────────────────────────
    print("\n>>> STEP 4 — Tuning K\n")
    best_k, cv_scores = tune_k(X_train_pca, y_train)

    # ── Step 5: Train final model ─────────────────────────────────────────────
    print("\n>>> STEP 5 — Training Final Model\n")
    start       = time.time()
    final_model = KNeighborsClassifier(n_neighbors=best_k, metric="cosine")
    final_model.fit(X_train_pca, y_train)
    train_time  = time.time() - start
    print(f"  Done in {train_time:.4f}s")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n>>> STEP 6 — Test Set Evaluation\n")
    y_pred   = final_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  {'─'*40}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print(f"  Best K        : {best_k}")
    print(f"  PCA components: {best_n}")
    print(f"  Train time    : {train_time:.4f}s")
    print(f"  {'─'*40}")

    # ── Step 7: Classification report ─────────────────────────────────────────
    print("\n>>> STEP 7 — Classification Report\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Step 8: Plots ─────────────────────────────────────────────────────────
    print("\n>>> STEP 8 — Saving Plots\n")
    plot_sweep(sweep_results, best_n,
               title="AT&T KNN — PCA Component Sweep (5-Fold CV)",
               save_path=os.path.join(MODEL_DIR, "knn_att_pca_sweep.png"))
    plot_k_curve(K_VALUES, cv_scores, best_k)
    plot_confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"  DONE — Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55)

    return {"accuracy": accuracy, "model": final_model, "pca": pca,
            "best_k": best_k, "best_n": best_n}


if __name__ == "__main__":
    main()
