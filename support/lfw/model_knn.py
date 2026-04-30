# model_knn.py
# KNN face recognition — LFW dataset, Chapter 2.
# Location: support/lfw/model_knn.py
# -------------------------------------------------------
# Run from repo root: python -m support.lfw.model_knn

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from support.lfw.load_data_lfw import load_dataset
from support.lfw.preprocessing import prepare_data

K_VALUES  = list(range(1, 21))
CV_FOLDS  = 5

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "lfw", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test):
    print("=" * 55)
    print(" KNN — Tuning K with cross-validation")
    print("=" * 55)

    cv_scores = []
    for k in K_VALUES:
        knn    = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        scores = cross_val_score(knn, X_train_pca, y_train, cv=CV_FOLDS, scoring="accuracy")
        mean_score = scores.mean()
        cv_scores.append(mean_score)
        print(f"  K={k:2d}  →  CV accuracy: {mean_score*100:.2f}%  (±{scores.std()*100:.2f}%)")

    best_k  = K_VALUES[np.argmax(cv_scores)]
    best_cv = max(cv_scores)
    print(f"\n  Best K : {best_k}  (CV accuracy: {best_cv*100:.2f}%)")
    _plot_k_curve(K_VALUES, cv_scores, best_k)

    start_time  = time.time()
    knn_final   = KNeighborsClassifier(n_neighbors=best_k, metric="cosine")
    knn_final.fit(X_train_pca, y_train)
    train_time  = time.time() - start_time

    y_pred   = knn_final.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy : {accuracy*100:.2f}%")

    return {
        "model_name" : "KNN",
        "accuracy"   : accuracy,
        "y_pred"     : y_pred,
        "train_time" : train_time,
        "model"      : knn_final,
        "best_k"     : best_k,
        "cv_scores"  : np.array(cv_scores),
    }


def _plot_k_curve(k_values, cv_scores, best_k):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, [s*100 for s in cv_scores], marker="o", linewidth=2,
            color="steelblue", label="CV Accuracy")
    ax.scatter([best_k], [cv_scores[k_values.index(best_k)]*100],
               color="red", s=120, zorder=5, label=f"Best K={best_k}")
    ax.set_xlabel("K (Number of Neighbours)", fontsize=12)
    ax.set_ylabel("Cross-Validation Accuracy (%)", fontsize=12)
    ax.set_title("LFW KNN — Accuracy vs K  (Bias-Variance Tradeoff)", fontsize=13)
    ax.set_xticks(k_values)
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "knn_k_curve.png")
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"\n  Saved → {plot_path}")


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm     = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("LFW KNN — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "knn_confusion_matrix.png")
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved → {save_path}")


def main():
    print("=" * 55)
    print("  KNN Face Recognition — Chapter 2 (LFW)")
    print("=" * 55)

    print("\n>>> STEP 1 — Loading dataset\n")
    X, y = load_dataset()

    print("\n>>> STEP 2 — PCA Component Sweep\n")
    best_n   = 150
    best_val = 0
    print("Sweeping PCA components...")
    for n_comp in range(10, 210, 10):
        X_tr, X_v, X_te, y_tr, y_v, y_te, _ = prepare_data(X, y, n_components=n_comp, verbose=False)
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(X_tr, y_tr)
        acc = accuracy_score(y_v, knn.predict(X_v))
        print(f"  n_components={n_comp:>4}  K=1  val_acc={acc*100:.1f}%")
        if acc > best_val:
            best_val = acc
            best_n   = n_comp
    print(f"\n  Auto-selected n_components={best_n}  (val_acc={best_val*100:.1f}%)")

    print(f"\n>>> STEP 3 — Preprocessing (n_components={best_n})\n")
    X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca = prepare_data(
        X, y, n_components=best_n
    )

    print("\n>>> STEP 4 — Training KNN\n")
    results = train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test)

    print("\n>>> STEP 5 — Classification Report\n")
    print(classification_report(y_test, results["y_pred"], zero_division=0))

    print("\n>>> STEP 6 — Saving Confusion Matrix\n")
    plot_confusion_matrix(y_test, results["y_pred"])

    print("\n" + "=" * 55)
    print(f"  DONE — Test Accuracy: {results['accuracy']*100:.2f}%")
    print("=" * 55)

    # ── Save results to JSON ──────────────────────────────────────────────────
    json_data = {
        "dataset"      : "LFW",
        "model"        : "KNN",
        "accuracy"     : results["accuracy"],
        "precision"    : float(precision_score(y_test, results["y_pred"], average="macro", zero_division=0)),
        "recall"       : float(recall_score   (y_test, results["y_pred"], average="macro", zero_division=0)),
        "f1"           : float(f1_score       (y_test, results["y_pred"], average="macro", zero_division=0)),
        "train_time"   : results["train_time"],
        "n_components" : best_n,
        "params"       : f"K={results['best_k']}, metric=cosine",
        "cv_mean"      : float(results["cv_scores"].mean() * 100),
        "cv_std"       : float(results["cv_scores"].std()  * 100),
    }
    json_path = os.path.join(MODEL_DIR, "knn_lfw_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Results saved → {json_path}")

    return {
        **results,
        "y_test"       : y_test,
        "n_components" : best_n,
        "dataset"      : "LFW",
        "params"       : f"K={results['best_k']}, metric=cosine",
    }


if __name__ == "__main__":
    main()