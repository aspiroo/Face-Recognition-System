# model_svm_att.py
# SVM face recognition on AT&T (ORL) dataset — Chapter 1 baseline.
# 40 subjects, 10 images each, 80/20 split = 8 train / 2 test per person.
#
# Pipeline:
#   load AT&T → PCA sweep (CV on train) → joint C+gamma tuning (CV) → evaluate
#
# No validation set — AT&T is too small for three-way split.
# All tuning done via 5-fold stratified CV on training set.
#
# Run from repo root: python -m support.att.model_svm_att
# -------------------------------------------------------

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from support.att.load_normalized_data import load_dataset
from support.att.preprocessing_att    import (
    prepare_data, sweep_components,
    plot_sweep, plot_eigenfaces
)

from sklearn.svm             import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "att", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# C + GAMMA TUNING VIA CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def tune_svm(X_train_pca, y_train):
    """
    Joint C + gamma grid search using 5-fold CV on training set.
    No validation set — AT&T is too small (8 train images per person).

    Returns:
        best_C     : float or int
        best_gamma : float or str
        results    : dict (C, gamma) → cv_accuracy
    """
    C_values     = [0.1, 1, 5, 10, 50, 100]
    gamma_values = ["scale", "auto", 0.0001, 0.001, 0.01, 0.1]

    best_acc   = 0
    best_C     = 10
    best_gamma = "scale"
    results    = {}
    skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("  Tuning C and gamma via 5-fold CV on training set...")
    print(f"  {'C':>6}  {'Gamma':>8}  {'CV Accuracy':>13}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*13}")

    for C in C_values:
        for gamma in gamma_values:
            clf    = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
            scores = cross_val_score(clf, X_train_pca, y_train,
                                     cv=skf, scoring="accuracy", n_jobs=-1)
            acc    = scores.mean()
            results[(C, gamma)] = acc
            marker = "  ←" if acc > best_acc else ""
            print(f"  {C:>6}  {str(gamma):>8}  {acc*100:>12.2f}%{marker}")

            if acc > best_acc:
                best_acc   = acc
                best_C     = C
                best_gamma = gamma

    print(f"\n  Best C={best_C}, gamma={best_gamma}  (CV acc={best_acc*100:.2f}%)")
    return best_C, best_gamma, results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_c_gamma_heatmap(results, best_C, best_gamma, save_path=None):
    """Heatmap of CV accuracy across C and gamma — same style as LFW SVM."""
    C_values     = sorted(set(k[0] for k in results))
    gamma_values = list(dict.fromkeys(k[1] for k in results))

    matrix = np.zeros((len(gamma_values), len(C_values)))
    for i, gamma in enumerate(gamma_values):
        for j, C in enumerate(C_values):
            matrix[i, j] = results.get((C, gamma), 0) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=matrix.min(), vmax=matrix.max())
    plt.colorbar(im, ax=ax, label="CV Accuracy (%)")

    ax.set_xticks(range(len(C_values)))
    ax.set_xticklabels([str(c) for c in C_values], fontsize=11)
    ax.set_yticks(range(len(gamma_values)))
    ax.set_yticklabels([str(g) for g in gamma_values], fontsize=11)
    ax.set_xlabel("C Value", fontsize=12)
    ax.set_ylabel("Gamma", fontsize=12)
    ax.set_title("AT&T SVM — Joint C + Gamma Tuning (5-Fold CV Accuracy %)", fontsize=13)

    for i in range(len(gamma_values)):
        for j in range(len(C_values)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if val > matrix.max() * 0.75 else "black")

    best_i = list(gamma_values).index(best_gamma)
    best_j = C_values.index(best_C)
    ax.add_patch(plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="blue", linewidth=3,
        label=f"Best: C={best_C}, γ={best_gamma}"
    ))
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "svm_att_c_gamma_heatmap.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm     = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("AT&T SVM — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "svm_att_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SVM — AT&T Dataset  (Chapter 1 Baseline)")
    print("=" * 55)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n>>> STEP 1 — Loading AT&T dataset\n")
    X, y = load_dataset()
    print(f"  {X.shape[0]} images | {len(np.unique(y))} subjects | "
          f"{X.shape[1]} features")

    # ── Step 2: PCA sweep ─────────────────────────────────────────────────────
    print("\n>>> STEP 2 — PCA Component Sweep\n")
    proxy = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    best_n, sweep_results = sweep_components(
        X, y, proxy_clf=proxy, label="SVM proxy C=10",
        whiten=True, step=5, max_n=150
    )

    # ── Step 3: Preprocess with best n ────────────────────────────────────────
    print("\n>>> STEP 3 — Preprocessing\n")
    X_train_pca, X_test_pca, y_train, y_test, pca = prepare_data(
        X, y, n_components=best_n, whiten=True
    )

    # ── Step 4: Tune C + gamma ────────────────────────────────────────────────
    print("\n>>> STEP 4 — Tuning C + Gamma\n")
    best_C, best_gamma, tune_results = tune_svm(X_train_pca, y_train)

    # ── Step 5: Train final model ─────────────────────────────────────────────
    print("\n>>> STEP 5 — Training Final Model\n")
    start      = time.time()
    final_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    final_model.fit(X_train_pca, y_train)
    train_time = time.time() - start
    print(f"  Done in {train_time:.2f}s")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n>>> STEP 6 — Test Set Evaluation\n")
    y_pred   = final_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  {'─'*40}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print(f"  Best C        : {best_C}")
    print(f"  Best gamma    : {best_gamma}")
    print(f"  PCA components: {best_n}")
    print(f"  Train time    : {train_time:.2f}s")
    print(f"  {'─'*40}")

    # ── Step 7: Classification report ─────────────────────────────────────────
    print("\n>>> STEP 7 — Classification Report\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Step 8: Plots ─────────────────────────────────────────────────────────
    print("\n>>> STEP 8 — Saving Plots\n")
    plot_sweep(sweep_results, best_n,
               title="AT&T SVM — PCA Component Sweep (5-Fold CV)",
               save_path=os.path.join(MODEL_DIR, "svm_att_pca_sweep.png"))
    plot_c_gamma_heatmap(tune_results, best_C, best_gamma)
    plot_confusion_matrix(y_test, y_pred)
    plot_eigenfaces(pca, save_path=os.path.join(MODEL_DIR, "svm_att_eigenfaces.png"))

    # ── Step 9: Save model ────────────────────────────────────────────────────
    print("\n>>> STEP 9 — Saving Model\n")
    joblib.dump(final_model, os.path.join(MODEL_DIR, "svm_att.pkl"))
    joblib.dump(pca,         os.path.join(MODEL_DIR, "pca_att_svm.pkl"))
    print(f"  Saved → support/svm_baseline/model/svm_att.pkl")

    print("\n" + "=" * 55)
    print(f"  DONE — Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55)

    return {"accuracy": accuracy, "model": final_model, "pca": pca,
            "best_C": best_C, "best_gamma": best_gamma, "best_n": best_n}


if __name__ == "__main__":
    main()
