# model_svm.py
# SVM face recognition model — Chapter 2.
# Uses Yale + LFW dataset (~1800 images, 90 subjects).
# Pipeline: load data → PCA (100 components) → SVM (RBF kernel) → evaluate
#
# Validation set is used to tune the C parameter.
# Test set is touched exactly once at the end for final accuracy.
#
# Unique contributions:
#   - C parameter tuning with validation accuracy curve
#   - 5-fold cross-validation for stability analysis (mean ± std)
#   - Confusion matrix saved to support/model/
#
# Run from repo root: python -m support.model_svm
# -------------------------------------------------------

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support.load_data_lfw_yale     import load_dataset
from support.preprocessing import prepare_data

from sklearn.svm             import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib

# ── Output directory ──────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "support", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# C PARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_c(X_train_pca, X_val_pca, y_train, y_val):
    """
    Try multiple C values and pick the one with the best validation accuracy.
    C controls the tradeoff between a smooth boundary (low C = underfitting)
    and fitting every training point (high C = overfitting).

    Returns:
        best_C    : float
        c_results : dict mapping C → val_accuracy
    """
    C_values  = [0.1, 1, 10, 50, 100, 500, 1000]
    c_results = {}

    print("  Tuning C parameter on validation set...")
    print(f"  {'C':>8}  {'Val Accuracy':>14}")
    print(f"  {'─'*8}  {'─'*14}")

    for C in C_values:
        clf = SVC(kernel="rbf", C=C, gamma="scale", random_state=42)
        clf.fit(X_train_pca, y_train)
        val_acc = accuracy_score(y_val, clf.predict(X_val_pca))
        c_results[C] = val_acc
        print(f"  {C:>8}  {val_acc * 100:>13.2f}%")

    best_C = max(c_results, key=c_results.get)
    print(f"\n  Best C: {best_C}  (val accuracy: {c_results[best_C] * 100:.2f}%)")
    return best_C, c_results


# ─────────────────────────────────────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(X_train_pca, X_val_pca, y_train, y_val, best_C):
    """
    5-fold CV on train+val combined using the best C.
    Reports mean ± std accuracy — shows how stable the model is
    across different random splits.

    Returns:
        scores : np.ndarray of 5 fold accuracies
    """
    X_cv = np.vstack([X_train_pca, X_val_pca])
    y_cv = np.concatenate([y_train, y_val])

    clf = SVC(kernel="rbf", C=best_C, gamma="scale", random_state=42)

    print(f"\n  Running 5-fold cross-validation (C={best_C})...")
    scores = cross_val_score(clf, X_cv, y_cv, cv=5, scoring="accuracy", n_jobs=-1)

    print(f"  Fold accuracies : {[f'{s*100:.1f}%' for s in scores]}")
    print(f"  Mean accuracy   : {scores.mean() * 100:.2f}%")
    print(f"  Std deviation   : ±{scores.std() * 100:.2f}%")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN + EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test):
    """
    Full SVM pipeline: tune C → cross-validate → train final → evaluate on test.
    Interface matches model_knn.py and model_rf.py for compare_models.py.

    Returns dict:
        model_name  : "SVM"
        accuracy    : float 0-1, final test accuracy
        y_pred      : np.ndarray of test predictions
        train_time  : float seconds
        model       : fitted SVC
        best_C      : float
        cv_scores   : np.ndarray of 5 CV fold accuracies
        c_results   : dict of C → val_accuracy
    """

    # ── Tune C ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  STEP 1 — C Parameter Tuning")
    print("─" * 45)
    best_C, c_results = tune_c(X_train_pca, X_val_pca, y_train, y_val)

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  STEP 2 — Cross-Validation")
    print("─" * 45)
    cv_scores = run_cross_validation(X_train_pca, X_val_pca, y_train, y_val, best_C)

    # ── Train final model on train + val ─────────────────────────────────────
    # Once C is selected, train on all non-test data for maximum coverage
    print("\n" + "─" * 45)
    print("  STEP 3 — Training Final Model")
    print("─" * 45)

    X_trainval = np.vstack([X_train_pca, X_val_pca])
    y_trainval = np.concatenate([y_train, y_val])

    print(f"  Training on {X_trainval.shape[0]} images (train + val)...")
    print(f"  C={best_C}, kernel=rbf, gamma=scale")

    start_time  = time.time()
    final_model = SVC(kernel="rbf", C=best_C, gamma="scale", random_state=42)
    final_model.fit(X_trainval, y_trainval)
    train_time  = time.time() - start_time
    print(f"  Done in {train_time:.2f}s")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  STEP 4 — Test Set Evaluation")
    print("─" * 45)

    y_pred   = final_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  Test Accuracy   : {accuracy * 100:.2f}%")
    print(f"  CV Mean ± Std   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  Train time      : {train_time:.2f}s")
    print(f"  {'─'*40}\n")

    return {
        "model_name" : "SVM",
        "accuracy"   : accuracy,
        "y_pred"     : y_pred,
        "train_time" : train_time,
        "model"      : final_model,
        "best_C"     : best_C,
        "cv_scores"  : cv_scores,
        "c_results"  : c_results
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_c_tuning(c_results, best_C, save_path=None):
    """
    Plot validation accuracy vs C value.
    Shows where accuracy peaks and why that C was chosen.
    """
    C_values = list(c_results.keys())
    accs     = [c_results[c] * 100 for c in C_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(C_values)), accs, marker="o", linewidth=2,
            markersize=8, color="steelblue")
    ax.axvline(x=C_values.index(best_C), color="red", linestyle="--",
               linewidth=1.5, label=f"Best C={best_C}")
    ax.set_xticks(range(len(C_values)))
    ax.set_xticklabels([str(c) for c in C_values])
    ax.set_xlabel("C Value")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("SVM — C Parameter Tuning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "svm_c_tuning.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_cv_scores(cv_scores, save_path=None):
    """
    Bar chart of cross-validation fold accuracies with mean line.
    Low variance across folds = model is stable and generalizable.
    """
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    accs  = cv_scores * 100
    mean  = accs.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(folds, accs, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axhline(y=mean, color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean:.1f}%")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"SVM — 5-Fold Cross-Validation  (std: ±{accs.std():.1f}%)")
    ax.legend()
    ax.set_ylim(0, min(100, accs.max() + 10))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "svm_cv_scores.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Confusion matrix heatmap. Diagonal = correct, off-diagonal = mistakes."""
    cm     = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=90)
    ax.set_title("SVM — Confusion Matrix", fontsize=14)
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "svm_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SVM Face Recognition — Chapter 2")
    print("=" * 55)

    print("\n>>> STEP 1 — Loading dataset\n")
    X, y = load_dataset()

    print("\n>>> STEP 2 — Preprocessing (PCA + 60/20/20 split)\n")
    X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca = prepare_data(X, y)

    print("\n>>> STEP 3 — Training SVM\n")
    results = train_and_evaluate(
        X_train_pca, X_val_pca, X_test_pca,
        y_train, y_val, y_test
    )

    print("\n>>> STEP 4 — Classification Report\n")
    print(classification_report(y_test, results["y_pred"]))

    print("\n>>> STEP 5 — Saving Plots\n")
    plot_c_tuning(results["c_results"], results["best_C"])
    plot_cv_scores(results["cv_scores"])
    plot_confusion_matrix(y_test, results["y_pred"])

    print("\n>>> STEP 6 — Saving Model\n")
    joblib.dump(results["model"], os.path.join(MODEL_DIR, "svm_yale_lfw.pkl"))
    joblib.dump(pca,              os.path.join(MODEL_DIR, "pca_yale_lfw.pkl"))
    print(f"  Saved → support/model/svm_yale_lfw.pkl")
    print(f"  Saved → support/model/pca_yale_lfw.pkl")

    print("\n" + "=" * 55)
    print(f"  DONE")
    print(f"  Test Accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"  CV Mean ± Std : {results['cv_scores'].mean()*100:.2f}% ± {results['cv_scores'].std()*100:.2f}%")
    print(f"  Best C        : {results['best_C']}")
    print("=" * 55)

    return results


if __name__ == "__main__":
    main()