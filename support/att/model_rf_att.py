# model_rf_att.py
# Random Forest face recognition on AT&T (ORL) dataset — Chapter 1 baseline.
# 40 subjects, 10 images each, 80/20 split = 8 train / 2 test per person.
#
# Pipeline:
#   load AT&T → PCA sweep (CV on train) → n_estimators + max_features tuning → evaluate
#
# All tuning via 5-fold stratified CV on training set.
# Feature importance plot unique to RF — shows which eigenfaces matter most.
#
# Run from repo root: python -m support.att.model_rf_att
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

from sklearn.ensemble        import RandomForestClassifier
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
# HYPERPARAMETER TUNING VIA CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def tune_rf(X_train_pca, y_train):
    """
    Joint n_estimators + max_features grid search via 5-fold CV.
    No validation set — AT&T too small for holdout tuning.

    Returns:
        best_n_est    : int
        best_max_feat : str or float
        results       : dict (n_est, max_feat) → cv_accuracy
    """
    n_estimators_values = [100, 200, 300, 500, 700]
    max_features_values = ["sqrt", "log2", 0.2, 0.3, 0.5]

    best_acc      = 0
    best_n_est    = 300
    best_max_feat = "sqrt"
    results       = {}
    skf           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("  Tuning n_estimators and max_features via 5-fold CV...")
    print(f"  {'n_est':>6}  {'max_feat':>10}  {'CV Accuracy':>13}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*13}")

    for n_est in n_estimators_values:
        for max_feat in max_features_values:
            rf     = RandomForestClassifier(
                n_estimators=n_est, max_features=max_feat,
                random_state=42, n_jobs=-1
            )
            scores = cross_val_score(rf, X_train_pca, y_train,
                                     cv=skf, scoring="accuracy", n_jobs=-1)
            acc    = scores.mean()
            results[(n_est, max_feat)] = acc
            marker = "  ←" if acc > best_acc else ""
            print(f"  {n_est:>6}  {str(max_feat):>10}  {acc*100:>12.2f}%{marker}")

            if acc > best_acc:
                best_acc      = acc
                best_n_est    = n_est
                best_max_feat = max_feat

    print(f"\n  Best n_estimators={best_n_est}, max_features={best_max_feat}")
    print(f"  CV accuracy: {best_acc*100:.2f}%")
    return best_n_est, best_max_feat, results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_tuning_heatmap(results, best_n_est, best_max_feat, save_path=None):
    """Heatmap of CV accuracy across n_estimators and max_features."""
    n_est_vals  = sorted(set(k[0] for k in results))
    max_feat_vals = list(dict.fromkeys(k[1] for k in results))

    matrix = np.zeros((len(max_feat_vals), len(n_est_vals)))
    for i, mf in enumerate(max_feat_vals):
        for j, ne in enumerate(n_est_vals):
            matrix[i, j] = results.get((ne, mf), 0) * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=matrix.min(), vmax=matrix.max())
    plt.colorbar(im, ax=ax, label="CV Accuracy (%)")

    ax.set_xticks(range(len(n_est_vals)))
    ax.set_xticklabels([str(n) for n in n_est_vals], fontsize=11)
    ax.set_yticks(range(len(max_feat_vals)))
    ax.set_yticklabels([str(m) for m in max_feat_vals], fontsize=11)
    ax.set_xlabel("n_estimators (Number of Trees)", fontsize=12)
    ax.set_ylabel("max_features", fontsize=12)
    ax.set_title("AT&T RF — Hyperparameter Tuning (5-Fold CV Accuracy %)", fontsize=13)

    for i in range(len(max_feat_vals)):
        for j in range(len(n_est_vals)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if val > matrix.max() * 0.75 else "black")

    best_i = list(max_feat_vals).index(best_max_feat)
    best_j = n_est_vals.index(best_n_est)
    ax.add_patch(plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="blue", linewidth=3,
        label=f"Best: n={best_n_est}, feat={best_max_feat}"
    ))
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "rf_att_tuning_heatmap.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_feature_importance(model, save_path=None):
    """Top 30 PCA component importances — unique RF contribution."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:30]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(indices)), importances[indices] * 100,
           color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"PC{indices[i]+1}" for i in range(len(indices))],
                       rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("PCA Component", fontsize=12)
    ax.set_ylabel("Feature Importance (%)", fontsize=12)
    ax.set_title("AT&T RF — Top 30 PCA Component Importances", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "rf_att_feature_importance.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm     = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("AT&T RF — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "rf_att_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Random Forest — AT&T Dataset  (Chapter 1 Baseline)")
    print("=" * 55)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n>>> STEP 1 — Loading AT&T dataset\n")
    X, y = load_dataset()
    print(f"  {X.shape[0]} images | {len(np.unique(y))} subjects | "
          f"{X.shape[1]} features")

    # ── Step 2: PCA sweep ─────────────────────────────────────────────────────
    print("\n>>> STEP 2 — PCA Component Sweep\n")
    proxy = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    best_n, sweep_results = sweep_components(
        X, y, proxy_clf=proxy, label="RF n=100 proxy",
        whiten=False, step=5, max_n=150
    )

    # ── Step 3: Preprocess with best n ────────────────────────────────────────
    print("\n>>> STEP 3 — Preprocessing\n")
    X_train_pca, X_test_pca, y_train, y_test, pca = prepare_data(
        X, y, n_components=best_n, whiten=False
    )

    # ── Step 4: Tune hyperparameters ──────────────────────────────────────────
    print("\n>>> STEP 4 — Tuning Hyperparameters\n")
    best_n_est, best_max_feat, tune_results = tune_rf(X_train_pca, y_train)

    # ── Step 5: Train final model ─────────────────────────────────────────────
    print("\n>>> STEP 5 — Training Final Model\n")
    start       = time.time()
    final_model = RandomForestClassifier(
        n_estimators=best_n_est, max_features=best_max_feat,
        random_state=42, n_jobs=-1
    )
    final_model.fit(X_train_pca, y_train)
    train_time  = time.time() - start
    print(f"  Done in {train_time:.2f}s")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n>>> STEP 6 — Test Set Evaluation\n")
    y_pred   = final_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  {'─'*40}")
    print(f"  Test Accuracy  : {accuracy * 100:.2f}%")
    print(f"  n_estimators   : {best_n_est}")
    print(f"  max_features   : {best_max_feat}")
    print(f"  PCA components : {best_n}")
    print(f"  Train time     : {train_time:.2f}s")
    print(f"  {'─'*40}")

    # ── Step 7: Classification report ─────────────────────────────────────────
    print("\n>>> STEP 7 — Classification Report\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Step 8: Plots ─────────────────────────────────────────────────────────
    print("\n>>> STEP 8 — Saving Plots\n")
    plot_sweep(sweep_results, best_n,
               title="AT&T RF — PCA Component Sweep (5-Fold CV)",
               save_path=os.path.join(MODEL_DIR, "rf_att_pca_sweep.png"))
    plot_tuning_heatmap(tune_results, best_n_est, best_max_feat)
    plot_feature_importance(final_model)
    plot_confusion_matrix(y_test, y_pred)

    # ── Step 9: Save model ────────────────────────────────────────────────────
    print("\n>>> STEP 9 — Saving Model\n")
    joblib.dump(final_model, os.path.join(MODEL_DIR, "rf_att.pkl"))
    joblib.dump(pca,         os.path.join(MODEL_DIR, "pca_att_rf.pkl"))
    print(f"  Saved → support/svm_baseline/model/rf_att.pkl")

    print("\n" + "=" * 55)
    print(f"  DONE — Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55)

    return {"accuracy": accuracy, "model": final_model, "pca": pca,
            "best_n_est": best_n_est, "best_max_feat": best_max_feat, "best_n": best_n}


if __name__ == "__main__":
    main()
