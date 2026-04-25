# model_rf.py
# Random Forest face recognition model — Chapter 2.
# Uses LFW dataset (~1227 images, 19 subjects).
# Pipeline: load data → PCA → RF (auto n_estimators + max_features tuning) → evaluate
#
# Key contributions:
#   - Auto-selects best n_components via sweep (same as KNN)
#   - Tunes n_estimators and max_features on validation set
#   - Feature importance plot (unique to RF — shows which eigenfaces matter most)
#   - 5-fold stratified cross-validation
#   - Confusion matrix saved to support/model/
#
# Run from repo root: python -m support.model_rf
# -------------------------------------------------------

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support.load_data_lfw  import load_dataset
from support.preprocessing  import prepare_data

from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "support", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PCA COMPONENT SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def sweep_components(X, y):
    """
    Auto-select best n_components for RF by sweeping with a fast proxy (RF n=100).
    RF is less sensitive to dimensionality than KNN but still benefits from
    finding the right number of PCA components.

    Returns:
        best_n : int — n_components with highest validation accuracy
    """
    print("Sweeping PCA components (proxy: RF n_estimators=100)...")
    print(f"  {'n_components':>14}  {'Variance':>10}  {'Val Acc':>10}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*10}")

    best_n   = 100
    best_val = 0

    for n_comp in range(10, 210, 10):
        X_tr, X_v, X_te, y_tr, y_v, y_te, pca = prepare_data(
            X, y, n_components=n_comp, verbose=False
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        acc     = accuracy_score(y_v, rf.predict(X_v))
        var     = pca.explained_variance_ratio_.sum() * 100
        marker  = "  ←" if acc > best_val else ""
        print(f"  {n_comp:>14}  {var:>9.1f}%  {acc*100:>9.1f}%{marker}")

        if acc > best_val:
            best_val = acc
            best_n   = n_comp

    print(f"\n  Auto-selected n_components={best_n}  (val_acc={best_val*100:.1f}%)")
    return best_n


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_rf(X_train_pca, X_val_pca, y_train, y_val):
    """
    Grid search over n_estimators and max_features on validation set.

    n_estimators — number of trees. More trees = more stable but slower.
        Too few → high variance. Sweet spot: 200–500 for this dataset size.

    max_features — features considered at each split.
        "sqrt" → sqrt(n_components) — standard for classification
        "log2" → log2(n_components) — fewer features, more randomness
        0.3/0.5 → fraction of features — between the two extremes

    Returns:
        best_n_est    : int
        best_max_feat : str or float
        results       : dict mapping (n_est, max_feat) → val_accuracy
    """
    n_estimators_values = [100, 200, 300, 400, 500 , 600, 700, 800, 900, 1000]
    max_features_values = ["sqrt", "log2", 0.2 ,0.3, 0.4, 0.5 , 0.6, 0.7, 0.8, 0.9]

    best_acc      = 0
    best_n_est    = 200
    best_max_feat = "sqrt"
    results       = {}

    print("  Tuning n_estimators and max_features on validation set...")
    print(f"  {'n_est':>6}  {'max_feat':>10}  {'Val Accuracy':>14}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*14}")

    for n_est in n_estimators_values:
        for max_feat in max_features_values:
            rf = RandomForestClassifier(
                n_estimators=n_est,
                max_features=max_feat,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_pca, y_train)
            val_acc = accuracy_score(y_val, rf.predict(X_val_pca))
            results[(n_est, max_feat)] = val_acc
            marker = "  ←" if val_acc > best_acc else ""
            print(f"  {n_est:>6}  {str(max_feat):>10}  {val_acc*100:>13.2f}%{marker}")

            if val_acc > best_acc:
                best_acc      = val_acc
                best_n_est    = n_est
                best_max_feat = max_feat

    print(f"\n  Best n_estimators={best_n_est}, max_features={best_max_feat}")
    print(f"  Val accuracy: {best_acc*100:.2f}%")
    return best_n_est, best_max_feat, results


# ─────────────────────────────────────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(X_train_pca, X_val_pca, y_train, y_val, best_n_est, best_max_feat):
    """
    5-fold stratified CV on train+val combined using best hyperparameters.
    Reports mean ± std — shows model stability across different splits.
    """
    X_cv = np.vstack([X_train_pca, X_val_pca])
    y_cv = np.concatenate([y_train, y_val])

    rf  = RandomForestClassifier(
        n_estimators=best_n_est,
        max_features=best_max_feat,
        random_state=42,
        n_jobs=-1
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  Running 5-fold stratified CV (n_est={best_n_est}, max_feat={best_max_feat})...")
    scores = cross_val_score(rf, X_cv, y_cv, cv=skf, scoring="accuracy", n_jobs=-1)

    print(f"  Fold accuracies : {[f'{s*100:.1f}%' for s in scores]}")
    print(f"  Mean accuracy   : {scores.mean() * 100:.2f}%")
    print(f"  Std deviation   : ±{scores.std() * 100:.2f}%")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN + EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test):
    """
    Full RF pipeline:
        tune hyperparams → cross-validate → train final on train+val → test once.

    Returns dict:
        model_name, accuracy, y_pred, train_time,
        model, best_n_est, best_max_feat, cv_scores, tune_results
    """

    print("\n" + "─" * 45)
    print("  STEP 1 — Hyperparameter Tuning")
    print("─" * 45)
    best_n_est, best_max_feat, tune_results = tune_rf(
        X_train_pca, X_val_pca, y_train, y_val
    )

    print("\n" + "─" * 45)
    print("  STEP 2 — Cross-Validation")
    print("─" * 45)
    cv_scores = run_cross_validation(
        X_train_pca, X_val_pca, y_train, y_val, best_n_est, best_max_feat
    )

    print("\n" + "─" * 45)
    print("  STEP 3 — Training Final Model")
    print("─" * 45)

    X_trainval = np.vstack([X_train_pca, X_val_pca])
    y_trainval = np.concatenate([y_train, y_val])

    print(f"  Training on {X_trainval.shape[0]} images (train + val)...")
    print(f"  n_estimators={best_n_est}, max_features={best_max_feat}")

    start_time  = time.time()
    final_model = RandomForestClassifier(
        n_estimators=best_n_est,
        max_features=best_max_feat,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_trainval, y_trainval)
    train_time = time.time() - start_time
    print(f"  Done in {train_time:.2f}s")

    print("\n" + "─" * 45)
    print("  STEP 4 — Test Set Evaluation")
    print("─" * 45)

    y_pred   = final_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  Test Accuracy     : {accuracy * 100:.2f}%")
    print(f"  CV Mean ± Std     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  n_estimators      : {best_n_est}")
    print(f"  max_features      : {best_max_feat}")
    print(f"  Train time        : {train_time:.2f}s")
    print(f"  {'─'*40}\n")

    return {
        "model_name"    : "RF",
        "accuracy"      : accuracy,
        "y_pred"        : y_pred,
        "train_time"    : train_time,
        "model"         : final_model,
        "best_n_est"    : best_n_est,
        "best_max_feat" : best_max_feat,
        "cv_scores"     : cv_scores,
        "tune_results"  : tune_results
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_tuning_heatmap(tune_results, best_n_est, best_max_feat, save_path=None):
    """
    Heatmap of validation accuracy across n_estimators and max_features.
    Best cell highlighted with blue border.
    """
    n_est_values  = sorted(set(k[0] for k in tune_results))
    max_feat_vals = list(dict.fromkeys(k[1] for k in tune_results))

    matrix = np.zeros((len(max_feat_vals), len(n_est_values)))
    for i, mf in enumerate(max_feat_vals):
        for j, ne in enumerate(n_est_values):
            matrix[i, j] = tune_results.get((ne, mf), 0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=matrix.min(), vmax=matrix.max())
    plt.colorbar(im, ax=ax, label="Validation Accuracy (%)")

    ax.set_xticks(range(len(n_est_values)))
    ax.set_xticklabels([str(n) for n in n_est_values], fontsize=11)
    ax.set_yticks(range(len(max_feat_vals)))
    ax.set_yticklabels([str(m) for m in max_feat_vals], fontsize=11)
    ax.set_xlabel("n_estimators (Number of Trees)", fontsize=12)
    ax.set_ylabel("max_features", fontsize=12)
    ax.set_title("Random Forest — Hyperparameter Tuning (Validation Accuracy %)", fontsize=13)

    for i in range(len(max_feat_vals)):
        for j in range(len(n_est_values)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if val > (matrix.max() * 0.75) else "black")

    # Highlight best cell
    best_i = list(max_feat_vals).index(best_max_feat)
    best_j = n_est_values.index(best_n_est)
    ax.add_patch(plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="blue", linewidth=3,
        label=f"Best: n={best_n_est}, feat={best_max_feat}"
    ))
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "rf_tuning_heatmap.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_feature_importance(model, n_components, save_path=None):
    """
    Bar chart of top PCA component importances.
    Unique to Random Forest — shows which eigenfaces contributed most
    to the classification decisions. Higher importance = more discriminative.
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:30]  # top 30

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(indices)),
           importances[indices] * 100,
           color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"PC{indices[i]+1}" for i in range(len(indices))],
                       rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("PCA Component", fontsize=12)
    ax.set_ylabel("Feature Importance (%)", fontsize=12)
    ax.set_title("Random Forest — Top 30 PCA Component Importances", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "rf_feature_importance.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_cv_scores(cv_scores, save_path=None):
    """Bar chart of CV fold accuracies with mean line."""
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
    ax.set_title(f"RF — 5-Fold Cross-Validation  (std: ±{accs.std():.1f}%)")
    ax.legend()
    ax.set_ylim(0, min(100, accs.max() + 10))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "rf_cv_scores.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Confusion matrix heatmap."""
    cm     = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("Random Forest — Confusion Matrix", fontsize=14)
    plt.tight_layout()

    save_path = save_path or os.path.join(MODEL_DIR, "rf_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Random Forest Face Recognition — Chapter 2")
    print("=" * 55)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n>>> STEP 1 — Loading dataset\n")
    X, y = load_dataset()

    # ── Step 2: Auto-select best n_components ─────────────────────────────────
    print("\n>>> STEP 2 — PCA Component Sweep\n")
    best_n = sweep_components(X, y)

    # ── Step 3: Preprocess with best n_components ─────────────────────────────
    print("\n>>> STEP 3 — Preprocessing (60/20/20 split + PCA)\n")
    X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca = prepare_data(
        X, y, n_components=best_n
    )

    # ── Step 4: Train and evaluate ────────────────────────────────────────────
    print("\n>>> STEP 4 — Training Random Forest\n")
    results = train_and_evaluate(
        X_train_pca, X_val_pca, X_test_pca,
        y_train, y_val, y_test
    )

    # ── Step 5: Classification report ─────────────────────────────────────────
    print("\n>>> STEP 5 — Classification Report\n")
    print(classification_report(y_test, results["y_pred"], zero_division=0))

    # ── Step 6: Save plots ────────────────────────────────────────────────────
    print("\n>>> STEP 6 — Saving Plots\n")
    plot_tuning_heatmap(
        results["tune_results"], results["best_n_est"], results["best_max_feat"]
    )
    plot_feature_importance(results["model"], best_n)
    plot_cv_scores(results["cv_scores"])
    plot_confusion_matrix(y_test, results["y_pred"])

    # ── Step 7: Save model ────────────────────────────────────────────────────
    print("\n>>> STEP 7 — Saving Model\n")
    joblib.dump(results["model"], os.path.join(MODEL_DIR, "rf_lfw.pkl"))
    joblib.dump(pca,              os.path.join(MODEL_DIR, "pca_rf_lfw.pkl"))
    print(f"  Saved → support/model/rf_lfw.pkl")
    print(f"  Saved → support/model/pca_rf_lfw.pkl")

    print("\n" + "=" * 55)
    print(f"  DONE")
    print(f"  Test Accuracy   : {results['accuracy'] * 100:.2f}%")
    print(f"  CV Mean ± Std   : {results['cv_scores'].mean()*100:.2f}% ± {results['cv_scores'].std()*100:.2f}%")
    print(f"  n_estimators    : {results['best_n_est']}")
    print(f"  max_features    : {results['best_max_feat']}")
    print(f"  PCA components  : {best_n}")
    print("=" * 55)

    return results


if __name__ == "__main__":
    main()