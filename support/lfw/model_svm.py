# model_svm.py
# SVM face recognition — LFW dataset, Chapter 2.
# Location: support/lfw/model_svm.py
# -------------------------------------------------------
# Run from repo root: python -m support.lfw.model_svm

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from support.lfw.load_data_lfw  import load_dataset
from support.lfw.preprocessing  import prepare_data

from sklearn.svm             import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "lfw", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def tune_c(X_train_pca, X_val_pca, y_train, y_val):
    C_values     = [1, 2, 3, 4, 5, 7, 8, 9, 10, 50, 100]
    gamma_values = ["scale", "auto", 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.1, 0.5, 1]

    best_acc   = 0
    best_C     = 5
    best_gamma = "scale"
    c_results  = {}

    print("  Tuning C and gamma on validation set...")
    print(f"  {'C':>6}  {'Gamma':>8}  {'Val Accuracy':>14}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*14}")

    for C in C_values:
        for gamma in gamma_values:
            clf = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
            clf.fit(X_train_pca, y_train)
            val_acc = accuracy_score(y_val, clf.predict(X_val_pca))
            c_results[(C, gamma)] = val_acc
            marker = "  ←" if val_acc > best_acc else ""
            print(f"  {C:>6}  {str(gamma):>8}  {val_acc*100:>13.2f}%{marker}")
            if val_acc > best_acc:
                best_acc   = val_acc
                best_C     = C
                best_gamma = gamma

    print(f"\n  Best C={best_C}, gamma={best_gamma}  (val accuracy: {best_acc*100:.2f}%)")
    return best_C, best_gamma, c_results


def run_cross_validation(X_train_pca, X_val_pca, y_train, y_val, best_C, best_gamma):
    X_cv = np.vstack([X_train_pca, X_val_pca])
    y_cv = np.concatenate([y_train, y_val])
    clf  = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n  Running 5-fold stratified CV (C={best_C}, gamma={best_gamma})...")
    scores = cross_val_score(clf, X_cv, y_cv, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  Fold accuracies : {[f'{s*100:.1f}%' for s in scores]}")
    print(f"  Mean accuracy   : {scores.mean()*100:.2f}%")
    print(f"  Std deviation   : ±{scores.std()*100:.2f}%")
    return scores


def train_and_evaluate(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test):
    print("\n" + "─" * 45)
    print("  STEP 1 — C + Gamma Parameter Tuning")
    print("─" * 45)
    best_C, best_gamma, c_results = tune_c(X_train_pca, X_val_pca, y_train, y_val)

    print("\n" + "─" * 45)
    print("  STEP 2 — Cross-Validation")
    print("─" * 45)
    cv_scores = run_cross_validation(X_train_pca, X_val_pca, y_train, y_val, best_C, best_gamma)

    print("\n" + "─" * 45)
    print("  STEP 3 — Training Final Model")
    print("─" * 45)
    X_trainval = np.vstack([X_train_pca, X_val_pca])
    y_trainval = np.concatenate([y_train, y_val])
    print(f"  Training on {X_trainval.shape[0]} images (train + val)...")

    start_time  = time.time()
    final_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    final_model.fit(X_trainval, y_trainval)
    train_time  = time.time() - start_time
    print(f"  Done in {train_time:.2f}s")

    print("\n" + "─" * 45)
    print("  STEP 4 — Test Set Evaluation")
    print("─" * 45)
    y_pred   = final_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  Test Accuracy   : {accuracy*100:.2f}%")
    print(f"  CV Mean ± Std   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  Best C          : {best_C}")
    print(f"  Best gamma      : {best_gamma}")
    print(f"  Train time      : {train_time:.2f}s")
    print(f"  {'─'*40}\n")

    return {
        "model_name" : "SVM",
        "accuracy"   : accuracy,
        "y_pred"     : y_pred,
        "train_time" : train_time,
        "model"      : final_model,
        "best_C"     : best_C,
        "best_gamma" : best_gamma,
        "cv_scores"  : cv_scores,
        "c_results"  : c_results,
    }


def plot_c_gamma_heatmap(c_results, best_C, best_gamma, save_path=None):
    C_values     = sorted(set(k[0] for k in c_results))
    gamma_values = list(dict.fromkeys(k[1] for k in c_results))
    matrix = np.zeros((len(gamma_values), len(C_values)))
    for i, gamma in enumerate(gamma_values):
        for j, C in enumerate(C_values):
            matrix[i, j] = c_results.get((C, gamma), 0) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Validation Accuracy (%)")
    ax.set_xticks(range(len(C_values))); ax.set_xticklabels([str(c) for c in C_values])
    ax.set_yticks(range(len(gamma_values))); ax.set_yticklabels([str(g) for g in gamma_values])
    ax.set_xlabel("C Value"); ax.set_ylabel("Gamma")
    ax.set_title("SVM — Joint C + Gamma Tuning (Validation Accuracy %)")
    for i in range(len(gamma_values)):
        for j in range(len(C_values)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if val > matrix.max()*0.75 else "black")
    best_i = list(gamma_values).index(best_gamma)
    best_j = C_values.index(best_C)
    ax.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                                fill=False, edgecolor="blue", linewidth=3))
    plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "svm_c_gamma_heatmap.png")
    plt.savefig(save_path, dpi=150); print(f"  Saved → {save_path}"); plt.show()


def plot_cv_scores(cv_scores, save_path=None):
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    accs  = cv_scores * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(folds, accs, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axhline(y=accs.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {accs.mean():.1f}%")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{acc:.1f}%", ha="center", va="bottom")
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"SVM — 5-Fold Cross-Validation  (std: ±{accs.std():.1f}%)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "svm_cv_scores.png")
    plt.savefig(save_path, dpi=150); print(f"  Saved → {save_path}"); plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("SVM — Confusion Matrix"); plt.tight_layout()
    save_path = save_path or os.path.join(MODEL_DIR, "svm_confusion_matrix.png")
    plt.savefig(save_path, dpi=150); print(f"  Saved → {save_path}"); plt.show()


def main():
    print("=" * 55)
    print("  SVM Face Recognition — Chapter 2 (LFW)")
    print("=" * 55)

    print("\n>>> STEP 1 — Loading dataset\n")
    X, y = load_dataset()

    print("\n>>> STEP 2 — Preprocessing (PCA + 60/20/20 split)\n")
    X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca = prepare_data(X, y)
    n_components = X_train_pca.shape[1]

    print("\n>>> STEP 3 — Training SVM\n")
    results = train_and_evaluate(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test)

    print("\n>>> STEP 4 — Classification Report\n")
    print(classification_report(y_test, results["y_pred"], zero_division=0))

    print("\n>>> STEP 5 — Saving Plots\n")
    plot_c_gamma_heatmap(results["c_results"], results["best_C"], results["best_gamma"])
    plot_cv_scores(results["cv_scores"])
    plot_confusion_matrix(y_test, results["y_pred"])

    print("\n>>> STEP 6 — Saving Model\n")
    joblib.dump(results["model"], os.path.join(MODEL_DIR, "svm_lfw.pkl"))
    joblib.dump(pca,              os.path.join(MODEL_DIR, "pca_lfw.pkl"))
    print(f"  Saved → support/lfw/model/svm_lfw.pkl")

    print("\n" + "=" * 55)
    print(f"  DONE — Test Accuracy: {results['accuracy']*100:.2f}%")
    print("=" * 55)

    # ── Save results to JSON for compare_models.py ──
    import json
    from sklearn.metrics import precision_score, recall_score, f1_score

    json_data = {
        "dataset"      : "LFW",
        "model"        : "SVM",
        "accuracy"     : results["accuracy"],
        "precision" : float(precision_score(y_test, results["y_pred"], average="macro", zero_division=0)),
        "recall"    : float(recall_score   (y_test, results["y_pred"], average="macro", zero_division=0)),
        "f1"        : float(f1_score       (y_test, results["y_pred"], average="macro", zero_division=0)),
        "train_time"   : results["train_time"],
        "n_components" : n_components,
        "params"       : f"C={results['best_C']}, gamma={results['best_gamma']}",
        "cv_mean"      : float(results["cv_scores"].mean() * 100),
        "cv_std"       : float(results["cv_scores"].std()  * 100),
    }

    json_path = os.path.join(MODEL_DIR, "svm_lfw_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Results saved → {json_path}")

    # ── Return enriched dict for compare_models.py ────────────────────────────
    return {
        **results,
        "y_test"       : y_test,
        "n_components" : n_components,
        "dataset"      : "LFW",
        "params"       : f"C={results['best_C']}, γ={results['best_gamma']}",
    }


if __name__ == "__main__":
    main()