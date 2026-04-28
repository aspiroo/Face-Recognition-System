# compare_models.py
# Reads saved JSON results from each model and produces comparison.
# Run AFTER running each model individually.
# Location: support/compare_models.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "support", "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── JSON file locations ───────────────────────────────────────────────────────
JSON_FILES = [
    os.path.join(ROOT, "support", "att", "model", "svm_att_results.json"),
    os.path.join(ROOT, "support", "att", "model", "knn_att_results.json"),
    os.path.join(ROOT, "support", "att", "model", "rf_att_results.json"),
    os.path.join(ROOT, "support", "lfw", "model", "svm_lfw_results.json"),
    os.path.join(ROOT, "support", "lfw", "model", "knn_lfw_results.json"),
    os.path.join(ROOT, "support", "lfw", "model", "rf_lfw_results.json"),
    os.path.join(ROOT, "support", "lfw", "model", "cnn_lfw_results.json"),
]


def load_results():
    rows = []
    missing = []
    for path in JSON_FILES:
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
        else:
            missing.append(path)

    if missing:
        print("\n  WARNING — Missing result files (run these models first):")
        for p in missing:
            print(f"    ✗  {os.path.relpath(p, ROOT)}")
        print()

    return rows


def print_table(rows):
    sep = "─" * 110
    print(f"\n{sep}")
    print(f"  {'Dataset':<8} {'Model':<6} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'CV Mean':>9} {'CV Std':>8} "
          f"{'Time(s)':>8} {'PCA n':>6}  Parameters")
    print(sep)

    for r in rows:
        cv_mean = f"{r['cv_mean']:>8.2f}%" if r["cv_mean"] is not None else f"{'N/A':>9}"
        cv_std  = f"±{r['cv_std']:>6.2f}%" if r["cv_std"]  is not None else f"{'':>8}"
        print(
            f"  {r['dataset']:<8} {r['model']:<6} "
            f"{r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
            f"{r['recall']*100:>7.2f}% {r['f1']*100:>7.2f}% "
            f"{cv_mean} {cv_std} "
            f"{r['train_time']:>7.2f}s {str(r['n_components']):>6}  {r['params']}"
        )
    print(sep)


def save_table(rows):
    path = os.path.join(OUTPUT_DIR, "comparison_table.txt")
    sep  = "─" * 110
    lines = [sep,
             f"  {'Dataset':<8} {'Model':<6} {'Accuracy':>10} {'Precision':>10} "
             f"{'Recall':>8} {'F1':>8} {'CV Mean':>9} {'CV Std':>8} "
             f"{'Time(s)':>8} {'PCA n':>6}  Parameters",
             sep]
    for r in rows:
        cv_mean = f"{r['cv_mean']:>8.2f}%" if r["cv_mean"] is not None else f"{'N/A':>9}"
        cv_std  = f"±{r['cv_std']:>6.2f}%" if r["cv_std"]  is not None else f"{'':>8}"
        lines.append(
            f"  {r['dataset']:<8} {r['model']:<6} "
            f"{r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
            f"{r['recall']*100:>7.2f}% {r['f1']*100:>7.2f}% "
            f"{cv_mean} {cv_std} "
            f"{r['train_time']:>7.2f}s {str(r['n_components']):>6}  {r['params']}"
        )
    lines.append(sep)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Table saved → {path}")


def plot_accuracy_bar(rows):
    att = [r for r in rows if r["dataset"] == "AT&T"]
    lfw = [r for r in rows if r["dataset"] == "LFW"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Accuracy Comparison", fontsize=15)

    for ax, group, title in [(ax1, att, "AT&T — Chapter 1"),
                              (ax2, lfw, "LFW — Chapter 2")]:
        if not group:
            ax.set_visible(False)
            continue
        names = [r["model"] for r in group]
        accs  = [r["accuracy"] * 100 for r in group]
        bars  = ax.bar(names, accs, color=colors[:len(names)], alpha=0.85, edgecolor="white", width=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_ylim(0, min(110, max(accs) + 15))
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_accuracy.png")
    plt.savefig(path, dpi=150); print(f"  Saved → {path}")


def plot_metrics_grouped(rows):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for dataset in ["AT&T", "LFW"]:
        group = [r for r in rows if r["dataset"] == dataset]
        if not group:
            continue
        names = [r["model"] for r in group]
        x     = np.arange(len(names))
        width = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            vals = [r[metric] * 100 for r in group]
            bars = ax.bar(x + i*width - 1.5*width, vals, width,
                          label=label, color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Model"); ax.set_ylabel("Score (%)")
        ax.set_title(f"{dataset} — Accuracy / Precision / Recall / F1")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
        ax.set_ylim(0, 115); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"comparison_metrics_{dataset.lower().replace('&','')}.png")
        plt.savefig(path, dpi=150); print(f"  Saved → {path}")


def plot_train_time(rows):
    labels = [f"{r['dataset']} {r['model']}" for r in rows]
    times  = [r["train_time"] for r in rows]
    colors = ["#2196F3"]*len([r for r in rows if r["dataset"]=="AT&T"]) + \
             ["#FF9800"]*len([r for r in rows if r["dataset"]=="LFW"])
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, times, color=colors, alpha=0.85, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                f"{t:.2f}s", va="center", fontsize=10)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_train_time.png")
    plt.savefig(path, dpi=150); print(f"  Saved → {path}")


def plot_cv_vs_test(rows):
    cv_rows = [r for r in rows if r["cv_mean"] is not None]
    if not cv_rows:
        return
    labels = [f"{r['dataset']} {r['model']}" for r in cv_rows]
    means  = [r["cv_mean"] for r in cv_rows]
    stds   = [r["cv_std"]  for r in cv_rows]
    accs   = [r["accuracy"] * 100 for r in cv_rows]
    x      = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x-0.2, means, 0.35, label="CV Mean", alpha=0.7, color="steelblue",
           yerr=stds, capsize=5, edgecolor="white")
    ax.bar(x+0.2, accs,  0.35, label="Test Accuracy", alpha=1.0, color="darkorange",
           edgecolor="white")
    ax.set_xlabel("Model"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("CV Mean ± Std vs Test Accuracy")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 115); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_cv_vs_test.png")
    plt.savefig(path, dpi=150); print(f"  Saved → {path}")


def main():
    print("\n" + "="*60)
    print("  FACE RECOGNITION — Model Comparison")
    print("="*60)

    rows = load_results()

    if not rows:
        print("  No results found. Run models first then compare.")
        return

    print_table(rows)
    save_table(rows)

    print("\n  Generating plots...")
    plot_accuracy_bar(rows)
    plot_metrics_grouped(rows)
    plot_train_time(rows)
    plot_cv_vs_test(rows)

    print(f"\n  All outputs → support/comparison/")


if __name__ == "__main__":
    main()