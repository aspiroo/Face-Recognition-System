# main.py
# Entry point for the AT&T Face Recognition System.
# Runs the full pipeline: Load Data → Train → Evaluate → Recognize
# -------------------------------------------------------
# Run from repo root: python main.py
4
import sys
import os

# Add support/ to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "support"))

from support.train_model import train
from support.evaluate    import evaluate
from support.recognize   import recognize


def print_banner():
    print("\n" + "=" * 55)
    print("   AT&T Face Recognition System")
    print("   CSE445 — Machine Learning Project")
    print("=" * 55)
    print("  Contributors:")
    print("    Safwan Ismaun Amin      (2311443042)")
    print("    Ananya Sarkar           (2231005042)")
    print("    Nudrat Rahman Tushin    (2231058642)")
    print("    Muzahidur Rahman Saim   (2221758042)")
    print("=" * 55 + "\n")


def print_menu():
    print("\nWhat would you like to do?")
    print("  [1] Run full pipeline  (Train → Evaluate → Recognize)")
    print("  [2] Train model only")
    print("  [3] Evaluate model only")
    print("  [4] Recognize a random face only")
    print("  [0] Exit")
    return input("\nEnter choice: ").strip()


def run_full_pipeline():
    print("\n" + "=" * 55)
    print("  FULL PIPELINE — Train → Evaluate → Recognize")
    print("=" * 55)

    # ── Stage 1: Train ───────────────────────────────────────
    print("\n>>> STAGE 1 — TRAINING\n")
    pca, svm, accuracy = train()

    # ── Stage 2: Evaluate ────────────────────────────────────
    print("\n>>> STAGE 2 — EVALUATION\n")
    evaluate()

    # ── Stage 3: Recognize ───────────────────────────────────
    print("\n>>> STAGE 3 — RECOGNITION (random test image)\n")
    recognize()

    print("\n" + "=" * 55)
    print(f"  PIPELINE COMPLETE — Final Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55 + "\n")


def main():
    print_banner()

    while True:
        choice = print_menu()

        if choice == "1":
            run_full_pipeline()

        elif choice == "2":
            print("\n>>> TRAINING\n")
            train()

        elif choice == "3":
            print("\n>>> EVALUATING\n")
            evaluate()

        elif choice == "4":
            print("\n>>> RECOGNIZING\n")
            recognize()

        elif choice == "0":
            print("\nExiting. Goodbye!\n")
            break

        else:
            print("\n  Invalid choice. Please enter 0, 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()