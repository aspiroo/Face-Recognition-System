# main.py
# Terminal menu for Face Recognition System.
# Location: repo root
# Run: python main.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header():
    print("=" * 60)
    print("       FACE RECOGNITION SYSTEM — Main Menu")
    print("=" * 60)


def section(title):
    print(f"\n  {'─'*50}")
    print(f"    {title}")
    print(f"  {'─'*50}\n")


def pause():
    input("\n  Press Enter to return to menu...")


MAIN_MENU = """
  CHAPTER 1 — AT&T Dataset  (40 subjects, 80/20 split)
  ─────────────────────────────────────────────────────
  [1]  SVM   — train + evaluate (AT&T)
  [2]  KNN   — train + evaluate (AT&T)
  [3]  RF    — train + evaluate (AT&T)
  [4]  Run ALL Chapter 1 models

  CHAPTER 2 — LFW Dataset  (19 subjects, 60/20/20 split)
  ─────────────────────────────────────────────────────
  [5]  SVM   — train + evaluate (LFW)
  [6]  KNN   — train + evaluate (LFW)
  [7]  RF    — train + evaluate (LFW)
  [8]  CNN   — train + evaluate (LFW)
  [9]  Run ALL Chapter 2 models

  UTILITIES
  ─────────────────────────────────────────────────────
  [10] Recognize — predict a random face (AT&T SVM)
  [11] Evaluate  — evaluate saved AT&T SVM model
  [12] Compare   — run all models and compare results  


  [0]  Exit
  ─────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 1 — AT&T
# ─────────────────────────────────────────────────────────────────────────────

def run_att_svm():
    section("AT&T — SVM")
    from support.att.model_svm_att import main
    return main()


def run_att_knn():
    section("AT&T — KNN")
    from support.att.model_knn_att import main
    return main()


def run_att_rf():
    section("AT&T — Random Forest")
    from support.att.model_rf_att import main
    return main()


def run_att_all():
    section("AT&T — Running ALL models")

    print("  [1/3] SVM...")
    from support.att.model_svm_att import main as svm_main
    r_svm = svm_main()

    print("\n  [2/3] KNN...")
    from support.att.model_knn_att import main as knn_main
    r_knn = knn_main()

    print("\n  [3/3] Random Forest...")
    from support.att.model_rf_att import main as rf_main
    r_rf = rf_main()

    print("\n" + "=" * 60)
    print("  AT&T — Final Results Summary")
    print("=" * 60)
    print(f"  SVM  : {r_svm['accuracy']*100:.2f}%")
    print(f"  KNN  : {r_knn['accuracy']*100:.2f}%")
    print(f"  RF   : {r_rf['accuracy']*100:.2f}%")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 2 — LFW
# ─────────────────────────────────────────────────────────────────────────────

def run_lfw_svm():
    section("LFW — SVM")
    from support.lfw.model_svm import main
    return main()


def run_lfw_knn():
    section("LFW — KNN")
    from support.lfw.model_knn import main
    return main()


def run_lfw_rf():
    section("LFW — Random Forest")
    from support.lfw.model_rf import main
    return main()


def run_lfw_cnn():
    section("LFW — CNN")
    from support.lfw.model_cnn import main
    return main()


def run_lfw_all():
    section("LFW — Running ALL models")

    print("  [1/4] SVM...")
    from support.lfw.model_svm import main as svm_main
    r_svm = svm_main()

    print("\n  [2/4] KNN...")
    from support.lfw.model_knn import main as knn_main
    r_knn = knn_main()

    print("\n  [3/4] Random Forest...")
    from support.lfw.model_rf import main as rf_main
    r_rf = rf_main()

    print("\n  [4/4] CNN...")
    from support.lfw.model_cnn import main as cnn_main
    r_cnn = cnn_main()

    print("\n" + "=" * 60)
    print("  LFW — Final Results Summary")
    print("=" * 60)
    print(f"  SVM  : {r_svm['accuracy']*100:.2f}%")
    print(f"  KNN  : {r_knn['accuracy']*100:.2f}%")
    print(f"  RF   : {r_rf['accuracy']*100:.2f}%")
    print(f"  CNN  : {r_cnn['accuracy']*100:.2f}%")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def run_recognize():
    section("Recognize — Random AT&T Face (SVM)")
    from support.att.recognize import recognize
    recognize()


def run_evaluate():
    section("Evaluate — Saved AT&T SVM Model")
    from support.att.evaluate import evaluate
    evaluate()

def run_compare():
    section("Full Model Comparison — All 7 Models")
    from support.compare_models import main
    main()


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS = {
    "1"  : run_att_svm,
    "2"  : run_att_knn,
    "3"  : run_att_rf,
    "4"  : run_att_all,
    "5"  : run_lfw_svm,
    "6"  : run_lfw_knn,
    "7"  : run_lfw_rf,
    "8"  : run_lfw_cnn,
    "9"  : run_lfw_all,
    "10" : run_recognize,
    "11" : run_evaluate,
    "12" : run_compare, 
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    while True:
        clear()
        header()
        print(MAIN_MENU)

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            print("\n  Goodbye!\n")
            sys.exit(0)

        action = ACTIONS.get(choice)

        if action is None:
            print(f"\n  Invalid choice '{choice}' — please try again.")
            pause()
            continue

        clear()
        try:
            action()
            pause()
        except KeyboardInterrupt:
            print("\n\n  Interrupted — returning to menu.")
            pause()
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            pause()


if __name__ == "__main__":
    main()