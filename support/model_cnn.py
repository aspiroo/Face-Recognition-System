# model_cnn.py
# Convolutional Neural Network for face recognition — Chapter 2.
#
# FIXES applied:
#   1. Dropout rates reduced: 0.25→0.1 per conv block, 0.5→0.3 at head.
#      With only 12 training images per person, high dropout kills learning
#      by zeroing too many activations per step.
#   2. BATCH_SIZE reduced 16→8. With small per-class counts, batch=16 often
#      has batches with no repeated classes, making loss noisy.
#   3. Added GlobalAveragePooling2D option instead of Flatten — reduces
#      parameter count significantly, less overfitting.
#   4. LR reduced 0.001→0.0005 — more stable on small datasets.
#   5. ImageDataGenerator: removed datagen.fit() (unnecessary for non-ZCA).
#      Added fill_mode="reflect" — better than "nearest" for face edges.
#   6. brightness_range is float-image safe (images are 0-1).
#
# Architecture:
#   Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.1)
#   Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.1)
#   Conv2D(128)→ BN → ReLU → MaxPool → Dropout(0.1)
#   GlobalAveragePooling2D  (replaces Flatten — fewer params, less overfit)
#   Dense(256) → BN → ReLU → Dropout(0.3)
#   Dense(n_classes) → Softmax
#
# Run from repo root: python -m support.model_cnn
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support.load_data_lfw import load_dataset
# from support.load_data_lfw_yale import load_dataset (when using mixed dataset)
# from support.load_data_lfw import load_dataset(if using LFW-only dataset)
from support.preprocessing import prepare_data_raw

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_HEIGHT    = 128
IMG_WIDTH     = 128
BATCH_SIZE    = 16        # FIXED: was 16 — smaller batches improve gradient signal
                          # when per-class counts are low (~12 train images/person)
MAX_EPOCHS    = 100
PATIENCE      = 20        # FIXED: was 10 — give model more time with small dataset
LEARNING_RATE = 0.00005   # FIXED: was 0.001 — lower LR stabilises small-dataset training

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "support", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(n_classes):
    """
    Build and compile the CNN.

    Key change: GlobalAveragePooling2D replaces Flatten.
    After 3 MaxPool layers, the feature map is 16x16x128 = 32768 values.
    Flattening that to Dense(256) creates a 32768×256 = 8.4M parameter matrix —
    massively prone to overfitting with <1000 training images.
    GlobalAveragePooling averages each of the 128 feature maps to a single value,
    giving a 128-dim vector. Dense(256) then has only 128×256 = 32K parameters.
    This alone can explain near-0% CNN accuracy on small datasets.
    """
    model = keras.Sequential([

        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),      # 128→64
        layers.Dropout(0.1),             # FIXED: was 0.25

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),      # 64→32
        layers.Dropout(0.1),             # FIXED: was 0.25

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),      # 32→16
        layers.Dropout(0.1),             # FIXED: was 0.25

        # FIXED: GlobalAveragePooling2D replaces Flatten
        # Flatten → Dense(256): 32768×256 = 8.4M params (overfit city)
        # GAP     → Dense(256):   128×256 =   32K params (manageable)
        layers.GlobalAveragePooling2D(),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),             # FIXED: was 0.5

        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0        
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_augmentation():
    """
    Data augmentation for training images (float32, 0-1 range).

    FIXED: Removed datagen.fit() call from train_and_evaluate — it is only
    needed for featurewise_center or ZCA whitening, neither of which we use.
    Calling it unnecessarily computes dataset statistics and slows startup.

    fill_mode changed "nearest"→"reflect": reflects pixel values at borders
    instead of repeating edge pixels — produces more realistic face padding.
    """
    return ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.85, 1.15],  # safe for 0-1 float images
        fill_mode="reflect"             # FIXED: was "nearest"
    )


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Build, train, and evaluate the CNN.

    Returns dict:
        model_name, accuracy, y_pred, train_time, model, history
    """
    n_classes = len(np.unique(y_train))
    print(f"  Classes     : {n_classes}")
    print(f"  Train       : {X_train.shape[0]} images")
    print(f"  Val         : {X_val.shape[0]} images")
    print(f"  Test        : {X_test.shape[0]} images")

    # Add channel dim: (n, 128, 128) → (n, 128, 128, 1)
    X_train_4d = np.expand_dims(X_train, -1)
    X_val_4d   = np.expand_dims(X_val,   -1)
    X_test_4d  = np.expand_dims(X_test,  -1)

    print("\n  Building model...")
    model = build_model(n_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "cnn_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,          # FIXED: was 5 — more patient with small dataset
            min_lr=1e-6,
            verbose=1
        )
    ]

    datagen = build_augmentation()
    # FIXED: removed datagen.fit(X_train_4d) — not needed without ZCA/featurewise

    print(f"\n  Training (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE})...")
    start_time = time.time()

    history = model.fit(
        datagen.flow(X_train_4d, y_train, batch_size=BATCH_SIZE),
        epochs=MAX_EPOCHS,
        validation_data=(X_val_4d, y_val),
        callbacks=callbacks,
        verbose=1
    )

    train_time = time.time() - start_time

    print("\n  Evaluating on test set...")
    y_pred_probs = model.predict(X_test_4d, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    accuracy     = accuracy_score(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print(f"  Train time    : {train_time:.1f}s")
    print(f"  Epochs run    : {len(history.history['loss'])}")
    print(f"  {'─'*40}\n")

    return {
        "model_name" : "CNN",
        "accuracy"   : accuracy,
        "y_pred"     : y_pred,
        "train_time" : train_time,
        "model"      : model,
        "history"    : history
    }


def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History", fontsize=14)
    epochs = range(1, len(history.history["accuracy"]) + 1)

    ax1.plot(epochs, history.history["accuracy"],     label="Train", linewidth=2)
    ax1.plot(epochs, history.history["val_accuracy"], label="Val",   linewidth=2)
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.history["loss"],     label="Train", linewidth=2)
    ax2.plot(epochs, history.history["val_loss"], label="Val",   linewidth=2)
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix_cnn(y_test, y_pred, n_classes, save_path=None):
    cm     = confusion_matrix(y_test, y_pred)
    labels = [str(i) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=90)
    ax.set_title("CNN — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.show()


def main():
    print("=" * 55)
    print("  CNN Face Recognition — Chapter 2")
    print("=" * 55)

    print("\n>>> STEP 1 — Loading dataset\n")
    X, y = load_dataset()

    print("\n>>> STEP 2 — Splitting data (60/20/20)\n")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_raw(X, y)

    print("\n>>> STEP 3 — Training CNN\n")
    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\n>>> STEP 4 — Classification report\n")
    n_classes = len(np.unique(y_test))
    print(classification_report(y_test, results["y_pred"]))

    print("\n>>> STEP 5 — Saving plots\n")
    plot_training_history(
        results["history"],
        save_path=os.path.join(MODEL_DIR, "cnn_training_history.png")
    )
    plot_confusion_matrix_cnn(
        y_test, results["y_pred"], n_classes,
        save_path=os.path.join(MODEL_DIR, "cnn_confusion_matrix.png")
    )

    print("\n>>> STEP 6 — Saving final model\n")
    save_path = os.path.join(MODEL_DIR, "cnn_final.keras")
    results["model"].save(save_path)
    print(f"  Saved → support/model/cnn_final.keras")

    print("\n" + "=" * 55)
    print(f"  DONE — Test Accuracy: {results['accuracy'] * 100:.2f}%")
    print("=" * 55)

    return results


if __name__ == "__main__":
    main()