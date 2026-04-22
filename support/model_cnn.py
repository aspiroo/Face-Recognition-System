# model_cnn.py
# Convolutional Neural Network for face recognition.
# Uses Yale + LFW dataset via load_data.py and prepare_data_raw() from preprocessing.py.
#
# Key differences from SVM/KNN/RF:
#   - No PCA — CNN learns its own features through convolution
#   - Input is raw 2D images (128x128x1), not compressed vectors
#   - Trains over multiple epochs with early stopping on val_loss
#   - Uses Dropout + BatchNorm to combat overfitting
#   - Data augmentation expands the training set artificially
#
# Architecture:
#   Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
#   Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
#   Conv2D(128)→ BatchNorm → ReLU → MaxPool → Dropout(0.25)
#   Flatten → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
#   Dense(n_classes) → Softmax
#
# Regularization used:
#   Dropout       — randomly disables neurons during training, prevents co-adaptation
#   BatchNorm     — normalizes layer inputs, stabilizes and speeds up training
#   EarlyStopping — stops training when val_loss stops improving (patience=10)
#   Augmentation  — flips, rotations, zoom, brightness shifts on training images
#
# Run from repo root: python -m support.model_cnn
# -------------------------------------------------------

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support.load_data_lfw_yale import load_dataset
from support.preprocessing import prepare_data_raw

# ── TensorFlow import ─────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    print("ERROR: TensorFlow not installed.")
    print("Run: pip install tensorflow")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_HEIGHT   = 128
IMG_WIDTH    = 128
BATCH_SIZE   = 16
MAX_EPOCHS   = 100    # early stopping will kick in well before this
PATIENCE     = 10     # stop if val_loss doesn't improve for 10 epochs
LEARNING_RATE = 0.001

# ── Output directory ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "support", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def build_model(n_classes):
    """
    Build and compile the CNN.

    Architecture rationale:
    - 3 Conv blocks with increasing filters (32→64→128) — learns simple edges
      first, then complex patterns deeper in the network
    - BatchNorm after each Conv — normalizes activations, reduces internal
      covariate shift, allows higher learning rates
    - MaxPool after each block — reduces spatial dimensions, adds translation
      invariance
    - Dropout(0.1) after each Conv block — prevents overfitting in feature maps
    - Dense(256) bottleneck — compresses spatial features to a representation vector
    - Dropout(0.3) before output — heavy dropout at the classification head
    - Softmax output — produces probability distribution over n_classes

    Args:
        n_classes : int — number of subjects (output layer size)

    Returns:
        compiled keras Model
    """
    model = keras.Sequential([

        # ── Input ─────────────────────────────────────────────────────────────
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        # ── Block 1: 32 filters — learns low-level edges and textures ─────────
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),           # 128x128 → 64x64
        layers.Dropout(0.1),

        # ── Block 2: 64 filters — learns mid-level features (eyes, nose) ──────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),           # 64x64 → 32x32
        layers.Dropout(0.1),

        # ── Block 3: 128 filters — learns high-level face structure ───────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),           # 32x32 → 16x16
        layers.Dropout(0.1),

        # ── Classification head ───────────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),                  # heavy dropout before output

        # ── Output: one probability per subject ───────────────────────────────
        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",  # y labels are integers not one-hot
        metrics=["accuracy"]
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def build_augmentation():
    """
    Build the data augmentation generator for training images.

    Augmentation rationale:
    - horizontal_flip     : a face from the left looks like same person from right
    - rotation_range=10   : small head tilts are common in real photos
    - zoom_range=0.1      : slight zoom simulates different distances
    - width/height_shift  : face not always perfectly centered
    - brightness_range    : simulates different lighting conditions
    - NO vertical flip    : upside-down faces are not realistic

    Each training image is randomly transformed on the fly each epoch,
    effectively multiplying training data without storing extra images.
    """
    return ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Build, train, and evaluate the CNN.
    Follows the same interface contract as model_svm/knn/rf for compare_models.py.

    Args:
        X_train/val/test : np.ndarray (n, 128, 128) — raw 2D grayscale images
        y_train/val/test : np.ndarray (n,)           — integer subject labels

    Returns dict with keys:
        model_name  : "CNN"
        accuracy    : float (test accuracy 0–1)
        y_pred      : np.ndarray of predictions on test set
        train_time  : float (seconds)
        model       : trained keras Model
        history     : keras History object (for plotting training curves)
    """
    n_classes = len(np.unique(y_train))
    print(f"  Classes     : {n_classes}")
    print(f"  Train       : {X_train.shape[0]} images")
    print(f"  Val         : {X_val.shape[0]} images")
    print(f"  Test        : {X_test.shape[0]} images")

    # ── Add channel dimension: (n, 128, 128) → (n, 128, 128, 1) ─────────────
    # Keras Conv2D expects (batch, height, width, channels)
    # Grayscale = 1 channel
    X_train_4d = np.expand_dims(X_train, -1)
    X_val_4d   = np.expand_dims(X_val,   -1)
    X_test_4d  = np.expand_dims(X_test,  -1)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n  Building model...")
    model = build_model(n_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        # Stop training when val_loss hasn't improved for PATIENCE epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,   # revert to best epoch when stopping
            verbose=1
        ),
        # Save the best model to disk during training
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "cnn_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
        # Reduce learning rate if val_loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,          # halve the learning rate
            patience=5,          # after 5 epochs of no improvement
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ── Data augmentation ─────────────────────────────────────────────────────
    datagen = build_augmentation()
    datagen.fit(X_train_4d)

    # ── Train ─────────────────────────────────────────────────────────────────
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

    # ── Evaluate on test set ──────────────────────────────────────────────────
    # Test set is touched ONCE here — never during training or tuning
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


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy + loss curves over epochs.
    The gap between train and val curves shows how much overfitting occurred.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History", fontsize=14)

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    ax1.plot(epochs, history.history["accuracy"],     label="Train", linewidth=2)
    ax1.plot(epochs, history.history["val_accuracy"], label="Val",   linewidth=2)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history.history["loss"],     label="Train", linewidth=2)
    ax2.plot(epochs, history.history["val_loss"], label="Val",   linewidth=2)
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix_cnn(y_test, y_pred, n_classes, save_path=None):
    """Plot confusion matrix for CNN predictions."""
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


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  CNN Face Recognition — Chapter 2")
    print("=" * 55)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n>>> STEP 1 — Loading dataset\n")
    X, y = load_dataset()   # Yale + LFW default

    # ── Step 2: Split (no PCA for CNN) ───────────────────────────────────────
    print("\n>>> STEP 2 — Splitting data (60/20/20)\n")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_raw(X, y)

    # ── Step 3: Train and evaluate ────────────────────────────────────────────
    print("\n>>> STEP 3 — Training CNN\n")
    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)

    # ── Step 4: Classification report ────────────────────────────────────────
    print("\n>>> STEP 4 — Classification report\n")
    n_classes = len(np.unique(y_test))
    print(classification_report(y_test, results["y_pred"]))

    # ── Step 5: Save plots ────────────────────────────────────────────────────
    print("\n>>> STEP 5 — Saving plots\n")
    plot_training_history(
        results["history"],
        save_path=os.path.join(MODEL_DIR, "cnn_training_history.png")
    )
    plot_confusion_matrix_cnn(
        y_test, results["y_pred"], n_classes,
        save_path=os.path.join(MODEL_DIR, "cnn_confusion_matrix.png")
    )

    # ── Step 6: Save model ────────────────────────────────────────────────────
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