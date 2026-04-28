# model_cnn.py — LFW, Chapter 2. Location: support/lfw/model_cnn.py
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from support.lfw.load_data_lfw import load_dataset
from support.lfw.preprocessing import prepare_data_raw
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    print("ERROR: TensorFlow not installed."); sys.exit(1)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

BATCH_SIZE=32; MAX_EPOCHS=100; PATIENCE=10; LEARNING_RATE=0.001
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "lfw", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(n_classes, img_h, img_w):
    model=keras.Sequential([
        layers.Input(shape=(img_h,img_w,1)),
        layers.Conv2D(32,(3,3),padding="same",activation="relu"), layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),padding="same",activation="relu"), layers.MaxPooling2D((2,2)),
        layers.Conv2D(128,(3,3),padding="same",activation="relu"), layers.MaxPooling2D((2,2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256,activation="relu"),
        layers.Dense(n_classes,activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

def build_augmentation():
    return ImageDataGenerator(horizontal_flip=True,rotation_range=10,zoom_range=0.1)

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    n_classes=len(np.unique(y_train)); img_h=X_train.shape[1]; img_w=X_train.shape[2]
    print(f"  Classes:{n_classes}  Image:{img_h}x{img_w}  Train:{X_train.shape[0]}  Val:{X_val.shape[0]}  Test:{X_test.shape[0]}")
    X_train_4d=np.expand_dims(X_train,-1); X_val_4d=np.expand_dims(X_val,-1); X_test_4d=np.expand_dims(X_test,-1)
    model=build_model(n_classes,img_h,img_w); model.summary()
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss",patience=PATIENCE,restore_best_weights=True,verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR,"cnn_best.keras"),monitor="val_loss",save_best_only=True,verbose=0)
    ]
    datagen=build_augmentation()
    print(f"\n  Training (max {MAX_EPOCHS} epochs, patience={PATIENCE})...")
    start=time.time()
    history=model.fit(datagen.flow(X_train_4d,y_train,batch_size=BATCH_SIZE),epochs=MAX_EPOCHS,
                      validation_data=(X_val_4d,y_val),callbacks=callbacks,verbose=1)
    train_time=time.time()-start
    y_pred_probs=model.predict(X_test_4d,verbose=0); y_pred=np.argmax(y_pred_probs,axis=1)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"  Test Accuracy:{accuracy*100:.2f}%  Train time:{train_time:.1f}s  Epochs:{len(history.history['loss'])}")
    return {"model_name":"CNN","accuracy":accuracy,"y_pred":y_pred,"train_time":train_time,"model":model,"history":history}

def plot_training_history(history, save_path=None):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5)); fig.suptitle("CNN Training History")
    epochs=range(1,len(history.history["accuracy"])+1)
    ax1.plot(epochs,history.history["accuracy"],label="Train",linewidth=2)
    ax1.plot(epochs,history.history["val_accuracy"],label="Val",linewidth=2)
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True,alpha=0.3)
    ax2.plot(epochs,history.history["loss"],label="Train",linewidth=2)
    ax2.plot(epochs,history.history["val_loss"],label="Val",linewidth=2)
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True,alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}")
    plt.show()

def plot_confusion_matrix_cnn(y_test, y_pred, n_classes, save_path=None):
    cm=confusion_matrix(y_test,y_pred); labels=[str(i) for i in range(n_classes)]
    fig,ax=plt.subplots(figsize=(14,12))
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels).plot(ax=ax,colorbar=True,cmap="Blues",xticks_rotation=90)
    ax.set_title("CNN — Confusion Matrix"); plt.tight_layout()
    if save_path: plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}")
    plt.show()

def main():
    print("="*55); print("  CNN Face Recognition — Chapter 2 (LFW)"); print("="*55)
    print("\n>>> STEP 1 — Loading dataset\n"); X,y=load_dataset()
    print("\n>>> STEP 2 — Splitting data (60/20/20)\n")
    X_train,X_val,X_test,y_train,y_val,y_test=prepare_data_raw(X,y)
    print("\n>>> STEP 3 — Training CNN\n")
    results=train_and_evaluate(X_train,X_val,X_test,y_train,y_val,y_test)
    print("\n>>> STEP 4 — Classification Report\n")
    n_classes=len(np.unique(y_test))
    print(classification_report(y_test,results["y_pred"],zero_division=0))
    print("\n>>> STEP 5 — Saving Plots\n")
    plot_training_history(results["history"],save_path=os.path.join(MODEL_DIR,"cnn_training_history.png"))
    plot_confusion_matrix_cnn(y_test,results["y_pred"],n_classes,save_path=os.path.join(MODEL_DIR,"cnn_confusion_matrix.png"))
    print("\n>>> STEP 6 — Saving Model\n")
    results["model"].save(os.path.join(MODEL_DIR,"cnn_final.keras"))
    print(f"  Saved → support/lfw/model/cnn_final.keras")
    print("\n"+"="*55); print(f"  DONE — Test Accuracy: {results['accuracy']*100:.2f}%"); print("="*55)

    # ── Save results to JSON ──
    import json
    from sklearn.metrics import precision_score, recall_score, f1_score

    json_data = {
        "dataset"      : "LFW",
        "model"        : "CNN",
        "accuracy"     : results["accuracy"],
        "precision"    : float(precision_score(y_test, results["y_pred"], average="macro", zero_division=0)),
        "recall"       : float(recall_score   (y_test, results["y_pred"], average="macro", zero_division=0)),
        "f1"           : float(f1_score       (y_test, results["y_pred"], average="macro", zero_division=0)),
        "train_time"   : results["train_time"],
        "n_components" : "N/A",
        "params"       : "3xConv, GAP, lr=0.001",
        "cv_mean"      : None,
        "cv_std"       : None,
    }

    json_path = os.path.join(MODEL_DIR, "cnn_lfw_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Results saved → {json_path}")
    return {**results,"y_test":y_test,"n_components":"N/A","dataset":"LFW","params":"3xConv, GAP, lr=0.001"}

if __name__ == "__main__":
    main()