# model_svm_att.py — AT&T, Chapter 1. Location: support/att/model_svm_att.py
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from support.att.load_normalized_data import load_dataset
from support.att.preprocessing_att import prepare_data, sweep_components, plot_sweep, plot_eigenfaces
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "att", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def tune_svm(X_train_pca, y_train):
    C_values=[0.1,1,5,10,50,100]; gamma_values=["scale","auto",0.0001,0.001,0.01,0.1]
    best_acc=0; best_C=10; best_gamma="scale"; results={}
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    print(f"  {'C':>6}  {'Gamma':>8}  {'CV Acc':>10}")
    for C in C_values:
        for gamma in gamma_values:
            clf=SVC(kernel="rbf",C=C,gamma=gamma,random_state=42)
            scores=cross_val_score(clf,X_train_pca,y_train,cv=skf,scoring="accuracy",n_jobs=-1)
            acc=scores.mean(); results[(C,gamma)]=acc; marker="  ←" if acc>best_acc else ""
            print(f"  {C:>6}  {str(gamma):>8}  {acc*100:>9.2f}%{marker}")
            if acc>best_acc: best_acc=acc; best_C=C; best_gamma=gamma
    print(f"\n  Best C={best_C}, gamma={best_gamma}  ({best_acc*100:.2f}%)")
    return best_C, best_gamma, results

def plot_c_gamma_heatmap(results, best_C, best_gamma, save_path=None):
    C_values=sorted(set(k[0] for k in results)); gamma_values=list(dict.fromkeys(k[1] for k in results))
    matrix=np.zeros((len(gamma_values),len(C_values)))
    for i,g in enumerate(gamma_values):
        for j,C in enumerate(C_values): matrix[i,j]=results.get((C,g),0)*100
    fig,ax=plt.subplots(figsize=(11,6)); im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto")
    plt.colorbar(im,ax=ax); ax.set_xticks(range(len(C_values))); ax.set_xticklabels([str(c) for c in C_values])
    ax.set_yticks(range(len(gamma_values))); ax.set_yticklabels([str(g) for g in gamma_values])
    ax.set_xlabel("C"); ax.set_ylabel("Gamma"); ax.set_title("AT&T SVM — C+Gamma CV Accuracy %")
    for i in range(len(gamma_values)):
        for j in range(len(C_values)):
            val=matrix[i,j]; ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=9,
                                     color="white" if val>matrix.max()*0.75 else "black")
    bi=list(gamma_values).index(best_gamma); bj=C_values.index(best_C)
    ax.add_patch(plt.Rectangle((bj-0.5,bi-0.5),1,1,fill=False,edgecolor="blue",linewidth=3))
    plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"svm_att_c_gamma_heatmap.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm=confusion_matrix(y_test,y_pred); labels=sorted(np.unique(y_test))
    fig,ax=plt.subplots(figsize=(14,12))
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels).plot(
        ax=ax,colorbar=True,cmap="Blues",xticks_rotation=45)
    ax.set_title("AT&T SVM — Confusion Matrix"); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"svm_att_confusion_matrix.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def main():
    print("="*55); print("  SVM — AT&T Dataset (Chapter 1)"); print("="*55)
    print("\n>>> STEP 1 — Loading\n"); X,y=load_dataset()
    print("\n>>> STEP 2 — PCA Sweep\n")
    proxy=SVC(kernel="rbf",C=10,gamma="scale",random_state=42)
    best_n,sweep_results=sweep_components(X,y,proxy_clf=proxy,label="SVM proxy C=10",whiten=True,step=5,max_n=150)
    print("\n>>> STEP 3 — Preprocessing\n")
    X_train_pca,X_test_pca,y_train,y_test,pca=prepare_data(X,y,n_components=best_n,whiten=True)
    print("\n>>> STEP 4 — Tuning C+Gamma\n")
    best_C,best_gamma,tune_results=tune_svm(X_train_pca,y_train)
    print("\n>>> STEP 5 — Training Final Model\n")
    start=time.time()
    final=SVC(kernel="rbf",C=best_C,gamma=best_gamma,random_state=42)
    final.fit(X_train_pca,y_train)
    train_time=time.time()-start

    print("\n>>> STEP 6 — Evaluate\n")
    y_pred=final.predict(X_test_pca); accuracy=accuracy_score(y_test,y_pred)
    print(f"  Test Accuracy:{accuracy*100:.2f}%  C={best_C}  gamma={best_gamma}  n={best_n}")

    # ── CV score on training set with best params for reporting ──────────────
    print("\n>>> STEP 6b — CV Score (best params)\n")
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_report=cross_val_score(
        SVC(kernel="rbf",C=best_C,gamma=best_gamma,random_state=42),
        X_train_pca, y_train, cv=skf, scoring="accuracy", n_jobs=-1
    )
    print(f"  CV Folds : {[f'{s*100:.1f}%' for s in cv_report]}")
    print(f"  CV Mean  : {cv_report.mean()*100:.2f}%  ±{cv_report.std()*100:.2f}%")

    print("\n>>> STEP 7 — Classification Report\n")
    print(classification_report(y_test,y_pred,zero_division=0))

    print("\n>>> STEP 8 — Plots\n")
    plot_sweep(sweep_results,best_n,title="AT&T SVM — PCA Sweep",
               save_path=os.path.join(MODEL_DIR,"svm_att_pca_sweep.png"))
    plot_c_gamma_heatmap(tune_results,best_C,best_gamma)
    plot_confusion_matrix(y_test,y_pred)
    plot_eigenfaces(pca,save_path=os.path.join(MODEL_DIR,"svm_att_eigenfaces.png"))

    print("\n>>> STEP 9 — Saving Model\n")
    joblib.dump(final,os.path.join(MODEL_DIR,"svm_att.pkl"))
    joblib.dump(pca,os.path.join(MODEL_DIR,"pca_att_svm.pkl"))
    print(f"  Saved → support/att/model/svm_att.pkl")

    # ── Save JSON for compare_models.py ──────────────────────────────────────
    json_data = {
        "dataset"      : "AT&T",
        "model"        : "SVM",
        "accuracy"     : float(accuracy),
        "precision"    : float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall"       : float(recall_score   (y_test, y_pred, average="macro", zero_division=0)),
        "f1"           : float(f1_score       (y_test, y_pred, average="macro", zero_division=0)),
        "train_time"   : float(train_time),
        "n_components" : int(best_n),
        "params"       : f"C={best_C}, gamma={best_gamma}",
        "cv_mean"      : float(cv_report.mean() * 100),
        "cv_std"       : float(cv_report.std()  * 100),
    }
    json_path = os.path.join(MODEL_DIR, "svm_att_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Results saved → {json_path}")

    print("\n"+"="*55); print(f"  DONE — {accuracy*100:.2f}%"); print("="*55)

    return {
        "model_name"   : "SVM",
        "accuracy"     : accuracy,
        "y_pred"       : y_pred,
        "y_test"       : y_test,
        "train_time"   : train_time,
        "model"        : final,
        "best_C"       : best_C,
        "best_gamma"   : best_gamma,
        "n_components" : best_n,
        "dataset"      : "AT&T",
        "params"       : f"C={best_C}, gamma={best_gamma}",
        "cv_scores"    : cv_report,
    }

if __name__ == "__main__":
    main()