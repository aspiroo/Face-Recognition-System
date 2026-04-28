# model_knn_att.py — AT&T, Chapter 1. Location: support/att/model_knn_att.py
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from support.att.load_normalized_data import load_dataset
from support.att.preprocessing_att import prepare_data, sweep_components, plot_sweep
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "att", "model")
os.makedirs(MODEL_DIR, exist_ok=True)
K_VALUES=[1,2,3,4,5,6,7,8,9,10,11,13,15]

def tune_k(X_train_pca, y_train):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); cv_scores=[]; best_acc=0; best_k=1
    print(f"  {'K':>4}  {'CV Acc':>10}  {'Std':>8}")
    for k in K_VALUES:
        knn=KNeighborsClassifier(n_neighbors=k,metric="cosine")
        scores=cross_val_score(knn,X_train_pca,y_train,cv=skf,scoring="accuracy",n_jobs=-1)
        acc=scores.mean(); cv_scores.append(acc); marker="  ←" if acc>best_acc else ""
        print(f"  {k:>4}  {acc*100:>9.2f}%  ±{scores.std()*100:>5.2f}%{marker}")
        if acc>best_acc: best_acc=acc; best_k=k
    print(f"\n  Best K={best_k}  ({best_acc*100:.2f}%)"); return best_k, cv_scores

def plot_k_curve(k_values, cv_scores, best_k, save_path=None):
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(k_values,[s*100 for s in cv_scores],marker="o",linewidth=2,color="steelblue",label="CV Accuracy")
    ax.scatter([best_k],[cv_scores[k_values.index(best_k)]*100],color="red",s=120,zorder=5,label=f"Best K={best_k}")
    ax.set_xlabel("K"); ax.set_ylabel("CV Accuracy (%)"); ax.set_title("AT&T KNN — Accuracy vs K")
    ax.set_xticks(k_values); ax.legend(); ax.grid(True,alpha=0.3); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"knn_att_k_curve.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm=confusion_matrix(y_test,y_pred); labels=sorted(np.unique(y_test))
    fig,ax=plt.subplots(figsize=(14,12))
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels).plot(ax=ax,colorbar=True,cmap="Blues",xticks_rotation=45)
    ax.set_title("AT&T KNN — Confusion Matrix"); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"knn_att_confusion_matrix.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def main():
    print("="*55); print("  KNN — AT&T Dataset (Chapter 1)"); print("="*55)
    print("\n>>> STEP 1 — Loading\n"); X,y=load_dataset()
    print("\n>>> STEP 2 — PCA Sweep\n")
    proxy=KNeighborsClassifier(n_neighbors=1,metric="cosine")
    best_n,sweep_results=sweep_components(X,y,proxy_clf=proxy,label="KNN K=1 cosine",whiten=False,step=5,max_n=150)
    print("\n>>> STEP 3 — Preprocessing\n")
    X_train_pca,X_test_pca,y_train,y_test,pca=prepare_data(X,y,n_components=best_n,whiten=False)
    print("\n>>> STEP 4 — Tuning K\n"); best_k,cv_scores=tune_k(X_train_pca,y_train)
    print("\n>>> STEP 5 — Training Final Model\n")
    start=time.time(); final=KNeighborsClassifier(n_neighbors=best_k,metric="cosine")
    final.fit(X_train_pca,y_train); train_time=time.time()-start
    y_pred=final.predict(X_test_pca); accuracy=accuracy_score(y_test,y_pred)
    print(f"  Test Accuracy:{accuracy*100:.2f}%  K={best_k}  n={best_n}")
    print("\n>>> STEP 6 — Classification Report\n")
    print(classification_report(y_test,y_pred,zero_division=0))
    print("\n>>> STEP 7 — Plots\n")
    plot_sweep(sweep_results,best_n,title="AT&T KNN — PCA Sweep",save_path=os.path.join(MODEL_DIR,"knn_att_pca_sweep.png"))
    plot_k_curve(K_VALUES,cv_scores,best_k)
    plot_confusion_matrix(y_test,y_pred)
    print("\n"+"="*55); print(f"  DONE — {accuracy*100:.2f}%"); print("="*55)
    
    # ── Save results to JSON ──
    import json
    from sklearn.metrics import precision_score, recall_score, f1_score

    json_data = {
        "dataset"      : "AT&T",
        "model"        : "KNN",
        "accuracy"     : accuracy,
        "precision"    : float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall"       : float(recall_score   (y_test, y_pred, average="macro", zero_division=0)),
        "f1"           : float(f1_score       (y_test, y_pred, average="macro", zero_division=0)),
        "train_time"   : train_time,
        "n_components" : best_n,
        "params"       : f"K={best_k}, metric=cosine",
        "cv_mean"      : float(np.array(cv_scores).mean() * 100),
        "cv_std"       : float(np.array(cv_scores).std()  * 100),
    }

    json_path = os.path.join(MODEL_DIR, "knn_att_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Results saved → {json_path}")
    
    return {"model_name":"KNN","accuracy":accuracy,"y_pred":y_pred,"y_test":y_test,
            "train_time":train_time,"model":final,"best_k":best_k,
            "cv_scores":np.array(cv_scores),"n_components":best_n,
            "dataset":"AT&T","params":f"K={best_k}, metric=cosine"}

if __name__ == "__main__":
    main()