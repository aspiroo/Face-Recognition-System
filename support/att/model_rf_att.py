# model_rf_att.py — AT&T, Chapter 1. Location: support/att/model_rf_att.py
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from support.att.load_normalized_data import load_dataset
from support.att.preprocessing_att import prepare_data, sweep_components, plot_sweep
from sklearn.ensemble import RandomForestClassifier
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

def tune_rf(X_train_pca, y_train):
    n_est_vals=[100,200,300,500,700]; mf_vals=["sqrt","log2",0.2,0.3,0.5]
    best_acc=0; best_n_est=300; best_mf="sqrt"; results={}
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    print(f"  {'n_est':>6}  {'max_feat':>10}  {'CV Acc':>10}")
    for n_est in n_est_vals:
        for mf in mf_vals:
            rf=RandomForestClassifier(n_estimators=n_est,max_features=mf,random_state=42,n_jobs=-1)
            scores=cross_val_score(rf,X_train_pca,y_train,cv=skf,scoring="accuracy",n_jobs=-1)
            acc=scores.mean(); results[(n_est,mf)]=acc; marker="  ←" if acc>best_acc else ""
            print(f"  {n_est:>6}  {str(mf):>10}  {acc*100:>9.2f}%{marker}")
            if acc>best_acc: best_acc=acc; best_n_est=n_est; best_mf=mf
    print(f"\n  Best n_est={best_n_est}, max_feat={best_mf}  ({best_acc*100:.2f}%)")
    return best_n_est, best_mf, results

def plot_tuning_heatmap(results, best_n_est, best_mf, save_path=None):
    n_est_vals=sorted(set(k[0] for k in results)); mf_vals=list(dict.fromkeys(k[1] for k in results))
    matrix=np.zeros((len(mf_vals),len(n_est_vals)))
    for i,mf in enumerate(mf_vals):
        for j,ne in enumerate(n_est_vals): matrix[i,j]=results.get((ne,mf),0)*100
    fig,ax=plt.subplots(figsize=(11,5)); im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto")
    plt.colorbar(im,ax=ax); ax.set_xticks(range(len(n_est_vals))); ax.set_xticklabels([str(n) for n in n_est_vals])
    ax.set_yticks(range(len(mf_vals))); ax.set_yticklabels([str(m) for m in mf_vals])
    ax.set_xlabel("n_estimators"); ax.set_ylabel("max_features")
    ax.set_title("AT&T RF — Hyperparameter Tuning CV %")
    for i in range(len(mf_vals)):
        for j in range(len(n_est_vals)):
            val=matrix[i,j]; ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=9,
                                     color="white" if val>matrix.max()*0.75 else "black")
    bi=list(mf_vals).index(best_mf); bj=n_est_vals.index(best_n_est)
    ax.add_patch(plt.Rectangle((bj-0.5,bi-0.5),1,1,fill=False,edgecolor="blue",linewidth=3))
    plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"rf_att_tuning_heatmap.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def plot_feature_importance(model, save_path=None):
    imp=model.feature_importances_; idx=np.argsort(imp)[::-1][:30]
    fig,ax=plt.subplots(figsize=(12,5)); ax.bar(range(len(idx)),imp[idx]*100,color="steelblue",alpha=0.85)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([f"PC{idx[i]+1}" for i in range(len(idx))],rotation=45,ha="right",fontsize=9)
    ax.set_xlabel("PCA Component"); ax.set_ylabel("Importance (%)")
    ax.set_title("AT&T RF — Top 30 PCA Importances")
    ax.grid(True,alpha=0.3,axis="y"); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"rf_att_feature_importance.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm=confusion_matrix(y_test,y_pred); labels=sorted(np.unique(y_test))
    fig,ax=plt.subplots(figsize=(14,12))
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels).plot(
        ax=ax,colorbar=True,cmap="Blues",xticks_rotation=45)
    ax.set_title("AT&T RF — Confusion Matrix"); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"rf_att_confusion_matrix.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def main():
    print("="*55); print("  RF — AT&T Dataset (Chapter 1)"); print("="*55)
    print("\n>>> STEP 1 — Loading\n"); X,y=load_dataset()
    print("\n>>> STEP 2 — PCA Sweep\n")
    proxy=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    best_n,sweep_results=sweep_components(X,y,proxy_clf=proxy,label="RF n=100",whiten=False,step=5,max_n=150)
    print("\n>>> STEP 3 — Preprocessing\n")
    X_train_pca,X_test_pca,y_train,y_test,pca=prepare_data(X,y,n_components=best_n,whiten=False)
    print("\n>>> STEP 4 — Tuning\n")
    best_n_est,best_mf,tune_results=tune_rf(X_train_pca,y_train)
    print("\n>>> STEP 5 — Training Final Model\n")
    start=time.time()
    final=RandomForestClassifier(n_estimators=best_n_est,max_features=best_mf,random_state=42,n_jobs=-1)
    final.fit(X_train_pca,y_train)
    train_time=time.time()-start

    print("\n>>> STEP 6 — Evaluate\n")
    y_pred=final.predict(X_test_pca); accuracy=accuracy_score(y_test,y_pred)
    print(f"  Test Accuracy:{accuracy*100:.2f}%  n_est={best_n_est}  max_feat={best_mf}  n={best_n}")

    # ── CV score on training set with best params for reporting ──────────────
    print("\n>>> STEP 6b — CV Score (best params)\n")
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_report=cross_val_score(
        RandomForestClassifier(n_estimators=best_n_est,max_features=best_mf,random_state=42,n_jobs=-1),
        X_train_pca, y_train, cv=skf, scoring="accuracy", n_jobs=-1
    )
    print(f"  CV Folds : {[f'{s*100:.1f}%' for s in cv_report]}")
    print(f"  CV Mean  : {cv_report.mean()*100:.2f}%  ±{cv_report.std()*100:.2f}%")

    print("\n>>> STEP 7 — Classification Report\n")
    print(classification_report(y_test,y_pred,zero_division=0))

    print("\n>>> STEP 8 — Plots\n")
    plot_sweep(sweep_results,best_n,title="AT&T RF — PCA Sweep",
               save_path=os.path.join(MODEL_DIR,"rf_att_pca_sweep.png"))
    plot_tuning_heatmap(tune_results,best_n_est,best_mf)
    plot_feature_importance(final)
    plot_confusion_matrix(y_test,y_pred)

    print("\n>>> STEP 9 — Saving Model\n")
    joblib.dump(final,os.path.join(MODEL_DIR,"rf_att.pkl"))
    joblib.dump(pca,os.path.join(MODEL_DIR,"pca_att_rf.pkl"))
    print(f"  Saved → support/att/model/rf_att.pkl")

    # ── Save JSON for compare_models.py ──────────────────────────────────────
    json_data = {
        "dataset"      : "AT&T",
        "model"        : "RF",
        "accuracy"     : float(accuracy),
        "precision"    : float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall"       : float(recall_score   (y_test, y_pred, average="macro", zero_division=0)),
        "f1"           : float(f1_score       (y_test, y_pred, average="macro", zero_division=0)),
        "train_time"   : float(train_time),
        "n_components" : int(best_n),
        "params"       : f"n_est={best_n_est}, max_feat={best_mf}",
        "cv_mean"      : float(cv_report.mean() * 100),
        "cv_std"       : float(cv_report.std()  * 100),
    }
    json_path = os.path.join(MODEL_DIR, "rf_att_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Results saved → {json_path}")

    print("\n"+"="*55); print(f"  DONE — {accuracy*100:.2f}%"); print("="*55)

    return {
        "model_name"    : "RF",
        "accuracy"      : accuracy,
        "y_pred"        : y_pred,
        "y_test"        : y_test,
        "train_time"    : train_time,
        "model"         : final,
        "best_n_est"    : best_n_est,
        "best_max_feat" : best_mf,
        "cv_scores"     : cv_report,
        "n_components"  : best_n,
        "dataset"       : "AT&T",
        "params"        : f"n_est={best_n_est}, max_feat={best_mf}",
    }

if __name__ == "__main__":
    main()