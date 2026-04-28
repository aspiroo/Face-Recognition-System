# model_rf.py — LFW, Chapter 2. Location: support/lfw/model_rf.py
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from support.lfw.load_data_lfw import load_dataset
from support.lfw.preprocessing import prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "support", "lfw", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def sweep_components(X, y):
    print("Sweeping PCA components..."); best_n=100; best_val=0
    for n_comp in range(10, 210, 10):
        X_tr,X_v,X_te,y_tr,y_v,y_te,pca = prepare_data(X,y,n_components=n_comp,verbose=False)
        rf = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
        rf.fit(X_tr,y_tr); acc=accuracy_score(y_v,rf.predict(X_v))
        var=pca.explained_variance_ratio_.sum()*100; marker="  ←" if acc>best_val else ""
        print(f"  n={n_comp:>4}  var={var:>5.1f}%  acc={acc*100:>5.1f}%{marker}")
        if acc>best_val: best_val=acc; best_n=n_comp
    print(f"\n  Auto-selected n_components={best_n}"); return best_n

def tune_rf(X_train_pca, X_val_pca, y_train, y_val):
    n_est_vals=[100,200,300,400,500,600,700,800,900,1000]
    mf_vals=["sqrt","log2",0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    best_acc=0; best_n_est=200; best_mf="sqrt"; results={}
    print(f"  {'n_est':>6}  {'max_feat':>10}  {'Val Acc':>10}")
    for n_est in n_est_vals:
        for mf in mf_vals:
            rf=RandomForestClassifier(n_estimators=n_est,max_features=mf,random_state=42,n_jobs=-1)
            rf.fit(X_train_pca,y_train); acc=accuracy_score(y_val,rf.predict(X_val_pca))
            results[(n_est,mf)]=acc; marker="  ←" if acc>best_acc else ""
            print(f"  {n_est:>6}  {str(mf):>10}  {acc*100:>9.2f}%{marker}")
            if acc>best_acc: best_acc=acc; best_n_est=n_est; best_mf=mf
    print(f"\n  Best n_est={best_n_est}, max_feat={best_mf}  ({best_acc*100:.2f}%)")
    return best_n_est, best_mf, results

def run_cv(X_train_pca, X_val_pca, y_train, y_val, best_n_est, best_mf):
    X_cv=np.vstack([X_train_pca,X_val_pca]); y_cv=np.concatenate([y_train,y_val])
    rf=RandomForestClassifier(n_estimators=best_n_est,max_features=best_mf,random_state=42,n_jobs=-1)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores=cross_val_score(rf,X_cv,y_cv,cv=skf,scoring="accuracy",n_jobs=-1)
    print(f"  CV: {[f'{s*100:.1f}%' for s in scores]}  mean={scores.mean()*100:.2f}%  ±{scores.std()*100:.2f}%")
    return scores

def train_and_evaluate(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test):
    print("\n── STEP 1: Tuning"); best_n_est,best_mf,tune_results=tune_rf(X_train_pca,X_val_pca,y_train,y_val)
    print("\n── STEP 2: CV"); cv_scores=run_cv(X_train_pca,X_val_pca,y_train,y_val,best_n_est,best_mf)
    print("\n── STEP 3: Train final")
    X_tv=np.vstack([X_train_pca,X_val_pca]); y_tv=np.concatenate([y_train,y_val])
    start=time.time()
    final=RandomForestClassifier(n_estimators=best_n_est,max_features=best_mf,random_state=42,n_jobs=-1)
    final.fit(X_tv,y_tv); train_time=time.time()-start
    y_pred=final.predict(X_test_pca); accuracy=accuracy_score(y_test,y_pred)
    print(f"  Test Accuracy: {accuracy*100:.2f}%  Train time: {train_time:.2f}s")
    return {"model_name":"RF","accuracy":accuracy,"y_pred":y_pred,"train_time":train_time,
            "model":final,"best_n_est":best_n_est,"best_max_feat":best_mf,
            "cv_scores":cv_scores,"tune_results":tune_results}

def plot_tuning_heatmap(tune_results, best_n_est, best_mf, save_path=None):
    n_est_vals=sorted(set(k[0] for k in tune_results)); mf_vals=list(dict.fromkeys(k[1] for k in tune_results))
    matrix=np.zeros((len(mf_vals),len(n_est_vals)))
    for i,mf in enumerate(mf_vals):
        for j,ne in enumerate(n_est_vals): matrix[i,j]=tune_results.get((ne,mf),0)*100
    fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto")
    plt.colorbar(im,ax=ax); ax.set_xticks(range(len(n_est_vals))); ax.set_xticklabels([str(n) for n in n_est_vals])
    ax.set_yticks(range(len(mf_vals))); ax.set_yticklabels([str(m) for m in mf_vals])
    ax.set_xlabel("n_estimators"); ax.set_ylabel("max_features")
    ax.set_title("RF — Hyperparameter Tuning (Val Accuracy %)")
    for i in range(len(mf_vals)):
        for j in range(len(n_est_vals)):
            val=matrix[i,j]; ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=9,
                                     color="white" if val>matrix.max()*0.75 else "black")
    bi=list(mf_vals).index(best_mf); bj=n_est_vals.index(best_n_est)
    ax.add_patch(plt.Rectangle((bj-0.5,bi-0.5),1,1,fill=False,edgecolor="blue",linewidth=3))
    plt.tight_layout(); save_path=save_path or os.path.join(MODEL_DIR,"rf_tuning_heatmap.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def plot_feature_importance(model, save_path=None):
    imp=model.feature_importances_; idx=np.argsort(imp)[::-1][:30]
    fig,ax=plt.subplots(figsize=(12,5)); ax.bar(range(len(idx)),imp[idx]*100,color="steelblue",alpha=0.85)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels([f"PC{idx[i]+1}" for i in range(len(idx))],rotation=45,ha="right",fontsize=9)
    ax.set_xlabel("PCA Component"); ax.set_ylabel("Importance (%)"); ax.set_title("RF — Top 30 PCA Component Importances")
    ax.grid(True,alpha=0.3,axis="y"); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"rf_feature_importance.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def plot_cv_scores(cv_scores, save_path=None):
    folds=[f"Fold {i+1}" for i in range(len(cv_scores))]; accs=cv_scores*100
    fig,ax=plt.subplots(figsize=(8,5)); bars=ax.bar(folds,accs,color="steelblue",alpha=0.8,edgecolor="white")
    ax.axhline(y=accs.mean(),color="red",linestyle="--",linewidth=2,label=f"Mean: {accs.mean():.1f}%")
    for bar,acc in zip(bars,accs): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{acc:.1f}%",ha="center",va="bottom")
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"RF — 5-Fold CV  (std: ±{accs.std():.1f}%)")
    ax.legend(); ax.grid(True,alpha=0.3,axis="y"); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"rf_cv_scores.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm=confusion_matrix(y_test,y_pred); labels=sorted(np.unique(y_test))
    fig,ax=plt.subplots(figsize=(12,10))
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels).plot(ax=ax,colorbar=True,cmap="Blues",xticks_rotation=45)
    ax.set_title("Random Forest — Confusion Matrix"); plt.tight_layout()
    save_path=save_path or os.path.join(MODEL_DIR,"rf_confusion_matrix.png")
    plt.savefig(save_path,dpi=150); print(f"  Saved → {save_path}"); plt.show()

def main():
    print("="*55); print("  Random Forest — Chapter 2 (LFW)"); print("="*55)
    print("\n>>> STEP 1 — Loading dataset\n"); X,y=load_dataset()
    print("\n>>> STEP 2 — PCA Sweep\n"); best_n=sweep_components(X,y)
    print(f"\n>>> STEP 3 — Preprocessing (n={best_n})\n")
    X_train_pca,X_val_pca,X_test_pca,y_train,y_val,y_test,pca=prepare_data(X,y,n_components=best_n)
    print("\n>>> STEP 4 — Training RF\n")
    results=train_and_evaluate(X_train_pca,X_val_pca,X_test_pca,y_train,y_val,y_test)
    print("\n>>> STEP 5 — Classification Report\n")
    print(classification_report(y_test,results["y_pred"],zero_division=0))
    print("\n>>> STEP 6 — Saving Plots\n")
    plot_tuning_heatmap(results["tune_results"],results["best_n_est"],results["best_max_feat"])
    plot_feature_importance(results["model"])
    plot_cv_scores(results["cv_scores"])
    plot_confusion_matrix(y_test,results["y_pred"])
    print("\n>>> STEP 7 — Saving Model\n")
    joblib.dump(results["model"],os.path.join(MODEL_DIR,"rf_lfw.pkl"))
    joblib.dump(pca,os.path.join(MODEL_DIR,"pca_rf_lfw.pkl"))
    print(f"  Saved → support/lfw/model/rf_lfw.pkl")
    print("\n"+"="*55); print(f"  DONE — Test Accuracy: {results['accuracy']*100:.2f}%"); print("="*55)

    # ── Save results to JSON ──
    import json
    from sklearn.metrics import precision_score, recall_score, f1_score

    json_data = {
        "dataset"      : "LFW",
        "model"        : "RF",
        "accuracy"     : results["accuracy"],
        "precision"    : float(precision_score(y_test, results["y_pred"], average="macro", zero_division=0)),
        "recall"       : float(recall_score   (y_test, results["y_pred"], average="macro", zero_division=0)),
        "f1"           : float(f1_score       (y_test, results["y_pred"], average="macro", zero_division=0)),
        "train_time"   : results["train_time"],
        "n_components" : best_n,
        "params"       : f"n_est={results['best_n_est']}, max_feat={results['best_max_feat']}",
        "cv_mean"      : float(results["cv_scores"].mean() * 100),
        "cv_std"       : float(results["cv_scores"].std()  * 100),
    }

    json_path = os.path.join(MODEL_DIR, "rf_lfw_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Results saved → {json_path}")

    return {**results,"y_test":y_test,"n_components":best_n,"dataset":"LFW",
            "params":f"n_est={results['best_n_est']}, max_feat={results['best_max_feat']}"}

if __name__ == "__main__":
    main()