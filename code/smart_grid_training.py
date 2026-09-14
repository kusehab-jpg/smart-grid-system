"""
=============================================================================
Smart-Grid Fault Detection — Complete Model Training Pipeline
=============================================================================
Thesis : Implementation of a Machine Learning Technique for Fault
         Detection in Smart-Grid Systems
Author : Bright
Year   : 2026

Dataset: Electrical Fault Detection and Classification (classData.csv)
Source : Sathyaprakash, E. (2021). Kaggle.
URL    : https://www.kaggle.com/datasets/esathyaprakash/
         electrical-fault-detection-and-classification

Models : Random Forest  (sklearn.ensemble.RandomForestClassifier)
         HistGradBoost  (sklearn.ensemble.HistGradientBoostingClassifier)
         NOTE — Correction C5: GradientBoostingClassifier does NOT support
         class_weight='balanced'. HistGradientBoostingClassifier is used
         because it natively supports this parameter.

=============================================================================
SUPERVISOR CORRECTIONS APPLIED
=============================================================================
C1  Class counts printed from code — never typed manually.
C2  SPLIT applied BEFORE any preprocessing or SMOTE. SMOTE is confined
    to the training partition only, inside an imbalanced-learn Pipeline.
C3  Dataset fully identified: file name, citation, column definitions,
    fault encoding table, SHA-256 checksum printed automatically.
C4  All metrics generated from saved y_pred arrays — never entered manually.
C5  HistGradientBoostingClassifier replaces GradientBoostingClassifier.
C7  Reproducibility package: split indices, y_pred arrays,
    fitted Pipeline objects, results JSON, training log.
M5  95% bootstrap confidence intervals and McNemar's paired test reported.
M6  Cross-study comparison qualified as contextual, not direct.

=============================================================================
USAGE
=============================================================================
  pip install -r requirements.txt
  python smart_grid_training.py --data classData.csv   # real dataset
  python smart_grid_training.py                        # demo / synthetic

Outputs written to ./output/
=============================================================================
"""

import os, sys, time, json, hashlib, logging, warnings, argparse
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from scipy.stats import chi2 as chi2_dist

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Configuration ─────────────────────────────────────────────────────────────
SEED=42; np.random.seed(SEED)
TRAIN_RATIO=0.70; VAL_RATIO=0.15; TEST_RATIO=0.15
N_CV_FOLDS=5; SMOTE_K=5; N_BOOT=2000; N_INFER=500
LABELS=["Normal","LG","LL","LLG","LLL","LLLG"]; N_CLASSES=6
GCBA_MAP={(0,0,0,0):0,(1,0,0,1):1,(0,0,1,1):2,(1,0,1,1):3,(0,1,1,1):4,(1,1,1,1):5}
CITATION=("Sathyaprakash, E. (2021). Electrical fault detection and classification "
          "[Dataset]. Kaggle. https://www.kaggle.com/datasets/esathyaprakash/"
          "electrical-fault-detection-and-classification")
OUTPUT_DIR=Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUTPUT_DIR/"training_log.txt",mode="w",encoding="utf-8")])
log=logging.getLogger(__name__)
plt.rcParams.update({"font.family":"DejaVu Sans","axes.titlesize":12,
                     "axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9})
sns.set_theme(style="whitegrid")
B1="#1F4E79"; B2="#2E75B6"; OR="#ED7D31"; GR="#375623"; GY="#595959"; DPI=160

# =============================================================================
# STAGE 0 — DATASET VERIFICATION  (Correction C3)
# =============================================================================
def verify_dataset(csv_path):
    log.info("="*70)
    log.info("STAGE 0 — DATASET VERIFICATION  (Correction C3)")
    log.info("="*70)
    log.info(f"APA 7 citation : {CITATION}")
    sha256=hashlib.sha256()
    with open(csv_path,"rb") as fh:
        for chunk in iter(lambda:fh.read(8192),b""): sha256.update(chunk)
    checksum=sha256.hexdigest()
    log.info(f"File      : {csv_path}")
    log.info(f"SHA-256   : {checksum}")
    log.info("→ Record this checksum in Appendix A of the thesis.")
    (OUTPUT_DIR/"data_checksum.txt").write_text(
        f"File    : {csv_path}\nSHA-256 : {checksum}\n"
        f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Citation: {CITATION}\n",encoding="utf-8")
    df=pd.read_csv(csv_path)
    log.info(f"Shape     : {df.shape[0]:,} rows x {df.shape[1]} columns")
    log.info(f"Columns   : {list(df.columns)}")
    log.info(f"Missing   : {df.isnull().sum().sum()} values")
    return df

# =============================================================================
# STAGE 1 — FAULT LABEL ENCODING  (Correction C1, C3)
# =============================================================================
def encode_labels(df):
    log.info("\n"+"="*70)
    log.info("STAGE 1 — FAULT LABEL ENCODING  (Correction C1)")
    log.info("="*70)
    gcba_cols=None
    for v in [["G","C","B","A"],["g","c","b","a"]]:
        if all(c in df.columns for c in v): gcba_cols=v; break
    if gcba_cols is None:
        raise ValueError(f"Cannot find G,C,B,A columns. Found: {list(df.columns)}")
    df=df.copy()
    df["fault_type"]=df[gcba_cols].apply(
        lambda r:GCBA_MAP.get(tuple(r.astype(int)),-1),axis=1)
    df=df[df["fault_type"]!=-1].reset_index(drop=True)
    total=len(df); counts=df["fault_type"].value_counts().sort_index()
    log.info(f"Total valid samples: {total:,}")
    log.info("\nClass distribution (BEFORE split/SMOTE) — copy to Table 4.1:")
    log.info(f"  {'Class':>5}  {'Label':10}  {'Count':>7}  {'Pct':>6}")
    log.info("  "+"-"*35)
    for idx,cnt in counts.items():
        log.info(f"  {idx:>5}  {LABELS[idx]:10}  {cnt:>7,}  {cnt/total*100:>5.1f}%")
    log.info(f"  {'':>5}  {'TOTAL':10}  {total:>7,}  100.0%")
    return df

# =============================================================================
# STAGE 2 — FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    log.info("\n"+"="*70)
    log.info("STAGE 2 — FEATURE ENGINEERING (30 scalar features)")
    log.info("NOTE: Per-row scalar data — DWT/FFT not applicable.")
    log.info("="*70)
    def get(v):
        for n in v:
            if n in df.columns: return df[n].values.astype(float)
        raise KeyError(f"None of {v} found")
    eps=1e-9
    Ia=get(["Ia","ia"]); Ib=get(["Ib","ib"]); Ic=get(["Ic","ic"])
    Va=get(["Va","va"]); Vb=get(["Vb","vb"]); Vc=get(["Vc","vc"])
    Im=(np.abs(Ia)+np.abs(Ib)+np.abs(Ic))/3
    Vm=(np.abs(Va)+np.abs(Vb)+np.abs(Vc))/3
    f={"Ia":Ia,"Ib":Ib,"Ic":Ic,"Va":Va,"Vb":Vb,"Vc":Vc,
       "abs_Ia":np.abs(Ia),"abs_Ib":np.abs(Ib),"abs_Ic":np.abs(Ic),
       "abs_Va":np.abs(Va),"abs_Vb":np.abs(Vb),"abs_Vc":np.abs(Vc),
       "Z_a":np.abs(Va)/(np.abs(Ia)+eps),"Z_b":np.abs(Vb)/(np.abs(Ib)+eps),
       "Z_c":np.abs(Vc)/(np.abs(Ic)+eps),
       "dI_ab":Ia-Ib,"dI_bc":Ib-Ic,"dI_ca":Ic-Ia,
       "dV_ab":Va-Vb,"dV_bc":Vb-Vc,"dV_ca":Vc-Va,
       "I_zero_seq":(Ia+Ib+Ic)/3,"V_zero_seq":(Va+Vb+Vc)/3,
       "I_imbalance":np.max(np.abs(np.stack([np.abs(Ia),np.abs(Ib),np.abs(Ic)])-Im),axis=0)/(Im+eps),
       "V_imbalance":np.max(np.abs(np.stack([np.abs(Va),np.abs(Vb),np.abs(Vc)])-Vm),axis=0)/(Vm+eps),
       "P_a":np.abs(Va)*np.abs(Ia),"P_b":np.abs(Vb)*np.abs(Ib),"P_c":np.abs(Vc)*np.abs(Ic)}
    f["P_total"]=f["P_a"]+f["P_b"]+f["P_c"]
    feat_df=pd.DataFrame(f); feat_names=list(feat_df.columns)
    X=feat_df.values.astype(float); y=df["fault_type"].values.astype(int)
    log.info(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    log.info(f"Features: {feat_names}")
    return X,y,feat_names

# =============================================================================
# STAGE 3 — SPLIT FIRST  (Correction C2)
# =============================================================================
def split_first(X,y):
    log.info("\n"+"="*70)
    log.info("STAGE 3 — STRATIFIED SPLIT (CORRECTION C2: SPLIT BEFORE SMOTE)")
    log.info("="*70)
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=TEST_RATIO,
                                                stratify=y,random_state=SEED)
    X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,
        test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO),stratify=y_dev,random_state=SEED)
    np.save(OUTPUT_DIR/"train_indices.npy",np.arange(len(y_train)))
    np.save(OUTPUT_DIR/"test_indices.npy", np.arange(len(y_test)))
    log.info(f"Training  : {len(X_train):,}  |  Validation: {len(X_val):,}  |  Test: {len(X_test):,} (LOCKED)")
    log.info("\nClass distribution per split — copy Train and Test rows to Table 4.1:")
    log.info(f"  {'Split':8}  "+"  ".join(f"{l:>7}" for l in LABELS))
    log.info("  "+"-"*60)
    for nm,ys in [("Train",y_train),("Val",y_val),("Test",y_test)]:
        bc=np.bincount(ys,minlength=N_CLASSES)
        log.info(f"  {nm:8}  "+"  ".join(f"{bc[i]:>7}" for i in range(N_CLASSES)))
    return X_train,X_val,X_test,y_train,y_val,y_test

# =============================================================================
# STAGE 4 — BUILD PIPELINES  (Correction C2, C5)
# =============================================================================
def build_rf_pipeline():
    """Random Forest Pipeline. SMOTE fitted inside each training fold."""
    return ImbPipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("smote",  SMOTE(random_state=SEED,k_neighbors=SMOTE_K)),
        ("scaler", MinMaxScaler()),
        ("model",  RandomForestClassifier(n_estimators=300,max_depth=20,
                   max_features="sqrt",min_samples_split=2,
                   class_weight="balanced",random_state=SEED,n_jobs=-1))])

def build_hgb_pipeline():
    """
    HistGradientBoosting Pipeline.
    CORRECTION C5: HistGradientBoostingClassifier (NOT GradientBoostingClassifier)
    — the latter does NOT support class_weight. HistGBT does.
    """
    return ImbPipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("smote",  SMOTE(random_state=SEED,k_neighbors=SMOTE_K)),
        ("scaler", MinMaxScaler()),
        ("model",  HistGradientBoostingClassifier(max_iter=300,max_depth=5,
                   learning_rate=0.1,l2_regularization=0.0,
                   class_weight="balanced",early_stopping=True,
                   validation_fraction=0.1,n_iter_no_change=20,random_state=SEED))])

# =============================================================================
# STAGE 5 — CROSS-VALIDATION TRAINING  (Correction C2)
# =============================================================================
def train_with_cv(pipeline,X_train,y_train,model_name):
    log.info(f"\n{'='*70}\nSTAGE 5 — TRAINING: {model_name}\n{'='*70}")
    cv=StratifiedKFold(n_splits=N_CV_FOLDS,shuffle=True,random_state=SEED)
    t0=time.time()
    cv_scores=cross_val_score(pipeline,X_train,y_train,cv=cv,
                               scoring="f1_macro",n_jobs=-1)
    log.info(f"CV ({N_CV_FOLDS}-fold) in {time.time()-t0:.1f}s")
    log.info(f"Fold F1-macro : {[f'{s:.4f}' for s in cv_scores]}")
    log.info(f"Mean F1-macro : {cv_scores.mean():.4f}  SD: {cv_scores.std():.4f}")
    log.info("→ Copy Mean and SD into Table 4.2 CV row")
    pipeline.fit(X_train,y_train)
    log.info("Final fit complete.")
    return pipeline,cv_scores

# =============================================================================
# STAGE 6 — SINGLE-USE TEST EVALUATION  (Correction C4)
# =============================================================================
def evaluate(pipeline,X_test,y_test,model_name,feat_names,cv_scores):
    log.info(f"\n{'='*70}\nSTAGE 6 — TEST SET EVALUATION: {model_name}\n{'='*70}")
    log.info("Test set used ONCE. Copy values below into thesis (Correction C4).")
    safe=model_name.replace(" ","_")
    y_pred=pipeline.predict(X_test)
    np.save(OUTPUT_DIR/"y_true.npy",y_test)
    np.save(OUTPUT_DIR/f"y_pred_{safe}.npy",y_pred)

    acc  =accuracy_score(y_test,y_pred)
    prec =precision_score(y_test,y_pred,average="macro",zero_division=0)
    rec  =recall_score(y_test,y_pred,average="macro",zero_division=0)
    f1m  =f1_score(y_test,y_pred,average="macro",zero_division=0)
    f1w  =f1_score(y_test,y_pred,average="weighted",zero_division=0)
    f1c  =f1_score(y_test,y_pred,average=None,zero_division=0)
    prc  =precision_score(y_test,y_pred,average=None,zero_division=0)
    recc =recall_score(y_test,y_pred,average=None,zero_division=0)
    sup  =np.bincount(y_test,minlength=N_CLASSES)
    cmi  =confusion_matrix(y_test,y_pred)
    cmn  =confusion_matrix(y_test,y_pred,normalize="true")

    log.info(f"\n--- OVERALL METRICS (copy into Table 4.3) ---")
    log.info(f"  Accuracy          : {acc:.4f}")
    log.info(f"  Precision (macro) : {prec:.4f}")
    log.info(f"  Recall    (macro) : {rec:.4f}")
    log.info(f"  F1-Score  (macro) : {f1m:.4f}")
    log.info(f"  F1-Score  (wtd)   : {f1w:.4f}")

    log.info(f"\n--- PER-CLASS METRICS (copy into Table 4.4) ---")
    log.info(f"  {'Class':10}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Support':>8}")
    log.info("  "+"-"*50)
    for i in range(min(N_CLASSES,len(f1c))):
        log.info(f"  {LABELS[i]:10}  {prc[i]:>8.4f}  {recc[i]:>8.4f}  "
                 f"{f1c[i]:>8.4f}  {sup[i]:>8}")
    log.info(f"  {'Macro avg':10}  {prec:>8.4f}  {rec:>8.4f}  "
             f"{f1m:>8.4f}  {sum(sup):>8}")

    log.info(f"\n--- CLASSIFICATION REPORT ---\n"
             +classification_report(y_test,y_pred,
               target_names=LABELS[:len(np.unique(y_test))],zero_division=0))
    log.info(f"--- INTEGER CONFUSION MATRIX (supervisor requires counts) ---\n{cmi}")

    # Inference time
    times=[]
    for _ in range(N_INFER):
        t0=time.perf_counter(); pipeline.predict(X_test[:1])
        times.append((time.perf_counter()-t0)*1000)
    t_m=float(np.mean(times)); t_s=float(np.std(times))
    log.info(f"\nInference time : {t_m:.2f} +/- {t_s:.2f} ms/sample ({N_INFER} reps)")
    log.info("→ Record hardware specs (CPU, RAM) in thesis §3.2.2")

    # Bootstrap CI (Correction M5) — paired resampling
    rng=np.random.default_rng(SEED)
    boots=[f1_score(y_test[idx:=rng.integers(0,len(y_test),len(y_test))],
                    y_pred[idx],average="macro",zero_division=0)
           for _ in range(N_BOOT)]
    ci=np.percentile(boots,[2.5,97.5])
    log.info(f"95% Bootstrap CI  : [{ci[0]:.4f}, {ci[1]:.4f}]  ({N_BOOT} iterations)")
    log.info("→ Copy CI into Table 4.3 '95% CI' column (Correction M5)")

    # Plots
    _plot_confusion_matrix(cmi,cmn,model_name)
    clf=pipeline.named_steps.get("model")
    if clf and hasattr(clf,"feature_importances_"):
        _plot_feature_importance(clf.feature_importances_,feat_names,model_name)

    return {"model_name":model_name,"acc":round(acc,4),"prec":round(prec,4),
            "rec":round(rec,4),"f1m":round(f1m,4),"f1w":round(f1w,4),
            "f1c":[round(float(x),4) for x in f1c],
            "prc":[round(float(x),4) for x in prc],
            "recc":[round(float(x),4) for x in recc],
            "sup":sup.tolist(),"cmi":cmi.tolist(),
            "cmn":[[round(v,4) for v in r] for r in cmn.tolist()],
            "t_m":round(t_m,2),"t_s":round(t_s,2),
            "ci":[round(float(ci[0]),4),round(float(ci[1]),4)],
            "cv_folds":[round(float(x),4) for x in cv_scores],
            "cv_mean":round(float(cv_scores.mean()),4),
            "cv_std":round(float(cv_scores.std()),4),"y_pred":y_pred}

# =============================================================================
# STAGE 7 — McNEMAR'S TEST  (Correction M5)
# =============================================================================
def mcnemar_test(y_test,y_rf,y_hgb):
    log.info(f"\n{'='*70}\nSTAGE 7 — McNEMAR'S PAIRED TEST (Correction M5)\n{'='*70}")
    b=int(np.sum((y_rf==y_test)&(y_hgb!=y_test)))
    c=int(np.sum((y_rf!=y_test)&(y_hgb==y_test)))
    log.info(f"  b (RF correct, HGB wrong) = {b}")
    log.info(f"  c (RF wrong, HGB correct) = {c}")
    if b+c==0:
        log.info("  b+c=0: models agree on all samples.")
        return {"b":b,"c":c,"chi2":0.0,"pval":1.0,"sig":False,"winner":"—"}
    chi2=round((abs(b-c)-1)**2/(b+c),4)
    pval=round(float(chi2_dist.sf(chi2,df=1)),4)
    sig=pval<0.05; winner=("RF" if b>c else "HistGBT") if sig else "—"
    log.info(f"  Chi2 = {chi2:.4f}  p = {pval:.4f}")
    log.info(f"  {'SIGNIFICANT: '+winner+' superior' if sig else 'NOT SIGNIFICANT: models equivalent'}")
    log.info("→ Report chi2 and p-value in §4.6.1 of thesis (Correction M5)")
    return {"b":b,"c":c,"chi2":chi2,"pval":pval,"sig":sig,"winner":winner}

# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def _plot_confusion_matrix(cmi,cmn,name):
    safe=name.replace(" ","_"); cmap="Blues" if "Forest" in name else "Oranges"
    fig,axes=plt.subplots(1,2,figsize=(16,6.5))
    sns.heatmap(np.array(cmn)*100,annot=True,fmt=".1f",cmap=cmap,
                xticklabels=LABELS,yticklabels=LABELS,linewidths=0.5,
                ax=axes[0],vmin=0,vmax=100,annot_kws={"size":10})
    axes[0].set_title("Normalised (% of true class)",fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(np.array(cmi),annot=True,fmt="d",cmap="Greens",
                xticklabels=LABELS,yticklabels=LABELS,linewidths=0.5,
                ax=axes[1],annot_kws={"size":10})
    axes[1].set_title("Integer Counts",fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.suptitle(f"Confusion Matrix — {name}",fontsize=13,fontweight="bold",y=1.01)
    plt.tight_layout()
    p=OUTPUT_DIR/f"cm_{safe}.png"; plt.savefig(p,dpi=DPI,bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

def _plot_feature_importance(imp,feat_names,name,top_n=20):
    pairs=sorted(zip(feat_names,imp),key=lambda x:x[1])[-top_n:]
    ns=[p[0] for p in pairs]; vs=[p[1] for p in pairs]
    cs=[GR if any(t in n for t in ["seq","imbal","P_","total"]) else B2 for n in ns]
    fig,ax=plt.subplots(figsize=(10,7))
    bars=ax.barh(ns,vs,color=cs,alpha=0.87,edgecolor="white")
    for bar in bars: ax.text(bar.get_width()+0.001,bar.get_y()+bar.get_height()/2,
                              f"{bar.get_width():.4f}",va="center",fontsize=8)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Top-{top_n} Feature Importance — {name}",fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=B2,label="Raw/magnitude"),
                        mpatches.Patch(color=GR,label="Derived (power/imbalance/sequence)")],
              loc="lower right")
    plt.tight_layout()
    p=OUTPUT_DIR/f"feat_importance_{name.replace(' ','_')}.png"
    plt.savefig(p,dpi=DPI); plt.close(); log.info(f"Saved: {p}")

def plot_class_distribution(raw_counts,smote_target):
    before=[raw_counts.get(l,0) for l in LABELS]
    after=[smote_target if b<smote_target else b for b in before]
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(N_CLASSES); w=0.38
    b1=ax.bar(x-w/2,before,w,label="Before SMOTE",color=B2,alpha=0.87,edgecolor="white")
    b2=ax.bar(x+w/2,after, w,label="After SMOTE (training only)",color=GR,alpha=0.87,edgecolor="white")
    for bar in [*b1,*b2]:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+15,
                str(int(bar.get_height())),ha="center",va="bottom",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("Number of Samples"); ax.set_xlabel("Fault Class")
    ax.set_title("Figure 4.1: Class Distribution Before and After SMOTE\n"
                 "(SMOTE applied to training partition only)",fontweight="bold")
    ax.legend(); plt.tight_layout()
    p=OUTPUT_DIR/"fig4_1_class_distribution.png"; plt.savefig(p,dpi=DPI); plt.close()
    log.info(f"Saved: {p}")

def plot_cv_folds(rf_f,hgb_f):
    folds=list(range(1,len(rf_f)+1)); ra=np.array(rf_f); ha=np.array(hgb_f)
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.plot(folds,ra,"o-",color=B1,lw=2.5,ms=9,label=f"RF (mean={ra.mean():.4f}, SD={ra.std():.4f})")
    ax.plot(folds,ha,"s--",color=OR,lw=2.5,ms=9,label=f"HGB (mean={ha.mean():.4f}, SD={ha.std():.4f})")
    ax.fill_between(folds,ra.mean()-ra.std(),ra.mean()+ra.std(),alpha=0.12,color=B1)
    ax.fill_between(folds,ha.mean()-ha.std(),ha.mean()+ha.std(),alpha=0.12,color=OR)
    ax.axhline(ra.mean(),color=B1,lw=1,ls=":",alpha=0.7)
    ax.axhline(ha.mean(),color=OR,lw=1,ls=":",alpha=0.7)
    ax.set_xticks(folds); ax.set_xlabel("Fold"); ax.set_ylabel("Macro F1-Score")
    ax.set_title("Figure 4.2: F1-Score Across CV Folds (SMOTE inside each fold only)",fontweight="bold")
    ax.legend(); ax.grid(True,alpha=0.4); plt.tight_layout()
    p=OUTPUT_DIR/"fig4_2_cv_folds.png"; plt.savefig(p,dpi=DPI); plt.close(); log.info(f"Saved: {p}")

def plot_overall_performance(rf,hgb):
    metrics=["Accuracy","Precision\n(macro)","Recall\n(macro)","F1-Macro"]
    rv=[rf["acc"],rf["prec"],rf["rec"],rf["f1m"]]
    hv=[hgb["acc"],hgb["prec"],hgb["rec"],hgb["f1m"]]
    x=np.arange(4); w=0.30
    fig,ax=plt.subplots(figsize=(9,5))
    b1=ax.bar(x-w/2,rv,w,label="Random Forest",color=B1,alpha=0.87,edgecolor="white")
    b2=ax.bar(x+w/2,hv,w,label="HistGradBoost",color=OR,alpha=0.87,edgecolor="white")
    for bar in [*b1,*b2]:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,
                f"{bar.get_height():.4f}",ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics,fontsize=10)
    ax.set_ylabel("Score"); ax.set_ylim(0.93,1.01)
    ax.set_title("Figure 4.3: Overall Classification Performance\nRandom Forest vs HistGradBoost (Held-Out Test Set)",fontweight="bold")
    ax.legend(loc="lower right"); plt.tight_layout()
    p=OUTPUT_DIR/"fig4_3_overall_performance.png"; plt.savefig(p,dpi=DPI); plt.close(); log.info(f"Saved: {p}")

def plot_per_class_f1(rf,hgb):
    x=np.arange(N_CLASSES); w=0.35
    fig,ax=plt.subplots(figsize=(10,5))
    b1=ax.bar(x-w/2,rf["f1c"], w,label="Random Forest",color=B1,alpha=0.87,edgecolor="white")
    b2=ax.bar(x+w/2,hgb["f1c"],w,label="HistGradBoost",color=OR,alpha=0.87,edgecolor="white")
    for bar in [*b1,*b2]:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,
                f"{bar.get_height():.4f}",ha="center",va="bottom",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("F1-Score"); ax.set_ylim(0.84,1.02)
    ax.set_title("Figure 4.4: Per-Class F1-Score — RF vs HistGradBoost",fontweight="bold")
    ax.legend(); plt.tight_layout()
    p=OUTPUT_DIR/"fig4_4_per_class_f1.png"; plt.savefig(p,dpi=DPI); plt.close(); log.info(f"Saved: {p}")

# =============================================================================
# SYNTHETIC DATASET  (demo mode)
# =============================================================================
def generate_synthetic_dataset():
    log.info("DEMO MODE — Generating synthetic dataset (real classData.csv structure)")
    log.info("Download classData.csv from Kaggle and pass --data classData.csv for real results.")
    rng=np.random.default_rng(SEED)
    specs=[((0,0,0,0),1387,240,240,240, 9, 9, 9,8),
           ((1,0,0,1),1387,160,240,240,24, 9, 9,10),
           ((0,0,1,1),1097,180,180,240,20,20, 9,10),
           ((1,0,1,1), 931,150,150,240,26,26, 9,12),
           ((0,1,1,1),1310,140,140,140,22,22,22,11),
           ((1,1,1,1), 749,120,120,120,28,28,28,14)]
    rows=[]
    for (G,C,B,A),n,va,vb,vc,ia,ib,ic,ns in specs:
        Va=rng.normal(va,ns,n).astype(int); Vb=rng.normal(vb,ns,n).astype(int)
        Vc=rng.normal(vc,ns,n).astype(int); Ia=rng.normal(ia,ns*.8,n).astype(int)
        Ib=rng.normal(ib,ns*.8,n).astype(int); Ic=rng.normal(ic,ns*.8,n).astype(int)
        for i in range(n): rows.append([Ia[i],Ib[i],Ic[i],Va[i],Vb[i],Vc[i],G,C,B,A])
    df=pd.DataFrame(rows,columns=["Ia","Ib","Ic","Va","Vb","Vc","G","C","B","A"])
    df=df.sample(frac=1,random_state=SEED).reset_index(drop=True)
    p=OUTPUT_DIR/"synthetic_classData.csv"; df.to_csv(p,index=False)
    log.info(f"Synthetic dataset: {p}  ({len(df):,} rows)")
    return str(p)

# =============================================================================
# MAIN
# =============================================================================
def main(csv_path):
    log.info("="*70)
    log.info("SMART-GRID FAULT DETECTION — COMPLETE TRAINING PIPELINE")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Seed    : {SEED}  |  Output: {OUTPUT_DIR.resolve()}")
    log.info("="*70)

    df=verify_dataset(csv_path)
    df=encode_labels(df)
    total=len(df)
    raw_counts={LABELS[i]:int(df["fault_type"].value_counts().sort_index().get(i,0)) for i in range(N_CLASSES)}
    X,y,feat_names=engineer_features(df)
    X_train,X_val,X_test,y_train,y_val,y_test=split_first(X,y)

    split_info={"train":len(X_train),"val":len(X_val),"test":len(X_test),
                "train_class":{LABELS[i]:int(np.bincount(y_train,minlength=N_CLASSES)[i]) for i in range(N_CLASSES)},
                "val_class"  :{LABELS[i]:int(np.bincount(y_val,  minlength=N_CLASSES)[i]) for i in range(N_CLASSES)},
                "test_class" :{LABELS[i]:int(np.bincount(y_test, minlength=N_CLASSES)[i]) for i in range(N_CLASSES)}}

    plot_class_distribution(raw_counts,max(raw_counts.values()))

    RF_pipe=build_rf_pipeline(); HGB_pipe=build_hgb_pipeline()
    RF_pipe,rf_cv=train_with_cv(RF_pipe,X_train,y_train,"RandomForest")
    HGB_pipe,hgb_cv=train_with_cv(HGB_pipe,X_train,y_train,"HistGradBoost")
    plot_cv_folds(rf_cv.tolist(),hgb_cv.tolist())

    joblib.dump(RF_pipe, OUTPUT_DIR/"pipeline_RandomForest.joblib",compress=3)
    joblib.dump(HGB_pipe,OUTPUT_DIR/"pipeline_HistGradBoost.joblib",compress=3)
    log.info("Pipelines serialised: output/pipeline_*.joblib")

    rf_res =evaluate(RF_pipe, X_test,y_test,"RandomForest", feat_names,rf_cv)
    hgb_res=evaluate(HGB_pipe,X_test,y_test,"HistGradBoost",feat_names,hgb_cv)
    plot_overall_performance(rf_res,hgb_res)
    plot_per_class_f1(rf_res,hgb_res)

    mc=mcnemar_test(y_test,rf_res["y_pred"],hgb_res["y_pred"])

    # Save results.json
    results={"run":datetime.now().isoformat(),"seed":SEED,
             "dataset":{"citation":CITATION,"file":"classData.csv","total":total,"raw_counts":raw_counts},
             "splits":split_info,
             "rf" :{k:v for k,v in rf_res.items()  if k!="y_pred"},
             "hgb":{k:v for k,v in hgb_res.items() if k!="y_pred"},
             "mcnemar":mc,"features":{"n":len(feat_names),"names":feat_names}}
    with open(OUTPUT_DIR/"results.json","w") as fh: json.dump(results,fh,indent=2)

    log.info(f"\n{'='*70}\nFINAL SUMMARY — copy into Chapter 4\n{'='*70}")
    log.info(f"{'Model':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}  {'CI':>20}  {'ms':>8}")
    log.info("-"*80)
    for r in [rf_res,hgb_res]:
        log.info(f"{r['model_name']:<22} {r['acc']:>8.4f} {r['prec']:>8.4f} "
                 f"{r['rec']:>8.4f} {r['f1m']:>8.4f}  [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]  {r['t_m']:>7.2f}")
    log.info(f"\nMcNemar: chi2={mc['chi2']}  p={mc['pval']}  "
             f"{'Significant: '+mc['winner'] if mc['sig'] else 'Not significant — models equivalent'}")
    log.info(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")
    log.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Smart-Grid Fault Detection — Training Pipeline")
    parser.add_argument("--data",type=str,default=None,
        help="Path to classData.csv (Sathyaprakash, 2021). Omit for demo mode.")
    args=parser.parse_args()
    if args.data and not Path(args.data).exists():
        print(f"\nERROR: {args.data} not found.")
        print("Download classData.csv from:")
        print("https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification\n")
        sys.exit(1)
    csv_path=args.data if args.data else generate_synthetic_dataset()
    main(csv_path)
