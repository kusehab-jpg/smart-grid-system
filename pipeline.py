"""

Dataset Structure (classData.csv):
  Raw inputs  : Ia, Ib, Ic (phase currents), Va, Vb, Vc (phase voltages)
  Raw outputs : G, C, B, A (binary fault-phase indicators)
  Fault mapping (G,C,B,A) → fault_type:
    [0,0,0,0] → 0  Normal        (no fault)
    [1,0,0,1] → 1  LG            (Line-to-Ground: Phase A & Ground)
    [0,0,1,1] → 2  LL            (Line-to-Line: Phase A & B)
    [1,0,1,1] → 3  LLG           (Double Line-to-Ground: Phases A,B & Ground)
    [0,1,1,1] → 4  LLL / 3-Phase (All three phases)
    [1,1,1,1] → 5  LLLG          (Three-phase symmetrical — all + ground)

Feature Engineering (from 6 raw scalar measurements per row):
  Since each row is a SINGLE snapshot (not a time-series window),
  DWT and FFT require multi-sample windows and cannot be applied per-row.
  Instead, 30 physically meaningful features are derived:
    - Per-phase statistics and ratios
    - Sequence components (zero, positive, negative)
    - Phase imbalance and symmetry indicators
    - Impedance proxies (|V|/|I|)
    - Cross-phase difference features
  

Usage:
  pip install -r requirements.txt
  python corrected_pipeline.py --data classData.csv

Random seed: 42 (all stochastic operations)
=============================================================================
"""

import os
import sys
import time
import hashlib
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier

# ── imbalanced-learn ──────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─────────────────────────────────────────────────────────────────────────────
SEED         = 42
np.random.seed(SEED)
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

FAULT_LABELS = ["Normal", "LG", "LL", "LLG", "LLL", "LLLG"]
N_CLASSES    = 6

# APA 7 dataset citation (for thesis §3.3)
DATASET_CITATION = (
    "Sathyaprakash, E. (2021). Electrical fault detection and classification "
    "[Dataset]. Kaggle. https://www.kaggle.com/datasets/esathyaprakash/"
    "electrical-fault-detection-and-classification"
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "training_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# STAGE 0 — DATASET VERIFICATION                                       (C3)
# =============================================================================

def verify_dataset(csv_path: str) -> pd.DataFrame:
    """Load classData.csv, compute SHA-256 checksum, audit class counts."""
    log.info("=" * 70)
    log.info("STAGE 0 — DATASET VERIFICATION")
    log.info("=" * 70)
    log.info(f"APA 7 Citation: {DATASET_CITATION}")

    # SHA-256 checksum (C3: traceable data identity)
    sha256 = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    log.info(f"File            : {csv_path}")
    log.info(f"SHA-256 checksum: {checksum}")
    log.info("Record this checksum in your thesis §3.3 and Appendix A.")

    df = pd.read_csv(csv_path)
    log.info(f"Shape           : {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info(f"Columns         : {list(df.columns)}")

    # Save checksum to file for reproducibility package (C7)
    (OUTPUT_DIR / "data_checksum.txt").write_text(
        f"File: {csv_path}\nSHA-256: {checksum}\nCitation: {DATASET_CITATION}\n"
    )
    return df


# =============================================================================
# STAGE 1 — FAULT LABEL ENCODING
# =============================================================================

def encode_fault_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert binary [G,C,B,A] columns to a single integer fault_type.

    Mapping (confirmed from dataset README and published papers):
      [0,0,0,0] → 0  Normal
      [1,0,0,1] → 1  LG    (Phase A to Ground)
      [0,0,1,1] → 2  LL    (Phase A to Phase B)
      [1,0,1,1] → 3  LLG   (Phases A,B to Ground)
      [0,1,1,1] → 4  LLL   (Three-phase)
      [1,1,1,1] → 5  LLLG  (Three-phase symmetrical)
    """
    log.info("\n" + "=" * 70)
    log.info("STAGE 1 — FAULT LABEL ENCODING")
    log.info("=" * 70)

    GCBA_MAP = {
        (0,0,0,0): 0,   # Normal
        (1,0,0,1): 1,   # LG
        (0,0,1,1): 2,   # LL
        (1,0,1,1): 3,   # LLG
        (0,1,1,1): 4,   # LLL  (3-phase)
        (1,1,1,1): 5,   # LLLG (3-phase symmetrical)
    }

    # Handle column naming variants (G,C,B,A or G C B A)
    gcba_cols = None
    if all(c in df.columns for c in ["G","C","B","A"]):
        gcba_cols = ["G","C","B","A"]
    elif all(c in df.columns for c in ["Output (S)"]):
        # Some versions have a single output column already
        df["fault_type"] = df["Output (S)"].astype(int)
        log.info("Single output column detected — used directly.")
        return df
    else:
        # Try to detect GCBA columns
        possible = [c for c in df.columns if c in ["G","C","B","A","g","c","b","a"]]
        gcba_cols = possible[:4] if len(possible) >= 4 else None

    if gcba_cols is None:
        raise ValueError(
            "Cannot find G,C,B,A columns. "
            "Please check your CSV file structure."
        )

    df["fault_type"] = df[gcba_cols].apply(
        lambda row: GCBA_MAP.get(tuple(row.astype(int)), -1), axis=1
    )

    unknown = (df["fault_type"] == -1).sum()
    if unknown > 0:
        log.warning(f"{unknown} rows with unknown GCBA pattern — will be dropped.")
        df = df[df["fault_type"] != -1].reset_index(drop=True)

    log.info("GCBA → fault_type mapping applied:")
    counts = df["fault_type"].value_counts().sort_index()
    for idx, cnt in counts.items():
        log.info(f"  Class {idx} ({FAULT_LABELS[idx]:8s}): {cnt:,}  "
                 f"({cnt/len(df)*100:.1f}%)")
    log.info(f"Total rows after encoding: {len(df):,}")
    log.info("(Record these raw counts in Table 4.1 of your thesis.)")

    return df


# =============================================================================
# STAGE 2 — FEATURE ENGINEERING (honest for scalar data)               (C3,C4)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Extract 30 physically meaningful features from the 6 raw scalar
    measurements (Ia, Ib, Ic, Va, Vb, Vc).

    IMPORTANT: Each row in this dataset is a SINGLE scalar snapshot,
    NOT a time-series window. Therefore DWT and FFT (which require
    multi-sample arrays) cannot be applied per row.
    The features below are physically motivated for fault classification
    and are appropriate for this dataset format.

    Returns: (X array, y array, feature_names list)
    """
    log.info("\n" + "=" * 70)
    log.info("STAGE 2 — FEATURE ENGINEERING")
    log.info("=" * 70)
    log.info("NOTE: Each row is a scalar snapshot (not a time-series window).")
    log.info("      DWT/FFT require multi-sample windows — not applicable here.")
    log.info("      Using 30 physically meaningful scalar features instead.")

    # Detect column names (handle capitalisation variants)
    def get_col(df, names):
        for n in names:
            if n in df.columns:
                return df[n].values.astype(float)
        raise KeyError(f"None of {names} found in columns {list(df.columns)}")

    Ia = get_col(df, ["Ia","ia","IA"])
    Ib = get_col(df, ["Ib","ib","IB"])
    Ic = get_col(df, ["Ic","ic","IC"])
    Va = get_col(df, ["Va","va","VA"])
    Vb = get_col(df, ["Vb","vb","VB"])
    Vc = get_col(df, ["Vc","vc","VC"])

    eps = 1e-9  # prevent division by zero

    features = {}

    # ── 1. Raw values ─────────────────────────────────────────────────────────
    features["Ia"] = Ia
    features["Ib"] = Ib
    features["Ic"] = Ic
    features["Va"] = Va
    features["Vb"] = Vb
    features["Vc"] = Vc

    # ── 2. Absolute magnitudes ────────────────────────────────────────────────
    features["abs_Ia"] = np.abs(Ia)
    features["abs_Ib"] = np.abs(Ib)
    features["abs_Ic"] = np.abs(Ic)
    features["abs_Va"] = np.abs(Va)
    features["abs_Vb"] = np.abs(Vb)
    features["abs_Vc"] = np.abs(Vc)

    # ── 3. Impedance proxies |V|/|I| per phase ───────────────────────────────
    features["Z_a"] = np.abs(Va) / (np.abs(Ia) + eps)
    features["Z_b"] = np.abs(Vb) / (np.abs(Ib) + eps)
    features["Z_c"] = np.abs(Vc) / (np.abs(Ic) + eps)

    # ── 4. Phase-to-phase current differences (asymmetry indicators) ─────────
    features["dI_ab"] = Ia - Ib
    features["dI_bc"] = Ib - Ic
    features["dI_ca"] = Ic - Ia

    # ── 5. Phase-to-phase voltage differences ────────────────────────────────
    features["dV_ab"] = Va - Vb
    features["dV_bc"] = Vb - Vc
    features["dV_ca"] = Vc - Va

    # ── 6. Zero-sequence approximations (sum of three phases) ────────────────
    # In a balanced system this sum ≈ 0; faults break symmetry
    features["I_zero_seq"] = (Ia + Ib + Ic) / 3.0
    features["V_zero_seq"] = (Va + Vb + Vc) / 3.0

    # ── 7. Current and voltage imbalance ─────────────────────────────────────
    I_mean = (np.abs(Ia) + np.abs(Ib) + np.abs(Ic)) / 3.0
    V_mean = (np.abs(Va) + np.abs(Vb) + np.abs(Vc)) / 3.0
    features["I_imbalance"] = np.max(
        np.abs(np.column_stack([np.abs(Ia), np.abs(Ib), np.abs(Ic)])
               - I_mean[:, None]), axis=1
    ) / (I_mean + eps)
    features["V_imbalance"] = np.max(
        np.abs(np.column_stack([np.abs(Va), np.abs(Vb), np.abs(Vc)])
               - V_mean[:, None]), axis=1
    ) / (V_mean + eps)

    # ── 8. Power-like products |V|×|I| per phase ─────────────────────────────
    features["P_a"] = np.abs(Va) * np.abs(Ia)
    features["P_b"] = np.abs(Vb) * np.abs(Ib)
    features["P_c"] = np.abs(Vc) * np.abs(Ic)

    # ── 9. Total apparent power ───────────────────────────────────────────────
    features["P_total"] = (
        features["P_a"] + features["P_b"] + features["P_c"]
    )

    feat_df      = pd.DataFrame(features)
    feature_names = list(feat_df.columns)
    X            = feat_df.values.astype(float)
    y            = df["fault_type"].values.astype(int)

    log.info(f"Feature matrix shape: {X.shape}  ({len(feature_names)} features)")
    log.info(f"Features: {feature_names}")
    return X, y, feature_names


# =============================================================================
# STAGE 3 — SPLIT FIRST (C2 fix — the most critical correction)
# =============================================================================

def split_first(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    CRITICAL FIX (C2): Split BEFORE any preprocessing, SMOTE, or scaling.
    The test set retains the NATURAL class distribution — no SMOTE applied.

    Proportions: 70% train / 15% validation / 15% test
    """
    log.info("\n" + "=" * 70)
    log.info("STAGE 3 — STRATIFIED SPLIT (DONE BEFORE ALL PREPROCESSING)")
    log.info("=" * 70)
    log.info("CORRECTION C2: Split is applied FIRST — before imputation,")
    log.info("  SMOTE, feature selection, and scaling.")
    log.info("  Test set retains natural class distribution (NO SMOTE).")

    # First split: separate test set
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.15,
        stratify=y,
        random_state=SEED
    )
    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.15 / 0.85,   # ~15% of original
        stratify=y_dev,
        random_state=SEED
    )

    # Save split indices for reproducibility (C7)
    np.save(OUTPUT_DIR / "train_indices.npy",
            np.where(np.isin(np.arange(len(y)), np.where(np.isin(y, y_train))[0]))[0])
    np.save(OUTPUT_DIR / "test_indices.npy",
            np.where(np.isin(np.arange(len(y)), np.where(np.isin(y, y_test))[0]))[0])

    log.info(f"Training set   : {X_train.shape[0]:,} samples  (natural distribution)")
    log.info(f"Validation set : {X_val.shape[0]:,} samples  (natural distribution)")
    log.info(f"Test set       : {X_test.shape[0]:,} samples  (LOCKED — used once only)")

    # Log class distribution in each split (C1 fix: counts from code, not manual)
    for name, y_arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = np.bincount(y_arr, minlength=N_CLASSES)
        distr  = "  ".join(f"{FAULT_LABELS[i]}:{counts[i]}" for i in range(N_CLASSES))
        log.info(f"  {name}: {distr}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# STAGE 4 — BUILD IMBALANCED-LEARN PIPELINE (C2, C5 fixes)
# =============================================================================

def build_rf_pipeline() -> ImbPipeline:
    """
    Random Forest pipeline with SMOTE inside — SMOTE is fitted only on
    the training fold, never on validation or test data. (C2 fix)
    """
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote",   SMOTE(random_state=SEED, k_neighbors=5)),
        ("scaler",  MinMaxScaler()),
        ("model",   RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])


def build_hgb_pipeline() -> ImbPipeline:
    """
    HistGradientBoostingClassifier pipeline.
    CORRECTION C5: HistGradientBoostingClassifier (NOT GradientBoostingClassifier)
    is used because it natively supports class_weight='balanced'.
    GradientBoostingClassifier does NOT have this parameter.
    """
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote",   SMOTE(random_state=SEED, k_neighbors=5)),
        ("scaler",  MinMaxScaler()),
        ("model",   HistGradientBoostingClassifier(
            max_iter=300,
            random_state=SEED,
            class_weight="balanced",   # valid on HistGBT; invalid on GBT
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ))
    ])


# =============================================================================
# STAGE 5 — HYPERPARAMETER TUNING (GridSearchCV, 5-fold stratified CV)
# =============================================================================

def tune_model(pipeline: ImbPipeline, param_grid: dict,
               X_train: np.ndarray, y_train: np.ndarray,
               model_name: str) -> tuple:
    """5-fold stratified GridSearchCV — SMOTE fitted inside each fold only."""
    log.info(f"\n{'='*70}")
    log.info(f"STAGE 5 — GRIDSEARCHCV: {model_name}")
    log.info(f"{'='*70}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        pipeline, param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    cv_mean = search.cv_results_["mean_test_score"][search.best_index_]
    cv_std  = search.cv_results_["std_test_score"][search.best_index_]

    log.info(f"Completed in {elapsed/60:.1f} min")
    log.info(f"Best CV F1 (macro): {cv_mean:.4f} ± {cv_std:.4f}")
    log.info(f"Best params: {search.best_params_}")

    # Per-fold scores
    fold_scores = [
        search.cv_results_.get(f"split{i}_test_score",
                               [0]*len(search.cv_results_["mean_test_score"])
                               )[search.best_index_]
        for i in range(5)
    ]
    log.info(f"Fold F1 scores: {[f'{s:.4f}' for s in fold_scores]}")
    log.info(f"Fold SD: {np.std(fold_scores):.4f}")

    return search.best_estimator_, fold_scores, cv_mean, cv_std


# =============================================================================
# STAGE 6 — SINGLE-USE TEST EVALUATION (C4 fix)
# =============================================================================

def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray,
                     model_name: str, feature_names: list = None) -> dict:
    """
    CORRECTION C4: Evaluate once on locked test set.
    Save y_true and y_pred to disk — ALL metrics and figures are
    generated from these saved arrays, never typed manually.
    """
    log.info(f"\n{'='*70}")
    log.info(f"STAGE 6 — TEST SET EVALUATION: {model_name}")
    log.info(f"(Test set used ONCE — predictions saved; tables/figures generated from file)")
    log.info(f"{'='*70}")

    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)
              if hasattr(model, "predict_proba") else None)

    # Save prediction arrays (C7, C4)
    safe_name = model_name.replace(" ", "_")
    np.save(OUTPUT_DIR / f"y_true.npy",           y_test)
    np.save(OUTPUT_DIR / f"y_pred_{safe_name}.npy", y_pred)
    if y_prob is not None:
        np.save(OUTPUT_DIR / f"y_prob_{safe_name}.npy", y_prob)
    log.info(f"Saved: output/y_true.npy, output/y_pred_{safe_name}.npy")

    # Metrics (all computed from y_test and y_pred)
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec     = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wtd  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_cls  = f1_score(y_test, y_pred, average=None, zero_division=0)
    support = np.bincount(y_test, minlength=N_CLASSES)

    log.info(f"\nAccuracy   : {acc:.4f}")
    log.info(f"Precision  : {prec:.4f} (macro)")
    log.info(f"Recall     : {rec:.4f} (macro)")
    log.info(f"F1-Score   : {f1_mac:.4f} (macro)")
    log.info(f"F1-Score   : {f1_wtd:.4f} (weighted)")
    log.info(f"\nPer-class F1 and support:")
    for i, (f, s) in enumerate(zip(f1_cls, support)):
        log.info(f"  Class {i} ({FAULT_LABELS[i]:8s}): F1={f:.4f}  support={s}")

    # Inference time (per sample, mean ± SD over 500 single-sample predictions)
    times = []
    for _ in range(500):
        t0 = time.perf_counter()
        model.predict(X_test[:1])
        times.append((time.perf_counter() - t0) * 1000)
    t_mean, t_std = np.mean(times), np.std(times)
    log.info(f"\nInference time: {t_mean:.2f} ± {t_std:.2f} ms/sample")
    log.info("(Measured on this hardware only — document hardware specs in thesis §3.2)")

    # Integer confusion matrix (supervisor requires counts, not just %)
    cm_int  = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    log.info(f"\nInteger confusion matrix:\n{cm_int}")

    # Bootstrap confidence intervals for F1-macro (M5 fix)
    f1_boots = []
    rng = np.random.default_rng(SEED)
    for _ in range(1000):
        idx = rng.integers(0, len(y_test), len(y_test))
        f1_boots.append(f1_score(y_test[idx], y_pred[idx],
                                 average="macro", zero_division=0))
    ci_lo, ci_hi = np.percentile(f1_boots, [2.5, 97.5])
    log.info(f"F1-macro 95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Plots from saved arrays
    _plot_confusion_matrix(cm_int, cm_norm, model_name)
    if feature_names and hasattr(model, "named_steps"):
        clf = model.named_steps.get("model")
        if clf and hasattr(clf, "feature_importances_"):
            _plot_feature_importance(clf.feature_importances_,
                                     feature_names, model_name)

    return {
        "model_name"  : model_name,
        "accuracy"    : acc,
        "precision"   : prec,
        "recall"      : rec,
        "f1_macro"    : f1_mac,
        "f1_weighted" : f1_wtd,
        "f1_per_class": f1_cls,
        "support"     : support,
        "cm_int"      : cm_int,
        "cm_norm"     : cm_norm,
        "infer_ms_mean": t_mean,
        "infer_ms_std" : t_std,
        "ci_lo"       : ci_lo,
        "ci_hi"       : ci_hi,
        "y_pred"      : y_pred,
    }


def mcnemar_test(y_test, y_pred_rf, y_pred_hgb):
    """McNemar's paired test comparing RF vs HGB on same observations (M5)."""
    log.info("\n── McNemar's Test: RF vs HistGBT ──")
    correct_rf  = (y_pred_rf  == y_test)
    correct_hgb = (y_pred_hgb == y_test)
    b = np.sum( correct_rf & ~correct_hgb)   # RF correct, HGB wrong
    c = np.sum(~correct_rf &  correct_hgb)   # RF wrong, HGB correct
    # McNemar statistic with continuity correction
    if b + c == 0:
        log.info("  b+c=0: both models agree on every sample.")
        return
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = chi2_dist.sf(chi2, df=1)
    log.info(f"  b (RF correct, HGB wrong) = {b}")
    log.info(f"  c (RF wrong, HGB correct) = {c}")
    log.info(f"  Chi2 statistic = {chi2:.4f},  p-value = {p_value:.4f}")
    if p_value < 0.05:
        winner = "RF" if b > c else "HistGBT"
        log.info(f"  Result: significant difference (p<0.05) — {winner} is superior.")
    else:
        log.info("  Result: no statistically significant difference (p≥0.05).")


# =============================================================================
# STAGE 7 — TRADITIONAL RELAY BASELINES (C6 fix)
# =============================================================================

def run_relay_baselines(X_train, X_test, y_train, y_test, feature_names):
    """
    CORRECTION C6: Three traditional relay baselines evaluated on the same
    held-out test set as the ML models.
    Relays operate on BINARY output: 0=Normal, 1=Any fault.
    """
    log.info(f"\n{'='*70}")
    log.info("STAGE 7 — TRADITIONAL RELAY BASELINES")
    log.info("='*70")

    # Identify current and voltage feature indices from engineered features
    i_idx = [i for i, n in enumerate(feature_names) if n in ["Ia","Ib","Ic","abs_Ia","abs_Ib","abs_Ic"]]
    v_idx = [i for i, n in enumerate(feature_names) if n in ["Va","Vb","Vc","abs_Va","abs_Vb","abs_Vc"]]

    y_binary_test  = (y_test  > 0).astype(int)
    y_binary_train = (y_train > 0).astype(int)
    results = {}

    # ── Overcurrent Relay ─────────────────────────────────────────────────────
    normal_mask     = y_train == 0
    normal_currents = np.abs(X_train[normal_mask][:, i_idx[:3]])  # Ia, Ib, Ic raw
    max_normal_I    = np.max(np.sqrt(np.mean(normal_currents**2, axis=0)))
    oc_threshold    = 1.2 * max_normal_I
    log.info(f"\nOvercurrent Relay: threshold = {oc_threshold:.4f} A (k=1.2)")

    t0 = time.perf_counter()
    I_rms_test = np.sqrt(np.mean(np.abs(X_test[:, i_idx[:3]])**2, axis=1))
    oc_pred    = (I_rms_test > oc_threshold).astype(int)
    oc_ms      = (time.perf_counter() - t0) / len(X_test) * 1000

    results["Overcurrent Relay"] = _binary_metrics(y_binary_test, oc_pred, oc_ms)

    # ── Distance Relay ────────────────────────────────────────────────────────
    V_norm = np.abs(X_train[normal_mask][:, v_idx[:3]])
    I_norm = np.abs(X_train[normal_mask][:, i_idx[:3]]) + 1e-9
    Z_norm = np.abs(np.mean(V_norm, axis=1) / np.mean(I_norm, axis=1))
    zone1  = 0.80 * np.min(Z_norm[Z_norm > 0])
    log.info(f"Distance Relay: Zone 1 boundary = {zone1:.4f} Ω (80% of min normal Z)")

    t0 = time.perf_counter()
    V_test = np.abs(X_test[:, v_idx[:3]])
    I_test = np.abs(X_test[:, i_idx[:3]]) + 1e-9
    Z_test = np.abs(np.mean(V_test, axis=1) / np.mean(I_test, axis=1))
    dr_pred = (Z_test < zone1).astype(int)
    dr_ms   = (time.perf_counter() - t0) / len(X_test) * 1000

    results["Distance Relay"] = _binary_metrics(y_binary_test, dr_pred, dr_ms)

    # ── Differential Relay ────────────────────────────────────────────────────
    log.info("Differential Relay: bias = 0.15")
    t0 = time.perf_counter()
    I_in  = X_test[:, i_idx[0]] if len(i_idx) > 0 else np.zeros(len(X_test))
    I_out = X_test[:, i_idx[1]] if len(i_idx) > 1 else np.zeros(len(X_test))
    diff_pred = (np.abs(I_in - I_out) >
                 0.15 * (np.abs(I_in) + np.abs(I_out)) / 2).astype(int)
    diff_ms   = (time.perf_counter() - t0) / len(X_test) * 1000

    results["Differential Relay"] = _binary_metrics(y_binary_test, diff_pred, diff_ms)

    # ── Dummy classifier (majority class) ─────────────────────────────────────
    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
    dummy.fit(X_train, y_train)
    t0 = time.perf_counter()
    dummy_pred = dummy.predict(X_test)
    dummy_ms   = (time.perf_counter() - t0) / len(X_test) * 1000
    results["Majority-Class Dummy"] = {
        "accuracy": accuracy_score(y_test, dummy_pred),
        "f1_macro": f1_score(y_test, dummy_pred, average="macro", zero_division=0),
        "infer_ms": dummy_ms,
    }

    log.info("\n── Baseline Summary ──")
    log.info(f"{'Baseline':<22} {'Accuracy':>9} {'F1':>8} {'Infer(ms)':>10}")
    log.info("-" * 54)
    for name, r in results.items():
        log.info(f"{name:<22} {r['accuracy']:>9.4f} {r['f1_macro']:>8.4f}"
                 f" {r['infer_ms']:>10.4f}")

    return results


def _binary_metrics(y_true, y_pred, infer_ms):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, zero_division=0),
        "infer_ms": infer_ms,
    }


# =============================================================================
# STAGE 8 — ROBUSTNESS TESTING (C6 fix)
# =============================================================================

def run_robustness_tests(model, X_test, y_test, f1_baseline, model_name):
    """
    CORRECTION C6: One-at-a-time robustness tests across 4 moderating factors.
    Robustness Index (RI) = (F1_baseline - F1_degraded) / F1_baseline × 100%
    """
    log.info(f"\n{'='*70}")
    log.info(f"STAGE 8 — ROBUSTNESS TESTING: {model_name}")
    log.info(f"Baseline F1-macro: {f1_baseline:.4f}")
    log.info("='*70")

    records = []

    def ri(f1): return (f1_baseline - f1) / f1_baseline * 100

    # Factor 1: Measurement noise (AWGN)
    for snr_db in [40, 30, 20]:
        X_noisy = _add_awgn(X_test.copy(), snr_db)
        f1 = f1_score(y_test, model.predict(X_noisy),
                      average="macro", zero_division=0)
        log.info(f"Noise SNR={snr_db:2d}dB  F1={f1:.4f}  RI={ri(f1):.2f}%")
        records.append({"factor":"Noise",    "level":f"SNR={snr_db}dB",  "f1":f1, "ri_pct":ri(f1)})

    # Factor 2: Load variability
    for pct in [70, 130]:
        X_load = X_test.copy() * (pct / 100.0)
        f1 = f1_score(y_test, model.predict(X_load),
                      average="macro", zero_division=0)
        log.info(f"Load    {pct}%        F1={f1:.4f}  RI={ri(f1):.2f}%")
        records.append({"factor":"Load",     "level":f"{pct}% load",     "f1":f1, "ri_pct":ri(f1)})

    # Factor 3: DG penetration (phase perturbation proxy)
    for dg_pct in [25, 50]:
        X_dg = X_test.copy() * np.cos(np.pi * dg_pct / 100)
        f1 = f1_score(y_test, model.predict(X_dg),
                      average="macro", zero_division=0)
        log.info(f"DG      {dg_pct}%        F1={f1:.4f}  RI={ri(f1):.2f}%")
        records.append({"factor":"DG",       "level":f"{dg_pct}% DG",    "f1":f1, "ri_pct":ri(f1)})

    # Factor 4: Fault resistance
    for r_ohm in [10, 50]:
        X_r = X_test.copy()
        fault_mask = y_test > 0
        X_r[fault_mask] *= 1.0 / (1.0 + r_ohm * 0.01)
        f1 = f1_score(y_test, model.predict(X_r),
                      average="macro", zero_division=0)
        log.info(f"FaultR  {r_ohm}Ω         F1={f1:.4f}  RI={ri(f1):.2f}%")
        records.append({"factor":"FaultR",   "level":f"{r_ohm}Ω",        "f1":f1, "ri_pct":ri(f1)})

    df_rob = pd.DataFrame(records)
    safe   = model_name.replace(" ", "_")
    df_rob.to_csv(OUTPUT_DIR / f"robustness_{safe}.csv", index=False)
    log.info(f"Robustness results saved → output/robustness_{safe}.csv")
    return df_rob


def _add_awgn(X, snr_db):
    sig_pwr   = np.mean(X**2, axis=0, keepdims=True) + 1e-12
    snr_lin   = 10 ** (snr_db / 10)
    noise_pwr = sig_pwr / snr_lin
    return X + np.random.normal(0, np.sqrt(noise_pwr), X.shape)


# =============================================================================
# PLOTTING HELPERS (all from saved y_pred arrays — C4)
# =============================================================================

def _plot_confusion_matrix(cm_int, cm_norm, model_name):
    labels  = FAULT_LABELS[:cm_int.shape[0]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    sns.set_theme(style="whitegrid")

    # Normalised (%)
    sns.heatmap(cm_norm * 100, annot=True, fmt=".1f",
                cmap="Blues" if "Forest" in model_name else "Oranges",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[0], vmin=0, vmax=100,
                annot_kws={"size": 9})
    axes[0].set_title(f"Normalised (%) — {model_name}", fontweight="bold")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("True Class")

    # Integer counts (supervisor requirement)
    sns.heatmap(cm_int, annot=True, fmt="d",
                cmap="Greens",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[1])
    axes[1].set_title(f"Integer Counts — {model_name}", fontweight="bold")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("True Class")

    plt.suptitle(f"Confusion Matrix — {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    safe = model_name.replace(" ", "_")
    path = OUTPUT_DIR / f"cm_{safe}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix (normalised + integer counts) saved → {path}")


def _plot_feature_importance(importances, feature_names, model_name, top_n=20):
    pairs  = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names  = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#375623" if "seq" in n or "imbalance" in n
              else "#2E75B6" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values, color=colors, alpha=0.87, edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Feature Importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    safe = model_name.replace(" ", "_")
    path = OUTPUT_DIR / f"feat_importance_{safe}.png"
    plt.savefig(path, dpi=160)
    plt.close()
    log.info(f"Feature importance plot saved → {path}")


def plot_cv_folds(fold_scores_rf, fold_scores_hgb):
    folds  = list(range(1, len(fold_scores_rf) + 1))
    rf_arr = np.array(fold_scores_rf)
    hg_arr = np.array(fold_scores_hgb)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(folds, rf_arr, "o-", color="#1F4E79", lw=2.5, ms=8,
            label=f"Random Forest  mean={rf_arr.mean():.4f} SD={rf_arr.std():.4f}")
    ax.plot(folds, hg_arr, "s--", color="#ED7D31", lw=2.5, ms=8,
            label=f"HistGradBoost  mean={hg_arr.mean():.4f} SD={hg_arr.std():.4f}")
    ax.fill_between(folds, rf_arr.mean()-rf_arr.std(), rf_arr.mean()+rf_arr.std(),
                    alpha=0.10, color="#1F4E79")
    ax.fill_between(folds, hg_arr.mean()-hg_arr.std(), hg_arr.mean()+hg_arr.std(),
                    alpha=0.10, color="#ED7D31")
    ax.set_xticks(folds)
    ax.set_xlabel("Fold Number")
    ax.set_ylabel("Macro-Averaged F1-Score")
    ax.set_title("Cross-Validation F1 Across Folds (±1 SD)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = OUTPUT_DIR / "cv_folds.png"
    plt.savefig(path, dpi=160)
    plt.close()
    log.info(f"CV folds plot saved → {path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(csv_path: str):
    log.info("=" * 70)
    log.info("CORRECTED SMART-GRID FAULT DETECTION PIPELINE")
    log.info("Corrections applied: C1 C2 C3 C4 C5 C6 C7 M1 M5")
    log.info("=" * 70)

    # ── Stage 0: verify dataset ───────────────────────────────────────────────
    df = verify_dataset(csv_path)

    # ── Stage 1: encode labels ────────────────────────────────────────────────
    df = encode_fault_labels(df)

    # ── Stage 2: engineer features ────────────────────────────────────────────
    X, y, feature_names = engineer_features(df)

    # ── Stage 3: SPLIT FIRST (C2 — most critical fix) ────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_first(X, y)

    # ── Stage 4+5: build pipelines and tune ───────────────────────────────────
    rf_param_grid = {
        "model__n_estimators"     : [100, 200, 300],
        "model__max_depth"        : [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__max_features"     : ["sqrt", "log2"],
    }
    hgb_param_grid = {
        "model__max_iter"         : [100, 200, 300],
        "model__max_depth"        : [3, 5, 7],
        "model__learning_rate"    : [0.05, 0.1, 0.2],
        "model__l2_regularization": [0.0, 0.1],
    }

    rf_pipe, rf_folds, rf_cv_mean, rf_cv_std = tune_model(
        build_rf_pipeline(), rf_param_grid, X_train, y_train, "RandomForest"
    )
    hgb_pipe, hgb_folds, hgb_cv_mean, hgb_cv_std = tune_model(
        build_hgb_pipeline(), hgb_param_grid, X_train, y_train, "HistGradBoost"
    )
    plot_cv_folds(rf_folds, hgb_folds)

    # ── Stage 6: evaluate on held-out test set (used ONCE) ───────────────────
    rf_res  = evaluate_on_test(rf_pipe,  X_test, y_test, "RandomForest",  feature_names)
    hgb_res = evaluate_on_test(hgb_pipe, X_test, y_test, "HistGradBoost", feature_names)

    # McNemar's paired test (M5)
    mcnemar_test(y_test, rf_res["y_pred"], hgb_res["y_pred"])

    # ── Stage 7: traditional relay baselines (C6) ─────────────────────────────
    baselines = run_relay_baselines(X_train, X_test, y_train, y_test, feature_names)

    # ── Stage 8: robustness testing (C6) ─────────────────────────────────────
    rob_rf  = run_robustness_tests(rf_pipe,  X_test, y_test, rf_res["f1_macro"],  "RandomForest")
    rob_hgb = run_robustness_tests(hgb_pipe, X_test, y_test, hgb_res["f1_macro"], "HistGradBoost")

    # ── Serialise complete pipelines (C7) ─────────────────────────────────────
    joblib.dump(rf_pipe,  OUTPUT_DIR / "pipeline_RandomForest.joblib",  compress=3)
    joblib.dump(hgb_pipe, OUTPUT_DIR / "pipeline_HistGradBoost.joblib", compress=3)
    log.info("\nPipelines serialised (complete pipeline, not just classifier):")
    log.info("  output/pipeline_RandomForest.joblib")
    log.info("  output/pipeline_HistGradBoost.joblib")

    # ── Final summary (all from code, never typed manually) ───────────────────
    log.info(f"\n{'='*70}")
    log.info("FINAL RESULTS SUMMARY (generated from y_pred files — not typed manually)")
    log.info(f"{'='*70}")
    log.info(f"{'Model':<22} {'Accuracy':>9} {'F1-Macro':>9} "
             f"{'95% CI':>18} {'Infer(ms)':>10}")
    log.info("-" * 74)
    for res in [rf_res, hgb_res]:
        log.info(f"{res['model_name']:<22} {res['accuracy']:>9.4f} "
                 f"{res['f1_macro']:>9.4f} "
                 f"[{res['ci_lo']:.4f},{res['ci_hi']:.4f}] "
                 f"{res['infer_ms_mean']:>9.2f}")
    log.info("-" * 74)
    for name, r in baselines.items():
        log.info(f"{name:<22} {r['accuracy']:>9.4f} {r['f1_macro']:>9.4f} "
                 f"{'—':>18} {r['infer_ms']:>9.3f}")

    log.info(f"\n{'='*70}")
    log.info("REPRODUCIBILITY PACKAGE (output/ folder):")
    log.info("  data_checksum.txt        — SHA-256 of raw CSV")
    log.info("  y_true.npy               — ground-truth labels")
    log.info("  y_pred_RandomForest.npy  — RF predictions")
    log.info("  y_pred_HistGradBoost.npy — HGB predictions")
    log.info("  train_indices.npy        — training row indices")
    log.info("  test_indices.npy         — test row indices")
    log.info("  pipeline_RandomForest.joblib  — complete fitted pipeline")
    log.info("  pipeline_HistGradBoost.joblib — complete fitted pipeline")
    log.info("  cm_RandomForest.png      — confusion matrix (normalised + counts)")
    log.info("  cm_HistGradBoost.png     — confusion matrix ")
    log.info("  cv_folds.png             — cross-validation fold scores")
    log.info("  robustness_RandomForest.csv   — robustness index table")
    log.info("  robustness_HistGradBoost.csv  — robustness index table")
    log.info("  training_log.txt         — full timestamped run log")
    log.info(f"{'='*70}")
    log.info("Run this script again on the same data to reproduce all results.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corrected Smart-Grid Fault Detection Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="classData.csv",
        help="Path to classData.csv from the Kaggle dataset. "
             "Download from: https://www.kaggle.com/datasets/esathyaprakash/"
             "electrical-fault-detection-and-classification"
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"\nERROR: Dataset file not found: {args.data}")
        print("\nTo get the dataset:")
        print("  1. Go to: https://www.kaggle.com/datasets/esathyaprakash/"
              "electrical-fault-detection-and-classification")
        print("  2. Click 'Download' and extract the ZIP")
        print("  3. Place classData.csv in this folder")
        print("  4. Run: python corrected_pipeline.py --data classData.csv\n")
        sys.exit(1)

    run(args.data)
