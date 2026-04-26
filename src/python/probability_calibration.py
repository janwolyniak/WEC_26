import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, PredefinedSplit
from sklearn.metrics import brier_score_loss, balanced_accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, clone

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
TUNING_DIR = ROOT / "artifacts" / "tuning"
MODELS_DIR = ROOT / "artifacts" / "models"
DOCS_DIR = ROOT / "docs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "scored_after"
RANDOM_SEED = 17

# MUST MATCH model_tuning.py exactly
METADATA_COLS = {
    "player_appearance_id", "player_id", "fixture_id", "date",
    "scored_after", "fold", "partition_role",
    "minute_out", "subbed", "jersey_number",
}
CATEGORICAL_COLS = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
BOOLEAN_COLS     = ["is_home"]

def get_numeric_cols(df):
    exclude = METADATA_COLS | set(CATEGORICAL_COLS) | set(BOOLEAN_COLS) | {"exposure"}
    return [c for c in df.columns if c not in exclude]

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def _prep_xgb(numeric_cols, cat_cols=CATEGORICAL_COLS, bool_cols=BOOLEAN_COLS):
    bool_tr = FunctionTransformer(to_float)
    from sklearn.preprocessing import OrdinalEncoder
    cat_tr_xgb = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ct = ColumnTransformer([
        ("cat",  cat_tr_xgb,                       cat_cols),
        ("bool", bool_tr,                          bool_cols),
        ("num",  SimpleImputer(strategy="median"), numeric_cols),
    ], remainder="passthrough")
    ct.set_output(transform='pandas')
    return ct

# --- Custom Calibrators ---
class BetaCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1e10):
        self.C = C
        self.clf = LogisticRegression(C=self.C)
    
    def _transform(self, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.column_stack([np.log(p), -np.log(1 - p)])
    
    def fit(self, p, y):
        X = self._transform(p)
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self
    
    def predict_proba(self, p):
        X = self._transform(p)
        return self.clf.predict_proba(X)
        
    def predict(self, p):
        return (self.predict_proba(p)[:, 1] >= 0.5).astype(int)

class SplineCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_knots=5, degree=3, C=1e10):
        self.n_knots = n_knots
        self.degree = degree
        self.C = C
        self.pipeline = Pipeline([
            ('spline', SplineTransformer(n_knots=self.n_knots, degree=self.degree)),
            ('lr', LogisticRegression(C=self.C))
        ])

    def fit(self, p, y):
        self.pipeline.fit(p.reshape(-1, 1), y)
        self.classes_ = self.pipeline.named_steps['lr'].classes_
        return self

    def predict_proba(self, p):
        return self.pipeline.predict_proba(p.reshape(-1, 1))
        
    def predict(self, p):
        return self.pipeline.predict(p.reshape(-1, 1))

# --- Poisson Wrapper ---
class PoissonXGBWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    
    def __init__(self, **params):
        self.params = params
        self.classes_ = np.array([0, 1])
        self.model_ = None

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y):
        X_fit = X.copy()
        exposure_col = "exposure" if "exposure" in X_fit.columns else "remainder__exposure"
        if isinstance(X_fit, pd.DataFrame) and exposure_col in X_fit.columns:
            exposure = X_fit[exposure_col].clip(lower=1)
            X_fit = X_fit.drop(columns=[exposure_col])
        else:
            exposure = np.ones(len(X_fit))

        self.model_ = xgb.XGBRegressor(
            **self.params, 
            objective='count:poisson'
        )
        self.model_.fit(X_fit, y, base_margin=np.log(exposure))
        return self

    def predict_proba(self, X):
        X_pred = X.copy()
        exposure_col = "exposure" if "exposure" in X_pred.columns else "remainder__exposure"
        if isinstance(X_pred, pd.DataFrame) and exposure_col in X_pred.columns:
            exposure = X_pred[exposure_col].clip(lower=1)
            X_pred = X_pred.drop(columns=[exposure_col])
        else:
            exposure = np.ones(len(X_pred))
            
        lambda_val = self.model_.predict(X_pred, base_margin=np.log(exposure))
        p1 = 1 - np.exp(-lambda_val)
        p1 = np.clip(p1, 0, 1)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# --- Final Model Wrapper ---
class FullyCalibratedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, calibrator):
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p_raw = self.base_estimator.predict_proba(X)[:, 1]
        return self.calibrator.predict_proba(p_raw)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class IsotonicCalibrator:
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip")
    def fit(self, p, y):
        self.iso.fit(p, y)
        return self
    def predict_proba(self, p):
        cal = self.iso.predict(p)
        cal = np.clip(cal, 0, 1)
        return np.column_stack([1 - cal, cal])
    def predict(self, p):
        return (self.predict_proba(p)[:, 1] >= 0.5).astype(int)

class SigmoidCalibrator:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression()
    def fit(self, p, y):
        self.lr.fit(p.reshape(-1, 1), y)
        return self
    def predict_proba(self, p):
        return self.lr.predict_proba(p.reshape(-1, 1))
    def predict(self, p):
        return self.lr.predict(p.reshape(-1, 1))

def run(version="v3", use_poisson=False):
    print(f"Running calibration for version: {version} (Poisson: {use_poisson})")
    dev_df = pd.read_csv(FEATURES_DIR / f"features_{version}_dev.csv")
    
    numeric_cols = get_numeric_cols(dev_df)
    actual_cat_cols = [c for c in CATEGORICAL_COLS if c in dev_df.columns]
    actual_bool_cols = [c for c in BOOLEAN_COLS if c in dev_df.columns]
    cols_to_use = numeric_cols + actual_cat_cols + actual_bool_cols
    if use_poisson and "exposure" in dev_df.columns:
        cols_to_use = cols_to_use + ["exposure"]

    with open(TUNING_DIR / f"best_params_{version}.json") as f:
        best_params = json.load(f)["XGBoost"]["params"]

    # Global imbalance for scale_pos_weight
    pos = dev_df[TARGET].sum()
    neg = len(dev_df) - pos
    spw = neg / pos if pos > 0 else 1.0

    if use_poisson:
        base_clf = Pipeline([
            ("prep", _prep_xgb(numeric_cols, cat_cols=actual_cat_cols, bool_cols=actual_bool_cols)),
            ("clf",  PoissonXGBWrapper(**best_params, random_state=RANDOM_SEED)),
        ])
    else:
        base_clf = Pipeline([
            ("prep", _prep_xgb(numeric_cols, cat_cols=actual_cat_cols, bool_cols=actual_bool_cols)),
            ("clf",  xgb.XGBClassifier(
                **best_params,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
            )),
        ])

    folds = sorted(dev_df["fold"].unique())
    oof_uncalib = np.zeros(len(dev_df))
    
    # Reset index for consistent integer positional indexing
    dev_df = dev_df.reset_index(drop=True)
    y_all = dev_df[TARGET].values

    print("Generating uncalibrated OOF probabilities...")
    for fold_id in folds:
        va_mask = dev_df["fold"] == fold_id
        tr_mask = ~va_mask
        
        X_tr, y_tr = dev_df.loc[tr_mask, cols_to_use], y_all[tr_mask]
        X_va = dev_df.loc[va_mask, cols_to_use]
        
        clf = clone(base_clf)
        clf.fit(X_tr, y_tr)
        
        # positional boolean masking works correctly on a reset-indexed numpy array
        oof_uncalib[va_mask] = clf.predict_proba(X_va)[:, 1]

    print("Calibrating OOF probabilities via internal 5-fold CV...")
    oof_sig, oof_iso, oof_beta, oof_spline = [np.zeros(len(dev_df)) for _ in range(4)]
    kf = GroupKFold(n_splits=5)

    for tr_idx, va_idx in kf.split(oof_uncalib, groups=dev_df["fixture_id"]):
        p_tr, y_tr_cal = oof_uncalib[tr_idx], y_all[tr_idx]
        p_va = oof_uncalib[va_idx]

        oof_sig[va_idx]    = LogisticRegression(C=1e10).fit(p_tr.reshape(-1, 1), y_tr_cal).predict_proba(p_va.reshape(-1, 1))[:, 1]
        oof_iso[va_idx]    = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_tr, y_tr_cal).predict(p_va)
        oof_beta[va_idx]   = BetaCalibrator().fit(p_tr, y_tr_cal).predict_proba(p_va)[:, 1]
        oof_spline[va_idx] = SplineCalibrator().fit(p_tr, y_tr_cal).predict_proba(p_va)[:, 1]

    briers = {
        "uncalibrated": brier_score_loss(y_all, oof_uncalib),
        "sigmoid":      brier_score_loss(y_all, oof_sig),
        "isotonic":     brier_score_loss(y_all, oof_iso),
        "beta":         brier_score_loss(y_all, oof_beta),
        "spline":       brier_score_loss(y_all, oof_spline)
    }
    for k, v in briers.items(): print(f"Brier {k:12}: {v:.5f}")

    best_method = min(briers, key=briers.get)
    print(f"Best method found: {best_method}")
    
    # Final model retrained on full dev set
    final_base = clone(base_clf).fit(dev_df[cols_to_use], y_all)
    
    if best_method == "isotonic":
        cal = IsotonicCalibrator().fit(oof_uncalib, y_all)
        final_model = FullyCalibratedModel(final_base, cal)
    elif best_method == "sigmoid":
        cal = SigmoidCalibrator().fit(oof_uncalib, y_all)
        final_model = FullyCalibratedModel(final_base, cal)
    elif best_method == "beta":
        cal = BetaCalibrator().fit(oof_uncalib, y_all)
        final_model = FullyCalibratedModel(final_base, cal)
    else: # spline
        cal = SplineCalibrator().fit(oof_uncalib, y_all)
        final_model = FullyCalibratedModel(final_base, cal)

    model_path = MODELS_DIR / f"calibrated_xgboost_{version}.pkl"
    joblib.dump(final_model, model_path)
    print(f"Saved calibrated model to {model_path}")

    # Optimal threshold scanning
    best_oof = {"uncalibrated": oof_uncalib, "sigmoid": oof_sig, "isotonic": oof_iso, "beta": oof_beta, "spline": oof_spline}[best_method]
    thresholds = np.linspace(0.01, 0.99, 99)
    best_bal_acc, best_thresh = 0.0, 0.5
    metrics_records = []

    for t in thresholds:
        preds = (best_oof >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_all, preds).ravel() if len(np.unique(preds)) > 1 else (0,0,0,0)
        ba = balanced_accuracy_score(y_all, preds)
        metrics_records.append({"threshold": t, "balanced_accuracy": ba, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                                "precision": tp/(tp+fp) if (tp+fp)>0 else 0, "recall": tp/(tp+fn) if (tp+fn)>0 else 0, "f1": f1_score(y_all, preds)})
        if ba > best_bal_acc:
            best_bal_acc, best_thresh = ba, t

    thresh_df = pd.DataFrame(metrics_records)
    best_row = thresh_df.iloc[np.argmin(np.abs(thresholds - best_thresh))]

    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, probs in [("Uncalib", oof_uncalib), ("Sigmoid", oof_sig), ("Isotonic", oof_iso), ("Beta", oof_beta), ("Spline", oof_spline)]:
        fop, mpv = calibration_curve(y_all, probs, n_bins=10)
        plt.plot(mpv, fop, "s-", label=name)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves"); plt.legend(); plt.xlim([0, 1]); plt.ylim([0, 1])

    plt.subplot(1, 2, 2)
    plt.plot(thresholds, thresh_df['balanced_accuracy'], label="Balanced Accuracy")
    plt.plot(thresholds, thresh_df['f1'], label="F1 Score")
    plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best Thresh ({best_thresh:.2f})")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title("Threshold Selection"); plt.legend(); plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.tight_layout(); plt.savefig(MODELS_DIR / "calibration_and_thresholds.png", dpi=300)

    _write_policy_note(briers, best_method, best_thresh, best_row, version)

def _write_policy_note(briers, best_method, best_thresh, best_row, version):
    lines = [
        "# Probability Calibration and Decision Policy",
        "",
        "## 1. Calibration Comparison",
        "",
        "| Method | Brier Score |",
        "|---|---:|",
    ]
    for k, v in briers.items(): lines.append(f"| {k.title()} | {v:.5f} |")
    lines += [
        "",
        f"**Decision:** Selected **{best_method.title()}** calibration.",
        f"Final model: `artifacts/models/calibrated_xgboost_{version}.pkl`.",
        "",
        "## 2. Threshold Selection",
        "",
        f"Optimal threshold found by maximizing Balanced Accuracy on OOF predictions.",
        "",
        "### Optimal Threshold Results",
        f"- **Threshold:** {best_thresh:.2f}",
        f"- **Balanced Accuracy:** {best_row['balanced_accuracy']:.4f}",
        f"- **F1 Score:** {best_row['f1']:.4f}",
        f"- **Precision:** {best_row['precision']:.4f}",
        f"- **Recall:** {best_row['recall']:.1%}",
        "",
        "### Confusion Matrix at Optimal Threshold",
        f"- True Positives (TP): {int(best_row['tp'])}",
        f"- False Positives (FP): {int(best_row['fp'])}",
        f"- True Negatives (TN): {int(best_row['tn'])}",
        f"- False Negatives (FN): {int(best_row['fn'])}",
        "",
        "## 3. Tactical Interpretation",
        "",
        f"By setting the decision threshold to **{best_thresh:.2f}**, the model identifies potential goalscorers with a recall of {best_row['recall']:.1%}.",
    ]
    (DOCS_DIR / "threshold_policy.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved policy note to {DOCS_DIR / 'threshold_policy.md'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--poisson", action="store_true")
    args = parser.parse_args()
    run(version=args.version, use_poisson=args.poisson)
