import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, PredefinedSplit
from sklearn.metrics import brier_score_loss, balanced_accuracy_score, precision_recall_curve, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
TUNING_DIR = ROOT / "artifacts" / "tuning"
MODELS_DIR = ROOT / "artifacts" / "models"
DOCS_DIR = ROOT / "docs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "scored_after"
RANDOM_SEED = 17

METADATA_COLS = {
    "player_appearance_id", "player_id", "fixture_id", "date",
    "scored_after", "fold", "partition_role",
    "minute_out", "subbed", "jersey_number",
}
CATEGORICAL_COLS = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
BOOLEAN_COLS     = ["is_home"]

def get_numeric_cols(df):
    exclude = METADATA_COLS | set(CATEGORICAL_COLS) | set(BOOLEAN_COLS)
    return [c for c in df.columns if c not in exclude]

def _cat_tr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def _prep_xgb(numeric_cols):
    bool_tr = FunctionTransformer(to_float)
    cat_tr_xgb = FunctionTransformer(to_category)
    ct = ColumnTransformer([
        ("cat",  cat_tr_xgb,                       CATEGORICAL_COLS),
        ("bool", bool_tr,                          BOOLEAN_COLS),
        ("num",  SimpleImputer(strategy="median"), numeric_cols),
    ])
    ct.set_output(transform='pandas')
    return ct

def run():
    dev_df = pd.read_csv(FEATURES_DIR / "features_v3_dev.csv")
    numeric_cols = get_numeric_cols(dev_df)
    feature_cols = numeric_cols + CATEGORICAL_COLS + BOOLEAN_COLS

    with open(TUNING_DIR / "best_params.json") as f:
        best_params = json.load(f)["XGBoost"]["params"]

    y_all = dev_df[TARGET].values
    spw = ((len(y_all) - y_all.sum()) / y_all.sum()) if y_all.sum() > 0 else 1.0

    base_clf = Pipeline([
        ("prep", _prep_xgb(numeric_cols)),
        ("clf",  xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=RANDOM_SEED,
        )),
    ])

    folds = sorted(dev_df["fold"].unique())
    oof_uncalib = np.zeros(len(dev_df))

    test_fold = np.zeros(len(dev_df))
    for i, fold_id in enumerate(folds):
        test_fold[dev_df["fold"] == fold_id] = i
    ps = PredefinedSplit(test_fold)

    print("Generating uncalibrated OOF probabilities...")
    for i, fold_id in enumerate(folds):
        tr_idx = dev_df[dev_df["fold"] != fold_id].index
        va_idx = dev_df[dev_df["fold"] == fold_id].index
        
        X_tr, y_tr = dev_df.loc[tr_idx, feature_cols], y_all[tr_idx]
        X_va = dev_df.loc[va_idx, feature_cols]
        
        clf = Pipeline([
            ("prep", _prep_xgb(numeric_cols)),
            ("clf",  xgb.XGBClassifier(
                **best_params,
                scale_pos_weight=spw,
                eval_metric="logloss",
                enable_categorical=True,
                random_state=RANDOM_SEED,
            )),
        ])
        clf.fit(X_tr, y_tr)
        oof_uncalib[va_idx] = clf.predict_proba(X_va)[:, 1]

    print("Calibrating OOF probabilities via internal 5-fold CV...")
    oof_sigmoid = np.zeros(len(y_all))
    oof_isotonic = np.zeros(len(y_all))
    kf = GroupKFold(n_splits=5)

    for tr_idx, va_idx in kf.split(oof_uncalib, groups=dev_df["fixture_id"]):
        X_tr, y_tr_cal = oof_uncalib[tr_idx], y_all[tr_idx]
        X_va = oof_uncalib[va_idx]

        lr = LogisticRegression()
        lr.fit(X_tr.reshape(-1, 1), y_tr_cal)
        oof_sigmoid[va_idx] = lr.predict_proba(X_va.reshape(-1, 1))[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(X_tr, y_tr_cal)
        oof_isotonic[va_idx] = iso.predict(X_va)

    brier_uncalib = brier_score_loss(y_all, oof_uncalib)
    brier_sigmoid = brier_score_loss(y_all, oof_sigmoid)
    brier_isotonic = brier_score_loss(y_all, oof_isotonic)

    print(f"Brier Uncalibrated: {brier_uncalib:.5f}")
    print(f"Brier Sigmoid:      {brier_sigmoid:.5f}")
    print(f"Brier Isotonic:     {brier_isotonic:.5f}")

    best_method_name = "isotonic" if brier_isotonic < brier_sigmoid else "sigmoid"
    best_oof = oof_isotonic if brier_isotonic < brier_sigmoid else oof_sigmoid

    print(f"Fitting final CalibratedClassifierCV using method='{best_method_name}'...")
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method=best_method_name, cv=ps)
    calibrated_clf.fit(dev_df[feature_cols], y_all)

    model_path = MODELS_DIR / "calibrated_xgboost_v3.pkl"
    joblib.dump(calibrated_clf, model_path)
    print(f"Saved calibrated model to {model_path}")

    print("Selecting optimal threshold...")
    thresholds = np.linspace(0.01, 0.99, 99)
    best_bal_acc = 0.0
    best_thresh = 0.5
    metrics_records = []

    for t in thresholds:
        preds = (best_oof >= t).astype(int)
        if len(np.unique(preds)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_all, preds).ravel()
        else:
            if preds[0] == 1:
                tn, fp, fn, tp = 0, len(y_all)-sum(y_all), 0, sum(y_all)
            else:
                tn, fp, fn, tp = len(y_all)-sum(y_all), 0, sum(y_all), 0

        bal_acc = balanced_accuracy_score(y_all, preds)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(y_all, preds)

        metrics_records.append({
            "threshold": t,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        })

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = t

    thresh_df = pd.DataFrame(metrics_records)
    thresh_df.to_csv(MODELS_DIR / "threshold_metrics.csv", index=False)

    best_row = thresh_df[thresh_df["threshold"] == best_thresh].iloc[0]

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, probs in [("Uncalibrated", oof_uncalib), ("Sigmoid", oof_sigmoid), ("Isotonic", oof_isotonic)]:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_all, probs, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(thresh_df["threshold"], thresh_df["balanced_accuracy"], label="Balanced Accuracy")
    plt.plot(thresh_df["threshold"], thresh_df["f1"], label="F1 Score")
    plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best Thresh ({best_thresh:.2f})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics by Threshold")
    plt.legend()

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "calibration_and_thresholds.png", dpi=300)
    print(f"Saved plots to {MODELS_DIR / 'calibration_and_thresholds.png'}")

    _write_policy_note(brier_uncalib, brier_sigmoid, brier_isotonic, best_method_name, best_thresh, best_row)
    print(f"Saved policy note to {DOCS_DIR / 'threshold_policy.md'}")

def _write_policy_note(brier_uncalib, brier_sigmoid, brier_isotonic, best_method_name, best_thresh, best_row):
    lines = [
        "# Probability Calibration and Decision Policy",
        "",
        "Generated by `src/python/probability_calibration.py` (Step 14 of `docs/plan.md`).",
        "",
        "## 1. Probability Calibration",
        "",
        "We compared the uncalibrated out-of-fold (OOF) probabilities of the best XGBoost model against two calibration methods: Platt Scaling (Sigmoid) and Isotonic Regression.",
        "",
        "| Method | Brier Score |",
        "|---|---:|",
        f"| Uncalibrated | {brier_uncalib:.5f} |",
        f"| Platt (Sigmoid) | {brier_sigmoid:.5f} |",
        f"| Isotonic | {brier_isotonic:.5f} |",
        "",
        f"**Decision:** Selected **{best_method_name.title()}** calibration because it achieved the lowest Brier score.",
        "The final model artifact (`artifacts/models/calibrated_xgboost_v3.pkl`) uses `CalibratedClassifierCV` with this method, respecting the 5-fold grouped cross-validation scheme to prevent leakage.",
        "",
        "## 2. Operating Threshold Selection",
        "",
        "To convert continuous probabilities into binary predictions, we must define an operating threshold. Given the heavy class imbalance (~5.8% positives), the default 0.5 threshold predicts almost exclusively negative.",
        "",
        "We scanned thresholds from 0.01 to 0.99 to maximize **Balanced Accuracy**, our primary predefined evaluation metric.",
        "",
        "### Optimal Threshold",
        f"- **Threshold:** {best_thresh:.2f}",
        f"- **Balanced Accuracy:** {best_row['balanced_accuracy']:.4f}",
        f"- **Precision:** {best_row['precision']:.4f}",
        f"- **Recall:** {best_row['recall']:.4f}",
        f"- **F1 Score:** {best_row['f1']:.4f}",
        "",
        "### Confusion Matrix at Optimal Threshold",
        f"- True Positives (TP): {int(best_row['tp'])}",
        f"- False Positives (FP): {int(best_row['fp'])}",
        f"- True Negatives (TN): {int(best_row['tn'])}",
        f"- False Negatives (FN): {int(best_row['fn'])}",
        "",
        "## 3. Tactical Interpretation",
        "",
        f"By setting the decision threshold to **{best_thresh:.2f}**, the model optimizes the trade-off between successfully identifying goalscorers (Recall = {best_row['recall']:.1%}) and avoiding excessive false alarms (Specificity = {best_row['tn']/(best_row['tn']+best_row['fp']):.1%}).",
        "",
        "**Coach's Perspective:**",
        f"A probability output above {best_thresh*100:.0f}% serves as a high-signal indicator that a player's physical and technical metrics align with impending goalscoring. Because goalscoring is inherently rare, precision is naturally low ({best_row['precision']:.1%}), meaning many flagged players will not ultimately score. However, this threshold ensures coaches are alerted to genuine offensive surges without being overwhelmed by alerts for every player on the pitch. This matches the tactical goal of identifying *potential* rather than guaranteeing an outcome."
    ]
    (DOCS_DIR / "threshold_policy.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    run()
