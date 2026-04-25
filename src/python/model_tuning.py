"""
Step 13: Model Expansion and Hyperparameter Tuning

Implements plan.md §13:
  1. Compact candidate set: LogisticRegression and XGBoost (avoid model zoo sprawl).
  2. Structured hyperparameter search under grouped CV using Optuna (TPE sampler).
  3. Track every trial: params, seed, per-fold and mean metrics, feature set label.
  4. Select best model per primary metric (mean CV PR AUC, per user request).

Feature set: v2 (artifacts/features/features_v2_dev.csv).
Leakage exclusions enforced: minute_out, subbed (docs/leakage_checklist.md critical failures).

Trial budgets:
  LogisticRegression: 50 trials
  XGBoost:            60 trials

Outputs:
  artifacts/tuning/lr_trials.csv          - all LR trial records
  artifacts/tuning/xgb_trials.csv         - all XGBoost trial records
  artifacts/tuning/best_params.json       - best params per model
  artifacts/tuning/finalist_summary.csv   - finalist comparison table
  docs/model_tuning.md                    - tuning summary report
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
TUNING_DIR   = ROOT / "artifacts" / "tuning"
DOCS_DIR     = ROOT / "docs"
TUNING_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "scored_after"
RANDOM_SEED = 17  # matches split_config.json

# Columns excluded from feature space
# minute_out and subbed: critical leakage per docs/leakage_checklist.md
# jersey_number: player identifier, not a feature
METADATA_COLS = {
    "player_appearance_id", "player_id", "fixture_id", "date",
    "scored_after", "fold", "partition_role",
    "minute_out", "subbed", "jersey_number",   # leakage / identifiers
}
CATEGORICAL_COLS = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
BOOLEAN_COLS     = ["is_home"]

N_TRIALS_LR  = 50
N_TRIALS_XGB = 60


# ---------------------------------------------------------------------------
# Feature list derivation
# ---------------------------------------------------------------------------
def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """All columns not in METADATA_COLS, CATEGORICAL_COLS, or BOOLEAN_COLS."""
    exclude = METADATA_COLS | set(CATEGORICAL_COLS) | set(BOOLEAN_COLS)
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Preprocessor builders
# ---------------------------------------------------------------------------
def _cat_tr() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def _prep_lr(numeric_cols: list[str]) -> ColumnTransformer:
    num_tr = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc",  StandardScaler())])
    bool_tr = FunctionTransformer(lambda x: x.astype(float))
    return ColumnTransformer([
        ("num",  num_tr,    numeric_cols),
        ("cat",  _cat_tr(), CATEGORICAL_COLS),
        ("bool", bool_tr,   BOOLEAN_COLS),
    ])

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def _prep_xgb(numeric_cols: list[str]) -> ColumnTransformer:
    bool_tr = FunctionTransformer(to_float)
    cat_tr_xgb = FunctionTransformer(to_category)
    ct = ColumnTransformer([
        ("cat",  cat_tr_xgb,                       CATEGORICAL_COLS),
        ("bool", bool_tr,                          BOOLEAN_COLS),
        ("num",  SimpleImputer(strategy="median"), numeric_cols),
    ])
    ct.set_output(transform='pandas')
    return ct


# ---------------------------------------------------------------------------
# CV evaluation helper (returns per-fold and mean metrics)
# ---------------------------------------------------------------------------
def _cv_eval(clf_factory, dev_df: pd.DataFrame,
             feature_cols: list[str], fold_ids: list[str]) -> dict:
    fold_ba, fold_auc, fold_pr, fold_br = [], [], [], []
    for fold_id in fold_ids:
        tr = dev_df[dev_df["fold"] != fold_id]
        va = dev_df[dev_df["fold"] == fold_id]
        X_tr, y_tr = tr[feature_cols], tr[TARGET]
        X_va, y_va = va[feature_cols], va[TARGET]
        clf = clf_factory(y_tr)
        clf.fit(X_tr, y_tr)
        y_pred  = clf.predict(X_va)
        y_proba = clf.predict_proba(X_va)[:, 1]
        fold_ba.append(balanced_accuracy_score(y_va, y_pred))
        fold_auc.append(roc_auc_score(y_va, y_proba) if y_va.nunique() > 1 else np.nan)
        fold_pr.append(average_precision_score(y_va, y_proba) if y_va.nunique() > 1 else np.nan)
        fold_br.append(brier_score_loss(y_va, y_proba))
    return {
        "mean_balanced_accuracy": float(np.nanmean(fold_ba)),
        "std_balanced_accuracy":  float(np.nanstd(fold_ba)),
        "mean_roc_auc":           float(np.nanmean(fold_auc)),
        "mean_pr_auc":            float(np.nanmean(fold_pr)),
        "mean_brier_score":       float(np.nanmean(fold_br)),
        "fold_balanced_accuracy": fold_ba,
    }


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------
def make_objective_lr(dev_df, numeric_cols, feature_cols, fold_ids, optimize_for="pr_auc"):
    def objective(trial: optuna.Trial):
        penalty_solver = trial.suggest_categorical(
            "penalty_solver", ["l2_lbfgs", "l2_saga", "l1_saga"]
        )
        penalty, solver = penalty_solver.split("_", 1)
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)

        def factory(y_tr):
            return Pipeline([
                ("prep", _prep_lr(numeric_cols)),
                ("clf",  LogisticRegression(
                    C=C, penalty=penalty, solver=solver,
                    class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
                )),
            ])

        result = _cv_eval(factory, dev_df, feature_cols, fold_ids)
        for key, val in result.items():
            if key != "fold_balanced_accuracy":
                trial.set_user_attr(key, val)
                
        if optimize_for == "both":
            return result["mean_pr_auc"], result["mean_balanced_accuracy"]
        elif optimize_for == "balanced_accuracy":
            return result["mean_balanced_accuracy"]
        return result["mean_pr_auc"]

    return objective


def make_objective_xgb(dev_df, numeric_cols, feature_cols, fold_ids, optimize_for="pr_auc"):
    def objective(trial: optuna.Trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 2, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "subsample":         trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        def factory(y_tr):
            spw = ((len(y_tr) - y_tr.sum()) / y_tr.sum()) if y_tr.sum() > 0 else 1.0
            return Pipeline([
                ("prep", _prep_xgb(numeric_cols)),
                ("clf",  xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=spw,
                    eval_metric="logloss",
                    enable_categorical=True,
                    random_state=RANDOM_SEED,
                )),
            ])

        result = _cv_eval(factory, dev_df, feature_cols, fold_ids)
        for key, val in result.items():
            if key != "fold_balanced_accuracy":
                trial.set_user_attr(key, val)
                
        if optimize_for == "both":
            return result["mean_pr_auc"], result["mean_balanced_accuracy"]
        elif optimize_for == "balanced_accuracy":
            return result["mean_balanced_accuracy"]
        return result["mean_pr_auc"]

    return objective


# ---------------------------------------------------------------------------
# Trial records extractor
# ---------------------------------------------------------------------------
def _extract_trials(study: optuna.Study, model_name: str) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "model": model_name, "feature_set": "v2"}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(optimize_for: str = "pr_auc", train_lr: bool = True, train_xgb: bool = True) -> None:
    dev_df    = pd.read_csv(FEATURES_DIR / "features_v3_dev.csv")
    numeric_cols = get_numeric_cols(dev_df)
    feature_cols = numeric_cols + CATEGORICAL_COLS + BOOLEAN_COLS
    fold_ids     = sorted(dev_df["fold"].unique())

    print(f"Feature space: {len(numeric_cols)} numeric + {len(CATEGORICAL_COLS)} cat "
          f"+ {len(BOOLEAN_COLS)} bool = {len(feature_cols)} total")
    print(f"Development rows: {len(dev_df)}, Folds: {fold_ids}")
    
    directions = ["maximize", "maximize"] if optimize_for == "both" else None
    direction = "maximize" if optimize_for != "both" else None

    # ---- Logistic Regression ----
    if train_lr:
        print(f"\nTuning LogisticRegression ({N_TRIALS_LR} trials)...")
        sampler_lr = optuna.samplers.TPESampler(seed=RANDOM_SEED, multivariate=True)
        study_lr = optuna.create_study(directions=directions, direction=direction, sampler=sampler_lr)
        study_lr.optimize(
            make_objective_lr(dev_df, numeric_cols, feature_cols, fold_ids, optimize_for),
            n_trials=N_TRIALS_LR,
            show_progress_bar=True,
        )
        lr_trials  = _extract_trials(study_lr,  "LogisticRegression")
        lr_trials.to_csv(TUNING_DIR  / "lr_trials.csv",  index=False)
        best_lr  = study_lr.best_trials[0] if optimize_for == "both" else study_lr.best_trial
    else:
        study_lr = None
        lr_trials = None
        best_lr = None

    # ---- XGBoost ----
    if train_xgb:
        print(f"\nTuning XGBoost ({N_TRIALS_XGB} trials)...")
        sampler_xgb = optuna.samplers.TPESampler(seed=RANDOM_SEED, multivariate=True)
        study_xgb = optuna.create_study(directions=directions, direction=direction, sampler=sampler_xgb)
        study_xgb.optimize(
            make_objective_xgb(dev_df, numeric_cols, feature_cols, fold_ids, optimize_for),
            n_trials=N_TRIALS_XGB,
            show_progress_bar=True,
        )
        xgb_trials = _extract_trials(study_xgb, "XGBoost")
        xgb_trials.to_csv(TUNING_DIR / "xgb_trials.csv", index=False)
        best_xgb = study_xgb.best_trials[0] if optimize_for == "both" else study_xgb.best_trial
    else:
        study_xgb = None
        xgb_trials = None
        best_xgb = None

    best_params_path = TUNING_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
    else:
        best_params = {}

    # ---- Best params ----
    if train_lr and best_lr is not None:
        best_params["LogisticRegression"] = {
            "params": best_lr.params,
            **{k: v for k, v in best_lr.user_attrs.items()},
        }
    if train_xgb and best_xgb is not None:
        best_params["XGBoost"] = {
            "params": best_xgb.params,
            **{k: v for k, v in best_xgb.user_attrs.items()},
        }
        
    (TUNING_DIR / "best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )

    # ---- Finalist summary ----
    finalist_rows = []
    models_to_report = []
    if train_lr and best_lr is not None:
        row = {"model": "LogisticRegression", "best_trial": best_lr.number}
        row.update(best_lr.params)
        row.update(best_lr.user_attrs)
        finalist_rows.append(row)
        models_to_report.append(("LogisticRegression", lr_trials, study_lr))
        
    if train_xgb and best_xgb is not None:
        row = {"model": "XGBoost", "best_trial": best_xgb.number}
        row.update(best_xgb.params)
        row.update(best_xgb.user_attrs)
        finalist_rows.append(row)
        models_to_report.append(("XGBoost", xgb_trials, study_xgb))

    if finalist_rows:
        # Load existing finalist summary if it exists to preserve untouched models
        finalist_summary_path = TUNING_DIR / "finalist_summary.csv"
        if finalist_summary_path.exists():
            existing_df = pd.read_csv(finalist_summary_path)
            new_df = pd.DataFrame(finalist_rows)
            # Remove updated models from existing
            existing_df = existing_df[~existing_df["model"].isin(new_df["model"])]
            finalist_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            finalist_df = pd.DataFrame(finalist_rows)
            
        finalist_df.to_csv(TUNING_DIR / "finalist_summary.csv", index=False)

    _write_report(models_to_report, best_params, optimize_for)
    print("\nDone. Outputs in artifacts/tuning/ and docs/model_tuning.md")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _write_report(models_to_report, best_params, optimize_for) -> None:
    lines: list[str] = []
    a = lines.append

    a("# Model Tuning Report — Step 13")
    a("")
    a("Generated by `src/python/model_tuning.py` (Step 13 of `docs/plan.md`).")
    a("")
    a(f"Primary selection metric: **{optimize_for}**.")
    a("Feature set: **v2** (85 v1 features + 38 event-table features).")
    a("")

    a("## 1. Candidate Models and Search Spaces")
    a("")
    a("| Model | Tuned Parameters | Fixed Parameters |")
    a("|---|---|---|")
    a("| LogisticRegression | C ∈ [1e-3, 100] (log), penalty/solver ∈ {l2_lbfgs, l2_saga, l1_saga} | class_weight=balanced, max_iter=2000 |")
    a("| XGBoost | n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda | scale_pos_weight (fold-computed), eval_metric=logloss |")
    a("")

    a("## 2. Best Hyperparameters")
    a("")
    for model_name in ["LogisticRegression", "XGBoost"]:
        if model_name not in best_params:
            continue
        bp = best_params[model_name]
        a(f"### {model_name}")
        a("")
        a(f"Best trial PR AUC: **{bp.get('mean_pr_auc', float('nan')):.4f}**")
        a(f"Best trial Balanced Accuracy: **{bp.get('mean_balanced_accuracy', float('nan')):.4f}**")
        a("")
        a("| Parameter | Value |")
        a("|---|---|")
        for k, v in bp["params"].items():
            val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
            a(f"| {k} | {val_str} |")
        a("")

    a("## 3. Performance Summary")
    a("")
    a("| Model | Trials | Best PR AUC | Best Bal. Acc. | Best ROC AUC | Best Brier |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for model_name, trials_df, study in models_to_report:
        bt = study.best_trials[0] if optimize_for == "both" else study.best_trial
        a(f"| {model_name} | {len(trials_df)} | {bt.user_attrs.get('mean_pr_auc', float('nan')):.4f}"
          f" | {bt.user_attrs.get('mean_balanced_accuracy', float('nan')):.4f}"
          f" | {bt.user_attrs.get('mean_roc_auc', float('nan')):.4f}"
          f" | {bt.user_attrs.get('mean_brier_score', float('nan')):.4f} |")
    a("")

    a("## 4. Model Selection Decision")
    a("")
    lr_pr  = best_params.get("LogisticRegression", {}).get("mean_pr_auc", 0)
    xgb_pr = best_params.get("XGBoost", {}).get("mean_pr_auc", 0)
    winner = "LogisticRegression" if lr_pr >= xgb_pr else "XGBoost"
    a(f"Primary metric (PR AUC): LR = {lr_pr:.4f}, XGBoost = {xgb_pr:.4f}.")
    a(f"**Selected finalist (by PR AUC fallback): {winner}**.")
    a("")
    a("Both models are retained as finalists for the RQ ablation matrix (Step 15). "
      "Final selection after holdout evaluation (Step 17).")
    a("")
    a("## 5. Next Steps (Step 14)")
    a("Calibrate probabilities on OOF predictions using Platt scaling and isotonic regression. "
      "Select operating threshold using development data only.")

    (DOCS_DIR / "model_tuning.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", type=str, default="pr_auc", choices=["pr_auc", "balanced_accuracy", "both"], help="Metric to optimize")
    args = parser.parse_args()
    run(optimize_for=args.optimize)
