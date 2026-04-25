"""
Step 13b: XGBoost Hyperparameter Tuning

Feature set: v2 (artifacts/features/features_v2_dev.csv).
Outputs:
  artifacts/tuning/xgb_trials.csv
  artifacts/tuning/best_params_xgb.json
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT         = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
TUNING_DIR   = ROOT / "artifacts" / "tuning"
TUNING_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "scored_after"
RANDOM_SEED = 17

METADATA_COLS = {
    "player_appearance_id", "player_id", "fixture_id", "date",
    "scored_after", "fold", "partition_role",
    "minute_out", "subbed", "jersey_number",
}
CATEGORICAL_COLS = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
BOOLEAN_COLS     = ["is_home"]

N_TRIALS_XGB = 60

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    exclude = METADATA_COLS | set(CATEGORICAL_COLS) | set(BOOLEAN_COLS)
    return [c for c in df.columns if c not in exclude]

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

def make_objective_xgb(dev_df, numeric_cols, feature_cols, fold_ids):
    def objective(trial: optuna.Trial) -> float:
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
        return result["mean_pr_auc"]

    return objective

def _extract_trials(study: optuna.Study, model_name: str) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "model": model_name, "feature_set": "v2"}
        row.update(t.params)
        row.update(t.user_attrs)
        row["mean_pr_auc"] = t.value
        rows.append(row)
    return pd.DataFrame(rows)

def run() -> None:
    dev_df    = pd.read_csv(FEATURES_DIR / "features_v2_dev.csv")
    numeric_cols = get_numeric_cols(dev_df)
    feature_cols = numeric_cols + CATEGORICAL_COLS + BOOLEAN_COLS
    fold_ids     = sorted(dev_df["fold"].unique())
    
    print(f"\nTuning XGBoost ({N_TRIALS_XGB} trials)...")
    sampler_xgb = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study_xgb = optuna.create_study(direction="maximize", sampler=sampler_xgb)
    study_xgb.optimize(
        make_objective_xgb(dev_df, numeric_cols, feature_cols, fold_ids),
        n_trials=N_TRIALS_XGB,
        show_progress_bar=True,
    )

    xgb_trials = _extract_trials(study_xgb, "XGBoost")
    xgb_trials.to_csv(TUNING_DIR / "xgb_trials.csv", index=False)

    best_xgb = study_xgb.best_trial
    best_params_xgb = {
        "params": best_xgb.params,
        "mean_pr_auc": best_xgb.value,
        **{k: v for k, v in best_xgb.user_attrs.items()},
    }
    
    (TUNING_DIR / "best_params_xgb.json").write_text(
        json.dumps(best_params_xgb, indent=2), encoding="utf-8"
    )
    print("Done. Outputs in artifacts/tuning/")

if __name__ == "__main__":
    run()
