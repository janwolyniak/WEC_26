import json
import logging
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
TUNING_DIR = ROOT / "artifacts" / "tuning"
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

def get_numeric_cols(df):
    exclude = METADATA_COLS | set(CATEGORICAL_COLS) | set(BOOLEAN_COLS)
    return [c for c in df.columns if c not in exclude]

def _cat_tr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

def to_float(x):
    return x.astype(float)

def _prep_if(numeric_cols):
    num_tr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    bool_tr = FunctionTransformer(to_float)
    return ColumnTransformer([
        ("num",  num_tr,    numeric_cols),
        ("cat",  _cat_tr(), CATEGORICAL_COLS),
        ("bool", bool_tr,   BOOLEAN_COLS),
    ])

def _cv_eval(clf_factory, dev_df, feature_cols, fold_ids):
    fold_ba, fold_auc, fold_pr = [], [], []
    for fold_id in fold_ids:
        tr = dev_df[dev_df["fold"] != fold_id]
        va = dev_df[dev_df["fold"] == fold_id]
        X_tr, y_tr = tr[feature_cols], tr[TARGET]
        X_va, y_va = va[feature_cols], va[TARGET]
        
        clf = clf_factory()
        clf.fit(X_tr) # Unsupervised learning
        
        # decision_function gives anomaly score. Lower is more abnormal.
        # We assume goalscoring is an anomaly. Thus, we invert the scores
        # so that higher score -> more likely to be an anomaly (goal).
        y_scores = -clf.decision_function(X_va)
        
        # predict returns -1 for outlier, 1 for inlier
        preds = clf.predict(X_va)
        y_pred = np.where(preds == -1, 1, 0)
        
        fold_ba.append(balanced_accuracy_score(y_va, y_pred))
        fold_auc.append(roc_auc_score(y_va, y_scores) if y_va.nunique() > 1 else np.nan)
        fold_pr.append(average_precision_score(y_va, y_scores) if y_va.nunique() > 1 else np.nan)
        
    return {
        "mean_balanced_accuracy": float(np.nanmean(fold_ba)),
        "mean_roc_auc":           float(np.nanmean(fold_auc)),
        "mean_pr_auc":            float(np.nanmean(fold_pr)),
    }

def make_objective_if(dev_df, numeric_cols, feature_cols, fold_ids):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
            "contamination": trial.suggest_float("contamination", 0.01, 0.15),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }

        def factory():
            return Pipeline([
                ("prep", _prep_if(numeric_cols)),
                ("clf",  IsolationForest(**params, random_state=RANDOM_SEED)),
            ])

        result = _cv_eval(factory, dev_df, feature_cols, fold_ids)
        for key, val in result.items():
            trial.set_user_attr(key, val)
        return result["mean_pr_auc"]

    return objective

def run():
    dev_df = pd.read_csv(FEATURES_DIR / "features_v3_dev.csv")
    numeric_cols = get_numeric_cols(dev_df)
    feature_cols = numeric_cols + CATEGORICAL_COLS + BOOLEAN_COLS
    fold_ids = sorted(dev_df["fold"].unique())

    print("Tuning Isolation Forest (50 trials)...")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective_if(dev_df, numeric_cols, feature_cols, fold_ids),
        n_trials=50,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_params = {
        "params": best_trial.params,
        "mean_pr_auc": best_trial.value,
        **{k: v for k, v in best_trial.user_attrs.items()},
    }

    print(f"\nBest PR AUC: {best_trial.value:.4f}")
    print("Best params:", best_trial.params)
    print("Best mean_balanced_accuracy:", best_trial.user_attrs["mean_balanced_accuracy"])
    print("Best mean_roc_auc:", best_trial.user_attrs["mean_roc_auc"])

    out_path = TUNING_DIR / "best_params_isolation_forest.json"
    out_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(TUNING_DIR / "if_trials.csv", index=False)

if __name__ == "__main__":
    run()
