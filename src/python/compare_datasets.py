"""
Compare performance of the best optimized models on:
1. Our v2 Feature Set (baseline)
2. Friend's Dataset (data/players_quarters_final_step1.csv)
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
BEST_PARAMS_PATH = ROOT / "artifacts" / "tuning" / "best_params.json"
FRIEND_DATA_PATH = ROOT / "data" / "players_quarters_final_step1.csv"
OUR_DATA_PATH    = ROOT / "artifacts" / "features" / "features_v3_dev.csv"
ASSIGNMENTS_PATH = ROOT / "data" / "splits" / "fixture_assignments.csv"

RANDOM_SEED = 17

def get_numeric_cols(df: pd.DataFrame, exclude_set: set) -> list[str]:
    return [c for c in df.columns if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])]

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def _prep_lr(numeric_cols, cat_cols, bool_cols):
    num_tr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_tr = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    bool_tr = Pipeline([("to_float", FunctionTransformer(to_float)), ("imp", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer([
        ("num", num_tr, numeric_cols),
        ("cat", cat_tr, cat_cols),
        ("bool", bool_tr, bool_cols),
    ])

def _prep_xgb(numeric_cols, cat_cols, bool_cols):
    cat_tr = FunctionTransformer(to_category)
    bool_tr = FunctionTransformer(to_float)
    ct = ColumnTransformer([
        ("cat", cat_tr, cat_cols),
        ("bool", bool_tr, bool_cols),
        ("num", SimpleImputer(strategy="median"), numeric_cols),
    ])
    ct.set_output(transform='pandas')
    return ct

def run_cv(df, params, model_name):
    target = "scored_after"
    meta = {"player_appearance_id", "player_id", "fixture_id", "date", "scored_after", "fold", "partition_role", "minute_out", "subbed", "jersey_number", "minute_in_window", "minute_out_window", "minutes_in_game"}
    cat_cols = [c for c in ["checkpoint", "checkpoint_period", "position", "formation", "player_id"] if c in df.columns]
    bool_cols = [c for c in ["is_home"] if c in df.columns]
    num_cols = get_numeric_cols(df, meta | set(cat_cols) | set(bool_cols))
    
    folds = sorted(df["fold"].unique())
    results = []
    
    for fold_id in folds:
        tr_df = df[df["fold"] != fold_id]
        va_df = df[df["fold"] == fold_id]
        
        X_tr, y_tr = tr_df[num_cols + cat_cols + bool_cols], tr_df[target]
        X_va, y_va = va_df[num_cols + cat_cols + bool_cols], va_df[target]
        
        if model_name == "LogisticRegression":
            ps = params.get("penalty_solver", "l2_saga")
            penalty, solver = ps.split("_", 1)
            clf = Pipeline([
                ("prep", _prep_lr(num_cols, cat_cols, bool_cols)),
                ("clf", LogisticRegression(C=params.get("C", 1.0), penalty=penalty, solver=solver, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED))
            ])
        else:
            spw = ((len(y_tr) - y_tr.sum()) / y_tr.sum()) if y_tr.sum() > 0 else 1.0
            clf = Pipeline([
                ("prep", _prep_xgb(num_cols, cat_cols, bool_cols)),
                ("clf", xgb.XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss", enable_categorical=True, random_state=RANDOM_SEED))
            ])
        
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        preds = clf.predict(X_va)
        
        results.append({
            "pr_auc": average_precision_score(y_va, probs),
            "bal_acc": balanced_accuracy_score(y_va, preds),
            "roc_auc": roc_auc_score(y_va, probs),
            "brier": brier_score_loss(y_va, probs)
        })
    
    return pd.DataFrame(results).mean()

def run():
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    
    # Load our data
    our_df = pd.read_csv(OUR_DATA_PATH)
    
    # Load friend data and merge splits
    friend_df = pd.read_csv(FRIEND_DATA_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    friend_df = friend_df.merge(assignments[["fixture_id", "fold", "partition_role"]], on="fixture_id", how="left")
    friend_df = friend_df[friend_df["partition_role"] == "development"].copy()
    
    summary = []
    
    for model in ["LogisticRegression", "XGBoost"]:
        params = best_params[model]["params"]
        print(f"Evaluating {model}...")
        
        our_res = run_cv(our_df, params, model)
        summary.append({
            "model": model,
            "dataset": "Our v2",
            "pr_auc": our_res["pr_auc"],
            "bal_acc": our_res["bal_acc"],
            "roc_auc": our_res["roc_auc"],
            "brier": our_res["brier"]
        })
        
        friend_res = run_cv(friend_df, params, model)
        summary.append({
            "model": model,
            "dataset": "Friend Step1",
            "pr_auc": friend_res["pr_auc"],
            "bal_acc": friend_res["bal_acc"],
            "roc_auc": friend_res["roc_auc"],
            "brier": friend_res["brier"]
        })

    report_df = pd.DataFrame(summary)
    print("\nComparison Results:")
    print(report_df.to_string(index=False))
    
    report_df.to_csv(ROOT / "artifacts" / "tuning" / "dataset_comparison.csv", index=False)

if __name__ == "__main__":
    run()
