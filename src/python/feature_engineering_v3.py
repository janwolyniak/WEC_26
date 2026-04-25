"""
Step 12.5: Feature Engineering Iteration 3 (Formation, Fresh Legs, Team Aggregates)

Implements:
1. Formation parsing: defenders, attackers, midfielders.
2. Formation context: lone striker system, back three, attacker ratio.
3. Fresh legs / substitute features.
4. Team-level aggregates: team momentum, shot share.
5. High-value event combinations.

Outputs:
  artifacts/features/features_v3_dev.csv
  artifacts/features/features_v3_holdout.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
BASELINE_DIR = ROOT / "artifacts" / "baseline"
DOCS_DIR     = ROOT / "docs"

TARGET = "scored_after"

def engineer_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1. Formation Parsing
    splits = out["formation"].astype(str).str.split("-")
    
    def get_defs(s):
        try: return float(s[0])
        except: return 0.0
        
    def get_atts(s):
        try: return float(s[-1])
        except: return 0.0
        
    def get_mids(s):
        try: return sum(float(x) for x in s[1:-1])
        except: return 0.0

    out["formation_defenders"] = splits.apply(get_defs)
    out["formation_attackers"] = splits.apply(get_atts)
    out["formation_midfielders"] = splits.apply(get_mids)
    
    out["is_lone_striker_system"] = (out["formation_attackers"] == 1).astype(float)
    out["is_back_three"] = (out["formation_defenders"].isin([3, 5])).astype(float)
    
    # 2. Formation Interactions
    out["attacker_in_lone_system"] = out.get("is_pos_A", 0) * out["is_lone_striker_system"]
    out["defender_in_back_three"] = out.get("is_pos_D", 0) * out["is_back_three"]
    
    total_outfield = out["formation_defenders"] + out["formation_midfielders"] + out["formation_attackers"]
    out["attacker_ratio"] = out["formation_attackers"] / total_outfield.replace(0, 10.0)

    # 3. Fresh legs / substitute features
    out["is_substitute"] = (out["minute_in"] > 1).astype(float)
    # fresh_legs_advantage = elapsed time / total checkpoint min
    out["fresh_legs_advantage"] = out.get("elapsed_min", 0) / out["checkpoint_cont_min"].replace(0, 1.0)
    out["sub_x_sprints"] = out["is_substitute"] * out.get("last15_sprints", 0).astype(float)

    # 4. High-value combinations
    out["high_threat_sprints"] = out.get("last15_sprints", 0).astype(float) * out.get("last15_shots_top_third", 0).astype(float)
    out["effective_presser"] = out.get("last15_pressure_count", 0).astype(float) * out.get("last15_pressure_turnover_rate", 0).astype(float)

    # 5. Team-level metrics
    # Team is grouped by fixture_id, checkpoint, and is_home
    team_grp = out.groupby(["fixture_id", "checkpoint", "is_home"])
    
    out["team_cumul_shots"] = team_grp["cumul_shots"].transform("sum")
    out["player_shot_share"] = out["cumul_shots"].astype(float) / out["team_cumul_shots"].replace(0, 1.0)
    
    out["team_last15_momentum_sprints"] = team_grp["last15_sprints"].transform("sum")
    out["team_last15_momentum_shots"] = team_grp["last15_shots"].transform("sum")

    return out

# ---------------------------------------------------------------------------
# Feature Column Management
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pipeline helpers (identical structure)
# ---------------------------------------------------------------------------
def _cat_tr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_pipeline_lr(numeric_cols, categorical_cols, boolean_cols) -> Pipeline:
    num_tr = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc",  StandardScaler())])
    bool_tr = FunctionTransformer(lambda x: x.astype(float))
    prep = ColumnTransformer([
        ("num",  num_tr,    numeric_cols),
        ("cat",  _cat_tr(), categorical_cols),
        ("bool", bool_tr,   boolean_cols),
    ])
    return Pipeline([("preprocessor", prep),
                     ("classifier",   LogisticRegression(class_weight="balanced",
                                                         max_iter=1000, random_state=42))])

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def build_pipeline_xgb(numeric_cols, categorical_cols, boolean_cols,
                        scale_pos_weight: float) -> Pipeline:
    bool_tr = FunctionTransformer(to_float)
    cat_tr_xgb = FunctionTransformer(to_category)
    prep = ColumnTransformer([
        ("cat",  cat_tr_xgb, categorical_cols),
        ("bool", bool_tr,   boolean_cols),
        ("num",  SimpleImputer(strategy="median"),   numeric_cols),
    ])
    prep.set_output(transform='pandas')
    return Pipeline([("preprocessor", prep),
                     ("classifier",   xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                                        eval_metric="logloss",
                                                        enable_categorical=True,
                                                        random_state=42))])


def cv_metrics(dev_df, numeric_cols, categorical_cols, boolean_cols) -> list[dict]:
    feature_cols = numeric_cols + categorical_cols + boolean_cols
    records = []
    for model_name in ["LogisticRegression", "XGBoost"]:
        for fold_id in sorted(dev_df["fold"].unique()):
            tr_mask  = dev_df["fold"] != fold_id
            val_mask = dev_df["fold"] == fold_id
            X_tr,  y_tr  = dev_df.loc[tr_mask,  feature_cols], dev_df.loc[tr_mask,  TARGET]
            X_val, y_val = dev_df.loc[val_mask, feature_cols], dev_df.loc[val_mask, TARGET]
            spw = ((len(y_tr) - y_tr.sum()) / y_tr.sum()) if y_tr.sum() > 0 else 1.0
            clf = (build_pipeline_lr(numeric_cols, categorical_cols, boolean_cols)
                   if model_name == "LogisticRegression"
                   else build_pipeline_xgb(numeric_cols, categorical_cols, boolean_cols, spw))
            clf.fit(X_tr, y_tr)
            y_pred  = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1]
            records.append({
                "model": model_name, "fold": fold_id,
                "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
                "roc_auc":  roc_auc_score(y_val, y_proba) if y_val.nunique() > 1 else np.nan,
                "pr_auc":   average_precision_score(y_val, y_proba) if y_val.nunique() > 1 else np.nan,
                "brier_score": brier_score_loss(y_val, y_proba),
            })
    return records


def run() -> None:
    # 1. Load v2 feature sets
    dev_v2  = pd.read_csv(FEATURES_DIR / "features_v2_dev.csv")
    hold_v2 = pd.read_csv(FEATURES_DIR / "features_v2_holdout.csv")

    # 2. Engineer v3
    dev_v3  = engineer_v3_features(dev_v2)
    hold_v3 = engineer_v3_features(hold_v2)

    # 3. Save v3
    dev_v3.to_csv(FEATURES_DIR / "features_v3_dev.csv", index=False)
    hold_v3.to_csv(FEATURES_DIR / "features_v3_holdout.csv", index=False)

    numeric_cols = get_numeric_cols(dev_v3)

    print(f"Total features in v3: {len(numeric_cols) + len(CATEGORICAL_COLS) + len(BOOLEAN_COLS)}")

    # 4. Evaluate CV
    print("[CV] evaluating v3...")
    ablation_results = {"v3": cv_metrics(dev_v3, numeric_cols, CATEGORICAL_COLS, BOOLEAN_COLS)}

    # Load baseline v2 metrics
    v2_metrics_path = BASELINE_DIR / "incremental_gain_v2.csv"
    if v2_metrics_path.exists():
        v2_records = pd.read_csv(v2_metrics_path)
        ablation_results["v2"] = v2_records[v2_records["feature_set"] == "v2"].to_dict("records")

    metric_cols = ["balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]
    summary_rows: list[dict] = []
    fold_rows: list[dict] = []

    for label, records in ablation_results.items():
        df = pd.DataFrame(records)
        fold_df = df.copy(); fold_df["feature_set"] = label
        fold_rows.append(fold_df)
        mean = df.groupby("model")[metric_cols].mean()
        for model, row in mean.iterrows():
            summary_rows.append({"feature_set": label, "model": model,
                                  **{m: round(row[m], 4) for m in metric_cols}})

    summary_df = pd.DataFrame(summary_rows)
    pd.concat(fold_rows).to_csv(BASELINE_DIR / "incremental_gain_v3.csv", index=False)
    summary_df.to_csv(BASELINE_DIR / "ablation_summary_v3.csv", index=False)

    print(summary_df.sort_values(["model", "feature_set"]))
    print("Done generating v3.")

if __name__ == "__main__":
    run()
