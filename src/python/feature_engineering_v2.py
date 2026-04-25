"""
Step 12: Feature Engineering Iteration 2 (Supplementary Tables)

Implements plan.md §12 and docs/event_aggregation_specification.md:
  1. Pass-derived features: count, accuracy rate, top/middle share.
  2. Pressure-derived features: count, accuracy rate, turnover rate,
     top share, pass-angle observed share.
  3. Run features not in main table: count, distance sum, peak speed,
     mean max speed, sprint share, top share.
  4. Shot features (selective, low coverage): count, top share,
     under-pressure rate, regular-play share.
  5. Merge all families at checkpoint level with strict causal cutoff.
  6. Ablation evaluation: each family individually + full v2.

Causal windows per event_aggregation_specification.md:
  - cumul:  event_cont_min <= checkpoint_cont_min
  - last15: (checkpoint_cont_min - 15) < event_cont_min <= checkpoint_cont_min

Join key: player_appearance_id
Zero-event windows: count → 0, rate/share → 0 (see spec §Join Key)

Outputs:
  artifacts/features/features_v2_dev.csv        – full v2 development set
  artifacts/features/features_v2_holdout.csv    – full v2 holdout set
  artifacts/features/features_v2_pass_dev.csv   – v1 + pass (ablation)
  artifacts/features/features_v2_pressure_dev.csv
  artifacts/features/features_v2_run_dev.csv
  artifacts/baseline/incremental_gain_v2.csv    – ablation metrics vs v1
  docs/feature_engineering_v2.md               – incremental gain report
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
NORM_DIR     = ROOT / "artifacts" / "data_intake" / "normalized"
FEATURES_DIR = ROOT / "artifacts" / "features"
BASELINE_DIR = ROOT / "artifacts" / "baseline"
DOCS_DIR     = ROOT / "docs"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Period offset for continuous-minute conversion (per event_aggregation_specification.md)
PERIOD_OFFSET: dict[str, float] = {
    "half_1": 0.0, "half_2": 45.0, "extra_time_1": 90.0, "extra_time_2": 105.0
}

TARGET = "scored_after"


# ---------------------------------------------------------------------------
# Generic windowing helper
# ---------------------------------------------------------------------------
def _attach_windows(events: pd.DataFrame, checkpoints: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-join event rows with checkpoint rows for the same player_appearance_id,
    then split into cumul and last-15 windows.
    Returns (cumul_df, last15_df), each still containing all original event columns
    plus 'checkpoint' and 'checkpoint_cont_min'.
    """
    ev = events.copy()
    ev["event_cont_min"] = (
        ev["period"].map(PERIOD_OFFSET).fillna(0).astype(float)
        + ev["minute"].astype(float)
    )
    cp = checkpoints[["player_appearance_id", "checkpoint", "checkpoint_cont_min"]]
    merged = ev.merge(cp, on="player_appearance_id", how="inner")

    cumul  = merged[merged["event_cont_min"] <= merged["checkpoint_cont_min"]].copy()
    last15 = merged[
        (merged["event_cont_min"] >  merged["checkpoint_cont_min"] - 15.0) &
        (merged["event_cont_min"] <= merged["checkpoint_cont_min"])
    ].copy()
    return cumul, last15


def _fill_zeros(agg: pd.DataFrame, base: pd.DataFrame, fill_val: float = 0.0
                ) -> pd.DataFrame:
    """
    Left-join agg onto base (player_appearance_id × checkpoint), filling missing
    windows with fill_val per the spec: 'count → 0, rate/share → 0'.
    """
    keys = ["player_appearance_id", "checkpoint"]
    result = base[keys].merge(agg, on=keys, how="left")
    num_cols = result.select_dtypes(include="number").columns.difference(keys)
    result[num_cols] = result[num_cols].fillna(fill_val)
    return result


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe rate: 0 when denominator is 0 (no events → no share)."""
    return np.where(denominator > 0, numerator / denominator, 0.0)


# ---------------------------------------------------------------------------
# Per-family aggregation functions
# ---------------------------------------------------------------------------

def _agg_pass(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Aggregates for one time window of pass events.
    prefix: 'cumul_pass' or 'last15_pass'
    """
    grp = df.groupby(["player_appearance_id", "checkpoint"])
    count      = grp["id"].count().rename(f"{prefix}_count")
    acc_sum    = grp["accurate"].apply(lambda s: s.astype(float).sum())
    top_sum    = grp["stage"].apply(lambda s: (s == "top").sum())
    middle_sum = grp["stage"].apply(lambda s: (s == "middle").sum())

    out = pd.concat([count, acc_sum, top_sum, middle_sum], axis=1)
    out.columns = [f"{prefix}_count", "acc_sum", "top_sum", "middle_sum"]
    out = out.reset_index()

    out[f"{prefix}_accurate_rate"] = _rate(out["acc_sum"],    out[f"{prefix}_count"])
    out[f"{prefix}_top_share"]     = _rate(out["top_sum"],    out[f"{prefix}_count"])
    out[f"{prefix}_middle_share"]  = _rate(out["middle_sum"], out[f"{prefix}_count"])
    return out.drop(columns=["acc_sum", "top_sum", "middle_sum"])


def _agg_pressure(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """prefix: 'cumul_pressure' or 'last15_pressure'"""
    grp = df.groupby(["player_appearance_id", "checkpoint"])
    count     = grp["id"].count().rename(f"{prefix}_count")
    acc_sum   = grp["accurate"].apply(lambda s: s.astype(float).sum())
    turn_sum  = grp["press_induced_outcome"].apply(lambda s: (s == "turnover").sum())
    top_sum   = grp["stage"].apply(lambda s: (s == "top").sum())
    angle_sum = grp["pass_angle"].apply(lambda s: s.notna().sum())

    out = pd.concat([count, acc_sum, turn_sum, top_sum, angle_sum], axis=1)
    out.columns = [f"{prefix}_count", "acc_s", "turn_s", "top_s", "angle_s"]
    out = out.reset_index()

    out[f"{prefix}_accurate_rate"]              = _rate(out["acc_s"],   out[f"{prefix}_count"])
    out[f"{prefix}_turnover_rate"]              = _rate(out["turn_s"],  out[f"{prefix}_count"])
    out[f"{prefix}_top_share"]                  = _rate(out["top_s"],   out[f"{prefix}_count"])
    out[f"{prefix}_pass_angle_observed_share"]  = _rate(out["angle_s"], out[f"{prefix}_count"])
    return out.drop(columns=["acc_s", "turn_s", "top_s", "angle_s"])


def _agg_run(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """prefix: 'cumul_run' or 'last15_run'"""
    grp = df.groupby(["player_appearance_id", "checkpoint"])
    count      = grp["id"].count().rename(f"{prefix}_count")
    dist_sum   = grp["distance"].sum()
    peak_spd   = grp["max_speed"].max()
    mean_spd   = grp["max_speed"].mean()
    sprint_sum = grp["run_type"].apply(lambda s: (s == "sprint").sum())
    top_sum    = grp["stage"].apply(lambda s: (s == "top").sum())

    out = pd.concat([count, dist_sum, peak_spd, mean_spd, sprint_sum, top_sum], axis=1)
    out.columns = [f"{prefix}_count", f"{prefix}_distance_sum",
                   f"{prefix}_peak_speed", f"{prefix}_mean_max_speed",
                   "sprint_s", "top_s"]
    out = out.reset_index()

    out[f"{prefix}_sprint_share"] = _rate(out["sprint_s"], out[f"{prefix}_count"])
    out[f"{prefix}_top_share"]    = _rate(out["top_s"],    out[f"{prefix}_count"])
    return out.drop(columns=["sprint_s", "top_s"])


def _agg_shot(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """prefix: 'cumul_shot' or 'last15_shot'"""
    grp = df.groupby(["player_appearance_id", "checkpoint"])
    count    = grp["id"].count().rename(f"{prefix}_count")
    top_sum  = grp["stage"].apply(lambda s: (s == "top").sum())
    press_sum = grp["under_pressure"].apply(lambda s: s.astype(float).sum())
    reg_sum  = grp["play_pattern"].apply(lambda s: (s == "regular_play").sum())

    out = pd.concat([count, top_sum, press_sum, reg_sum], axis=1)
    out.columns = [f"{prefix}_count", "top_s", "press_s", "reg_s"]
    out = out.reset_index()

    out[f"{prefix}_top_share"]              = _rate(out["top_s"],   out[f"{prefix}_count"])
    out[f"{prefix}_under_pressure_rate"]    = _rate(out["press_s"], out[f"{prefix}_count"])
    out[f"{prefix}_regular_play_share"]     = _rate(out["reg_s"],   out[f"{prefix}_count"])
    return out.drop(columns=["top_s", "press_s", "reg_s"])


# ---------------------------------------------------------------------------
# Build one event-family feature block for a partition (dev or holdout)
# ---------------------------------------------------------------------------
def build_event_features(event_df: pd.DataFrame, checkpoints: pd.DataFrame,
                          agg_fn, prefix: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed on (player_appearance_id, checkpoint)
    with both cumul and last15 features, zero-filled for missing windows.
    """
    cumul_ev, last15_ev = _attach_windows(event_df, checkpoints)

    keys = ["player_appearance_id", "checkpoint"]
    cumul_agg  = _fill_zeros(_agg_fn_call(agg_fn, cumul_ev,  f"cumul_{prefix}"),  checkpoints, 0.0)
    last15_agg = _fill_zeros(_agg_fn_call(agg_fn, last15_ev, f"last15_{prefix}"), checkpoints, 0.0)

    result = cumul_agg.merge(last15_agg, on=keys, how="outer")
    return result


def _agg_fn_call(fn, df, prefix):
    if df.empty:
        return pd.DataFrame(columns=["player_appearance_id", "checkpoint"])
    return fn(df, prefix)


# ---------------------------------------------------------------------------
# Pipeline helpers (identical structure to v1)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    # --- Load v1 feature sets ---
    dev_v1  = pd.read_csv(FEATURES_DIR / "features_v1_dev.csv")
    hold_v1 = pd.read_csv(FEATURES_DIR / "features_v1_holdout.csv")

    cp_dev  = dev_v1[["player_appearance_id", "checkpoint", "checkpoint_cont_min"]]
    cp_hold = hold_v1[["player_appearance_id", "checkpoint", "checkpoint_cont_min"]]

    # --- Load event tables (normalized) ---
    pass_ev  = pd.read_csv(NORM_DIR / "player_appearance_pass_normalized.csv")
    press_ev = pd.read_csv(NORM_DIR / "player_appearance_behaviour_under_pressure_normalized.csv")
    run_ev   = pd.read_csv(NORM_DIR / "player_appearance_run_normalized.csv")
    shot_ev  = pd.read_csv(NORM_DIR / "player_appearance_shot_limited_normalized.csv")

    # Shot: pre-filter own-goal perspective rows (per spec)
    shot_ev = shot_ev[shot_ev["own_goal_player_appearance_id"].isna()].copy()

    event_specs = [
        ("pass",     pass_ev,  _agg_pass),
        ("pressure", press_ev, _agg_pressure),
        ("run",      run_ev,   _agg_run),
        ("shot",     shot_ev,  _agg_shot),
    ]

    # --- Build event features for dev and holdout ---
    family_feats_dev:  dict[str, pd.DataFrame] = {}
    family_feats_hold: dict[str, pd.DataFrame] = {}

    for family, ev_df, agg_fn in event_specs:
        print(f"Aggregating {family} events...")
        family_feats_dev[family]  = build_event_features(ev_df, cp_dev,  agg_fn, family)
        family_feats_hold[family] = build_event_features(ev_df, cp_hold, agg_fn, family)

    # --- Merge all families onto v1 ---
    keys = ["player_appearance_id", "checkpoint"]

    def merge_families(base, families):
        out = base.copy()
        for feats in families.values():
            out = out.merge(feats, on=keys, how="left")
        return out

    dev_v2  = merge_families(dev_v1,  family_feats_dev)
    hold_v2 = merge_families(hold_v1, family_feats_hold)

    dev_v2.to_csv(FEATURES_DIR  / "features_v2_dev.csv",     index=False)
    hold_v2.to_csv(FEATURES_DIR / "features_v2_holdout.csv", index=False)

    # --- Identify feature column buckets for v2 ---
    v1_numeric_cols = [
        c for c in dev_v1.columns
        if c.startswith("last15_") or c.startswith("cumul_")
        or c.startswith("ratio_") or c.startswith("pace_delta_")
        or c.startswith("pm_") or c.startswith("rate15_")
        or c.startswith("is_pos_") or c.startswith("home_x_")
        or c.startswith("attacker_x_") or c.startswith("midfielder_x_")
        or c.startswith("late_x_")
        or c in ("checkpoint_min", "checkpoint_cont_min", "elapsed_min",
                 "minute_in", "is_late_game")
    ]
    cat_cols  = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
    bool_cols = ["is_home"]

    # New event-derived numeric features
    event_numeric: dict[str, list[str]] = {}
    for family in family_feats_dev:
        cols = [c for c in family_feats_dev[family].columns
                if c not in ("player_appearance_id", "checkpoint")]
        event_numeric[family] = cols

    all_event_num = [c for cols in event_numeric.values() for c in cols]
    v2_numeric_cols = v1_numeric_cols + all_event_num

    # --- Ablation: save per-family augmented dev sets ---
    for family in event_numeric:
        aug = dev_v1.merge(family_feats_dev[family], on=keys, how="left")
        aug.to_csv(FEATURES_DIR / f"features_v2_{family}_dev.csv", index=False)

    # --- CV evaluation: ablations ---
    ablation_results: dict[str, list[dict]] = {}

    # Baseline v1 (reference)
    print("[CV] v1 baseline...")
    ablation_results["v1"] = cv_metrics(dev_v1, v1_numeric_cols, cat_cols, bool_cols)

    # Each family on top of v1
    for family in event_numeric:
        print(f"[CV] v1 + {family}...")
        aug_dev = dev_v1.merge(family_feats_dev[family], on=keys, how="left")
        ablation_results[f"v1+{family}"] = cv_metrics(
            aug_dev, v1_numeric_cols + event_numeric[family], cat_cols, bool_cols
        )

    # Full v2
    print("[CV] full v2...")
    ablation_results["v2"] = cv_metrics(dev_v2, v2_numeric_cols, cat_cols, bool_cols)

    # --- Summarise and save ---
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
    pd.concat(fold_rows).to_csv(BASELINE_DIR / "incremental_gain_v2.csv", index=False)
    summary_df.to_csv(BASELINE_DIR / "ablation_summary_v2.csv", index=False)

    _write_report(summary_df, event_numeric, dev_v2)
    print("Done. Outputs in artifacts/features/, artifacts/baseline/, "
          "docs/feature_engineering_v2.md")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _write_report(summary_df, event_numeric, dev_v2) -> None:
    lines: list[str] = []
    a = lines.append

    a("# Feature Engineering v2 — Incremental Gain Report")
    a("")
    a("Generated by `src/python/feature_engineering_v2.py` (Step 12 of `docs/plan.md`).")
    a("")

    # Feature inventory
    a("## 1. New Feature Families (Supplementary Tables)")
    a("")
    a("| Family | Source | Features | Count |")
    a("|---|---|---|---:|")
    for family, cols in event_numeric.items():
        a(f"| {family} | `player_appearance_{family}*.csv` | "
          f"{', '.join(cols[:3])}{'…' if len(cols)>3 else ''} | {len(cols)} |")
    total_new = sum(len(c) for c in event_numeric.values())
    a(f"| **Total new** | — | — | **{total_new}** |")
    a("")

    # Ablation table
    a("## 2. Ablation Results (Mean CV Performance)")
    a("")
    a("Each row adds one family to the v1 base. 'v2' = all families combined.")
    a("")
    a("| Feature Set | Model | Bal. Acc. | ROC AUC | PR AUC | Brier |")
    a("|---|---|---:|---:|---:|---:|")
    for _, row in summary_df.sort_values(["feature_set", "model"]).iterrows():
        a(f"| {row['feature_set']} | {row['model']}"
          f" | {row['balanced_accuracy']:.4f} | {row['roc_auc']:.4f}"
          f" | {row['pr_auc']:.4f} | {row['brier_score']:.4f} |")
    a("")

    # Delta vs v1
    a("## 3. Delta vs v1 Baseline")
    a("")
    a("| Feature Set | Model | ΔBal. Acc. | ΔROC AUC | ΔPR AUC | ΔBrier |")
    a("|---|---|---:|---:|---:|---:|")
    metric_cols = ["balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]
    v1_ref = summary_df[summary_df["feature_set"] == "v1"].set_index("model")
    for _, row in summary_df[summary_df["feature_set"] != "v1"].sort_values(
            ["feature_set", "model"]).iterrows():
        model = row["model"]
        if model not in v1_ref.index:
            continue
        deltas = {m: round(row[m] - v1_ref.loc[model, m], 4) for m in metric_cols}
        a(f"| {row['feature_set']} | {model}"
          f" | {deltas['balanced_accuracy']:+.4f} | {deltas['roc_auc']:+.4f}"
          f" | {deltas['pr_auc']:+.4f} | {deltas['brier_score']:+.4f} |")
    a("")

    # v2 feature count
    total_cols = len([c for c in dev_v2.columns if c not in
                      ("player_appearance_id", "player_id", "fixture_id", "date",
                       "scored_after", "fold", "partition_role")])
    a(f"## 4. Total Feature Space in v2: {total_cols} columns")
    a("")
    a("Ablation-ready datasets saved to `artifacts/features/features_v2_{{family}}_dev.csv`.")
    a("")
    a("## 5. Next Steps (Step 13)")
    a("Proceed to hyperparameter tuning for XGBoost and Logistic Regression under "
      "grouped CV using the v2 feature set.")

    (DOCS_DIR / "feature_engineering_v2.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
