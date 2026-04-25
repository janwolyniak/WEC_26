"""
Step 11: Feature Engineering Iteration 1 (Main Table)

Implements plan.md §11:
  1. Relative intensity features: last15/cumul ratios and pace deltas.
  2. Per-minute exposure normalisation (per-minute cumulative rate, last15 rate).
  3. Interaction features with position, formation, is_home.
  4. Leakage correction: drop minute_out and subbed (flagged in docs/leakage_checklist.md).
  5. Refit under same grouped CV protocol and compare vs baseline.

Outputs:
  artifacts/features/features_v1_dev.csv        – engineered development partition
  artifacts/features/features_v1_holdout.csv    – engineered holdout partition
  artifacts/baseline/incremental_gain_v1.csv    – metric deltas vs baseline
  docs/feature_engineering_v1.md               – incremental gain report
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
ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
ARTIFACTS_DIR = ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
DOCS_DIR = ROOT / "docs"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Base physical metrics present in main table
# ---------------------------------------------------------------------------
PHYSICAL_METRICS = [
    "sprints", "hsr", "distance", "mean_max_speed", "peak_speed",
    "shots", "shots_on_target", "shots_under_press", "shots_top_third",
]

# Map checkpoint_period to the minute offset of that period's start
PERIOD_OFFSET = {"half_1": 0, "half_2": 45, "extra_time_1": 90}

# Positions for indicator features
POSITIONS = ["A", "D", "M", "G"]

EPSILON = 1e-6  # avoid division by zero


# ---------------------------------------------------------------------------
# Feature engineering
# All features are computed row-by-row from pre-checkpoint data only:
#   - ratios / deltas use last15 and cumul columns (both causal per leakage audit)
#   - per-minute features use checkpoint_cont_min - minute_in (entry time is causal)
#   - interactions multiply existing row-level scalars
#   - minute_out and subbed are EXCLUDED (leakage per docs/leakage_checklist.md)
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- Continuous match minute at checkpoint ---
    out["checkpoint_cont_min"] = (
        out["checkpoint_period"].map(PERIOD_OFFSET).fillna(0).astype(float)
        + out["checkpoint_min"].astype(float)
    )

    # --- Minutes elapsed since player entered pitch (floor at 1 to avoid /0) ---
    out["elapsed_min"] = (
        out["checkpoint_cont_min"] - out["minute_in"].astype(float)
    ).clip(lower=1.0)

    # --- Per physical metric features ---
    for m in PHYSICAL_METRICS:
        c_col = f"cumul_{m}"
        l_col = f"last15_{m}"
        if c_col not in out.columns or l_col not in out.columns:
            continue

        c = out[c_col].astype(float)
        l = out[l_col].astype(float)

        # 1. Recency ratio: how much of total activity happened in last 15 min
        out[f"ratio_{m}"] = l / (c + EPSILON)

        # 2. Expected last-15 pace based on cumulative rate over elapsed time
        expected_pace = (c / out["elapsed_min"]) * 15.0
        out[f"pace_delta_{m}"] = l - expected_pace

        # 3. Per-minute cumulative rate
        out[f"pm_{m}"] = c / out["elapsed_min"]

        # 4. Per-minute last-15 rate (standardised to per-minute)
        out[f"rate15_{m}"] = l / 15.0

    # --- Position binary indicators ---
    for pos in POSITIONS:
        out[f"is_pos_{pos}"] = (out["position"] == pos).astype(float)

    # --- Late-game indicator (H2_30, H2_45) ---
    out["is_late_game"] = out["checkpoint"].isin(["H2_30", "H2_45"]).astype(float)

    # --- Context interactions ---
    is_home_f = out["is_home"].astype(float)
    out["home_x_last15_sprints"] = is_home_f * out["last15_sprints"].astype(float)
    out["home_x_last15_shots"]   = is_home_f * out["last15_shots"].astype(float)
    out["home_x_rate15_sprints"] = is_home_f * out.get("rate15_sprints",
                                    out["last15_sprints"].astype(float) / 15.0)

    out["attacker_x_cumul_shots"]           = out["is_pos_A"] * out["cumul_shots"].astype(float)
    out["attacker_x_cumul_shots_on_target"] = out["is_pos_A"] * out["cumul_shots_on_target"].astype(float)
    out["midfielder_x_cumul_shots"]         = out["is_pos_M"] * out["cumul_shots"].astype(float)

    out["late_x_rate15_sprints"] = out["is_late_game"] * out.get(
        "rate15_sprints", out["last15_sprints"].astype(float) / 15.0
    )

    return out


# ---------------------------------------------------------------------------
# Feature column lists (after engineering)
# ---------------------------------------------------------------------------
def get_feature_lists(df: pd.DataFrame):
    base_numeric = [
        c for c in df.columns
        if c.startswith("last15_") or c.startswith("cumul_")
    ] + ["checkpoint_min", "checkpoint_cont_min", "elapsed_min", "minute_in"]
    # NOTE: minute_out and subbed excluded — leakage (docs/leakage_checklist.md)

    engineered_numeric = [
        c for c in df.columns
        if c.startswith("ratio_") or c.startswith("pace_delta_")
        or c.startswith("pm_") or c.startswith("rate15_")
        or c in ("is_late_game", "home_x_last15_sprints", "home_x_last15_shots",
                 "home_x_rate15_sprints", "attacker_x_cumul_shots",
                 "attacker_x_cumul_shots_on_target", "midfielder_x_cumul_shots",
                 "late_x_rate15_sprints")
        or c.startswith("is_pos_")
    ]

    categorical = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
    boolean     = ["is_home"]

    return base_numeric + engineered_numeric, categorical, boolean


# ---------------------------------------------------------------------------
# Build sklearn pipelines
# ---------------------------------------------------------------------------
def _cat_tr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_pipeline_lr(numeric_cols, categorical_cols, boolean_cols) -> Pipeline:
    num_tr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])
    bool_tr = FunctionTransformer(lambda x: x.astype(float))
    prep = ColumnTransformer([
        ("num",  num_tr,    numeric_cols),
        ("cat",  _cat_tr(), categorical_cols),
        ("bool", bool_tr,   boolean_cols),
    ])
    return Pipeline([
        ("preprocessor", prep),
        ("classifier",   LogisticRegression(class_weight="balanced",
                                            max_iter=1000, random_state=42)),
    ])

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def build_pipeline_xgb(numeric_cols, categorical_cols, boolean_cols,
                        scale_pos_weight: float) -> Pipeline:
    num_imp  = SimpleImputer(strategy="median")  # no scaler for trees
    bool_tr  = FunctionTransformer(to_float)
    cat_tr_xgb = FunctionTransformer(to_category)
    prep = ColumnTransformer([
        ("cat",  cat_tr_xgb,  categorical_cols),
        ("bool", bool_tr,    boolean_cols),
        ("num",  num_imp,    numeric_cols),
    ])
    prep.set_output(transform='pandas')
    return Pipeline([
        ("preprocessor", prep),
        ("classifier",   xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                            eval_metric="logloss",
                                            enable_categorical=True,
                                            random_state=42)),
    ])


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc":  roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else np.nan,
        "pr_auc":   average_precision_score(y_true, y_proba) if y_true.nunique() > 1 else np.nan,
        "brier_score": brier_score_loss(y_true, y_proba),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    TARGET = "scored_after"

    # Load and engineer features
    df = pd.read_csv(SPLITS_DIR / "modeling_row_folds.csv")
    df = engineer_features(df)

    numeric_cols, categorical_cols, boolean_cols = get_feature_lists(df)
    feature_cols = numeric_cols + categorical_cols + boolean_cols

    dev_df  = df[df["partition_role"] == "development"].copy()
    hold_df = df[df["partition_role"] == "holdout"].copy()

    # Save engineered partitions
    dev_df.to_csv(FEATURES_DIR / "features_v1_dev.csv", index=False)
    hold_df.to_csv(FEATURES_DIR / "features_v1_holdout.csv", index=False)
    print(f"Saved features_v1 partitions — dev: {len(dev_df)} rows, "
          f"holdout: {len(hold_df)} rows, "
          f"feature cols: {len(feature_cols)}")

    # Cross-validation
    fold_records: list[dict] = []

    for model_name in ["LogisticRegression", "XGBoost"]:
        print(f"[{model_name}] Running CV folds...")
        for fold_id in sorted(dev_df["fold"].unique()):
            train_mask = dev_df["fold"] != fold_id
            val_mask   = dev_df["fold"] == fold_id

            X_tr  = dev_df.loc[train_mask, feature_cols]
            y_tr  = dev_df.loc[train_mask, TARGET]
            X_val = dev_df.loc[val_mask,   feature_cols]
            y_val = dev_df.loc[val_mask,   TARGET]

            spw = ((len(y_tr) - y_tr.sum()) / y_tr.sum()) if y_tr.sum() > 0 else 1.0

            clf = (build_pipeline_lr(numeric_cols, categorical_cols, boolean_cols)
                   if model_name == "LogisticRegression"
                   else build_pipeline_xgb(numeric_cols, categorical_cols, boolean_cols, spw))

            clf.fit(X_tr, y_tr)
            y_pred  = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1]

            fold_records.append({
                "model": model_name,
                "fold": fold_id,
                "feature_set": "v1",
                **metrics(y_val, pd.Series(y_pred), pd.Series(y_proba)),
            })

    fold_df_v1 = pd.DataFrame(fold_records)
    fold_df_v1.to_csv(BASELINE_DIR / "fold_metrics_v1.csv", index=False)

    # Compare against baseline
    baseline_fold = pd.read_csv(BASELINE_DIR / "fold_metrics.csv")
    baseline_fold["feature_set"] = "baseline"

    summary_v1  = fold_df_v1.groupby("model")[["balanced_accuracy", "roc_auc",
                                                "pr_auc", "brier_score"]].mean()
    summary_b   = baseline_fold.groupby("model")[["balanced_accuracy", "roc_auc",
                                                   "pr_auc", "brier_score"]].mean()

    metric_cols = ["balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]
    gain_rows: list[dict] = []
    for model in summary_v1.index:
        if model not in summary_b.index:
            continue
        row = {"model": model}
        for mc in metric_cols:
            row[f"{mc}_baseline"] = round(summary_b.loc[model, mc], 4)
            row[f"{mc}_v1"]       = round(summary_v1.loc[model, mc], 4)
            delta = summary_v1.loc[model, mc] - summary_b.loc[model, mc]
            # Brier: lower is better, so negate the sign of gain
            row[f"{mc}_delta"]    = round(delta, 4)
        gain_rows.append(row)

    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(BASELINE_DIR / "incremental_gain_v1.csv", index=False)

    _write_report(gain_df, fold_df_v1, baseline_fold, numeric_cols, categorical_cols, boolean_cols)
    print("Done. Outputs written to artifacts/features/, artifacts/baseline/, "
          "and docs/feature_engineering_v1.md")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _write_report(gain_df, fold_v1, fold_baseline, numeric_cols,
                  categorical_cols, boolean_cols) -> None:
    lines: list[str] = []
    a = lines.append

    a("# Feature Engineering v1 — Incremental Gain Report")
    a("")
    a("Generated by `src/python/feature_engineering_v1.py` (Step 11 of `docs/plan.md`).")
    a("")

    # Feature inventory
    a("## 1. Engineered Feature Inventory")
    a("")
    a(f"Total numeric features: {len(numeric_cols)}")
    a(f"Categorical features:   {len(categorical_cols)}")
    a(f"Boolean features:       {len(boolean_cols)}")
    a("")
    a("### New feature families added")
    a("")
    a("| Family | Pattern | Description |")
    a("|---|---|---|")
    a("| Recency ratio | `ratio_{metric}` | `last15 / (cumul + ε)` — share of total activity in last 15 min |")
    a("| Pace delta    | `pace_delta_{metric}` | `last15 − expected_last15` — surge/decline relative to cumulative pace |")
    a("| Per-minute cumul | `pm_{metric}` | `cumul / elapsed_min` — average intensity per minute since kick-off |")
    a("| Per-minute last15 | `rate15_{metric}` | `last15 / 15` — activity rate over last 15-min window |")
    a("| Elapsed time  | `elapsed_min`, `checkpoint_cont_min` | Continuous match clock and player active minutes |")
    a("| Position indicators | `is_pos_{A,D,M,G}` | Binary flags per position category |")
    a("| Late-game flag | `is_late_game` | 1 if checkpoint ∈ {H2_30, H2_45} |")
    a("| Home interactions | `home_x_*` | is_home × recent sprint/shot rates |")
    a("| Position interactions | `attacker_x_*`, `midfielder_x_*` | Position × cumulative shot counts |")
    a("| Late interactions | `late_x_rate15_sprints` | Late-game × recent sprint rate |")
    a("")
    a("### Leakage corrections applied")
    a("")
    a("- `minute_out` **removed** — carries post-checkpoint exit time (critical leakage, "
      "see `docs/leakage_checklist.md`).")
    a("- `subbed` **removed** — substitution flag encodes future event in 96.4% of rows "
      "(critical leakage, see `docs/leakage_checklist.md`).")
    a("")

    # Incremental gain table
    a("## 2. Incremental Gain vs Baseline")
    a("")
    a("Positive Δ = improvement for Bal. Acc., ROC AUC, PR AUC. "
      "Negative Δ = improvement for Brier score.")
    a("")
    a("| Model | Metric | Baseline | v1 | Δ |")
    a("|---|---|---:|---:|---:|")
    metric_labels = {
        "balanced_accuracy": "Balanced Accuracy",
        "roc_auc":           "ROC AUC",
        "pr_auc":            "PR AUC",
        "brier_score":       "Brier Score",
    }
    for _, row in gain_df.iterrows():
        for mc, label in metric_labels.items():
            delta = row[f"{mc}_delta"]
            sign  = "+" if delta > 0 else ""
            # For Brier, improvement = negative delta
            if mc == "brier_score":
                sign = "-" if delta < 0 else "+"
            a(f"| {row['model']} | {label} | {row[f'{mc}_baseline']:.4f}"
              f" | {row[f'{mc}_v1']:.4f} | {sign}{delta:.4f} |")
    a("")

    # Per-fold breakdown
    a("## 3. Per-Fold CV Detail (v1)")
    a("")
    a("| Model | Fold | Bal. Acc. | ROC AUC | PR AUC | Brier |")
    a("|---|---|---:|---:|---:|---:|")
    for _, r in fold_v1.sort_values(["model", "fold"]).iterrows():
        a(f"| {r['model']} | {r['fold']} | {r['balanced_accuracy']:.4f}"
          f" | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['brier_score']:.4f} |")
    a("")

    # Interpretation
    a("## 4. Interpretation and Next Steps")
    a("")
    a("### What the v1 features add")
    a("- **Pace delta and recency ratio** surface whether a player is accelerating or declining"
      " relative to their match-average. A positive `pace_delta_shots` signals a shooting surge"
      " in the last 15 minutes — a pattern directly relevant to `scored_after`.")
    a("- **Per-minute rates** normalise for time on pitch, removing the confound that longer"
      " exposures naturally accumulate higher raw counts.")
    a("- **Position indicators** allow the model to adjust expected baselines per position"
      " without relying entirely on OHE encoding inside the categorical branch.")
    a("- **Interaction terms** test whether home advantage or late-game context modulates"
      " the relevance of physical intensity signals.")
    a("")
    a("### Leakage correction impact")
    a("Removing `minute_out` and `subbed` eliminates two critical leakage sources. Any metric"
      " change between baseline and v1 partly reflects this correction: the baseline numbers"
      " were slightly optimistic due to these leaking columns.")
    a("")
    a("### Next steps (Step 12)")
    a("Proceed to supplementary table feature extraction: pass accuracy, pressure turnover rate,"
      " and progressive run aggregates as specified in `docs/feature_extension_plan.md`.")

    (DOCS_DIR / "feature_engineering_v1.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
