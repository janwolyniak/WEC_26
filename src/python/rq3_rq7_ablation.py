from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_intake import NA_TOKENS, normalize_dataframe, project_root


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")


CHECKPOINT_TO_CONT_MIN = {
    "H1_15": 15,
    "H1_30": 30,
    "H1_45": 45,
    "H2_15": 60,
    "H2_30": 75,
    "H2_45": 90,
    "ET1_15": 105,
}
PERIOD_OFFSETS = {"half_1": 0, "half_2": 45, "extra_time_1": 90, "extra_time_2": 105}
CATEGORICAL_CONTEXT = ["checkpoint", "position", "formation", "is_home", "subbed"]
NUMERIC_CONTEXT = ["minute_in", "minute_out", "minutes_available_before_checkpoint"]
CONTEXT_FEATURES = CATEGORICAL_CONTEXT + NUMERIC_CONTEXT
BASE_BEHAVIOR_FEATURES = [
    "last15_sprints",
    "last15_hsr",
    "last15_distance",
    "last15_mean_max_speed",
    "last15_peak_speed",
    "last15_shots",
    "last15_shots_on_target",
    "last15_shots_under_press",
    "last15_shots_top_third",
    "cumul_sprints",
    "cumul_hsr",
    "cumul_distance",
    "cumul_mean_max_speed",
    "cumul_peak_speed",
    "cumul_shots",
    "cumul_shots_on_target",
    "cumul_shots_under_press",
    "cumul_shots_top_third",
]
DIRECT_SHOT_SPRINT_FEATURES = [
    "last15_sprints",
    "last15_hsr",
    "last15_shots",
    "last15_shots_on_target",
    "last15_shots_under_press",
    "last15_shots_top_third",
    "cumul_sprints",
    "cumul_hsr",
    "cumul_shots",
    "cumul_shots_on_target",
    "cumul_shots_under_press",
    "cumul_shots_top_third",
]
COUNT_LIKE_FAMILIES = [
    "sprints",
    "hsr",
    "distance",
    "shots",
    "shots_on_target",
    "shots_under_press",
    "shots_top_third",
]
ALL_BASE_FAMILIES = [
    "sprints",
    "hsr",
    "distance",
    "mean_max_speed",
    "peak_speed",
    "shots",
    "shots_on_target",
    "shots_under_press",
    "shots_top_third",
]
EVENT_SPECS: dict[str, list[dict[str, Any]]] = {
    "pass": [
        {"name": "pass_count", "kind": "count"},
        {"name": "pass_accurate_rate", "kind": "mean", "column": "accurate"},
        {"name": "pass_top_share", "kind": "share_eq", "column": "stage", "value": "top"},
        {"name": "pass_middle_share", "kind": "share_eq", "column": "stage", "value": "middle"},
    ],
    "pressure": [
        {"name": "pressure_count", "kind": "count"},
        {"name": "pressure_accurate_rate", "kind": "mean", "column": "accurate"},
        {"name": "pressure_turnover_rate", "kind": "share_eq", "column": "press_induced_outcome", "value": "turnover"},
        {"name": "pressure_top_share", "kind": "share_eq", "column": "stage", "value": "top"},
        {"name": "pressure_pass_angle_observed_share", "kind": "non_null_share", "column": "pass_angle"},
    ],
}


def load_normalized_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, keep_default_na=True, na_values=NA_TOKENS, low_memory=False)
    return normalize_dataframe(raw)


def prepare_base_panel(root: Path) -> pd.DataFrame:
    base = load_normalized_csv(root / "data" / "splits" / "modeling_row_folds.csv")
    base["checkpoint_cont_min"] = base["checkpoint"].astype(str).map(CHECKPOINT_TO_CONT_MIN)
    base["minutes_available_before_checkpoint"] = (
        base["checkpoint_cont_min"] - pd.to_numeric(base["minute_in"], errors="coerce") + 1
    ).clip(lower=1)
    return base


def add_event_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_cont_min"] = out["period"].astype(str).map(PERIOD_OFFSETS) + pd.to_numeric(out["minute"], errors="coerce")
    return out


def aggregate_event_window(window: pd.DataFrame, specs: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for spec in specs:
        name = f"{prefix}_{spec['name']}"
        if window.empty:
            features[name] = 0.0
        elif spec["kind"] == "count":
            features[name] = float(len(window))
        elif spec["kind"] == "mean":
            features[name] = float(pd.to_numeric(window[spec["column"]], errors="coerce").fillna(0.0).mean())
        elif spec["kind"] == "share_eq":
            features[name] = float(window[spec["column"]].astype(str).eq(spec["value"]).mean())
        elif spec["kind"] == "non_null_share":
            features[name] = float(window[spec["column"]].notna().mean())
        else:
            raise ValueError(f"Unsupported event aggregate kind: {spec['kind']}")
    return features


def build_event_family_features(base: pd.DataFrame, event_df: pd.DataFrame, family: str) -> pd.DataFrame:
    specs = EVENT_SPECS[family]
    zero_features = aggregate_event_window(pd.DataFrame(), specs, "last15") | aggregate_event_window(
        pd.DataFrame(), specs, "cumul"
    )
    grouped = {
        int(player_appearance_id): frame.sort_values(["event_cont_min", "id"]).reset_index(drop=True)
        for player_appearance_id, frame in event_df.groupby("player_appearance_id", dropna=False)
    }

    rows: list[dict[str, Any]] = []
    for row in base[["player_appearance_id", "fixture_id", "checkpoint", "checkpoint_cont_min"]].itertuples(index=False):
        current = grouped.get(int(row.player_appearance_id))
        if current is None:
            features = zero_features.copy()
        else:
            cumul_window = current[current["event_cont_min"] <= row.checkpoint_cont_min]
            last15_window = cumul_window[cumul_window["event_cont_min"] > row.checkpoint_cont_min - 15]
            features = aggregate_event_window(last15_window, specs, "last15") | aggregate_event_window(cumul_window, specs, "cumul")
        rows.append(
            {
                "player_appearance_id": int(row.player_appearance_id),
                "fixture_id": int(row.fixture_id),
                "checkpoint": row.checkpoint,
                **features,
            }
        )
    return pd.DataFrame(rows)


def build_relative_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    relative_features: list[str] = []
    history_features: list[str] = []
    recent_minutes = np.minimum(df["minutes_available_before_checkpoint"].to_numpy(dtype=float), 15.0)
    exposure_minutes = np.maximum(df["minutes_available_before_checkpoint"].to_numpy(dtype=float), 1.0)

    for family in ALL_BASE_FAMILIES:
        last_col = f"last15_{family}"
        cumul_col = f"cumul_{family}"
        delta_col = f"delta_vs_cumul_{family}"
        ratio_col = f"ratio_{family}"
        df[delta_col] = pd.to_numeric(df[last_col], errors="coerce").fillna(0.0) - pd.to_numeric(df[cumul_col], errors="coerce").fillna(0.0)
        df[ratio_col] = pd.to_numeric(df[last_col], errors="coerce").fillna(0.0) / np.maximum(
            pd.to_numeric(df[cumul_col], errors="coerce").fillna(0.0), 1.0
        )
        relative_features.extend([delta_col, ratio_col])

    for family in COUNT_LIKE_FAMILIES:
        last_values = pd.to_numeric(df[f"last15_{family}"], errors="coerce").fillna(0.0)
        cumul_values = pd.to_numeric(df[f"cumul_{family}"], errors="coerce").fillna(0.0)
        history_col = f"history_{family}"
        excess_col = f"excess_share_{family}"
        df[history_col] = (cumul_values - last_values).clip(lower=0.0)
        df[excess_col] = (last_values / np.maximum(recent_minutes, 1.0)) - (cumul_values / exposure_minutes)
        history_features.append(history_col)
        relative_features.append(excess_col)

    return df, relative_features, history_features


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical_columns = [column for column in feature_columns if column in CATEGORICAL_CONTEXT]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0)), ("scaler", StandardScaler())]), numeric_columns),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
        ]
    )


def make_pipeline(feature_columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(feature_columns)),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=17)),
        ]
    )


def score_predictions(y_true: np.ndarray, pred_prob: np.ndarray) -> dict[str, float]:
    pred_label = (pred_prob >= 0.5).astype(int)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred_label),
        "pr_auc": average_precision_score(y_true, pred_prob),
        "brier_score": brier_score_loss(y_true, pred_prob),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, pred_prob) if len(np.unique(y_true)) > 1 else np.nan
    return metrics


def evaluate_feature_set(dev_df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fold_artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for fold in sorted(dev_df["fold"].dropna().unique()):
        train_df = dev_df[dev_df["fold"] != fold].copy()
        valid_df = dev_df[dev_df["fold"] == fold].copy()
        pipeline = make_pipeline(feature_columns)
        pipeline.fit(train_df[feature_columns], train_df["scored_after"].astype(int))
        pred_prob = pipeline.predict_proba(valid_df[feature_columns])[:, 1]
        rows.append({"fold": fold, **score_predictions(valid_df["scored_after"].to_numpy(dtype=int), pred_prob)})
        fold_artifacts.append({"fold": fold, "pipeline": pipeline, "valid_df": valid_df[feature_columns + ["scored_after"]].copy()})
    return pd.DataFrame(rows), fold_artifacts


def mean_metric_row(rq: str, variant: str, blocks: list[str], features: list[str], metrics_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rq": rq,
        "variant": variant,
        "feature_blocks": " + ".join(blocks),
        "feature_count": len(features),
        **{metric: float(metrics_df[metric].mean()) for metric in ["balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]},
    }


def extract_relative_coefficients(fold_artifacts: list[dict[str, Any]], relative_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in fold_artifacts:
        prep = artifact["pipeline"].named_steps["prep"]
        coef = artifact["pipeline"].named_steps["model"].coef_[0]
        for feature_name, value in zip(prep.get_feature_names_out(), coef, strict=True):
            clean_name = feature_name.replace("num__", "")
            if clean_name in relative_features:
                rows.append({"fold": artifact["fold"], "feature": clean_name, "coefficient": float(value)})
    summary = pd.DataFrame(rows).groupby("feature", as_index=False).agg(
        mean_coefficient=("coefficient", "mean"),
        abs_mean_coefficient=("coefficient", lambda series: float(np.abs(series).mean())),
    )
    return summary.sort_values("abs_mean_coefficient", ascending=False).reset_index(drop=True)


def extract_context_importance(fold_artifacts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in fold_artifacts:
        valid_df = artifact["valid_df"]
        importance = permutation_importance(
            artifact["pipeline"],
            valid_df.drop(columns="scored_after"),
            valid_df["scored_after"].astype(int),
            scoring="roc_auc",
            n_repeats=10,
            random_state=17,
        )
        for feature, mean_drop in zip(valid_df.drop(columns="scored_after").columns.tolist(), importance.importances_mean, strict=True):
            if feature in CONTEXT_FEATURES:
                rows.append({"fold": artifact["fold"], "feature": feature, "mean_importance_drop": float(mean_drop)})
    return (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)
        .agg(mean_importance_drop=("mean_importance_drop", "mean"))
        .sort_values("mean_importance_drop", ascending=False)
        .reset_index(drop=True)
    )


def build_conclusions(results: pd.DataFrame, rq4_delta: pd.DataFrame, rq5_table: pd.DataFrame, rq6_gain: float, rq6_coeffs: pd.DataFrame, rq7_importance: pd.DataFrame) -> dict[str, str]:
    rq3 = results[results["rq"] == "RQ3"].set_index("variant")
    rq3_gap = float(rq3.loc["direct_shots_sprints_plus_context", "balanced_accuracy"] - rq3.loc["full_base_plus_context", "balanced_accuracy"])
    rq4_nonbaseline = rq4_delta[rq4_delta["variant"] != "base_plus_context"].copy()
    rq4_best = rq4_nonbaseline.sort_values("balanced_accuracy_delta", ascending=False).iloc[0]
    rq5_best = rq5_table.sort_values(["balanced_accuracy", "roc_auc"], ascending=[False, False]).iloc[0]
    rq6_top = rq6_coeffs.iloc[0]
    rq7_top = rq7_importance.iloc[0]
    rq7 = results[results["rq"] == "RQ7"].set_index("variant")
    rq7_context_gain = float(rq7.loc["context_only", "balanced_accuracy"] - rq7.loc["base_behavior_plus_context", "balanced_accuracy"])
    rq6_direction = "improves" if rq6_gain > 0 else "reduces"
    rq3_text = (
        f"Direct sprint-and-shot features outperform the broader base model by {rq3_gap:+.4f} Balanced Accuracy, "
        "so the leaner sprint-shot view is adequate in this CV setup."
        if rq3_gap >= 0
        else f"Direct sprint-and-shot features underperform the broader base model by {abs(rq3_gap):.4f} Balanced Accuracy, "
        "so they should be treated as informative but incomplete."
    )
    rq4_text = (
        f"{rq4_best['variant']} is the least harmful event extension, but it still shifts Balanced Accuracy by {rq4_best['balanced_accuracy_delta']:+.4f} versus the base+context reference."
        if float(rq4_best["balanced_accuracy_delta"]) <= 0
        else f"{rq4_best['variant']} gives the best event-family lift with a Balanced Accuracy delta of {rq4_best['balanced_accuracy_delta']:+.4f}."
    )
    return {
        "RQ3": rq3_text,
        "RQ4": rq4_text,
        "RQ5": f"{rq5_best['variant']} is the strongest temporal specification under the frozen CV protocol.",
        "RQ6": f"Adding relative-intensity features {rq6_direction} Balanced Accuracy by {rq6_gain:+.4f}; the strongest coefficient by magnitude is {rq6_top['feature']} ({rq6_top['mean_coefficient']:+.4f}).",
        "RQ7": f"The context-only model leads the context-augmented behavior model by {rq7_context_gain:+.4f} Balanced Accuracy, and {rq7_top['feature']} is the strongest context variable by mean ROC AUC permutation drop ({rq7_top['mean_importance_drop']:.4f}).",
    }


def run_analysis() -> dict[str, Any]:
    root = project_root()
    artifact_dir = root / "artifacts" / "rq3_rq7_ablation"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    base = prepare_base_panel(root)
    pass_df = add_event_time(load_normalized_csv(root / "data" / "player_appearance_pass.csv"))
    pressure_df = add_event_time(load_normalized_csv(root / "data" / "player_appearance_behaviour_under_pressure.csv"))

    pass_features = build_event_family_features(base, pass_df[pass_df["player_appearance_id"].isin(base["player_appearance_id"])], "pass")
    pressure_features = build_event_family_features(base, pressure_df[pressure_df["player_appearance_id"].isin(base["player_appearance_id"])], "pressure")

    modeling = base.merge(pass_features, on=["player_appearance_id", "fixture_id", "checkpoint"], how="left").merge(
        pressure_features, on=["player_appearance_id", "fixture_id", "checkpoint"], how="left"
    )
    modeling, relative_features, history_features = build_relative_features(modeling)

    development = modeling[modeling["partition_role"] == "development"].copy().reset_index(drop=True)
    pass_event_features = [column for column in development.columns if column.startswith("last15_pass_") or column.startswith("cumul_pass_")]
    pressure_event_features = [column for column in development.columns if column.startswith("last15_pressure_") or column.startswith("cumul_pressure_")]
    last15_features = [column for column in BASE_BEHAVIOR_FEATURES if column.startswith("last15_")]
    cumul_features = [column for column in BASE_BEHAVIOR_FEATURES if column.startswith("cumul_")]

    variants = [
        ("RQ3", "direct_shots_sprints_plus_context", ["direct_shots_sprints", "context"], DIRECT_SHOT_SPRINT_FEATURES + CONTEXT_FEATURES),
        ("RQ3", "full_base_plus_context", ["base_behavior", "context"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES),
        ("RQ4", "base_plus_context", ["base_behavior", "context"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES),
        ("RQ4", "base_plus_context_plus_pass", ["base_behavior", "context", "pass_event"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES + pass_event_features),
        ("RQ4", "base_plus_context_plus_pressure", ["base_behavior", "context", "pressure_event"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES + pressure_event_features),
        ("RQ4", "base_plus_context_plus_pass_pressure", ["base_behavior", "context", "pass_event", "pressure_event"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES + pass_event_features + pressure_event_features),
        ("RQ5", "last15_only_plus_context", ["last15", "context"], last15_features + CONTEXT_FEATURES),
        ("RQ5", "cumul_only_plus_context", ["cumul", "context"], cumul_features + CONTEXT_FEATURES),
        ("RQ5", "last15_plus_cumul_plus_context", ["last15", "cumul", "context"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES),
        ("RQ5", "history_counts_only_plus_context", ["history_counts", "context"], history_features + CONTEXT_FEATURES),
        ("RQ6", "base_plus_context", ["base_behavior", "context"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES),
        ("RQ6", "base_plus_context_plus_relative_intensity", ["base_behavior", "context", "relative_intensity"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES + relative_features),
        ("RQ7", "context_only", ["context"], CONTEXT_FEATURES),
        ("RQ7", "base_behavior_only", ["base_behavior"], BASE_BEHAVIOR_FEATURES),
        ("RQ7", "base_behavior_plus_context", ["base_behavior", "context"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES),
        ("RQ7", "base_behavior_plus_context_plus_pass_pressure", ["base_behavior", "context", "pass_event", "pressure_event"], BASE_BEHAVIOR_FEATURES + CONTEXT_FEATURES + pass_event_features + pressure_event_features),
    ]

    result_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[pd.DataFrame] = []
    fold_artifact_map: dict[str, list[dict[str, Any]]] = {}
    for rq, variant, blocks, features in variants:
        metrics_df, fold_artifacts = evaluate_feature_set(development, features)
        metrics_export = metrics_df.copy()
        metrics_export.insert(0, "variant", variant)
        metrics_export.insert(0, "rq", rq)
        fold_metric_rows.append(metrics_export)
        result_rows.append(mean_metric_row(rq, variant, blocks, features, metrics_df))
        fold_artifact_map[variant] = fold_artifacts

    results = pd.DataFrame(result_rows).sort_values(["rq", "balanced_accuracy", "roc_auc"], ascending=[True, False, False]).reset_index(drop=True)
    fold_metrics = pd.concat(fold_metric_rows, ignore_index=True)
    rq4_reference = results[results["variant"] == "base_plus_context"].iloc[0]
    rq4_delta = results[results["rq"] == "RQ4"].copy()
    rq4_delta["balanced_accuracy_delta"] = rq4_delta["balanced_accuracy"] - float(rq4_reference["balanced_accuracy"])
    rq4_delta["roc_auc_delta"] = rq4_delta["roc_auc"] - float(rq4_reference["roc_auc"])
    rq5_table = results[results["rq"] == "RQ5"].copy().reset_index(drop=True)
    rq6_coefficients = extract_relative_coefficients(fold_artifact_map["base_plus_context_plus_relative_intensity"], relative_features)
    rq7_importance = extract_context_importance(fold_artifact_map["base_behavior_plus_context"])
    rq6_gain = float(
        results.loc[results["variant"] == "base_plus_context_plus_relative_intensity", "balanced_accuracy"].iloc[0]
        - results.loc[results["variant"] == "base_plus_context", "balanced_accuracy"].iloc[0]
    )
    support_summary = pd.DataFrame(
        [
            {
                "family": "pass",
                "cumul_support_share": float((development["cumul_pass_count"] > 0).mean()),
                "last15_support_share": float((development["last15_pass_count"] > 0).mean()),
            },
            {
                "family": "pressure",
                "cumul_support_share": float((development["cumul_pressure_count"] > 0).mean()),
                "last15_support_share": float((development["last15_pressure_count"] > 0).mean()),
            },
        ]
    )
    conclusions = build_conclusions(results, rq4_delta, rq5_table, rq6_gain, rq6_coefficients, rq7_importance)

    results.to_csv(artifact_dir / "rq3_rq7_ablation_results.csv", index=False)
    fold_metrics.to_csv(artifact_dir / "rq3_rq7_fold_metrics.csv", index=False)
    rq4_delta.to_csv(artifact_dir / "rq4_incremental_deltas.csv", index=False)
    rq5_table.to_csv(artifact_dir / "rq5_temporal_comparison.csv", index=False)
    rq6_coefficients.to_csv(artifact_dir / "rq6_relative_intensity_coefficients.csv", index=False)
    rq7_importance.to_csv(artifact_dir / "rq7_context_importance.csv", index=False)
    support_summary.to_csv(artifact_dir / "event_support_summary.csv", index=False)

    return {
        "artifact_dir": artifact_dir,
        "results": results,
        "rq4_delta": rq4_delta,
        "rq5_table": rq5_table,
        "rq6_coefficients": rq6_coefficients,
        "rq7_importance": rq7_importance,
        "support_summary": support_summary,
        "conclusions": conclusions,
    }


if __name__ == "__main__":
    run_analysis()
