from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import random
from typing import Any

import numpy as np
import pandas as pd

from data_intake import NA_TOKENS, normalize_dataframe, project_root


@dataclass(frozen=True)
class SplitConfig:
    random_seed: int = 17
    holdout_fixture_fraction: float = 0.2
    n_dev_folds: int = 5
    holdout_restarts: int = 20
    fold_restarts: int = 20
    holdout_row_share_tolerance: float = 0.03
    holdout_target_rate_tolerance: float = 0.015
    holdout_home_away_share_tolerance: float = 0.03
    holdout_player_wmae_tolerance: float = 0.18
    cv_row_share_tolerance: float = 0.02
    cv_target_rate_tolerance: float = 0.01
    cv_home_away_share_tolerance: float = 0.02
    cv_player_wmae_tolerance: float = 0.22


def load_base_table() -> pd.DataFrame:
    root = project_root()
    path = root / "data" / "players_quarters_final.csv"
    raw = pd.read_csv(path, keep_default_na=True, na_values=NA_TOKENS, low_memory=False)
    return normalize_dataframe(raw)


def build_fixture_player_matrix(base: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(base["fixture_id"], base["player_id"])
    return matrix.sort_index(axis=0).sort_index(axis=1)


def build_fixture_metadata(base: pd.DataFrame) -> pd.DataFrame:
    grouped = base.groupby("fixture_id", dropna=False)
    metadata = grouped.agg(
        row_count=("fixture_id", "size"),
        positive_count=("scored_after", "sum"),
        unique_players=("player_id", "nunique"),
        unique_player_appearances=("player_appearance_id", "nunique"),
        home_rows=("is_home", lambda series: int(series.fillna(False).astype(bool).sum())),
        away_rows=("is_home", lambda series: int((~series.fillna(False).astype(bool)).sum())),
    ).reset_index()

    metadata["target_rate"] = metadata["positive_count"] / metadata["row_count"]
    metadata["team_proxy_count"] = 2
    return metadata.sort_values("fixture_id").reset_index(drop=True)


def build_score_inputs(
    fixture_metadata: pd.DataFrame,
    fixture_player_matrix: pd.DataFrame,
) -> dict[str, Any]:
    fixture_ids = fixture_metadata.index.tolist()
    aligned_players = fixture_player_matrix.loc[fixture_ids]
    return {
        "fixture_ids": fixture_ids,
        "fixture_to_position": {fixture_id: idx for idx, fixture_id in enumerate(fixture_ids)},
        "row_count": fixture_metadata.loc[fixture_ids, "row_count"].to_numpy(dtype=float),
        "positive_count": fixture_metadata.loc[fixture_ids, "positive_count"].to_numpy(dtype=float),
        "home_rows": fixture_metadata.loc[fixture_ids, "home_rows"].to_numpy(dtype=float),
        "away_rows": fixture_metadata.loc[fixture_ids, "away_rows"].to_numpy(dtype=float),
        "player_rows": aligned_players.to_numpy(dtype=float),
    }


def score_bucket(
    fixture_ids: tuple[int, ...],
    score_inputs: dict[str, Any],
    reference_totals: dict[str, Any],
    target_share: float,
    score_weights: dict[str, float],
) -> float:
    if not fixture_ids:
        return math.inf

    positions = [score_inputs["fixture_to_position"][fixture_id] for fixture_id in fixture_ids]
    rows = float(score_inputs["row_count"][positions].sum())
    positives = float(score_inputs["positive_count"][positions].sum())
    home_rows = float(score_inputs["home_rows"][positions].sum())
    away_rows = float(score_inputs["away_rows"][positions].sum())
    target_rate = positives / rows if rows > 0 else 0.0

    player_rows = score_inputs["player_rows"][positions].sum(axis=0)
    total_player_rows = reference_totals["player_rows"]
    player_share = np.divide(
        player_rows,
        total_player_rows,
        out=np.zeros_like(player_rows, dtype=float),
        where=total_player_rows > 0,
    )
    player_weighted_mae = float(
        np.average(np.abs(player_share - target_share), weights=total_player_rows)
    )

    score = 0.0
    score += score_weights["row_share"] * abs((rows / reference_totals["rows"]) - target_share)
    score += score_weights["target_rate"] * abs(target_rate - reference_totals["target_rate"])
    score += score_weights["home_share"] * abs((home_rows / reference_totals["home_rows"]) - target_share)
    score += score_weights["away_share"] * abs((away_rows / reference_totals["away_rows"]) - target_share)
    score += score_weights["player_wmae"] * player_weighted_mae
    return float(score)


def total_assignment_score(
    buckets: dict[str, tuple[int, ...]],
    score_inputs: dict[str, Any],
    reference_totals: dict[str, Any],
    target_share_by_bucket: dict[str, float],
    score_weights: dict[str, float],
) -> float:
    score = 0.0
    for bucket_name, fixture_ids in buckets.items():
        score += score_bucket(
            fixture_ids=fixture_ids,
            score_inputs=score_inputs,
            reference_totals=reference_totals,
            target_share=target_share_by_bucket[bucket_name],
            score_weights=score_weights,
        )
    return float(score)


def optimize_holdout(
    fixture_metadata: pd.DataFrame,
    fixture_player_matrix: pd.DataFrame,
    config: SplitConfig,
) -> set[int]:
    score_inputs = build_score_inputs(fixture_metadata, fixture_player_matrix)
    fixture_ids = score_inputs["fixture_ids"]
    holdout_size = int(round(len(fixture_ids) * config.holdout_fixture_fraction))
    target_share = holdout_size / len(fixture_ids)
    reference_totals = {
        "rows": float(score_inputs["row_count"].sum()),
        "target_rate": float(score_inputs["positive_count"].sum() / score_inputs["row_count"].sum()),
        "home_rows": float(score_inputs["home_rows"].sum()),
        "away_rows": float(score_inputs["away_rows"].sum()),
        "player_rows": score_inputs["player_rows"].sum(axis=0),
    }
    score_weights = {
        "row_share": 3.0,
        "target_rate": 40.0,
        "home_share": 2.0,
        "away_share": 2.0,
        "player_wmae": 12.0,
    }

    rng = random.Random(config.random_seed)
    best_score: float | None = None
    best_holdout: tuple[int, ...] | None = None

    for _ in range(config.holdout_restarts):
        holdout = tuple(sorted(rng.sample(fixture_ids, holdout_size)))
        development = tuple(sorted(set(fixture_ids) - set(holdout)))
        current_score = total_assignment_score(
            buckets={"development": development, "holdout": holdout},
            score_inputs=score_inputs,
            reference_totals=reference_totals,
            target_share_by_bucket={"development": 1.0 - target_share, "holdout": target_share},
            score_weights=score_weights,
        )

        improved = True
        while improved:
            improved = False
            dev_set = set(development)
            holdout_set = set(holdout)
            for holdout_fixture in holdout:
                for dev_fixture in development:
                    trial_holdout = tuple(sorted((holdout_set - {holdout_fixture}) | {dev_fixture}))
                    trial_development = tuple(sorted((dev_set - {dev_fixture}) | {holdout_fixture}))
                    trial_score = total_assignment_score(
                        buckets={"development": trial_development, "holdout": trial_holdout},
                        score_inputs=score_inputs,
                        reference_totals=reference_totals,
                        target_share_by_bucket={"development": 1.0 - target_share, "holdout": target_share},
                        score_weights=score_weights,
                    )
                    if trial_score + 1e-12 < current_score:
                        holdout = trial_holdout
                        development = trial_development
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break

        if best_score is None or current_score < best_score:
            best_score = current_score
            best_holdout = holdout

    if best_holdout is None:
        raise RuntimeError("Failed to optimize holdout split.")

    return set(best_holdout)


def optimize_dev_folds(
    dev_fixture_metadata: pd.DataFrame,
    dev_fixture_player_matrix: pd.DataFrame,
    config: SplitConfig,
) -> dict[str, set[int]]:
    score_inputs = build_score_inputs(dev_fixture_metadata, dev_fixture_player_matrix)
    fixture_ids = score_inputs["fixture_ids"]
    fold_size = len(fixture_ids) // config.n_dev_folds
    reference_totals = {
        "rows": float(score_inputs["row_count"].sum()),
        "target_rate": float(score_inputs["positive_count"].sum() / score_inputs["row_count"].sum()),
        "home_rows": float(score_inputs["home_rows"].sum()),
        "away_rows": float(score_inputs["away_rows"].sum()),
        "player_rows": score_inputs["player_rows"].sum(axis=0),
    }
    score_weights = {
        "row_share": 3.0,
        "target_rate": 80.0,
        "home_share": 2.0,
        "away_share": 2.0,
        "player_wmae": 12.0,
    }

    best_score: float | None = None
    best_folds: dict[str, tuple[int, ...]] | None = None

    for seed in range(config.fold_restarts):
        rng = random.Random(config.random_seed + seed)
        shuffled = fixture_ids[:]
        rng.shuffle(shuffled)
        fold_names = [f"fold_{idx}" for idx in range(config.n_dev_folds)]
        folds = {
            fold_name: tuple(sorted(shuffled[idx * fold_size : (idx + 1) * fold_size]))
            for idx, fold_name in enumerate(fold_names)
        }
        current_score = total_assignment_score(
            buckets=folds,
            score_inputs=score_inputs,
            reference_totals=reference_totals,
            target_share_by_bucket={fold_name: 1.0 / config.n_dev_folds for fold_name in fold_names},
            score_weights=score_weights,
        )

        improved = True
        while improved:
            improved = False
            for left_idx, left_name in enumerate(fold_names):
                for right_name in fold_names[left_idx + 1 :]:
                    for left_fixture in folds[left_name]:
                        for right_fixture in folds[right_name]:
                            left_trial = tuple(
                                sorted((set(folds[left_name]) - {left_fixture}) | {right_fixture})
                            )
                            right_trial = tuple(
                                sorted((set(folds[right_name]) - {right_fixture}) | {left_fixture})
                            )
                            trial_folds = dict(folds)
                            trial_folds[left_name] = left_trial
                            trial_folds[right_name] = right_trial
                            trial_score = total_assignment_score(
                                buckets=trial_folds,
                                score_inputs=score_inputs,
                                reference_totals=reference_totals,
                                target_share_by_bucket={
                                    fold_name: 1.0 / config.n_dev_folds for fold_name in fold_names
                                },
                                score_weights=score_weights,
                            )
                            if trial_score + 1e-12 < current_score:
                                folds = trial_folds
                                current_score = trial_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if best_score is None or current_score < best_score:
            best_score = current_score
            best_folds = folds

    if best_folds is None:
        raise RuntimeError("Failed to optimize development folds.")

    return {fold_name: set(fixture_ids) for fold_name, fixture_ids in best_folds.items()}


def build_bucket_diagnostics(
    base: pd.DataFrame,
    fixture_player_matrix: pd.DataFrame,
    buckets: dict[str, set[int]],
    expected_share_by_bucket: dict[str, float],
    bucket_type: str,
) -> pd.DataFrame:
    all_rows = float(len(base))
    all_positive = float(base["scored_after"].sum())
    all_target_rate = all_positive / all_rows
    all_players = int(base["player_id"].nunique())
    all_home_rows = float(base["is_home"].fillna(False).astype(bool).sum())
    all_away_rows = float((~base["is_home"].fillna(False).astype(bool)).sum())
    all_player_rows = fixture_player_matrix.sum(axis=0).to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for bucket_name, fixture_ids in buckets.items():
        subset = base[base["fixture_id"].isin(fixture_ids)].copy()
        player_rows = fixture_player_matrix.loc[sorted(fixture_ids)].sum(axis=0).to_numpy(dtype=float)
        player_share = np.divide(
            player_rows,
            all_player_rows,
            out=np.zeros_like(player_rows, dtype=float),
            where=all_player_rows > 0,
        )
        expected_share = expected_share_by_bucket[bucket_name]

        rows.append(
            {
                "bucket_type": bucket_type,
                "bucket": bucket_name,
                "fixture_count": int(len(fixture_ids)),
                "row_count": int(len(subset)),
                "row_share": float(len(subset) / all_rows),
                "row_share_abs_dev": float(abs((len(subset) / all_rows) - expected_share)),
                "positive_count": int(subset["scored_after"].sum()),
                "target_rate": float(subset["scored_after"].mean()),
                "target_rate_abs_dev": float(abs(subset["scored_after"].mean() - all_target_rate)),
                "unique_players": int(subset["player_id"].nunique()),
                "unique_player_share": float(subset["player_id"].nunique() / all_players),
                "home_rows": int(subset["is_home"].fillna(False).astype(bool).sum()),
                "away_rows": int((~subset["is_home"].fillna(False).astype(bool)).sum()),
                "home_share_abs_dev": float(
                    abs((subset["is_home"].fillna(False).astype(bool).sum() / all_home_rows) - expected_share)
                ),
                "away_share_abs_dev": float(
                    abs(((~subset["is_home"].fillna(False).astype(bool)).sum() / all_away_rows) - expected_share)
                ),
                "player_row_share_weighted_mae": float(
                    np.average(np.abs(player_share - expected_share), weights=all_player_rows)
                ),
                "player_row_share_max_abs_dev": float(np.max(np.abs(player_share - expected_share))),
            }
        )

    return pd.DataFrame(rows).sort_values(["bucket_type", "bucket"]).reset_index(drop=True)


def evaluate_thresholds(diagnostics: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    threshold_map = {
        "holdout": {
            "row_share_abs_dev": config.holdout_row_share_tolerance,
            "target_rate_abs_dev": config.holdout_target_rate_tolerance,
            "home_share_abs_dev": config.holdout_home_away_share_tolerance,
            "away_share_abs_dev": config.holdout_home_away_share_tolerance,
            "player_row_share_weighted_mae": config.holdout_player_wmae_tolerance,
        },
        "cv": {
            "row_share_abs_dev": config.cv_row_share_tolerance,
            "target_rate_abs_dev": config.cv_target_rate_tolerance,
            "home_share_abs_dev": config.cv_home_away_share_tolerance,
            "away_share_abs_dev": config.cv_home_away_share_tolerance,
            "player_row_share_weighted_mae": config.cv_player_wmae_tolerance,
        },
    }

    for bucket_type, metrics in threshold_map.items():
        subset = diagnostics[diagnostics["bucket_type"] == bucket_type]
        for metric_name, threshold in metrics.items():
            observed = float(subset[metric_name].max())
            rows.append(
                {
                    "bucket_type": bucket_type,
                    "metric": metric_name,
                    "threshold": threshold,
                    "observed_max": observed,
                    "passes": observed <= threshold + 1e-12,
                }
            )

    return pd.DataFrame(rows)


def build_split_outputs(
    base: pd.DataFrame,
    holdout_fixtures: set[int],
    dev_folds: dict[str, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixture_rows: list[dict[str, Any]] = []
    for fixture_id in sorted(base["fixture_id"].unique()):
        if fixture_id in holdout_fixtures:
            fold = "holdout"
            partition_role = "holdout"
        else:
            fold = next(fold_name for fold_name, ids in dev_folds.items() if fixture_id in ids)
            partition_role = "development"
        fixture_rows.append(
            {
                "fixture_id": int(fixture_id),
                "fold": fold,
                "partition_role": partition_role,
            }
        )

    fixture_assignments = pd.DataFrame(fixture_rows)
    row_assignments = base.merge(fixture_assignments, on="fixture_id", how="left", validate="many_to_one")
    return fixture_assignments, row_assignments


def write_split_balance_report(
    report_path: Path,
    fixture_assignments: pd.DataFrame,
    diagnostics: pd.DataFrame,
    threshold_checks: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Split Balance Diagnostics Report")
    lines.append("")
    lines.append("This report was generated by `src/python/split_strategy.py`.")
    lines.append("")

    holdout_fixtures = fixture_assignments.loc[
        fixture_assignments["partition_role"] == "holdout", "fixture_id"
    ].tolist()
    lines.append("## Final Holdout Fixtures")
    lines.append("")
    lines.append(", ".join(str(value) for value in holdout_fixtures))
    lines.append("")

    lines.append("## Development Fold Fixtures")
    lines.append("")
    for fold_name, subset in fixture_assignments[fixture_assignments["partition_role"] == "development"].groupby("fold"):
        fixture_text = ", ".join(str(value) for value in subset["fixture_id"].tolist())
        lines.append(f"- `{fold_name}`: {fixture_text}")
    lines.append("")

    lines.append("## Bucket Diagnostics")
    lines.append("")
    lines.append(
        "| Type | Bucket | Fixtures | Rows | Row Share | Target Rate | Unique Players | "
        "Home Dev | Away Dev | Player Row Share WMAE |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in diagnostics.iterrows():
        lines.append(
            "| "
            + f"{row['bucket_type']} | {row['bucket']} | {int(row['fixture_count'])} | {int(row['row_count'])} | "
            + f"{float(row['row_share']):.4f} | {float(row['target_rate']):.4f} | {int(row['unique_players'])} | "
            + f"{float(row['home_share_abs_dev']):.4f} | {float(row['away_share_abs_dev']):.4f} | "
            + f"{float(row['player_row_share_weighted_mae']):.4f} |"
        )

    lines.append("")
    lines.append("## Threshold Checks")
    lines.append("")
    lines.append("| Type | Metric | Threshold | Observed Max | Passes |")
    lines.append("|---|---|---:|---:|---|")
    for _, row in threshold_checks.iterrows():
        lines.append(
            "| "
            + f"{row['bucket_type']} | {row['metric']} | {float(row['threshold']):.4f} | "
            + f"{float(row['observed_max']):.4f} | {bool(row['passes'])} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_evaluation_protocol(
    protocol_path: Path,
    config: SplitConfig,
    fixture_assignments: pd.DataFrame,
) -> None:
    holdout_count = int((fixture_assignments["partition_role"] == "holdout").sum())
    dev_count = int((fixture_assignments["partition_role"] == "development").sum())

    lines: list[str] = []
    lines.append("# Evaluation Protocol Note")
    lines.append("")
    lines.append("This note was generated by `src/python/split_strategy.py`.")
    lines.append("")
    lines.append("## Frozen Split Strategy")
    lines.append("")
    lines.append(f"- Fixture-grouped final holdout size: {holdout_count} fixtures.")
    lines.append(f"- Development size: {dev_count} fixtures.")
    lines.append(f"- Development CV: {config.n_dev_folds}-fold grouped cross-validation by `fixture_id`.")
    lines.append("- Split artifacts are fixed on disk and must be reused across all experiments.")
    lines.append("- Holdout fixtures remain untouched until final model freeze.")
    lines.append("- If explicit team IDs are unavailable, team balance is monitored via the documented `fixture_id + is_home` proxy together with player-composition diagnostics.")
    lines.append("")
    lines.append("## Metric Policy")
    lines.append("")
    lines.append("- Primary model-selection metric: mean development-fold Balanced Accuracy.")
    lines.append("- Secondary metrics reported for every experiment: ROC AUC, PR AUC, and Brier score.")
    lines.append("- Fold-wise metrics must be archived to support uncertainty intervals and paired ablation comparisons.")
    lines.append("")
    lines.append("## Fixed Vs Tunable")
    lines.append("")
    lines.append("- Fixed: split assignments, random seed, grouped evaluation protocol, no holdout access during development, and fold-contained preprocessing/feature engineering.")
    lines.append("- Tunable later: model family, regularization/class-weight settings, tree depth/learning-rate style parameters, and approved feature-set variants.")
    lines.append("- Forbidden: regrouping fixtures, peeking at holdout performance during tuning, or fitting preprocessing on combined train+validation data.")
    lines.append("")
    lines.append("## Leakage Guardrails")
    lines.append("")
    lines.append("- Every training fold must exclude validation fixtures completely.")
    lines.append("- All feature engineering, imputation, encoding, scaling, and calibration must run inside the active training fold only.")
    lines.append("- Temporal leakage findings from `docs/leakage_checklist.md` remain binding for downstream modeling tables.")

    protocol_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = SplitConfig()
    root = project_root()
    data_splits_dir = root / "data" / "splits"
    artifacts_dir = root / "artifacts" / "splits"
    docs_dir = root / "docs"

    data_splits_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_table()
    fixture_player_matrix = build_fixture_player_matrix(base)
    fixture_metadata = build_fixture_metadata(base).set_index("fixture_id", drop=True)

    holdout_fixtures = optimize_holdout(fixture_metadata, fixture_player_matrix, config)

    dev_base = base[~base["fixture_id"].isin(holdout_fixtures)].copy()
    dev_fixture_player_matrix = build_fixture_player_matrix(dev_base)
    dev_fixture_metadata = build_fixture_metadata(dev_base).set_index("fixture_id", drop=True)
    dev_folds = optimize_dev_folds(dev_fixture_metadata, dev_fixture_player_matrix, config)

    fixture_assignments, row_assignments = build_split_outputs(base, holdout_fixtures, dev_folds)

    holdout_diagnostics = build_bucket_diagnostics(
        base=base,
        fixture_player_matrix=fixture_player_matrix,
        buckets={
            "development": set(fixture_assignments.loc[fixture_assignments["partition_role"] == "development", "fixture_id"]),
            "holdout": holdout_fixtures,
        },
        expected_share_by_bucket={
            "development": 1.0 - (len(holdout_fixtures) / fixture_metadata.shape[0]),
            "holdout": len(holdout_fixtures) / fixture_metadata.shape[0],
        },
        bucket_type="holdout",
    )
    cv_diagnostics = build_bucket_diagnostics(
        base=dev_base,
        fixture_player_matrix=dev_fixture_player_matrix,
        buckets=dev_folds,
        expected_share_by_bucket={fold_name: 1.0 / config.n_dev_folds for fold_name in dev_folds},
        bucket_type="cv",
    )
    diagnostics = pd.concat([holdout_diagnostics, cv_diagnostics], ignore_index=True)
    threshold_checks = evaluate_thresholds(diagnostics, config)

    fixture_metadata.reset_index().to_csv(data_splits_dir / "fixture_metadata.csv", index=False)
    fixture_assignments.to_csv(data_splits_dir / "fixture_assignments.csv", index=False)
    row_assignments.to_csv(data_splits_dir / "modeling_row_folds.csv", index=False)
    diagnostics.to_csv(artifacts_dir / "split_balance_diagnostics.csv", index=False)
    threshold_checks.to_csv(artifacts_dir / "split_balance_threshold_checks.csv", index=False)
    (artifacts_dir / "split_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    write_split_balance_report(
        docs_dir / "split_balance_report.md",
        fixture_assignments=fixture_assignments,
        diagnostics=diagnostics,
        threshold_checks=threshold_checks,
    )
    write_evaluation_protocol(
        docs_dir / "evaluation_protocol.md",
        config=config,
        fixture_assignments=fixture_assignments,
    )


if __name__ == "__main__":
    main()
