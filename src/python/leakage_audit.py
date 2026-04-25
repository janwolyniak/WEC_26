from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_intake import DATASET_FILES, NA_TOKENS, normalize_dataframe, project_root


STATUS_ORDER = {"fail": 0, "pending": 1, "pass": 2}
PERIOD_OFFSETS = {"half_1": 0, "half_2": 45, "extra_time_1": 90, "extra_time_2": 105}
CHECKPOINT_TO_TIME = {
    "H1_15": ("half_1", 15),
    "H1_30": ("half_1", 30),
    "H1_45": ("half_1", 45),
    "H2_15": ("half_2", 15),
    "H2_30": ("half_2", 30),
    "H2_45": ("half_2", 45),
    "ET1_15": ("extra_time_1", 15),
}
TRANSFORM_TOKENS = ("rank", "pct", "percentile", "zscore", "z_score", "norm", "normalized", "standardized")
FLOAT_TOLERANCE = 0.01


@dataclass
class LeakageCheck:
    category: str
    check_name: str
    status: str
    severity: str
    issue_count: int
    total_rows: int
    details: str
    evidence: str
    resolution_action: str

    def to_record(self) -> dict[str, Any]:
        rate = 0.0
        if self.total_rows > 0:
            rate = (self.issue_count / self.total_rows) * 100.0

        return {
            "category": self.category,
            "check_name": self.check_name,
            "status": self.status,
            "severity": self.severity,
            "issue_count": self.issue_count,
            "total_rows": self.total_rows,
            "issue_rate_pct": round(rate, 4),
            "details": self.details,
            "evidence": self.evidence,
            "resolution_action": self.resolution_action,
        }


def load_datasets() -> dict[str, pd.DataFrame]:
    root = project_root()
    data_dir = root / "data"

    frames: dict[str, pd.DataFrame] = {}
    for filename in DATASET_FILES:
        path = data_dir / filename
        raw = pd.read_csv(path, keep_default_na=True, na_values=NA_TOKENS, low_memory=False)
        frames[path.stem] = normalize_dataframe(raw)

    return frames


def add_check(
    checks: list[LeakageCheck],
    category: str,
    check_name: str,
    status: str,
    severity: str,
    issue_count: int,
    total_rows: int,
    details: str,
    evidence: str,
    resolution_action: str,
) -> None:
    checks.append(
        LeakageCheck(
            category=category,
            check_name=check_name,
            status=status,
            severity=severity,
            issue_count=issue_count,
            total_rows=total_rows,
            details=details,
            evidence=evidence,
            resolution_action=resolution_action,
        )
    )


def add_checkpoint_minutes(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["checkpoint_cont_min"] = (
        out["checkpoint"].map(lambda value: PERIOD_OFFSETS[CHECKPOINT_TO_TIME[str(value)][0]])
        + out["checkpoint"].map(lambda value: CHECKPOINT_TO_TIME[str(value)][1])
    )
    return out


def add_event_minutes(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_cont_min"] = out["period"].map(PERIOD_OFFSETS).astype("Float64") + pd.to_numeric(
        out["minute"],
        errors="coerce",
    )
    return out


def feature_tolerance(feature: str) -> float:
    if "distance" in feature or "speed" in feature:
        return FLOAT_TOLERANCE
    return 0.0


def compare_feature_sets(
    base: pd.DataFrame,
    reconstructed: pd.DataFrame,
    features: list[str],
    source_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []

    for feature in features:
        tolerance = feature_tolerance(feature)
        base_values = pd.to_numeric(base[feature], errors="coerce").fillna(0.0)
        reconstructed_values = pd.to_numeric(reconstructed[feature], errors="coerce").fillna(0.0)
        abs_diff = (base_values - reconstructed_values).abs()
        mismatch_mask = abs_diff > tolerance

        summary_rows.append(
            {
                "source": source_name,
                "feature": feature,
                "tolerance": tolerance,
                "mismatch_rows": int(mismatch_mask.sum()),
                "max_abs_diff": round(float(abs_diff.max()), 6),
            }
        )

        mismatch_subset = base.loc[mismatch_mask, ["player_appearance_id", "checkpoint"]].copy()
        if mismatch_subset.empty:
            continue

        mismatch_subset["source"] = source_name
        mismatch_subset["feature"] = feature
        mismatch_subset["base_value"] = base_values[mismatch_mask].astype(float).values
        mismatch_subset["reconstructed_value"] = reconstructed_values[mismatch_mask].astype(float).values
        mismatch_subset["abs_diff"] = abs_diff[mismatch_mask].astype(float).values
        mismatch_rows.append(mismatch_subset)

    summary_df = pd.DataFrame(summary_rows)
    mismatch_df = pd.concat(mismatch_rows, ignore_index=True) if mismatch_rows else pd.DataFrame(
        columns=[
            "player_appearance_id",
            "checkpoint",
            "source",
            "feature",
            "base_value",
            "reconstructed_value",
            "abs_diff",
        ]
    )
    return summary_df, mismatch_df


def reconstruct_run_features(base: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    run_groups = {key: frame for key, frame in run.groupby("player_appearance_id", dropna=True)}
    rows: list[dict[str, Any]] = []

    for row in base[["player_appearance_id", "checkpoint_cont_min"]].itertuples(index=False):
        events = run_groups.get(row.player_appearance_id)
        if events is None:
            cumulative = pd.DataFrame(columns=run.columns)
        else:
            cumulative = events[events["event_cont_min"] <= row.checkpoint_cont_min]
        last15 = cumulative[cumulative["event_cont_min"] > row.checkpoint_cont_min - 15]

        rows.append(
            {
                "last15_sprints": int((last15["run_type"] == "sprint").sum()),
                "last15_hsr": int((last15["run_type"] == "hsr").sum()),
                "last15_distance": float(last15["distance"].sum()) if not last15.empty else 0.0,
                "last15_mean_max_speed": float(last15["max_speed"].mean()) if not last15.empty else 0.0,
                "last15_peak_speed": float(last15["max_speed"].max()) if not last15.empty else 0.0,
                "cumul_sprints": int((cumulative["run_type"] == "sprint").sum()),
                "cumul_hsr": int((cumulative["run_type"] == "hsr").sum()),
                "cumul_distance": float(cumulative["distance"].sum()) if not cumulative.empty else 0.0,
                "cumul_mean_max_speed": float(cumulative["max_speed"].mean()) if not cumulative.empty else 0.0,
                "cumul_peak_speed": float(cumulative["max_speed"].max()) if not cumulative.empty else 0.0,
            }
        )

    return pd.DataFrame(rows)


def reconstruct_shot_features(base: pd.DataFrame, shot: pd.DataFrame) -> pd.DataFrame:
    shot_groups = {key: frame for key, frame in shot.groupby("player_appearance_id", dropna=True)}
    rows: list[dict[str, Any]] = []

    for row in base[["player_appearance_id", "checkpoint_cont_min"]].itertuples(index=False):
        events = shot_groups.get(row.player_appearance_id)
        if events is None:
            cumulative = pd.DataFrame(columns=shot.columns)
        else:
            cumulative = events[events["event_cont_min"] <= row.checkpoint_cont_min]

        cumulative = cumulative[cumulative["own_goal_player_appearance_id"].isna()]
        last15 = cumulative[cumulative["event_cont_min"] > row.checkpoint_cont_min - 15]

        rows.append(
            {
                "last15_shots": int(len(last15)),
                "last15_shots_under_press": int(last15["under_pressure"].fillna(False).astype(bool).sum()),
                "last15_shots_top_third": int((last15["stage"] == "top").sum()),
                "cumul_shots": int(len(cumulative)),
                "cumul_shots_under_press": int(cumulative["under_pressure"].fillna(False).astype(bool).sum()),
                "cumul_shots_top_third": int((cumulative["stage"] == "top").sum()),
            }
        )

    return pd.DataFrame(rows)


def run_temporal_checks(
    datasets: dict[str, pd.DataFrame],
    checks: list[LeakageCheck],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = add_checkpoint_minutes(datasets["players_quarters_final"])
    run = add_event_minutes(datasets["player_appearance_run"])
    shot = add_event_minutes(datasets["player_appearance_shot_limited"])

    column_risk_rows: list[dict[str, Any]] = []
    feature_summaries: list[pd.DataFrame] = []
    feature_mismatches: list[pd.DataFrame] = []

    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="shot_outcome_absent_from_contest_tables",
        status="pass" if "outcome" not in shot.columns and "outcome" not in base.columns else "fail",
        severity="critical",
        issue_count=int(("outcome" in shot.columns) or ("outcome" in base.columns)),
        total_rows=len(base),
        details="Audit that the redacted shot outcome field is absent from contest modeling inputs.",
        evidence="No `outcome` column is present in `player_appearance_shot_limited` or `players_quarters_final`.",
        resolution_action="Keep outcome-like labels excluded from any modeling table.",
    )

    minute_in_violations = int((pd.to_numeric(base["minute_in"], errors="coerce") > base["checkpoint_cont_min"]).sum())
    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="minute_in_is_historical_at_checkpoint",
        status="fail" if minute_in_violations > 0 else "pass",
        severity="critical",
        issue_count=minute_in_violations,
        total_rows=len(base),
        details="Entry time must be known by the checkpoint for the column to be leakage-safe.",
        evidence=f"{minute_in_violations} rows have `minute_in` after the checkpoint minute.",
        resolution_action="Exclude malformed rows or repair entry-time encoding before modeling.",
    )

    future_minute_out = (pd.to_numeric(base["minute_out"], errors="coerce") > base["checkpoint_cont_min"]).fillna(False)
    future_minute_out_count = int(future_minute_out.sum())
    column_risk_rows.append(
        {
            "column": "minute_out",
            "issue_count": future_minute_out_count,
            "total_rows": len(base),
            "details": "Exit time is after the checkpoint in most rows, so it contains future match information.",
            "recommended_action": "Drop `minute_out` from any leakage-safe modeling feature set.",
        }
    )
    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="minute_out_contains_future_exit_time",
        status="fail" if future_minute_out_count > 0 else "pass",
        severity="critical",
        issue_count=future_minute_out_count,
        total_rows=len(base),
        details="Exit time is future information whenever the player remains on the pitch beyond the checkpoint.",
        evidence=f"{future_minute_out_count} of {len(base)} rows have `minute_out` later than the checkpoint.",
        resolution_action="Do not use `minute_out` directly; replace it with leakage-safe status variables if needed.",
    )

    future_subbed = (
        base["subbed"].fillna(False).astype(bool)
        & (pd.to_numeric(base["minute_out"], errors="coerce") > base["checkpoint_cont_min"]).fillna(False)
    )
    future_subbed_count = int(future_subbed.sum())
    column_risk_rows.append(
        {
            "column": "subbed",
            "issue_count": future_subbed_count,
            "total_rows": len(base),
            "details": "The flag identifies later substitutions-off, which is future information at earlier checkpoints.",
            "recommended_action": "Drop `subbed` or replace it with a pre-checkpoint availability/status feature.",
        }
    )
    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="subbed_contains_future_substitution_info",
        status="fail" if future_subbed_count > 0 else "pass",
        severity="critical",
        issue_count=future_subbed_count,
        total_rows=len(base),
        details="The current `subbed` flag is not checkpoint-causal because it marks later substitutions.",
        evidence=f"{future_subbed_count} rows have `subbed=True` while the player leaves after the checkpoint.",
        resolution_action="Exclude `subbed` from modeling unless it is re-derived using only pre-checkpoint information.",
    )

    suspicious_columns = [
        column
        for column in base.columns
        if column != "scored_after" and any(token in column.lower() for token in TRANSFORM_TOKENS)
    ]
    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="forbidden_normalization_columns_absent",
        status="pass" if not suspicious_columns else "pending",
        severity="major",
        issue_count=len(suspicious_columns),
        total_rows=len(base.columns),
        details="Scan for columns whose names suggest ranks, percentiles, z-scores, or normalizations.",
        evidence=(
            "No suspicious transformation columns were found."
            if not suspicious_columns
            else f"Columns requiring provenance review: {', '.join(sorted(suspicious_columns))}."
        ),
        resolution_action="Require explicit per-checkpoint provenance for any future normalization/ranking features.",
    )

    run_features = [
        "last15_sprints",
        "last15_hsr",
        "last15_distance",
        "last15_mean_max_speed",
        "last15_peak_speed",
        "cumul_sprints",
        "cumul_hsr",
        "cumul_distance",
        "cumul_mean_max_speed",
        "cumul_peak_speed",
    ]
    shot_features = [
        "last15_shots",
        "last15_shots_under_press",
        "last15_shots_top_third",
        "cumul_shots",
        "cumul_shots_under_press",
        "cumul_shots_top_third",
    ]

    run_reconstructed = reconstruct_run_features(base, run)
    shot_reconstructed = reconstruct_shot_features(base, shot)
    run_summary, run_mismatches = compare_feature_sets(base, run_reconstructed, run_features, "run")
    shot_summary, shot_mismatches = compare_feature_sets(base, shot_reconstructed, shot_features, "shot")
    feature_summaries.extend([run_summary, shot_summary])
    feature_mismatches.extend([run_mismatches, shot_mismatches])

    run_mismatch_rows = int(run_summary["mismatch_rows"].sum())
    shot_mismatch_rows = int(shot_summary["mismatch_rows"].sum())

    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="run_features_match_checkpoint_cutoff_reconstruction",
        status="fail" if run_mismatch_rows > 0 else "pass",
        severity="critical",
        issue_count=run_mismatch_rows,
        total_rows=len(base) * len(run_features),
        details="Reconstruct all run-derived features using only events at or before the checkpoint.",
        evidence=(
            "All 10 run-derived features matched the causal reconstruction within rounding tolerance."
            if run_mismatch_rows == 0
            else f"{run_mismatch_rows} run-feature cells differ from the causal reconstruction."
        ),
        resolution_action="If mismatches appear, rebuild run aggregates strictly from pre-checkpoint events.",
    )

    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="shot_features_match_checkpoint_cutoff_reconstruction",
        status="fail" if shot_mismatch_rows > 0 else "pass",
        severity="critical",
        issue_count=shot_mismatch_rows,
        total_rows=len(base) * len(shot_features),
        details="Reconstruct all auditable shot-derived features using only pre-checkpoint shot events.",
        evidence=(
            "All 6 auditable shot-derived features matched the causal reconstruction exactly."
            if shot_mismatch_rows == 0
            else f"{shot_mismatch_rows} shot-feature cells differ from the causal reconstruction."
        ),
        resolution_action="If mismatches appear, rebuild shot aggregates with a strict checkpoint cutoff.",
    )

    add_check(
        checks=checks,
        category="within_match_temporal",
        check_name="shots_on_target_not_independently_reconstructable_from_redacted_shot_table",
        status="pending",
        severity="major",
        issue_count=2,
        total_rows=2,
        details="`last15_shots_on_target` and `cumul_shots_on_target` depend on shot outcome, which is redacted.",
        evidence="The contest shot table omits `outcome`, so the two on-target aggregates cannot be independently rebuilt.",
        resolution_action="Treat these columns as trusted organizer aggregates unless non-redacted provenance is provided.",
    )

    feature_summary_df = pd.concat(feature_summaries, ignore_index=True)
    feature_mismatch_df = pd.concat(feature_mismatches, ignore_index=True)
    column_risk_df = pd.DataFrame(column_risk_rows)
    return feature_summary_df, feature_mismatch_df, column_risk_df


def load_split_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported split artifact format: {path.suffix}")


def run_fold_checks(root: Path, checks: list[LeakageCheck]) -> pd.DataFrame:
    diagnostics: list[dict[str, Any]] = []
    split_paths: list[Path] = []
    for directory in [root / "data" / "splits", root / "artifacts" / "splits"]:
        if directory.exists():
            split_paths.extend(sorted(directory.glob("*.csv")))
            split_paths.extend(sorted(directory.glob("*.parquet")))

    if not split_paths:
        add_check(
            checks=checks,
            category="fold_leakage",
            check_name="split_artifacts_available_for_overlap_audit",
            status="pending",
            severity="major",
            issue_count=1,
            total_rows=1,
            details="Fold leakage audit requires persisted split assignments.",
            evidence="No split artifacts were found under `data/splits` or `artifacts/splits`.",
            resolution_action="Save split assignments, then rerun the leakage audit before modeling.",
        )
        return pd.DataFrame(
            columns=["artifact", "split_column", "key_overlap_rows", "fixture_overlap_rows", "status"]
        )

    for path in split_paths:
        frame = load_split_frame(path)
        split_column = next((col for col in ["split", "fold", "set", "partition"] if col in frame.columns), None)
        has_key_cols = {"player_appearance_id", "checkpoint"}.issubset(frame.columns)

        if split_column is None or not has_key_cols:
            diagnostics.append(
                {
                    "artifact": str(path.relative_to(root)),
                    "split_column": split_column or "",
                    "key_overlap_rows": pd.NA,
                    "fixture_overlap_rows": pd.NA,
                    "status": "pending",
                }
            )
            add_check(
                checks=checks,
                category="fold_leakage",
                check_name=f"split_schema_supported_{path.name}",
                status="pending",
                severity="major",
                issue_count=1,
                total_rows=1,
                details="Split artifact exists but cannot be audited without key columns and a split label column.",
                evidence=f"Columns found: {', '.join(frame.columns.astype(str))}.",
                resolution_action="Store `player_appearance_id`, `checkpoint`, and one of `split`/`fold`/`set`/`partition`.",
            )
            continue

        key_overlap_rows = int(
            frame.groupby(["player_appearance_id", "checkpoint"], dropna=False)[split_column].nunique().gt(1).sum()
        )
        fixture_overlap_rows = pd.NA
        if "fixture_id" in frame.columns:
            fixture_overlap_rows = int(frame.groupby("fixture_id", dropna=False)[split_column].nunique().gt(1).sum())

        diagnostics.append(
            {
                "artifact": str(path.relative_to(root)),
                "split_column": split_column,
                "key_overlap_rows": key_overlap_rows,
                "fixture_overlap_rows": fixture_overlap_rows,
                "status": "fail" if key_overlap_rows > 0 else "pass",
            }
        )
        add_check(
            checks=checks,
            category="fold_leakage",
            check_name=f"player_checkpoint_overlap_{path.name}",
            status="fail" if key_overlap_rows > 0 else "pass",
            severity="critical",
            issue_count=key_overlap_rows,
            total_rows=len(frame),
            details="No player-match checkpoint may appear in more than one split/fold assignment.",
            evidence=(
                f"Artifact `{path.relative_to(root)}` has {key_overlap_rows} overlapping player-checkpoint keys."
            ),
            resolution_action="Regenerate split assignments so each player-checkpoint key appears in exactly one split.",
        )

    return pd.DataFrame(diagnostics)


def write_leakage_checklist(report_path: Path, findings_df: pd.DataFrame) -> None:
    failed = findings_df[findings_df["status"] == "fail"].copy()
    pending = findings_df[findings_df["status"] == "pending"].copy()
    signed_off = failed.empty and pending.empty

    lines: list[str] = []
    lines.append("# Leakage Checklist")
    lines.append("")
    lines.append("This checklist was generated by `src/python/leakage_audit.py`.")
    lines.append("")
    lines.append(f"## Sign-off Status: {'signed off' if signed_off else 'not signed off'}")
    lines.append("")

    if signed_off:
        lines.append("All implemented leakage checks passed.")
    else:
        if not failed.empty:
            lines.append("### Blocking Failures")
            lines.append("")
            for _, row in failed.iterrows():
                lines.append(f"- `{row['check_name']}`: {row['evidence']}")
            lines.append("")
        if not pending.empty:
            lines.append("### Pending Items")
            lines.append("")
            for _, row in pending.iterrows():
                lines.append(f"- `{row['check_name']}`: {row['evidence']}")
            lines.append("")

    lines.append("## Checklist")
    lines.append("")
    lines.append("| Category | Check | Status | Severity | Issues | Evidence | Action |")
    lines.append("|---|---|---|---|---:|---|---|")
    for _, row in findings_df.iterrows():
        lines.append(
            "| "
            + f"{row['category']} | {row['check_name']} | {row['status']} | {row['severity']} | "
            + f"{int(row['issue_count'])} | {row['evidence']} | {row['resolution_action']} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_temporal_report(
    report_path: Path,
    findings_df: pd.DataFrame,
    feature_summary_df: pd.DataFrame,
    column_risk_df: pd.DataFrame,
    split_diagnostics_df: pd.DataFrame,
) -> None:
    temporal_findings = findings_df[findings_df["category"] == "within_match_temporal"].copy()
    fold_findings = findings_df[findings_df["category"] == "fold_leakage"].copy()

    lines: list[str] = []
    lines.append("# Temporal Leakage Diagnostics Report")
    lines.append("")
    lines.append("This report was generated by `src/python/leakage_audit.py`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Leakage classes covered: within-match temporal leakage, fold leakage")
    lines.append("- Event families reconciled: run-derived features and auditable shot-derived features")
    lines.append("- Tolerance for floating-point reconciliation: 0.01")
    lines.append("")
    lines.append("## Temporal Findings")
    lines.append("")
    lines.append("| Check | Status | Issues | Evidence |")
    lines.append("|---|---|---:|---|")
    for _, row in temporal_findings.iterrows():
        lines.append(
            "| "
            + f"{row['check_name']} | {row['status']} | {int(row['issue_count'])} | {row['evidence']} |"
        )

    lines.append("")
    lines.append("## Base Columns With Future Information")
    lines.append("")
    lines.append("| Column | Issues | Details | Recommended Action |")
    lines.append("|---|---:|---|---|")
    for _, row in column_risk_df.iterrows():
        lines.append(
            "| "
            + f"{row['column']} | {int(row['issue_count'])} | {row['details']} | {row['recommended_action']} |"
        )

    lines.append("")
    lines.append("## Event Feature Reconciliation")
    lines.append("")
    lines.append("| Source | Feature | Mismatch Rows | Max Abs Diff |")
    lines.append("|---|---|---:|---:|")
    for _, row in feature_summary_df.iterrows():
        lines.append(
            "| "
            + f"{row['source']} | {row['feature']} | {int(row['mismatch_rows'])} | {float(row['max_abs_diff']):.6f} |"
        )

    lines.append("")
    lines.append("## Fold Leakage Status")
    lines.append("")
    if split_diagnostics_df.empty:
        lines.append("Split overlap checks are pending because no split artifacts were found.")
    else:
        lines.append("| Artifact | Split Column | Key Overlap Rows | Fixture Overlap Rows | Status |")
        lines.append("|---|---|---:|---:|---|")
        for _, row in split_diagnostics_df.iterrows():
            fixture_value = "" if pd.isna(row["fixture_overlap_rows"]) else int(row["fixture_overlap_rows"])
            key_value = "" if pd.isna(row["key_overlap_rows"]) else int(row["key_overlap_rows"])
            lines.append(
                "| "
                + f"{row['artifact']} | {row['split_column']} | {key_value} | {fixture_value} | {row['status']} |"
            )

    lines.append("")
    lines.append("## Fold Findings")
    lines.append("")
    lines.append("| Check | Status | Issues | Evidence |")
    lines.append("|---|---|---:|---|")
    for _, row in fold_findings.iterrows():
        lines.append(
            "| "
            + f"{row['check_name']} | {row['status']} | {int(row['issue_count'])} | {row['evidence']} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = project_root()
    docs_dir = root / "docs"
    out_dir = root / "artifacts" / "leakage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets()
    checks: list[LeakageCheck] = []

    feature_summary_df, feature_mismatch_df, column_risk_df = run_temporal_checks(datasets, checks)
    split_diagnostics_df = run_fold_checks(root, checks)

    findings_df = pd.DataFrame([check.to_record() for check in checks])
    findings_df["status_rank"] = findings_df["status"].map(STATUS_ORDER)
    findings_df = findings_df.sort_values(
        ["status_rank", "category", "severity", "issue_count", "check_name"],
        ascending=[True, True, True, False, True],
    ).drop(columns=["status_rank"]).reset_index(drop=True)

    findings_df.to_csv(out_dir / "findings.csv", index=False)
    feature_summary_df.to_csv(out_dir / "feature_reconciliation_summary.csv", index=False)
    feature_mismatch_df.to_csv(out_dir / "feature_reconciliation_mismatches.csv", index=False)
    column_risk_df.to_csv(out_dir / "base_column_risks.csv", index=False)
    split_diagnostics_df.to_csv(out_dir / "split_overlap_diagnostics.csv", index=False)

    write_leakage_checklist(docs_dir / "leakage_checklist.md", findings_df)
    write_temporal_report(
        docs_dir / "temporal_leakage_report.md",
        findings_df,
        feature_summary_df,
        column_risk_df,
        split_diagnostics_df,
    )


if __name__ == "__main__":
    main()
