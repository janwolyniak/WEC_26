from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_intake import DATASET_FILES, NA_TOKENS, normalize_dataframe, project_root


SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "info": 3}


@dataclass
class Finding:
    dataset: str
    check_name: str
    severity: str
    issue_count: int
    total_rows: int
    details: str
    resolution_action: str

    def to_record(self) -> dict[str, Any]:
        rate = 0.0
        if self.total_rows > 0:
            rate = (self.issue_count / self.total_rows) * 100.0

        return {
            "dataset": self.dataset,
            "check_name": self.check_name,
            "severity": self.severity,
            "issue_count": self.issue_count,
            "total_rows": self.total_rows,
            "issue_rate_pct": round(rate, 4),
            "details": self.details,
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


def to_int_set(series: pd.Series) -> set[int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return set()
    return set(numeric.astype("Int64").astype(int).tolist())


def add_finding(
    findings: list[Finding],
    dataset: str,
    check_name: str,
    severity: str,
    issue_count: int,
    total_rows: int,
    details: str,
    resolution_action: str,
) -> None:
    findings.append(
        Finding(
            dataset=dataset,
            check_name=check_name,
            severity=severity,
            issue_count=issue_count,
            total_rows=total_rows,
            details=details,
            resolution_action=resolution_action,
        )
    )


def run_duplicate_checks(datasets: dict[str, pd.DataFrame], findings: list[Finding]) -> None:
    for name, df in datasets.items():
        row_count = len(df)
        duplicate_rows = int(df.duplicated().sum())

        add_finding(
            findings=findings,
            dataset=name,
            check_name="duplicate_rows",
            severity="major" if duplicate_rows > 0 else "info",
            issue_count=duplicate_rows,
            total_rows=row_count,
            details="Count of fully duplicated rows.",
            resolution_action=(
                "Drop exact duplicates after confirming they are not legitimate repeated observations."
                if duplicate_rows > 0
                else "No action required."
            ),
        )

        if "id" in df.columns:
            duplicate_ids = int(df["id"].dropna().duplicated().sum())
            add_finding(
                findings=findings,
                dataset=name,
                check_name="duplicate_primary_id",
                severity="critical" if duplicate_ids > 0 else "info",
                issue_count=duplicate_ids,
                total_rows=row_count,
                details="Duplicate values in `id` column (expected unique event identifier).",
                resolution_action=(
                    "Investigate source extraction; keep one record per id or repair key collisions before modeling."
                    if duplicate_ids > 0
                    else "No action required."
                ),
            )

    base = datasets["players_quarters_final"]
    composite_dup = int(base.duplicated(subset=["player_appearance_id", "checkpoint"]).sum())
    add_finding(
        findings=findings,
        dataset="players_quarters_final",
        check_name="duplicate_player_checkpoint_key",
        severity="critical" if composite_dup > 0 else "info",
        issue_count=composite_dup,
        total_rows=len(base),
        details="Duplicate rows for composite key (`player_appearance_id`, `checkpoint`).",
        resolution_action=(
            "Keep a single row per player/checkpoint after verifying source consistency."
            if composite_dup > 0
            else "No action required."
        ),
    )


def run_key_integrity_checks(
    datasets: dict[str, pd.DataFrame],
    findings: list[Finding],
) -> pd.DataFrame:
    base = datasets["players_quarters_final"]
    base_ids = to_int_set(base["player_appearance_id"])

    global_ids: set[int] = set()
    for df in datasets.values():
        for col in df.columns:
            if col.endswith("player_appearance_id"):
                global_ids.update(to_int_set(df[col]))

    cardinality_records: list[dict[str, Any]] = []

    for name, df in datasets.items():
        if "player_appearance_id" in df.columns:
            ids = pd.to_numeric(df["player_appearance_id"], errors="coerce")
            valid_ids = ids.dropna().astype("Int64").astype(int)

            unmatched_to_base = int((~valid_ids.isin(base_ids)).sum())
            severity = "major" if unmatched_to_base > 0 else "info"
            add_finding(
                findings=findings,
                dataset=name,
                check_name="join_key_coverage_vs_players_quarters_final",
                severity=severity,
                issue_count=unmatched_to_base,
                total_rows=len(df),
                details=(
                    "Rows with `player_appearance_id` not present in players_quarters_final. "
                    "May happen for appearances with no eligible checkpoint rows."
                ),
                resolution_action=(
                    "Inspect missing appearance IDs and define handling policy (drop, map, or keep for event-only analyses)."
                    if unmatched_to_base > 0
                    else "No action required."
                ),
            )

            counts = df.groupby("player_appearance_id", dropna=True).size()
            if not counts.empty:
                cardinality_records.append(
                    {
                        "dataset": name,
                        "linked_key": "player_appearance_id",
                        "unique_keys": int(counts.shape[0]),
                        "mean_rows_per_key": round(float(counts.mean()), 4),
                        "median_rows_per_key": round(float(counts.median()), 4),
                        "p90_rows_per_key": round(float(counts.quantile(0.9)), 4),
                        "max_rows_per_key": int(counts.max()),
                    }
                )

        fk_columns = [
            col
            for col in df.columns
            if col.endswith("player_appearance_id") and col != "player_appearance_id"
        ]

        for fk_col in fk_columns:
            numeric_fk = pd.to_numeric(df[fk_col], errors="coerce")
            non_null_fk = numeric_fk.dropna().astype("Int64").astype(int)
            invalid_fk = int((~non_null_fk.isin(global_ids)).sum())
            severity = "critical" if invalid_fk > 0 else "info"
            add_finding(
                findings=findings,
                dataset=name,
                check_name=f"foreign_key_validity_{fk_col}",
                severity=severity,
                issue_count=invalid_fk,
                total_rows=len(df),
                details=(
                    f"Non-null `{fk_col}` values not present in global player_appearance_id universe "
                    "assembled from all tables."
                ),
                resolution_action=(
                    "Repair invalid foreign keys or set unresolved IDs to missing after source validation."
                    if invalid_fk > 0
                    else "No action required."
                ),
            )

    return pd.DataFrame(cardinality_records)


def range_violation_count(series: pd.Series, lower: float | None, upper: float | None) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=numeric.index)

    if lower is not None:
        mask = mask | (numeric < lower)
    if upper is not None:
        mask = mask | (numeric > upper)

    return int(mask.fillna(False).sum())


def run_range_and_consistency_checks(datasets: dict[str, pd.DataFrame], findings: list[Finding]) -> None:
    run = datasets["player_appearance_run"]

    checks = [
        ("min_speed_negative", run["min_speed"], 0.0, None, "Speed must be non-negative."),
        ("max_speed_negative", run["max_speed"], 0.0, None, "Speed must be non-negative."),
        ("max_speed_above_15_mps", run["max_speed"], None, 15.0, "Flag potentially implausible run speed values."),
        ("distance_negative", run["distance"], 0.0, None, "Run distance must be non-negative."),
        ("event_minute_outside_0_60", run["minute"], 0.0, 60.0, "Event minute should be within local period range."),
    ]

    for name, series, lower, upper, detail in checks:
        violations = range_violation_count(series, lower, upper)
        add_finding(
            findings=findings,
            dataset="player_appearance_run",
            check_name=name,
            severity="major" if violations > 0 else "info",
            issue_count=violations,
            total_rows=len(run),
            details=detail,
            resolution_action=(
                "Review source records and clip/drop impossible values after manual validation."
                if violations > 0
                else "No action required."
            ),
        )

    max_lt_min = int((pd.to_numeric(run["max_speed"], errors="coerce") < pd.to_numeric(run["min_speed"], errors="coerce")).sum())
    add_finding(
        findings=findings,
        dataset="player_appearance_run",
        check_name="max_speed_lower_than_min_speed",
        severity="major" if max_lt_min > 0 else "info",
        issue_count=max_lt_min,
        total_rows=len(run),
        details="Check that per-run max speed is not lower than min speed.",
        resolution_action=(
            "Swap or correct min/max speed values after source verification."
            if max_lt_min > 0
            else "No action required."
        ),
    )

    for table_name in [
        "player_appearance_pass",
        "player_appearance_behaviour_under_pressure",
        "player_appearance_shot_limited",
    ]:
        df = datasets[table_name]
        minute_viol = range_violation_count(df["minute"], 0.0, 60.0)
        add_finding(
            findings=findings,
            dataset=table_name,
            check_name="event_minute_outside_0_60",
            severity="major" if minute_viol > 0 else "info",
            issue_count=minute_viol,
            total_rows=len(df),
            details="Event minute should be within local period range.",
            resolution_action=(
                "Inspect period/minute encoding and correct or remove inconsistent events."
                if minute_viol > 0
                else "No action required."
            ),
        )

    main = datasets["players_quarters_final"]

    minute_order_viol = int((pd.to_numeric(main["minute_in"], errors="coerce") > pd.to_numeric(main["minute_out"], errors="coerce")).sum())
    add_finding(
        findings=findings,
        dataset="players_quarters_final",
        check_name="minute_in_greater_than_minute_out",
        severity="critical" if minute_order_viol > 0 else "info",
        issue_count=minute_order_viol,
        total_rows=len(main),
        details="Consistency rule: minute_in <= minute_out.",
        resolution_action=(
            "Correct substitution timing records or remove malformed rows before modeling."
            if minute_order_viol > 0
            else "No action required."
        ),
    )

    checkpoint_map = {
        "H1_15": ("half_1", 15),
        "H1_30": ("half_1", 30),
        "H1_45": ("half_1", 45),
        "H2_15": ("half_2", 15),
        "H2_30": ("half_2", 30),
        "H2_45": ("half_2", 45),
        "ET1_15": ("extra_time_1", 15),
    }

    invalid_label = int((~main["checkpoint"].astype("string").isin(checkpoint_map.keys())).sum())
    add_finding(
        findings=findings,
        dataset="players_quarters_final",
        check_name="invalid_checkpoint_label",
        severity="major" if invalid_label > 0 else "info",
        issue_count=invalid_label,
        total_rows=len(main),
        details="Checkpoint must be one of the contest-defined labels.",
        resolution_action=(
            "Map non-standard checkpoint labels to valid values or remove inconsistent rows."
            if invalid_label > 0
            else "No action required."
        ),
    )

    period_mismatch = 0
    minute_mismatch = 0
    checkpoint_str = main["checkpoint"].astype("string")
    checkpoint_period = main["checkpoint_period"].astype("string")
    checkpoint_min = pd.to_numeric(main["checkpoint_min"], errors="coerce")

    for label, (expected_period, expected_minute) in checkpoint_map.items():
        mask = checkpoint_str == label
        period_mismatch += int((checkpoint_period[mask] != expected_period).sum())
        minute_mismatch += int((checkpoint_min[mask] != expected_minute).sum())

    add_finding(
        findings=findings,
        dataset="players_quarters_final",
        check_name="checkpoint_period_mismatch",
        severity="major" if period_mismatch > 0 else "info",
        issue_count=period_mismatch,
        total_rows=len(main),
        details="Checkpoint label and checkpoint_period should be consistent.",
        resolution_action=(
            "Recompute checkpoint_period from checkpoint label and validate against source."
            if period_mismatch > 0
            else "No action required."
        ),
    )

    add_finding(
        findings=findings,
        dataset="players_quarters_final",
        check_name="checkpoint_minute_mismatch",
        severity="major" if minute_mismatch > 0 else "info",
        issue_count=minute_mismatch,
        total_rows=len(main),
        details="Checkpoint label and checkpoint_min should be consistent.",
        resolution_action=(
            "Recompute checkpoint_min from checkpoint label and validate against source."
            if minute_mismatch > 0
            else "No action required."
        ),
    )

    invalid_target = int((~pd.to_numeric(main["scored_after"], errors="coerce").isin([0, 1])).sum())
    add_finding(
        findings=findings,
        dataset="players_quarters_final",
        check_name="scored_after_not_binary",
        severity="critical" if invalid_target > 0 else "info",
        issue_count=invalid_target,
        total_rows=len(main),
        details="Target `scored_after` must be binary (0/1).",
        resolution_action=(
            "Repair target encoding and exclude unresolved rows from supervised modeling."
            if invalid_target > 0
            else "No action required."
        ),
    )


def write_data_quality_report(
    report_path: Path,
    findings_df: pd.DataFrame,
    cardinality_df: pd.DataFrame,
) -> None:
    non_zero = findings_df[findings_df["issue_count"] > 0].copy()
    severity_counts = non_zero["severity"].value_counts().to_dict() if not non_zero.empty else {}

    lines: list[str] = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append("This report was generated by `src/python/data_quality_audit.py`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Datasets audited: 5")
    lines.append("- Checks covered: duplicate records, key integrity, join cardinality, range plausibility, consistency constraints")
    lines.append("")
    lines.append("## Outcome Summary")
    lines.append("")

    if non_zero.empty:
        lines.append("No data quality violations were detected in the implemented checks.")
    else:
        lines.append("| Severity | Number of failed checks |")
        lines.append("|---|---:|")
        for sev in ["critical", "major", "minor"]:
            lines.append(f"| {sev} | {int(severity_counts.get(sev, 0))} |")

    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append("| Dataset | Check | Severity | Issues | Rate (%) | Resolution Action |")
    lines.append("|---|---|---|---:|---:|---|")

    for _, row in findings_df.iterrows():
        lines.append(
            "| "
            + f"{row['dataset']} | {row['check_name']} | {row['severity']} | "
            + f"{int(row['issue_count'])} | {float(row['issue_rate_pct']):.4f} | {row['resolution_action']} |"
        )

    lines.append("")
    lines.append("## Join Cardinality (player_appearance_id)")
    lines.append("")

    if cardinality_df.empty:
        lines.append("No player_appearance_id cardinality statistics were available.")
    else:
        lines.append("| Dataset | Unique keys | Mean rows/key | Median rows/key | P90 rows/key | Max rows/key |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in cardinality_df.iterrows():
            lines.append(
                "| "
                + f"{row['dataset']} | {int(row['unique_keys'])} | {float(row['mean_rows_per_key']):.4f} | "
                + f"{float(row['median_rows_per_key']):.4f} | {float(row['p90_rows_per_key']):.4f} | {int(row['max_rows_per_key'])} |"
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_cleaning_rules(
    rules_path: Path,
    findings_df: pd.DataFrame,
) -> None:
    failed = findings_df[findings_df["issue_count"] > 0].copy()

    lines: list[str] = []
    lines.append("# Cleaning Rules")
    lines.append("")
    lines.append("This document was generated by `src/python/data_quality_audit.py`.")
    lines.append("")
    lines.append("## Core Rules")
    lines.append("")
    lines.append("1. Standardize missing-value tokens (`NULL`, empty strings, etc.) to a single NA representation.")
    lines.append("2. Enforce stable column dtypes (booleans, numerics, categories, datetime) before auditing/modeling.")
    lines.append("3. Preserve uniqueness of primary event IDs (`id`) where present.")
    lines.append("4. Keep key integrity for all `*_player_appearance_id` links.")
    lines.append("5. Enforce range/consistency constraints: non-negative distances/speeds, minute bounds, and checkpoint consistency.")
    lines.append("6. Keep `scored_after` strictly binary (0/1).")
    lines.append("")
    lines.append("## Current Project Actions")
    lines.append("")

    if failed.empty:
        lines.append("No corrective actions are required for the currently implemented checks.")
    else:
        lines.append("| Dataset | Check | Severity | Issues | Action |")
        lines.append("|---|---|---|---:|---|")
        for _, row in failed.iterrows():
            lines.append(
                "| "
                + f"{row['dataset']} | {row['check_name']} | {row['severity']} | {int(row['issue_count'])} | "
                + f"{row['resolution_action']} |"
            )

    rules_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = project_root()
    docs_dir = root / "docs"
    out_dir = root / "artifacts" / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets()
    findings: list[Finding] = []

    run_duplicate_checks(datasets, findings)
    cardinality_df = run_key_integrity_checks(datasets, findings)
    run_range_and_consistency_checks(datasets, findings)

    findings_df = pd.DataFrame([f.to_record() for f in findings])
    findings_df["severity_rank"] = findings_df["severity"].map(SEVERITY_ORDER)
    findings_df = findings_df.sort_values(
        ["severity_rank", "issue_count", "dataset", "check_name"],
        ascending=[True, False, True, True],
    ).drop(columns=["severity_rank"]).reset_index(drop=True)

    findings_df.to_csv(out_dir / "findings.csv", index=False)
    cardinality_df.to_csv(out_dir / "join_cardinality.csv", index=False)

    write_data_quality_report(docs_dir / "data_quality_report.md", findings_df, cardinality_df)
    write_cleaning_rules(docs_dir / "cleaning_rules.md", findings_df)


if __name__ == "__main__":
    main()
