from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import pandas as pd


NA_TOKENS = ["NULL", "null", "None", "none", "N/A", "n/a", "NA", "na", ""]

DATASET_FILES = [
    "players_quarters_final.csv",
    "player_appearance_pass.csv",
    "player_appearance_behaviour_under_pressure.csv",
    "player_appearance_run.csv",
    "player_appearance_shot_limited.csv",
]

BOOL_COLUMNS = {"is_home", "subbed", "accurate", "under_pressure"}
DATE_COLUMNS = {"date"}
CATEGORICAL_HINTS = {
    "period",
    "stage",
    "position",
    "formation",
    "checkpoint",
    "checkpoint_period",
    "run_type",
    "body_part",
    "technique",
    "play_pattern",
    "press_induced_outcome",
}

PRIMARY_ID_COLUMNS = {"id"}
KEY_COLUMNS = {
    "player_appearance_id",
    "player_id",
    "fixture_id",
    "addressee_player_appearance_id",
    "pressing_player_appearance_id",
    "own_goal_player_appearance_id",
    "block_player_appearance_id",
}
TARGET_COLUMNS = {"scored_after"}

COLUMN_META: dict[str, tuple[str, str]] = {
    "player_appearance_id": ("Identifier of a player appearance in a specific match.", "id"),
    "player_id": ("Identifier of a player across appearances.", "id"),
    "fixture_id": ("Identifier of a match fixture.", "id"),
    "date": ("Match date.", "date"),
    "checkpoint": ("Checkpoint label in match timeline.", "category"),
    "checkpoint_period": ("Match period of checkpoint.", "category"),
    "checkpoint_min": ("Local minute of checkpoint within period.", "minute"),
    "position": ("Player position category.", "category"),
    "is_home": ("Flag indicating whether the player's team is home.", "boolean"),
    "formation": ("Team formation at checkpoint.", "category"),
    "minute_in": ("Continuous match minute when player entered pitch.", "minute"),
    "minute_out": ("Continuous match minute when player left pitch.", "minute"),
    "subbed": ("Flag indicating whether player was substituted on/off.", "boolean"),
    "jersey_number": ("Player jersey number.", "count"),
    "period": ("Match period for event.", "category"),
    "id": ("Unique event identifier.", "id"),
    "minute": ("Minute of event within period.", "minute"),
    "stage": ("Pitch zone of event origin.", "category"),
    "accurate": ("Flag indicating event pass accuracy.", "boolean"),
    "run_type": ("Type of run event.", "category"),
    "min_speed": ("Minimum speed during run.", "m/s"),
    "max_speed": ("Maximum speed during run.", "m/s"),
    "distance": ("Distance covered during run.", "meters"),
    "body_part": ("Body part used for shot.", "category"),
    "technique": ("Shot technique.", "category"),
    "play_pattern": ("Attacking play pattern.", "category"),
    "possession": ("Possession sequence identifier.", "id"),
    "under_pressure": ("Flag indicating action under pressure.", "boolean"),
    "press_induced_outcome": ("Outcome of action under pressure.", "category"),
    "pass_angle": ("Angle of resulting pass.", "degrees"),
    "scored_after": ("Target indicator: scored after checkpoint.", "binary"),
}

BASE_METRICS: dict[str, tuple[str, str]] = {
    "sprints": ("Number of sprints (maximal-speed runs)", "count"),
    "hsr": ("Number of high-speed runs", "count"),
    "distance": ("Distance covered in high-intensity runs", "meters"),
    "mean_max_speed": ("Mean of per-run maximum speed", "m/s"),
    "peak_speed": ("Peak maximum speed", "m/s"),
    "shots": ("Number of shots (excluding own goals)", "count"),
    "shots_on_target": ("Number of shots on target", "count"),
    "shots_under_press": ("Number of shots taken under pressure", "count"),
    "shots_top_third": ("Number of shots from attacking third", "count"),
}


@dataclass
class DatasetProfile:
    name: str
    rows: int
    cols: int
    missing_cells: int
    duplicate_rows: int
    memory_mb: float
    dtypes: dict[str, int]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_bool_series(series: pd.Series) -> pd.Series:
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n"}

    as_string = series.astype("string").str.strip().str.lower()
    mapped = as_string.map(lambda x: True if x in true_values else (False if x in false_values else pd.NA))
    return mapped.astype("boolean")


def numeric_cast(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric.astype("Float64")

    is_integer_like = ((numeric.dropna() % 1) == 0).all()
    return numeric.astype("Int64") if is_integer_like else numeric.astype("Float64")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].replace({"": pd.NA})

    for col in out.columns:
        if col in DATE_COLUMNS:
            out[col] = pd.to_datetime(out[col], errors="coerce")
            continue

        if col in BOOL_COLUMNS:
            out[col] = normalize_bool_series(out[col])
            continue

        if str(out[col].dtype) in {"string", "object"}:
            values = out[col].astype("string")
            non_null = int(values.notna().sum())

            if non_null > 0:
                candidate_bool = values.dropna().str.lower().isin(
                    {"true", "false", "t", "f", "1", "0", "yes", "no", "y", "n"}
                ).all()
                if candidate_bool:
                    out[col] = normalize_bool_series(values)
                    continue

                numeric = pd.to_numeric(values, errors="coerce")
                numeric_success = int(numeric.notna().sum())
                if numeric_success / non_null >= 0.98:
                    out[col] = numeric_cast(values)
                    continue

                cardinality = int(values.nunique(dropna=True))
                if col in CATEGORICAL_HINTS or (0 < cardinality <= 30 and cardinality / max(len(values), 1) <= 0.5):
                    out[col] = values.astype("category")
                else:
                    out[col] = values.astype("string")

    return out


def infer_role(column: str) -> str:
    if column in TARGET_COLUMNS:
        return "target"
    if column in PRIMARY_ID_COLUMNS:
        return "id"
    if column in KEY_COLUMNS or column.endswith("_id"):
        return "key"
    return "feature"


def infer_meta(column: str) -> tuple[str, str]:
    if column in COLUMN_META:
        return COLUMN_META[column]

    if column.startswith("last15_"):
        base = column.replace("last15_", "", 1)
        if base in BASE_METRICS:
            desc, unit = BASE_METRICS[base]
            return (f"{desc} in the 15 minutes before checkpoint.", unit)

    if column.startswith("cumul_"):
        base = column.replace("cumul_", "", 1)
        if base in BASE_METRICS:
            desc, unit = BASE_METRICS[base]
            return (f"Cumulative {desc.lower()} from match start to checkpoint.", unit)

    if column.endswith("_id"):
        return ("Identifier column used for linking data.", "id")

    return ("Recorded or derived variable from source event data.", "unknown")


def allowed_values(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return ""

    if str(series.dtype) in {"boolean", "category"} or non_null.nunique() <= 20:
        unique_values = sorted({str(v) for v in non_null.unique()})
        return " | ".join(unique_values)

    return ""


def profile_dataset(name: str, df: pd.DataFrame) -> DatasetProfile:
    dtype_counts: dict[str, int] = {}
    for dtype_name in df.dtypes.astype(str):
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

    return DatasetProfile(
        name=name,
        rows=len(df),
        cols=len(df.columns),
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        memory_mb=float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        dtypes=dtype_counts,
    )


def build_outputs() -> None:
    root = project_root()
    data_dir = root / "data"
    dictionary_dir = data_dir / "dictionary"
    docs_dir = root / "docs"
    out_dir = root / "artifacts" / "data_intake"
    normalized_dir = out_dir / "normalized"

    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    dictionary_dir.mkdir(parents=True, exist_ok=True)

    schema_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    dtype_registry: dict[str, dict[str, str]] = {}

    for filename in DATASET_FILES:
        path = data_dir / filename
        raw = pd.read_csv(path, keep_default_na=True, na_values=NA_TOKENS, low_memory=False)
        normalized = normalize_dataframe(raw)

        dataset_name = path.stem
        profile = profile_dataset(dataset_name, normalized)

        dtype_registry[dataset_name] = {col: str(dtype) for col, dtype in normalized.dtypes.items()}

        for dtype_name, count in sorted(profile.dtypes.items()):
            schema_rows.append(
                {
                    "dataset": profile.name,
                    "rows": profile.rows,
                    "columns": profile.cols,
                    "missing_cells": profile.missing_cells,
                    "duplicate_rows": profile.duplicate_rows,
                    "memory_mb": round(profile.memory_mb, 3),
                    "dtype": dtype_name,
                    "dtype_count": count,
                }
            )

        for col in normalized.columns:
            series = normalized[col]
            definition, unit = infer_meta(col)

            dictionary_rows.append(
                {
                    "dataset": profile.name,
                    "column": col,
                    "normalized_dtype": str(series.dtype),
                    "candidate_role": infer_role(col),
                    "definition": definition,
                    "unit": unit,
                    "allowed_values": allowed_values(series),
                    "missing_count": int(series.isna().sum()),
                    "missing_pct": round(float(series.isna().mean() * 100), 4),
                    "unique_non_null": int(series.nunique(dropna=True)),
                    "example_values": " | ".join(
                        [str(v) for v in series.dropna().astype(str).head(5).tolist()]
                    ),
                }
            )

        normalized.to_csv(normalized_dir / f"{dataset_name}_normalized.csv", index=False)

    schema_df = pd.DataFrame(schema_rows).sort_values(["dataset", "dtype"]).reset_index(drop=True)
    dictionary_df = pd.DataFrame(dictionary_rows).sort_values(["dataset", "column"]).reset_index(drop=True)

    schema_df.to_csv(out_dir / "schema_summary.csv", index=False)
    dictionary_df.to_csv(dictionary_dir / "data_dictionary_v1.csv", index=False)

    with open(out_dir / "normalized_dtypes.json", "w", encoding="utf-8") as f:
        json.dump(dtype_registry, f, indent=2)

    write_schema_markdown(schema_df, docs_dir / "schema_summary.md")
    write_dictionary_markdown(dictionary_df, dictionary_dir / "data_dictionary_v1.md")



def write_schema_markdown(schema_df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Schema Summary")
    lines.append("")
    lines.append("This file was generated by `src/python/data_intake.py`.")
    lines.append("")

    for dataset in schema_df["dataset"].unique():
        subset = schema_df[schema_df["dataset"] == dataset]
        head = subset.iloc[0]
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- Rows: {int(head['rows'])}")
        lines.append(f"- Columns: {int(head['columns'])}")
        lines.append(f"- Missing cells: {int(head['missing_cells'])}")
        lines.append(f"- Duplicate rows: {int(head['duplicate_rows'])}")
        lines.append(f"- Memory usage: {float(head['memory_mb']):.3f} MB")
        lines.append("")
        lines.append("| Normalized dtype | Number of columns |")
        lines.append("|---|---:|")
        for _, row in subset.iterrows():
            lines.append(f"| {row['dtype']} | {int(row['dtype_count'])} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")



def write_dictionary_markdown(dictionary_df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Data Dictionary v1")
    lines.append("")
    lines.append("This file was generated by `src/python/data_intake.py`.")
    lines.append("")

    for dataset in dictionary_df["dataset"].unique():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Column | Role | Dtype | Unit | Missing % | Unique non-null |")
        lines.append("|---|---|---|---|---:|---:|")
        subset = dictionary_df[dictionary_df["dataset"] == dataset]
        for _, row in subset.iterrows():
            lines.append(
                "| "
                + f"{row['column']} | {row['candidate_role']} | {row['normalized_dtype']} | "
                + f"{row['unit']} | {row['missing_pct']:.4f} | {int(row['unique_non_null'])} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
