# WEC 2026 Goal-Scoring Research Reproduction Guide

This repository contains a reproducible research pipeline for the WEC 2026 case study: predicting whether a player will score later in the match (`scored_after`) from checkpoint-level football data and supplementary event tables.

The codebase mixes exploratory notebooks with script-based research artifacts. If you want to reproduce the research end to end, use the Python scripts in the order below.

## 1. What You Need

- Python 3.11 or newer
- Raw CSV files placed in `data/`
- Enough time for model tuning and SHAP analysis; these are the slowest steps

Install the required packages in a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install pandas numpy scipy scikit-learn xgboost optuna shap joblib matplotlib seaborn jupyter
```

## 2. Expected Input Data

The following files must exist in `data/`:

- `players_quarters_final.csv`
- `player_appearance_pass.csv`
- `player_appearance_behaviour_under_pressure.csv`
- `player_appearance_run.csv`
- `player_appearance_shot_limited.csv`

These are the core contest datasets used by the scripts.

## 3. Recommended Reproduction Order

Run all commands from the repository root:

```bash
cd /path/to/WEC2026
```

### Step 1: Build normalized data intake artifacts

```bash
python3 src/python/data_intake.py
```

This generates:

- `artifacts/data_intake/normalized/`
- `artifacts/data_intake/normalized_dtypes.json`
- `docs/schema_summary.md`
- `data/dictionary/data_dictionary_v1.md`

### Step 2: Freeze modeling scope for event tables

```bash
python3 src/python/modeling_scope.py
```

This keeps only event-table rows whose `player_appearance_id` belongs to the base modeling table and writes scope artifacts to `artifacts/modeling_scope/`.

### Step 3: Run the data quality audit

```bash
python3 src/python/data_quality_audit.py
```

Main outputs:

- `artifacts/data_quality/findings.csv`
- `artifacts/data_quality/join_cardinality.csv`
- `docs/data_quality_report.md`
- `docs/cleaning_rules.md`

### Step 4: Create the fixed train/validation/holdout split

```bash
python3 src/python/split_strategy.py
```

Main outputs:

- `data/splits/fixture_metadata.csv`
- `data/splits/fixture_assignments.csv`
- `data/splits/modeling_row_folds.csv`
- `artifacts/splits/split_balance_diagnostics.csv`
- `docs/split_balance_report.md`
- `docs/evaluation_protocol.md`

### Step 5: Run the leakage audit

```bash
python3 src/python/leakage_audit.py
```

Main outputs:

- `artifacts/leakage_audit/`
- `docs/leakage_checklist.md`
- `docs/temporal_leakage_report.md`

### Step 6: Train the baseline models

```bash
python3 src/python/baseline_pipeline.py
python3 src/python/baseline_diagnostics.py
```

Main outputs:

- `artifacts/baseline/fold_metrics.csv`
- `artifacts/baseline/baseline_metrics.csv`
- `artifacts/baseline/feature_importances.csv`
- `artifacts/baseline/fold_metrics_ci.csv`
- `artifacts/baseline/subgroup_metrics.csv`
- `docs/baseline_metrics.md`
- `docs/baseline_diagnostics.md`

### Step 7: Build consolidated engineered feature sets

```bash
python3 src/python/feature_engineering_consolidated.py
```

This generates:

- `artifacts/features/features_final_dev.csv`
- `artifacts/features/features_final_holdout.csv`

To generate the optimized "Top 20" subset used for the final model:

```bash
# First run a full SHAP analysis to get importances (optional if already computed)
# python3 src/python/shap_analysis.py --version final --poisson

# Then extract top 20
python3 src/python/extract_top_n_features.py --n 20
```

This generates `artifacts/features/features_final_top20_dev.csv`.

### Step 8: Tune and Interpret the Finalist Models

```bash
# Tune the Top 20 model with Poisson regression
python3 src/python/model_tuning.py --version final_top20 --poisson --optimize balanced_accuracy
```

Main outputs:

- `artifacts/tuning/xgb_trials.csv`
- `artifacts/tuning/best_params_final_top20.json`
- `artifacts/tuning/pr_curves_final_top20.png`
- `docs/model_tuning.md`

### Step 9: Calibrate the Final Model and Select Threshold

```bash
python3 src/python/probability_calibration.py --version final_top20 --poisson
```

Main outputs:

- `artifacts/models/calibrated_xgboost_final_top20.pkl`
- `artifacts/models/calibration_and_thresholds.png`
- `docs/threshold_policy.md`

### Step 10: Final SHAP Analysis

```bash
python3 src/python/shap_analysis.py --version final_top20 --poisson
```

Main outputs:

- `reports/figures/shap_summary_xgboost_final_top20.png`
- `reports/figures/shap_importance_xgboost_final_top20.png`
- `artifacts/models/shap_importance_final_top20.csv`


### Step 11: Run the RQ3-RQ7 ablation study

```bash
python3 src/python/rq3_rq7_ablation.py
```

Main outputs:

- `artifacts/rq3_rq7_ablation/rq3_rq7_ablation_results.csv`
- `artifacts/rq3_rq7_ablation/rq3_rq7_fold_metrics.csv`
- `artifacts/rq3_rq7_ablation/rq4_incremental_deltas.csv`
- `artifacts/rq3_rq7_ablation/rq5_temporal_comparison.csv`
- `artifacts/rq3_rq7_ablation/rq6_relative_intensity_coefficients.csv`
- `artifacts/rq3_rq7_ablation/rq7_context_importance.csv`

## 4. Optional Notebooks

The notebooks are useful for exploratory analysis and figure development, but they are not the cleanest reproduction path. If you want to inspect the research process, start with:

- `notebooks/eda_block_a_global_structure_and_target_behavior.ipynb`
- `notebooks/eda_block_b_feature_behavior_and_signal_discovery.ipynb`
- `notebooks/eda_block_c_event_table_extension_feasibility.ipynb`
- `notebooks/rq3_rq7_ablation_matrix.ipynb`

There are also sequence-model notebooks that are adjacent to, but not required for, the main tabular pipeline.

## 5. Recommended Reading Order for the Written Outputs

After the scripts finish, the quickest way to review the reproduced research is:

1. `docs/governance_protocol.md`
2. `docs/evaluation_protocol.md`
3. `docs/data_quality_report.md`
4. `docs/temporal_leakage_report.md`
5. `docs/baseline_metrics.md`
6. `docs/baseline_diagnostics.md`
7. `docs/feature_engineering_v1.md`
8. `docs/feature_engineering_v2.md`
9. `docs/model_tuning.md`
10. `docs/threshold_policy.md`

## 6. Notes on Current Script Layout

- `model_tuning.py` is the main tuning entry point. It supports `--version` and `--poisson` flags for the final pipeline.
- `feature_engineering_consolidated.py` replaces the previous `v1-v6` sequence for faster reproduction.
- The older `tune_lr.py`, `tune_xgb.py`, and `model_tuning_report.py` scripts are legacy pieces and are not required if you use the unified tuning script.

## 7. Final Deliverables You Should Expect

If reproduction succeeds, you should have:

- fixed split artifacts in `data/splits/` and `artifacts/splits/`
- cleaned audit and governance documents in `docs/`
- baseline and engineered-feature comparisons in `artifacts/baseline/`
- tuned model summaries in `artifacts/tuning/`
- a calibrated final XGBoost model (`calibrated_xgboost_final_top20.pkl`) in `artifacts/models/`
- SHAP figures in `reports/figures/`

- ablation-study outputs in `artifacts/rq3_rq7_ablation/`
