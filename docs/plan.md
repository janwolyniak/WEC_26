# WEC 2026 Project Plan

## Objective
Develop a scientifically rigorous, reproducible solution for predicting whether a player will score later in the match (`scored_after`) and answer RQ1-RQ7 with clear evidence.

## Scientific Standards To Follow Throughout
- Prevent leakage at every stage (time, match, and target leakage).
- Use grouped validation by match (`fixture_id`) and enforce split balance for target prevalence, participating teams, and participating players.
- Report uncertainty (confidence intervals), not only point estimates.
- Keep all transformations inside training folds and keep all per-match feature calculations time-causal (only data available up to the checkpoint).
- Make all outputs reproducible: fixed seeds, versioned code, saved configs, saved datasets.
- Prioritize both predictive power and interpretability.

## Chronological Execution Plan

### 1) Project Initialization And Governance
1. Confirm scope, target definition, and expected deliverables.
2. Define acceptance criteria for success:
- Primary metrics: Balanced Accuracy, ROC AUC.
- Secondary metrics: PR AUC, Brier score, calibration quality.
3. Freeze experiment protocol before model tuning:
- Data split strategy.
- Model comparison criteria.
- Statistical testing approach.
4. Set project structure for reproducibility:
- `notebooks/` for exploration.
- `src/` for reusable pipelines and feature code.
- `reports/figures/` for plots.
- `artifacts/` for model outputs and intermediate datasets.

Deliverables:
- Written protocol note.
- Standard folder and naming structure.

### 2) Data Intake And Schema Lock
1. Load and profile all five datasets:
- `players_quarters_final.csv`
- `player_appearance_pass.csv`
- `player_appearance_behaviour_under_pressure.csv`
- `player_appearance_run.csv`
- `player_appearance_shot_limited.csv`
2. Normalize dtypes and missing values (e.g., string `NULL` to missing).
3. Create initial data dictionary:
- Variable definitions.
- Units.
- Allowed categorical values.
- Candidate role (`id`, feature, target, key).

Deliverables:
- Schema summary.
- Data dictionary v1.

### 3) Data Quality And Integrity Audit
1. Check duplicate rows and duplicate IDs per table.
2. Validate key integrity and join cardinalities using `player_appearance_id`.
3. Validate ranges and plausibility:
- Speeds, distances, minutes, checkpoint labels.
4. Check consistency rules:
- `minute_in <= minute_out`.
- Checkpoint minute and checkpoint label agreement.
- `scored_after` in `{0,1}` only.
5. Create a data quality report with issue severity and resolution actions.

Deliverables:
- Data quality report.
- Cleaning rules document.

### 4) Leakage Audit (Mandatory Before Modeling)
1. Separate leakage checks into two classes: fold leakage and within-match temporal leakage.
2. Verify no post-checkpoint information enters features.
3. Confirm shot `outcome` is absent from contest modeling table.
4. Ensure all engineered event features are cut off at checkpoint time.
5. Ensure per-match/player transformations (normalization, ranking, percentiles, z-scores) use only information available up to each checkpoint.
6. Ensure no player-match checkpoint appears in both train and validation folds.

Deliverables:
- Leakage checklist signed off.
- Temporal leakage diagnostics report.

### 5) Split Strategy And Evaluation Protocol
1. Build fixture-level split metadata: target prevalence, player participation profile, and team participation profile.
2. Create final holdout split by `fixture_id` (never touched during model development) with balancing constraints for target prevalence, teams, and players.
3. On remaining data, use grouped cross-validation by `fixture_id` with custom stratified fold assignment that jointly balances target prevalence, teams, and players.
4. Validate fold balance with quantitative checks (per-fold deviation thresholds for target prevalence and team/player representation); adjust assignment if thresholds fail.
5. Save split indices and split-balance diagnostics to disk for full reproducibility.
6. Predefine what will be tuned and what will remain fixed.

Deliverables:
- `data/splits/` artifacts.
- Split balance diagnostics report.
- Evaluation protocol note.

### 6) EDA Block A: Global Structure And Target Behavior
1. Row/column counts and unique entities per table.
2. Class imbalance analysis of `scored_after` overall and by:
- Checkpoint.
- Position.
- Home/away.
- Formation.
3. Temporal distribution of positives by checkpoint sequence.
4. Participation dynamics:
- Distribution of `minute_in`, `minute_out`, substitution status.

Deliverables:
- Core descriptive tables.
- Imbalance and temporal plots.

### 7) EDA Block B: Feature Behavior And Signal Discovery
1. Distribution analysis for all `last15_*` and `cumul_*` features.
2. Correlation and redundancy mapping for numeric features.
3. Missingness heatmaps and missingness-target association checks.
4. Univariate signal scans:
- Effect sizes and monotonic trends vs target.
5. Segment analysis by position and game context.

Deliverables:
- EDA notebook/report.
- Candidate feature shortlist.

### 8) EDA Block C: Event-Table Extension Feasibility
1. Quantify event coverage per `player_appearance_id` for pass/pressure/run/shot tables.
2. Check whether event-derived features add enough density to justify inclusion.
3. Define aggregation windows aligned to checkpoint logic:
- Last 15 minutes.
- Cumulative from match start.
4. Identify robust event features (stable, non-sparse, interpretable).

Deliverables:
- Feature extension plan.
- Event aggregation specification.

### 9) Baseline Pipeline Construction
1. Build one leakage-safe baseline pipeline using only `players_quarters_final.csv`.
2. Include preprocessing inside pipeline:
- Imputation.
- Encoding.
- Scaling where needed.
3. Train baseline models:
- Class-weighted Logistic Regression (interpretability baseline).
- One tree-boosting baseline (CatBoost/LightGBM/XGBoost).
4. Evaluate via grouped CV using predefined metrics.

Deliverables:
- Baseline metrics table.
- First interpretation outputs.

### 10) Baseline Diagnostics And Error Analysis
1. Compute fold-level metrics and confidence intervals.
2. Evaluate by subgroup:
- Position.
- Checkpoint.
- Home/away.
3. Inspect confusion patterns and probability calibration.
4. Identify major failure modes and candidate fixes.

Deliverables:
- Diagnostic report.
- Prioritized improvement list.

### 11) Feature Engineering Iteration 1 (Main Table)
1. Create relative intensity features:
- Ratios or deltas between `last15_*` and `cumul_*` trends.
2. Normalize exposure where appropriate (per-minute style features).
3. Add interaction features with context variables (position, formation, home).
4. Refit and compare against baseline under same CV protocol.

Deliverables:
- Engineered feature set v1.
- Incremental gain report.

### 12) Feature Engineering Iteration 2 (Supplementary Tables)
1. Build pass-derived features:
- Volume, accuracy, top-third share, progression proxies.
2. Build pressure-derived features:
- Pressed action count, turnover-under-pressure rate, directional pass profile.
3. Build additional run/shot micro-aggregates not already present in base table.
4. Merge features at checkpoint level with strict time cutoff.
5. Apply only time-causal post-merge transformations (no full-match look-ahead statistics).
6. Refit and benchmark incremental performance.

Deliverables:
- Extended feature set v2.
- Ablation-ready datasets.

### 13) Model Expansion And Hyperparameter Tuning
1. Select a compact candidate set (avoid model zoo sprawl).
2. Run structured hyperparameter search under grouped CV (Optuna).
3. Track every run (params, seed, metrics, feature set).
4. Select best model by predefined primary metric and stability.

Deliverables:
- Tuning summary table.
- Candidate finalist models.

### 14) Probability Calibration And Decision Policy
1. Calibrate probabilities (Platt or isotonic on CV predictions).
2. Evaluate post-calibration Brier score and calibration curves.
3. Select operating threshold using predefined criterion (not holdout).
4. Document tactical interpretation of threshold choice.

Deliverables:
- Calibrated model artifact.
- Threshold policy note.

### 15) RQ-Focused Experimental Matrix (Core Scientific Section)
Run controlled ablation experiments to answer RQ1-RQ7:

1. RQ1: Final predictive quality
- Report Balanced Accuracy and ROC AUC with confidence intervals.
2. RQ2: Determinants of scoring
- SHAP/permutation importance and effect plots.
3. RQ3: Sprints + shots only
- Compare reduced feature model vs full model.
4. RQ4: Added value of pass and pressure data
- Stepwise add these families and report deltas.
5. RQ5: Short-term vs cumulative influence
- Compare `last15`-only, `cumul`-only, combined.
6. RQ6: Relative short-term intensity impact
- Test added relative-intensity features.
7. RQ7: External/context factors
- Context-only and context-augmented comparisons.

Deliverables:
- RQ results tables.
- Evidence-backed conclusions per RQ.

### 16) Robustness And Sensitivity Analysis
1. Repeat evaluations with alternative grouped split seeds and alternative balanced fold constructions.
2. Bootstrap fixture-level confidence intervals.
3. Check model stability by subgroup and sparse scenarios.
4. Stress-test major assumptions:
- Handling of missing values.
- Outlier clipping vs no clipping.
- Alternative class weighting.

Deliverables:
- Robustness appendix.

### 17) Final Model Freeze And Holdout Evaluation
1. Freeze final feature set, model class, hyperparameters, threshold.
2. Retrain on full development data only.
3. Evaluate once on untouched holdout fixtures.
4. Record holdout metrics, calibration, and subgroup behavior.

Deliverables:
- Final holdout performance report.
- Frozen model card.

### 18) Interpretation, Football Insight, And Practical Recommendations
1. Translate top model signals into football language for coaches.
2. Distinguish actionable vs non-actionable factors.
3. Discuss expected use cases and limits in real match analysis.
4. Add caveats on causal interpretation (predictive model, not causal proof).

Deliverables:
- Insight narrative section.
- Practical recommendations list.

### 19) Paper Writing And Reproducibility Package
1. Write paper with standard scientific structure:
- Introduction.
- Data.
- Methods.
- Results.
- Discussion.
- Limitations.
- Conclusion.
2. Include reproducibility details:
- Environment versions.
- Exact split strategy.
- Exact final feature list.
3. Save and provide final modeling dataset used for analysis.
4. Package code so a reviewer can rerun end-to-end with minimal steps.

Deliverables:
- Final manuscript.
- Reproducible code bundle.
- Final analysis dataset.

### 20) Final QA And Submission
1. Re-run end-to-end pipeline from clean state.
2. Confirm all reported numbers match saved artifacts.
3. Verify all figures/tables are correctly labeled and referenced.
4. Complete submission checklist and submit before deadline.
5. Archive final version with timestamp and hashes.

Deliverables:
- Submission-ready package.
- Internal audit checklist signed.

## Recommended Minimal Tooling
- Data and modeling: Python (pandas, numpy, scikit-learn, LightGBM/CatBoost, shap).
- Visualization: matplotlib/seaborn/plotly.
- Experiment tracking: MLflow or structured logs.
- Reproducibility: `requirements.txt` or `pyproject.toml`, fixed seeds, saved splits.

## Definition Of Done
The project is complete when all of the following are true:
- All RQ1-RQ7 are answered with explicit experiments and evidence.
- Final model is validated with grouped CV and untouched holdout.
- Uncertainty is reported (confidence intervals).
- Leakage checks are documented and passed.
- Results are interpretable and connected to football context.
- Full submission package is reproducible end-to-end.