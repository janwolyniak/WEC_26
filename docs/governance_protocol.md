# WEC 2026 Governance Protocol

## 1. Scope Confirmation
This project delivers a scientifically rigorous, reproducible solution to estimate whether a player will score later in the match.

- Primary prediction target: scored_after (binary).
- Tournament context: UEFA U21 EURO 2025 (provided contest data).
- Unit of analysis in base table: player at match checkpoint.

Research goals to address in final paper:
- RQ1-RQ7 from contest brief (predictive quality, determinants, feature family value, temporal signal, contextual effects).

## 2. Expected Deliverables
Mandatory deliverables for completion:
- Research paper with methods, results, and discussion.
- Fully reproducible codebase for all tables and figures.
- Final modeling dataset(s) used for reported analyses.
- Saved model artifacts and evaluation outputs.

Suggested internal artifacts for reproducibility:
- Fixed data split indices.
- Experiment run log (params, metrics, seed, feature set).
- Versioned figures and tables used in manuscript.

## 3. Success Criteria (Pre-Registered)
Primary model quality criteria:
- Balanced Accuracy.
- PR AUC.

Secondary criteria:
- Brier score.
- Probability calibration quality (curve and calibration error).

Scientific criteria:
- Leakage controls documented and passed.
- Temporal causality checks passed: each checkpoint uses only information available at or before that checkpoint.
- Split quality checks passed: no fixture overlap plus balanced team and player representation across folds.
- Uncertainty reported with confidence intervals.
- Results interpretable and mapped to football context.

## 4. Frozen Experiment Protocol Before Tuning
These protocol choices are frozen before model tuning.

### 4.1 Split Strategy
- Final holdout split grouped by fixture_id.
- No fixture may appear in both development and holdout sets.
- Holdout assignment must be balanced for target prevalence, team participation, and player participation.
- If explicit team IDs are unavailable in a modeling table, use a documented team proxy (for example fixture side defined by fixture_id + is_home) and validate balance with player composition diagnostics.
- Model development performed only on development split.

### 4.2 Cross-Validation Strategy
- Grouped CV using fixture_id to prevent cross-match leakage.
- Use custom fold assignment that jointly balances target prevalence, team participation, and player participation.
- Stratification by target proportion remains mandatory within that grouped assignment.
- Enforce and report per-fold balance tolerances for target prevalence and team/player coverage.
- All preprocessing and feature engineering executed inside CV folds only.
- Fold-only preprocessing controls train-validation contamination and must be combined with separate within-match temporal causality controls.

### 4.3 Model Comparison Rules
- Primary ranking metric: mean CV Balanced Accuracy.
- Secondary reported metrics: ROC AUC, PR AUC, and Brier score.
- Stability criterion: low fold variance across key metrics.
- Preference for simpler/interpretable model when performance is statistically indistinguishable.

### 4.4 Statistical Inference Rules
- Report fold-wise metrics plus 95% confidence intervals.
- Use fixture-level bootstrap for uncertainty where applicable.
- Use paired model comparisons on matched folds for ablation deltas.

### 4.5 Temporal Causality Rules (Within-Match)
- For any checkpoint t, feature construction may use only events with timestamp <= t in that fixture.
- Full-match aggregates, ranks, percentiles, or z-scores are forbidden for earlier checkpoints.
- Rolling and cumulative features must be anchored at checkpoint time with no forward-looking window.
- If player/match normalization is used, normalization statistics must be computed from pre-checkpoint data only.
- Temporal leakage diagnostics must be produced and archived before model comparison.

## 5. Reproducibility Rules
- Fixed random seeds for all stochastic procedures.
- Persist exact split indices used in each run.
- Persist split-balance diagnostics for each split configuration.
- Persist temporal leakage diagnostics for each feature-set version.
- Save feature definitions and preprocessing configuration.
- Keep immutable record of final model class and hyperparameters.
- Ensure end-to-end rerun from clean environment produces same core metrics (within numerical tolerance).

## 6. Project Structure
- notebooks/: exploratory analysis and quick diagnostics.
- src/: reusable code for data prep, features, modeling, evaluation.
- reports/figures/: manuscript-ready plots.
- artifacts/: saved models, split files, prediction outputs, tables.
- docs/: governance and methodological documentation.

## 7. Governance Compliance And Maintenance
- This protocol applies across the entire project lifecycle.
- Any change affecting data splits, leakage controls, metrics, or model-comparison rules must be documented with rationale and date.
- All reported results must reference the protocol version in force at experiment time.
- Compliance requires documented evidence for: split integrity, balance diagnostics, temporal causality checks, and reproducibility artifacts.
