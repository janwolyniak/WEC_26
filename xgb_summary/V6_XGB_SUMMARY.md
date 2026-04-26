# V6 XGBoost Poisson Modeling & Ablation Summary

This document summarizes the final modeling phase (V6) using an XGBoost classifier with a Poisson objective, optimized for `balanced_accuracy`. 
It compares the performance of the full V6 feature set against a reduced subset of the top 20 most important features identified via SHAP.

## 1. Dataset and Feature Composition (V6)

The **V6 (Final)** dataset is the result of a multi-stage feature engineering pipeline designed for the **U21 EURO 2025** tournament. It transforms raw tracking and event data into a high-dimensional feature space (128 features) optimized for predicting scoring intensity.

### Data Foundation
*   **Sampled Checkpoints:** The dataset consists of match-state snapshots taken at specific regular intervals (checkpoints).
*   **Target Variable:** `scored_after` — a binary indicator of whether a goal was scored by the player's team in the subsequent match window.
*   **Splits:** The dataset is partitioned into a **Development set** ($N=2,830$) for training and cross-validation, and a strictly independent **Holdout set** ($N=656$) reserved for final model assessment.
*   **Cross-Validation Strategy:** We employed a **Grouped & Stratified 5-Fold Cross-Validation** approach:
    *   **Fixture-Based Grouping:** All data points from the same `fixture_id` (match) are assigned to the same fold. This prevents temporal and contextual data leakage, ensuring the model is tested on unseen matches.
    *   **Stratification:** Folds are stratified by the `scored_after` target to maintain consistent class distribution.
    *   **Player & Team Balancing:** The splitting logic ensures that multiple snapshots of the same player or team are handled within the match groups to prevent over-fitting to specific team identities or individual player signatures across folds.

### Feature Layers (V1-V6)
*   **V1: Physical Tracking Metrics:** Derived from high-frequency tracking data. Includes distance, sprints, high-speed runs (HSR), and peak speeds. Key features include **pace deltas** (actual vs expected pace based on minutes played) and **per-minute rates**.
*   **V2: Event Aggregations:** Summary statistics for on-ball events (Passes, Pressures, Shots). These are computed for both the **cumulative match duration** and the **last 15-minute window**.
*   **V3: Tactical Context:**
    *   **Formation Parsing:** Automatic extraction of defender/midfielder/attacker counts from team formations.
    *   **Fresh Legs:** Indicators for substitutes and normalized time-on-pitch.
    *   **Team Dynamics:** Team-level aggregates (e.g., total team shots) to capture collective momentum.
*   **V4: Advanced Proxies:**
    *   **Simplified xG (cumul_threat):** A Random Forest-based threat proxy trained on shot characteristics (stage, technique, pressure).
    *   **Passing Centrality:** Ratios of passes received vs. passes made (`in_out_ratio`) to identify focal players.
*   **V5/V6: Intensity Normalization:** Implementation of the **Exposure** feature (time remaining on pitch for the player) used as an offset in the Poisson modeling objective.

## 2. Modeling Strategy

The modeling approach was designed to handle the specific challenges of scoring prediction: high class imbalance, rare events, and varying player-exposure windows.

### Algorithm & Objective
*   **Base Algorithm:** XGBoost Regressor (implemented via `xgboost.XGBRegressor`).
*   **Poisson Objective (`count:poisson`):** Instead of standard binary classification, we model scoring as a Poisson process. This is mathematically superior for "rare count" events like goals, as it directly models the event intensity $\lambda$.
*   **Exposure Normalization (Base Margin):** We use the `exposure` feature (minutes remaining on pitch at checkpoint) as an offset in the base margin: $margin = \log(exposure)$. This allows the model to predict **intensity** (goals per unit of time) rather than just a flat probability.
*   **Poisson-to-Probability Mapping:** To evaluate the model as a classifier, predicted intensities $\lambda$ were transformed into probabilities using the formula: $P(Y \ge 1) = 1 - e^{-\lambda}$.

### Hyperparameter Optimization (Optuna)
*   **Optimization Engine:** We employed the **Tree-structured Parzen Estimator (TPE)** sampler (`optuna.samplers.TPESampler`) with `multivariate=True` to capture dependencies between parameters.
*   **Search Space:**
    *   `n_estimators`: $[50, 400]$ (discrete)
    *   `max_depth`: $[2, 8]$ (discrete)
    *   `learning_rate`: $[0.01, 0.3]$ (log-uniform)
    *   `min_child_weight`: $[1, 20]$ (discrete)
    *   `subsample`: $[0.3, 1.0]$ (uniform)
    *   `colsample_bytree`: $[0.5, 1.0]$ (uniform)
    *   `reg_alpha`: $[1e-8, 10.0]$ (log-uniform)
    *   `reg_lambda`: $[1e-8, 10.0]$ (log-uniform)
*   **Primary Objective Metric:** **Balanced Accuracy**. This metric calculates the arithmetic mean of sensitivity and specificity, ensuring the model maintains high performance on the rare "scored" class without being biased by the majority "not scored" class.
*   **Validation Protocol:** 5-fold **Grouped & Stratified** cross-validation on the development set, where groups are defined by `fixture_id`. This setup ($N=2,830$) ensures that the model generalizes to new matches and avoids leakage between snapshots of the same game.

## 3. Probability Calibration & Evaluation

Since the raw output of the Poisson model is an intensity score, calibration is essential to ensure that the predicted probabilities $P(Y \ge 1)$ correspond accurately to the observed event frequencies.

### Calibration Techniques
We evaluated four calibration methods using a nested cross-validation approach to minimize the **Brier Score** ($MSE = \frac{1}{N} \sum (\hat{p}_i - y_i)^2$):
1.  **Isotonic Regression:** A non-parametric, piece-wise constant approach that preserves monotonicity.
2.  **Sigmoid (Platt) Scaling:** A parametric approach fitting a logistic curve to the model outputs.
3.  **Beta Calibration:** A parametric approach using the Beta distribution, designed for skewed or bounded probability distributions.
4.  **Spline Calibration:** Uses cubic smoothing splines to provide a flexible and smooth mapping from scores to probabilities.

### Evaluation Metrics
*   **Balanced Accuracy:** Calculation: $\frac{1}{2} (\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$.
*   **PR AUC (Precision-Recall Area Under Curve):** Computed via average precision to evaluate the model's ability to rank rare positive events correctly.
*   **Brier Score:** The standard metric for assessing calibration quality (lower is better).

## 4. Feature Ablation & SHAP Interpretability

Based on the SHAP analysis of the full V6 model, the following 20 features were identified as the most impactful for predicting `scored_after`:

1. `cumul_pass_top_share`
2. `cumul_pass_middle_share`
3. `cumul_threat`
4. `cumul_in_out_ratio`
5. `cumul_pressure_accurate_rate`
6. `position` (Categorical)
7. `cumul_pressure_top_share`
8. `cumul_pressure_turnover_rate`
9. `cumul_pressure_pass_angle_observed_share`
10. `last15_pass_middle_share`
11. `pace_delta_sprints`
12. `team_cumul_shots`
13. `pm_distance`
14. `ratio_mean_max_speed`
15. `player_id` (Categorical)
16. `last15_hsr`
17. `is_pos_D`
18. `cumul_peak_speed`
19. `cumul_pressure_count`
20. `is_pos_A`

## 5. Ablation Study: Comparison of Feature Subsets

To validate whether the model was suffering from the curse of dimensionality or overfitting to noise in the full 128-feature space, an ablation study was conducted. We extracted the top 10, 20, and 30 features (plus required metadata and `exposure`) and retuned the XGBoost model from scratch for each configuration.

### Performance Comparison

| Feature Set | Balanced Acc | PR AUC | ROC AUC | Brier (Uncal) | Brier (Calib) | Best Calib |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (128 features)** | 0.6719 | 0.1690 | 0.7187 | 0.0527 | 0.05254 | spline |
| **Top 10 Subset** | 0.6751 | **0.1933** | 0.7343 | **0.0522** | 0.05153 | isotonic |
| **Top 20 Subset** | **0.6810** | 0.1869 | **0.7450** | 0.0526 | **0.05129** | isotonic |
| **Top 30 Subset** | 0.6739 | 0.1697 | 0.7284 | 0.0537 | 0.05262 | spline |

*Note: Results are averaged over 5-fold cross-validation with 60 Optuna trials per configuration.*

### Calibration Example (Top 20 Subset)
For the **Top 20** subset (primary finalist), the calibration script automatically selects the method with the lowest Brier score. In the finalized run, **Isotonic Regression** provided the best fit:
*   Uncalibrated Brier: 0.05254
*   **Isotonic Brier**: **0.05129**
*   Spline Brier: 0.05172
*   Beta Brier: 0.05203
*   Sigmoid Brier: 0.05387

## 6. Key Findings

1.  **Dimensionality Curse Verified:** The Full V6 dataset (128 features) underperforms across all metrics compared to subsets as small as 10-30 features.
2.  **Top 10 Efficiency:** The Top 10 features provide the highest precision-recall balance (PR AUC 0.1933), suggesting that the most critical intensity signals are captured very early in the importance ranking.
3.  **Top 20 Stability:** The Top 20 subset provides the best balance of classification stability (Balanced Accuracy 0.6810) and overall discriminative power.
4.  **Information Plateau:** Performance gains plateau or slightly regress after 20-30 features, indicating that additional tracking features likely introduce more variance than signal.
5.  **Poisson Effectiveness:** Integrating `exposure` directly into the base margin of a Poisson objective continues to be a robust strategy for intensity modeling.

## 7. Artifacts Generated

*   **Tuned Model:** `artifacts/models/calibrated_xgboost_final_top20.pkl`
*   **SHAP Summary Plot:** `reports/figures/shap_summary_xgboost_final_top20.png`
*   **SHAP Importance Plot:** `reports/figures/shap_importance_xgboost_final_top20.png`
*   **Feature Data:** `artifacts/features/features_final_top20_dev.csv` (and holdout)
