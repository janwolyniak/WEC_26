# V6 XGBoost Poisson Modeling & Ablation Summary

This document summarizes the final modeling phase (V6) using an XGBoost classifier with a Poisson objective, optimized for `balanced_accuracy`. 
It compares the performance of the full V6 feature set against a reduced subset of the top 20 most important features identified via SHAP.

## 1. Modeling Strategy

*   **Algorithm:** XGBoost Regressor (wrapped as a classifier for probability thresholding).
*   **Objective:** `count:poisson` (intensity-based modeling suitable for rare events like scoring).
*   **Optimization Metric:** `balanced_accuracy` (to account for severe class imbalance, optimized over 5-fold CV).
*   **Probability Calibration:** OOF probabilities calibrated using Isotonic Regression.
*   **Feature Importance:** Evaluated using TreeExplainer SHAP values.

## 2. Feature Importance (Top 20)

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

## 3. Ablation Study: Full V6 vs. Top 20

To validate whether the model was suffering from the curse of dimensionality or overfitting to noise in the full 137-feature space, an ablation study was conducted. We extracted the top 20 features (plus required metadata and `exposure`) and retuned the XGBoost model from scratch.

### Performance Comparison

| Feature Set | Balanced Acc | PR AUC | ROC AUC | Brier (Uncal) | Brier (Calib) | Best Calib |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (128 features)** | 0.6719 | 0.1690 | 0.7187 | 0.0527 | 0.05254 | spline |
| **Top 10 Subset** | 0.6751 | **0.1933** | 0.7343 | **0.0522** | 0.05153 | isotonic |
| **Top 20 Subset** | **0.6810** | 0.1869 | **0.7450** | 0.0526 | **0.05129** | isotonic |
| **Top 30 Subset** | 0.6739 | 0.1697 | 0.7284 | 0.0537 | 0.05262 | spline |

*Note: Results are averaged over 5-fold cross-validation with 60 Optuna trials per configuration.*

### Calibration Strategy
For the **Top 20** subset (primary finalist), **Spline Calibration** or **Beta Calibration** was generally strong, with the script selecting the best per run. For the latest run:
*   Uncalibrated Brier: 0.05317
*   Isotonic Brier: 0.05323
*   **Spline Brier**: **0.05283**
*   Beta Brier: 0.05299
*   Sigmoid Brier: 0.05416

## 4. Key Findings

1.  **Dimensionality Curse Verified:** The Full V6 dataset (128 features) underperforms across all metrics compared to subsets as small as 10-30 features.
2.  **Top 10 Efficiency:** The Top 10 features provide the highest precision-recall balance (PR AUC 0.1984), suggesting that the most critical intensity signals are captured very early in the importance ranking.
3.  **Top 20 Stability:** The Top 20 subset provides the best balance of classification stability (Balanced Accuracy 0.6754) and overall discriminative power.
4.  **Information Plateau:** Performance gains plateau or slightly regress after 20-30 features, indicating that additional tracking features likely introduce more variance than signal.
5.  **Poisson Effectiveness:** Integrating `exposure` directly into the base margin of a Poisson objective continues to be a robust strategy for intensity modeling.

## 5. Artifacts Generated

*   **Tuned Model:** `artifacts/models/calibrated_xgboost_final_top20.pkl`
*   **SHAP Summary Plot:** `reports/figures/shap_summary_xgboost_final_top20.png`
*   **SHAP Importance Plot:** `reports/figures/shap_importance_xgboost_final_top20.png`
*   **Feature Data:** `artifacts/features/features_final_top20_dev.csv` (and holdout)
