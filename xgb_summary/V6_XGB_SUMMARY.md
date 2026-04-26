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

| Metric | Full V6 (128 features) | Top 20 Subset (21 features) |
| :--- | :--- | :--- |
| **Balanced Accuracy** | 0.6519 | **0.6809** |
| **PR AUC** | 0.1577 | **0.1868** |
| **ROC AUC** | 0.7066 | **0.7449** |
| **Brier Score (Uncalibrated)** | 0.0532 | **0.0525** |

### Calibration Strategy
For the Top 20 subset, **Isotonic Regression** was found to be the most effective calibration technique based on 5-fold CV Brier scores:
*   Uncalibrated Brier: 0.05254
*   Isotonic Brier: **0.05129**
*   Spline Brier: 0.05172
*   Beta Brier: 0.05203
*   Sigmoid Brier: 0.05387

## 4. Key Findings

1.  **Significant Dimensionality Reduction:** Reducing the feature space by over 80% (from 128 to 21 active features) led to a **strict improvement** across all tracking metrics.
2.  **Reduced Overfitting:** The full V6 dataset contained noisy or redundant tracking features. The top 20 feature model is more parsimonious and generalizes better out-of-fold.
3.  **Contextual Features Dominate:** Cumulative tracking metrics (e.g., `cumul_pass_top_share`, `cumul_threat`, `cumul_pressure_accurate_rate`) heavily dominated the SHAP importance, validating the hypothesis that physical context built up over the match is highly predictive.
4.  **Poisson Effectiveness:** Integrating `exposure` directly into the base margin of a Poisson objective effectively normalized the risk window, allowing the model to focus purely on the contextual probability of an event.

## 5. Artifacts Generated

*   **Tuned Model:** `artifacts/models/calibrated_xgboost_final_top20.pkl`
*   **SHAP Summary Plot:** `reports/figures/shap_summary_xgboost_final_top20.png`
*   **SHAP Importance Plot:** `reports/figures/shap_importance_xgboost_final_top20.png`
*   **Feature Data:** `artifacts/features/features_final_top20_dev.csv` (and holdout)
