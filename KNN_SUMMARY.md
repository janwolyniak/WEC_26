# K-Nearest Neighbors (KNN) Modeling Summary

## 1. Modeling Rationale

This document summarizes the KNN line of modeling built on `data/knn_dataset.csv` and documented in [knn_pipeline.ipynb](/Users/jan/Documents/competitions/hackatons/WEC_26/notebooks/knn_pipeline.ipynb). The KNN model was not treated as the expected final contest winner. Instead, it served four scientific purposes that are directly useful for the research paper:

1. It established a simple, transparent, nonparametric baseline on the checkpoint-level data.
2. It tested whether local similarity in player-match states is predictive of later scoring.
3. It showed how strongly the results depend on feature scaling, feature selection, and threshold choice.
4. It provided an interpretable contrast against more structured models such as gradient boosting and neural architectures.

In practical terms, the KNN experiments answer a narrow but important question: if two checkpoint rows are similar in recent/cumulative performance and match context, can a neighborhood vote recover later-goal risk with acceptable discrimination?

## 2. Figures

The following paper-ready figures accompany this summary:

- KNN development trajectory: [knn_evolution_metrics.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/knn_evolution_metrics.png)
- KNN threshold trade-offs: [knn_threshold_tradeoff.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/knn_threshold_tradeoff.png)

## 3. Data Representation And Evaluation Protocol

The KNN notebook uses `data/knn_dataset.csv`, which contains `3,486` rows and `31` columns. The target prevalence is approximately `5.82%`, so the task is a strongly imbalanced binary classification problem.

The feature space combines:

- contextual variables:
  - `checkpoint`
  - `position`
  - `formation`
  - `minute_in_window`
  - `is_home`
  - `subbed`
- recent-performance aggregates:
  - `last15_sprints`
  - `last15_hsr`
  - `last15_shots`
  - `last15_shots_on_target`
  - `last15_shots_under_press`
  - `last15_shots_top_third`
  - pass activity and accuracy aggregates
- cumulative aggregates from match start:
  - `cumul_sprints`
  - `cumul_hsr`
  - `cumul_shots`
  - `cumul_shots_on_target`
  - `cumul_shots_under_press`
  - `cumul_shots_top_third`
  - cumulative pass activity and accuracy aggregates
- exposure variable:
  - `minutes_in_game`

The row identifier `scored_after_eval_key` is used to recover a fixture-like group identifier for grouped evaluation. The KNN notebook uses `StratifiedGroupKFold` with `5` splits, which is a sound compromise between leakage control and class-balance stability in a small rare-event setting.

The tuning and comparison metrics are:

- primary metric:
  - PR AUC
- secondary metrics:
  - balanced accuracy
  - ROC AUC
  - Brier score

This protocol fits the research setting well. PR AUC is the correct ranking metric for a low-prevalence target, while balanced accuracy and ROC AUC provide additional discrimination diagnostics.

## 4. Model Specification

The initial KNN pipeline uses a leakage-safe preprocessing stack:

- numeric variables:
  - median imputation
  - standardization
- categorical variables:
  - most-frequent imputation
  - one-hot encoding
- boolean variables:
  - conversion to integer
  - most-frequent imputation

The classifier is `KNeighborsClassifier`, tuned over a compact grid:

- `n_neighbors` in `{5, 11, 21, 31}`
- `weights` in `{"uniform", "distance"}`
- Minkowski distance with:
  - `p = 1` for Manhattan distance
  - `p = 2` for Euclidean distance

This is an appropriate design for a paper baseline because it is small enough to explain clearly, but broad enough to show whether local geometry is meaningful.

## 5. Model Development

### 5.1 Original KNN On The Full Feature Space

The first tuning stage searched the compact KNN grid on the original feature set. The notebook records the current best tested setup as:

- `n_neighbors = 31`
- `weights = "distance"`
- `p = 1`
- mean PR AUC = `0.0963`
- mean ROC AUC = `0.6152`

This result is modest but informative. It shows that local neighborhood structure is not random; there is signal in the checkpoint-level representation. However, the result is not strong enough to support KNN as the final modeling family.

### 5.2 Removing ID-Like Features

The next extension explicitly removed:

- `player_id`
- `jersey_number`

This was an important methodological correction. Both variables are dangerous in a distance-based model:

- `player_id` is an arbitrary identifier, not a football mechanism.
- `jersey_number` is semantically weak and may act like pseudo-identity.
- In a KNN model, even weakly informative or spurious numeric identifiers can distort the neighborhood geometry.

After removing these variables and rerunning the compact search, the best reduced-feature configuration became:

- `n_neighbors = 31`
- `weights = "distance"`
- `p = 2`
- mean PR AUC = `0.0967`
- mean ROC AUC = `0.6251`
- mean balanced accuracy at threshold `0.5` = `0.5000`
- mean Brier score = `0.0555`

The gain in PR AUC was small, but the gain in ROC AUC was more noticeable. This suggests that the original KNN geometry was indeed being polluted by ID-like variables, and removing them made the neighborhood structure slightly more coherent.

### 5.3 Threshold Tuning On Reduced-Feature KNN

The reduced-feature KNN line then introduced threshold tuning on pooled out-of-fold probabilities instead of fixing the decision threshold at `0.5`.

This is a crucial methodological step in an imbalanced setting:

- KNN probabilities are often poorly calibrated for rare-event classification.
- A threshold of `0.5` is usually too conservative when the positive class rate is below `10%`.
- Operational utility depends as much on the decision threshold as on the underlying probability ranking.

The best reduced-feature threshold was:

- threshold = `0.05`

At that threshold, the notebook reports:

- balanced accuracy = `0.6098`
- recall = `0.7143`
- precision = `0.0820`
- predicted positive rate approximately `0.50`
- pooled OOF PR AUC = `0.0878`
- pooled OOF ROC AUC = `0.5962`

This result is important for the research paper because it demonstrates the distinction between ranking quality and decision behavior. The reduced-feature KNN is not an especially strong ranking model, but with an aggressive threshold it becomes a recall-oriented alerting model.

The trade-off is clear:

- benefit:
  - much better sensitivity to future scorers
- cost:
  - low precision
  - many false positives
  - roughly half of rows are flagged as positive

### 5.4 PCA Before KNN

The final KNN extension inserted PCA after preprocessing and before distance calculation. This tests whether a compact latent representation improves neighborhood behavior by:

- compressing correlated numeric and one-hot dimensions
- smoothing local geometry
- reducing the impact of sparse high-dimensional expansions

The PCA search tested:

- `n_components` in `{8, 12, 16, 20, 24}`
- the same compact KNN grid as before

The best PCA configuration ranked by mean PR AUC was:

- `n_components = 12`
- `n_neighbors = 31`
- `weights = "distance"`
- `p = 1`
- mean PR AUC = `0.1079`
- mean ROC AUC = `0.6026`
- mean balanced accuracy at threshold `0.5` = `0.5000`
- mean Brier score = `0.0559`

The direct comparison recorded in the notebook is:

| Model variant | Mean PR AUC | Mean ROC AUC | Mean Balanced Accuracy | Mean Brier Score |
|---|---:|---:|---:|---:|
| reduced KNN without PCA | 0.0967 | 0.6251 | 0.5000 | 0.0555 |
| reduced KNN with PCA | 0.1079 | 0.6026 | 0.5000 | 0.0559 |

This is a meaningful result. PCA improved PR AUC substantially for KNN, even though ROC AUC declined. In a low-prevalence setting, that is a defensible trade: the PCA-extended KNN ranks positives earlier in the probability list, which is more relevant to the paper’s main predictive objective than preserving the exact ROC ordering.

Threshold tuning for the PCA model selected:

- threshold = `0.08`
- balanced accuracy = `0.5709`
- precision = `0.0851`
- recall = `0.4236`
- predicted positive rate = `0.2900`
- pooled OOF PR AUC = `0.0878`
- pooled OOF ROC AUC = `0.5962`

Compared with the reduced-feature no-PCA threshold solution, the PCA model is more selective and less recall-heavy.

These stage-wise changes are visualized in [knn_evolution_metrics.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/knn_evolution_metrics.png), while the threshold sensitivity of the reduced and PCA-enhanced variants is shown in [knn_threshold_tradeoff.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/knn_threshold_tradeoff.png).

## 6. Design Logic And Justification

The KNN pipeline evolved in a scientifically coherent order.

### 6.1 Why Remove ID-Like Features

Distance-based models assume that every coordinate in feature space contributes meaningfully to similarity. That assumption is violated by arbitrary identifiers. In KNN, this is more harmful than in many linear or tree models because the entire prediction mechanism depends on a sensible distance geometry.

Removing `player_id` and `jersey_number` therefore improved methodological validity even before considering predictive performance.

### 6.2 Why Tune The Threshold

The research task is imbalanced. A model can have a useful probability ranking while making poor hard classifications at threshold `0.5`. Threshold tuning was necessary to show what the same reduced-feature KNN can do in a recall-oriented coaching scenario.

### 6.3 Why Add PCA

KNN is vulnerable to high-dimensional sparse representations. One-hot encoded context and multiple correlated aggregates make Euclidean or Manhattan neighborhoods noisy. PCA was introduced to determine whether a lower-dimensional latent space could restore local structure. The PR AUC gain suggests that it did.

## 7. Results And Research-Question Interpretation

The contest’s research questions are broader than model leaderboard performance. KNN contributes usefully even if it is not the best final model.

### 7.1 RQ1: How Well Can Player Behavior Predict Later Scoring?

The KNN family demonstrates that simple local similarity contains signal, but only moderate signal.

The strongest KNN ranking result in the notebook is:

- PCA-extended KNN:
  - mean PR AUC = `0.1079`

This supports the claim that checkpoint-level behavior and context are predictive, but that a local distance-based learner is not the most efficient extractor of that signal.

### 7.2 RQ2: Which Aspects Of Behavior Determine Scoring Probability?

KNN does not provide feature coefficients or native importance scores, so it is weak for direct determinant analysis. Its main contribution to RQ2 is indirect:

- removing identifier-like variables improved behavior
- PCA improved ranking performance, implying that the useful signal is distributed across correlated feature combinations rather than a few isolated axes

Thus, KNN suggests that determinants are multivariate and interaction-heavy, but it does not identify them sharply.

### 7.3 RQ3: Are Sprints And Shots Alone Sufficient?

The KNN notebook does not run a dedicated sprint-and-shot-only ablation. However, because the KNN feature space includes passes and context, and because KNN still performs only moderately, the KNN line does not support the idea that simple raw volume similarity in physical and shooting behavior is sufficient on its own.

### 7.4 RQ4: Does Passing Data Improve Predictive Performance?

KNN does not isolate passing data in a formal ablation. Still, the KNN dataset includes pass aggregates, and the moderate performance suggests that pass information may help, but not enough to make a naive local similarity model competitive with stronger nonlinear learners.

### 7.5 RQ5: Short-Term Vs Cumulative Influence

The KNN dataset combines both recent (`last15_*`) and cumulative (`cumul_*`) aggregates in the same geometry. Since KNN does not disentangle them, it does not answer RQ5 directly. Its value here is only as a baseline using both scales simultaneously.

### 7.6 RQ6: Does Short-Term Intensity Relative To Overall Level Matter?

Not directly. The KNN notebook does not construct explicit relative-intensity features. This omission is itself informative: KNN on the simpler aggregate space works, but only modestly, which supports the later move toward richer engineered sequence-aware features and more expressive nonlinear models.

### 7.7 RQ7: Do External/Context Factors Matter?

Because context variables are included in the feature space and one-hot encoded, the KNN model does incorporate them. The fact that the model performs above chance suggests that context helps define useful neighborhoods. However, KNN cannot isolate which contextual factors matter most in an interpretable way.

## 8. Discussion

The KNN family should be interpreted as a methodological baseline rather than a final football-analytic explanation engine.

Its main strengths are:

- conceptual simplicity
- full nonlinearity without parametric assumptions
- good didactic value for explaining grouped CV, threshold tuning, and class imbalance

Its main weaknesses are:

- weak interpretability for feature-level football insights
- sensitivity to feature scaling and representation
- poor robustness in sparse or high-dimensional spaces
- threshold dependence under strong class imbalance

The most scientifically useful conclusion is that local similarity is present but not strong enough to stand alone. Better answers to the contest questions require models that can represent structured interactions more efficiently.

## 9. Paper-Ready Takeaway

For the paper, the KNN results support the following statement:

> A simple distance-based learner can recover non-trivial signal from checkpoint-level football behavior, confirming that similar player-match states tend to share some later scoring risk. However, its predictive quality remains modest, and its performance is highly sensitive to representation choices such as removal of identifier-like variables, threshold selection, and latent compression. KNN therefore functions best as a transparent baseline rather than as the primary explanatory or predictive model.

If a compact results paragraph is needed, the most defensible KNN headline is:

> The strongest KNN ranking result was obtained after PCA compression of the reduced feature space (`12` components, `k=31`, distance weighting, Manhattan distance), yielding mean PR AUC `0.1079`. Removing ID-like variables improved the neighborhood geometry, and threshold tuning increased balanced accuracy in recall-oriented settings, but the KNN family remained primarily a baseline rather than a final candidate for answering the full set of research questions.
