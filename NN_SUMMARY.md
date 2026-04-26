# Sequence-Aware Neural Modeling Summary

## 1. Modeling Rationale

This document summarizes the sequence-aware and FT-transformer-oriented modeling line built on `data/sequencing_dataset.csv`, together with the intermediate baseline and neural comparison notebooks:

- [sequencing_model_pipeline.ipynb](/Users/jan/Documents/competitions/hackatons/WEC_26/notebooks/sequencing_model_pipeline.ipynb)
- [sequencing_dual_branch_nn.ipynb](/Users/jan/Documents/competitions/hackatons/WEC_26/notebooks/sequencing_dual_branch_nn.ipynb)
- [ft_transformer_search.ipynb](/Users/jan/Documents/competitions/hackatons/WEC_26/notebooks/ft_transformer_search.ipynb)

The central modeling decision was to treat the available “sequence” information in a form that matches the actual data representation. The project does not start from raw event-by-event tensors. Instead, `sequencing_dataset.csv` is a checkpoint-level wide table in which sequential football behavior has already been summarized into structured families of features. This has two important implications:

1. A plain RNN or LSTM is not the most natural first neural architecture.
2. The modeling problem is better understood as structured tabular learning with sequence-aware feature families.

The neural line therefore evolved in a staged way:

- first, benchmark tabular baselines on the sequence-aware table
- second, test whether trimmed sequence families help
- third, build a custom dual-branch neural model for context plus sequence fusion
- fourth, move to an FT-transformer-style model that is better suited to interaction-rich tabular structure
- fifth, extend the FT-transformer search toward stronger objectives, including focal and Poisson-style rare-event modeling

## 2. Figures

The following paper-ready figures accompany this summary:

- XGBoost sequence ablation: [sequence_xgb_ablation.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/sequence_xgb_ablation.png)
- neural benchmark comparison: [neural_benchmark_comparison.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/neural_benchmark_comparison.png)
- sequence-to-neural evolution: [sequence_nn_evolution.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/sequence_nn_evolution.png)
- FT-transformer search comparison: [ft_transformer_search_comparison.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/ft_transformer_search_comparison.png)

## 3. Data Representation And Evaluation Protocol

The sequence-aware modeling line uses `data/sequencing_dataset.csv`, which contains `3,486` rows and `99` columns. The target prevalence is again approximately `5.82%`, so the task remains a rare-event binary classification problem.

The dataset consists of:

- `27` non-sequence context and aggregate columns after excluding target and identifiers
- `24` `seq15m_*` features
- `14` `seqpos_*` features
- `30` `seqextra_*` features

Conceptually, the feature families capture:

- `seq15m_*`:
  minute-level short-window structure, pressure volumes, directionality, recency, burstiness, and local timing
- `seqpos_*`:
  possession-level linking, escalation, and structured run/shot behavior
- `seqextra_*`:
  extra composition variables such as last-stage, body part, play pattern, and technique distributions

The split protocol reuses the project’s frozen grouped evaluation metadata. `scored_after_eval_key` is joined back to fixture-level split assignments so every experiment remains aligned with the same development folds and holdout logic used elsewhere in the project.

The primary comparison metric is PR AUC. Secondary metrics are balanced accuracy, ROC AUC, and Brier score. This is the correct setup for the contest’s rare-event goal-prediction framing.

## 4. Model Development

### 4.1 Sequence-Aware Tabular Benchmark

The first notebook, `sequencing_model_pipeline.ipynb`, asked a practical question:

> If we stay in the wide checkpoint representation and add richer sequence-aware features, do standard tabular learners improve meaningfully?

The notebook compared:

- Logistic Regression
- XGBoost
- a small PyTorch MLP benchmark

It evaluated multiple feature sets:

- `base_context`
- `base_plus_sequence`
- `full_engineered`
- trimmed `seq15m` and `seqpos` subsets

### 4.1.1 Why A Tabular Benchmark Came First

This was the correct first step for several reasons:

- the input was already engineered and aggregated
- a tree model is usually the right stress test for nonlinear tabular signal
- a small MLP provides a low-cost neural reference point
- it lets the project answer whether more complex sequence-aware engineering is even helping before escalating model capacity

### 4.1.2 Full Sequence Block: Too Noisy

The notebook showed that naively adding the entire sequence block degraded performance.

For `XGBoost`:

| Feature set | Mean PR AUC | Mean Balanced Accuracy | Mean ROC AUC | Mean Brier Score |
|---|---:|---:|---:|---:|
| `base_context` | `0.1161` | `0.5541` | `0.6424` | `0.1073` |
| `base_plus_sequence` | `0.0942` | `0.5310` | `0.5931` | `0.0873` |
| `full_engineered` | `0.0942` | `0.5310` | `0.5931` | `0.0873` |

This is an important substantive result. It means that richer sequence-aware representation is not automatically beneficial. The raw additional families included substantial noise or redundancy relative to the sample size.

### 4.1.3 Trimming The Sequence Families

The next logical move was ablation-driven trimming. The project narrowed the sequence set to:

#### Trimmed `seq15m` features

- `seq15m_pressure_total`
- `seq15m_pressure_turnover_total`
- `seq15m_longest_active_streak`
- `seq15m_recency_weighted_pressure_total`
- `seq15m_active_minute_share`
- `seq15m_late_burst_count`
- `seq15m_shots_same_or_next_min_after_pressure`

#### Trimmed `seqpos` features

- `seqpos_strict_link_rate`
- `seqpos_mean_shots_per_possession`
- `seqpos_possessions_with_shots_share`
- `seqpos_pressure_shot_escalation_share`
- `seqpos_forward_pressure_shot_share`
- `seqpos_linked_pressure_total`
- `seqpos_mean_runs_per_possession`

This selection was not arbitrary. It came from two sources:

- signal concentration in model importance tables
- better football interpretability than the noisier `seqextra_*` family

The trimmed ablation results for `XGBoost` were:

| Feature set | Mean PR AUC | Mean Balanced Accuracy | Mean ROC AUC | Mean Brier Score |
|---|---:|---:|---:|---:|
| `base_context` | `0.1161` | `0.5541` | `0.6424` | `0.1073` |
| `base_plus_seq15m_trimmed` | `0.1111` | `0.5456` | `0.6230` | `0.1005` |
| `base_plus_seqpos_trimmed` | `0.1049` | `0.5473` | `0.6202` | `0.1020` |
| `base_plus_trimmed` | `0.1048` | `0.5595` | `0.6251` | `0.0951` |

These results matter for the paper because they show a nuanced outcome:

- trimming clearly helped compared with the full sequence block
- however, trimmed sequence features still did not beat `base_context` on PR AUC
- the combined trimmed model slightly improved balanced accuracy, but not the primary ranking metric

This means the sequence features were directionally useful, but the initial representation and model family were not yet extracting their value efficiently.

This ablation pattern is visualized directly in [sequence_xgb_ablation.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/sequence_xgb_ablation.png).

### 4.2 Why An RNN Was Rejected

At this point, an obvious idea would have been to try RNN, GRU, or LSTM models. That path was explicitly rejected as the main next step.

The reason is methodological, not stylistic:

- the current dataset is not a raw ordered action tensor
- it is a structured checkpoint table with engineered summaries
- an RNN would therefore consume a representation that has already collapsed most temporal order
- the model would inherit sequence complexity without receiving true sequence granularity

This is why the project pivoted to architectures that are natively stronger on interaction-rich tabular data.

### 4.3 Dual-Branch Neural Benchmark

The next notebook, `sequencing_dual_branch_nn.ipynb`, asked a more targeted question:

> Can a custom neural architecture extract useful nonlinear interactions by treating base context and trimmed sequence features as separate information streams?

The notebook compared:

- `xgb_base_context`
- `xgb_base_plus_trimmed`
- `dual_branch_residual_mlp`
- `ft_transformer_lite`

### 4.3.1 Dual-Branch Residual MLP Design

The dual-branch architecture was deliberately aligned with the semantic structure of the data:

- branch 1:
  base-context features plus categorical embeddings
- branch 2:
  trimmed sequence features only
- fusion:
  concatenation followed by a small dense prediction head
- regularization:
  dropout
  residual blocks
  weighted BCE loss

This architecture was motivated by a reasonable hypothesis: sequence features may not help when mixed naively with all other variables, but may become more useful when given a dedicated representation branch before fusion.

### 4.3.2 Results Of The Dual-Branch Benchmark

The neural comparison results were:

| Model | Mean PR AUC | Mean Balanced Accuracy at 0.5 | Mean ROC AUC | Mean Brier Score | Tuned Threshold | OOF Balanced Accuracy at Tuned Threshold |
|---|---:|---:|---:|---:|---:|---:|
| `ft_transformer_lite` | `0.1210` | `0.5724` | `0.6253` | `0.2442` | `0.50` | `0.5731` |
| `xgb_base_context` | `0.1060` | `0.5587` | `0.6300` | `0.1194` | `0.21` | `0.6021` |
| `xgb_base_plus_trimmed` | `0.1039` | `0.5507` | `0.6127` | `0.1018` | `0.07` | `0.5913` |
| `dual_branch_residual_mlp` | `0.1019` | `0.6127` | `0.6440` | `0.2377` | `0.49` | `0.6182` |

The substantive conclusion is very clear:

- the dual-branch residual MLP did not improve PR AUC enough to justify replacing the stronger baselines
- however, it did produce the strongest balanced accuracy after threshold tuning
- the FT-transformer-style model was the first neural architecture to improve PR AUC over the tree benchmark

This result is exactly the kind of methodological story a paper should tell: the first custom neural design improved classification balance but not ranking, whereas the transformer-style tabular architecture improved the ranking objective that mattered most.

The cross-model comparison is summarized in [neural_benchmark_comparison.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/neural_benchmark_comparison.png), and the full development arc from sequence ablation to neural modeling is shown in [sequence_nn_evolution.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/sequence_nn_evolution.png).

### 4.4 Why The FT-Transformer Worked Better

The FT-transformer-style model was a better fit than the dual-branch MLP for the following reasons.

### 4.4.1 Tokenized Feature Interactions

The model treats features as tokens rather than as one undifferentiated dense vector. This is attractive in the present setting because the signal is likely to depend on:

- interactions between context and recent form
- interactions among different sequence summaries
- local combinations such as pressure intensity plus possession linkage plus finishing context

These are exactly the kinds of relationships that self-attention can model more flexibly than a shallow feed-forward network.

### 4.4.2 Better Handling Of Mixed Data Types

The checkpoint rows combine:

- continuous aggregates
- sparse categorical context
- nonlinear rare-event signals

FT-style architectures are explicitly designed for this mixed tabular regime. That is a better match to the current representation than either a naive MLP or an RNN.

### 4.4.3 Rare-Event Learning

The contest target is strongly imbalanced. The FT-transformer line is therefore being extended with:

- BCE-based variants
- focal-loss variants
- Poisson/intensity-based rare-event variants

This is an appropriate escalation because once the model family is structurally aligned with the data, the next performance gains are more likely to come from better rare-event optimization than from indiscriminate architectural complexity.

### 4.5 FT-Transformer Search And Poisson/Intensity Framing

The heavier search notebook, `ft_transformer_search.ipynb`, was created to exploit the available compute budget in a disciplined way rather than through ad hoc experimentation.

It does three important things:

1. It scales up the FT-transformer capacity:
   - larger token widths
   - deeper encoders
   - more attention heads
2. It compares rare-event objectives:
   - BCE
   - focal loss
   - Poisson-style intensity loss
3. It separates internal interpretation from public contest reporting:
   - public output:
     later-goal probability
   - internal Poisson interpretation:
     scoring intensity `lambda`

This is particularly important for the research paper, because it allows the model to remain aligned with the contest question while still supporting an intensity-based football interpretation.

For Poisson variants, the logic is:

- the model predicts `log(lambda)`
- `lambda` is interpreted as post-checkpoint scoring intensity
- reported goal probability is:
  - `P(score later) = 1 - exp(-lambda)`

This is the correct way to use a Poisson-like rare-event formulation here. It allows the model to remain a probability predictor for evaluation while giving a coherent latent-process interpretation for RQ2 and the mechanism-oriented questions.

### 4.5.1 Search Results

The completed FT-transformer search produced the following ranking:

| Model | Loss | Token width | Layers | Heads | Mean PR AUC | Mean Balanced Accuracy at 0.5 | Mean ROC AUC | Mean Brier Score | Tuned Threshold | OOF Balanced Accuracy at Tuned Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ft_wide_poisson` | Poisson | 64 | 3 | 8 | `0.1262` | `0.5059` | `0.6359` | `0.0551` | `0.06` | `0.5939` |
| `ft_deep_focal` | focal | 48 | 4 | 6 | `0.1250` | `0.6129` | `0.6369` | `0.2178` | `0.48` | `0.6176` |
| `ft_wide_focal` | focal | 64 | 3 | 8 | `0.1177` | `0.5422` | `0.6312` | `0.2417` | `0.45` | `0.5553` |
| `ft_repro_bce` | BCE | 32 | 2 | 4 | `0.1038` | `0.5876` | `0.6346` | `0.2404` | `0.44` | `0.5989` |
| `ft_wide_bce` | BCE | 64 | 3 | 8 | `0.1028` | `0.5986` | `0.6050` | `0.2270` | `0.41` | `0.6006` |
| `ft_deep_poisson` | Poisson | 48 | 4 | 6 | `0.0914` | `0.5000` | `0.5708` | `0.1044` | `0.05` | `0.5272` |

Three conclusions are especially important.

First, the best PR AUC in the whole executed sequence-aware neural line is now achieved by `ft_wide_poisson` with mean PR AUC `0.1262`. This is the strongest current answer to the contest’s primary predictive question.

Second, the best threshold-tuned balanced accuracy among the FT-search models is achieved by `ft_deep_focal` at `0.6176`, almost matching the dual-branch residual MLP while also preserving much stronger ranking quality.

Third, the search validates the broader modeling hypothesis: once the representation is compact and sequence-aware, better interaction modeling and rare-event objectives matter more than simply adding more handcrafted features.

The complete FT-search comparison is visualized in [ft_transformer_search_comparison.png](/Users/jan/Documents/competitions/hackatons/WEC_26/plots/ft_transformer_search_comparison.png).

## 5. Design Logic And Justification

The entire evolution of the neural line was driven by explicit failures or partial successes in the previous stage.

### 5.1 Why Build The Sequence-Aware Table First

Because the available event information had already been aggregated into a checkpoint-aligned wide dataset, the first requirement was to see whether structured sequence summaries add value at all.

### 5.2 Why Trim The Sequence Families

The full sequence block hurt performance. That made trimming mandatory. The purpose of trimming was not merely regularization; it was to retain the most interpretable football signals while removing low-value or noisy expansions, especially from `seqextra_*`.

### 5.3 Why Move From XGBoost To Custom Neural Models

The trimmed tree benchmarks still left open the possibility that sequence information was useful but difficult to extract through standard additive tree splits. A custom architecture was therefore justified.

### 5.4 Why Dual-Branch First

The dual-branch residual MLP was a semantically honest neural test. It asked whether base context and sequence information should be represented separately before fusion. This was a controlled step up in complexity.

### 5.5 Why FT-Transformer Next

The dual-branch MLP improved balanced accuracy but not ranking quality. That suggested that the main remaining challenge was feature interaction learning, not merely nonlinear fusion. The FT-transformer was therefore a stronger next step than an even deeper MLP.

### 5.6 Why Poisson/Intensity Variants

The outcome is rare, and the research questions are naturally compatible with an event-rate story. Poisson-based variants therefore offer a principled way to ask whether the model learns latent scoring intensity better than plain direct probability classification.

## 6. Results And Research-Question Interpretation

### 6.1 RQ1: How Well Can Player Behavior Predict Later Scoring?

This modeling line gives the project its strongest answer to RQ1 so far.

Key results:

- best tabular sequence-stage result:
  - `XGBoost` on `base_context`
  - mean PR AUC = `0.1161`
- best neural result from the initial neural benchmark:
  - `FTTransformerLite`
  - mean PR AUC = `0.1210`
- best result from the completed FT-transformer search:
  - `ft_wide_poisson`
  - mean PR AUC = `0.1262`

Thus, player on-field behavior measured at checkpoints does predict later scoring above trivial baselines, and the updated evidence indicates that a stronger transformer-style neural model can exploit this structure better than both the current tree benchmark and the earlier lighter transformer baseline.

### 6.2 RQ2: Which Aspects Of Player Behavior Determine Scoring Probability?

This is where the sequence-aware line becomes especially valuable.

The trimmed features that survived ablation are not random technical artifacts. They encode specific football mechanisms:

- pressure volume and pressure-induced turnovers
- active-streak persistence
- recency-weighted pressure signal
- possession strictness and linkage
- shot escalation within possessions
- forward-pressure shot share
- run and shot volume per possession

These features support an interpretable football story:

- scoring risk is associated not only with raw shot or sprint totals
- it is also associated with how pressured actions connect to possessions, how activity clusters temporally, and whether possessions escalate toward shots

The Poisson/intensity framing strengthens this interpretation further because it allows these features to be described as drivers of latent scoring intensity rather than only direct classifiers of a binary outcome. The fact that `ft_wide_poisson` is now the best PR AUC model makes that interpretation substantively relevant rather than merely theoretical.

### 6.3 RQ3: Are Sprints And Shots Alone Sufficient?

The sequence-aware results argue against a “sprints and shots alone are sufficient” interpretation.

Why:

- the base checkpoint model works reasonably well
- but additional structured pressure and possession features contain meaningful football information
- even when they do not immediately improve PR AUC in a naive full-block expansion, the trimmed and neuralized sequence line indicates that richer contextualized attacking-process features matter

Therefore, RQ3 should likely be answered cautiously: sprints and shots are useful, but not sufficient for the richest predictive and interpretive account.

### 6.4 RQ4: Does Passing Data And Pressure Behavior Help?

The sequence-aware line strongly supports the importance of pressure-linked and possession-linked information, even though the raw full feature expansion was too noisy.

The trimmed sequence features that remain most credible are dominated by:

- pressure intensity
- pressure-induced turnover structure
- possession linkage
- possession escalation toward shots

This suggests that the answer to RQ4 is yes, but with an important qualifier:

> passing and pressure information helps when represented in a compact, behaviorally structured way; it does not help when added as a broad, noisy feature block without careful selection.

### 6.5 RQ5: Does Short-Term Or Cumulative Performance Matter More?

The sequence-aware line points toward the importance of recent local structure rather than only cumulative totals.

Evidence:

- many retained sequence signals are explicitly recent-window constructs
- `seq15m_*` features survived trimming more cleanly than broad extra composition features
- features like active streak, recency-weighted pressure, and shots near pressure are inherently short-horizon signals

This suggests that the short-term patterning of behavior contributes meaningfully beyond static cumulative state.

### 6.6 RQ6: Does Elevated Recent Intensity Relative To Overall Level Matter?

The sequence-aware modeling line is especially relevant here.

Features such as:

- recency-weighted pressure totals
- active-minute share
- late burst count
- longest active streak

directly operationalize the idea that local bursts or concentrated activity matter. Even when these features do not universally dominate the benchmark on their own, their survival through trimming indicates that relative short-term intensity is a plausible component of later scoring risk.

### 6.7 RQ7: Do External/Context Factors Matter?

Yes, and the neural line confirms this in two ways:

1. Context variables are retained in every meaningful benchmark.
2. Neural models that embed categorical context and let it interact with sequence summaries are more plausible final models than sequence-only designs.

The dual-branch and FT-transformer architectures both assume that context and behavior should be jointly represented rather than separated analytically.

## 7. Discussion

The current sequence-aware and neural evidence supports a balanced conclusion:

- sequence features are real and behaviorally meaningful
- full unfiltered sequence expansion is too noisy
- compact sequence families are more defensible
- the first custom neural architecture improved classification balance but not ranking
- the FT-transformer family is the strongest executed current direction for improving the primary metric
- within that family, a medium-width Poisson/intensity model currently gives the best ranking performance, while a deeper focal-loss model gives the strongest threshold-tuned class balance

In other words, the project has already learned something important:

> the bottleneck is not whether sequence information matters, but how that sequence information is represented and how its interactions are modeled.

## 8. Limitations

Several limitations should be stated clearly in the paper.

### 8.1 Representation Is Still Aggregated

Even the neural line is learning from aggregated sequence summaries, not raw event chains. This limits how strongly one can interpret the model as a genuine event-sequence learner.

### 8.2 Small-Sample Rare-Event Setting

The dataset is not large. With only `3,486` rows and around `5.8%` positives, high-capacity neural models are vulnerable to instability and metric variance.

### 8.3 Calibration Remains Weaker In Neural Models

The current neural models have relatively poor Brier scores compared with the tree baselines. That does not invalidate them, but it means the final probability layer should be handled carefully if the paper emphasizes probabilistic decision support.

## 9. Paper-Ready Takeaway

For the paper, the strongest concise statement is:

> Sequence-aware modeling improved the conceptual fidelity of the football representation, but naive inclusion of all engineered sequence features degraded performance. Ablation revealed that the most credible signals were concentrated in compact pressure- and possession-oriented summaries, especially those describing pressure intensity, possession linkage, escalation toward shots, and temporally concentrated activity. A custom dual-branch residual MLP improved balanced accuracy after threshold tuning but did not improve the primary ranking metric. By contrast, the FT-transformer family consistently provided the strongest ranking results, and the current best executed model is a Poisson/intensity-based FT-transformer configuration (`ft_wide_poisson`, mean PR AUC `0.1262`). This indicates that the main remaining modeling gain lies in learning higher-order interactions among context, recent form, and compact sequence summaries, while also treating later scoring as a rare-event intensity process rather than only a direct binary classification target.

If a shorter result paragraph is needed:

> The best executed neural result is now obtained by the Poisson-based FT-transformer configuration `ft_wide_poisson`, which reaches mean PR AUC `0.1262`, above both the earlier `FTTransformerLite` benchmark (`0.1210`) and the best earlier XGBoost sequence benchmark (`0.1161`). The strongest threshold-tuned balanced accuracy within the FT-search is delivered by `ft_deep_focal` (`0.6176`), while the dual-branch residual MLP remains competitive on class-balance-oriented decision performance. Overall, the evidence suggests that compact sequence features are useful when modeled with an architecture that can represent complex tabular interactions, and that rare-event-aware objectives can further improve the answer to the primary predictive question.
