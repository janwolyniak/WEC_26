# `notebooks/data_cleaning_feature_selection_general.ipynb`

## General intake and scope

- Set up the notebook as the shared starting point for the cleaning and feature-selection workflow.
- Defined the current scope:
  - load the five contest tables from `data/`
  - keep a raw view for special missing-token checks
  - build a normalized view with the same rules as `src/python/data_intake.py`
  - summarize missing values at dataset and column level
  - expose column names and dtypes for later linking and feature-selection decisions
- Explicitly marked the following as not yet in scope:
  - imputation or dropping rules
  - dataset joins
  - final feature shortlist decisions

## Project setup and shared intake logic

- Imported the project intake module dynamically from `src/python/data_intake.py`.
- Resolved the project root by searching upward until both `data/` and `src/` were found.
- Reused the central intake definitions from `src/python/data_intake.py`:
  - `DATASET_FILES`
  - `NA_TOKENS`
  - `normalize_dataframe`
- Configured notebook display defaults and silenced warnings.

## Raw and normalized dataset loading

- Defined `TRACKED_MISSING_TOKENS` as:
  - empty string
  - `NULL`
  - `null`
  - `None`
  - `none`
  - `N/A`
  - `n/a`
  - `NA`
  - `na`
  - `nan`
- Created `load_raw_table(path)` to read CSVs with `keep_default_na=False` so raw missing tokens stay visible.
- Created `load_normalized_table(path)` to:
  - read CSVs with `na_values=NA_TOKENS`
  - apply `normalize_dataframe(df)`
- Loaded all contest files twice:
  - `raw_tables` for raw token inspection
  - `tables` for normalized downstream work

## Dataset inventory

- Built `dataset_inventory(name, df)` to summarize each dataset with:
  - dataset name
  - row count
  - column count
  - total missing cells
  - percent missing
  - duplicate row count
- Combined the per-dataset summaries into a sorted `inventory` table and displayed it.

## Raw missing-token scan

- Implemented `raw_missing_token_summary(df)` to scan every column for the tracked missing tokens.
- Counted token occurrences after trimming string values.
- Returned a sorted summary with:
  - `column`
  - `token`
  - `count`
- Ran this scan for every raw dataset and displayed either:
  - the token-count summary, or
  - a message that no tracked raw missing tokens were found

## Normalized missingness and schema overview

- Implemented `column_missing_summary(df)` to summarize every normalized column with:
  - column name
  - dtype
  - non-null count
  - missing count
  - missing percent
  - number of unique non-null values
- Built `dataset_missing_overview` for each normalized dataset with:
  - number of columns containing missing values
  - number of rows containing missing values
  - total missing cells
- Displayed:
  - a dataset-level normalized missingness overview
  - a column-level schema and missingness report for each dataset

## Checkpoint analysis for `players_quarters_final.csv`

- Focused on `players_quarters_final.csv` for checkpoint-based minute analysis.
- Grouped rows by `checkpoint`.
- Computed grouped statistics for `minute_in` and `minute_out`:
  - row count
  - min
  - median
  - mean
  - max
- Rounded grouped averages and medians to two decimals.

## Engineered checkpoint-window features for `minute_in` and `minute_out`

- Defined checkpoint windows:
  - `H1_15` = minutes 1 to 15
  - `H1_30` = minutes 16 to 30
  - `H1_45` = minutes 31 to 45
  - `H2_15` = minutes 46 to 60
  - `H2_30` = minutes 61 to 75
  - `H2_45` = minutes 76 to 90
  - `ET1_15` = minutes 91 to 105
  - `ET2_15` = minutes 106 to 120
- Created binary indicator columns for every window for both:
  - `minute_in`
  - `minute_out`
- Added all engineered minute-window columns back into `tables["players_quarters_final.csv"]`.
- Built a feature summary for the engineered columns showing:
  - feature name
  - active row count
  - zero row count
  - unique values
- Built an outside-window summary to count rows whose `minute_in` or `minute_out` did not fall into any defined checkpoint window.
- Displayed a preview including original minute columns and a subset of the engineered indicators.

### Exact engineered features added at this stage

- Added the following binary `minute_in` checkpoint features:
  - `minute_in_H1_15`
  - `minute_in_H1_30`
  - `minute_in_H1_45`
  - `minute_in_H2_15`
  - `minute_in_H2_30`
  - `minute_in_H2_45`
  - `minute_in_ET1_15`
  - `minute_in_ET2_15`
- Added the following binary `minute_out` checkpoint features:
  - `minute_out_H1_15`
  - `minute_out_H1_30`
  - `minute_out_H1_45`
  - `minute_out_H2_15`
  - `minute_out_H2_30`
  - `minute_out_H2_45`
  - `minute_out_ET1_15`
  - `minute_out_ET2_15`

## Shared-column map for future linking

- Implemented `build_column_presence_map(table_dict)` across all loaded tables.
- Built a table showing:
  - every unique column name across datasets
  - whether the column exists in each dataset
  - in how many datasets the column appears
- Sorted this output by `dataset_count` and column name.
- Displayed it as a preparation step for later linking decisions.

## Step 1: Replace legacy `minute_in` and `minute_out`

- Defined the intention of Step 1:
  - treat `df` as the imported target dataset
  - treat `clean_df` as the modeling copy
  - replace raw minute columns with denser checkpoint-based representations

### Compression of checkpoint indicators

- Created `minute_window_order` from the checkpoint-window keys.
- Created `minute_window_to_idx` mapping each checkpoint window to an index starting from 1.
- Implemented `compress_checkpoint_features(df, base_column)` to:
  - inspect the engineered binary checkpoint columns for a base column
  - compress each row into a single checkpoint-window label
  - calculate diagnostics for rows with zero active windows
  - calculate diagnostics for rows with multiple active windows
- Ran the compression for:
  - `minute_in`
  - `minute_out`
- Built and displayed `compression_diagnostics`.

### Replacement modeling columns

- Copied the checkpoint dataset into `clean_df`.
- Created `minutes_in_game` as:
  - `minute_out + 1 - minute_in`
- Created ordered categorical replacements:
  - `minute_in_window`
  - `minute_out_window`
- Dropped legacy raw timing columns:
  - `checkpoint_period`
  - `checkpoint_min`
  - `minute_in`
  - `minute_out`
- Stored this intermediate modeling table as `tables["players_quarters_final_step1.csv"]`.
- Summarized missingness and uniqueness for:
  - `minutes_in_game`
  - `minute_in_window`
  - `minute_out_window`
- Displayed a preview of the replacement features together with selected minute-indicator columns.

### Exact replacement features added to `clean_df`

- Added `minutes_in_game`
- Added `minute_in_window`
- Added `minute_out_window`

### Removal of temporary minute-indicator columns

- Dropped all temporary binary indicator columns for:
  - `minute_in_H1_15`
  - `minute_in_H1_30`
  - `minute_in_H1_45`
  - `minute_in_H2_15`
  - `minute_in_H2_30`
  - `minute_in_H2_45`
  - `minute_in_ET1_15`
  - `minute_in_ET2_15`
  - `minute_out_H1_15`
  - `minute_out_H1_30`
  - `minute_out_H1_45`
  - `minute_out_H2_15`
  - `minute_out_H2_30`
  - `minute_out_H2_45`
  - `minute_out_ET1_15`
  - `minute_out_ET2_15`
- Overwrote `tables["players_quarters_final_step1.csv"]` with the cleaner Step-1 modeling version.

# `notebooks/sequencing_model_pipeline.ipynb`

## Sequence-aware modeling pipeline

- Added a new notebook that uses `data/sequencing_dataset.csv` as the modeling table for sequence-aware goal forecasting.
- Recovered the frozen split metadata for the sequencing table by joining `scored_after_eval_key` back to:
  - `data/splits/modeling_row_folds.csv`
  - `data/splits/fixture_assignments.csv`
- Kept the existing grouped evaluation protocol:
  - development partition only for iteration
  - grouped folds already frozen on disk
  - PR AUC as the main comparison metric
  - ROC AUC and Brier score also reported

## Added feature-engineering layer

- Added a compact second layer of derived features on top of the existing sequence aggregates from `sequencing_dataset.csv`.
- Implemented derived families including:
  - short-term vs cumulative activity ratios for sprints, HSR, and shots
  - shot efficiency and under-pressure rates
  - pass accuracy rates for passed and received actions
  - pressure-to-run, pressure-to-shot, and run-to-shot ratios
  - directional balance features from `seq15m_*` shares
  - recency-weighted activity shares
  - possession-level density and transition-rate features
  - run-stage balance and entropy
  - shot body-part, play-pattern, and technique composition summaries

## Model comparison

- Implemented three model families in the notebook:
  - `LogisticRegression`
  - `XGBoost`
  - a small PyTorch `MLP` benchmark
- Structured the comparison around three feature sets:
  - `base_context`
  - `base_plus_sequence`
  - `full_engineered`
- Explicitly did not make RNN/LSTM/GRU the default path in this notebook because the current sequencing dataset is already a wide checkpoint-level representation rather than a raw event-by-event tensor.

## Outputs written by the notebook

- Configured the notebook to export artifacts to `artifacts/sequencing_model_pipeline/`.
- The notebook writes:
  - fold-level metrics
  - model summary table
  - out-of-fold predictions
  - derived feature list
  - top-feature tables for fitted models
  - a plot for the best model's top features

## Sequence-family trimming and ablation update

- Updated `notebooks/sequencing_model_pipeline.ipynb` to add ablation-driven trimming focused on the strongest `seq15m` and `seqpos` signals.
- Added explicit trimmed sequence registries:
  - `SEQ15M_TRIMMED_FEATURES`
  - `SEQPOS_TRIMMED_FEATURES`
- Added new comparison sets:
  - `base_plus_seq15m_trimmed`
  - `base_plus_seqpos_trimmed`
  - `base_plus_trimmed`
- Added balanced accuracy to the notebook metrics and exported summaries.
- Added a dedicated `xgb_ablation_summary.csv` artifact for quick comparison of the trimmed XGBoost variants.

## Result of trimmed ablation

- Full sequence inclusion remained too noisy for the current table representation.
- Trimming improved the sequence variants materially relative to the full sequence block.
- For XGBoost:
  - `base_context` remained best on PR AUC
  - `base_plus_trimmed` achieved the best balanced accuracy among the XGBoost sequence variants
- This means the trimmed sequence families are much more defensible than the full sequence block, but they still do not beat the plain context baseline on the primary selection metric.

# `notebooks/sequencing_dual_branch_nn.ipynb`

## Neural benchmark follow-up

- Added a new notebook for the next neural modeling step on top of the trimmed sequence setup.
- The notebook benchmarks:
  - `xgb_base_context`
  - `xgb_base_plus_trimmed`
  - `dual_branch_residual_mlp`
  - `ft_transformer_lite`
- Reused the same frozen fold protocol and the same trimmed `seq15m` + `seqpos` feature block.

## Dual-branch architecture

- Implemented a dual-branch residual MLP where:
  - branch 1 receives base context features
  - branch 2 receives trimmed sequence features
  - categorical base-context variables are embedded
  - both branches are fused in a small dense prediction head
- Used weighted BCE loss to address class imbalance.
- Added separate threshold tuning on pooled out-of-fold predictions instead of assuming `0.5` is optimal.

## FT-transformer fallback

- Implemented a lightweight FT-transformer-style fallback in the same notebook so the neural comparison did not stop at the dual-branch MLP.
- The FT-style model tokenizes numeric features and categorical embeddings and passes them through a small transformer encoder before classification.

## Result of neural comparison

- The dual-branch residual MLP improved balanced accuracy, but did not improve PR AUC enough to justify replacing the tree benchmark.
- The FT-transformer-style model achieved the best PR AUC among the models in this notebook.
- Exported artifacts to `artifacts/sequencing_dual_branch_nn/`, including:
  - fold metrics
  - pooled out-of-fold predictions
  - model summary
  - threshold summary

# `notebooks/ft_transformer_search.ipynb`

## FT-transformer search scaffold

- Added a dedicated FT-transformer search notebook for a heavier architecture sweep under the same frozen fold protocol.
- Kept the same trimmed sequence setup used in the earlier neural benchmarks.
- Configured the notebook to export search outputs to `artifacts/ft_transformer_search/`.

## Added objective variants

- Included BCE-based FT-transformer configurations.
- Included focal-loss FT-transformer configurations.
- Added Poisson-based rare-event variants:
  - `ft_wide_poisson`
  - `ft_deep_poisson`

## Poisson formulation

- Implemented the Poisson option as an intensity model rather than a direct binary-probability loss.
- The model predicts log-intensity.
- Training uses Poisson negative log-likelihood on the binary event target.
- Validation and threshold tuning convert intensity to probability using:
  - `p(goal) = 1 - exp(-lambda)`
- This keeps the output compatible with the same PR AUC, ROC AUC, Brier score, and threshold-tuning workflow as the other FT-transformer variants.

## Research-question alignment

- Updated the FT-transformer search notebook so Poisson-based runs are explicitly framed in a way that best supports the contest research questions.
- The notebook now distinguishes between:
  - public output for reporting: later-goal probability
  - internal model interpretation for Poisson variants: scoring intensity `lambda`
- Added notebook-level guidance that Poisson runs should answer the research questions through:
  - `goal_probability = 1 - exp(-lambda)`
  - while still allowing feature effects to be discussed as changes in scoring intensity
- Updated pooled out-of-fold exports so Poisson-capable runs can save:
  - `goal_probability`
  - `log_lambda_rate`
  - `lambda_rate`
  - interpretation mode metadata
- This makes the notebook better suited for RQ1-RQ7 reporting without changing the contest-facing evaluation target.

## Execution status

- The notebook structure and objective options were implemented.
- The search notebook has not been rerun yet after adding the Poisson variants.

# `notebooks/knn_pipeline.ipynb`

## KNN modeling baseline and extensions

- Added and executed a notebook that models `data/knn_dataset.csv` directly.
- The notebook validates:
  - required columns
  - unique `scored_after_eval_key`
  - binary `scored_after`
- The notebook derives `group_id` from the first segment of `scored_after_eval_key` and uses grouped, stratified 5-fold cross-validation.
- Implemented the base KNN pipeline with:
  - median imputation + scaling for numeric features
  - most-frequent imputation + one-hot encoding for categorical features
  - boolean-to-integer conversion for boolean features
- The initial best baseline KNN configuration in the notebook run was:
  - `n_neighbors=31`
  - `weights="distance"`
  - `p=1`
- Baseline mean metrics:
  - PR AUC: `0.0963`
  - ROC AUC: `0.6152`
  - balanced accuracy: `0.5000`
  - Brier score: `0.0558`

## Reduced-feature KNN extension

- Extended the notebook only by appending new cells; the original KNN workflow was left unchanged.
- Added a follow-up KNN experiment that drops the ID-like distance features:
  - `player_id`
  - `jersey_number`
- Rebuilt the reduced feature-role table inside the appended cells so the modified experiment stays local to the extension.
- Reran the compact KNN search on the reduced feature set.
- Best reduced-feature KNN configuration in the notebook run:
  - `n_neighbors=31`
  - `weights="distance"`
  - `p=2`
- Best reduced-feature mean metrics:
  - PR AUC: `0.0967`
  - ROC AUC: `0.6251`
  - balanced accuracy: `0.5000`
  - Brier score: `0.0555`

## Reduced-feature threshold tuning

- Added threshold tuning based on out-of-fold probabilities from the best reduced-feature KNN model.
- Evaluated thresholds from `0.05` to `0.30` and ranked them by balanced accuracy.
- Best reduced-feature threshold in the notebook run:
  - threshold: `0.05`
  - balanced accuracy: `0.6098`
  - precision: `0.0820`
  - recall: `0.7143`
  - predicted positive rate: `0.5075`
- This confirmed that the default `0.5` threshold is too conservative for the rare-positive target.

## PCA before KNN extension

- Extended the notebook again by appending a PCA-based KNN block after the reduced-feature experiment.
- Added a PCA pipeline that runs:
  - the reduced-feature preprocessing
  - dense float conversion
  - an additional scaling step
  - PCA
  - KNN classification
- Tested PCA component counts:
  - `8`
  - `12`
  - `16`
  - `20`
  - `24`
- Reused the compact KNN hyperparameter search on top of the PCA representation.
- Best PCA-extended KNN configuration in the notebook run:
  - `n_components=12`
  - `n_neighbors=31`
  - `weights="distance"`
  - `p=1`
- Best PCA mean metrics:
  - PR AUC: `0.1079`
  - ROC AUC: `0.6026`
  - balanced accuracy: `0.5000`
  - Brier score: `0.0559`
- Relative to the reduced non-PCA KNN:
  - PR AUC improved from `0.0967` to `0.1079`
  - ROC AUC decreased from `0.6251` to `0.6026`
  - Brier score became slightly worse

## PCA threshold tuning

- Added threshold tuning for the best PCA-extended KNN model using out-of-fold probabilities.
- Best PCA threshold in the notebook run:
  - threshold: `0.08`
  - balanced accuracy: `0.5709`
  - precision: `0.0851`
  - recall: `0.4236`
  - predicted positive rate: `0.2900`
- The PCA variant improved PR AUC, but the earlier reduced-feature non-PCA threshold-tuned model still produced the stronger balanced-accuracy result.
