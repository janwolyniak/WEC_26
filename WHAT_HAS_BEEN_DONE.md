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

## KNN-specific clean dataset export

- Added a dedicated KNN export path built from the latest `clean_df`.
- Created `scored_after_eval_key` so the target remains traceable to the correct:
  - match
  - checkpoint
  - player
- Built this key from:
  - `fixture_id`
  - `checkpoint`
  - `player_id`
- Created a compact `knn_target_lookup` view inside the notebook to verify that the key-to-target mapping is unique.
- Added a duplicate-key validation step that raises an error if `scored_after_eval_key` is not unique.
- Built `knn_df` as a KNN-ready dataset and dropped columns requested for removal:
  - `player_appearance_id`
  - `date`
  - `minute_out_window`
  - `fixture_id`
- Kept `scored_after_eval_key` in the KNN dataset so predictions can later be joined back to the correct target row for accuracy checks.
- Stored the result as `tables["knn_dataset.csv"]`.
- Updated the notebook export step so it now saves:
  - `players_quarters_final.csv`
  - `players_quarters_final_step1.csv`
  - `knn_dataset.csv`
- Displayed a reduced preview with:
  - `player_appearance_id`
  - `checkpoint`
  - `minutes_in_game`
  - `minute_in_window`
  - `minute_out_window`

## Notebook cleanup for KNN-only scope

- Simplified `notebooks/data_cleaning_feature_selection_general.ipynb` so it now focuses only on building the final KNN dataset.
- Removed the broader intake, inventory, missingness, exploratory summaries, and intermediate export flow from the notebook.
- Kept only the logic needed to build the final KNN feature set:
  - checkpoint-window compression
  - `minutes_in_game`
  - pass-based last-15 features
  - cumulative pass features
  - `scored_after_eval_key`
- Made the final KNN column list explicit in the notebook so the output schema is easier to inspect and control.
- Reduced the notebook outputs to a minimal export summary and a preview of the final KNN table.
- Narrowed the export behavior so the notebook now writes only `data/knn_dataset.csv`.

## Bug fix for direct minute-window mapping

- Fixed the cleaned KNN notebook so it no longer expects engineered source columns like `minute_in_H1_15`.
- Replaced the old checkpoint-compression helper with direct mapping from raw `minute_in` values into `minute_in_window`.
- Removed the unused `minute_out_window` creation step from the slim KNN notebook.
- This makes the notebook compatible with the current `data/players_quarters_final.csv` schema, which only contains raw `minute_in` and `minute_out`.

## New sequencing dataset builder

- Added `notebooks/sequencing_dataset_builder.ipynb`.
- Built the new sequencing dataset starting from `data/knn_dataset.csv` rather than from the broader modeling tables.
- Reused the previously generated geometry-aware sequence feature artifact:
  - `artifacts/sequential_goal_forecasting_geometry/sequence_feature_table.csv`
- Reconstructed the row link from `players_quarters_final.csv` using:
  - `player_id`
  - `fixture_id`
  - `checkpoint`
  - `scored_after_eval_key`
- Merged the geometry sequence features onto the KNN base through `scored_after_eval_key`.

### Additional event-derived sequencing features added

- From `player_appearance_behaviour_under_pressure.csv`:
  - `seqextra_pass_angle_observed_count`
  - `seqextra_pass_angle_mean_deg`
  - `seqextra_pass_angle_median_deg`
  - `seqextra_pass_angle_std_deg`
  - `seqextra_pass_angle_last_deg`
- From `player_appearance_run.csv`:
  - `seqextra_run_distinct_possessions`
  - `seqextra_run_possession_transition_count`
  - `seqextra_run_last_stage`
  - `seqextra_run_stage_bottom_share`
  - `seqextra_run_stage_middle_share`
  - `seqextra_run_stage_top_share`
- From `player_appearance_shot_limited.csv`:
  - `seqextra_shot_last_body_part`
  - `seqextra_shot_last_technique`
  - `seqextra_shot_last_play_pattern`
  - body-part share features
  - technique share features
  - play-pattern share features

### Sequencing window and export behavior

- Computed the extra event-derived summaries over the same last-15-minute window ending at each checkpoint.
- Dropped helper linkage columns before final export:
  - `player_appearance_id`
  - `fixture_id`
  - `checkpoint_cont_min`
- Configured the notebook to export the final merged table to:
  - `data/sequencing_dataset.csv`

## Pass-based feature engineering

### Absolute minute and checkpoint mapping

- Copied `player_appearance_pass.csv` into `pass_df`.
- Defined `period_offsets`:
  - `half_1` -> `0`
  - `half_2` -> `45`
  - `extra_time_1` -> `90`
  - `extra_time_2` -> `105`
- Created `minute_abs` as:
  - `minute + mapped period offset`
- Implemented `map_absolute_minute_to_checkpoint(minute_abs)` to assign each pass event to one checkpoint window.
- Created a checkpoint column in `pass_df` using that mapping.
- Filtered pass events so only checkpoints present in the target quarter-level dataset were retained.

### Last-15 checkpoint pass features

- Aggregated pass events by `player_appearance_id` and `checkpoint` to create:
  - `last15_pass_passed`
  - `last15_pass_passed_accurate`
- Aggregated received pass events by `addressee_player_appearance_id` and `checkpoint` to create:
  - `last15_pass_received`
  - `last15_pass_received_accurate`
- Renamed the addressee key back to `player_appearance_id` for merging.
- Left-joined both pass aggregations into `clean_df`.
- Filled missing pass-feature values with zero.
- Cast all pass features to `int64`.
- Stored the updated table again as `tables["players_quarters_final_step1.csv"]`.
- Displayed:
  - a row preview
  - descriptive statistics for the checkpoint pass features

### Exact pass features added to `clean_df`

- Added `last15_pass_received`
- Added `last15_pass_passed`
- Added `last15_pass_received_accurate`
- Added `last15_pass_passed_accurate`

## Cumulative pass features by player

- Built `checkpoint_order_for_cumsum` from the defined checkpoint order restricted to checkpoints actually present in `clean_df`.
- Defined the cumulative feature mapping:
  - `last15_pass_passed` -> `cumul_pass_passed`
  - `last15_pass_passed_accurate` -> `cumul_pass_passed_accurate`
  - `last15_pass_received` -> `cumul_pass_received`
  - `last15_pass_received_accurate` -> `cumul_pass_received_accurate`
- Added helper ordering columns:
  - `_row_order`
  - ordered categorical `_checkpoint_order`
- Sorted rows by player and checkpoint order.
- Computed cumulative sums grouped by `player_appearance_id`.
- Renamed the cumulative outputs using the target feature names.
- Concatenated cumulative features back onto the modeling table.
- Restored original row order and dropped the temporary ordering columns.
- Saved the result back to `tables["players_quarters_final_step1.csv"]`.
- Displayed:
  - a row preview of current and cumulative pass features
  - descriptive statistics for the cumulative pass features

### Exact cumulative features added to `clean_df`

- Added `cumul_pass_passed`
- Added `cumul_pass_passed_accurate`
- Added `cumul_pass_received`
- Added `cumul_pass_received_accurate`

## Exact feature inventory added by the notebook

- Temporary engineered checkpoint indicators:
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
- Final retained Step-1 features:
  - `minutes_in_game`
  - `minute_in_window`
  - `minute_out_window`
  - `last15_pass_received`
  - `last15_pass_passed`
  - `last15_pass_received_accurate`
  - `last15_pass_passed_accurate`
  - `cumul_pass_passed`
  - `cumul_pass_passed_accurate`
  - `cumul_pass_received`
  - `cumul_pass_received_accurate`
- Explicitly removed after the replacement step:
  - `checkpoint_period`
  - `checkpoint_min`
  - `minute_in`
  - `minute_out`
- Temporary checkpoint indicator features were also dropped after the compressed replacement columns were created.

## Export

- Exported the final Step-1 modeling table to:
  - `data/players_quarters_final_step1.csv`
- Printed the save path and final shape after export.
## 2026-04-25

- Added notebook [sequential_goal_forecasting_geometry.ipynb](/Users/jan/Documents/competitions/hackatons/WEC_26/notebooks/sequential_goal_forecasting_geometry.ipynb) to implement the first sequence-oriented goal-forecasting extension.
- The notebook loads `players_quarters_final`, `player_appearance_behaviour_under_pressure`, `player_appearance_run`, and `player_appearance_shot_limited`, reconstructs continuous event time with the existing checkpoint-causal mapping, and builds both:
  - a `minute_hybrid` last-15-minute ordered sequence view
  - a strict `possession_sequence` view with unmatched pressure retained as residual checkpoint summaries
- Exported reproducible artifacts to `artifacts/sequential_goal_forecasting_geometry/`:
  - `sequence_feature_table.csv`
  - `minute_sequence_long.csv`
  - `possession_sequence_long.csv`
  - `validation_checks.csv`
  - `feature_support.csv`
  - `metrics_table.csv`
  - `fold_metrics_table.csv`
  - `interpretation_table.csv`
  - `subgroup_stability_table.csv`
- Implemented validation checks confirming:
  - no `seq15m` rows use post-checkpoint events
  - no `seqpos` rows use post-checkpoint events
  - sequence joins preserve checkpoint-row cardinality
  - ambiguous pressure-to-possession matches are excluded from direct possession attribution
- Implemented grouped development-fold evaluation with ablations for:
  - base only
  - base + `minute_hybrid`
  - base + `possession_sequence`
  - base + both
- Included interpretation and subgroup-stability outputs for the geometry-aware sequence feature families.
- Added a second pass inside `sequential_goal_forecasting_geometry.ipynb` for a trimmed supported-core sequence subset.
- The trimmed subset keeps 10 higher-support `seq15m_...` features and 10 higher-support `seqpos_...` features, and exports the selection rationale to `artifacts/sequential_goal_forecasting_geometry/trimmed_feature_selection.csv`.
- Trimmed ablation results improved over the full sequence families:
  - `logreg + base_plus_minute_hybrid_trimmed` reached balanced accuracy `0.5711` vs `0.5624` for the full minute-hybrid family
  - `logreg + base_plus_possession_sequence_trimmed` reached balanced accuracy `0.5647` vs `0.5608` for the full possession family
  - `rf + base_plus_both_trimmed` achieved the strongest tree-based sequence variant with ROC AUC `0.6574` and PR AUC `0.1245`
- The baseline `logreg + base_only` model still remains the best balanced-accuracy reference at `0.5831`, so the current interpretation is that sequence features are promising but not yet net-positive enough to replace the simpler baseline.
- Added a third pass inside `sequential_goal_forecasting_geometry.ipynb` for redundancy-pruned lean sequence subsets and exported the rationale to `artifacts/sequential_goal_forecasting_geometry/lean_feature_selection.csv`.
- The leaner set keeps 5 minute-hybrid features and 5 possession-sequence features after removing highly correlated pairs such as:
  - `seqpos_mean_run_distance_per_possession` vs `seqpos_max_run_distance_per_possession`
  - `seq15m_pressure_total` vs `seq15m_recency_weighted_pressure_total`
  - `seq15m_angle_concentration` vs `seq15m_pressure_angle_availability_rate`
  - `seqpos_mean_runs_per_possession` vs `seqpos_mean_possession_span`
- Leaner ablation results further improved the sequence track:
  - `logreg + base_plus_possession_sequence_leaner` reached balanced accuracy `0.5806`, nearly matching the base-only logistic benchmark `0.5831`
  - `logreg + base_plus_minute_hybrid_leaner` reached balanced accuracy `0.5750`
  - `logreg + base_plus_both_leaner` reached balanced accuracy `0.5681`
  - `rf + base_plus_possession_sequence_leaner` achieved the strongest tree balanced accuracy among sequence variants at `0.5141`
  - `rf + base_plus_both_leaner` achieved the best lean combined ranking metrics with ROC AUC `0.6575` and PR AUC `0.1303`
- The lean interpretation table is now dominated by a smaller, more stable set of effects:
  - possession run-distance intensity
  - pressure volume under pressure
  - pressure-induced turnovers
  - possession count / runs-per-possession structure
  - unmatched-pressure share
- Added a fold-safe hyperparameter sweep for the `base_plus_possession_sequence_leaner` logistic track inside `sequential_goal_forecasting_geometry.ipynb`.
- Exported new tuning artifacts to `artifacts/sequential_goal_forecasting_geometry/`:
  - `tuned_possession_tuning_table.csv`
  - `tuned_possession_oof_predictions.csv`
  - `tuned_possession_interpretation_table.csv`
  - `tuned_possession_subgroup_stability_table.csv`
- The best tuned configuration is:
  - penalty = `l1`
  - C = `0.10`
  - class_weight = `balanced`
- This tuned possession-only logistic model improved the lean possession track from balanced accuracy `0.5806` to `0.6006`, which now exceeds the earlier base-only logistic benchmark `0.5831`.
- Tuned possession-only interpretation is now sparse and stable:
  - strongest positive effect: `seqpos_mean_run_distance_per_possession`
  - strongest negative effect: `seqpos_unmatched_pressure_share`
  - smaller positive effects remain for `seqpos_linked_pressure_total` and `seqpos_possession_count`
- Added SHAP-based interpretation for the tuned possession-only logistic model so the notebook now explicitly answers RQ2 from `docs/WEC2026_Problem_description.md`.
- Exported new RQ2 / SHAP artifacts to `artifacts/sequential_goal_forecasting_geometry/`:
  - `tuned_possession_shap_table.csv`
  - `tuned_possession_shap_bar.png`
  - `tuned_possession_shap_beeswarm.png`
- The SHAP determinant ranking confirms the compact behavioural answer to RQ2:
  - strongest positive determinant: `seqpos_mean_run_distance_per_possession` (possession run-distance intensity)
  - strongest negative determinant: `seqpos_unmatched_pressure_share` (pressure events that do not map cleanly to possessions)
  - weaker positive determinants: `seqpos_linked_pressure_total` and `seqpos_possession_count`
  - near-zero determinant after L1 shrinkage: `seqpos_mean_runs_per_possession`
- The sequence notebook now covers both:
  - RQ1 via grouped predictive metrics and ablation comparisons
  - RQ2 via tuned sparse-model coefficients plus SHAP-based behavioural importance and directionality
- Extended the RQ2 block in `notebooks/sequential_goal_forecasting_geometry.ipynb` to incorporate leakage-safe variables coming from `notebooks/data_cleaning_feature_selection_general.ipynb` / `data/players_quarters_final_step1.csv`.
- Included these additional RQ2 variables in the expanded behavioural model:
  - `minute_in_window`
  - `last15_pass_passed`
  - `last15_pass_passed_accurate`
  - `last15_pass_received`
  - `last15_pass_received_accurate`
  - `cumul_pass_passed`
  - `cumul_pass_passed_accurate`
  - `cumul_pass_received`
  - `cumul_pass_received_accurate`
- Explicitly excluded leakage-prone Step-1 variables from the expanded RQ2 model:
  - `subbed`
  - `minutes_in_game`
  - `minute_out_window`
- Added an expanded sparse logistic tuning pass for RQ2 using lean possession-sequence features plus the leakage-safe Step-1 variables.
- Exported new expanded-RQ2 artifacts to `artifacts/sequential_goal_forecasting_geometry/`:
  - `rq2_expanded_tuning_table.csv`
  - `rq2_step1_shap_table.csv`
  - `rq2_step1_shap_bar.png`
  - `rq2_step1_shap_beeswarm.png`
  - `rq2_expanded_interpretation_table.csv`
  - `rq2_expanded_subgroup_stability_table.csv`
- The best expanded RQ2 configuration is:
  - penalty = `l1`
  - C = `0.10`
  - class_weight = `balanced`
- Best expanded RQ2 validation metrics:
  - balanced accuracy = `0.5956`
  - ROC AUC = `0.6194`
  - PR AUC = `0.1016`
  - Brier score = `0.2136`
- This expanded model does not beat the tuned possession-only model on balanced accuracy (`0.6006`), but it gives a broader behavioural answer to RQ2 by combining possession-sequence structure with pass-behaviour variables from the general cleaning notebook.
- SHAP on the added Step-1 variables shows the strongest extra behavioural effects come from:
  - `cumul_pass_passed`: strongest added negative determinant
  - `cumul_pass_received_accurate`: strong added positive determinant
  - `last15_pass_received_accurate`: positive recent receiving signal
  - `last15_pass_received`: negative recent receiving-volume signal in the tuned sparse model
- The combined expanded interpretation table is now led by:
  - `cumul_pass_passed`
  - `seqpos_mean_run_distance_per_possession`
  - `cumul_pass_received_accurate`
  - `last15_pass_received_accurate`
  - `last15_pass_received`
- Replaced the leaky full-match `minutes_in_game` concept in the expanded RQ2 notebook block with a checkpoint-safe cumulative playtime variable:
  - `minutes_in_game_to_checkpoint`
  - definition: `checkpoint_cont_min - minute_in + 1`
  - source: merged from `players_quarters_final`, where the same quantity already exists as `minutes_available_before_checkpoint`
- Kept the raw Step-1 `minutes_in_game` excluded, because it still encodes full-match exposure beyond the checkpoint.
- Re-ran `notebooks/sequential_goal_forecasting_geometry.ipynb` with this cumulative playtime feature included in the expanded sparse logistic RQ2 model and in the Step-1 SHAP analysis.
- Updated expanded RQ2 results after adding checkpoint-safe playtime:
  - best config remains `penalty=l1`, `C=0.10`, `class_weight=balanced`
  - balanced accuracy = `0.5923`
  - ROC AUC = `0.6157`
  - PR AUC = `0.1022`
  - Brier score = `0.2156`
- The new cumulative playtime feature is retained by the sparse model:
  - `minutes_in_game_to_checkpoint` coefficient = `-0.1779`
  - SHAP rank among the added Step-1 variables = `5`
  - interpretation in this multivariable setting: more minutes already played by the checkpoint is associated with a lower later scoring probability, conditional on the other possession and passing variables in the model
- Audited the movement-derived variables after identifying implausible computer-vision outputs:
  - `last15_distance` has a maximum of `8028.31` meters in a 15-minute window
  - `29` checkpoint rows exceed `5.5 km` in `last15_distance`
  - `last15_peak_speed` reaches about `50.9 km/h`
- Treated the whole distance/speed family as unreliable for modeling:
  - removed base features `last15_distance`, `last15_mean_max_speed`, `last15_peak_speed`, `cumul_distance`, `cumul_mean_max_speed`, `cumul_peak_speed`
  - removed sequence features derived from run distance / speed, including possession run-distance summaries
  - kept non-distance run structure features such as run counts and stage shares
- Updated `notebooks/sequential_goal_forecasting_geometry.ipynb` so the active RQ1/RQ2 analysis now excludes these movement-magnitude variables entirely.
- This exclusion surfaced a preprocessing issue in the notebook: `SimpleImputer(fill_value=0.0)` failed once the remaining numeric block became mostly integer-valued, so it was corrected to `fill_value=0`.
- Re-ran the full sequence notebook after the exclusion.
- Updated results on the cleaned feature space:
  - `logreg + base_only` balanced accuracy = `0.5776`
  - `logreg + base_plus_possession_sequence_leaner` balanced accuracy = `0.5756`
  - the best expanded RQ2 configuration is now `penalty=l1`, `C=0.03`, `class_weight=balanced`
  - expanded RQ2 balanced accuracy = `0.6029`
  - expanded RQ2 ROC AUC = `0.6191`
  - expanded RQ2 PR AUC = `0.1006`
- The tuned possession-only interpretation changed materially after removing distance/speed variables:
  - strongest remaining negative effect: `seqpos_unmatched_pressure_share`
  - smaller positive effects: `seqpos_linked_pressure_total`, `seqpos_possession_count`
  - `seqpos_mean_runs_per_possession` shrinks to zero
- The expanded RQ2 story also changed:
  - strongest determinant is now `minutes_in_game_to_checkpoint`
  - next nonzero added determinant is `cumul_pass_passed`
  - most other added pass variables shrink to zero in the sparse fit
- Confirmed how `minutes_in_game` was created in `notebooks/data_cleaning_feature_selection_general.ipynb`:
  - it is deterministically defined as `minute_out + 1 - minute_in`
  - so it is “fixed” by construction in the Step-1 table
  - but it is a full-match exposure variable, not checkpoint-safe, which is why the notebook continues to use `minutes_in_game_to_checkpoint` instead
- Updated `notebooks/data_cleaning_feature_selection_general.ipynb` so the cleaned checkpoint datasets themselves now enforce the same movement-data exclusion and no-leakage playtime rule.
- In the cleaning notebook:
  - dropped distance/speed-derived columns from `players_quarters_final.csv` before downstream feature engineering
  - changed `minutes_in_game` in `players_quarters_final_step1.csv` from full-match exposure to checkpoint-cumulative exposure
  - new definition: `checkpoint_cont_min - minute_in + 1`, clipped at `0`
  - exported both cleaned datasets back to disk from the notebook
- Re-executed `notebooks/data_cleaning_feature_selection_general.ipynb` and rewrote:
  - `data/players_quarters_final.csv`
  - `data/players_quarters_final_step1.csv`
- Post-run validation confirms:
  - `players_quarters_final.csv` contains no columns with `distance` or `speed` in the name
  - `players_quarters_final_step1.csv` contains no columns with `distance` or `speed` in the name
  - `players_quarters_final_step1.csv` retains `minutes_in_game`, but it is now checkpoint-cumulative and leakage-safe
  - exact validation result: `0` mismatches against the formula derived from `players_quarters_final.csv`
- Current cleaned dataset shapes after the rewrite:
  - `players_quarters_final.csv`: `(3486, 43)`
  - `players_quarters_final_step1.csv`: `(3486, 34)`
- Added a dedicated RQ3-RQ7 implementation path:
  - `src/python/rq3_rq7_ablation.py`
  - `notebooks/rq3_rq7_ablation_matrix.ipynb`
- The new module uses the frozen grouped split file `data/splits/modeling_row_folds.csv` as the modeling base so all RQ3-RQ7 experiments stay aligned with the fixed development/holdout protocol.
- Implemented leakage-safe checkpoint-aligned event aggregation for:
  - pass events from `player_appearance_pass.csv`
  - pressure events from `player_appearance_behaviour_under_pressure.csv`
- Added deterministic derived feature blocks for the RQ matrix:
  - context/exposure variables including `minutes_available_before_checkpoint`
  - history-only carry-over features for count-like base variables
  - relative-intensity features comparing recent activity against cumulative level
- Implemented frozen-CV logistic ablations for:
  - RQ3: direct shots-and-sprints sufficiency vs broader base behavior
  - RQ4: incremental value of pass and pressure aggregates
  - RQ5: `last15` vs `cumul` vs combined vs historical carry-over
  - RQ6: added value and directionality of relative-intensity features
  - RQ7: context-only vs behavior-only vs context-augmented models
- Configured the new workflow to export compact paper-facing artifacts under `artifacts/rq3_rq7_ablation/`, including:
  - mean CV result tables
  - fold metrics
  - RQ4 deltas
  - RQ5 temporal comparison
  - RQ6 relative-intensity coefficient ranking
  - RQ7 context permutation-importance ranking

## KNN pipeline notebook

- Added a new executed notebook at `notebooks/knn_pipeline.ipynb`.
- Built the notebook directly on top of `data/knn_dataset.csv` instead of regenerating features.
- Added dataset validation checks for:
  - required columns
  - unique `scored_after_eval_key`
  - binary `scored_after`
- Derived `group_id` from the first segment of `scored_after_eval_key` so evaluation stays grouped at the match-like level.
- Defined explicit feature roles for:
  - numeric predictors
  - categorical predictors
  - boolean predictors
- Implemented a leakage-safe sklearn pipeline with:
  - median imputation + standard scaling for numeric features
  - most-frequent imputation + one-hot encoding for categorical features
  - boolean-to-integer conversion for boolean features
  - `KNeighborsClassifier` as the final estimator
- Used grouped and stratified 5-fold cross-validation via `StratifiedGroupKFold`.
- Reported the following metrics for every run:
  - PR AUC
  - ROC AUC
  - balanced accuracy
  - Brier score
- Added a baseline KNN run and a compact hyperparameter search over:
  - `n_neighbors`
  - `weights`
  - Minkowski distance with `p in {1, 2}`
- Ranked the search results by mean PR AUC to match the rare-positive classification setup.
- The executed notebook currently selects `n_neighbors=31`, `weights="distance"`, `p=1` as the best tested configuration.
- Best mean cross-validated metrics in the current notebook run:
  - PR AUC: `0.0963`
  - ROC AUC: `0.6152`
  - balanced accuracy: `0.5000`
  - Brier score: `0.0558`

## KNN notebook extension: ID-drop and threshold tuning

- Extended `notebooks/knn_pipeline.ipynb` only by appending new cells; the original notebook cells were left unchanged.
- Added a follow-up KNN experiment that drops the ID-like distance features:
  - `player_id`
  - `jersey_number`
- Rebuilt the reduced feature-role table inside the appended cells so the modified experiment is explicit and local to the notebook extension.
- Reran the same compact KNN hyperparameter search on the reduced feature set.
- The best reduced-feature configuration is now:
  - `n_neighbors=31`
  - `weights="distance"`
  - `p=2`
- Best reduced-feature mean cross-validated metrics:
  - PR AUC: `0.0967`
  - ROC AUC: `0.6251`
  - balanced accuracy: `0.5000`
  - Brier score: `0.0555`
- Added a threshold-tuning extension based on out-of-fold probabilities from the reduced-feature best KNN model.
- Evaluated thresholds from `0.05` to `0.30` and ranked them by balanced accuracy.
- Best threshold in the current notebook run:
  - threshold: `0.05`
  - balanced accuracy: `0.6098`
  - precision: `0.0820`
  - recall: `0.7143`
  - predicted positive rate: `0.5075`
- This confirms that the default `0.5` threshold was too conservative for the rare-positive target and that threshold tuning materially improves classification sensitivity.
