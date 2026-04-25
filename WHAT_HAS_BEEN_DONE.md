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
- Displayed a reduced preview with:
  - `player_appearance_id`
  - `checkpoint`
  - `minutes_in_game`
  - `minute_in_window`
  - `minute_out_window`

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
