# Event Aggregation Specification

This document implements the point 8 aggregation deliverable from `docs/plan.md`.

## Objective

Define a leakage-safe and reproducible method for turning event tables into checkpoint-level features aligned with the main modeling table.

Unit of output:

- one row per `player_appearance_id` and checkpoint in `players_quarters_final.csv`

## Time Alignment Rules

### 1. Continuous Match Minute

Convert event timestamps to a continuous match minute:

- `half_1`: `0 + minute`
- `half_2`: `45 + minute`
- `extra_time_1`: `90 + minute`
- `extra_time_2`: `105 + minute`

Convert checkpoints to continuous minute:

- `H1_15` -> `15`
- `H1_30` -> `30`
- `H1_45` -> `45`
- `H2_15` -> `60`
- `H2_30` -> `75`
- `H2_45` -> `90`
- `ET1_15` -> `105`

### 2. Causal Windows

For a checkpoint time `t`:

- cumulative window: `event_cont_min <= t`
- last-15 window: `t - 15 < event_cont_min <= t`

These definitions are mandatory. No event after checkpoint time may enter any feature.

## Join Key

Events are joined to checkpoint rows by:

- `player_appearance_id`

If an event family has no rows for a given checkpoint window:

- count-like features become `0`
- rate/share features become `0`

This keeps the representation dense and model-friendly while preserving the meaning of “no events observed”.

## Approved Aggregates

### Pass Table

Source: `player_appearance_pass.csv`

For each window:

- `pass_count`: number of pass events
- `pass_accurate_rate`: mean of `accurate`
- `pass_top_share`: share of passes with `stage == "top"`
- `pass_middle_share`: share of passes with `stage == "middle"`

### Pressure Table

Source: `player_appearance_behaviour_under_pressure.csv`

For each window:

- `pressure_count`: number of pressure-behavior events
- `pressure_accurate_rate`: mean of `accurate`
- `pressure_turnover_rate`: share with `press_induced_outcome == "turnover"`
- `pressure_top_share`: share with `stage == "top"`
- `pressure_pass_angle_observed_share`: share with non-null `pass_angle`

### Run Table

Source: `player_appearance_run.csv`

For each window:

- `run_count`: number of run events
- `run_distance_sum`: sum of `distance`
- `run_peak_speed`: max of `max_speed`
- `run_mean_max_speed`: mean of `max_speed`
- `run_sprint_share`: share with `run_type == "sprint"`
- `run_top_share`: share with `stage == "top"`

### Shot Table

Source: `player_appearance_shot_limited.csv`

Pre-filter:

- exclude own-goal perspective rows by keeping only rows with null `own_goal_player_appearance_id`

For each window:

- `shot_count`: number of non-own-goal shots
- `shot_top_share`: share with `stage == "top"`
- `shot_under_pressure_rate`: mean of `under_pressure`
- `shot_regular_play_share`: share with `play_pattern == "regular_play"`

Optional later only if sparsity remains acceptable:

- `shot_blocked_share`

## Forbidden Aggregates

- any feature using shot `outcome`
- any feature using future events after checkpoint time
- any full-match rank, percentile, z-score, or normalization computed using post-checkpoint information
- any aggregate that mixes train and validation fixtures during preprocessing

## Naming Convention

Use the window prefix plus the aggregate name:

- `cumul_pass_count`
- `last15_pass_accurate_rate`
- `cumul_pressure_turnover_rate`
- `last15_run_peak_speed`
- `cumul_shot_under_pressure_rate`

This keeps supplementary families distinct from the current base-table variables.

## Validation Checks

Every implementation should verify:

1. the event timestamp mapping to continuous minute is correct,
2. no event with `event_cont_min > checkpoint_cont_min` is included,
3. row counts are reproducible on rerun,
4. zero-event windows are encoded consistently,
5. feature support rates remain within the ranges observed in Block C.

## Intended Use In Point 12

These aggregates are the approved starting specification for Feature Engineering Iteration 2.
They should be merged to the checkpoint table only after the causal windowing step is complete.
