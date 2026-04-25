# Feature Extension Plan

This plan implements the point 8 deliverable from `docs/plan.md` and is aligned with the exploratory outputs from `notebooks/eda_block_c_event_table_extension_feasibility.ipynb`.

## Decision Summary

- Extend the feature space first with pass, pressure, and run aggregates.
- Add shot-derived features selectively because shot density is materially lower than the other event families.
- Keep all event features checkpoint-causal using the `last15` and `cumul` windows defined in `docs/event_aggregation_specification.md`.
- Reuse the fixed development/holdout split artifacts; no event engineering should touch holdout rows during iteration.

## Coverage-Based Feasibility

Development-partition checkpoint coverage from the implemented Block C analysis:

| Family | Cumulative coverage | Last-15 coverage | Inclusion decision |
|---|---:|---:|---|
| pass | 98.41% | 96.75% | include |
| pressure | 92.79% | 82.08% | include |
| run | 96.47% | 93.22% | include |
| shot | 28.83% | 15.05% | selective only |

Interpretation:

- Pass, pressure, and run all have enough density to support both volume and composition-rate features.
- Shot data is informative but sparse. It should be treated as a compact additive family, not a large feature expansion block.

## Recommended Feature Families

### 1. Pass Features

Recommended in both `cumul` and `last15` windows:

- `pass_count`
- `pass_accurate_rate`
- `pass_top_share`
- `pass_middle_share`

Rationale:

- Coverage is near-complete.
- These features are interpretable and stable.
- They complement the current main table, which does not yet expose passing volume or directional field-zone mix.

### 2. Pressure Features

Recommended in both `cumul` and `last15` windows:

- `pressure_count`
- `pressure_accurate_rate`
- `pressure_turnover_rate`
- `pressure_top_share`
- `pressure_pass_angle_observed_share`

Rationale:

- Pressure data remains dense enough even in the last-15 window.
- These features directly address the contest theme around behavior under pressure.
- `pressure_turnover_rate` is especially attractive because it is interpretable and behaviorally specific.

### 3. Run Features

Recommended in both `cumul` and `last15` windows:

- `run_count`
- `run_distance_sum`
- `run_peak_speed`
- `run_mean_max_speed`
- `run_sprint_share`
- `run_top_share`

Rationale:

- Run coverage is high and stable.
- These features preserve the main sprint/intensity signal while adding richer spatial and composition information.
- They remain close to football meaning and should be easy to communicate in the paper.

### 4. Shot Features

Recommended selectively:

- `shot_count`
- `shot_top_share`
- `shot_under_pressure_rate`
- `shot_regular_play_share`

Do not prioritize initially:

- `shot_blocked_share`
- any lower-support subtype shares beyond the four listed above

Rationale:

- Shot sparsity is acceptable for a compact family but too low for a broad subtype expansion.
- The retained features are the least sparse and the most interpretable.
- Outcome-dependent shot variables remain forbidden because shot outcome is redacted and target-linked.

## Implementation Order

1. Add pass aggregates.
2. Add pressure aggregates.
3. Add run aggregates not already represented in the main table.
4. Add the compact shot family last.
5. Evaluate incremental gain family by family under the fixed grouped CV protocol.

## Exclusion Rules

- Do not create aggregates that depend on shot outcome.
- Do not create full-match or post-checkpoint event summaries.
- Do not explode sparse categorical event values into wide one-hot count matrices in the first extension pass.
- Do not add highly granular subtype shares unless support is demonstrated to be stable across checkpoints.

## Success Criterion For Moving To Feature Engineering Iteration 2

Proceed with the event-table extension when:

- the aggregation code reproduces checkpoint-causal windows exactly,
- the added families remain sufficiently dense in the development partition,
- and each family can be evaluated separately in an ablation against the baseline main-table model.
