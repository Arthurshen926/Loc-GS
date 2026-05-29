# LSF v6 Saturated Coverage Coreset Design

## Goal

Fix the v5 coverage coreset non-monotonic failure mode where larger edit
capacity improves some scenes but overshoots hard scenes such as OldHospital and
StMarysChurch.

## Problem

v5 ranks candidates by hard-query coverage gain, but it keeps assigning positive
value to queries that are already sufficiently covered by the native source map.
The 512-capacity run confirmed the failure mode:

- q160 GreatCourt and OldHospital improved more than v5-128;
- q80 OldHospital regressed by `+0.8970 cm`;
- q160 StMarysChurch regressed by `+0.2623 cm`.

This is not a scalar threshold issue. It is an objective issue: the selector
needs a saturation target and a hard-tail controller.

## Method

LSF v6 keeps the same paper-safe output contract as v5:

- one fixed same-budget map per scene;
- explicit candidate pool;
- no test feedback;
- no per-query branch selection;
- fixed STDLoc-compatible inference path.

The selector changes only the coverage objective.

For each hard query, compute native source coverage from the current sampled set.
Then build a saturation target:

- `none`: v5 behavior, no cap;
- `native_percentile`: one global cap from a percentile of native hard-query
  coverage;
- `native_fraction`: per-query cap from a fraction of each query's native
  coverage.

Candidate positive gain above the cap receives zero or low marginal value.
Queries in the lowest coverage-ratio tail receive a bonus controlled by
`tail_cvar_alpha`. A candidate can also be required to add at least
`hard_query_min_gain` to under-covered queries before it is considered.

## Exporter Interface

Add explicit v6 parameters to `export_lsf_solver_aware_map.py`:

```text
--coverage_saturation_mode none|native_percentile|native_fraction
--coverage_saturation_percentile 0.75
--coverage_saturation_fraction 1.0
--easy_query_gain_decay 0.0
--tail_cvar_alpha 0.0
--tail_query_gain_boost 1.0
--hard_query_min_gain 0.0
```

Defaults preserve v5 behavior.

## Evaluation Plan

Main v6 recipe:

```text
--selection_policy coverage
--max_edits 512
--coverage_saturation_mode native_percentile
--coverage_saturation_percentile 0.75
--tail_cvar_alpha 0.1
--tail_query_gain_boost 1.0
--hard_query_min_gain 0.0
```

Evaluate Cambridge train-dev q80/q160 against the same SPGS-prior fixed poselib
cfg used in v5.

## Acceptance Criteria

- All v6 maps remain same-budget 16384 and split-audit passed.
- OldHospital q80 does not repeat the v5-512 overshoot.
- StMarysChurch q160 does not repeat the v5-512 overshoot.
- q80 and q160 macro dense median TE are both non-worse than v5-128 raw macro,
  or the precision-primary gate improves over v5-128 on at least one split
  without making the other split worse.
- Recall drops remain warnings, not hard acceptance criteria.
