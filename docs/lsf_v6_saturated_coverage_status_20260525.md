# LSF v6 Saturated Coverage Status - 2026-05-25

## Summary

LSF v6 implemented a saturated hard-query coverage objective for large-edit
solver-coverage coreset exports. The intended fix was to stop v5-512 from
over-serving already covered queries and hurting hard scenes.

The implementation is functional and audited, but the q80/q160 result does not
clear the method-level bar. v6 reduces the worst v5-512 OldHospital q80
overshoot, but it still underperforms v5-128 raw macro accuracy. The guarded
drop-protection follow-up also does not recover the loss.

Conclusion: v6 is useful negative evidence for the current selection-only
objective. It should not replace v5-128 as the main result. The next method
sprint should move beyond sampled-index editing toward solver-weighted landmark
feature construction and dense correspondence verification.

## Code Changes

- `loc_gs/stdloc_native/solver_coverage_coreset.py`
  - Added coverage saturation modes: `none`, `native_percentile`,
    `native_fraction`.
  - Added under-covered tail query boosting via `tail_cvar_alpha`.
  - Added `hard_query_min_gain` for under-covered query gain filtering.
  - Added v6.1 drop-side protection with
    `saturation_drop_protection_weight`.
- `loc_gs/scripts/export_lsf_solver_aware_map.py`
  - Added explicit v6 CLI arguments and manifest hyperparameters.
- Tests:
  - `tests/test_solver_coverage_coreset.py`
  - `tests/test_export_lsf_solver_aware_map.py`

Targeted test status before evaluation:

```text
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_solver_coverage_coreset.py \
  tests/test_export_lsf_solver_aware_map.py

22 passed in 32.49s
```

## Artifacts

Map exports:

```text
output/stdloc/map_cambridge_spgs_lsf_v6_saturated512_20260525
output/stdloc/map_cambridge_spgs_lsf_v6_guarded512_20260525
```

Evaluation roots:

```text
output/stdloc_native/lsf_v6_saturated512_spgs_prior_20260525
output/stdloc_native/lsf_v6_guarded512_spgs_prior_20260525
```

Gate reports:

```text
output/reports/lsf_v6_saturated512_spgs_prior_20260525/q80_precision_acceptance.json
output/reports/lsf_v6_saturated512_spgs_prior_20260525/q160_precision_acceptance.json
output/reports/lsf_v6_guarded512_spgs_prior_20260525/q80_precision_acceptance.json
output/reports/lsf_v6_guarded512_spgs_prior_20260525/q160_precision_acceptance.json
```

All guarded map manifests passed the structural audit:

| Scene | Sampled | Edits | Split Audit | Saturation | Tail Q | Safe Drops |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| GreatCourt | 16384 | 411 | passed | native_percentile | 7 | 0 |
| KingsCollege | 16384 | 206 | passed | native_percentile | 7 | 0 |
| OldHospital | 16384 | 512 | passed | native_percentile | 7 | 0 |
| ShopFacade | 16384 | 296 | passed | native_percentile | 7 | 0 |
| StMarysChurch | 16384 | 433 | passed | native_percentile | 7 | 0 |

## Raw Train-Dev Results

Metrics are dense-stage deltas against the same native SPGS-prior baseline.
Negative `median_te_cm` is better. Recall deltas are absolute fractions.

### q80

| Variant | Macro TE Delta | Wins | R@10 Delta | R@5 Delta | R@2 Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5-128 | -0.099655 cm | 4/5 | +0.0100 | +0.0025 | +0.0075 |
| v5-512 | +0.122703 cm | 3/5 | -0.0025 | +0.0050 | +0.0075 |
| v6-sat512 | +0.056509 cm | 3/5 | +0.0025 | +0.0050 | +0.0075 |
| v6-guarded512 | +0.056509 cm | 3/5 | +0.0025 | +0.0050 | +0.0075 |

Per-scene dense TE deltas:

| Scene | v5-128 | v5-512 | v6-sat512 | v6-guarded512 |
| --- | ---: | ---: | ---: | ---: |
| GreatCourt | -0.3548 | -0.2673 | -0.2311 | -0.2311 |
| KingsCollege | +0.2463 | +0.2665 | +0.2841 | +0.2841 |
| OldHospital | -0.0164 | +0.8970 | +0.4963 | +0.4963 |
| ShopFacade | -0.1612 | -0.1232 | -0.1076 | -0.1076 |
| StMarysChurch | -0.2122 | -0.1594 | -0.1592 | -0.1592 |

### q160

| Variant | Macro TE Delta | Wins | R@10 Delta | R@5 Delta | R@2 Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5-128 | -0.119556 cm | 3/5 | +0.0000 | -0.0025 | +0.0100 |
| v5-512 | -0.091075 cm | 3/5 | -0.00125 | -0.00125 | +0.00875 |
| v6-sat512 | +0.020673 cm | 2/5 | -0.0000 | +0.0000 | +0.0100 |
| v6-guarded512 | +0.024726 cm | 2/5 | -0.0000 | +0.0000 | +0.0100 |

Per-scene dense TE deltas:

| Scene | v5-128 | v5-512 | v6-sat512 | v6-guarded512 |
| --- | ---: | ---: | ---: | ---: |
| GreatCourt | -0.3548 | -0.4048 | -0.2716 | -0.2716 |
| KingsCollege | -0.0151 | -0.0141 | -0.0141 | -0.0141 |
| OldHospital | -0.2562 | -0.3156 | +0.1191 | +0.1191 |
| ShopFacade | +0.0142 | +0.0169 | +0.0325 | +0.0325 |
| StMarysChurch | +0.0141 | +0.2623 | +0.2375 | +0.2577 |

## Gate Results

The precision-primary scene-level gate still passes because it can select native
for scenes where the candidate regresses. This is a reconstruction-time
diagnostic gate, not a per-query branch selector and not sufficient as a main
method claim.

| Report | Gate Macro TE Delta | Selected Candidate Scenes |
| --- | ---: | --- |
| v5-128 q80 | -0.148925 cm | GreatCourt, OldHospital, ShopFacade, StMarysChurch |
| v5-128 q160 | -0.125219 cm | GreatCourt, KingsCollege, OldHospital |
| v5-512 q80 | -0.109980 cm | GreatCourt, ShopFacade, StMarysChurch |
| v5-512 q160 | -0.146908 cm | GreatCourt, KingsCollege, OldHospital |
| v6-sat512 q80 | -0.099576 cm | GreatCourt, ShopFacade, StMarysChurch |
| v6-sat512 q160 | -0.057146 cm | GreatCourt, KingsCollege |
| v6-guarded512 q80 | -0.099576 cm | GreatCourt, ShopFacade, StMarysChurch |
| v6-guarded512 q160 | -0.057146 cm | GreatCourt, KingsCollege |

## Interpretation

v6 addressed part of the failure mode but not the main bottleneck:

- OldHospital q80 improved relative to v5-512 (`+0.8970 cm` -> `+0.4963 cm`),
  but remains a clear regression against native and much worse than v5-128.
- StMarysChurch q160 remains regressed under large edits.
- The drop-protection follow-up changes sampled sets, but it does not produce
  useful metric movement.
- This confirms that large static sampled-index editing is not enough. The
  bottleneck is still in matching representation and dense residual acceptance,
  not only in query coverage allocation.

## Decision

Do not promote v6-sat512 or v6-guarded512 as the next main recipe.

Keep v6 in the codebase because it provides a tested saturation controller and a
negative result for large static coreset editing. For the mainline, retain
v5-128 as the stronger selection-only baseline and start the next sprint on:

1. solver-weighted landmark feature fusion;
2. dense match verification instead of dense locability prior;
3. only then revisit large-edit support selection with the stronger
   representation.
