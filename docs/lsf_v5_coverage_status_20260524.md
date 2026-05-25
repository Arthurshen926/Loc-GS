# LSF v5 Solver-Coverage Coreset Status - 2026-05-24

## Scope

This round implements the ChatGPT-new.md mainline as a solver-level support
selector rather than another unary matchability selector.

- Method: same-budget LSF solver-coverage coreset.
- Real query path: matching -> fixed poselib PnP -> STDLoc dense refinement.
- Eval cfg: `third_party/stdloc/configs/stdloc_spgs_cambridge_dense1.yaml`.
- Split: Cambridge train-dev (`eval_split=train`) with `q80` and `q160`
  meaning `max_test_cameras=80` and `max_test_cameras=160`.
- Candidate maps:
  - `output/stdloc/map_cambridge_spgs_lsf_v5_coverage_20260524`
  - `output/stdloc/map_cambridge_spgs_lsf_v5_coverage512_20260524`
- Eval outputs:
  - `output/stdloc_native/lsf_v5_coverage_spgs_prior_20260524`
  - `output/stdloc_native/lsf_v5_coverage512_spgs_prior_20260524`

## Implemented

- Added `loc_gs.stdloc_native.solver_coverage_coreset`.
- Added exporter policy `--selection_policy coverage`.
- Added hard guardrail `--require_candidate_pool`.
- Manifest now records `candidate_pool_path`, `candidate_pool_required`, and
  `selection_policy`.
- Coverage selector uses hard-query candidate gain/source loss from solver
  admissibility artifacts, with dense-worsen/ambiguity penalties from the same
  thresholds.
- Drop search is bounded and audited by `drop_cost_evaluations`; it preserves
  same-budget output and safe-core landmarks.

## Map Audit

All five v5 128-edit maps passed:

- sampled count: 16384 -> 16384
- split audit: passed
- policy: `solver_coverage_coreset`
- candidate pool: explicit `candidate_pool_lsf_v2_native_top4096.pt`
- safe-core dropped count: 0

The 512-capacity variant also passed map audit. Actual admissible replacements:

| Scene | v5-512 admissible edits |
| --- | ---: |
| GreatCourt | 391 |
| KingsCollege | 198 |
| OldHospital | 512 |
| ShopFacade | 292 |
| StMarysChurch | 426 |

## Raw Results

Dense median translation error deltas are candidate minus baseline in cm.
Negative is better.

### v5 128

| Split | GreatCourt | KingsCollege | OldHospital | ShopFacade | StMarysChurch | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q80 | -0.3548 | +0.2463 | -0.0164 | -0.1612 | -0.2122 | -0.0997 |
| q160 | -0.3548 | -0.0151 | -0.2562 | +0.0142 | +0.0141 | -0.1196 |

Recall deltas:

| Split | R@10 | R@5 | R@2 |
| --- | ---: | ---: | ---: |
| q80 | +0.0100 | +0.0025 | +0.0075 |
| q160 | +0.0000 | -0.0025 | +0.0100 |

### v5 512

| Split | GreatCourt | KingsCollege | OldHospital | ShopFacade | StMarysChurch | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q80 | -0.2673 | +0.2665 | +0.8970 | -0.1232 | -0.1594 | +0.1227 |
| q160 | -0.4048 | -0.0141 | -0.3156 | +0.0169 | +0.2623 | -0.0911 |

The 512-capacity variant is not monotonic. It improves q160 GreatCourt and
OldHospital more than 128, but it regresses q80 OldHospital and q160
StMarysChurch. This means larger edit capacity needs an explicit saturation or
tail-risk constraint before it can become the main recipe.

## Precision-Primary Gate

Gate policy:

- median TE regression is a hard rejection;
- recall drops are warnings;
- no per-query branch selection;
- selected map is decided at reconstruction/eval-run level.

Reports:

- `output/reports/lsf_v5_coverage_spgs_prior_20260524/q80_precision_acceptance.json`
- `output/reports/lsf_v5_coverage_spgs_prior_20260524/q160_precision_acceptance.json`
- `output/reports/lsf_v5_coverage512_spgs_prior_20260524/q80_precision_acceptance.json`
- `output/reports/lsf_v5_coverage512_spgs_prior_20260524/q160_precision_acceptance.json`

Gate-selected macro dense TE deltas:

| Recipe | q80 | q160 |
| --- | ---: | ---: |
| v5 128 | -0.1489 | -0.1252 |
| v5 512 | -0.1100 | -0.1469 |

Current best fixed train-dev interpretation:

- q80: use v5 128. It selects v5 on GreatCourt, OldHospital, ShopFacade, and
  StMarysChurch; KingsCollege stays native.
- q160: v5 512 has the stronger gate macro, but only because ShopFacade and
  StMarysChurch stay native. v5 128 is more stable raw all-v5.

## Current Claim Status

Positive and defensible train-dev claim:

> Solver-level feedback can select same-budget 3DGS support that improves
> STDLoc-style dense pose precision on Cambridge train-dev without changing the
> evaluator or using per-query branches.

What is still not ready:

- Not a Cambridge test claim.
- Not yet a SOTA claim.
- 512-capacity result shows the method still lacks a robust coverage saturation
  controller.

## Next Method-Level Fix

The next improvement should not be another scalar threshold sweep. The observed
failure mode is capacity overshoot: once enough hard-query support is covered,
additional edits can damage scene-specific geometry.

Recommended v6 direction:

1. Add query-tail saturation to the coreset objective.
   Stop adding support for a query once its normalized solver support exceeds a
   percentile target from native source coverage.
2. Add scene-level edit budgeting from admissible coverage mass.
   The 512 cap should become an upper bound, not a target.
3. Add sparse-anchor preservation as a solver term, not detector score only.
   Penalize drops that reduce native PnP inlier diversity even when dense support
   looks good.
4. Keep dense LSF as a residual-support teacher only.
   It should shape dense refinement support, not become the sparse selector.

This keeps the main claim aligned with localization feedback guided feature
selection/reconstruction, while addressing the current non-monotonic behavior.
