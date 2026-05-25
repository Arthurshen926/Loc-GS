# LSF-Loc Submission Progress, 2026-05-22

This note records the current paper-safety state after switching ULF-Loc to a
paper-reported external reference and adding scene-level fixed-recipe acceptance
reports. All numbers below are Cambridge train-dev diagnostics, not Cambridge
test or SOTA claims.

## Mainline

The active claim remains:

> LSF-Loc learns solver-level localization support from self-localization
> feedback and uses it for reconstruction-time support selection, instead of
> optimizing unary matchability alone.

The real query path remains native STDLoc-compatible: feature matching, OpenCV
PROSAC/RANSAC PnP, then STDLoc dense refinement. The scene-level acceptance
report is a fixed map acceptance policy, not a per-query branch selector.

## ULF-Loc Reference Handling

ULF-Loc is now tracked as a paper-reported external reference, not a local
reproduction blocker. The generated reference is:

- `output/lsf_reports/ulfloc_alignment_reference_paper_reported_20260522.json`

The report explicitly records `code_reproduced=false`,
`reference_mode=paper_reported`, and `reporting_allowed=true`. These numbers
must be labeled as paper-reported and not locally reproduced.

## Fixed Recipe Evidence

Strict q80 scene-level acceptance:

- report: `output/lsf_reports/fixed_recipe_scene_acceptance_q80_20260522.json`
- selected maps: GreatCourt v2, KingsCollege v2, OldHospital native,
  ShopFacade v3, StMarysChurch v3
- macro dense delta vs native: median -1.4095 cm, R@5 +0.0125,
  R@2 +0.0075

Strict q160 scene-level acceptance:

- report: `output/lsf_reports/fixed_recipe_scene_acceptance_q160_20260522.json`
- selected maps: GreatCourt native, KingsCollege native, OldHospital native,
  ShopFacade v3, StMarysChurch native
- macro dense delta vs native: median -0.2470 cm, R@5 +0.0100,
  R@2 +0.00625

The strict q160 recipe is conservative but positive. It accepts only the scene
whose candidate passes all predeclared dense gates.

Aggregate board artifacts:

- `output/lsf_reports/experiment_board_train_dev_q80_q160_20260522.json`
- `output/lsf_reports/experiment_board_train_dev_q80_q160_20260522.md`
- `output/lsf_reports/frozen_recipe_manifest_train_dev_20260522.json`
- `output/lsf_reports/experiment_board_train_dev_q80_q160_profile_20260522.json`
- `output/lsf_reports/experiment_board_train_dev_q80_q160_profile_20260522.md`
- `output/lsf_reports/frozen_profile_coverage_q5_20260522.json`

The board distinguishes artifact audit status from fixed-recipe acceptance.
OldHospital q160 v3 and StMarysChurch q160 v2 are marked `rejected` even though
their LSF feedback split audits pass.

The frozen recipe manifest records both q80 and q160 scene-level map choices,
the fixed gate thresholds, candidate rejection reasons, and single-path
constraints. Its acceptance and split-audit checks pass for the selected
train-dev runs, so the recipe is frozen for a future full-eval launch. The
experiment board still reports `Submit-ready: no` because runtime/memory
reporting is incomplete.

## Full Train Split Validation

The q160 frozen recipe has now been validated on the full Cambridge train
split, still without using Cambridge test results.

- validation report:
  `output/lsf_reports/frozen_recipe_validation_train_full_q160_recipe_20260522.json`
- run root:
  `output/stdloc_native/full_train_q160_recipe_20260522`
- experiment board:
  `output/lsf_reports/experiment_board_full_train_q160_recipe_20260522.json`
- selected maps: GreatCourt native, KingsCollege native, OldHospital native,
  ShopFacade v3, StMarysChurch native
- macro dense delta vs native: median -0.2720 cm, R@5 +0.00952,
  R@2 +0.00519
- validation checks: split audits passed, single-path evaluator, feedback
  disabled at eval, feedback-bank split not test, required sidecars present

The full train split result is positive but conservative. The only promoted LSF
gain is ShopFacade v3:

- dense median translation: 6.6549 cm -> 5.2948 cm
- dense median rotation: 0.3656 deg -> 0.3068 deg
- dense R@5cm/5deg: 0.4459 -> 0.4935
- dense R@2cm/2deg: 0.1385 -> 0.1645

OldHospital v3 full-train remains a rejected diagnostic:

- dense median translation regresses: 620.2114 cm -> 638.7456 cm
- dense median rotation regresses: 10.8549 deg -> 12.0391 deg
- dense R@5/R@2 slightly improve, but the strict gate rejects the median and
  rotation regression

StMarysChurch native full-train is very weak in absolute terms, so the frozen
recipe keeps it neutral rather than claiming an LSF improvement. StMarysChurch
v3 full-train is also not promotable under the strict gate:

- dense median translation slightly improves: 4369.8833 cm -> 4364.6019 cm
- dense median rotation improves: 106.1449 deg -> 103.8163 deg
- dense R@2cm/2deg improves: 0.03161 -> 0.03228
- dense R@5cm/5deg regresses: 0.10155 -> 0.10020
- judgment: diagnostic only; the R@5 regression keeps StMarysChurch native in
  the frozen selected recipe

StMarysChurch v2 full-train was attempted as an optional old-recipe diagnostic,
but the process stopped before completion. The log reached `1078/1487` queries,
no `summary.json` or `metrics_summary.json` was written, and GPU/process checks
showed no running evaluator. This run has no valid metric and remains excluded
from every table.

## Precision-Primary Diagnostic Gate

A precision-primary gate has been added as a reporting diagnostic, not as a
replacement for the strict frozen-recipe gate. It treats dense median
translation/rotation as primary, recall as a bounded safety check, and rejects
cases whose absolute pose median remains unusable.

- ShopFacade v3:
  `output/lsf_reports/precision_primary_shop_v3_full_train_20260523.json`
  passes. It improves dense median translation, dense median rotation, R@5, and
  R@2, while staying below the absolute pose thresholds.
- OldHospital v3:
  `output/lsf_reports/precision_primary_old_v3_full_train_20260523.json`
  fails. R@5/R@2 improve slightly, but median translation and rotation both
  regress, and the absolute median pose remains outside the paper-success
  threshold.
- StMarysChurch v3:
  `output/lsf_reports/precision_primary_st_v3_full_train_20260523.json`
  fails. Median translation/rotation and R@2 improve with tolerable R@5 drop,
  but the absolute median pose remains unusable, so it cannot be promoted as a
  successful localization result.

This confirms that shifting the paper emphasis toward pose precision is
reasonable, but it does not make the current method globally effective. The
safe claim is still localized: ShopFacade is a real positive result; OldHospital
and StMarysChurch remain hard-scene boundaries.

## Runtime And Memory Profiling

The frozen recipe now has q5 native profiler coverage for all unique selected
maps. This is a reporting sidecar only, not the main accuracy result.

- coverage report: `output/lsf_reports/frozen_profile_coverage_q5_20260522.json`
- profile root: `output/stdloc_native/profile_q5_frozen_20260522`
- coverage checks: all 8 unique selected maps profiled, runtime present,
  memory present, at least 5 profiled queries, no test-split profiles
- peak GPU memory range: 2402-2600 MB across the frozen selected maps
- mean total latency range: 8664-15874 ms/query in the q5 profile runs

This closes the runtime/memory coverage gap for the frozen train-dev recipe.
The q80/q160 accuracy rows remain the accuracy evidence; the q5 rows are only
used to attach STDLoc/ULF-Loc-style reporting fields.

## New Hard-Scene Runs

OldHospital q160 v3:

- output:
  `output/stdloc_native/train_dev_q160_20260522/old_lsf_v3_hard_edits32_pointcloud_opencv/OldHospital`
- dense delta vs native: median +3.7766 cm, rotation -0.5222 deg,
  R@5 +0.01875, R@2 +0.025
- judgment: rejected by strict gate because median translation regresses,
  although it is better than v2 and improves recall.

StMarysChurch q160 v2:

- output:
  `output/stdloc_native/train_dev_q160_20260522/st_lsf_v2_native_edits32_pointcloud_opencv/StMarysChurch`
- dense delta vs native: median -322.6769 cm, rotation -2.1678 deg,
  R@10 +0.00625, R@5 -0.0125, R@2 +0.0125
- judgment: rejected by strict gate because R@5 regresses, despite improving
  median translation and R@2.

ShopFacade q80 v3:

- output:
  `output/stdloc_native/train_dev_q80_20260522/shop_lsf_v3_hard_edits32_opencv/ShopFacade`
- dense delta vs native: median -6.5917 cm, rotation -0.3359 deg,
  R@5 +0.0500, R@2 +0.0375
- judgment: accepted by strict gate.

## Audit State

New LSF candidate eval bundles include `manifest.json`, `command.txt`,
`metrics_summary.json`, `split_audit.json`, and `git_status.txt`.

`write_native_eval_audit_bundle` now traces upstream map/proposal/feedback
manifests and preserves a passed feedback-bank split audit when available.
This fixes the previous reporting-layer issue where LSF candidate evals were
marked `unknown` despite upstream feedback-bank audits passing.

Selected native train-dev bundles used by the frozen recipe now have native
dataset train/test disjoint split audits. Older non-selected diagnostic bundles
may still have `split_audit=unknown`; keep them out of paper-facing test tables.

Full-train bundles now also record a non-empty `checkpoint_path` using the
native STDLoc map detector checkpoint, for example
`detector/30000_detector.pth`. The frozen validation report additionally checks
that feedback-bank split status comes from `split_audit.json`, and that
selector/residual/rho feedback are disabled at eval time.

## Next Iteration

1. Freeze the current conservative scene-level recipe as the train-dev
   candidate: done in
   `output/lsf_reports/frozen_recipe_manifest_train_dev_20260522.json`.
2. Profile the frozen selected maps for runtime/memory with the native profiler:
   done in `output/lsf_reports/frozen_profile_coverage_q5_20260522.json`.
3. Validate the q160 frozen recipe on full train split: done in
   `output/lsf_reports/frozen_recipe_validation_train_full_q160_recipe_20260522.json`.
4. Add a precision-primary diagnostic gate: done in
   `output/lsf_reports/precision_primary_shop_v3_full_train_20260523.json`,
   `output/lsf_reports/precision_primary_old_v3_full_train_20260523.json`, and
   `output/lsf_reports/precision_primary_st_v3_full_train_20260523.json`.
5. Treat the incomplete StMarysChurch v2 full-train diagnostic as invalid unless
   it is explicitly relaunched from scratch with a fresh audit bundle; use the
   completed v3 diagnostic only to define the next hard-scene fix, not to
   reselect the current frozen recipe.
6. Do not launch Cambridge test-facing evaluation until hard-scene diagnostics
   are resolved into a predeclared fixed recipe with runtime/memory reporting
   attached to the final table.
