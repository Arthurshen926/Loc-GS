# Solver-Feedback Active Pose Augmentation Status - 2026-06-08

## Scope

This round implements a non-RL active pose augmentation controller for sparse-only
solver feedback. It does not use Cambridge test queries and does not modify the
STDLoc/ULF evaluator.

The module has two jobs:

1. Select rendered/source poses that are likely to expose sparse solver-supported
   landmarks.
2. Build a view-landmark fusion plan that keeps only clean PnP-inlier
   correspondence evidence and deboosts high-score outliers or artifact-heavy
   rays.

## New Code

- `loc_gs/feedback/active_pose_augmentation.py`
  - `select_active_pose_candidates(...)`
  - `candidate_poses_from_observations(...)`
  - `build_view_landmark_reliability(...)`
  - `build_active_pose_augmentation(...)`
- `loc_gs/scripts/build_solver_feedback_active_pose_augmentation.py`
  - writes `manifest.json`, `command.txt`, `metrics_summary.json`,
    `split_audit.json`, `git_status.txt`
  - supports explicit candidate-pose files or inferred candidates from
    non-test observation records.
- `tests/test_active_pose_augmentation.py`

## Smoke Result

Input:

- Scene: `ShopFacade`
- Split: `selfmap_train`
- Source observations:
  `output/ulfloc_streaming_observations_fulltrain/ShopFacade_20260603/observations.jsonl`
- Source split audit: passed, no test split.
- Smoke cap: first 50,000 observations.

Output:

- `output/active_pose_augmentation/ShopFacade_selfmap_smoke_minpos1_20260608`

Key metrics:

- candidate poses: 60
- eligible poses: 38
- selected poses: 16
- observations used: 50,000
- view-landmark pairs: 237,729
- selected fusion view-landmark pairs: 27,690
- fusion landmarks: 18,600
- deboosted view-landmark pairs: 208,054

Top selected active poses are early `seq2/frame*.png` views with roughly
600-1700 solver-positive landmarks per view. Top selected fusion pairs have
low reprojection errors, around 0.13-0.94 px in the inspected rows.

## Current Interpretation

The active selection mechanism is functional: real self-map solver traces
produce non-empty pose and view-landmark selections, and the selected fusion
pairs are low-reprojection PnP inlier evidence.

The follow-up implementation connects this plan into descriptor fusion by
extending the ULF pair-cache builder with `row_source_view_id`. Descriptor fusion
can now filter candidate query-descriptor observations by active
`(source_view_id, gaussian_id)` pairs before applying the conservative
trust-region descriptor update.

## Sparse-Only Eval

Protocol:

- Scene: `ShopFacade`
- Feedback source: self-map train cameras only.
- Eval split: train-dev seed13 20%.
- Official Cambridge test is not used.
- Evaluator: `loc_gs.scripts.eval_ulfloc_sparse_only`.
- Detector: disabled unless explicitly noted.

Artifacts:

- self-map pair cache:
  `output/reports/ulfloc_scene_matcher_cache_shopfacade_selfmap_train_active_schema2_20260608`
- active a0.1 descriptor artifact:
  `output/reports/ulfloc_solver_weighted_fusion_shopfacade_active_selfmap_a01_20260608/selected_landmark_descriptors.pt`
- active a0.1 eval:
  `output/reports/ulfloc_sparse_only_shopfacade_active_solver_fused_selfmap_a01_20260608`

Results:

| run | median TE cm | R@10cm/5d | R@5cm/5d | notes |
| --- | ---: | ---: | ---: | --- |
| native same-path baseline | 2.9780 | 0.9783 | 0.7174 | native descriptors |
| active solver-fused a0.1 | 2.6290 | 0.9565 | 0.7174 | 742 updated landmarks |
| active solver-fused a0.05 | 3.0918 | 0.9348 | 0.6957 | too weak/unstable |
| active solver-fused a0.1 + train-dev geomw detector | 2.9712 | 0.9348 | 0.7174 | diagnostic only; detector trained on train-dev |

Interpretation:

- Active solver-feedback descriptor fusion gives a clear median translation
  improvement on ShopFacade train-dev: `2.9780 -> 2.6290 cm` (`-0.3490 cm`).
- R@5 is unchanged.
- R@10 drops by one query, so this is not yet a strict recall improvement.
- The train-dev geomw detector diagnostic does not help this active descriptor;
  it weakens median and R@10 relative to active descriptor alone.
- Current active plan is based on a 50k-observation smoke artifact. A full,
  compact active-plan export should be built next before scaling to all scenes.

## Verification

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_active_pose_augmentation.py \
  tests/test_rendered_feedback_augmentation.py \
  tests/test_ray_attributed_solver_feedback.py \
  tests/test_resample_ulfloc_with_solver_feedback.py \
  tests/test_sparse_solver_set_selection.py \
  tests/test_sparse_pnp_validation.py
```

Result: `153 passed`.

Additional final verification after connecting active plan into descriptor
fusion:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_active_pose_augmentation.py \
  tests/test_rendered_feedback_augmentation.py \
  tests/test_ray_attributed_solver_feedback.py \
  tests/test_resample_ulfloc_with_solver_feedback.py \
  tests/test_sparse_solver_set_selection.py \
  tests/test_sparse_pnp_validation.py \
  tests/test_solver_weighted_feature_fusion.py \
  tests/test_build_ulfloc_scene_matcher_pair_cache_cli.py \
  tests/test_build_ulfloc_solver_weighted_feature_log.py \
  tests/test_eval_ulfloc_sparse_only.py
```

Result: `177 passed`.

## Full Compact Export - 2026-06-09

The 50k smoke cap has been removed for the available self-map observation
artifacts. Compact export keeps the full `landmark_fusion_plan` but omits the
large `view_landmark_reliability` table from the main artifact, retaining only a
small audit sample. No official Cambridge test split is used for active-plan
construction.

Five-scene train-dev/self-map active-plan artifacts:

| scene | observations | selected poses | view-landmark pairs | selected fusion pairs | fusion landmarks | split audit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GreatCourt | 1,770,672 | 128 | 8,765,892 | 1,957,827 | 316,738 | passed, no test |
| KingsCollege | 1,475,075 | 128 | 7,290,583 | 1,508,526 | 282,444 | passed, no test |
| OldHospital | 1,086,327 | 128 | 5,291,019 | 518,084 | 166,370 | passed, no test |
| ShopFacade | 258,771 | 128 | 1,183,833 | 126,241 | 79,625 | passed, no test |
| StMarysChurch | 1,545,438 | 128 | 6,242,011 | 420,034 | 205,582 | passed, no test |

Artifact roots:

- `output/active_pose_augmentation/GreatCourt_traindev_selfmap_full_compact_20260608`
- `output/active_pose_augmentation/KingsCollege_traindev_selfmap_full_compact_20260608`
- `output/active_pose_augmentation/OldHospital_traindev_selfmap_full_compact_20260608`
- `output/active_pose_augmentation/ShopFacade_traindev_selfmap_full_compact_20260608`
- `output/active_pose_augmentation/StMarysChurch_traindev_selfmap_full_compact_20260608`

ShopFacade fulltrain/self-map active-plan artifact:

- `output/active_pose_augmentation/ShopFacade_selfmap_full_compact_20260609`
- observations: 324,033
- selected poses: 128
- view-landmark pairs: 1,483,811
- selected fusion view-landmark pairs: 165,390
- fusion landmarks: 95,971
- split audit: passed, no test
- `max_observations=0`

Compact export sizes:

- `view_landmark_reliability.json` is about 45-49 KB for all exported scenes.
- `active_pose_augmentation.json` and `landmark_fusion_plan.json` still contain
  the complete fusion plan, so large scenes remain tens of MB. This is expected:
  descriptor fusion needs the complete `(gaussian_id, selected_view_ids)` plan.

## Full Active Descriptor Fusion Check

ShopFacade fulltrain active plan was connected into descriptor fusion using the
audited self-map train pair cache:

- pair cache:
  `output/reports/ulfloc_scene_matcher_cache_shopfacade_selfmap_train_active_schema2_20260608`
- active plan:
  `output/active_pose_augmentation/ShopFacade_selfmap_full_compact_20260609/landmark_fusion_plan.json`
- descriptor artifact:
  `output/reports/ulfloc_solver_weighted_fusion_shopfacade_active_full_selfmap_a01_20260609/selected_landmark_descriptors.pt`
- ULF log:
  `output/reports/ulfloc_log_shopfacade_active_full_solver_fused_selfmap_a01_20260609`
- sparse-only eval:
  `output/reports/ulfloc_sparse_only_shopfacade_active_full_solver_fused_selfmap_a01_20260609`

Descriptor-fusion metadata:

| run | active-plan landmarks | active-plan positive pairs | fused positive pairs | updated landmarks | min native cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50k smoke active a0.1 | 18,600 | 4,515 | 1,567 | 742 | 0.9959 |
| full active a0.1 | 95,971 | 32,106 | 9,375 | 2,930 | 0.9956 |

ShopFacade train-dev sparse-only metrics:

| run | median TE cm | delta cm | R@10cm/5d | delta R10 | R@5cm/5d | delta R5 | mean inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native same-path baseline | 2.9780 | 0.0000 | 0.9783 | 0.0000 | 0.7174 | 0.0000 | 279.93 |
| active 50k a0.1 | 2.6290 | -0.3490 | 0.9565 | -0.0217 | 0.7174 | 0.0000 | 282.28 |
| active full a0.1 | 2.9491 | -0.0289 | 1.0000 | +0.0217 | 0.7826 | +0.0652 | 288.96 |

Interpretation:

- Removing the 50k cap makes descriptor feedback much broader:
  `742 -> 2930` updated landmarks and `1567 -> 9375` fused positive pairs.
- Full active feedback improves recall and mean inliers over the native
  same-path baseline, but the median gain is only `-0.0289 cm`.
- The earlier 50k smoke active plan remains better for median precision, but it
  loses one R@10 query. Therefore active feedback is not monotonic with more
  feedback; the next method step should prune active views/pairs by query-level
  validation utility rather than simply expanding the feedback set.
- The full active result is useful evidence that solver feedback can improve
  strict recall through descriptor fusion, but it is not yet a SOTA-level sparse
  result.

## Five-Scene Sparse Validation - 2026-06-09

After the five-scene compact active-plan export, self-map train pair caches were
built for all Cambridge scenes with `camera_split=train` and
`row_source_view_id_available=true`. These caches are explicitly audited as
non-test sources and can be consumed by active descriptor fusion.

Self-map train pair-cache roots:

- `output/reports/ulfloc_scene_matcher_cache_greatcourt_selfmap_train_active_schema2_20260609`
- `output/reports/ulfloc_scene_matcher_cache_kingscollege_selfmap_train_active_schema2_20260609`
- `output/reports/ulfloc_scene_matcher_cache_oldhospital_selfmap_train_active_schema2_20260609`
- `output/reports/ulfloc_scene_matcher_cache_shopfacade_selfmap_train_seed13_active_schema2_20260609`
- `output/reports/ulfloc_scene_matcher_cache_stmaryschurch_selfmap_train_active_schema2_20260609`

Active descriptor-fusion scale:

| scene | active-plan landmarks | active-plan positive pairs | fused positive pairs | updated landmarks | min native cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 316,738 | 426,810 | 170,917 | 9,843 | 0.9956 |
| KingsCollege | 282,444 | 209,730 | 89,323 | 7,109 | 0.9951 |
| OldHospital | 166,370 | 86,549 | 31,453 | 4,804 | 0.9955 |
| ShopFacade | 79,625 | 25,335 | 7,007 | 2,511 | 0.9958 |
| StMarysChurch | 205,582 | 165,122 | 44,390 | 5,772 | 0.9952 |

Sparse-only train-dev metrics:

| scene | base TE | active TE | dTE | base R10 | active R10 | dR10 | base R5 | active R5 | dR5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 14.6535 | 13.4105 | -1.2430 | 0.3616 | 0.3844 | +0.0228 | 0.1303 | 0.1433 | +0.0130 |
| KingsCollege | 17.6219 | 17.2944 | -0.3274 | 0.2418 | 0.2705 | +0.0287 | 0.0533 | 0.0820 | +0.0287 |
| OldHospital | 15.5136 | 15.4392 | -0.0744 | 0.3352 | 0.3464 | +0.0112 | 0.1229 | 0.1117 | -0.0112 |
| ShopFacade | 3.1258 | 3.3275 | +0.2016 | 0.9130 | 0.9130 | +0.0000 | 0.6957 | 0.7391 | +0.0435 |
| StMarysChurch | 7.0973 | 6.6161 | -0.4812 | 0.6812 | 0.6946 | +0.0134 | 0.3557 | 0.3993 | +0.0436 |
| MACRO | 11.6024 | 11.2175 | -0.3849 | 0.5066 | 0.5218 | +0.0152 | 0.2716 | 0.2951 | +0.0235 |

Summary artifact:

- `output/reports/ulfloc_active_full_solver_fused_five_scene_sparse_summary_20260609`

Artifact audit:

- The five active sparse-only eval directories were rerun after adding explicit
  sparse-only split-audit fields.
- Each directory has `manifest.json`, `command.txt`, `metrics_summary.json`,
  `split_audit.json`, and `git_status.txt`.
- Each `split_audit.json` reports `official_test_used=false` and
  `test_split_used=false` for `train_dev_seed13_20p_sparse_validation`.

Interpretation:

- Full active solver-feedback descriptor fusion is now five-scene positive in
  sparse-only train-dev: macro median TE improves by `0.3849 cm`, R@10 improves
  by `1.52 pp`, and R@5 improves by `2.35 pp`.
- Four of five scenes improve median TE; ShopFacade regresses by `+0.2016 cm`
  while R@5 improves.
- R@2 is not yet stable and drops in macro. This should not be used as a main
  claim yet.
- The evidence now supports a narrower sparse-stage claim: self-localization
  solver feedback can improve ULF-style landmark descriptor fusion under a
  conservative trust region. It does not yet prove a full SOTA claim.
- The next method step should not expand feedback volume further. It should add
  query/view validation pruning to remove feedback pairs that improve recall but
  hurt scene-specific median precision, especially ShopFacade and OldHospital
  strict R@5 behavior.
