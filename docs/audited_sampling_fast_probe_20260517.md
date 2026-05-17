# Audited Sampling/Fast Probe, 2026-05-17

## Scope

This is a ShopFacade q20 diagnostic follow-up to
`docs/sampling_field_budget_20260516.md` and
`docs/speed_accuracy_probe_20260517.md`.

The selector training cache was regenerated with source image ids and passed the
Cambridge split disjointness audit:

```text
output/unified_lff_v2/episode_caches_audited_20260517/ShopFacade/scene_match_pairs.pt
output/unified_lff_v2/episode_caches_audited_20260517/ShopFacade/scene_match_pairs.json
```

The cache metadata records `feedback_bank_split_name=selfmap_train_rendered`,
`phase_counts={train: 30000, rendered: 30000}`, 103 official ShopFacade test
ids, and zero train/rendered overlap with the test ids.

The q20 profiles below are still diagnostic because they use the test split for
measurement and have not been promoted through a full paper-facing audit table.
No STDLoc vendored evaluator or metric was modified.

## Code Changes

- `calibrate_landmark_matchability.py` now writes source image ids and a split
  audit into new pair/listwise caches.
- `train_unified_lff.py` preserves embedded cache split audits and records the
  selector loss weights in `command.txt` / `manifest.json`.
- `profile_stdloc_native.py` supports explicit, audited STDLoc config overrides
  for fixed fast-mode probes:
  `--sparse_max_iterations`, `--sparse_min_iterations`,
  `--dense_max_iterations`, and `--dense_min_iterations`. It also supports
  profiler-only prior-weight overrides for diagnostics:
  `--sparse_landmark_prior_weight` and `--dense_locability_prior_weight`.
- `selector_resampling.py` / `export_selector_resampled_map.py` now support an
  optional self-map hard-negative risk penalty. The risk is aggregated from
  episode-cache reprojection errors, mapped through `base_gaussian_id`, and
  recorded in the selector-resampling manifest.
- The same resampling path now supports all-Gaussian candidate pools with a
  source-retention fraction. This lets diagnostics keep a fixed fraction of the
  native sampled pool before filling from all Gaussians.
- `selector_resampling.py` / `export_selector_resampled_map.py` also support a
  self-map positive-support prior. It aggregates high-score true positives from
  the audited episode cache and records the source split audit in the manifest.
- The same path now supports a sparse-pose information prior. It aggregates
  high-score, low-reprojection self-map inliers with query-score and descriptor
  margin weighting, then records the audited cache metadata in the manifest.
- The same resampling path now supports a hard-query support prior. It
  aggregates low-reprojection self-map inliers but upweights low-margin
  ambiguous queries, so the prior tests hard-case protection rather than only
  easy high-margin pose evidence.
- The resampling path now also supports query-coverage reservation. Instead of
  adding another scalar landmark prior, it greedily reserves landmarks that
  cover low-margin self-map queries with low-reprojection inliers. This is a
  reconstruction-time map export rule, not per-query branch selection.
- Selector-resampled exports now also write a top-level `manifest.json` alias
  with git commit, command, split, hyperparameters, data roots, and feedback
  flags, alongside the legacy `selector_resampling_manifest.json`.
- The resampling path also supports a strict-support reservation guard. It
  reserves a fixed fraction of the map budget for high-score, low-reprojection
  self-map inliers before the broader selector/pose score fills the remaining
  slots. This is a reconstruction-time map export rule, not query-time branch
  selection.
- `train_unified_lff.py` now supports opt-in geometry-aware selector training
  from audited self-map reprojection evidence. The global `pose_target_*`
  target blends landmark-level reliability into the gate target; the newer
  `lambda_selector_pose_pair` loss ranks candidates within each self-map query
  by low reprojection error, query score, margin, and candidate cosine. Both
  paths are disabled by default and recorded in training manifests/checkpoints.

## Audited Selector Artifacts

Selector-only checkpoints:

```text
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_budget001/unified_lff_v2.pt
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_budget005/unified_lff_v2.pt
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_gate15_budget001/unified_lff_v2.pt
```

All three training sidecars report `split=selfmap_train_rendered`,
`split_audit.audit_status=passed`, and `diagnostic=false`.

Maps profiled here:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/budget001_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/budget001_selector12288_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/budget001_hn05t07_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_budget001_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_budget001_selector12288_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_budget001_selector4096_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_budget001_selector2048_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_pos07w05_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_pos07w05_selector4096_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_allg_budget001_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_allg_budget001_selector12288_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_allg_ret75_budget001_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_allg_ret50_budget001_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_pose05t07_allg_ret75_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_pose05t07_strict3px_f05_allg_ret75_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_pose025t07_strict3px_f05_allg_ret75_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_pose05t07_strict3px_f05_allg_ret90_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/fp07_pose025t07_strict3px_f05_allg_ret90_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_r001_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f25_r002_selector8192_cov8/ShopFacade
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_lr3e3_qcov_f50_r002_selector8192_cov8/ShopFacade
```

## q20 Results

All rows are ShopFacade test split, `max_test_cameras=20`,
`warmup_cameras=2`, same native STDLoc wrapper, one dense iteration, and no
parallel timing contention. Fast rows use:

```text
sparse max/min iterations = 20000 / 100
dense max/min iterations = 500 / 50
```

| Run | Landmarks | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Sparse ms | Dense ms | Sparse inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native12288 | 12288 | 2.216 / 0.1172 | 1.00 | 0.90 | 0.30 | 394.5 | 391.9 | 200.3 | 168.4 | 325.9 |
| native12288 fast | 12288 | 2.216 / 0.1172 | 1.00 | 0.90 | 0.30 | 383.0 | 356.5 | 212.0 | 145.8 | 325.9 |
| native4096 fast | 4096 | 2.686 / 0.1164 | 1.00 | 0.90 | 0.25 | 482.9 | 476.5 | 309.9 | 147.3 | 226.1 |
| audited fp07 selector4096 fast | 4096 | 2.686 / 0.1164 | 1.00 | 0.85 | 0.25 | 493.0 | 484.5 | 318.7 | 148.8 | 226.1 |
| audited fp07 pos07w05 selector4096 fast | 4096 | 2.516 / 0.1207 | 1.00 | 0.95 | 0.15 | 491.6 | 494.8 | 319.6 | 146.4 | 228.3 |
| native2048 fast | 2048 | 2.508 / 0.1124 | 1.00 | 0.90 | 0.30 | 493.8 | 496.4 | 320.1 | 148.1 | 157.2 |
| audited fp07 selector2048 fast | 2048 | 2.570 / 0.1137 | 1.00 | 0.90 | 0.30 | 494.4 | 489.2 | 322.4 | 146.4 | 156.6 |
| native8192 | 8192 | 2.277 / 0.1172 | 0.95 | 0.90 | 0.35 | 468.2 | 482.2 | 268.0 | 174.6 | 299.1 |
| native8192 fast | 8192 | 2.277 / 0.1172 | 0.95 | 0.90 | 0.35 | 423.3 | 409.9 | 250.2 | 147.8 | 299.1 |
| audited selector8192 | 8192 | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 481.4 | 495.0 | 282.5 | 173.0 | 298.9 |
| audited selector8192 fast | 8192 | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 408.4 | 384.3 | 236.4 | 146.3 | 298.9 |
| audited selector8192 hard-neg fast | 8192 | 2.217 / 0.1125 | 0.95 | 0.85 | 0.35 | 419.3 | 404.2 | 245.5 | 148.5 | 297.6 |
| audited fp07 selector8192 fast | 8192 | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 401.5 | 394.3 | 229.0 | 147.3 | 299.0 |
| audited fp07 pos07w05 selector8192 fast | 8192 | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 424.3 | 422.9 | 251.2 | 147.6 | 299.4 |
| native8192 fast prior005 | 8192 | 2.321 / 0.1156 | 0.95 | 0.90 | 0.30 | 414.5 | 400.7 | 244.9 | 143.9 | 297.1 |
| audited fp07 selector8192 fast prior005 | 8192 | 2.323 / 0.1162 | 0.95 | 0.90 | 0.30 | 386.3 | 379.5 | 219.3 | 141.6 | 296.9 |
| native8192 train-cal budget | 8192 | 2.277 / 0.1172 | 0.95 | 0.90 | 0.35 | 320.3 | 296.1 | 166.2 | 128.9 | 299.0 |
| audited fp07 selector8192 train-cal budget | 8192 | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 330.6 | 318.4 | 177.6 | 127.7 | 298.9 |
| audited fp07 pose05 ret75 train-cal budget | 8192 | 2.475 / 0.1172 | 0.95 | 0.90 | 0.20 | 306.9 | 307.8 | 156.9 | 125.0 | 281.9 |
| audited fp07 pose05 strict3px f05 ret75 train-cal budget | 8192 | 2.270 / 0.1049 | 1.00 | 0.95 | 0.30 | 313.5 | 289.5 | 159.2 | 129.2 | 305.4 |
| audited fp07 allg selector8192 fast | 8192 | 2.457 / 0.1047 | 1.00 | 0.95 | 0.35 | 494.0 | 493.7 | 324.9 | 143.4 | 206.9 |
| audited fp07 allg ret75 selector8192 fast | 8192 | 2.459 / 0.1195 | 0.95 | 0.90 | 0.20 | 471.3 | 471.5 | 300.7 | 144.9 | 280.1 |
| audited fp07 allg ret50 selector8192 fast | 8192 | 2.678 / 0.1137 | 1.00 | 0.95 | 0.25 | 484.7 | 479.9 | 312.0 | 146.9 | 249.2 |
| audited selector12288 fast | 12288 | 2.216 / 0.1172 | 1.00 | 0.90 | 0.25 | 364.3 | 360.2 | 195.8 | 143.0 | 325.9 |
| audited fp07 selector12288 fast | 12288 | 2.216 / 0.1172 | 1.00 | 0.90 | 0.25 | 393.3 | 374.2 | 224.9 | 143.0 | 325.9 |
| native12288 fast prior005 | 12288 | 2.225 / 0.1201 | 1.00 | 0.85 | 0.30 | 390.4 | 376.0 | 221.2 | 144.4 | 323.1 |
| audited fp07 selector12288 fast prior005 | 12288 | 2.291 / 0.1195 | 1.00 | 0.85 | 0.30 | 399.6 | 397.4 | 222.1 | 151.7 | 322.9 |
| audited fp07 allg selector12288 fast | 12288 | 2.544 / 0.1002 | 1.00 | 0.95 | 0.30 | 463.3 | 455.7 | 289.6 | 148.0 | 249.9 |

## Self-Map Prior Calibration

The sparse-prior check was repeated on the ShopFacade train camera split
(`eval_split=train`, `max_test_cameras=20`, `warmup_cameras=2`) so the q20 test
rows above are not used to choose a prior weight. These are calibration
diagnostics only, not paper-facing train/test results. The profile sidecars
record `split_audit.status=unknown` and `paper_safe=false` because they do not
attach a full calibration/self-map disjointness audit.

Artifacts:

```text
output/unified_lff_v2/profile_20260517/train_q20_native8192_fast_prior000_s20k100_d500_50_solo
output/unified_lff_v2/profile_20260517/train_q20_native8192_fast_prior005_s20k100_d500_50_solo
output/unified_lff_v2/profile_20260517/train_q20_audited_fp07_budget001_selector8192_cov8_fast_prior000_s20k100_d500_50_solo
output/unified_lff_v2/profile_20260517/train_q20_audited_fp07_budget001_selector8192_cov8_fast_prior005_s20k100_d500_50_solo
```

| Run | Split | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train native8192 fast prior000 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 340.9 | 323.1 | 179.9 | 88.7 | 91.2 | 135.3 | 352.1 |
| train native8192 fast prior005 | train | 1.525 / 0.0489 | 0.95 | 0.95 | 0.65 | 347.3 | 343.1 | 187.0 | 89.9 | 97.2 | 134.5 | 349.9 |
| train fp07 selector8192 fast prior000 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 338.5 | 330.7 | 172.5 | 88.1 | 84.3 | 140.2 | 352.4 |
| train fp07 selector8192 fast prior005 | train | 1.571 / 0.0502 | 0.95 | 0.95 | 0.65 | 366.4 | 356.7 | 196.1 | 89.4 | 106.7 | 144.5 | 350.2 |

This calibration slice does not support selecting `sparse_landmark_prior_weight`
`0.05`: recall is unchanged, while latency rises for both native 8k and fp07
8k. The q20 test prior005 speed row should therefore remain a test diagnostic,
not a chosen setting.

## Train-Calibrated Budget Control

The fixed fast budget was also moved onto the train camera split before running
one q20 test diagnostic. The selected budget was:

```text
sparse max/min iterations = 5000 / 50
dense max/min iterations = 100 / 10
```

Train calibration rows:

| Run | Split | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train native8192 fast s20k100 d500/50 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 340.9 | 323.1 | 179.9 | 88.7 | 91.2 | 135.3 | 352.1 |
| train fp07 selector8192 fast s20k100 d500/50 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 338.5 | 330.7 | 172.5 | 88.1 | 84.3 | 140.2 | 352.4 |
| train native8192 budget s5k50 d250/25 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 321.3 | 284.4 | 173.1 | 60.9 | 112.2 | 122.5 | 352.1 |
| train fp07 selector8192 budget s5k50 d250/25 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 322.4 | 306.3 | 170.0 | 60.8 | 109.2 | 126.7 | 352.4 |
| train native8192 budget s5k50 d100/10 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 316.0 | 299.4 | 177.9 | 60.9 | 117.0 | 112.4 | 352.1 |
| train fp07 selector8192 budget s5k50 d100/10 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 318.7 | 305.2 | 177.6 | 60.9 | 116.8 | 115.2 | 352.4 |
| train fp07 pose05 ret75 budget s5k50 d100/10 | train | 1.816 / 0.0575 | 1.00 | 0.95 | 0.65 | 303.0 | 278.2 | 160.3 | 61.1 | 99.2 | 117.1 | 324.7 |
| train fp07 pose05 strict3px f05 ret75 budget s5k50 d100/10 | train | 1.521 / 0.0470 | 1.00 | 0.95 | 0.65 | 312.1 | 288.4 | 169.9 | 57.4 | 112.4 | 116.6 | 364.9 |
| train fp07 pose025 strict3px f05 ret90 budget s5k50 d100/10 | train | 1.509 / 0.0496 | 1.00 | 0.95 | 0.65 | 283.9 | 274.1 | 140.6 | 55.4 | 85.2 | 117.6 | 382.6 |

Fixed q20 test diagnostic:

| Run | Split | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test native8192 traincal s5k50 d100/10 | test | 2.277 / 0.1172 | 0.95 | 0.90 | 0.35 | 320.3 | 296.1 | 166.2 | 60.7 | 105.5 | 128.9 | 299.0 |
| test fp07 selector8192 traincal s5k50 d100/10 | test | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 330.6 | 318.4 | 177.6 | 60.7 | 116.9 | 127.7 | 298.9 |
| test fp07 pose05 ret75 traincal s5k50 d100/10 | test | 2.475 / 0.1172 | 0.95 | 0.90 | 0.20 | 306.9 | 307.8 | 156.9 | 60.9 | 96.0 | 125.0 | 281.9 |
| test fp07 pose05 strict3px f05 ret75 traincal s5k50 d100/10 | test | 2.270 / 0.1049 | 1.00 | 0.95 | 0.30 | 313.5 | 289.5 | 159.2 | 61.1 | 98.1 | 129.2 | 305.4 |

This is useful evidence for a fixed evaluator-budget speed lever, but not for a
Loc-GS selector speed claim: the selector map preserves recall under the frozen
budget, but native8192 is faster on both train calibration and q20 test.

The pose-information ret75 map uses the audited self-map cache with
`pose_information_score_threshold=0.7` and `pose_information_weight=0.5`; its
manifest reports `audit_status=passed`, 1,479 positive inlier pairs, and 473
pose-informed landmarks. It is a speed/accuracy tradeoff diagnostic only: it is
faster than native8192 under the frozen q20 budget but drops strict R2 from 0.35
to 0.20.

The strict-support guarded pose map adds a fixed `strict_support_fraction=0.05`
reservation from the same audited cache with `score_threshold=0.7` and
`reprojection_threshold_px=3.0`. Its manifest reports `audit_status=passed`,
981 strict positive pairs, and 386 strict-support landmarks. On train-q20 it
improves median, loose R10, and latency versus native8192 under the frozen
budget without changing R5/R2. On the fixed q20 test diagnostic it improves
median, R10, R5, and latency versus native8192, but strict R2 remains lower
than native8192 (0.30 vs 0.35), so it is not a main result.

A train-only calibration then increased source retention to 90% and reduced
the pose-information weight to 0.25, still with the same strict 3px guard. The
map keeps 7,443 of the native source-sampled landmarks and reserves 386 strict
support landmarks. Its solo train-q20 row is the best train-calibrated point so
far under this budget: it improves native8192 train median and latency while
keeping R5/R2 unchanged and raising loose R10 to 1.00. No additional q20 test
row was run for this variant in order to avoid test-driven recipe iteration.

Larger train-split check (`eval_split=train`, `max_test_cameras=80`,
`warmup_cameras=2`, same frozen sparse/dense caps):

| Run | Split | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-q80 native8192 budget s5k50 d100/10 | train | 1.652 / 0.1382 | 0.95 | 0.925 | 0.55 | 283.3 | 261.8 | 158.4 | 58.9 | 99.5 | 99.1 | 221.4 |
| train-q80 fp07 source-pool selector8192 budget s5k50 d100/10 | train | 1.652 / 0.1382 | 0.95 | 0.925 | 0.55 | 280.1 | 263.2 | 155.4 | 58.8 | 96.6 | 99.1 | 221.4 |
| train-q80 fp07 pose025 strict3px f05 ret90 budget s5k50 d100/10 | train | 1.786 / 0.1351 | 0.925 | 0.850 | 0.5375 | 280.6 | 262.0 | 155.7 | 57.8 | 97.9 | 99.3 | 223.5 |
| train-q80 fp07 pose025 strict3px f05 protnative95 budget s5k50 d100/10 | train | 1.796 / 0.1372 | 0.9375 | 0.8875 | 0.5125 | 282.9 | 262.0 | 156.1 | 58.0 | 98.1 | 100.9 | 229.6 |
| train-q80 fp07 pose025 protnative98 budget s5k50 d100/10 | train | 1.805 / 0.1368 | 0.9375 | 0.8875 | 0.5375 | 291.9 | 284.3 | 164.9 | 59.1 | 105.7 | 101.2 | 219.2 |

The q80 train check rejects the ret90/pose0.25/strict3px map as a frozen
candidate: it is only 2.7 ms faster on mean latency, but median translation and
all dense recall thresholds are worse than native8192. Query-level comparison
shows R10 recovered/lost = 1/3, R5 recovered/lost = 0/6, and R2
recovered/lost = 2/3. The large failures are concentrated around
`seq2/frame00038.png`, `seq2/frame00039.png`, and `seq2/frame00048.png`, where
native remains within 5 cm but the candidate has meter-scale dense errors.

Root-cause isolation on the same q80 train slice used parallel runs for
accuracy only; timing from these rows is ignored because parallel profiling is
known to distort latency. The source-pool selector map overlaps native8192 on
8,185 / 8,192 landmarks and exactly matches native8192 dense metrics, while
the all-Gaussian maps drop 766-6,024 native8192 landmarks and lose hard cases:

| Run | Native8192 overlap | Added / dropped | Median cm/deg | R10 | R5 | R2 | R10 recovered/lost | R5 recovered/lost | R2 recovered/lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native8192 | 8192 | 0 / 0 | 1.652 / 0.1382 | 0.950 | 0.925 | 0.550 | - | - | - |
| fp07 source-pool selector8192 | 8185 | 7 / 7 | 1.652 / 0.1382 | 0.950 | 0.925 | 0.550 | 0 / 0 | 0 / 0 | 0 / 0 |
| fp07 all-gaussian selector8192 | 2168 | 6024 / 6024 | 2.584 / 0.1832 | 0.8125 | 0.725 | 0.3625 | 1 / 12 | 1 / 17 | 7 / 22 |
| fp07 pose05 strict3px ret75 | 6197 | 1995 / 1995 | 1.820 / 0.1397 | 0.925 | 0.8375 | 0.5375 | 1 / 3 | 1 / 8 | 6 / 7 |
| fp07 pose025 strict3px ret90 | 7426 | 766 / 766 | 1.786 / 0.1351 | 0.925 | 0.850 | 0.5375 | 1 / 3 | 0 / 6 | 2 / 3 |

This points to the map-level replacement step as the failure source, not to the
selector score on the native sampled pool. The next map objective should be a
native-supported replacement guard: new all-Gaussian landmarks must either
improve train/self-map hard-case evidence or be added only after protecting the
native8192 hard-case support set.

Follow-up guarded exports added an explicit protected-source reservation to the
resampler. The 95% native-protected strict-support map keeps 7,840 / 8,192
native landmarks and recovers the obvious meter-scale failures around
`seq2/frame00038.png`, `seq2/frame00039.png`, and `seq2/frame00048.png`, but it
still loses q80 train aggregate recall: R10/R5/R2 becomes
0.9375/0.8875/0.5125. The no-strict 98% native-protected map keeps 8,031 /
8,192 native landmarks and still loses R10/R5/R2 to 0.9375/0.8875/0.5375 while
running slower than native8192. This rejects same-budget all-Gaussian
replacement as the next frozen candidate, even with an explicit native-source
guard. The source-pool selector solo timing is the only q80 train point that
matches native accuracy and improves mean latency (280.1 ms vs 283.3 ms), but
the median latency is slightly worse and this remains train-only evidence.

The source-pool timing signal was then repeated as three paired q80 train runs,
with native and selector profiled sequentially on the same GPU per pair and the
three pairs run across GPUs 0/1/2. Accuracy stayed exactly matched, but the
runtime advantage did not hold under paired load: selector total mean was slower
by +67.5, +64.1, and +53.7 ms/query. These paired timings are not paper-facing
because of shared CPU/disk load, but they are enough to reject a strong
source-pool runtime claim from the earlier single solo delta.

An additive full-mode map was also checked: protect all native8192 landmarks and
fill the remaining 4,096 slots to make a 12,288-landmark selector/pose map. It
preserves native8192 exactly, but compared with native12288 on the same q80
train slice it loses dense median and recall: native12288 gives
1.533 cm / 0.1018 deg with R10/R5/R2 = 0.9625/0.9000/0.6500, while the additive
selector12288 map gives 1.576 cm / 0.1233 deg with R10/R5/R2 =
0.9500/0.8875/0.6250. This rejects the current additive fill as a full-mode
candidate.

Finally, three stronger selector-only checkpoints were trained from the audited
self-map cache. The best training variants raise self-map listwise accuracy from
0.5532 to 0.5604 and increase gate spread from std 0.0039 to about 0.0074. The
rankstrong source-pool export changes only 11 / 8,192 native landmarks, and its
solo q80 train profile keeps R10/R5 unchanged while improving R2 from 0.5500 to
0.5625 by recovering `seq2/frame00068.png`. It is slower than native8192
(304.8 ms vs 283.3 ms mean latency), so this is an accuracy-only diagnostic
signal rather than a speed claim.

Two geometry-aware selector-training probes were then run on the same audited
cache. The global pose-target probe `posew05_score07` used 1,479 positive
self-map pairs over 473 landmarks, but changed only one sampled index relative
to `rankstrong_gate01_bias5` and matched rankstrong dense q80 train metrics
exactly. Stronger global weighting with `posew10_score00_gate1` produced an
identical sampled set to rankstrong.

The pair-level pose loss was active but still too weak at the current selector
score scale. Three selector-only checkpoints were trained with
`lambda_selector_pose_pair` and audited self-map reprojection utilities:

```text
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_posepair_w1_t8s0_rankstrong_gate01_bias5/unified_lff_v2.pt
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_posepair_w2_t8s0_rankstrong_gate01_bias5/unified_lff_v2.pt
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_posepair_w2_t8s07_rankstrong_gate01_bias5/unified_lff_v2.pt
```

The t8/s0 variants used 9,440 positive pairs over 1,899 landmarks; t8/s0.7 used
1,479 positive pairs over 473 landmarks. The exported 8k source-pool maps
changed only 1-3 sampled ids versus rankstrong. The two profiled maps matched
rankstrong dense q80 train accuracy exactly: 1.653 cm / 0.1382 deg with
R10/R5/R2 = 0.950/0.925/0.5625. These profiles were parallel accuracy checks,
so their timing is ignored.

Score-scale isolation found why these losses barely move the source-pool map:
the clean native source `sampled_scores.pkl` has mean/std 0.2922/0.1297 and
range 0.0000-0.6967, while the rankstrong selector logit has std 0.0294 and
the sigmoid gate has std 0.00735. With the default export blend, source scores
dominate selector differences.

Three selector-amplified source-pool exports were checked on train-q80 for
accuracy only. These rows used parallel profiling, so timing is intentionally
ignored:

| Run | Native overlap | Rankstrong overlap | Median cm/deg | R10 | R5 | R2 | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rankstrong source-pool | 8181 | 8192 | 1.653 / 0.1382 | 0.950 | 0.925 | 0.5625 | baseline selector gain, slower solo timing |
| rankstrong sw10 | 8106 | 8117 | 1.655 / 0.1395 | 0.950 | 0.925 | 0.5625 | preserves recall, worsens median |
| rankstrong minmax | 8071 | 8082 | 1.648 / 0.1391 | 0.950 | 0.9125 | 0.5875 | improves R2/median translation, loses R5 |
| rankstrong rank | 4744 | 4752 | 2.468 / 0.1467 | 0.850 | 0.7875 | 0.4375 | harmful normalization |

The minmax row is useful diagnostic evidence that selector normalization can
change strict recall on the native source pool, but it is not a candidate to
freeze because it trades away R5. The rank transform is rejected.

A finer minmax weight sweep was then run between the safe and unsafe endpoints.
All rows are train-q80 parallel accuracy diagnostics, with timing ignored:

| Selector weight | Rankstrong changed ids | Median cm/deg | R10 | R5 | R2 | Outcome |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | 63 | 1.674 / 0.1375 | 0.950 | 0.925 | 0.5625 | preserves recall, worsens median translation |
| 0.30 | 74 | 1.655 / 0.1395 | 0.950 | 0.925 | 0.5625 | preserves recall, worsens median pose |
| 0.35 | 84 | 1.674 / 0.1395 | 0.950 | 0.9125 | 0.5625 | loses R5 without gaining R2 |
| 0.40 | 90 | 1.692 / 0.1395 | 0.950 | 0.9125 | 0.5625 | loses R5 and median |
| 0.50 | 100 | 1.646 / 0.1391 | 0.950 | 0.9125 | 0.5875 | gains R2/median translation, loses R5 |
| 0.75 | 108 | 1.648 / 0.1391 | 0.950 | 0.9125 | 0.5875 | gains R2/median translation, loses R5 |

This rejects a simple weighted minmax source-pool export as the next frozen
candidate. The transition is sharp: preserving R5 gives no strict-R2 gain, and
the weights that improve R2 drop R5.

A stronger training-side scale probe then raised selector variance directly
instead of changing export normalization. Three selector-only checkpoints were
trained from the same audited self-map cache, all with
`split_audit.audit_status=passed` and no test data:

```text
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_rankstrong_lr3e3_e48_gate005_bias5/unified_lff_v2.pt
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_posepair_w10_m1_gate005_bias5/unified_lff_v2.pt
output/unified_lff_v2/train_selector_only_audited_20260517/ShopFacade_posepair_w20_m1_gate0_bias5/unified_lff_v2.pt
```

Relative to the earlier rankstrong checkpoint, gate std increased from about
0.0074 to 0.047-0.056, and the source-pool exports changed 44-60 sampled ids
instead of 1-11. The two pose-pair runs again used 9,440 self-map positive
pairs over 1,899 landmarks. Train-q80 profiles were parallel accuracy checks,
so timing is ignored:

| Run | Native overlap | Rankstrong overlap | Median cm/deg | R10 | R5 | R2 | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rankstrong source-pool | 8181 | 8192 | 1.653 / 0.1382 | 0.950 | 0.925 | 0.5625 | baseline selector gain, slower solo timing |
| rankstrong lr3e3 e48 gate005 | 8124 | 8132 | 1.679 / 0.1395 | 0.950 | 0.925 | 0.5625 | same recall as rankstrong, worse median |
| posepair w10 m1 gate005 | 8137 | 8147 | 1.669 / 0.1362 | 0.950 | 0.9125 | 0.5500 | angle median improves, but R5/R2 drop |
| posepair w20 m1 gate0 | 8138 | 8148 | 1.676 / 0.1418 | 0.950 | 0.925 | 0.5500 | no strict-R2 gain, worse median |

This is a useful implementation check: the selector-scale bottleneck can be
moved during training, and the pair-level geometry loss is active. It is still
negative evidence for the current recipe because the larger gate spread does
not produce a q80 train Pareto point over either native8192 or the earlier
rankstrong source-pool map.

A hard-query support prior was then added to the resampler to test whether
low-margin self-map inliers can protect hard train cases better than the earlier
pose-information prior, which weighted high-margin/high-confidence evidence.
The prior used the same audited cache with `score_threshold=0.0`,
`reprojection_threshold_px=8.0`, and records 9,440 positive pairs over 1,899
landmarks. The exported maps stayed source-pool only; no all-Gaussian
replacement or test data was used. Train-q80 profiles were again parallel
accuracy checks, so timing is ignored:

| Run | Native overlap | Rankstrong overlap | Median cm/deg | R10 | R5 | R2 | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rankstrong source-pool | 8181 | 8192 | 1.653 / 0.1382 | 0.950 | 0.925 | 0.5625 | baseline selector gain, slower solo timing |
| rankstrong hard-query w0.25 | 8163 | 8170 | 1.653 / 0.1382 | 0.950 | 0.925 | 0.5500 | matches native, loses rankstrong R2 gain |
| rankstrong hard-query w1.0 | 8146 | 8153 | 1.795 / 0.1430 | 0.950 | 0.925 | 0.5250 | stronger hard-query prior hurts median/R2 |
| lr3e3 hard-query w0.5 | 8140 | 8151 | 1.688 / 0.1418 | 0.950 | 0.925 | 0.5500 | preserves native recall, worse median |

This rejects the first hard-query support prior as a frozen candidate. It is a
useful diagnostic because it shows that simply reweighting self-map inliers
toward ambiguous queries is still not enough to preserve the single rankstrong
strict-R2 recovery while moving the source-pool set.

A query-coverage reservation objective was then added to test the next
structural hypothesis from the hard-query result: a map-level reservation should
cover the low-margin self-map queries as sets, rather than assign a scalar prior
to individual landmarks. The reservation uses the same audited cache
(`feedback_bank_split_name=selfmap_train_rendered`, split audit passed), stays
source-pool only, and records the cache path plus coverage metadata in each
resampling manifest.

The first two rows below are from parallel q80 train accuracy checks, so their
timing is ignored. The qcov f10/r001 row was then repeated as a solo profile for
timing:

| Run | Native overlap | Rankstrong overlap | Qcov reserved | Hard queries covered | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rankstrong source-pool | 8181 | 8192 | 0 | - | 1.653 / 0.1382 | 0.950 | 0.925 | 0.5625 | 304.8 | 287.6 | baseline selector gain, slower than native solo |
| rankstrong qcov f10/r001 | 8168 | 8178 | 60 | 506 / 506 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | 275.2 | 262.4 | train-only candidate: better median than native/rankstrong and same rankstrong recall |
| rankstrong qcov f25/r002 | 8150 | 8159 | 110 | 1250 / 1250 | 1.629 / 0.1382 | 0.950 | 0.925 | 0.5500 | ignored | ignored | rejects: loses rankstrong strict-R2 gain |
| lr3e3 qcov f50/r002 | 8137 | 8147 | 151 | 2483 / 2483 | 1.664 / 0.1362 | 0.950 | 0.925 | 0.5750 | ignored | ignored | improves R2, but worsens translation median |

Against native8192 solo, qcov f10/r001 changes dense metrics by
-0.042 cm / -0.0039 deg and +0.0125 strict R2 with no R10/R5 loss. Against
rankstrong solo, it keeps R10/R5/R2 fixed and improves median by
-0.043 cm / -0.0039 deg. The first solo timing was faster than the old
native8192 solo row, but this speed signal was not stable after a corrected
same-GPU repeat below. This is the first train-q80 row in this branch that
improves median pose and preserves rankstrong strict recall. It is still
train-only and should be repeated on self-map validation / other scenes before
any q20 or test diagnostic.

A small neighboring sweep was then run on the same train-q80 slice to check
whether qcov f10/r001 is a fragile point. These rows were parallel accuracy
checks, so timing is ignored:

| Run | Qcov reserved | Hard queries covered | Median cm/deg | R10 | R5 | R2 | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rankstrong qcov f05/r0005 | 35 | 265 / 265 | 1.653 / 0.1343 | 0.950 | 0.925 | 0.5625 | preserves R2, but loses the translation median gain |
| rankstrong qcov f10/r001 | 60 | 506 / 506 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | best balanced train-q80 accuracy point |
| rankstrong qcov f10/r002 | 60 | 506 / 506 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | identical to f10/r001 because only 60 source-pool qcov candidates are available before dedupe |
| rankstrong qcov f15/r0015 | 85 | 746 / 746 | 1.631 / 0.1382 | 0.950 | 0.925 | 0.5500 | loses the rankstrong strict-R2 gain |
| rankstrong qcov f25/r002 | 110 | 1250 / 1250 | 1.629 / 0.1382 | 0.950 | 0.925 | 0.5500 | loses the rankstrong strict-R2 gain |

The sweep supports a narrow reservation basin around the lowest 10% margin
queries. Covering fewer hard queries keeps the R2 gain but not the translation
median gain; covering more hard queries drifts back to native strict R2.

The f10/r001 timing was also repeated sequentially after a corrected native8192
repeat on the same GPU. Accuracy stayed unchanged, but the speed signal did not
replicate: native8192 repeat was 274.9 ms mean / 263.0 ms median, while qcov
f10/r001 repeat was 281.5 ms mean / 265.9 ms median. Treat qcov f10/r001 as a
train-only accuracy/strict-recall candidate, not a speed candidate.

A sparse-prior follow-up then enabled `sampled_scores.pkl` during sparse
matching on the qcov f10/r001 map. This is still train-q80 calibration evidence:
the prior weight was not selected on the test split, inference remains the same
single STDLoc-compatible path, and the parallel rows are used only for accuracy.

| Run | Prior weight | Profile mode | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Outcome |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qcov f10/r001 repeat | 0.00 | solo | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | 281.5 | 265.9 | no-prior qcov accuracy baseline |
| qcov f10/r001 prior002 | 0.02 | parallel accuracy | 1.642 / 0.1263 | 0.9375 | 0.9125 | 0.5750 | ignored | ignored | rejects: loses R10/R5 |
| qcov f10/r001 prior005 | 0.05 | parallel accuracy | 1.639 / 0.1173 | 0.950 | 0.9125 | 0.6125 | ignored | ignored | improves R2, but loses R5 |
| qcov f10/r001 prior0075 | 0.075 | parallel accuracy | 1.685 / 0.1187 | 0.9375 | 0.9125 | 0.6000 | ignored | ignored | rejects: loses R10/R5 |
| qcov f10/r001 prior010 | 0.10 | parallel accuracy | 1.645 / 0.1300 | 0.950 | 0.925 | 0.5875 | ignored | ignored | preserves R10/R5 and improves R2 |
| qcov f10/r001 prior010 | 0.10 | solo | 1.645 / 0.1300 | 0.950 | 0.925 | 0.5875 | 276.9 | 258.7 | strict-R2 diagnostic; no speed claim |
| qcov f10/r001 prior01125 | 0.1125 | parallel accuracy | 1.638 / 0.1490 | 0.975 | 0.950 | 0.6000 | ignored | ignored | best train-q80 recall point, rotation worse |
| qcov f10/r001 prior01125 | 0.1125 | solo | 1.638 / 0.1490 | 0.975 | 0.950 | 0.6000 | 278.3 | 262.9 | recall candidate; no native speed win |
| qcov f10/r001 prior0125 | 0.125 | parallel accuracy | 1.625 / 0.1424 | 0.9625 | 0.950 | 0.5625 | ignored | ignored | improves R10/R5, but no R2 gain over qcov |
| qcov f10/r001 prior01375 | 0.1375 | parallel accuracy | 1.627 / 0.1256 | 0.9625 | 0.925 | 0.5625 | ignored | ignored | improves R10 only; no R5/R2 gain |
| qcov f05/r0005 prior010 | 0.10 | parallel accuracy | 1.674 / 0.1300 | 0.950 | 0.925 | 0.5750 | ignored | ignored | mild R2 gain, worse translation median |
| qcov f15/r0015 prior0125 | 0.125 | parallel accuracy | 1.696 / 0.1310 | 0.9625 | 0.9375 | 0.5500 | ignored | ignored | better R10/R5, but loses qcov R2 |

The prior01125 solo row is now the best train-q80 recall point in this branch:
versus corrected native8192 it changes dense recall by +2.5 pp R10, +2.5 pp
R5, and +5.0 pp R2, while keeping median translation slightly better
(1.638 cm vs 1.652 cm). It is not a clean pose-median or speed Pareto point:
rotation median worsens to 0.1490 deg, no-prior qcov still has better median
pose, and mean latency remains slower than corrected native8192 (278.3 ms vs
274.9 ms) even though median latency is essentially tied. Treat prior01125 as
the next train/self-map validation candidate, not as a paper-facing setting.

A shifted train-slice check then used `test_stride=3`. ShopFacade has 231 train
cameras, so this selects 77 cameras total and leaves 75 measured queries after
the two warmup cameras. The slice partially overlaps the first train-q80 set but
adds later train frames without using Cambridge test queries.

| Run | Profile mode | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train-s3/q75 native8192 | solo | 2.133 / 0.1193 | 0.9867 | 0.8667 | 0.4667 | 282.4 | 261.6 | shifted native baseline |
| train-s3/q75 qcov f10/r001 | parallel accuracy | 2.024 / 0.1193 | 0.9867 | 0.8667 | 0.4800 | ignored | ignored | improves median/R2 only |
| train-s3/q75 qcov f10/r001 prior01125 | parallel accuracy | 1.806 / 0.1211 | 0.9867 | 0.8800 | 0.5333 | ignored | ignored | shifted-slice recall gain |
| train-s3/q75 qcov f10/r001 prior01125 | solo | 1.806 / 0.1211 | 0.9867 | 0.8800 | 0.5333 | 295.2 | 282.3 | confirms recall; slower than native |

This shifted-slice check strengthens the train-only recall evidence for
qcov+prior01125: it improves translation median, R5, and strict R2 versus
native8192 on a broader train slice. It also confirms that this is not a speed
candidate: same-GPU solo latency is worse by +12.8 ms mean and +20.7 ms median.

## Interpretation

Supported:

- The cache audit gap is fixed for future selector training caches.
- Fixed fast-mode iteration caps are now reproducible and recorded in profiling
  manifests.
- On this q20 slice, audited selector8192 fast is better than native8192 fast at
  the same landmark budget and recall: 408.4 ms vs 423.3 ms mean latency.
- The fp07 selector-only training variant, where Gaussian advantage suppresses
  only high-score self-map false positives, improves the 8k fast diagnostic to
  401.5 ms mean latency while preserving native8192-fast R10/R5/R2.
- Audited selector12288 fast is faster than native12288 fast, 364.3 ms vs
  383.0 ms mean latency, with the same median and R10/R5.
- The hard-negative risk path is now implemented and auditable. The q20
  diagnostic used `hard_negative_weight=0.5` and
  `hard_negative_score_threshold=0.7`; its manifest reports
  `audit_status=passed` for the source cache and 2,099 high-score hard-negative
  pairs.
- The all-Gaussian candidate pool can improve loose q20 recall at 8k:
  R10/R5/R2 becomes 1.00/0.95/0.35, versus native8192 fast
  0.95/0.90/0.35. This is an accuracy diagnostic, not a usable speed result.
- The positive-support path is implemented and auditable. The smoke probe used
  `positive_support_score_threshold=0.7` and `positive_support_weight=0.5`;
  the manifest reports `audit_status=passed`, 1,479 positive pairs, and 473
  positive landmarks from the self-map cache.
- Sparse landmark prior overrides now let diagnostics exercise
  `sampled_scores.pkl`, which the base `stdloc_cambridge.yaml` otherwise leaves
  unused.
- The sparse-prior selection path is now moved off the test split: the new
  train/self-map prior calibration writes full profile manifests and keeps the
  q20 test prior005 rows as diagnostics only.
- Train-camera calibration found a fixed evaluator budget that preserves the
  8k native/fp07 recall pattern while reducing q20 test latency versus the
  earlier fast cap.
- Sparse-pose information export is implemented and auditable. On train-q20 it
  improves loose R10 and latency under the frozen budget, which makes it worth
  retaining as a diagnostic signal.
- Strict-support reservation is implemented and auditable. It repairs much of
  the first pose-information strict-recall loss on q20 (R2 0.20 -> 0.30) while
  preserving the speed benefit relative to native8192 under the frozen budget.
- The q20 train-only ret90/pose0.25/strict3px result was a useful stress test
  for the guard, but the larger q80 train check shows it does not generalize
  within the ShopFacade train split.
- Root-cause isolation shows the selector score is not the main problem on
  this slice: source-pool selector8192 matches native8192, while all-Gaussian
  replacement drops native hard-case support and degrades recall.
- Geometry-aware selector training is implemented and auditable at both
  landmark-target and pair-ranking levels. The pair loss uses only the audited
  self-map cache and keeps inference on the single STDLoc-compatible path.
- Selector score scale is now a concrete bottleneck: current selector logits
  are much lower variance than native source scores, so default source-pool
  exports barely change even when the geometry loss is active.
- Stronger selector-scale training can overcome that ordering bottleneck: the
  latest audited checkpoints raise gate std by about 6-8x and change 44-60
  source-pool ids, while still using only self-map supervision.
- Hard-query support export is implemented and auditable. It uses only the
  self-map cache and gives a direct diagnostic for low-margin hard-case
  protection without altering descriptors, test splits, or the evaluator.
- Query-coverage reservation is implemented and auditable. The f10/r001
  source-pool map gives a stronger train-q80 signal than scalar hard-query
  priors: it improves median translation/rotation versus native8192 and
  rankstrong while preserving rankstrong R10/R5/R2.
- Sparse-prior weighting on the qcov f10/r001 map can improve train-q80 recall:
  prior01125 reaches R10/R5/R2 = 0.975/0.950/0.600. This is a calibration
  diagnostic and next validation candidate, not a paper-facing result. A
  shifted train-s3/q75 slice preserves the recall direction, with R5/R2 rising
  from native 0.8667/0.4667 to 0.8800/0.5333.

Not supported:

- A paper-facing speed/accuracy claim. The selector12288 fast row loses R2 on
  this q20 slice, and q20 is too small for method selection.
- A realtime claim. The best diagnostic row is still about 364 ms/query on this
  machine.
- A strong accuracy claim. The audited selector-only maps do not dominate native
  12k across median, recall, and latency.
- The first hard-negative penalty probe is not better than the non-risk fast
  row: it improves median translation/rotation on q20, but drops R5 from 0.90
  to 0.85 and increases mean latency from 408.4 ms to 419.3 ms.
- The fp07 12k variant does not repair the selector12288 R2 drop; it keeps
  R2=0.25 and is slower than native12288 fast.
- The all-Gaussian maps are too slow under the current sparse matching path.
  The best all-Gaussian 8k recall row takes 494.0 ms mean latency and has only
  206.9 sparse inliers, suggesting the new landmarks improve final dense
  alignment but make sparse pose harder.
- Retaining part of the native sampled pool did not solve the speed/accuracy
  tradeoff. The ret75 map keeps 6,144 source-pool landmarks but drops R2 to
  0.20 and still takes 471.3 ms. The ret50 map keeps 4,096 source-pool
  landmarks and recovers R10/R5 to 1.00/0.95, but still takes 484.7 ms and
  drops R2 to 0.25.
- The 4k/2k fast-cap extension does not produce a Pareto point. Native 4k and
  2k are slower than native 8k/12k because sparse pose time rises above
  220 ms/query. The audited fp07 4k row also drops R5 to 0.85, and fp07 2k is
  essentially tied with native 2k while remaining about 494 ms/query.
- The positive-support smoke is not a main improvement. At 4k it raises R5 to
  0.95 but drops R2 to 0.15 and remains slow at 491.6 ms. At 8k it preserves
  the fp07/native recall pattern, but loses the fp07 speed gain:
  424.3 ms vs 401.5 ms.
- The sparse landmark prior is not yet a main speed/accuracy win. The 8k fp07
  test prior row is fast, but drops R2 from 0.35 to 0.30. The 12k test prior
  rows keep R10/R2 but drop R5 to 0.85, and the selector 12k prior row is
  slower than native 12k prior. The train/self-map prior calibration is also
  negative: prior005 preserves recall but increases mean latency for both
  native8192 and fp07 selector8192.
- A query-coverage paper-facing claim. The f10/r001 result is promising but is
  still one ShopFacade train-q80 calibration slice. The f25/r002 variant loses
  the rankstrong strict-R2 gain, and the lr3e3 f50/r002 variant trades better
  R2 for worse translation median. No test split or full Cambridge conclusion
  should be inferred.
- A query-coverage speed claim. The first solo f10/r001 run was faster than the
  old native8192 solo row, but a corrected same-GPU sequential repeat reversed
  the mean-latency delta. The robust evidence is accuracy/strict recall, not
  realtime performance.
- A qcov sparse-prior frozen-candidate claim. Prior0075 loses R10/R5,
  prior005 loses R5, prior0125 gives no strict-R2 gain over qcov no-prior, and
  even the stronger prior01125 recall row worsens rotation median and does not
  beat corrected native8192 on mean latency. On the shifted train-s3/q75 solo
  pair it is slower than native by +12.8 ms mean and +20.7 ms median.
- The train-calibrated budget is not a selector contribution by itself. Under
  the frozen budget, native8192 is faster than fp07 selector8192 on both train
  calibration and q20 test while keeping the same recall.
- The first pose-information ret75 map is not a main result. The q20 test row is
  faster, but strict R2 drops from 0.35 to 0.20 and median translation worsens.
- The strict-support guarded pose map is still not a main result. It improves
  median, R10/R5, and latency on the fixed q20 diagnostic, but strict R2 remains
  below native8192 (0.30 vs 0.35).
- The new ret90/pose0.25 strict-support row is train-only calibration evidence.
  It should not be described as a test result or paper-facing improvement until
  the recipe is frozen and evaluated without further test-driven adjustment.
- The larger train split does not support freezing ret90/pose0.25/strict3px:
  it loses dense median, R10, R5, and R2 versus native8192 while saving only
  2.7 ms/query on mean latency.
- Explicit native-source protection reduces the catastrophic all-Gaussian
  losses but still does not produce a same-budget candidate. The protnative95
  strict-support map and the no-strict protnative98 map both lose q80 train
  recall versus native8192; the latter is also slower.
- All-Gaussian replacement without a native-supported hard-case guard is not a
  viable main path. Even with the new explicit guard, replacing 161-352
  native8192 landmarks at the same 8k budget still loses native-supported
  cases on q80 train.
- The source-pool selector q80 solo run matches native8192 dense accuracy and
  improves mean latency by 3.2 ms, but median latency is slightly worse and the
  effect is too small to be paper-facing without repeats and cross-scene audit.
- Paired q80 train repeats reject that source-pool speed signal as unstable:
  accuracy remains matched, but selector is slower by 54-68 ms/query under
  paired multi-GPU load.
- The additive 12k map that preserves all native8192 landmarks and fills 4,096
  selector/pose landmarks is worse than native12288 on q80 train.
- Stronger selector-only training produces the first source-pool q80 strict-R2
  gain without R10/R5 loss, but the gain is one recovered train query and the
  solo runtime is slower than native8192.
- Pose-reliability selector targets are now implemented in
  `train_unified_lff.py` and recorded in training manifests/checkpoints. The
  targets are opt-in, come from the audited self-map reprojection cache, and do
  not touch test queries or evaluator behavior.
- The first pose-target probes were neutral. `posew05_score07` used 1,479
  positive self-map pairs over 473 landmarks and changed only one sampled index
  relative to `rankstrong_gate01_bias5`; its q80 train dense metrics are exactly
  the same as rankstrong (1.653 cm / 0.1382 deg, R10/R5/R2
  0.950/0.925/0.5625). A stronger `posew10_score00_gate1` export is identical
  to the rankstrong sampled set.
- The first pair-level pose-supervision probes are also neutral. They record
  the intended self-map support counts and nonzero pose-pair losses, but change
  only 1-3 sampled ids versus rankstrong and do not improve q80 train accuracy.
- Selector amplification is not yet a main result. `sw10` keeps the rankstrong
  recall pattern but worsens median pose; `minmax` improves strict R2 and median
  translation but drops R5; rank normalization badly degrades all dense recall
  thresholds.
- The finer minmax weight sweep does not find a hidden Pareto point. Weights
  0.25/0.30 preserve R5/R2 but worsen median pose, while weights 0.35-0.75
  either lose R5 without gaining R2 or gain R2 only by losing R5.
- Stronger training-side selector scaling is not a frozen candidate. It moves
  the source-pool map more than the first pose-target and pose-pair probes, but
  the best rank-only variant only matches rankstrong recall with worse median,
  and the pose-pair variants either lose R5/R2 or fail to recover the
  rankstrong strict-R2 gain.
- The first hard-query support prior is not a frozen candidate. Mild weighting
  removes the rankstrong strict-R2 gain, and stronger weighting worsens median
  translation/rotation and strict R2.
- A first non-ShopFacade audited-cache transfer was completed for KingsCollege.
  The cache, fixed selector-only checkpoint, and qcov export are:

```text
output/unified_lff_v2/episode_caches_audited_20260517/KingsCollege/scene_match_pairs.pt
output/unified_lff_v2/train_selector_only_audited_20260517/KingsCollege_rankstrong_gate01_bias5/unified_lff_v2.pt
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_r001_selector8192_cov8/KingsCollege
```

  The cache records `feedback_bank_split_name=selfmap_train_rendered` and
  `split_audit.audit_status=passed` with no overlap against KingsCollege test
  image ids. The fixed recipe was intentionally reused from ShopFacade rather
  than selected from evaluation metrics. Profiling KingsCollege needed
  `--images .` because this scene has images directly under `seq*/`; the
  ShopFacade `--images processed` path does not exist for KingsCollege.
- KingsCollege train-q80 same-GPU solo diagnostics are mixed, not a frozen
  candidate:

| Run | Median cm/deg | R10 | R5 | R2 | Mean ms | Median ms | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| native8192 | 12.471 / 0.1805 | 0.3375 | 0.1000 | 0.0250 | 293.1 | 280.6 | Kings train-q80 baseline |
| rankstrong qcov f10/r001 | 12.272 / 0.1813 | 0.3250 | 0.1125 | 0.0250 | 290.8 | 281.6 | improves translation median/R5, loses R10 |

  This is useful cross-scene evidence that the audited pipeline works and can
  move the map, but it weakens the hypothesis that qcov f10/r001 is already a
  robust frozen recipe. The profile split audits are still `unknown` and
  `paper_safe=false`, so the rows remain train-only diagnostics.

## Five-Scene Train-Q80 Transfer Status

The remaining Cambridge scenes were then processed with the same fixed
rankstrong/qcov recipe, without using test queries or evaluation results for
training, calibration, or recipe selection. The new audited caches/checkpoints
and exports are:

```text
output/unified_lff_v2/episode_caches_audited_20260517/{GreatCourt,OldHospital,StMarysChurch}/scene_match_pairs.pt
output/unified_lff_v2/train_selector_only_audited_20260517/{GreatCourt,OldHospital,StMarysChurch}_rankstrong_gate01_bias5/unified_lff_v2.pt
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_r001_selector8192_cov8/{GreatCourt,OldHospital,StMarysChurch}
```

All three cache/train/export split audits passed at the self-map artifact
level with `split=selfmap_train_rendered` and zero overlap against the official
test image ids. The profile sidecars still mark split audit as `unknown`
because profile outputs record the evaluator split but do not themselves attach
a full self-map disjointness audit, so these rows remain diagnostic.

| Scene | Native cm/deg | Qcov cm/deg | Native R10/R5/R2 | Qcov R10/R5/R2 | Native mean/median ms | Qcov mean/median ms | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ShopFacade | 1.652 / 0.1382 | 1.610 / 0.1343 | 0.950 / 0.925 / 0.550 | 0.950 / 0.925 / 0.5625 | 274.9 / 263.0 | 281.5 / 265.9 | accuracy/R2 up, slower |
| KingsCollege | 12.471 / 0.1805 | 12.272 / 0.1813 | 0.3375 / 0.1000 / 0.0250 | 0.3250 / 0.1125 / 0.0250 | 293.1 / 280.6 | 290.8 / 281.6 | mixed |
| GreatCourt | 6.756 / 0.0345 | 6.817 / 0.0366 | 0.8500 / 0.3125 / 0.0000 | 0.8375 / 0.2750 / 0.0250 | 626.4 / 635.0 | 622.6 / 634.1 | loose recall/median worse |
| OldHospital | 9.020 / 0.1859 | 9.738 / 0.1962 | 0.5125 / 0.3000 / 0.0000 | 0.5125 / 0.2500 / 0.0000 | 650.2 / 663.6 | 634.2 / 674.4 | worse median/R5 |
| StMarysChurch | 11.701 / 0.3951 | 9.847 / 0.3614 | 0.4875 / 0.3250 / 0.1000 | 0.5000 / 0.3500 / 0.1000 | 653.4 / 664.7 | 642.8 / 679.6 | accuracy/recall up, median latency worse |

This rejects qcov f10/r001 as a frozen cross-scene recipe. It has real positive
signals on ShopFacade and StMarysChurch, but GreatCourt and OldHospital lose
recall/median pose, and no scene provides a clean accuracy/recall/runtime
Pareto win versus native8192.

## Qcov 12k Budgeted Pareto Follow-Up

The same audited query-coverage objective was then exported at a 12,288-landmark
budget to test whether the 8k transfer was failing mainly from over-aggressive
budget pressure:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_r001_selector12288_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q40_q80_qcov12_budgeted_pareto.json
```

All five exports inherit `split=selfmap_train_rendered` and passed cache split
audits. The q40 smoke was mixed but speed-positive, so a train-q80 same-GPU
confirmation was run. Profile split audits remain `unknown`, so these are
diagnostic train rows only.

| Scene | Native12 cm/deg | Qcov12 cm/deg | Native R10/R5/R2 | Qcov12 R10/R5/R2 | Native mean/median ms | Qcov12 mean/median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 1.533 / 0.1018 | 1.533 / 0.1018 | 0.9625 / 0.9000 / 0.6500 | 0.9625 / 0.9000 / 0.6500 | 641.1 / 658.3 | 617.6 / 648.8 |
| KingsCollege | 12.060 / 0.1867 | 12.180 / 0.1848 | 0.3000 / 0.1000 / 0.0250 | 0.3250 / 0.1000 / 0.0250 | 657.2 / 665.2 | 653.6 / 658.6 |
| GreatCourt | 5.914 / 0.0330 | 5.911 / 0.0333 | 0.8250 / 0.4000 / 0.0250 | 0.8250 / 0.4125 / 0.0250 | 620.4 / 623.9 | 604.7 / 612.1 |
| OldHospital | 7.743 / 0.1695 | 8.458 / 0.1763 | 0.5500 / 0.3000 / 0.0000 | 0.5375 / 0.2875 / 0.0000 | 483.8 / 480.6 | 442.8 / 454.1 |
| StMarysChurch | 5.407 / 0.2010 | 5.530 / 0.2020 | 0.6125 / 0.4500 / 0.1500 | 0.6000 / 0.4500 / 0.1500 | 478.0 / 475.9 | 469.1 / 467.2 |

This is a useful speed/accuracy frontier diagnostic, not a frozen candidate.
It improves same-budget mean/median latency on every train-q80 scene pair, but
OldHospital and StMarysChurch lose accuracy or R10, and KingsCollege trades
translation median for R10/rotation. The next viable direction is to preserve
the qcov12 sparse-stage speed signal while adding a self-map guard against
scene-specific recall losses.

Guard follow-up:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_scoreonly_protnative100_selector12288_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_qcov12_guard_followup.json
```

The score-only protected export keeps the native12 sampled ids exactly and
therefore tests selector/source scoring plus locability without landmark
replacement. It preserves native12 accuracy on all five train-q80 scenes. The
speed result is weaker than qcov12: GreatCourt and StMarysChurch gain useful
latency, OldHospital gains only mean latency while losing median latency, and
ShopFacade/KingsCollege are tied or slower.

| Scene | Native12 mean/median ms | Score-only mean/median ms | Accuracy |
| --- | ---: | ---: | --- |
| ShopFacade | 641.1 / 658.3 | 641.4 / 664.3 | unchanged |
| KingsCollege | 657.2 / 665.2 | 658.0 / 664.5 | unchanged |
| GreatCourt | 620.4 / 623.9 | 603.2 / 612.4 | unchanged |
| OldHospital | 483.8 / 480.6 | 481.6 / 492.4 | unchanged |
| StMarysChurch | 478.0 / 475.9 | 469.7 / 471.9 | unchanged |

A smaller hard-query reservation with 99.5% native-source protection was tested
only on the two qcov12 loss scenes. It helps StMarysChurch
(5.383 / 0.1911, R10/R5/R2 0.6250/0.4625/0.1500, 472.8/475.4 ms), but hurts
OldHospital (8.884 / 0.1819, R10/R5/R2 0.5375/0.3000/0.0000, 494.2/498.9 ms).
This is still a mixed diagnostic, not a global recipe.

A smaller qcov reservation without native protection was then checked on the
same two loss scenes:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f005_r001_selector12288_cov8/{OldHospital,StMarysChurch}
output/unified_lff_v2/reports/20260517_train_q80_qcov12_lowfraction_loss_scene_followup.json
```

It lowers `query_coverage_fraction` to 0.005 while keeping the 12k source-pool
budget, but it exactly matches the qcov12 loss metrics on both scenes:
OldHospital is 8.458 / 0.1763 with R10/R5/R2 = 0.5375/0.2875/0.0000, and
StMarysChurch is 5.530 / 0.2020 with R10/R5/R2 =
0.6000/0.4500/0.1500. The profiles were parallel accuracy checks, so timing is
ignored. This rejects the idea that the loss is solved by simply reducing qcov
reservation count; the admitted qcov landmarks need a stronger self-map
geometry filter or native-hard-case protection.

A sampled-set audit then separated the query-coverage prefix from the
source-pool fill. The f005 and f10 maps are identical on both loss scenes
because the available source-pool qcov candidates saturate before the f005
budget. A no-qcov source-pool fill simulation remains within 4 symdiff IDs of
qcov12 on OldHospital and 2 on StMarysChurch, while still dropping 119/44
native12 IDs. Raising `source_score_weight` from 1.0 through 8.0 leaves those
dropped-native counts unchanged. Hard-query support barely overlaps the
dropped native IDs: 0 on OldHospital and 1 on StMarysChurch.

Protected fill isolation:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_scoreonly_protnative100_blend010_selector12288_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_scoreonly_protnative100_blend020_selector12288_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_protnative100_selector12288_cov8/{OldHospital,StMarysChurch}
output/unified_lff_v2/reports/20260517_train_q80_protected_qcov_fill_isolation.json
output/unified_lff_v2/reports/20260517_train_q80_scoreonly_blend020_followup.json
```

The blend010 score-only export preserves all native12 sampled IDs and matches
native12 macro train-q80 dense accuracy exactly: 6.531 cm / 0.1384 deg with
R10/R5/R2 = 0.6500/0.4300/0.1700. Timing is ignored because these were
parallel accuracy profiles. The blend020 repeat gives the same accuracy-clean
result with all native12 sampled IDs preserved; it is also parallel-only timing
evidence and remains a safety control rather than a speed candidate.

The qcov f10 plus `protected_source_fraction=1.0` export reserves the qcov
prefix, then fills the remaining budget from native12. It changes only 2/1
native IDs on OldHospital/StMarysChurch and recovers the native12 metrics:

| Scene | Native12 dense | Qcov12 dense | Qcov+protnative100 dense | Native IDs dropped | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| OldHospital | 7.743 / 0.1695 | 8.458 / 0.1763 | 7.743 / 0.1695 | 2 | recovers native12 |
| StMarysChurch | 5.407 / 0.2010 | 5.530 / 0.2020 | 5.406 / 0.2010 | 1 | recovers native12 |

These rows are structural diagnostics, not a paper candidate. They show the
OldHospital/StMarysChurch qcov12 losses come from broad source-pool
replacement/fill, not from the qcov prefix by itself.

Native12-subset budget follow-up:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_sourcescore10240_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_native12sub10240_selector_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_native12sub10240_followup.json
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_sourcescore11264_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_native12sub11264_selector_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_native12sub11264_followup.json
```

This tried to get speed by dropping only native12 IDs, avoiding the unsafe
non-native source-pool replacement. The 10,240 selector subset is slightly
better than its source-score control, but still regresses native12 macro dense
accuracy: 7.215 cm / 0.1540 deg with R10/R5/R2 =
0.6275/0.4125/0.1575, versus native12 6.531 cm / 0.1384 deg and
0.6500/0.4300/0.1700. The less aggressive 11,264 subset still loses median,
R10, and R2: 7.060 cm / 0.1592 deg with R10/R5/R2 =
0.6400/0.4300/0.1550. It is rejected before solo timing. The remaining loss
scenes are OldHospital and StMarysChurch, and the dropped IDs overlap only
4 self-map-supported landmarks on each loss scene, so a simple support guard is
unlikely to repair this route.

Fixed prior005 follow-up:

```text
output/unified_lff_v2/profile_20260517/{scene}_train_q80_native12288_prior005_traincal_s5k50_d100_10_parallel_acc
output/unified_lff_v2/profile_20260517/{scene}_train_q80_scoreonly_blend020_prior005_traincal_s5k50_d100_10_parallel_acc
output/unified_lff_v2/reports/20260517_train_q80_scoreonly_blend020_prior005_followup.json
```

This keeps the native12 sampled set and enables fixed global
`--sparse_landmark_prior_weight 0.05` plus
`--dense_locability_prior_weight 0.05` for both native12 and score-only
blend020. It is rejected before solo timing. Native12 prior005 already hurts
the no-prior native12 baseline: 6.816 cm / 0.1503 deg with R10/R5/R2 =
0.6450/0.4275/0.1700. Score-only blend020 prior005 is still worse than
native12 no-prior on median, R10, and R2: 6.894 cm / 0.1516 deg with
0.6400/0.4300/0.1650. OldHospital and StMarysChurch remain the loss scenes.

Dense early-exit simulation:

```text
output/unified_lff_v2/reports/20260517_train_q80_native12_dense_early_exit_sweep.json
```

This uses existing native12 train-q80 solo per-query sparse/dense results and
timing. It simulates a fixed inlier-only policy: keep the sparse pose and skip
dense refinement when sparse inliers exceed a threshold. It is not paper-facing
because any threshold would need predeclared self-map validation. The diagnostic
is negative: threshold 700 skips 3.0% of queries and saves about 4.3 ms mean,
but drops R10/R5 from 0.6500/0.4300 to 0.6425/0.4275. The only non-regressive
threshold is 800, which skips 1/400 queries and saves about 0.33 ms mean.
Sparse inlier count alone is not a useful dense early-exit signal.

Dense-oracle ceiling audit:

```text
output/unified_lff_v2/reports/20260517_train_q80_native12_dense_oracle_ceiling.json
```

Using the same existing native12 train-q80 solo profiles, dense refinement
improves translation on 333/400 queries and worsens it on 67/400. A strict GT
oracle that skips dense only when sparse has no worse translation and rotation
error would skip 42/400 queries, improve macro median from 6.531 to 6.236 cm,
raise R10/R5/R2 from 0.6500/0.4300/0.1700 to 0.6750/0.4550/0.1750, and save
2.0% mean latency. This is diagnostic only and cannot be used as a deployed
per-query branch selector. A looser threshold-safe oracle has larger speed
headroom, skipping 245/400 queries and saving 12.5% mean latency, but it
worsens macro median by 0.607 cm. The observable signals already logged before
dense are too weak for a policy: sparse inliers have only AUC 0.543 for the
safe-skip target. The next justified step is wrapper-side reliability
instrumentation on train/self-map data, not an early-exit implementation.

Reliability-signal follow-up:

```text
output/unified_lff_v2/profile_20260517/{scene}_train_q80_native12288_reliability_traincal_s5k50_d100_10_parallel
output/unified_lff_v2/reports/20260517_train_q80_native12_reliability_signal_audit.json
```

The profiling wrapper now records solve-pose reliability metadata in
`timing_profile.json` without changing vendored STDLoc behavior: match count,
inlier count/ratio, and all-match/inlier reprojection residual
mean/median/p90. A ShopFacade q2 smoke confirmed the fields are emitted, then
the five train-q80 native12 profiles were rerun in parallel for reliability
analysis only. Timing from these parallel profiles is ignored. The result is
still not strong enough for a fast-mode candidate: the best oriented strict
sparse-dominates-dense predictor is `neg_sparse_inlier_reprojection_p90_px`
with AUC 0.581. The best single-feature threshold selected on this diagnostic
split skips 14/400 queries (3.5%) with no macro median/recall regression. This
is better than raw high-inlier early exit, but it remains train-slice selected
and should not be implemented without predeclared self-map calibration plus a
shifted train validation.

Pose-pair training follow-up:

```text
output/unified_lff_v2/train_selector_only_audited_20260517/{GreatCourt,OldHospital,StMarysChurch}_posepair_w2_t8s0_rankstrong_gate01_bias5
output/unified_lff_v2/selector_only_audited_resample_20260517/posepair_w2_t8s0_qcov_f10_r001_selector12288_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_posepair_qcov12_followup.json
```

This used audited self-map caches and selector-only native descriptors with
`lambda_selector_pose_pair=2.0`, `selector_pose_pair_margin=0.2`,
`pose_pair_reprojection_threshold_px=8.0`, and
`pose_pair_score_threshold=0.0`. The training split audits pass, and the export
query-coverage audits pass, but the candidate does not materially change the
qcov12 map:

| Scene | Delta vs rankstrong qcov12 sampled set | Pose-pair train-q80 dense | R10/R5/R2 | Conclusion |
| --- | ---: | ---: | ---: | --- |
| GreatCourt | +0 ids | 5.911 / 0.0333 | 0.8250/0.4125/0.0250 | same as qcov12 |
| OldHospital | +0 ids | 8.458 / 0.1763 | 0.5375/0.2875/0.0000 | does not fix native12 regression |
| StMarysChurch | +1 id | 5.531 / 0.2020 | 0.6000/0.4500/0.1500 | does not fix native12 median/R10 regression |

The three profiles were run concurrently on GPUs 0/1/2, so their timing is
audit material only and should not be compared against solo timing rows. The
branch is rejected for candidate selection because it leaves source-pool
ordering essentially unchanged and does not repair the qcov12 accuracy losses.

Additive qcov follow-up:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_plus_qcov12_union_idx_20260517
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_plus_qcov12_union_traincal/{scene}
output/unified_lff_v2/reports/20260517_train_q80_native12_plus_qcov12_additive_followup.json
```

This export keeps all native12 sampled IDs and adds the qcov12 non-native
sampled IDs as a protected static set. The set audits pass for all five scenes:
`split=selfmap_train_rendered`, native IDs missing = 0, qcov IDs missing = 0,
and no extra IDs beyond the union. Accuracy is still mixed:

| Scene | Native12 dense | Additive union dense | Native R10/R5/R2 | Additive R10/R5/R2 | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| GreatCourt | 5.914 / 0.0330 | 5.887 / 0.0329 | 0.8250/0.4000/0.0250 | 0.8375/0.4250/0.0250 | useful train gain |
| OldHospital | 7.743 / 0.1695 | 8.590 / 0.1695 | 0.5500/0.3000/0.0000 | 0.5375/0.3000/0.0000 | rejected |
| StMarysChurch | 5.407 / 0.2010 | 5.406 / 0.2010 | 0.6125/0.4500/0.1500 | 0.6125/0.4500/0.1500 | recovers qcov12 loss |

A capped variant that keeps all native12 IDs and at most 45 qcov12 non-native
additions was checked on OldHospital and StMarysChurch. StMarysChurch remains
recovered, but OldHospital still fails at 8.276 cm / 0.1819 deg with
R10/R5/R2 = 0.5375/0.2875/0.0000. The failure therefore is not just native-ID
dropout; added qcov landmarks can perturb sparse matching. The persistent
OldHospital failure case is `seq1/frame00027.png`: native12 is 6.885 cm, while
the additive maps give 26.897 cm. These additive maps should not be promoted to
test diagnostics.

A map-level churn guard remains diagnostic only: choose additive union only
when qcov12 adds at most 64 non-native IDs, otherwise use native12. That rule
uses self-map/map-set churn rather than query errors and gives macro train-q80
dense median 6.526 cm, R10 0.6525, R5 0.4350 versus native12 6.531 cm, 0.6500,
0.4300. The gain is too small to claim and needs predeclared validation plus
solo timing before any test run.

The rule was then predeclared and checked on a shifted train slice
(`test_stride=3`, `max_test_cameras=75`, `warmup_cameras=2`):

```text
output/unified_lff_v2/reports/20260517_train_s3_q75_churn64_guard_validation.json
```

It selects additive union for ShopFacade, KingsCollege, GreatCourt, and
StMarysChurch, and native12 for OldHospital. Against native12 on the same
shifted slice, macro dense median/R10/R2 are unchanged at 6.294 cm, 0.6907,
and 0.1840; macro R5 improves from 0.4507 to 0.4533, which is one GreatCourt
query. This validates the guard as accuracy-neutral train evidence, but not as
a speed candidate. The selected maps are native12-sized or larger and the runs
were parallel accuracy profiles, so timing is ignored and no solo timing was
run. Keep the guard off Cambridge test unless a future same-budget export rule
can retain this accuracy while preserving the qcov12 speed signal.

That future same-budget check was then run as a train-only diagnostic:

```text
output/unified_lff_v2/reports/20260517_train_q40_q80_qcov12_samebudget_churn_guard_offline.json
output/unified_lff_v2/reports/20260517_train_s3_q75_qcov12_samebudget_lowchurn_guard_validation.json
```

The qcov-added counts are ShopFacade 7, GreatCourt 41, StMarysChurch 45,
KingsCollege 46, and OldHospital 121. The exact train-q80 candidate is
threshold 7, which selects only ShopFacade, leaves dense q80 metrics unchanged,
and references -4.71 ms macro total-mean latency from the solo q80 rows.
Threshold 41 selects ShopFacade and GreatCourt, references -7.85 ms q80 macro
latency and +0.0025 q80 macro R5, but has a tiny train-q80 angle drift.

Shifted train-s3/q75 validation keeps threshold 7 exact: macro dense TE and
recalls are unchanged and angle improves by 0.000002 deg. Threshold 41 gains
one GreatCourt R5 query on the shifted slice, but still has a small macro angle
regression (+0.000247 deg). This same-budget guard is cleaner than the
additive guard, but it is not strong enough for a paper-facing fast mode:
threshold 7 affects only one scene, while threshold 41 is train-selected and
not strictly pose-neutral.

Next:

- Select any fast-mode caps only from train/self-map validation, not q20/full
  test outcomes.
- Add sparse-pose-aware sampling pressure before spending full Cambridge GPU
  time. The current all-Gaussian score improves dense recall on q20 but reduces
  sparse pose efficiency and fails the larger q80 train gate.
- Do not freeze another same-budget all-Gaussian replacement recipe from these
  rows. The protected-source guard is useful for diagnostics, but the remaining
  budget does not yet add enough value to offset dropped native landmarks.
- The simple pose-target and pose-pair losses are not enough at the current
  export scale. Prioritize calibrated selector normalization/temperature or a
  larger-margin geometry objective that changes source-pool ordering without
  becoming a per-query branch selector. A simple minmax export-weight sweep is
  now also rejected. Source-pool runtime stability, the current additive 12k
  fill, and the first geometry-aware selector blends all failed their train
  gates.
- Treat smaller landmark budgets as unsafe unless the sampling objective also
  improves PnP inlier geometry or confidence. The current score mostly changes
  selection/ranking, but it does not reduce sparse pose cost at 4k/2k.
- Treat qcov12 as a guarded-speed diagnostic: it reduces latency across all
  five train-q80 scene pairs, but the OldHospital/StMarysChurch losses block a
  paper-facing candidate. The same-budget low-churn guard validates only a
  small safe subset, so it does not solve the speed/accuracy goal.
- The first native-source guard shows the replacement set is the accuracy risk:
  score-only protection preserves accuracy but does not retain the full qcov12
  speed gain, while hq005/protnative995 remains scene-specific.
- Lowering qcov reservation to f005 on the two qcov12 loss scenes is also
  rejected: it matches the original qcov12 loss metrics and does not deserve
  solo timing. The protected-qcov fill isolation recovers both loss scenes
  while changing only 2/1 native IDs, so the unsafe part is source-pool fill,
  not the qcov prefix alone.
- Native12-subset budget reduction is rejected as a fast-mode route for now:
  10k and 11,264 selector subsets avoid non-native replacement, but still lose
  train-q80 median/R10/R2 before solo timing is justified.
- Fixed prior005 on score-only blend020 is also rejected. The native12 sampled
  set is preserved, but prior weighting hurts median/R10/R2 and does not
  create a safe speed/accuracy candidate.
- Inlier-only dense early exit is rejected as a practical fast-mode route:
  meaningful skip rates lose train-q80 recall/median, while the only
  non-regressive threshold skips just 1/400 queries. The dense-oracle ceiling
  has modest strict-error headroom, but logged pre-dense signals are too weak.
  The new residual/inlier-ratio instrumentation gives a best diagnostic
  threshold that skips 14/400 queries with no macro regression, but AUC is only
  0.581 and the threshold is train-slice selected. Revisit this only after
  predeclared self-map calibration and shifted train validation.
- The pose-pair qcov12 training follow-up is also rejected: it passes audit but
  does not move the sampled set enough to change the qcov12 failure modes.
- The native12-plus-qcov12 additive follow-up is rejected too. Full union and
  cap45 both preserve all native IDs, yet OldHospital still loses train-q80
  median/R10 because extra qcov landmarks perturb sparse matching. Treat the
  qcov-churn guard only as a map-level diagnostic: the shifted train-s3/q75
  validation is accuracy-neutral, but its gain is only one GreatCourt R5 query
  and the larger maps do not provide a same-budget speed mechanism. The
  separate same-budget low-churn guard preserves a speed mechanism but is too
  small at the exact-safe threshold.
- If sparse priors are pursued, select the prior weight from self-map
  validation. The q20 prior005 rows are diagnostics only.
- Do not promote these rows to paper-safe tables until full split/audit bundles
  are attached.
