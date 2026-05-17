# Loc-GS Speed/Accuracy Probe, 2026-05-17

## Scope

This is a diagnostic ShopFacade q20 probe for the sampling-field budget line in
`docs/sampling_field_budget_20260516.md`. It is not paper-facing because the
split audit is intentionally marked `unknown` until self-map/calibration ids are
audited against Cambridge test ids.

The profiled path is still the single STDLoc-compatible deployment path:

```text
native descriptors + sampled detector landmarks
  -> feature matching
  -> OpenCV PROSAC/RANSAC PnP
  -> STDLoc dense refinement
```

No vendored evaluator files under `third_party/stdloc` were edited. Profiling is
done by `loc_gs.scripts.profile_stdloc_native`, which wraps native STDLoc calls
from the Loc-GS side and writes `summary.json`, `timing_profile.json`,
`manifest.json`, `command.txt`, `metrics_summary.json`, `split_audit.json`, and
`git_status.txt`.

## Artifact Fix

The clean/resampled detector folders originally contained rebuilt
`sampled_idx.pkl` and `sampled_scores.pkl` but were missing native detector
support files such as `30000_detector.pth`. Existing budget maps were repaired
by materializing non-sampled detector support files from the clean native source
while preserving each map's sampled payload.

This makes the budget maps rerunnable without relying on external detector
symlinks:

```text
output/unified_lff_v2/native_rebuilt_detector_source_20260516/{scene}
output/unified_lff_v2/sampling_budget_20260516/*/{scene}
output/unified_lff_v2/fast_budget_20260517/*/{scene}
```

## Solo q20 Timing

All rows below are ShopFacade test split, `max_test_cameras=20`,
`warmup_cameras=2`, one dense iteration, same GPU, no parallel contention.
Latency columns are mean milliseconds per query.

| Run | Landmarks | Median cm | Median deg | R10 | R5 | R2 | Total ms | Sparse ms | Sparse pose ms | Dense ms | Sparse inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native2048 | 2048 | 2.507 | 0.112 | 1.00 | 0.90 | 0.30 | 1059.8 | 862.5 | 772.7 | 171.7 | 157.2 |
| selector2048 cov8 | 2048 | 2.468 | 0.115 | 0.95 | 0.90 | 0.25 | 1227.6 | 1032.0 | 932.8 | 169.8 | 148.2 |
| native4096 | 4096 | 2.686 | 0.116 | 1.00 | 0.90 | 0.25 | 572.6 | 378.4 | 299.5 | 168.7 | 226.2 |
| selector4096 cov8 | 4096 | 2.316 | 0.116 | 1.00 | 0.90 | 0.25 | 695.5 | 505.8 | 407.8 | 164.0 | 202.9 |
| native8192 | 8192 | 2.277 | 0.117 | 0.95 | 0.90 | 0.35 | 468.2 | 268.0 | 136.8 | 174.6 | 299.1 |
| selector8192 | 8192 | 2.250 | 0.118 | 1.00 | 0.95 | 0.35 | 477.5 | 281.2 | 176.0 | 170.7 | 269.8 |
| selector-only-train8192 cov8 | 8192 | 2.246 | 0.117 | 0.95 | 0.90 | 0.30 | 486.0 | 283.7 | 138.2 | 176.5 | 297.5 |
| native12288 | 12288 | 2.216 | 0.117 | 1.00 | 0.90 | 0.30 | 394.5 | 200.3 | 106.0 | 168.4 | 325.9 |
| selector12288 cov8 | 12288 | 2.192 | 0.127 | 1.00 | 0.85 | 0.40 | 450.7 | 253.3 | 117.4 | 171.9 | 313.9 |

## Fast-Cap Extension

The audited fp07 selector-only checkpoint was also exported at 4k and 2k
source-pool budgets, then profiled with the same fixed fast caps used in
`docs/audited_sampling_fast_probe_20260517.md`:

```text
sparse max/min iterations = 20000 / 100
dense max/min iterations = 500 / 50
```

| Run | Landmarks | Median cm/deg | R10 | R5 | R2 | Total ms | Sparse ms | Sparse pose ms | Dense ms | Sparse inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native4096 fast | 4096 | 2.686 / 0.1164 | 1.00 | 0.90 | 0.25 | 482.9 | 309.9 | 222.0 | 147.3 | 226.1 |
| audited fp07 selector4096 fast | 4096 | 2.686 / 0.1164 | 1.00 | 0.85 | 0.25 | 493.0 | 318.7 | 222.2 | 148.8 | 226.1 |
| audited fp07 pos07w05 selector4096 fast | 4096 | 2.516 / 0.1207 | 1.00 | 0.95 | 0.15 | 491.6 | 319.6 | 219.3 | 146.4 | 228.3 |
| native2048 fast | 2048 | 2.508 / 0.1124 | 1.00 | 0.90 | 0.30 | 493.8 | 320.1 | 228.7 | 148.1 | 157.2 |
| audited fp07 selector2048 fast | 2048 | 2.570 / 0.1137 | 1.00 | 0.90 | 0.30 | 494.4 | 322.4 | 228.1 | 146.4 | 156.6 |
| audited fp07 pos07w05 selector8192 fast | 8192 | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 424.3 | 251.2 | 133.7 | 147.6 | 299.4 |
| native8192 fast prior005 | 8192 | 2.321 / 0.1156 | 0.95 | 0.90 | 0.30 | 414.5 | 244.9 | 137.2 | 143.9 | 297.1 |
| audited fp07 selector8192 fast prior005 | 8192 | 2.323 / 0.1162 | 0.95 | 0.90 | 0.30 | 386.3 | 219.3 | 137.4 | 141.6 | 296.9 |
| native12288 fast prior005 | 12288 | 2.225 / 0.1201 | 1.00 | 0.85 | 0.30 | 390.4 | 221.2 | 108.7 | 144.4 | 323.1 |
| audited fp07 selector12288 fast prior005 | 12288 | 2.291 / 0.1195 | 1.00 | 0.85 | 0.30 | 399.6 | 222.1 | 108.8 | 151.7 | 322.9 |

This extension is negative evidence for a pure landmark-budget speed claim.
Below 8k, sparse pose time increases enough to erase any matching savings, and
the audited selector does not improve the 4k/2k Pareto frontier.
The positive-support smoke uses audited self-map high-score true positives. It
can raise loose R5 at 4k, but it hurts strict R2 and does not create a faster
row at 4k or 8k.
The prior005 rows explicitly enable `sampled_scores.pkl` in sparse matching via
`sparse_landmark_prior_weight=0.05`. This exposes a speed lever at 8k, but it
still trades away strict recall and is therefore not a paper-facing setting.

## Train-Split Prior Calibration

The sparse-prior setting was then checked on ShopFacade train cameras, using the
same q20 cap and fast-mode iteration caps. This keeps prior selection off the
test split. These rows are calibration diagnostics only; their sidecar
`split_audit.json` files record `status=unknown` and `paper_safe=false`.

| Run | Split | Median cm/deg | R10 | R5 | R2 | Total ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train native8192 fast prior000 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 340.9 | 179.9 | 88.7 | 91.2 | 135.3 | 352.1 |
| train native8192 fast prior005 | train | 1.525 / 0.0489 | 0.95 | 0.95 | 0.65 | 347.3 | 187.0 | 89.9 | 97.2 | 134.5 | 349.9 |
| train fp07 selector8192 fast prior000 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 338.5 | 172.5 | 88.1 | 84.3 | 140.2 | 352.4 |
| train fp07 selector8192 fast prior005 | train | 1.571 / 0.0502 | 0.95 | 0.95 | 0.65 | 366.4 | 196.1 | 89.4 | 106.7 | 144.5 | 350.2 |

On this self-map slice, `sparse_landmark_prior_weight=0.05` keeps recall fixed
but increases latency for both native8192 and fp07 selector8192. That means the
faster fp07 prior005 row on the test q20 slice should not be used for method
selection.

## Train-Calibrated Budget Control

Fast-mode iteration caps were also calibrated on ShopFacade train cameras before
one fixed q20 test diagnostic. The selected budget was:

```text
sparse max/min iterations = 5000 / 50
dense max/min iterations = 100 / 10
```

Train calibration:

| Run | Split | Median cm/deg | R10 | R5 | R2 | Total ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train native8192 fast s20k100 d500/50 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 340.9 | 179.9 | 88.7 | 91.2 | 135.3 | 352.1 |
| train fp07 selector8192 fast s20k100 d500/50 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 338.5 | 172.5 | 88.1 | 84.3 | 140.2 | 352.4 |
| train native8192 budget s5k50 d250/25 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 321.3 | 173.1 | 60.9 | 112.2 | 122.5 | 352.1 |
| train fp07 selector8192 budget s5k50 d250/25 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 322.4 | 170.0 | 60.8 | 109.2 | 126.7 | 352.4 |
| train native8192 budget s5k50 d100/10 | train | 1.559 / 0.0486 | 0.95 | 0.95 | 0.65 | 316.0 | 177.9 | 60.9 | 117.0 | 112.4 | 352.1 |
| train fp07 selector8192 budget s5k50 d100/10 | train | 1.559 / 0.0491 | 0.95 | 0.95 | 0.65 | 318.7 | 177.6 | 60.9 | 116.8 | 115.2 | 352.4 |
| train fp07 pose05 ret75 budget s5k50 d100/10 | train | 1.816 / 0.0575 | 1.00 | 0.95 | 0.65 | 303.0 | 160.3 | 61.1 | 99.2 | 117.1 | 324.7 |
| train fp07 pose05 strict3px f05 ret75 budget s5k50 d100/10 | train | 1.521 / 0.0470 | 1.00 | 0.95 | 0.65 | 312.1 | 169.9 | 57.4 | 112.4 | 116.6 | 364.9 |
| train fp07 pose025 strict3px f05 ret90 budget s5k50 d100/10 | train | 1.509 / 0.0496 | 1.00 | 0.95 | 0.65 | 283.9 | 140.6 | 55.4 | 85.2 | 117.6 | 382.6 |

Frozen q20 test diagnostic:

| Run | Split | Median cm/deg | R10 | R5 | R2 | Total ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test native8192 traincal s5k50 d100/10 | test | 2.277 / 0.1172 | 0.95 | 0.90 | 0.35 | 320.3 | 166.2 | 60.7 | 105.5 | 128.9 | 299.0 |
| test fp07 selector8192 traincal s5k50 d100/10 | test | 2.277 / 0.1180 | 0.95 | 0.90 | 0.35 | 330.6 | 177.6 | 60.7 | 116.9 | 127.7 | 298.9 |
| test fp07 pose05 ret75 traincal s5k50 d100/10 | test | 2.475 / 0.1172 | 0.95 | 0.90 | 0.20 | 306.9 | 156.9 | 60.9 | 96.0 | 125.0 | 281.9 |
| test fp07 pose05 strict3px f05 ret75 traincal s5k50 d100/10 | test | 2.270 / 0.1049 | 1.00 | 0.95 | 0.30 | 313.5 | 159.2 | 61.1 | 98.1 | 129.2 | 305.4 |

This is a train-selected evaluator-budget speed lever, not a selector
contribution: under the frozen budget native8192 remains faster than fp07
selector8192 while recall is unchanged.

The new pose-information ret75 map is faster under the frozen q20 budget, but it
is not a usable speed/accuracy win because strict R2 drops from 0.35 to 0.20.
It stays diagnostic until the sampling objective can preserve strict recall.

A strict-support guarded variant reserves 5% of the map budget for audited
high-score self-map inliers with reprojection error at or below 3px before
filling the rest with the pose/selector score. This guard improves train-q20
median, R10, and latency versus native8192 without losing R5/R2. On the fixed
q20 test diagnostic it restores much of the first pose-information R2 loss
(0.20 -> 0.30) and improves median/R10/R5/latency versus native8192, but it
still stays below native8192 strict R2 (0.30 vs 0.35).

The best train-only follow-up so far uses the same strict 3px guard, raises
source retention to 90%, and lowers pose-information weight to 0.25. Its solo
train-q20 profile improves native8192 train-calibrated median and latency while
keeping R5/R2 unchanged. It has not been run on the q20 test split, because
this is calibration evidence and should not become iterative test tuning.

The same ret90/pose0.25/strict3px map was then checked on a larger ShopFacade
train slice with `max_test_cameras=80`, still using the frozen train-calibrated
sparse/dense caps:

| Run | Split | Median cm/deg | R10 | R5 | R2 | Total ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Sparse inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-q80 native8192 budget s5k50 d100/10 | train | 1.652 / 0.1382 | 0.95 | 0.925 | 0.55 | 283.3 | 158.4 | 58.9 | 99.5 | 99.1 | 221.4 |
| train-q80 fp07 source-pool selector8192 budget s5k50 d100/10 | train | 1.652 / 0.1382 | 0.95 | 0.925 | 0.55 | 280.1 | 155.4 | 58.8 | 96.6 | 99.1 | 221.4 |
| train-q80 fp07 pose025 strict3px f05 ret90 budget s5k50 d100/10 | train | 1.786 / 0.1351 | 0.925 | 0.850 | 0.5375 | 280.6 | 155.7 | 57.8 | 97.9 | 99.3 | 223.5 |
| train-q80 fp07 pose025 strict3px f05 protnative95 budget s5k50 d100/10 | train | 1.796 / 0.1372 | 0.9375 | 0.8875 | 0.5125 | 282.9 | 156.1 | 58.0 | 98.1 | 100.9 | 229.6 |
| train-q80 fp07 pose025 protnative98 budget s5k50 d100/10 | train | 1.805 / 0.1368 | 0.9375 | 0.8875 | 0.5375 | 291.9 | 164.9 | 59.1 | 105.7 | 101.2 | 219.2 |

This larger train split rejects the ret90/pose0.25/strict3px recipe as a
candidate to freeze. It saves only 2.7 ms/query on mean latency and loses
median translation plus every dense recall threshold versus native8192. The
per-query comparison is also asymmetric: R10 recovered/lost = 1/3, R5
recovered/lost = 0/6, and R2 recovered/lost = 2/3.

Root-cause isolation on q80 train points to all-Gaussian replacement as the
failure source. These rows were run in parallel, so their latency is ignored;
only accuracy and recovered/lost counts are used:

| Run | Native8192 overlap | Added / dropped | Median cm/deg | R10 | R5 | R2 | R10 rec/lost | R5 rec/lost | R2 rec/lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native8192 | 8192 | 0 / 0 | 1.652 / 0.1382 | 0.950 | 0.925 | 0.550 | - | - | - |
| fp07 source-pool selector8192 | 8185 | 7 / 7 | 1.652 / 0.1382 | 0.950 | 0.925 | 0.550 | 0 / 0 | 0 / 0 | 0 / 0 |
| fp07 all-gaussian selector8192 | 2168 | 6024 / 6024 | 2.584 / 0.1832 | 0.8125 | 0.725 | 0.3625 | 1 / 12 | 1 / 17 | 7 / 22 |
| fp07 pose05 strict3px ret75 | 6197 | 1995 / 1995 | 1.820 / 0.1397 | 0.925 | 0.8375 | 0.5375 | 1 / 3 | 1 / 8 | 6 / 7 |
| fp07 pose025 strict3px ret90 | 7426 | 766 / 766 | 1.786 / 0.1351 | 0.925 | 0.850 | 0.5375 | 1 / 3 | 0 / 6 | 2 / 3 |

The source-pool selector reproduces native dense metrics almost exactly. The
large regressions appear only when native8192 landmarks are replaced by
all-Gaussian candidates. This means the next speed/accuracy path must protect
native-supported hard cases before it adds all-Gaussian selector/pose landmarks.

An explicit protected-source reservation was then added to test that hypothesis
without touching test queries or evaluator behavior. It reduces replacement
damage but does not rescue same-budget all-Gaussian maps: the protnative95
strict-support map keeps 7,840 / 8,192 native landmarks yet drops q80 train
R10/R5/R2 to 0.9375/0.8875/0.5125, and the no-strict protnative98 map keeps
8,031 / 8,192 native landmarks yet still drops R10/R5/R2 to
0.9375/0.8875/0.5375 while running slower than native8192. The source-pool
selector solo timing is the only q80 train point that preserves native dense
accuracy and improves mean latency (280.1 ms vs 283.3 ms), but the median
latency is slightly worse and the result remains train-only.

That source-pool timing signal was repeated with three paired q80 train runs
across GPUs 0/1/2. Each pair ran native and selector sequentially on the same
GPU. Dense accuracy remained exactly matched, but selector was slower by +67.5,
+64.1, and +53.7 ms/query under paired load, so the earlier solo mean-latency
gain is not robust enough to carry a speed claim.

Two structural follow-ups were also negative or weak. An additive 12k full-mode
map protected every native8192 landmark and filled 4,096 selector/pose-scored
all-Gaussian landmarks, but it underperformed native12288 on q80 train:
1.576 / 0.1233 with R10/R5/R2 = 0.9500/0.8875/0.6250 versus native12288
1.533 / 0.1018 with 0.9625/0.9000/0.6500. Stronger selector-only training
raised self-map listwise accuracy from 0.5532 to 0.5604 and produced a
rankstrong source-pool map that improves q80 train R2 from 0.5500 to 0.5625
with no R10/R5 loss, but it recovers only `seq2/frame00068.png` and is slower
than native8192 in the solo run (304.8 ms vs 283.3 ms).

## Selector-Only Training Probe

`train_unified_lff.py` now has a strict selector-only mode:

```text
native base descriptors frozen
descriptor residual frozen
residual gate frozen
candidate logits = native cosine logits + selector_bias_weight * selector_logit
export_descriptors = native descriptors
```

Five scene checkpoints were trained from the existing 60k rehearsal caches:

```text
output/unified_lff_v2/train_selector_only_20260517/{scene}/unified_lff_v2.pt
```

Each output writes `manifest.json`, `command.txt`, `metrics_summary.json`,
`split_audit.json`, and `git_status.txt`. The split audit is still `unknown`
because the old rehearsal caches contain train/rendered phase counts but not
complete source image id lists.

Final training diagnostics:

| Scene | Listwise acc | Loss | Residual norm | Audit |
| --- | ---: | ---: | ---: | --- |
| GreatCourt | 0.7181 | 2.7875 | 0.000000 | unknown |
| KingsCollege | 0.8727 | 1.9108 | 0.000000 | unknown |
| OldHospital | 0.6772 | 3.4389 | 0.000000 | unknown |
| ShopFacade | 0.6292 | 3.8470 | 0.000000 | unknown |
| StMarysChurch | 0.7193 | 3.2108 | 0.000000 | unknown |

The ShopFacade selector-only checkpoint was exported as an 8k cov8 resampled
map and profiled on the same q20 slice:

```text
output/unified_lff_v2/selector_only_resample_20260517/selector8192_cov8/ShopFacade
output/unified_lff_v2/profile_20260517/shop_q20_selectoronly8192_cov8_solo
```

It gives slightly better median translation than native8192, but it does not
improve R10/R5/R2 or latency. This is useful negative evidence: clean
selector-only training by itself is not enough to create the desired
speed-accuracy Pareto point.

## Interpretation

Supported by this probe:

- Same-budget selector sampling has an accuracy signal on this q20 slice:
  4k/8k/12k improve dense median translation versus native, and the older 8k
  selector improves R10 and R5.
- Dense refinement is stable across budget choices at about 164-175 ms/query.
- Sparse pose time, not dense rendering/refinement, is the main latency driver
  once the landmark budget is reduced too aggressively.
- Fixed fast caps reduce some low-budget latency, but they do not change the
  central conclusion: 4k/2k budgets are not faster than the 8k/12k fast rows
  because sparse pose remains harder.
- A positive-support prior is now available for sampling diagnostics and is
  backed by split-audited self-map cache metadata.
- Sparse prior-weight profiling is now supported without editing the vendored
  evaluator, and it confirms that `sampled_scores.pkl` can affect sparse-match
  latency when enabled.
- Prior-weight calibration has been moved to train cameras. The current
  self-map check does not support selecting prior005.
- A fixed sparse/dense iteration budget can reduce latency when chosen on train
  cameras, but it is not yet tied to selector sampling.
- Sparse-pose information can be exported from the audited self-map cache and
  changes the speed/accuracy tradeoff without changing descriptors or evaluator.
- A strict-support reservation guard can reduce the pose-information strict
  recall loss while keeping the map export single-path and auditable.
- Larger train-split validation is now in place and can reject q20-only
  calibration wins before they reach another test diagnostic.
- The q80 train failure has a concrete source: all-Gaussian replacement, not
  selector scoring inside the native sampled pool.
- Geometry-aware selector training targets are implemented and audited. The new
  `pose_target_*` options blend the existing selector target with a neutral
  self-map reprojection reliability target, and the training bundle records the
  pose-target settings and support counts.
- Pair-level geometry-aware selector training is also implemented and audited.
  The `lambda_selector_pose_pair` loss ranks candidates inside each self-map
  query by low reprojection error, query score, margin, and candidate cosine,
  using only the audited self-map cache. It is disabled by default and does not
  alter the single STDLoc-compatible inference path.
- Selector score scale has been isolated as a current bottleneck. Native source
  `sampled_scores.pkl` has much larger variance than the learned selector gate,
  so the default source-pool export blend can hide geometry-aware training
  signal.
- Training-side selector scaling can move that bottleneck: the latest audited
  ShopFacade checkpoints raise gate std from about 0.0074 to 0.047-0.056 and
  change 44-60 source-pool sampled ids while using only self-map supervision.
- Hard-query support is now available as a train/self-map export prior. It
  upweights low-margin self-map inliers and records split-audited cache
  metadata, giving a direct diagnostic for native hard-case protection.
- Query-coverage reservation is now available as a train/self-map export
  objective. Unlike the scalar hard-query prior, the first f10/r001 source-pool
  row improves q80 train median pose while preserving the stronger
  rank/listwise selector R10/R5/R2.
- Sparse-prior weighting can sharpen the qcov f10/r001 train-recall signal:
  prior01125 reaches R10/R5/R2 = 0.975/0.950/0.600. This is a calibration
  diagnostic only. A shifted train-s3/q75 slice preserves the recall direction,
  with native R5/R2 0.8667/0.4667 and qcov+prior01125 R5/R2 0.8800/0.5333.

Not supported:

- A paper-facing speed claim. The fair q20 solo probe still has every selector
  row slower than its same-budget native row, and the q80 query-coverage
  latency signal is not stable across solo repeats.
- A positive-support main claim. The 4k smoke improves R5 but drops R2, and the
  8k smoke loses the best fp07 speed point.
- A sparse-prior speed claim. The 8k fp07 prior row is faster but drops R2 to
  0.30, while the 12k prior rows drop R5 to 0.85. The train-camera prior005
  calibration also increases latency with unchanged recall. On qcov f10/r001,
  prior01125 gives the best train recall but is still slower than corrected
  native8192 on mean latency and worsens rotation median. The shifted train
  solo pair is also slower than native by +12.8 ms mean and +20.7 ms median, so
  it is not a frozen speed candidate.
- A selector speed claim from the train-calibrated budget. The frozen q20 test
  diagnostic keeps recall unchanged, but native8192 is faster than fp07
  selector8192 under the same budget.
- A pose-information main claim. The first ret75 pose-information map is faster
  but loses strict q20 R2, so it is only evidence about the next sampling target.
- A strict-support guarded pose-information main claim. The guard improves the
  fixed q20 diagnostic over the first pose map, but it still loses strict R2
  versus native8192.
- A ret90/pose0.25 strict-support main claim. It is only train-calibration
  evidence at this point; no q20-test or full-Cambridge row should be inferred.
- A ret90/pose0.25 strict-support frozen-candidate claim. The q80 train check
  loses dense median, R10, R5, and R2 versus native8192, so this recipe should
  not be advanced to another test diagnostic.
- A broad all-Gaussian replacement claim. The q80 train isolation shows
  replacement drops native hard-case support and produces asymmetric losses.
- A clean selector-only training claim. The new selector-only 8k profile does
  not beat the older selector 8k profile or native12288 on this q20 slice.
- A pose-target training claim. On ShopFacade q80 train, `posew05_score07`
  changes one source-pool sampled index relative to the stronger rank/listwise
  checkpoint and gives identical dense accuracy: 1.653 cm / 0.1382 deg,
  R10/R5/R2 0.950/0.925/0.5625. Its solo timing is still not a speed win
  versus native8192 (286.5 ms mean, 267.8 ms median, versus 283.3 ms and
  261.8 ms). The stronger `posew10_score00_gate1` export is exactly the same
  sampled set as rankstrong, so another profile would be redundant.
- A pose-pair training claim. The first pair-level probes used audited self-map
  pose utilities with 9,440 positive pairs over 1,899 landmarks for t8/s0 and
  1,479 pairs over 473 landmarks for t8/s0.7, but the exported source-pool maps
  changed only 1-3 sampled ids versus rankstrong. The profiled maps exactly
  match rankstrong q80 train accuracy: 1.653 cm / 0.1382 deg with R10/R5/R2
  0.950/0.925/0.5625.
- A selector-normalization claim. Amplifying rankstrong selector scores can move
  the native source-pool map, but the current transforms do not dominate:
  `sw10` preserves R10/R5/R2 while worsening median, `minmax` improves strict
  R2 to 0.5875 and median translation to 1.648 cm but drops R5 to 0.9125, and
  rank normalization collapses to R10/R5/R2 = 0.850/0.7875/0.4375. These were
  parallel accuracy diagnostics, so timing is ignored.
- A weighted-minmax source-pool claim. The fine sweep does not expose a hidden
  train-q80 Pareto point: weights 0.25/0.30 preserve R5/R2 but worsen median
  pose, weights 0.35/0.40 lose R5 without gaining R2, and weights 0.50/0.75
  gain R2 only by dropping R5 to 0.9125. These were also parallel accuracy
  diagnostics with timing ignored.
- A stronger training-side selector-scale claim. The lr3e3/e48 rank-only
  checkpoint preserves rankstrong R10/R5/R2 but worsens median to
  1.679 cm / 0.1395 deg, the pose-pair w10 checkpoint improves median angle to
  0.1362 deg but drops R5/R2 to 0.9125/0.5500, and the pose-pair w20 checkpoint
  loses the rankstrong strict-R2 gain while worsening median. These q80 train
  profiles were parallel accuracy diagnostics, so timing is ignored.
- A hard-query support claim. The source-pool hard-query prior uses only the
  audited self-map cache and changes 22-39 rankstrong sampled ids, but the mild
  w0.25 run drops R2 back to native 0.5500 and the stronger w1.0 run worsens
  median to 1.795 cm / 0.1430 deg with R2 0.5250. The lr3e3+w0.5 variant also
  preserves only native recall while worsening median. These were parallel
  accuracy diagnostics with timing ignored.
- A query-coverage paper-facing claim. The f10/r001 row is positive on one
  ShopFacade train-q80 slice, but f25/r002 loses the rankstrong R2 gain and the
  lr3e3 f50/r002 variant trades higher R2 for worse translation median.
- A query-coverage speed claim. The first f10/r001 solo run was faster than the
  old native8192 solo row, but a corrected same-GPU sequential repeat reversed
  the mean-latency delta.
- A realtime claim. The best q20 solo row here is native12288 at 394.5 ms/query
  mean latency, and the q80 train query-coverage row is still about
  275-281 ms/query.
- A paper-facing claim. The split audit is `unknown`, and q20 is only a small
  diagnostic subset.

## Query-Coverage Reservation Probe

The next structural probe changes the map objective from scalar self-map priors
to query-set coverage. `selector_resampling.py` now greedily reserves
source-pool landmarks that cover low-margin audited self-map queries with
low-reprojection inliers, then fills the remaining map budget with the normal
selector/source score. This still produces one static map and keeps inference on
the single STDLoc-compatible path; no per-query selector or test signal is used.
The qcov exports were regenerated with top-level `manifest.json` files recording
the git commit, command, split, hyperparameters, data roots, and feedback flags.

The first pass used the audited ShopFacade cache:

```text
output/unified_lff_v2/episode_caches_audited_20260517/ShopFacade/scene_match_pairs.pt
```

The f10/r001 candidate was repeated as a solo q80 train timing run; the f25 and
lr3e3 rows are parallel accuracy diagnostics with timing ignored.

| Run | Native overlap | Rankstrong overlap | Reserved | Covered hard queries | Median cm/deg | R10 | R5 | R2 | Total ms | Sparse ms | Sparse pose ms | Sparse match ms | Dense ms | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train-q80 native8192 budget s5k50 d100/10 | 8192 | - | 0 | - | 1.652 / 0.1382 | 0.950 | 0.925 | 0.5500 | 283.3 | 158.4 | 58.9 | 99.5 | 99.1 | native solo baseline |
| train-q80 rankstrong source-pool | 8181 | 8192 | 0 | - | 1.653 / 0.1382 | 0.950 | 0.925 | 0.5625 | 304.8 | 176.9 | 59.1 | 117.9 | 102.1 | strict-R2 gain, slower solo |
| train-q80 rankstrong qcov f10/r001 | 8168 | 8178 | 60 | 506 / 506 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | 275.2 | 149.2 | 58.7 | 90.5 | 100.3 | train-only accuracy/strict-recall candidate |
| train-q80 rankstrong qcov f25/r002 | 8150 | 8159 | 110 | 1250 / 1250 | 1.629 / 0.1382 | 0.950 | 0.925 | 0.5500 | ignored | ignored | ignored | ignored | ignored | loses rankstrong R2 gain |
| train-q80 lr3e3 qcov f50/r002 | 8137 | 8147 | 151 | 2483 / 2483 | 1.664 / 0.1362 | 0.950 | 0.925 | 0.5750 | ignored | ignored | ignored | ignored | ignored | higher R2, worse translation median |

The f10/r001 result is the first q80 train point in this branch that improves
dense median pose versus both native8192 and rankstrong while keeping
rankstrong R10/R5/R2. It remains calibration evidence only: it needs
repeat/cross-scene validation from train/self-map artifacts before another q20
test diagnostic is justified.

A tight neighboring sweep shows that the reservation basin is narrow. All rows
below are train-q80 parallel accuracy diagnostics, with timing ignored:

| Run | Reserved | Covered hard queries | Median cm/deg | R10 | R5 | R2 | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rankstrong qcov f05/r0005 | 35 | 265 / 265 | 1.653 / 0.1343 | 0.950 | 0.925 | 0.5625 | keeps strict-R2 gain, but not translation median gain |
| rankstrong qcov f10/r001 | 60 | 506 / 506 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | best balanced train-q80 accuracy point |
| rankstrong qcov f10/r002 | 60 | 506 / 506 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | same selected set/metrics as f10/r001 |
| rankstrong qcov f15/r0015 | 85 | 746 / 746 | 1.631 / 0.1382 | 0.950 | 0.925 | 0.5500 | loses rankstrong R2 gain |
| rankstrong qcov f25/r002 | 110 | 1250 / 1250 | 1.629 / 0.1382 | 0.950 | 0.925 | 0.5500 | loses rankstrong R2 gain |

The initial qcov f10/r001 solo row looked faster than the old native8192 solo
row, but a corrected same-GPU sequential repeat did not preserve that speed
delta:

| Repeat | Run | Median cm/deg | R10 | R5 | R2 | Total mean ms | Total median ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | native8192 | 1.652 / 0.1382 | 0.950 | 0.925 | 0.5500 | 283.3 | 261.8 |
| r1 | qcov f10/r001 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | 275.2 | 262.4 |
| r2 | native8192 | 1.652 / 0.1382 | 0.950 | 0.925 | 0.5500 | 274.9 | 263.0 |
| r2 | qcov f10/r001 | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | 281.5 | 265.9 |

This downgrades qcov f10/r001 from speed/accuracy candidate to
accuracy/strict-recall candidate. It is still the strongest train-q80
source-pool accuracy signal in this branch, but it does not yet advance the
realtime claim.

A sparse-prior follow-up then tested whether the qcov map could recover more
strict train cases by enabling `sampled_scores.pkl` in sparse matching. These
weights were checked only on train-q80; the parallel rows are accuracy-only and
their timing is ignored.

| Run | Prior weight | Profile mode | Median cm/deg | R10 | R5 | R2 | Total mean ms | Total median ms | Outcome |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qcov f10/r001 repeat | 0.00 | solo | 1.610 / 0.1343 | 0.950 | 0.925 | 0.5625 | 281.5 | 265.9 | no-prior qcov accuracy baseline |
| qcov f10/r001 prior002 | 0.02 | parallel accuracy | 1.642 / 0.1263 | 0.9375 | 0.9125 | 0.5750 | ignored | ignored | loses R10/R5 |
| qcov f10/r001 prior005 | 0.05 | parallel accuracy | 1.639 / 0.1173 | 0.950 | 0.9125 | 0.6125 | ignored | ignored | improves R2, but loses R5 |
| qcov f10/r001 prior0075 | 0.075 | parallel accuracy | 1.685 / 0.1187 | 0.9375 | 0.9125 | 0.6000 | ignored | ignored | loses R10/R5 |
| qcov f10/r001 prior010 | 0.10 | parallel accuracy | 1.645 / 0.1300 | 0.950 | 0.925 | 0.5875 | ignored | ignored | preserves R10/R5 and improves R2 |
| qcov f10/r001 prior010 | 0.10 | solo | 1.645 / 0.1300 | 0.950 | 0.925 | 0.5875 | 276.9 | 258.7 | strict-R2 diagnostic; no speed claim |
| qcov f10/r001 prior01125 | 0.1125 | parallel accuracy | 1.638 / 0.1490 | 0.975 | 0.950 | 0.6000 | ignored | ignored | best train-q80 recall point |
| qcov f10/r001 prior01125 | 0.1125 | solo | 1.638 / 0.1490 | 0.975 | 0.950 | 0.6000 | 278.3 | 262.9 | recall candidate; no native speed win |
| qcov f10/r001 prior0125 | 0.125 | parallel accuracy | 1.625 / 0.1424 | 0.9625 | 0.950 | 0.5625 | ignored | ignored | improves R10/R5, but no R2 gain over qcov |
| qcov f10/r001 prior01375 | 0.1375 | parallel accuracy | 1.627 / 0.1256 | 0.9625 | 0.925 | 0.5625 | ignored | ignored | improves R10 only; no R5/R2 gain |
| qcov f05/r0005 prior010 | 0.10 | parallel accuracy | 1.674 / 0.1300 | 0.950 | 0.925 | 0.5750 | ignored | ignored | mild R2 gain, worse translation median |
| qcov f15/r0015 prior0125 | 0.125 | parallel accuracy | 1.696 / 0.1310 | 0.9625 | 0.9375 | 0.5500 | ignored | ignored | better R10/R5, but loses qcov R2 |

Prior01125 improves dense recall over corrected native8192 by +2.5 pp R10,
+2.5 pp R5, and +5.0 pp R2, while keeping median translation slightly better
(1.638 cm vs 1.652 cm). It is not a clean speed/accuracy point: mean latency is
slower than the corrected native repeat, median latency is essentially tied,
rotation median is worse, and qcov no-prior still has the better median pose.

The candidate was then checked on a shifted train slice using `test_stride=3`.
With 231 ShopFacade train cameras, this gives 77 selected cameras and 75
measured queries after two warmup cameras. These rows are still train-only; the
parallel row timing is ignored.

| Run | Profile mode | Median cm/deg | R10 | R5 | R2 | Total mean ms | Total median ms | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train-s3/q75 native8192 | solo | 2.133 / 0.1193 | 0.9867 | 0.8667 | 0.4667 | 282.4 | 261.6 | shifted native baseline |
| train-s3/q75 qcov f10/r001 | parallel accuracy | 2.024 / 0.1193 | 0.9867 | 0.8667 | 0.4800 | ignored | ignored | improves median/R2 only |
| train-s3/q75 qcov f10/r001 prior01125 | parallel accuracy | 1.806 / 0.1211 | 0.9867 | 0.8800 | 0.5333 | ignored | ignored | shifted-slice recall gain |
| train-s3/q75 qcov f10/r001 prior01125 | solo | 1.806 / 0.1211 | 0.9867 | 0.8800 | 0.5333 | 295.2 | 282.3 | confirms recall; slower than native |

The shifted slice keeps the accuracy direction but reinforces the runtime
downgrade. Qcov+prior01125 improves translation median, R5, and R2 versus
native8192, but the same-GPU solo pair is slower by +12.8 ms mean and +20.7 ms
median.

The main technical finding is that reducing sampled landmark count does not
automatically reduce end-to-end latency. At low budgets, PROSAC/PnP spends more
time resolving weaker or fewer correspondences, which can dominate any matching
or dense-stage savings.

## KingsCollege Audited Transfer Probe

A first non-ShopFacade audited-cache transfer was run to test whether the
ShopFacade qcov recipe is obviously portable before spending any Cambridge test
budget. The generated artifacts are:

```text
output/unified_lff_v2/episode_caches_audited_20260517/KingsCollege/scene_match_pairs.pt
output/unified_lff_v2/train_selector_only_audited_20260517/KingsCollege_rankstrong_gate01_bias5/unified_lff_v2.pt
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_r001_selector8192_cov8/KingsCollege
```

The cache has `feedback_bank_split_name=selfmap_train_rendered`,
`split_audit.audit_status=passed`, and no overlap with KingsCollege test image
ids. The fixed rankstrong/qcov recipe was reused; no evaluation result was used
to choose it. Profiling KingsCollege requires `--images .` because that scene's
RGB files are under `seq*/`, while ShopFacade uses `processed/seq*/`.

Same-GPU train-q80 solo results:

| Run | Median cm/deg | R10 | R5 | R2 | Total mean ms | Total median ms | Sparse mean ms | Dense mean ms | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Kings native8192 | 12.471 / 0.1805 | 0.3375 | 0.1000 | 0.0250 | 293.1 | 280.6 | 138.9 | 129.0 | baseline |
| Kings rankstrong qcov f10/r001 | 12.272 / 0.1813 | 0.3250 | 0.1125 | 0.0250 | 290.8 | 281.6 | 139.2 | 126.4 | mixed transfer |

This is not a speed/accuracy Pareto point: translation median and R5 improve,
but R10 drops, rotation median is slightly worse, and median latency is slightly
slower. It is diagnostic evidence that the audited cross-scene path is working,
not evidence to promote qcov f10/r001 to a frozen recipe.

## Remaining Cambridge Transfer Probe

The same fixed rankstrong/qcov f10/r001 recipe was then transferred to
GreatCourt, OldHospital, and StMarysChurch. The caches were built from
train/rendered self-map evidence only, with no Cambridge test query signal used
for training, calibration, recipe selection, or map export. All new cache and
training sidecars report `split=selfmap_train_rendered`,
`split_audit.audit_status=passed`, and zero overlap with each scene's official
test image ids.

New artifacts:

```text
output/unified_lff_v2/episode_caches_audited_20260517/{GreatCourt,OldHospital,StMarysChurch}/scene_match_pairs.pt
output/unified_lff_v2/train_selector_only_audited_20260517/{GreatCourt,OldHospital,StMarysChurch}_rankstrong_gate01_bias5/unified_lff_v2.pt
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_r001_selector8192_cov8/{GreatCourt,OldHospital,StMarysChurch}
```

Cache and export diagnostics:

| Scene | Cache positives | Positive ratio | Selector acc | Reserved landmarks | Covered hard queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 6762 | 0.1127 | 0.6317 | 437 | 475 |
| OldHospital | 1901 | 0.0317 | 0.7092 | 124 | 136 |
| StMarysChurch | 1005 | 0.0168 | 0.8807 | 25 | 39 |

Same-GPU train-q80 profiles used the frozen train-calibrated caps
`sparse=5000/50`, `dense=100/10`, `warmup_cameras=2`, and `--images processed`
for these three scenes. They are train-only diagnostics; the profile
`split_audit.json` files remain `unknown`/`paper_safe=false` because profile
outputs do not attach the full self-map disjointness audit.

| Scene | Map | Median cm/deg | R10 | R5 | R2 | Total mean/median ms | Sparse mean/median ms | Dense mean/median ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | native8192 | 6.756 / 0.0345 | 0.8500 | 0.3125 | 0.0000 | 626.4 / 635.0 | 498.1 / 507.0 | 102.3 / 93.9 |
| GreatCourt | qcov f10/r001 | 6.817 / 0.0366 | 0.8375 | 0.2750 | 0.0250 | 622.6 / 634.1 | 492.1 / 512.1 | 104.4 / 97.3 |
| OldHospital | native8192 | 9.020 / 0.1859 | 0.5125 | 0.3000 | 0.0000 | 650.2 / 663.6 | 495.5 / 511.5 | 128.2 / 120.2 |
| OldHospital | qcov f10/r001 | 9.738 / 0.1962 | 0.5125 | 0.2500 | 0.0000 | 634.2 / 674.4 | 482.4 / 524.2 | 125.5 / 120.0 |
| StMarysChurch | native8192 | 11.701 / 0.3951 | 0.4875 | 0.3250 | 0.1000 | 653.4 / 664.7 | 515.1 / 531.6 | 112.5 / 111.4 |
| StMarysChurch | qcov f10/r001 | 9.847 / 0.3614 | 0.5000 | 0.3500 | 0.1000 | 642.8 / 679.6 | 514.8 / 546.8 | 102.2 / 104.1 |

Combined with ShopFacade and KingsCollege, this rejects qcov f10/r001 as a
frozen cross-scene recipe. ShopFacade and StMarysChurch show useful accuracy
signals, KingsCollege is mixed, and GreatCourt/OldHospital regress enough that
the method should not be advanced to test/full Cambridge evaluation.

## Qcov 12k Budgeted Pareto Probe

Because the 8k qcov map was too aggressive for GreatCourt and OldHospital, the
same audited f10/r001 query-coverage objective was exported at 12,288 landmarks
for all five scenes:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_r001_selector12288_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q40_q80_qcov12_budgeted_pareto.json
```

The exports use only `selfmap_train_rendered` caches and preserve native
descriptors. All export split audits pass, but the profile sidecars are still
`unknown`/`paper_safe=false`; these rows are train-only diagnostics.

Train-q80 same-GPU paired profiles against native12k:

| Scene | Native cm/deg | Qcov12 cm/deg | Native R10/R5/R2 | Qcov12 R10/R5/R2 | Native mean/median ms | Qcov12 mean/median ms | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ShopFacade | 1.533 / 0.1018 | 1.533 / 0.1018 | 0.9625 / 0.9000 / 0.6500 | 0.9625 / 0.9000 / 0.6500 | 641.1 / 658.3 | 617.6 / 648.8 | same accuracy, faster |
| KingsCollege | 12.060 / 0.1867 | 12.180 / 0.1848 | 0.3000 / 0.1000 / 0.0250 | 0.3250 / 0.1000 / 0.0250 | 657.2 / 665.2 | 653.6 / 658.6 | R10/rotation up, translation worse |
| GreatCourt | 5.914 / 0.0330 | 5.911 / 0.0333 | 0.8250 / 0.4000 / 0.0250 | 0.8250 / 0.4125 / 0.0250 | 620.4 / 623.9 | 604.7 / 612.1 | median/R5 up, rotation slightly worse |
| OldHospital | 7.743 / 0.1695 | 8.458 / 0.1763 | 0.5500 / 0.3000 / 0.0000 | 0.5375 / 0.2875 / 0.0000 | 483.8 / 480.6 | 442.8 / 454.1 | faster, accuracy/recall worse |
| StMarysChurch | 5.407 / 0.2010 | 5.530 / 0.2020 | 0.6125 / 0.4500 / 0.1500 | 0.6000 / 0.4500 / 0.1500 | 478.0 / 475.9 | 469.1 / 467.2 | faster, R10/median worse |

This is the clearest same-budget runtime signal so far: qcov12 reduces mean and
median latency on all five train-q80 scene pairs, mostly by lowering sparse
stage time. It is still not a frozen paper-facing candidate because OldHospital
and StMarysChurch lose accuracy/recall, and KingsCollege trades translation
median for R10/rotation. The useful next hypothesis is not another test run;
it is to keep the 12k sparse-stage speed mechanism while adding a
self-map-derived guard that prevents the OldHospital/StMarysChurch losses.

A first guard follow-up kept the native12 sampled set exactly
(`protected_source_fraction=1.0`) and changed only the selector/source scores
plus PLY locability field. This isolates score/locability effects from landmark
replacement:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_scoreonly_protnative100_selector12288_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_qcov12_guard_followup.json
```

| Scene | Native12 mean/median ms | Score-only mean/median ms | Accuracy delta | Outcome |
| --- | ---: | ---: | ---: | --- |
| ShopFacade | 641.1 / 658.3 | 641.4 / 664.3 | unchanged | no speed gain |
| KingsCollege | 657.2 / 665.2 | 658.0 / 664.5 | unchanged | essentially tied |
| GreatCourt | 620.4 / 623.9 | 603.2 / 612.4 | unchanged | useful speed gain |
| OldHospital | 483.8 / 480.6 | 481.6 / 492.4 | unchanged | weak/mixed speed |
| StMarysChurch | 478.0 / 475.9 | 469.7 / 471.9 | unchanged | useful speed gain |

This proves the OldHospital/StMarysChurch qcov12 accuracy losses come from
landmark replacement rather than the selector score field itself. It does not
yet produce a strong global speed claim: score-only preserves native accuracy,
but the speed benefit is meaningful on only two or three scenes.

A reduced hard-query reservation plus 99.5% native-source protection was also
checked on the two loss scenes:

| Scene | Method | Median cm/deg | R10 | R5 | R2 | Mean/median ms | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OldHospital | qcov hq005 protnative995 | 8.884 / 0.1819 | 0.5375 | 0.3000 | 0.0000 | 494.2 / 498.9 | worse than native and qcov12 |
| StMarysChurch | qcov hq005 protnative995 | 5.383 / 0.1911 | 0.6250 | 0.4625 | 0.1500 | 472.8 / 475.4 | accuracy improves, small speed gain |

This remains mixed and cannot be a global recipe. The StMarysChurch row is
useful evidence that guarded replacement can recover accuracy while retaining
some speed, but OldHospital shows the guard is not robust enough.

A lower query-coverage reservation was also checked on those two loss scenes:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f005_r001_selector12288_cov8/{OldHospital,StMarysChurch}
output/unified_lff_v2/reports/20260517_train_q80_qcov12_lowfraction_loss_scene_followup.json
```

This reduces `query_coverage_fraction` from 0.10 to 0.005 and keeps the same
12k source-pool budget. It does not repair either loss. OldHospital exactly
matches the qcov12 failure at 8.458 cm / 0.1763 deg with R10/R5/R2 =
0.5375/0.2875/0.0000, and StMarysChurch stays at the qcov12 loss point
5.530 cm / 0.2020 deg with R10/R5/R2 = 0.6000/0.4500/0.1500. These were
parallel accuracy checks, so timing is ignored. The problem is not simply too
many qcov-reserved landmarks; the next same-budget route needs a stricter
self-map geometry filter or native-hard-case protection that changes which
qcov landmarks are admitted.

A sampled-set audit then made the failure mode sharper: f005 and f10 are
identical on both loss scenes, because the available source-pool qcov candidate
set saturates below even the f005 budget. Removing query coverage entirely but
keeping the same rankstrong source-pool fill still lands within 4 symdiff IDs
of qcov12 on OldHospital and 2 symdiff IDs on StMarysChurch, while dropping
119/44 native12 IDs. Raising `source_score_weight` from 1.0 through 8.0 does
not change those dropped-native counts. Hard-query support also barely
overlaps the dropped native IDs: 0 on OldHospital and 1 on StMarysChurch. The
loss is therefore a source-pool fill/replacement issue, not the qcov prefix
alone.

A protected-qcov fill isolation followed:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_scoreonly_protnative100_blend010_selector12288_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_scoreonly_protnative100_blend020_selector12288_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_qcov_f10_protnative100_selector12288_cov8/{OldHospital,StMarysChurch}
output/unified_lff_v2/reports/20260517_train_q80_protected_qcov_fill_isolation.json
output/unified_lff_v2/reports/20260517_train_q80_scoreonly_blend020_followup.json
```

The score-only blend010 export preserves the native12 sampled set exactly on
all five scenes and keeps the train-q80 macro dense metrics identical to
native12: 6.531 cm / 0.1384 deg with R10/R5/R2 =
0.6500/0.4300/0.1700. Its profiles were parallel accuracy checks, so no speed
claim is attached. A stronger blend020 repeat gives the same result: native12
sampled IDs are unchanged on all five scenes, dense metrics remain identical,
and timing is still parallel-only audit material. Keep score-only blend sweeps
as a safety control, not as a candidate until a fair solo speed mechanism is
identified.

The qcov f10 plus `protected_source_fraction=1.0` loss-scene maps reserve the
qcov prefix first and then fill the rest from native12. They change only 2/1
native IDs relative to native12 and restore the native12 dense metrics:

| Scene | Native12 dense | Qcov12 dense | Qcov+protnative100 dense | Native IDs dropped | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| OldHospital | 7.743 / 0.1695 | 8.458 / 0.1763 | 7.743 / 0.1695 | 2 | recovers native12 |
| StMarysChurch | 5.407 / 0.2010 | 5.530 / 0.2020 | 5.406 / 0.2010 | 1 | recovers native12 |

Those profiles were also parallel accuracy checks, so timing is ignored. The
useful conclusion is structural: the qcov prefix is not by itself the source
of the OldHospital/StMarysChurch regressions; the broad source-pool fill that
replaces 121/45 native12 landmarks is the unsafe step.

A native12-subset budget follow-up tested whether a safer speed Pareto point
could be found by reducing landmarks only inside the native12 set, with no
non-native replacement:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_sourcescore10240_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_native12sub10240_selector_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_native12sub10240_followup.json
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_sourcescore11264_cov8/{scene}
output/unified_lff_v2/selector_only_audited_resample_20260517/rankstrong_native12sub11264_selector_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_native12sub11264_followup.json
```

Both screens are rejected before solo timing. At 10,240 landmarks, the selector
subset is slightly better than the 10k source-score control but regresses
native12 macro dense metrics: median 7.215 cm vs 6.531 cm, R10/R5/R2 =
0.6275/0.4125/0.1575 vs 0.6500/0.4300/0.1700. At 11,264 landmarks, R5 is
held but median/R10/R2 still regress: 7.060 cm, 0.6400/0.4300/0.1550. The
remaining losses are again OldHospital and StMarysChurch. Dropped native IDs
barely overlap self-map support at 11,264 (4 IDs on each loss scene), so a
simple support guard is unlikely to recover this budget-reduction route. Treat
native12-subset budget reduction as negative train evidence unless a new
mechanism protects pose-critical landmarks directly.

A fixed prior005 follow-up then tested whether the native-ID-preserving
score-only field becomes useful when its `sampled_scores.pkl` and PLY
locability are actually consumed by sparse/dense matching:

```text
output/unified_lff_v2/profile_20260517/{scene}_train_q80_native12288_prior005_traincal_s5k50_d100_10_parallel_acc
output/unified_lff_v2/profile_20260517/{scene}_train_q80_scoreonly_blend020_prior005_traincal_s5k50_d100_10_parallel_acc
output/unified_lff_v2/reports/20260517_train_q80_scoreonly_blend020_prior005_followup.json
```

Both native12 and score-only blend020 were run with fixed global
`--sparse_landmark_prior_weight 0.05` and
`--dense_locability_prior_weight 0.05`. This is rejected before solo timing:
native12 prior005 already regresses native12 no-prior to 6.816 cm /
0.1503 deg and R10/R5/R2 = 0.6450/0.4275/0.1700. Score-only blend020 prior005
is worse on median/R10/R2 relative to native12 no-prior: 6.894 cm /
0.1516 deg and 0.6400/0.4300/0.1650. The loss scenes are again OldHospital
and StMarysChurch. Fixed prior weighting is therefore not a safe way to turn
score-only locability into a fast-mode claim.

An inlier-only dense early-exit simulation was then run from existing native12
train-q80 solo profiles, without changing evaluator code or running Cambridge
test:

```text
output/unified_lff_v2/reports/20260517_train_q80_native12_dense_early_exit_sweep.json
```

The policy is a fixed sparse-inlier threshold: if sparse inliers exceed the
threshold, keep the sparse pose and skip dense refinement; otherwise use the
recorded dense pose. This is diagnostic only because any deployable threshold
would need predeclared self-map validation. The result is negative for speed:
threshold 700 skips 3.0% of queries and saves only about 4.3 ms mean, while
dropping R10/R5 from 0.6500/0.4300 to 0.6425/0.4275. The only non-regressive
threshold is 800, which skips 1/400 queries and saves about 0.33 ms mean. Do
not implement inlier-only dense early exit; sparse inlier count alone is too
weak as a reliability signal.

A dense-oracle ceiling audit was then computed from the same native12 train-q80
solo profiles:

```text
output/unified_lff_v2/reports/20260517_train_q80_native12_dense_oracle_ceiling.json
```

Dense refinement improves translation on 333/400 queries but worsens it on
67/400. A strict GT oracle that skips dense only when the sparse pose has no
worse translation and rotation error would skip 42/400 queries, improve macro
dense median from 6.531 to 6.236 cm, improve R10/R5/R2 from
0.6500/0.4300/0.1700 to 0.6750/0.4550/0.1750, and save 2.0% mean latency. This
is an oracle ceiling only. A looser threshold-safe oracle skips 245/400 queries
and saves 12.5% mean latency, but worsens macro median by 0.607 cm, so it is
not a clean metric point. The best currently logged pre-dense signal for the
safe-skip target is sparse inlier count with AUC 0.543, effectively near
random. Do not implement GT-oracle or inlier-only early exit; the next speed
route needs additional train/self-map calibrated reliability features such as
reprojection residual quantiles, inlier ratio, PROSAC confidence, or match-score
margins.

The profiler wrapper was then extended, without touching vendored STDLoc
evaluator code, to record solve-pose reliability metadata in
`timing_profile.json`: match count, inlier count/ratio, and all-match/inlier
reprojection residual mean/median/p90. A tiny ShopFacade smoke verified that
the emitted per-query fields are present. Five train-q80 native12 profiles were
then rerun in parallel for reliability analysis only:

```text
output/unified_lff_v2/profile_20260517/{scene}_train_q80_native12288_reliability_traincal_s5k50_d100_10_parallel
output/unified_lff_v2/reports/20260517_train_q80_native12_reliability_signal_audit.json
```

The new residual fields are still weak as a deployable fast-mode signal. The
best oriented predictor for strict sparse-dominates-dense oracle labels is
`neg_sparse_inlier_reprojection_p90_px` with AUC 0.581. The best
single-feature threshold selected on this diagnostic split skips 14/400
queries (3.5%) with no macro median/recall regression, but that threshold is
train-slice selected and not paper-facing. Do not implement dense early exit
from this evidence; any future rule needs predeclared self-map calibration and
a shifted train validation.

A geometry-aware training follow-up then tested whether the same qcov12 export
could be repaired by selector pose-pair supervision rather than another export
guard. GreatCourt, OldHospital, and StMarysChurch were trained from audited
self-map caches with `lambda_selector_pose_pair=2.0`,
`pose_pair_reprojection_threshold_px=8.0`, and
`pose_pair_score_threshold=0.0`:

```text
output/unified_lff_v2/train_selector_only_audited_20260517/{GreatCourt,OldHospital,StMarysChurch}_posepair_w2_t8s0_rankstrong_gate01_bias5
output/unified_lff_v2/selector_only_audited_resample_20260517/posepair_w2_t8s0_qcov_f10_r001_selector12288_cov8/{scene}
output/unified_lff_v2/reports/20260517_train_q80_posepair_qcov12_followup.json
```

The new training run passed the cache split audits, but it barely changed the
qcov12 sampled sets: GreatCourt and OldHospital are identical to rankstrong
qcov12, and StMarysChurch adds only one new landmark. Train-q80 dense accuracy
therefore stays effectively unchanged:

| Scene | Pose-pair sampled-set delta vs qcov12 | Pose-pair qcov12 median cm/deg | R10 | R5 | R2 | Accuracy outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GreatCourt | +0 ids | 5.911 / 0.0333 | 0.8250 | 0.4125 | 0.0250 | same as qcov12 |
| OldHospital | +0 ids | 8.458 / 0.1763 | 0.5375 | 0.2875 | 0.0000 | still worse than native12 |
| StMarysChurch | +1 id | 5.531 / 0.2020 | 0.6000 | 0.4500 | 0.1500 | still worse than native12 median/R10 |

These profiles were launched concurrently on GPUs 0/1/2, so their timing fields
are recorded in the report but not used as fair speed evidence against solo
baselines. The useful conclusion is structural: this pose-pair loss does not
move source-pool ordering enough to fix the qcov12 replacement risk. Do not
spend q20/full test budget on this branch.

An additive guard was then tested without changing evaluator behavior: export
all native12 sampled IDs plus qcov12 non-native additions as one protected
static sampled set. The index artifacts and exported maps are:

```text
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_plus_qcov12_union_idx_20260517
output/unified_lff_v2/selector_only_audited_resample_20260517/native12_plus_qcov12_union_traincal/{scene}
output/unified_lff_v2/reports/20260517_train_q80_native12_plus_qcov12_additive_followup.json
```

The export audits pass (`split=selfmap_train_rendered`, native IDs missing = 0,
qcov IDs missing = 0), but the branch is still rejected. It restores the
StMarysChurch qcov12 near-threshold loss and slightly improves GreatCourt, but
OldHospital remains worse than native12:

| Scene | Native12 dense cm/deg | Additive union cm/deg | Native R10/R5/R2 | Additive R10/R5/R2 | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| GreatCourt | 5.914 / 0.0330 | 5.887 / 0.0329 | 0.8250 / 0.4000 / 0.0250 | 0.8375 / 0.4250 / 0.0250 | useful train gain |
| OldHospital | 7.743 / 0.1695 | 8.590 / 0.1695 | 0.5500 / 0.3000 / 0.0000 | 0.5375 / 0.3000 / 0.0000 | rejected |
| StMarysChurch | 5.407 / 0.2010 | 5.406 / 0.2010 | 0.6125 / 0.4500 / 0.1500 | 0.6125 / 0.4500 / 0.1500 | recovers qcov12 loss |

A capped additive variant, native12 plus at most 45 qcov12 non-native additions
per scene, was also checked on the two loss/control scenes. StMarysChurch stays
recovered, but OldHospital still fails: 8.276 cm / 0.1819 deg with
R10/R5/R2 = 0.5375/0.2875/0.0000. The OldHospital failure is not simply
"dropped native support"; adding qcov landmarks can perturb sparse matching
even when every native12 landmark is retained. The worst persistent case is
`seq1/frame00027.png`, which is 6.885 cm on native12 but 26.897 cm with the
additive maps. Do not run these additive maps on Cambridge test.

One derived diagnostic remains worth studying, but not claiming: a static
map-churn guard that uses additive qcov only when qcov12 adds at most 64
non-native IDs, otherwise falls back to native12. This uses self-map/map-set
churn rather than train/test query errors. On existing train-q80 rows it gives
macro dense median 6.526 cm, R10 0.6525, and R5 0.4350 versus native12
6.531 cm, 0.6500, and 0.4300. The gain is too small for a paper claim and
needs a predeclared map-level validation rule plus solo timing before any test
run.

That churn64 rule was then predeclared and checked on a shifted train slice
(`test_stride=3`, `max_test_cameras=75`, `warmup_cameras=2`) against native12:

```text
output/unified_lff_v2/reports/20260517_train_s3_q75_churn64_guard_validation.json
```

The guard selects additive union for ShopFacade, KingsCollege, GreatCourt, and
StMarysChurch, and native12 for OldHospital. It is accuracy-clean on this
shifted slice but still only diagnostic:

| Run | Macro median cm/deg | R10 | R5 | R2 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| native12 | 6.294 / 0.1546 | 0.6907 | 0.4507 | 0.1840 | baseline |
| churn64 guard | 6.294 / 0.1546 | 0.6907 | 0.4533 | 0.1840 | +1 GreatCourt R5 query only |

No dense metric regresses, but the only macro gain is +0.0027 R5 and the
selected additive maps retain all native12 landmarks while adding more. The
parallel timing fields are audit material only, and no solo timing was run
because this larger-map branch has no credible same-budget speed mechanism. Do
not promote the guard to Cambridge test; use it only as evidence that low
map-churn additive support can be accuracy-neutral.

A separate same-budget qcov12 churn guard was checked offline, using the
qcov12 sampled-set churn against native12 rather than additive maps:

```text
output/unified_lff_v2/reports/20260517_train_q40_q80_qcov12_samebudget_churn_guard_offline.json
output/unified_lff_v2/reports/20260517_train_s3_q75_qcov12_samebudget_lowchurn_guard_validation.json
```

The qcov-added counts are ShopFacade 7, GreatCourt 41, StMarysChurch 45,
KingsCollege 46, and OldHospital 121. On train-q80, the strict exact candidate
is threshold 7, selecting only ShopFacade: dense metrics are unchanged and the
solo q80 macro total-mean reference is -4.71 ms. Threshold 41 selects
ShopFacade and GreatCourt, gives +0.0025 macro R5 and -7.85 ms q80 macro
total-mean reference, but has a tiny macro angle drift (+0.000062 deg).

The shifted train-s3/q75 validation keeps the same interpretation:

| Same-budget guard | Selected qcov scenes | Shifted macro delta cm/deg | R10 | R5 | R2 | q80 solo speed ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| threshold 7 | ShopFacade | +0.000 / -0.000002 | +0.0000 | +0.0000 | +0.0000 | -4.71 ms |
| threshold 41 | ShopFacade, GreatCourt | +0.000 / +0.000247 | +0.0000 | +0.0027 | +0.0000 | -7.85 ms |

This is cleaner than the additive guard because it keeps a same-budget speed
mechanism, but the effect is too small for the thread goal. Threshold 7 is safe
but only changes one scene; threshold 41 is still a train-selected diagnostic
with angle drift. Do not move either threshold to Cambridge test or paper-facing
tables.

## Parallel Timing Caveat

The earlier parallel q20 runs on separate GPUs were useful for smoke testing but
not for fair speed evidence. They showed much higher and inconsistent latency,
so only the solo table above should be used for speed reasoning.

## Next Evidence

1. Do not launch full Cambridge timing until a q20 Pareto point is found.
2. Investigate self-map-calibrated sparse-stage controls that preserve the
   single-path evaluator protocol, such as fixed PROSAC iteration caps or
   confidence thresholds selected without test queries. The current
   s5k50/d100_10 cap is a useful control baseline, not a Loc-GS contribution.
3. Improve selector training for inlier geometry, not just locability rank:
   false-positive suppression and pose-information targets should be selected
   on self-map validation.
4. Add a sparse-pose-aware sampling objective before more low-budget sweeps.
   Current source-pool selector sampling can preserve recall, but it does not
   reduce PnP work at 4k/2k. The first pose-information prior reduces sparse
   matching time; the strict-support guard helps but does not yet preserve R2.
5. Before another test diagnostic, freeze a candidate from train/self-map
   calibration. The ret90 + pose0.25 + strict3px candidate and the explicit
   native-protected all-Gaussian follow-ups failed the larger q80 train check.
6. Do not freeze another same-budget all-Gaussian replacement recipe from these
   rows. Even replacing only 161-352 native8192 landmarks loses native-supported
   q80 train cases; the next path should improve source-pool runtime stability,
   train geometry-aware selector targets, or use a separately justified
   additive/larger-budget map.
7. The source-pool runtime and additive/larger-budget branches are now weak:
   paired timing rejects a stable source-pool speed claim, and the additive 12k
   selector fill loses to native12288. The only positive train signal is the
   stronger rank/listwise selector recovering one extra R2 query without R10/R5
   loss. The first pose-target and pose-pair losses did not materially change
   the sampled set, while selector amplification exposed a real R2/R5 tradeoff.
   The follow-up weighted-minmax export sweep is also negative. Training-side
   scale amplification does affect ordering directly, but the q80 train result
   still trades median, R5, or R2. Hard-query support adds a different
   self-map prior, but it still trades away the rankstrong R2 recovery. The new
   query-coverage reservation is the first source-pool variant to improve
   train-q80 median pose while preserving rankstrong recall; treat it as the
   next train/self-map validation candidate, not as a test-ready speed claim.
8. If sparse prior weights are used, choose them on self-map validation rather
   than q20/full test diagnostics. The current train-q20 calibration argues
   against prior005, while the qcov train-q80 prior01125 row is only a recall
   diagnostic until validated beyond this split slice. The shifted train-s3/q75
   check keeps the recall direction but remains same-scene train evidence.
9. The fixed qcov f10/r001 transfer is now rejected as a frozen cross-scene
   candidate: it improves ShopFacade/StMarysChurch but is mixed on KingsCollege
   and regresses GreatCourt/OldHospital.
10. The qcov12 budget probe gives a real same-budget speed signal, but it is
   not a paper-facing candidate until the OldHospital and StMarysChurch
   accuracy losses are guarded on train/self-map evidence. The first pose-pair
   qcov12 follow-up is rejected because it leaves the sampled sets essentially
   unchanged and does not repair those losses. Lowering qcov reservation to
   f005 on the two loss scenes also fails, matching the original qcov12 loss
   metrics. The protected-qcov fill isolation recovers the loss scenes by
   changing only 2/1 native IDs, which pins the failure on source-pool fill
   rather than query coverage itself; it is not a speed candidate because it
   has only parallel timing. A same-budget map-churn guard is cleaner, but the
   exact shifted-valid threshold selects only ShopFacade and is too weak
   (-4.71 ms q80 macro reference); the broader threshold remains diagnostic.
11. Native12-subset budget reduction is also rejected as a fast-mode route for
   now. The 10k and 11,264 selector subsets use only native12 IDs, but they
   still lose train-q80 median/R10/R2 before any solo timing is justified.
12. Fixed prior005 on score-only blend020 is rejected as well. It uses the
   native12 sampled set and fixed sparse/dense prior weights, but native12
   prior005 already hurts the no-prior baseline and blend020 prior005 still
   loses median/R10/R2.
13. Inlier-only dense early exit is rejected as a practical fast-mode lever:
   thresholds with meaningful skip rates lose train-q80 recall/median, while
   the only non-regressive threshold skips just 1/400 queries. A GT dense-oracle
   audit shows modest strict-error headroom, but the logged pre-dense signals
   cannot predict safe skips. The new wrapper-side residual/inlier-ratio
   instrumentation improves the best diagnostic skip point to 14/400 queries,
   but the predictor AUC is only 0.581 and the threshold was train-slice
   selected; require self-map calibration and shifted train validation before
   revisiting this route.
14. Native12-plus-qcov12 additive maps are also rejected as frozen candidates:
   full union and cap45 both keep all native IDs, yet OldHospital still loses
   train-q80 R10/median because added qcov landmarks perturb sparse matching.
   A predeclared shifted-slice qcov-churn guard is accuracy-neutral, but its
   only gain is +0.0027 macro R5 and it has no same-budget speed mechanism.
   Keep it diagnostic and off Cambridge test. The same-budget low-churn guard
   preserves a speed mechanism but has the opposite problem: the safe effect is
   too small to matter.
15. Attach a real split audit before moving any timing result into a paper-safe
   table.
