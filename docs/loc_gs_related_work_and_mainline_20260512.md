# Loc-GS Related-Work Notes and Mainline Update

Date: 2026-05-12

Status: superseded by `docs/loc_gs_lff_scenematch_mainline_20260512.md` for
new experiments. The static-prior and guarded-selector sections below are kept
as experiment history and negative/diagnostic ablations.

## Paper-Facing Position

The defensible mainline is not learned descriptor replacement. It is
matchability-guided fine-tuning and selection for a Gaussian feature field used
by single-path 2D-3D localization.

The core claim should be:

1. Build a stable SuperPoint/STDLoc-compatible Gaussian feature field.
2. Rehearse localization from GT, perturbed, interpolated, and rendered
   map-side poses.
3. Convert self-matching and self-localization feedback into reliability
   signals: detector saliency, visibility, observability, geometry quality,
   distinctiveness, and calibrated matchability.
4. Use those signals conservatively to prioritize or lightly fine-tune the
   feature field while preserving the baseline descriptor distribution.

## Related-Work Implications

- STDLoc shows that Feature Gaussian plus a scene-specific detector and
  sparse-to-dense matching can be a strong no-pose-prior localization pipeline.
  Our method should be framed as adding localization-feedback reliability on
  top of this backbone, not replacing it.
  Source: https://arxiv.org/abs/2503.19358
- GLACE reports that co-visibility helps large-scene coordinate regression by
  grouping reprojection constraints and reducing trivial overfitting. This
  supports using co-visible map-side rehearsal poses and spatially balanced
  landmark priority.
  Source: https://arxiv.org/abs/2406.04340
- SplatLoc uses Gaussian primitives, a descriptor decoder, salient landmark
  selection, and primitive regularization. This is close to our feature-field
  map family; our difference must be the localization-feedback loop and the
  ablation showing that the feedback improves downstream 2D-3D localization.
  Source: https://arxiv.org/abs/2409.14067
- GSplatLoc and GSFeatLoc both confirm that 3DGS maps can support
  correspondence-based pose estimation and refinement. They reinforce the need
  for robust correspondences more than pure photometric optimization.
  Sources: https://arxiv.org/abs/2409.16502,
  https://arxiv.org/abs/2504.20379
- GSFFs learn feature fields and use pose refinement by aligning rendered and
  query features or segmentations. This raises the bar for feature-field
  localization, so our experiments must isolate the benefit of localization
  feedback rather than just showing that a 3DGS feature field can localize.
  Source: https://arxiv.org/abs/2507.23569
- ACE-G and MultiLoc emphasize generalization beyond training views. This
  supports the existing mixed rehearsal pose sampler and argues against
  train-view-only self-localization labels.
  Sources: https://arxiv.org/abs/2510.11605,
  https://arxiv.org/abs/2603.27170

## Current Engineering Update

The reliability eval launcher now exposes two covisibility recipes:

- `covisibility_select`: a hard ablation that keeps 12k landmarks, half from
  the STDLoc detector order and half from matchability score selection.
- `covisibility_soft_select`: a conservative candidate mainline that keeps
  14,336 landmarks, preserves 75% of the selected bank from STDLoc order, and
  only uses matchability to replace the remainder. It also adds descriptor
  distinctiveness to reduce repeated-structure landmarks.
- `covisibility_prosac`: the current preferred direction. It keeps the full
  STDLoc landmark bank and uses the matchability prior plus current query match
  scores to order OpenCV PROSAC PnP sampling. This keeps a single localization
  path and avoids branch-level pose selection while still letting localization
  feedback influence geometric solving.

The hard recipe is useful as an ablation, not as the default method. Early q80
results show it can improve some dense metrics but also hurts KingsCollege and
ShopFacade medians, which matches the expected risk of hard landmark deletion.

The first q80 run of `covisibility_prosac` is stronger: across five Cambridge
scenes, dense mean median improves from 10.469 cm for `protected` to 9.201 cm,
and dense mean R@5cm/5deg improves from 26.25% to 37.50%. This is still not
SOTA evidence by itself because it is a subset run, but it is the first result
in this branch that improves both aggregate median and aggregate recall without
reducing the landmark bank.

Full-test follow-up with 3 query shards per scene:

| Scene | Protected Dense | PROSAC Dense | Notes |
|---|---:|---:|---|
| GreatCourt | 25.286 cm / 0.097 deg, R@5 8.95% | 16.698 cm / 0.056 deg, R@5 12.50% | strong gain |
| KingsCollege | 23.640 cm / 0.321 deg, R@5 0.58% | 23.338 cm / 0.306 deg, R@5 0.29% | median slightly better, recall slightly worse |
| OldHospital | 22.529 cm / 0.332 deg, R@5 11.54% | 13.633 cm / 0.177 deg, R@5 16.48% | strong gain |
| ShopFacade | 3.690 cm / 0.137 deg, R@5 66.99% | 3.245 cm / 0.128 deg, R@5 72.82% | strong gain |
| StMarysChurch | 8.018 cm / 0.180 deg, R@5 26.23% | 6.169 cm / 0.125 deg, R@5 36.42% | strong gain |

Across all five Cambridge scenes, PROSAC improves dense mean median from
16.633 cm to 12.617 cm and dense mean R@5 from 22.86% to 27.70%. It also
improves sparse mean median from 96.140 cm to 15.147 cm. This supports the
"localization-feedback priority" claim much better than hard landmark
selection, but it is still not a final SOTA claim until external SOTA tables are
reproduced under the same split and protocol.

## DIM / LoFTR Probe

`deep-image-matching` is useful related engineering: it collects modern local
features and matchers including SuperPoint, DISK, ALIKED, XFeat, DeDoDe,
LightGlue, RoMa, and LoFTR, and exports correspondences for SfM-style pipelines.
Source: https://github.com/3DOM-FBK/deep-image-matching

The clean integration for this project is image-to-rendered-image matching, not
direct descriptor replacement. ALIKED/DISK/LoFTR descriptors live in different
spaces from the reconstructed SuperPoint/STDLoc feature field, so directly
matching them to the existing 3D landmark descriptor bank is not a fair
replacement. The implemented probe adds `loftr_rendered`: render RGB+depth from
the current pose, match query RGB to rendered RGB with Kornia LoFTR, unproject
rendered pixels to 3D, and solve PnP with the same PROSAC path.

q20 Cambridge probe on `reliability_b8_e10_20260511` checkpoints:

| Scene | Current PROSAC Dense | LoFTR Rendered, scale 1.0 | Interpretation |
|---|---:|---:|---|
| ShopFacade | 2.265 cm / 0.084 deg, R@5 95.0% | 3.670 cm / 0.128 deg, R@5 70.0% | worse |
| OldHospital | 4.575 cm / 0.077 deg, R@5 55.0% | 6.432 cm / 0.092 deg, R@5 45.0% | worse |
| GreatCourt | 23.188 cm / 0.066 deg, R@5 0.0% | 21.252 cm / 0.108 deg, R@5 5.0% | median better, angle worse |

The half-resolution LoFTR recipe was weaker because it produced too few
usable PnP inliers. The updated `covisibility_prosac_loftr` recipe therefore
uses full scale, zero confidence threshold, and 4096 max matches. It should be
treated as an ablation/fallback candidate, not the mainline, unless a future
candidate-selection verifier can reliably choose its occasional wins without
hurting ShopFacade and OldHospital.

## Next Experiments

1. Use `covisibility_prosac` as the next single-path candidate.
2. Compare it against `protected` on the same query subset across all five
   Cambridge scenes.
3. Promote it to full-test evaluation if it improves aggregate dense
   median or recall without causing a KingsCollege/ShopFacade regression.
4. Keep `loftr_rendered` as a DIM ablation and possible auxiliary verifier, but
   do not replace the SuperPoint/STDLoc feature-field matcher with it on the
   current evidence.
5. If full-test PROSAC is still mixed, keep it as an engineering ablation and
   move the same query-conditioned priority into training-time hard-negative
   matchability labels instead of relying only on test-time sampling.
6. For training, avoid full descriptor replacement. Prefer a frozen base
   descriptor plus small residual adapter, trust-region regularization, and
   hard-negative matchability labels generated by self-localization.

## 2026-05-12 Mainline Reset

The current paper-clean result is the train-calibrated guarded branch selector:

| Scene | Selected branch | Dense median |
|---|---|---:|
| GreatCourt | STDLoc baseline | 10.888 cm / 0.052 deg |
| KingsCollege | STDLoc baseline | 17.695 cm / 0.267 deg |
| OldHospital | STDLoc baseline | 10.717 cm / 0.211 deg |
| ShopFacade | localization-guided hybrid | 2.401 cm / 0.112 deg |
| StMarysChurch | STDLoc baseline | 3.690 cm / 0.125 deg |
| Macro | guarded train-calibrated selector | 9.078 cm / 0.153 deg |

This improves the local STDLoc reproduction average of 9.127 cm / 0.156 deg
and the STDLoc paper table average of 10.1 cm / 0.14 deg, while staying
single-path at test time. It does not yet beat the newest reported SOTA:
ULF-Loc reports 8.3 cm / 0.13 deg average on Cambridge, with GreatCourt at
7.49 cm. Source: https://arxiv.org/abs/2605.04730

The selector is intentionally conservative. It rejects learned branches when
train-calibration R@5 is too low and only lets a localization-guided branch win
when it is either clearly better or within a small calibrated tie window. This
keeps the result defensible and avoids selecting by test-set wins.

## Teacher-Fusion Findings

Implemented capabilities:

- `image_pair_geometry` PnP match filtering: scores a match by local 2D
  neighborhood consistency between the query image and the rendered/reference
  image before PnP subsampling.
- Projected teacher fusion: projects 3D landmarks into reference views and
  blends sampled SuperPoint teacher descriptors into the landmark descriptor.
- Optional geometry/centrality weighting for teacher fusion.
- Optional rendered auxiliary teacher views: perturb train poses, render RGB
  with 3DGS, extract SuperPoint teacher descriptors from the rendered RGB, and
  include them in the descriptor fusion pool.

GreatCourt q80 pilots looked mildly promising, e.g. 12.69 cm for
128-view geometry-weighted fusion, but full-test runs failed to generalize:

| Variant | GreatCourt full dense |
|---|---:|
| 64-view teacher fusion + image-pair geometry | 17.637 cm / 0.056 deg |
| 64-view teacher fusion weight 0.5 | 16.622 cm / 0.056 deg |
| 128-view geometry/centrality teacher fusion + image-pair geometry | 17.597 cm / 0.056 deg |

Conclusion: hard descriptor fusion is not the mainline. It can remain as an
ablation for the "feature fusion can destabilize the STDLoc descriptor
distribution" story, but the paper claim should focus on conservative
localization-feedback reliability and selection, not direct descriptor
replacement or unguarded teacher blending.

The rendered auxiliary view implementation is still useful for the original
definition of localization-guided reconstruction: it gives a clean way to
rehearse map-side self-localization under perturbed viewpoints. However, it
should feed reliability labels or residual fine-tuning losses, not directly
overwrite the descriptor bank.

## Immediate Next Direction

The next real accuracy path is to move the successful feedback signal earlier:

1. Generate held-out train and rendered-perturbed self-localization episodes.
2. Convert those episodes into per-landmark reliability labels: true-positive
   rate, false-positive rate, local geometric consistency, and view coverage.
3. Train only a small residual adapter or locability/matchability head under a
   trust region to preserve the STDLoc descriptor distribution.
4. Use the calibrated reliability at test time as a prior or PROSAC ordering,
   not as a multi-branch pose selector.

This matches the intended claim: downstream localization feedback affects the
map/reconstruction process and yields a localization-oriented feature field,
while inference remains one clean localization path.

## Query-Like Self-Localization Update

The next implementation step changes the calibration episodes from
feature-field self-consistency to query-like self-localization:

- training/rehearsal pair sampling now searches local and global candidate
  frames, then chooses the farthest pair that still satisfies the overlap
  floor, instead of stopping at the first small frame offset;
- rendered calibration queries now default to `rendered_rgb_teacher`: render a
  novel RGB view, run the frozen SuperPoint/STDLoc query extractor on that
  image, and label 2D-3D matches with the known rendered pose;
- calibration sidecars record pose-delta and overlap statistics, so we can
  verify that self-localization covers query-like viewpoints without using
  test GT.

The first five-scene v64 calibration used 64 train views plus 64 rendered RGB
rehearsal views per scene. The rendered views are no longer tiny perturbations:

| Scene | Mean pose delta | Mean rot delta | Mean overlap | Observed landmarks |
|---|---:|---:|---:|---:|
| GreatCourt | 17.61 m | 29.27 deg | 0.695 | 15,892 |
| KingsCollege | 33.33 m | 40.32 deg | 0.762 | 15,619 |
| OldHospital | 11.20 m | 28.94 deg | 0.749 | 14,921 |
| ShopFacade | 6.48 m | 31.95 deg | 0.687 | 15,981 |
| StMarysChurch | 20.80 m | 46.92 deg | 0.619 | 15,859 |

Pilot test on the first 80 queries per Cambridge scene with
`covisibility_prosac` plus the query-like calibrated matchability:

| Method on q80 | Macro median | Macro R@10cm/5deg | Macro R@5cm/5deg | Macro R@2cm/2deg |
|---|---:|---:|---:|---:|
| STDLoc baseline branch | 9.95 cm / 0.183 deg | 0.560 | 0.348 | 0.108 |
| Historical learned branch | 10.67 cm / 0.168 deg | 0.535 | 0.373 | 0.120 |
| Query-like calibrated prior | 8.90 cm / 0.111 deg | 0.608 | 0.373 | 0.123 |

Scene-level q80 deltas are mixed: KingsCollege, OldHospital, and
StMarysChurch improve, while GreatCourt regresses. This supports the revised
direction but also reinforces the need for a baseline-preserving guard before
claiming full-test SOTA.

Full-test follow-up changed the interpretation. The same query-like calibrated
prior does **not** generalize when applied as an unguarded full-test branch:

| Full-test branch | Macro median | Macro R@10cm/5deg | Macro R@5cm/5deg | Macro R@2cm/2deg |
|---|---:|---:|---:|---:|
| STDLoc baseline branch | 9.13 cm / 0.156 deg | 0.576 | 0.371 | 0.133 |
| Historical learned branch | 10.10 cm / 0.158 deg | 0.568 | 0.394 | 0.144 |
| Query-like calibrated prior | 12.56 cm / 0.159 deg | 0.487 | 0.282 | 0.090 |

This makes the v64 query-like prior a useful negative ablation rather than the
main method. It supports the paper story that localization feedback must be
baseline-preserving and confidence-gated; a static per-landmark reliability
prior, even when generated from broader rendered views, is still not enough for
full-scene query-conditioned ambiguity.

The clean three-branch guarded selector therefore uses a calibration R@5 floor
of 0.2 and a median-error guard. With the query-like branch included, it selects
the same defensible branch set as before:

| Scene | Selected branch |
|---|---|
| GreatCourt | STDLoc baseline |
| KingsCollege | STDLoc baseline |
| OldHospital | STDLoc baseline |
| ShopFacade | historical selected |
| StMarysChurch | STDLoc baseline |

The clean guarded full-test result remains 9.08 cm / 0.153 deg with macro
R@5cm/5deg 0.373 and macro R@2cm/2deg 0.133. This is paper-safe but not yet
SOTA-level; the next accuracy path should be query-conditioned reliability or
learning the feedback into the feature field, not stronger static landmark
filtering.
