# Loc-GS Related-Work Notes and Mainline Update

Date: 2026-05-12

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

## Next Experiments

1. Use `covisibility_prosac` as the next single-path candidate.
2. Compare it against `protected` on the same query subset across all five
   Cambridge scenes.
3. Promote it to full-test evaluation if it improves aggregate dense
   median or recall without causing a KingsCollege/ShopFacade regression.
4. If full-test PROSAC is still mixed, keep it as an engineering ablation and
   move the same query-conditioned priority into training-time hard-negative
   matchability labels instead of relying only on test-time sampling.
5. For training, avoid full descriptor replacement. Prefer a frozen base
   descriptor plus small residual adapter, trust-region regularization, and
   hard-negative matchability labels generated by self-localization.
