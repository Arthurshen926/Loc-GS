# Clean Render Visual Autopsy 20260530

This note checks whether the new Clean Render Generator contradicts the earlier
sparse-conditioned dense-control results. It uses existing train/self-map
diagnostic artifacts only.

## Sources

- `output/diagnostics/slcdp_train_selfmap_sparse_conditioned_failure_analysis_20260529/case_analysis.csv`
- `output/diagnostics/slcdp_train_selfmap_sparse_conditioned_failure_analysis_20260529/contact_sheets/*.jpg`
- per-case `match_summary.json` under `visualizations/none/` and
  `visualizations/sparse_conditioned/`

## Main Finding

The earlier sparse-conditioned scheme is genuinely effective on some hard cases.
The issue is not that Clean Render Generator fails to find those cases. When run
on the existing candidate lists, it selects the same effective clean-render
candidates for the large positive cases.

The actual problem is sharper:

> Current clean-render scoring separates dirty native renders from cleaner
> sparse-anchor renders, but it does not separate pose-improving clean renders
> from pose-regressing clean renders.

In other words, sparse-anchor render health is necessary but not sufficient for
final dense pose improvement.

## Candidate Selection Replay

| Case | Category | Old Selected | Clean Render Selected | Delta cm |
| --- | --- | --- | --- | ---: |
| GreatCourt #686 | regression | gated_base | gated_base | +27.59 |
| GreatCourt #691 | regression | gated_base | gated_base | +21.38 |
| GreatCourt #1372 | regression | gated_base | gated_ray_up-5.000 | +40.92 |
| GreatCourt #1392 | improvement | gated_ray_up-5.000 | gated_ray_up-5.000 | -804.53 |
| GreatCourt #1496 | improvement | gated_base | gated_base | -288.32 |
| GreatCourt #1520 | improvement | gated_base | gated_base | -376.85 |
| KingsCollege #346 | improvement | gated_ray_up-5.000 | gated_ray_up-5.000 | -24.34 |
| KingsCollege #620 | small_regression | gated_ray_centroid+5.000 | gated_ray_centroid+5.000 | +17.68 |
| StMarysChurch #667 | regression | gated_base | base after sparse-confidence fix | +27.79 |
| StMarysChurch #681 | improvement | gated_base | base after sparse-confidence fix | -8.72 |

The StMarysChurch changes come from a bug fix added during this autopsy:
non-base clean renders are now rejected when the preflight explicitly says
`sparse_confident=false`. That aligns the module with its own premise: the clean
render is sparse-anchor conditioned, so untrusted sparse anchors cannot justify a
non-native render candidate.

## Visual Interpretation

### GreatCourt #1392: real positive case

Contact sheet:

`output/diagnostics/slcdp_train_selfmap_sparse_conditioned_failure_analysis_20260529/contact_sheets/GreatCourt_01392_compare.jpg`

Native render/dense:

- sparse TE `701.6cm`, native dense `956.8cm`
- native sparse-ray depth has `visible_ratio=0.0`, `near_occluder_ratio=1.0`
- dense good ratio is almost zero: `0.0046`

Sparse-conditioned render:

- selected `gated_ray_up-5.000`
- dense TE improves to `152.3cm`
- sparse-ray depth becomes visible: `visible_ratio=1.0`, `near_occluder_ratio=0.0`
- dense good ratio rises to `0.2283`

This is exactly the intended failure recovery: native render is a smeared/occluded
view, and guided/gated rendering creates a much more usable dense view.

### KingsCollege #346: real positive case, but still not a pure match-count story

Contact sheet:

`output/diagnostics/slcdp_train_selfmap_sparse_conditioned_failure_analysis_20260529/contact_sheets/KingsCollege_00346_compare.jpg`

- native dense `41.0cm`
- sparse-conditioned dense `16.6cm`
- base sparse-ray preflight is catastrophic: `visible_ratio=0.0`,
  `near_occluder_ratio=1.0`
- selected `gated_ray_up-5.000`

The render cleanup is visible in the sparse-ray depth panel. However, dense good
ratio decreases in the diagnostic table, while pose improves. This means the
module should not be judged only by dense match count or GT-good match ratio; the
solver can improve when the changed render gives a better minimal set even if
many dense correspondences remain noisy.

### GreatCourt #691: false positive clean render

Contact sheet:

`output/diagnostics/slcdp_train_selfmap_sparse_conditioned_failure_analysis_20260529/contact_sheets/GreatCourt_00691_compare.jpg`

- sparse TE `25.5cm`
- native dense `32.4cm`
- sparse-conditioned dense `53.7cm`
- selected `gated_base`

The bottom-right sparse-ray-depth panel becomes much greener after gating:
`visible_ratio` rises from `0.144` to `0.963`, and `near_occluder_ratio` drops
from `0.843` to `0.026`. So clean-render metrics improve strongly.

But the native dense path was already stable:

- native dense solver inlier ratio `0.808`
- native step retained sparse inlier ratio `0.991`
- native dense update was small: translation `0.124m`, rotation `0.098deg`

Gating improves sparse-anchor render consistency but perturbs an already-good
dense basin. This is why "render looks cleaner" does not guarantee final pose
improvement.

### StMarysChurch #667: invalid sparse-anchor premise

Contact sheet:

`output/diagnostics/slcdp_train_selfmap_sparse_conditioned_failure_analysis_20260529/contact_sheets/StMarysChurch_00667_compare.jpg`

- sparse TE `550.1cm`
- native dense `386.2cm`
- sparse-conditioned dense `414.0cm`
- selected `gated_base`
- preflight says `sparse_confident=false`

Here the sparse anchors are not reliable enough to drive clean render generation.
The tree/occlusion-heavy view can be locally made more anchor-visible, but the
anchor set itself is from a wrong or weak sparse hypothesis. Clean Render
Generator now rejects this non-base candidate.

## Why Phase1 Looked Negative

Phase1 was not a full localization evaluation of the old sparse-conditioned
module. It was a diagnostic check asking whether the current render-health
metrics alone can decide which non-base clean render should be consumed.

They cannot:

- improvement cases have high visible ratio and feature cosine;
- regression cases also have high visible ratio and feature cosine;
- therefore these metrics are not sufficient as an acceptance signal.

This does not negate the old sparse-conditioned positives. It says the current
Clean Render Generator is only a render-quality scorer, not yet a safe
pose-improvement predictor.

## Current Bug Fix

`loc_gs.dense_support.clean_render_generator._anchor_safe` now rejects candidates
when `preflight.sparse_confident is False`.

Regression test:

`tests/test_clean_render_generator.py::test_clean_render_generator_rejects_non_base_when_sparse_anchors_are_not_confident`

## Remaining Root Cause

The remaining false positives are not fixed by a simple visibility threshold.
GreatCourt #691 and #1372 have confident sparse anchors and much cleaner
sparse-ray preflight after gating, yet final pose gets worse.

The missing signal is global dense-transition safety:

- native dense may already retain sparse anchors and make only a small update;
- a clean render may improve anchor visibility while reducing global dense
  feature consistency;
- gating currently protects sparse rays, not the full dense feature
  distribution.

Next implementation should not add another hard threshold directly to final
pose selection. It should add a diagnostic score for global consistency before
clean renders are consumed:

- native dense step stability: sparse-anchor retention, translation/rotation
  update size, dense solver inlier ratio;
- clean-render global match stability: MNN count, spatial coverage,
  feature entropy/variance, dense match concentration;
- off-anchor consistency: matches away from sparse anchors should not collapse
  or shift to a repeated facade mode.

Until those signals pass train/self-map validation, Clean Render Generator should
remain diagnostic-only.
