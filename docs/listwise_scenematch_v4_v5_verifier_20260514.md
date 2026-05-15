# Listwise SceneMatch verifier / label-quality experiments, 2026-05-14

## Goal

This round tested whether the localization-feedback story can be strengthened by
making the listwise SceneMatchNet supervision closer to train-time
self-localization:

- keep STDLoc/PLY descriptors and single-path PROSAC;
- train a query-conditioned top-K selector with reprojection-error verifier
  supervision;
- avoid per-scene hyperparameter tuning;
- evaluate all Cambridge scenes with the same recipe.

## Code changes

- `train_scene_matcher.py` now supports an optional listwise reprojection
  verifier loss via `--listwise_verifier_loss_weight` and
  `--listwise_verifier_sigma_px`.
- listwise pair caches now carry `reprojection_error`, `query_yx`, and
  `landmark_id`, so training can use self-localization geometry and future
  external matcher teachers can be attached without regenerating descriptors.
- `launch_cambridge_matchability_calibration.py` now exposes
  `--visibility_check`, which lets us generate relaxed diagnostic pair labels
  without hard rendered visibility filtering.

## Full Cambridge results

All rows use:

```text
PLY descriptors + STDLoc query detector + listwise SceneMatchNet top16
+ calibrated matchability prior + single-path OpenCV PROSAC + dense refinement
```

| Variant | Median cm | Median deg | R@10cm/5deg | R@5cm/5deg | R@2cm/2deg | Sparse R@10 | Sparse R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFF residual mainline | 12.456 | 0.156 | 0.485 | 0.284 | 0.085 | 0.406 | 0.196 |
| listwise v4 verifier, weight 0.25, top16 | 12.470 | 0.160 | 0.484 | 0.283 | 0.084 | 0.411 | 0.211 |
| listwise v4 verifier, weight 0.25, top32 | 12.486 | 0.156 | 0.490 | 0.281 | 0.084 | 0.412 | 0.209 |
| **listwise v4 verifier, weight 0.05, top16** | **12.411** | 0.158 | 0.487 | **0.287** | 0.085 | **0.416** | 0.208 |
| listwise v4 verifier, weight 0.05, top16, zscore logits | 12.428 | 0.160 | 0.483 | 0.286 | **0.086** | 0.413 | 0.209 |
| listwise v5 no-visibility labels, weight 0.05, top16 | 12.624 | 0.158 | 0.487 | 0.282 | 0.084 | 0.411 | 0.204 |
| top16 candidate oracle | 11.183 | 0.138 | 0.526 | 0.310 | 0.100 | 0.600 | 0.324 |

## Interpretation

The weaker verifier is useful but not decisive. Dropping the verifier weight
from `0.25` to `0.05` gives the best full-test median in this family and a small
gain over the current residual mainline, but the recall gains are not large
enough for a SOTA claim.

Relaxing visibility labels is a negative result. It fixes an important data
issue, especially on KingsCollege where the listwise positive ratio increases
from `0.031` to `0.202`, but it also introduces geometrically close false
positives that the current descriptor-only listwise model cannot disambiguate.
The full-test result degrades, so no-visibility labels should remain a
diagnostic/negative ablation.

The candidate oracle remains the main clue: top16 contains useful matches, but
the learned selector cannot reliably choose them. This points away from more
global score tuning and toward stronger candidate features or teachers:
LightGlue/DIM hard negatives, local geometric context, or descriptor-bank
improvement. The current single-path listwise selector supports the paper story
only as a modest localization-feedback reliability module, not yet as the final
SOTA component.
