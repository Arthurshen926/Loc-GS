# Loc-GS Sampling Field Budget Evidence, 2026-05-16

## Mainline

This round follows `ChatGPT-Loc-GS.md`: the paper-facing method is a
localization utility / sampling field on top of native STDLoc descriptors.
The deployed path remains single-path:

```text
native STDLoc descriptor map
  -> self-localization distilled selector / sampling field
  -> STDLoc-compatible sampled_idx + sampled_scores + locability
  -> descriptor matching + OpenCV PROSAC PnP
  -> STDLoc-style dense refinement
```

No descriptor replacement, test-time branch selection, or per-query multi-path
selection is used in these runs.

## Clean Baseline Correction

Previous detector folders in source STDLoc maps were contaminated by symlinked
exports. Clean detector payloads were rebuilt and materialized under:

```text
output/unified_lff_v2/native_rebuilt_detector_source_20260516/{scene}
```

Older selector/full results that used polluted detector payloads should remain
diagnostic only.

## q80 Budget Sweep

| Run | Queries | Landmarks | Median cm/deg | R10 | R5 | R2 | Delta vs same-budget native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16k native clean | 400 | 16384 | 7.2234 / 0.1084 | 0.5975 | 0.3750 | 0.1050 | - |
| 12k native | 400 | 12288 | 7.8068 / 0.1119 | 0.5900 | 0.3700 | 0.0900 | - |
| 12k selector cov8 | 400 | 12288 | 7.2673 / 0.1119 | 0.5875 | 0.3725 | 0.0925 | -0.539cm, R10 -0.0025, R5 +0.0025, R2 +0.0025 |
| 12k selector cov8 rank | 400 | 12288 | 7.4005 / 0.1119 | 0.5750 | 0.3800 | 0.1025 | -0.406cm, R10 -0.0150, R5 +0.0100, R2 +0.0125 |
| 8k native | 400 | 8192 | 9.6417 / 0.1312 | 0.5100 | 0.2925 | 0.0875 | - |
| 8k selector | 400 | 8192 | 9.0266 / 0.1327 | 0.5475 | 0.3150 | 0.0800 | -0.615cm, R10 +0.0375, R5 +0.0225, R2 -0.0075 |

## Full Cambridge Validation

| Run | Queries | Landmarks | Median cm/deg | R10 | R5 | R2 | Delta vs 12k native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16k native clean | 1918 | 16384 | 12.0377 / 0.1187 | 0.4359 | 0.2112 | 0.0469 | - |
| 12k native | 1918 | 12288 | 12.5967 / 0.1282 | 0.4192 | 0.2049 | 0.0407 | - |
| 12k selector cov8 | 1918 | 12288 | 12.2966 / 0.1219 | 0.4228 | 0.2044 | 0.0433 | -0.300cm, R10 +0.0036, R5 -0.0005, R2 +0.0026 |
| 12k selector cov8 rank | 1918 | 12288 | 13.0368 / 0.1342 | 0.4088 | 0.1966 | 0.0443 | +0.440cm, R10 -0.0104, R5 -0.0083, R2 +0.0036 |

## Interpretation

Supported:

- Selector-guided sampling with 3D coverage (`12k selector cov8`) improves the
  same-budget native 12k map on full Cambridge median, R10, and R2.
- The gain is achieved with native descriptors and one STDLoc-compatible
  inference path.
- q80 shows a stronger speed/landmark-budget signal: 12k selector cov8 nearly
  recovers the 16k native median while using 25% fewer sampled landmarks.

Not yet supported:

- Strong SOTA accuracy. Clean 16k native full remains better than the current
  12k selector cov8 full result.
- Rank-normalized selector as a main method. It improves q80 strict recall, but
  full Cambridge degrades median, R10, and R5.
- A descriptor-learning claim. These runs keep `descriptor_mode=native`.

## Code Notes

- `selector_transform=rank` was added only as an explicit resampling ablation.
  The default remains `identity`.
- `merge_cambridge_eval_results.py` now keys duplicate checks by
  `(scene, image_name)` so full Cambridge can merge scenes that share relative
  image names while still rejecting duplicate rows inside the same scene.

## Next Required Evidence

1. Add timing/profile metrics to separate sparse matching, PnP, and dense
   refinement cost. Whole-run wall-clock is not enough because dense refinement
   dominates end-to-end runtime.
2. Rebuild a paper-safe feedback cache with full train/rendered source ids and
   explicit disjointness from Cambridge test ids.
3. Improve selector training itself rather than test-set selector-strength
   search: selector-only training, false-positive suppression, and coverage /
   pose-information targets should be selected by self-map validation.
