# Sparse-Only PGSH Round 1 Diagnostic

Date: 2026-05-31

Scope: sparse stage only. Dense refinement is intentionally excluded.

## Implemented

- Added base-sparse reference reprojection filtering for patch-level sparse matches.
- Added precision-first patch scoring and score-ranked cluster selection.
- Added OpenCV PROSAC/MAGSAC diagnostic solver support with query-observable match ordering.
- Added sparse PnP reprojection-threshold override for diagnostic sweeps.
- Added capped patch-group coreset selection via `--pgsh_group_max_patches`.
- Added focused unit tests in `tests/test_patch_guided_sparse.py`.

## Key Diagnostic Result

KingsCollege test query #33 (`seq2/frame00034.png`) is a sparse failure case:

| Config | Sparse TE cm | Sparse RE deg | Decision |
| --- | ---: | ---: | --- |
| base sparse | 519.458 | 1.550 | native |
| reference filter + precision selection | 209.528 | 1.556 | PGSH |
| + PROSAC/MAGSAC reference ordering | 106.142 | 0.213 | PGSH |
| + 4px PnP threshold | 70.883 | 0.813 | PGSH |
| + cap top-2 patch group | 59.175 | 0.627 | PGSH |

Forced-good diagnostics showed that correct correspondences exist in #33, but
false consensus dominates the original sparse PnP. The improvement comes from
raising the effective correct-match ratio and reducing solver-level ambiguity,
not from dense refinement.

## Boundary

Unconditional PGSH is not safe. With the same cap-2 configuration:

| Query | Base TE cm | Unconditional PGSH TE cm | Base inliers |
| --- | ---: | ---: | ---: |
| KingsCollege #61 | 34.747 | 92.691 | 491 |
| KingsCollege #62 | 38.700 | 231.225 | 485 |
| KingsCollege #79 | 14.204 | 26.131 | 570 |

These are not sparse catastrophic failures. Patch-only hypotheses can lose
global geometric constraints and regress even with many locally consistent
matches.

## Protected Diagnostic Gate

Using the existing base-inlier gate (`pgsh_max_base_inliers_for_override=80`)
keeps stable native sparse poses and only applies PGSH to low-confidence sparse
failures:

| Query | Base TE cm | Gated PGSH TE cm | Decision | Base inliers |
| --- | ---: | ---: | --- | ---: |
| KingsCollege #33 | 519.458 | 59.175 | PGSH | 32 |
| KingsCollege #61 | 34.747 | 34.747 | fallback native | 491 |
| KingsCollege #62 | 38.700 | 38.700 | fallback native | 485 |
| KingsCollege #79 | 14.204 | 14.204 | fallback native | 570 |

This gate is diagnostic/failure-recovery evidence. Per-query branch selection
must not be presented as a paper-facing main path without a separate
paper-safety argument.

## Artifacts

- `output/diagnostics/sparse_only_pgsh/kings33_sparse_refbase96_precision_prosac_thr4_cap2_20260531/`
- `output/diagnostics/sparse_only_pgsh/kings_sparse_cap2_batch_20260531/`
- `output/diagnostics/sparse_only_pgsh/kings_sparse_cap2_gate80_20260531/`

## Next Sparse-Only Steps

1. Move from official-test diagnostics to train/self-map hard validation.
2. Mine all low-base-inlier Cambridge train queries and run the gated PGSH recipe.
3. Replace the diagnostic gate with a paper-safe sparse confidence policy or keep
   PGSH as an ablation/failure analysis only.
4. Add patch-pair conflict checks so false local consensus is suppressed before
   PnP, rather than only after patch selection.

## Round 2 Formal Sparse Path Update

After moving from the diagnostic PGSH entry point to
`eval_cambridge_hybrid` with `dense_iters=0`, the largest sparse-only signal is
not patch-level filtering. It is query keypoint availability.

On KingsCollege train, using the same STDLoc detector/map/solver path:

| Config | Queries | Median TE cm | Median RE deg | R@50cm/5 | R@10cm/5 | Avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k512 baseline | 1220 | 34.516 | 0.489 | 0.713 | 0.044 | 57.2 |
| k2048 baseline | 1220 | 25.779 | 0.336 | 0.774 | 0.109 | 167.1 |

On OldHospital train:

| Config | Queries | Median TE cm | Median RE deg | R@50cm/5 | R@10cm/5 | TE>5m | Avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k512 baseline | 895 | 76.776 | 1.403 | 0.344 | 0.015 | 0.199 | 21.5 |
| k2048 baseline | 895 | 38.019 | 0.734 | 0.590 | 0.078 | 0.125 | 65.1 |

Patch-consensus filtering was negative on KingsCollege train q261:

| Config | Sparse TE cm | Sparse RE deg | Avg inliers |
| --- | ---: | ---: | ---: |
| k2048 no filter | 15.752 | 0.292 | 263 |
| score top512 | 20.890 | 0.458 | 166 |
| patch-consensus top512 | 20.565 | 0.514 | 32 |
| patch-consensus top1024 | 19.147 | 0.470 | 48 |

Interpretation:

- Many earlier "sparse hard" examples were made worse by an insufficient
  keypoint budget or by the diagnostic path, not necessarily by dense-stage
  failures.
- Increasing keypoints directly improves Availability and inlier count, giving
  much larger gains than map-score edits or patch filtering.
- It is not a complete solution: KingsCollege k2048 improves median/R10 but
  worsens the extreme tail relative to k512 (`TE>5m` 0.008 -> 0.020). The next
  main sparse method should keep the k2048 availability benefit while adding
  ambiguity-aware PROSAC ordering or do-no-harm filtering.

Current paper-safe direction:

1. Treat 2048-keypoint sparse as the new strong sparse baseline candidate, not
   as an LSF contribution.
2. Keep PGSH as diagnostic/ablation unless replaced by a fixed single-path
   policy.
3. Focus method development on ambiguity-aware match ordering, descriptor
   margin, and solver geometry rather than patch branch selection.

## Round 2 Full-Train Sparse-Only Table

Common settings:

```text
eval_split=train
dense_iters=0
query_detector=stdloc
landmark_source=stdloc_detector
solver=opencv_prosac_magsac
sparse_reprojection_error=4.0
sparse_pnp_iterations=20000
```

| Scene | K | Queries | Median TE cm | Median RE deg | P90 TE cm | P95 TE cm | CVaR10 TE cm | TE>1m | TE>5m | R@50/5 | R@25/2 | R@10/5 | R@5/5 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 512 | 1532 | 70.34 | 0.456 | 10293.16 | 20034.26 | 2180581696.67 | 0.406 | 0.176 | 0.409 | 0.172 | 0.020 | 0.002 | 45.8 |
| GreatCourt | 2048 | 1532 | 42.94 | 0.240 | 602.28 | 10538.06 | 2180568652.09 | 0.289 | 0.109 | 0.551 | 0.270 | 0.050 | 0.011 | 121.0 |
| KingsCollege | 512 | 1220 | 34.52 | 0.489 | 84.34 | 113.16 | 850.28 | 0.073 | 0.008 | 0.713 | 0.343 | 0.044 | 0.006 | 57.2 |
| KingsCollege | 2048 | 1220 | 25.78 | 0.336 | 78.51 | 105.32 | 1314.04 | 0.057 | 0.020 | 0.774 | 0.486 | 0.109 | 0.020 | 167.1 |
| OldHospital | 512 | 895 | 76.78 | 1.403 | 2734.19 | 4082.86 | 4595.29 | 0.406 | 0.199 | 0.344 | 0.122 | 0.015 | 0.001 | 21.5 |
| OldHospital | 2048 | 895 | 38.02 | 0.734 | 1238.24 | 2461.83 | 2986.36 | 0.265 | 0.125 | 0.590 | 0.360 | 0.078 | 0.010 | 65.1 |
| ShopFacade | 512 | 231 | 18.33 | 1.001 | 849.00 | 1560.31 | 2075.47 | 0.190 | 0.117 | 0.745 | 0.584 | 0.264 | 0.091 | 22.5 |
| ShopFacade | 2048 | 231 | 8.06 | 0.409 | 37.91 | 237.73 | 716.49 | 0.082 | 0.030 | 0.900 | 0.827 | 0.593 | 0.307 | 82.6 |
| StMarysChurch | 512 | 1487 | 4773.32 | 107.924 | 9371.01 | 11738.01 | 13304.47 | 0.783 | 0.773 | 0.206 | 0.157 | 0.056 | 0.015 | 14.3 |
| StMarysChurch | 2048 | 1487 | 4204.18 | 90.819 | 8588.36 | 9970.58 | 10624.69 | 0.718 | 0.703 | 0.272 | 0.247 | 0.141 | 0.037 | 47.1 |

Five-scene conclusion:

- 2048 keypoints improves median TE and R@10/5 on every train scene.
- The gains are method-level in size, not small decimal changes.
- StMarysChurch remains a hard failure boundary: even after the availability
  boost, sparse median remains meter-scale. This points to descriptor
  distinguishability / false consensus / map or detector alignment issues.
- KingsCollege has a mild tail-risk regression at TE>5m despite better median
  and recall, so 2048 should be paired with fixed ambiguity-aware ordering or a
  solver geometry safeguard before becoming the frozen sparse recipe.

## Round 2 Candidate Expansion Finding

A concrete implementation bug was found and fixed:

```text
--landmark_candidate_source all_gaussians
```

previously fell back to `sampled_ids[:keep]` whenever no landmark score weights
were supplied. This silently disabled the intended all-Gaussian availability
audit. The no-rescoring path now exposes `ids_all[:keep]` as expected.

This matters. On StMarysChurch train query index 730:

| Candidate pool | Landmarks | Sparse TE cm | Sparse RE deg | Inliers |
| --- | ---: | ---: | ---: | ---: |
| sampled k2048 | 8192 | 7580.08 | 138.484 | 7 |
| all-Gaussian before fix | 8192 | 5966.65 | 77.015 | 7 |
| all-Gaussian after fix | 200000 | 3.10 | 0.125 | 644 |

On a 36-query StMarysChurch hard subset starting at query index 730:

| Candidate pool | Median TE cm | Median RE deg | R@50/5 | R@10/5 | TE>5m | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sampled k2048 | 6188.97 | 141.765 | 0.028 | 0.000 | 0.833 | 8.9 |
| all-Gaussian 200k | 3.57 | 0.103 | 1.000 | 1.000 | 0.000 | 608.5 |

On full StMarysChurch train:

| Candidate pool | Median TE cm | Median RE deg | P90 TE cm | P95 TE cm | TE>1m | TE>5m | R@50/5 | R@25/2 | R@10/5 | R@5/5 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sampled k2048 | 4204.18 | 90.819 | 8588.36 | 9970.58 | 0.718 | 0.703 | 0.272 | 0.247 | 0.141 | 0.037 | 47.1 |
| all-Gaussian 200k | 7.22 | 0.163 | 16.43 | 24.95 | 0.015 | 0.011 | 0.977 | 0.950 | 0.711 | 0.295 | 462.5 |

Interpretation:

- StMarysChurch was not primarily a dense failure. It was a sparse candidate
  availability failure induced by the native sampled sparse map.
- The next method should not be patch branch selection. It should be
  availability-expanded sparse map construction: use all-Gaussian/large-pool
  evidence to build a compact paper-safe sparse candidate map.
- A direct 200k-landmark query pool is a diagnostic upper bound and may be too
  large for final reporting; the publishable version should distill it into a
  fixed-size evidence-gated sparse map.

Full five-scene train sparse-only candidate expansion:

| Scene | Pool | Median TE cm | Median RE deg | P90 TE cm | P95 TE cm | TE>1m | TE>5m | R@50/5 | R@25/2 | R@10/5 | R@5/5 | Inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | sampled8k-k2048 | 42.94 | 0.240 | 602.28 | 10538.06 | 0.289 | 0.109 | 0.551 | 0.270 | 0.050 | 0.011 | 121.0 |
| GreatCourt | allgauss200k | 31.36 | 0.127 | 118.85 | 181.85 | 0.125 | 0.028 | 0.728 | 0.366 | 0.077 | 0.022 | 484.4 |
| KingsCollege | sampled8k-k2048 | 25.78 | 0.336 | 78.51 | 105.32 | 0.057 | 0.020 | 0.774 | 0.486 | 0.109 | 0.020 | 167.1 |
| KingsCollege | allgauss200k | 19.17 | 0.240 | 61.99 | 77.04 | 0.005 | 0.000 | 0.834 | 0.617 | 0.164 | 0.030 | 712.8 |
| OldHospital | sampled8k-k2048 | 38.02 | 0.734 | 1238.24 | 2461.83 | 0.265 | 0.125 | 0.590 | 0.360 | 0.078 | 0.010 | 65.1 |
| OldHospital | allgauss200k | 15.85 | 0.291 | 52.42 | 69.50 | 0.018 | 0.000 | 0.889 | 0.672 | 0.309 | 0.089 | 250.1 |
| ShopFacade | sampled8k-k2048 | 8.06 | 0.409 | 37.91 | 237.73 | 0.082 | 0.030 | 0.900 | 0.827 | 0.593 | 0.307 | 82.6 |
| ShopFacade | allgauss200k | 2.85 | 0.134 | 7.61 | 10.33 | 0.009 | 0.009 | 0.987 | 0.987 | 0.944 | 0.801 | 413.0 |
| StMarysChurch | sampled8k-k2048 | 4204.18 | 90.819 | 8588.36 | 9970.58 | 0.718 | 0.703 | 0.272 | 0.247 | 0.141 | 0.037 | 47.1 |
| StMarysChurch | allgauss200k | 7.22 | 0.163 | 16.43 | 24.95 | 0.015 | 0.011 | 0.977 | 0.950 | 0.711 | 0.295 | 462.5 |

Macro:

| Pool | Median TE cm | Median RE deg | P90 TE cm | P95 TE cm | TE>1m | TE>5m | R@50/5 | R@25/2 | R@10/5 | R@5/5 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sampled8k-k2048 | 863.80 | 18.508 | 2109.06 | 4662.71 | 0.282 | 0.197 | 0.617 | 0.438 | 0.194 | 0.077 | 96.6 |
| allgauss200k | 15.29 | 0.191 | 51.46 | 72.73 | 0.034 | 0.010 | 0.883 | 0.718 | 0.441 | 0.247 | 464.6 |

Compact StMarysChurch distillation:

| Pool | Median TE cm | Median RE deg | P90 TE cm | P95 TE cm | TE>1m | TE>5m | R@50/5 | R@25/2 | R@10/5 | R@5/5 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sampled8k-k2048 | 4204.18 | 90.819 | 8588.36 | 9970.58 | 0.718 | 0.703 | 0.272 | 0.247 | 0.141 | 0.037 | 47.1 |
| compact16k, no native core | 16.46 | 0.472 | 7862.38 | 9899.91 | 0.293 | 0.280 | 0.688 | 0.605 | 0.275 | 0.052 | 86.2 |
| compact16k, 50% native core | 16.33 | 0.457 | 3821.25 | 7187.71 | 0.202 | 0.159 | 0.748 | 0.632 | 0.283 | 0.062 | 80.3 |
| allgauss200k upper bound | 7.22 | 0.163 | 16.43 | 24.95 | 0.015 | 0.011 | 0.977 | 0.950 | 0.711 | 0.295 | 462.5 |

Compact conclusion:

- Keeping a native core is necessary for do-no-harm behavior.
- Global 16k scoring is not sufficient for tail coverage; it fixes median but
  still leaves sequence-level holes.
- The next implementation should build a coverage-aware compact map: native
  safe core plus all-Gaussian additions constrained by train-view/sequence/query
  coverage, not just global composite score.
