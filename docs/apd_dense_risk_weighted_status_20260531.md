# APD-Dense Risk-Weighted Status - 2026-05-31

## Scope

This is a train/self-map diagnostic check on Phase0 cases only. It is not
paper-facing because the current `locgsctl list-scenes` audit reports
`sampled_count=8192` while native STDLoc parity expects `16384`.

Case source:

```text
output/diagnostics/clean_render_phase0_benchmark_20260530/phase0_cases.csv
```

## Implemented

- APD final-pose usage is now optional through `--apd_use_refined_pose`.
- Added `--apd_risk_weighted` and `--apd_risk_max_beta`.
- Risk-weighted APD keeps native dense as the default pose and blends toward APD:

```text
T_final = Interp(T_native_dense, T_APD, beta)
beta = dense_damage_risk * apd_risk_max_beta * apd_refine_success
```

- Dense-damage risk uses only no-GT observable diagnostics:
  sparse inlier count, sparse-anchor residual conflict, dense-vs-sparse pose
  delta, and dense reprojection quality.

## Results

### A: sparse-good / dense-bad hard cases

Output:

```text
output/diagnostics/apd_dense_eval_phase0_A_all_risk_weighted_v2_20260531
```

| Method | Median TE cm | P90 TE cm | P95 TE cm | R@50cm/5deg | Dense-worsened count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native dense | 40.5384 | 53.2902 | 54.3606 | 0.8333 | 7 |
| Risk-weighted APD | 23.3105 macro / 31.7166 row median | 27.6513 | 29.7173 | 1.0000 | 5 |
| Forced APD + patch | 21.2964 | 24.0983 | 24.3879 | 1.0000 | 5 |
| Sparse-only | 13.4207 row median | - | - | - | - |

Risk-weighted APD recovers most of the hard-case damage relative to native
dense, but it still does not reach sparse-only. The remaining gap is caused by
APD pose refinement itself, not by the risk detector.

### D50: normal dense-good smoke

Output:

```text
output/diagnostics/apd_dense_eval_phase0_D50_risk_weighted_v2_20260531
```

| Method | Median TE cm | P90 TE cm | P95 TE cm | R@10cm/5deg | R@5cm/5deg | Dense-worsened count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native dense | 5.6578 | 9.4855 | 10.5965 | 0.9400 | 0.3600 | 0 |
| Risk-weighted APD | 5.6670 | 9.4989 | 10.5603 | 0.9400 | 0.3600 | 0 |
| Forced APD + patch | 10.7557 | - | - | 0.4400 | 0.1000 | 12 |

Risk-weighting protects normal cases from the large APD full-replacement
regression. Median changes by about `+0.009 cm`, with no R@10/R@5 drop and no
dense-worsened increase on this smoke subset.

## Risk Detector Check

Output:

```text
output/diagnostics/apd_dense_risk_features_phase0_AD_20260531/dense_damage_risk_report_v2.json
```

On the current A+D diagnostic set:

```text
query_count = 57
positive_count = 7
AUC = 1.0
precision@positive_count = 1.0
recall@positive_count = 1.0
```

This validates the current rule on the small Phase0 split, but it is not enough
to claim full-split generalization.

## Interpretation

The current best policy is:

```text
native dense default
+ no-GT dense-damage risk
+ APD as mitigation candidate only
```

This satisfies the immediate engineering goal better than APD full replacement:
it improves A-type hard cases while keeping D-type normal cases close to native.

The method is still not final:

- A-type hard cases remain worse than sparse-only.
- Median sparse-anchor monotonicity is not a sufficient safety constraint; it
  accepted APD poses that preserved median anchor residual while worsening TE.
- The next useful refinement is not more thresholds; it is improving APD's
  internal pose update so high-risk cases approach the sparse-only ceiling.

## Next Gate

Before official test, run train/self-map full-split policy simulation:

```text
native dense
forced APD
risk-weighted APD
risk-weighted APD + stronger anchor residual objective
oracle min(sparse, native, APD)
```

Required pass condition:

```text
median TE not worse than native
R@10/R@5 not worse than native
P95/CVaR improves
dense_worsened_count does not increase
Reg20 < Imp20
```
