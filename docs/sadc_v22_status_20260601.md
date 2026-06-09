# SADC v2.2 Status - 2026-06-01

## Scope

SADC v2.2 adds no-GT dense-damage activation to the sparse-anchored dense
correspondence path. The goal is to keep normal queries close to native dense
refinement while applying SADC only when sparse anchors indicate likely dense
damage.

This report is diagnostic, not paper-facing:

- current selected Cambridge maps are 8192 sampled landmarks while native
  parity expects 16384;
- official test was evaluated only after freezing the train/self-map threshold;
- official test results below were not used to tune thresholds.

## Implementation

New activation policy:

```text
activation_mode = dense_damage_risk
activation_min_risk = 0.20
activation_min_sparse_confidence = 0.25
```

The activation decision uses no GT. It consumes the existing dense-damage risk
observation built from sparse confidence, dense update risk, and sparse-anchor
conflict. If inactive, SADC leaves the output on the native dense path.

Code paths:

- `loc_gs/dense_support/sadc.py`
- `loc_gs/scripts/eval_sparse_conditioned_dense_control.py`
- `loc_gs/scripts/visualize_stdloc_hard_matches.py`

## Train/Self-Map Gate

Frozen from train/self-map diagnostic subsets:

| Split | Setting | Result |
| --- | --- | --- |
| A sparse-good dense-bad | risk >= 0.20 | median TE 40.54cm -> 36.46cm, delta -4.08cm |
| A sparse-good dense-bad | risk >= 0.20 | mean median delta -3.70cm, Reg20/Reg50 = 0 |
| D normal dense-good, 50 queries | risk >= 0.20 | all inactive, identical to native dense |
| C sparse-catastrophic, 50 queries | risk >= 0.20 | 1/50 active, identical aggregate metrics, Reg20/Reg50 = 0 |

The threshold was selected before official test evaluation.

## Official Test Diagnostic

Recipe:

```text
SADC dense
+ patch candidates
+ anchor-monotonic line search
+ dense-damage activation, risk >= 0.20, sparse_conf >= 0.25
```

Output roots:

```text
output/diagnostics/sadc_v22_full_test_patch_append_anchor_mono_risk020_20260601_*
```

### Per-Scene Summary

| Scene | Active | Median TE Base -> SADC | Mean TE Base -> SADC | R10 Delta | R5 Delta | P95 Delta | CVaR10 Delta | Reg20 / Reg50 | Imp20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 223/760 | 11.073 -> 11.155 (+0.082) | 156.670 -> 156.982 (+0.312) | +0.0026 | -0.0013 | -0.826 | +3.315 | 3 / 1 | 1 |
| KingsCollege | 39/343 | 22.373 -> 22.373 (+0.000) | 46.347 -> 25.419 (-20.928) | +0.0000 | +0.0000 | +0.542 | -205.030 | 0 / 0 | 1 |
| OldHospital | 117/182 | 11.913 -> 11.913 (+0.000) | 22.999 -> 23.063 (+0.064) | -0.0055 | -0.0110 | +0.262 | -0.507 | 0 / 0 | 1 |
| ShopFacade | 29/103 | 2.801 -> 2.825 (+0.025) | 4.852 -> 4.847 (-0.005) | +0.0000 | +0.0000 | -0.152 | -0.093 | 0 / 0 | 0 |
| StMarysChurch | 164/530 | 3.816 -> 3.848 (+0.032) | 13.407 -> 13.416 (+0.009) | +0.0000 | +0.0019 | +0.000 | +0.085 | 0 / 0 | 0 |

### Macro Metrics

| Metric | Base | SADC v2.2 | Delta |
| --- | ---: | ---: | ---: |
| median TE cm | 10.395 | 10.423 | +0.028 |
| mean TE cm | 48.855 | 44.745 | -4.110 |
| R50 | 0.9340 | 0.9337 | -0.0003 |
| R15 | 0.6766 | 0.6771 | +0.0005 |
| R10 | 0.5671 | 0.5666 | -0.0006 |
| R5 | 0.3561 | 0.3540 | -0.0021 |
| R2 | 0.1168 | 0.1169 | +0.0001 |
| P90 TE cm | 36.451 | 36.234 | -0.216 |
| P95 TE cm | 50.918 | 50.883 | -0.035 |
| CVaR10 TE cm | 382.426 | 341.980 | -40.446 |
| TE > 1m | 0.0194 | 0.0177 | -0.0017 |
| TE > 5m | 0.0024 | 0.0021 | -0.0003 |

### Micro Metrics

| Metric | Base | SADC v2.2 | Delta |
| --- | ---: | ---: | ---: |
| median TE cm | 8.979 | 8.962 | -0.017 |
| mean TE cm | 76.516 | 72.905 | -3.611 |
| R50 | 0.9239 | 0.9234 | -0.0005 |
| R15 | 0.6590 | 0.6601 | +0.0010 |
| R10 | 0.5391 | 0.5396 | +0.0005 |
| R5 | 0.3149 | 0.3139 | -0.0010 |
| R2 | 0.0819 | 0.0819 | +0.0000 |
| P90 TE cm | 41.701 | 41.432 | -0.269 |
| P95 TE cm | 64.148 | 65.025 | +0.878 |
| CVaR10 TE cm | 663.492 | 627.376 | -36.116 |
| TE > 1m | 0.0266 | 0.0255 | -0.0010 |
| TE > 5m | 0.0037 | 0.0037 | +0.0000 |

## Interpretation

SADC v2.2 is better behaved than full SADC replacement:

- normal scenes are mostly preserved;
- KingsCollege tail improves substantially because one catastrophic dense case
  is mitigated;
- dense-worse-vs-sparse count drops from 8 to 5 over the five official splits.

It is still not a SOTA/main-result candidate:

- macro median is slightly worse (+0.028cm);
- R5 is slightly worse (-0.21pp macro);
- GreatCourt still has false-positive activation with Reg20=3 and Reg50=1;
- OldHospital recall drops despite tail CVaR improving slightly.

The core remaining issue is not SADC mechanics but activation precision. The
risk score correctly catches some dense-damage tails, but on GreatCourt it also
activates when both sparse and dense are already in a repeated-structure or bad
basin. Anchor-monotonic line search limits some damage but cannot make an
incorrect sparse anchor basin safe.

The train/self-map C-class smoke check does not support simply raising the
sparse-confidence threshold: catastrophic sparse failures are mostly inactive
already. The remaining false positives need a global-consistency or
multi-modality proxy, not another scalar inlier-count gate.

## Next Step

Do not broaden SADC further. The next optimization should target activation
precision:

1. add a sparse-basin validity term: reject or downweight activation when sparse
   anchor geometry itself is poor, multi-modal, or already meter-level by no-GT
   proxies;
2. add a GreatCourt false-positive audit set from train/self-map only;
3. keep native dense as default and treat SADC as tail mitigation until train
   validation shows median and strict recall are non-worse.
