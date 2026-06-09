# SADC v2 Diagnostic Status 2026-06-01

## Scope

SADC is implemented as sparse-anchored dense correspondence construction, not
pose-level branch selection. The fixed diagnostic recipe evaluated here is:

```text
candidate_render_control = none
sadc_dense = true
sadc_include_patch_candidates = true
sadc_mode = patch_append
sadc_anchor_monotonic = true for v2.1
```

This keeps the STDLoc dense solver path and changes only the dense
correspondence pool. Native dense correspondences are preserved; patch
correspondences are appended only when they satisfy sparse-anchor consistency.
The v2.1 anchor-monotonic step is a deterministic trust-region over the dense
update using sparse PnP anchors and no GT.

All runs below are diagnostic because the current selected Cambridge maps are
8192 sampled landmarks while the native expected count is 16384.

## Implementation

- `loc_gs/dense_support/sadc.py`
  - `SADCCorrespondencePolicy`
  - `sanitize_sadc_correspondences`
  - `apply_sadc_anchor_monotonic_update`
- `loc_gs/scripts/visualize_stdloc_hard_matches.py`
  - SADC correspondence pool injection into dense matching diagnostics.
- `loc_gs/scripts/eval_sparse_conditioned_dense_control.py`
  - SADC CLI, manifest fields, row-level SADC and anchor-monotonic diagnostics.

## Train/Self-Map Gate

The recipe was fixed on train/self-map diagnostics before running official test.

| subset | base median TE | SADC median TE | delta | R5 delta | reg20 | imp20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A sparse-good dense-bad | 40.54cm | 36.46cm | -4.08cm | 0.0000 | 0 | 0 |
| D normal smoke, 50 GreatCourt | 5.66cm | 5.61cm | -0.05cm | +0.0200 | 0 | 0 |

## Official Cambridge Test Diagnostic

Fixed recipe, no test-set tuning after the train gate.

| scene | base med TE | SADC v2.1 med TE | delta | R10 delta | R5 delta | P95 delta | CVaR10 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 11.07 | 10.81 | -0.26 | +0.0118 | +0.0026 | -0.41 | -694.24 |
| KingsCollege | 22.37 | 22.50 | +0.12 | +0.0029 | +0.0000 | +0.54 | -204.77 |
| OldHospital | 11.91 | 11.50 | -0.41 | -0.0110 | -0.0110 | +0.26 | -0.51 |
| ShopFacade | 2.80 | 2.89 | +0.09 | +0.0000 | +0.0097 | -0.15 | -0.10 |
| StMarysChurch | 3.82 | 3.84 | +0.03 | +0.0000 | +0.0000 | +0.00 | -40.38 |
| macro | 10.40 | 10.31 | -0.09 | +0.0008 | +0.0003 | +0.05 | -188.00 |

## Hard-Tail Analysis

Official test row-level diagnostic, used only for analysis:

| subset | n | median delta | mean delta | imp20 | reg20 | imp50 | reg50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base TE > 50cm | 146 | +0.16 | -426.65 | 6 | 5 | 4 | 3 |
| base TE > 100cm | 51 | -0.09 | -1222.15 | 6 | 5 | 4 | 3 |
| base TE > 500cm | 7 | -1800.29 | -8941.31 | 4 | 2 | 4 | 1 |
| base dense worse than sparse by >20cm | 8 | -919.85 | -8536.82 | 6 | 0 | 4 | 0 |
| base dense worse than sparse by >5cm | 72 | -0.43 | -949.20 | 6 | 0 | 4 | 0 |

Anchor-monotonic selected `alpha < 1` for only 9/1918 test queries. It is
therefore a tail guard, not a broad median optimizer.

## Interpretation

SADC v2.1 achieves the intended first-order behavior on dense-damage hard
cases: when native dense strongly violates sparse anchors, the method often
prevents catastrophic dense drift. This is visible in the dense-worse-than-
sparse subset: 6 improvements over 20cm and no regressions over 20cm.

The method is not a SOTA main result. Macro median improves only 0.09cm, strict
recall changes are tiny, and OldHospital R5/R10 drop. The remaining failure mode
is clear: if sparse itself is in a wrong basin, anchor-monotonic can pull a
better dense result back toward a wrong sparse pose. GreatCourt contains the
largest such regressions.

## Next Step

Keep SADC v2.1 as a diagnostic tail-mitigation module. The next method-level
step should be a no-GT sparse-anchor reliability gate for anchor-monotonic, but
only if it can be validated on train/self-map hard and normal subsets before any
official test run. A safe gate cannot rely on test GT and must avoid becoming a
per-query oracle branch selector.
