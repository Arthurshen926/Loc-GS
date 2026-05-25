# LSF v4 SPGS Prior Precision Status, 2026-05-24

## Scope

This note records train-dev validation only. No Cambridge test query image,
pose, feature, descriptor, or result was used for selection.

The evaluated path keeps the vendored STDLoc evaluator unmodified and uses the
vendored poselib SPGS prior config:

```text
third_party/stdloc/configs/stdloc_spgs_cambridge_dense1.yaml
```

This is not the strict official baseline parity config
`third_party/stdloc/configs/stdloc_cambridge.yaml`. It is a fixed method config
that consumes:

- `detector/sampled_scores.pkl` through `sparse.landmark_score_path`
- PLY `locability_logit` through `dense.locability_prior_weight`

The strict official baseline parity check remains separate.

## Fixed Candidate Maps

The candidate maps are the existing LSF v4 scene recipe maps:

| scene | candidate map | edit budget |
| --- | --- | ---: |
| GreatCourt | `map_cambridge_spgs_lsf_v4_candidate_probe_native_adaptive1p_20260524/GreatCourt` | 10 |
| KingsCollege | `map_cambridge_spgs_lsf_v4_candidate_probe_native_20260524/KingsCollege` | 32 |
| OldHospital | `map_cambridge_spgs_lsf_v4_candidate_probe_native_adaptive1p_20260524/OldHospital` | 8 |
| ShopFacade | `map_cambridge_spgs_lsf_v4_candidate_probe_native_20260524/ShopFacade` | 32 |
| StMarysChurch | `map_cambridge_spgs_lsf_v4_candidate_probe_native_20260524/StMarysChurch` | 32 |

All five candidate maps have `sampled_idx=16384`, `detector/sampled_scores.pkl`,
and PLY `locability_logit`.

## Raw Selected-vs-Baseline Results

Output root:

```text
output/stdloc_native/lsf_v4_scene_recipe_spgs_prior_20260524
```

### q80

| scene | dense median TE delta | R10 delta | R5 delta | R2 delta |
| --- | ---: | ---: | ---: | ---: |
| GreatCourt | -0.2792 cm | -0.0125 | -0.0125 | -0.0125 |
| KingsCollege | +0.2382 cm | +0.0000 | +0.0250 | +0.0000 |
| OldHospital | +1.2219 cm | -0.0375 | +0.0375 | +0.0000 |
| ShopFacade | +0.0102 cm | +0.0000 | +0.0000 | +0.0250 |
| StMarysChurch | -0.3470 cm | +0.0500 | -0.0125 | -0.0125 |

Raw macro median TE delta is `+0.1688 cm`, so the all-selected recipe is not
accepted.

### q160

| scene | dense median TE delta | R10 delta | R5 delta | R2 delta |
| --- | ---: | ---: | ---: | ---: |
| GreatCourt | -0.3828 cm | -0.0062 | +0.0125 | -0.0063 |
| KingsCollege | +0.0525 cm | +0.0062 | +0.0062 | +0.0000 |
| OldHospital | -0.1968 cm | -0.0250 | +0.0125 | +0.0125 |
| ShopFacade | +0.1137 cm | +0.0000 | +0.0000 | +0.0187 |
| StMarysChurch | -0.0907 cm | +0.0250 | -0.0250 | +0.0125 |

Raw macro median TE delta is `-0.1008 cm`, with 3/5 scenes improving.

## Precision-Primary Scene-Level Acceptance

Policy:

- accept an LSF candidate map only when dense median TE does not regress;
- treat recall drops as warnings, not hard failures;
- select at scene level only, never per query.

Reports:

```text
output/reports/lsf_v4_scene_recipe_spgs_prior_20260524/q80_precision_acceptance.json
output/reports/lsf_v4_scene_recipe_spgs_prior_20260524/q160_precision_acceptance.json
output/reports/lsf_v4_scene_recipe_spgs_prior_20260524/frozen_precision_recipe.json
```

q80 selected maps:

```text
GreatCourt=LSF
KingsCollege=native
OldHospital=native
ShopFacade=native
StMarysChurch=LSF
```

q80 accepted macro median TE delta: `-0.1252 cm`.

q160 selected maps:

```text
GreatCourt=LSF
KingsCollege=native
OldHospital=LSF
ShopFacade=native
StMarysChurch=LSF
```

q160 accepted macro median TE delta: `-0.1341 cm`.

The frozen precision recipe currently passes the local paper-safety checks:

- all selected split audits passed;
- fixed single-path evaluator;
- no per-query branch selection;
- feedback disabled at evaluation;
- vendored STDLoc evaluator unmodified;
- selected runs are fixed poselib method runs.

## Interpretation

The previous official-default poselib run showed that sparse map edits alone are
too weak under `stdloc_cambridge.yaml`: q80 regressed and q160 improved only
slightly.

This SPGS-prior run validates Module 3 more directly: dense/sparse support must
be consumed by the evaluator, not merely written into the map. Once consumed,
the method has a positive precision-primary train-dev recipe, but the raw
all-selected result still fails on OldHospital q80, KingsCollege, and
ShopFacade.

The current defensible claim is therefore:

> LSF feedback can guide same-budget feature selection and dense support
> reconstruction for fixed STDLoc-style localization, but candidate maps must
> pass a reconstruction-time scene-level precision gate before paper-facing
> evaluation.

This remains a train-dev result. It does not authorize a Cambridge test or SOTA
claim yet.

## Dense-Only Diagnostic

A follow-up diagnostic isolates the dense support field from sparse sampled
landmark edits:

```text
configs/stdloc_spgs_cambridge_dense_lsf_only.yaml
output/stdloc/map_cambridge_spgs_lsf_v4_dense_lsf_only_20260524
output/stdloc_native/lsf_v4_dense_lsf_only_20260524
```

This setting keeps `sampled_idx` unchanged (`max_edits=0`), disables the sparse
landmark prior, and enables only `dense.locability_prior_weight=0.05`.

Raw dense-only results:

| split | macro dense TE delta | wins | macro R@10 | macro R@5 | macro R@2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q80 | +0.0223 cm | 0/5 | -0.0025 | +0.0000 | +0.0000 |
| q160 | +0.0179 cm | 2/5 | -0.0025 | +0.0012 | +0.0012 |

The dense-only all-selected candidate is therefore rejected. It shows that the
current dense LSF target is not yet strong enough as a standalone dense
verification prior. The positive SPGS-prior recipe above comes from the
combination of same-budget sparse support editing and prior-consumed dense
support, not from dense locability alone.

Diagnostic acceptance reports:

```text
output/reports/lsf_v4_dense_lsf_only_20260524/q80_precision_acceptance.json
output/reports/lsf_v4_dense_lsf_only_20260524/q160_precision_acceptance.json
```
