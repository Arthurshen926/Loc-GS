# LSF v4 Candidate-Probe Adaptive Status 2026-05-24

## Scope

This round evaluates a paper-safe train-dev variant of solver-feedback guided
map editing. It uses candidate-probe feedback and a fixed adaptive edit budget:

```text
effective_edits = min(32, max(4, floor(candidate_gain_count * 0.01)))
```

The budget is derived from self-map solver admissibility metadata, not from
per-query branch selection.

All results below are Cambridge train split train-dev checks with
`configs/stdloc_cambridge_opencv.yaml`. They are OpenCV/PROSAC/MAGSAC
STDLoc-style runs, not official untouched STDLoc parity runs.

## Effective Budgets

| Scene | Candidate gain count | Effective edits |
| --- | ---: | ---: |
| GreatCourt | 1077 | 10 |
| KingsCollege | 858 | 8 |
| OldHospital | 839 | 8 |
| ShopFacade | 462 | 4 |
| StMarysChurch | 746 | 7 |

## Dense Median Translation Error Delta

Delta is candidate minus baseline, in cm. Negative is better.

| Recipe | q80 macro | q80 wins | q160 macro | q160 wins | Main issue |
| --- | ---: | ---: | ---: | ---: | --- |
| max32 | -0.143 | 4/5 | +0.045 | 4/5 | OldHospital q160 regresses +0.412 |
| edits8 | -0.224 | 2/5 | -0.054 | 1/5 | Shop/Great/St small regressions |
| adaptive1p | -0.273 | 3/5 | -0.064 | 2/5 | Shop/St small regressions remain |

## Adaptive1p Per-Scene Delta

| Scene | q80 delta cm | q160 delta cm |
| --- | ---: | ---: |
| GreatCourt | -0.182 | -0.034 |
| KingsCollege | -0.078 | +0.030 |
| OldHospital | -1.214 | -0.348 |
| ShopFacade | +0.045 | +0.011 |
| StMarysChurch | +0.065 | +0.023 |

## Audit Status

All adaptive1p q80/q160 run directories contain:

```text
manifest.json
command.txt
metrics_summary.json
split_audit.json
git_status.txt
```

Split audits pass for all ten runs. The manifests also record:

```text
evaluator_variant = opencv_prosac_magsac
third_party_stdloc_evaluator_modified = true
official_stdloc_parity_candidate = false
```

This is intentional reporting: these results are valid for the OpenCV single
path method-development track, but must not be used as official native STDLoc
parity evidence.

## Current Interpretation

The precision explosion bug has been isolated away from metric scale or GT
units. Current medians are back in the expected centimeter range. The remaining
problem is selection stability: high edit budgets improve easy-scene medians but
can overfit q80 support and damage q160 tail queries; low/adaptive budgets
stabilize OldHospital but leave small regressions on ShopFacade and
StMarysChurch.

The strongest current method claim is therefore:

```text
Solver-feedback guided support editing can improve macro dense pose precision
on train-dev while preserving map budget, but the present selector is not yet
uniformly positive per scene.
```

## Next Work

1. Move OpenCV solver support out of modified `third_party/stdloc` or restore an
   untouched vendored parity path before any paper-facing native STDLoc claim.
2. Add a second fixed adaptive rule that penalizes candidate dense-worsen risk
   more strongly for low candidate-gain scenes, targeting ShopFacade and
   StMarysChurch without per-query branching.
3. Freeze one train-dev recipe only after q80/q160 pass with macro precision
   improvement and no hard-scene q160 regression.

## Follow-Up: Dense-Worsen Penalty

An `adaptive2_densepen05` variant was tested with the same adaptive budget and
an additional dense-worsen utility penalty of `0.5`. It is not selected:

| Recipe | q80 macro | q80 wins | q160 macro | q160 wins |
| --- | ---: | ---: | ---: | ---: |
| adaptive2_densepen05 | -0.240 | 3/5 | +0.041 | 0/5 |

This confirms that dense-worsen penalty alone is too blunt: it helps
StMarysChurch q80, but damages q160 tail behavior, especially OldHospital.

## Frozen Scene-Level Recipe Candidate

A precision-primary q160 scene-level acceptance report was written to:

```text
output/reports/lsf_v4_candidate_probe_scene_recipe_20260524/q160_precision_acceptance.json
```

The frozen manifest is:

```text
output/reports/lsf_v4_candidate_probe_scene_recipe_20260524/frozen_q160_precision_recipe.json
```

Selected maps:

| Scene | Selected candidate |
| --- | --- |
| GreatCourt | adaptive1p |
| KingsCollege | max32 |
| OldHospital | adaptive1p |
| ShopFacade | max32 |
| StMarysChurch | max32 |

This is a scene-level reconstruction-time recipe, not per-query branch
selection.

| Split | Macro dense TE delta | Wins | Macro R@10 | Macro R@5 | Macro R@2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q80 | -0.315 cm | 4/5 | +0.0025 | +0.0050 | -0.0025 |
| q160 | -0.114 cm | 5/5 | +0.0012 | +0.0088 | +0.0038 |

The frozen manifest is intentionally not paper-facing ready because the selected
runs use the OpenCV/PROSAC/MAGSAC variant. The vendored `third_party/stdloc`
evaluator has since been restored to an unmodified state, and OpenCV variant
configs are now rejected unless an explicit variant evaluator is present.

## Follow-Up: Vendored Poselib SPGS Prior Validation

The same scene recipe was re-evaluated with the unmodified vendored STDLoc
evaluator and the fixed poselib SPGS prior config:

```text
third_party/stdloc/configs/stdloc_spgs_cambridge_dense1.yaml
```

This config consumes both `detector/sampled_scores.pkl` and PLY
`locability_logit`, so it validates the support-consistent dense verification
path more directly than `stdloc_cambridge.yaml`.

The precision-primary scene-level acceptance reports are:

```text
output/reports/lsf_v4_scene_recipe_spgs_prior_20260524/q80_precision_acceptance.json
output/reports/lsf_v4_scene_recipe_spgs_prior_20260524/q160_precision_acceptance.json
output/reports/lsf_v4_scene_recipe_spgs_prior_20260524/frozen_precision_recipe.json
```

Accepted recipe:

| Split | Selected LSF scenes | Macro dense TE delta | Recall policy |
| --- | --- | ---: | --- |
| q80 | GreatCourt, StMarysChurch | -0.1252 cm | warnings only |
| q160 | GreatCourt, OldHospital, StMarysChurch | -0.1341 cm | warnings only |

See `docs/lsf_v4_spgs_prior_precision_status_20260524.md` for the full result
table and audit interpretation.
