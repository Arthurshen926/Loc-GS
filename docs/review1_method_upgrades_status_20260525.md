# Review1 Method Upgrades Status - 2026-05-25

## Current Decision

The v6 saturated coverage sprint showed that large static sampled-index editing
does not yet beat v5-128. The route is therefore adjusted from "more selection
tuning" to tested method components that can change the matching representation
and dense correspondence acceptance.

This document maps the seven `review1.md` directions to concrete code and the
next validation gate. These modules are offline/train-dev components. None of
them changes Cambridge test splits, metric definitions, or the vendored STDLoc
evaluator.

## Direction Coverage

| Review1 Direction | Code Status | Primary Files | Next Gate |
| --- | --- | --- | --- |
| 1. Query-conditioned support banks | Implemented as deterministic offline banks/router | `loc_gs/stdloc_native/support_banks.py`, `tests/test_support_banks.py` | Build train/self-map banks, then q80/q160 sparse smoke |
| 2. Solver-weighted landmark feature fusion | Implemented as tensor fusion core plus pair-cache artifact CLI with native fallback/trust region | `loc_gs/stdloc_native/solver_weighted_feature_fusion.py`, `loc_gs/scripts/build_solver_weighted_feature_fusion.py`, `tests/test_solver_weighted_feature_fusion.py`, `tests/test_build_solver_weighted_feature_fusion_cli.py` | Export fused descriptor bank for one scene, run sparse-only train-dev |
| 3. Dense match verifier | Implemented as multi-signal dense correspondence scorer/filter | `loc_gs/dense_support/dense_match_verifier.py`, `tests/test_dense_match_verifier.py` | Dense transition audit: sparse-correct to dense-wrong count |
| 4. Query-tail saturation controller | Implemented previously as v6; evaluated negative as main recipe | `loc_gs/stdloc_native/solver_coverage_coreset.py`, `docs/lsf_v6_saturated_coverage_status_20260525.md` | Keep as guardrail, not main route |
| 5. LSF detector target refinement | Implemented as offline heatmap target builder | `loc_gs/stdloc_native/detector_target_refinement.py`, `tests/test_detector_target_refinement.py` | Fine-tune detector only after sparse descriptor-fusion signal appears |
| 6. Negative support memory | Implemented as pairwise hard-negative conflict graph | `loc_gs/stdloc_native/negative_support_memory.py`, `tests/test_negative_support_memory.py` | Feed graph into future bank/resampling objectives |
| 7. Bank-level large-edit resampling | Implemented as safety policy for 10-30% replacement | `loc_gs/stdloc_native/bank_level_resampling.py`, `tests/test_bank_level_resampling.py` | Allow only with saturation and conflict graph controls |

## Test Evidence

The first RED run failed because all six modules were missing:

```text
6 errors during collection
ModuleNotFoundError for support_banks, solver_weighted_feature_fusion,
dense_match_verifier, detector_target_refinement, negative_support_memory,
and bank_level_resampling
```

After implementation:

```text
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_support_banks.py \
  tests/test_solver_weighted_feature_fusion.py \
  tests/test_build_solver_weighted_feature_fusion_cli.py \
  tests/test_dense_match_verifier.py \
  tests/test_detector_target_refinement.py \
  tests/test_negative_support_memory.py \
  tests/test_bank_level_resampling.py

7 method-component tests passed in targeted verification.
```

## Route Adjustment

The priority order is now:

1. **Solver-weighted landmark feature fusion**
   - Most likely to produce cm-level improvement because it changes descriptor
     matching, not just sampled IDs.
   - First experiment should be sparse-only. If sparse R@10/R@5 does not move,
     do not spend dense evaluation budget.

2. **Dense match verifier**
   - Directly targets the observed failure mode where dense refinement erases
     sparse gains.
   - First experiment should measure transition counts, not only median error.

3. **Negative support memory + support banks**
   - Use as guardrails for larger bank-level edits.
   - Do not promote large-edit maps unless both controls are active and
     train-dev hard scenes are non-regressing.

4. **Detector target refinement**
   - Defer until descriptor fusion or verifier produces a real sparse/dense
     signal; otherwise detector fine-tuning expands the search space too early.

## Paper Safety

- These modules do not use test feedback or test metrics.
- Query-conditioned support bank routing is an offline artifact/router, not a
  result-driven branch selector.
- Dense verifier is designed as a correspondence filter before dense refinement,
  not as a per-query oracle fallback.
- Large-edit policy explicitly rejects uncontrolled 10-30% replacement.

## Solver-Weighted Fusion Diagnostic

The first feature-fusion artifacts were generated from existing train/self-map
pair caches:

```text
output/review1_method_upgrades_20260525/solver_weighted_feature_fusion/ShopFacade/selected_landmark_descriptors.pt
  positive_pair_count: 7846
  source_split_name: train
  feedback_bank_split_name: selfmap_train_rendered

output/review1_method_upgrades_20260525/solver_weighted_feature_fusion/OldHospital/selected_landmark_descriptors.pt
  positive_pair_count: 8606
  source_split_name: train
  feedback_bank_split_name: selfmap_train_rendered
```

An initial materialization attempted to apply these artifacts to
`output/stdloc/map_cambridge_spgs_trainonly_20260524`. This was invalid:
the pair cache base descriptors are tied to `output/stdloc/map_cambridge_spgs`
and are nearly orthogonal to the train-only map rows at the same Gaussian ids
(ShopFacade mean cosine `0.0039`, OldHospital mean cosine `0.0074`).

The implementation now stores `base_descriptors` in every feature-fusion
artifact and `build_solver_weighted_feature_map` rejects incompatible source
maps by default. Targeted guard tests pass:

```text
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_solver_weighted_feature_fusion.py \
  tests/test_build_solver_weighted_feature_fusion_cli.py \
  tests/test_build_solver_weighted_feature_map_cli.py -q

6 passed
```

A strict, source-compatible diagnostic was then run against the original
8192-landmark map. This is **not paper-facing** because the source map is not
the clean train-only map, but it isolates the fusion mechanism without the id
mismatch:

| Scene | Stage | Median TE delta | R@5cm/5deg delta | Decision |
| --- | ---: | ---: | ---: | --- |
| ShopFacade q80 | sparse | `+0.3861cm` | `-0.0125` | negative |
| ShopFacade q80 | dense | `+0.3521cm` | `-0.0375` | negative |
| OldHospital q80 | sparse | `-0.4763cm` | `+0.0000` | weak local positive |
| OldHospital q80 | dense | `+0.3373cm` | `+0.0000` | negative |

Artifacts:

```text
output/review1_method_upgrades_20260525/solver_weighted_feature_fusion_strict_compatible/{ShopFacade,OldHospital}
output/stdloc/map_cambridge_spgs_solver_weighted_fusion_strict_compatible_20260525/{ShopFacade,OldHospital}
output/stdloc_native/solver_weighted_fusion_strict_compatible_20260525/{baseline,selected}/q80/{ShopFacade,OldHospital}
```

Conclusion: solver-weighted feature fusion should not be scaled as-is. Even
with source compatibility and conservative trust-region guards, it is not a
stable mechanism.

## Dense LSF Fixed Locability Diagnostic

A paper-safe export path was added for dense transition supervision that does
not modify `third_party/stdloc`, does not change `sampled_idx`, and does not
introduce per-query branch selection:

```text
loc_gs/scripts/export_dense_lsf_locability_map.py
tests/test_export_dense_lsf_locability_map_cli.py
```

The dense target distiller now preserves upstream feedback-bank audit metadata
so the exported map carries a real split audit:

```text
loc_gs/dense_support/distill_dense_lsf.py
tests/test_dense_lsf_distillation.py
```

Targeted verification:

```text
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_dense_lsf_distillation.py \
  tests/test_export_dense_lsf_locability_map_cli.py -q

4 passed
```

Train-only dense LSF targets were distilled from audited
`feedback_bank_v2` artifacts:

| Scene | Observed Gaussians | Selected Targets | Feedback Audit |
| --- | ---: | ---: | --- |
| ShopFacade | `12617` | `12350` | passed |
| OldHospital | `10427` | `10245` | passed |

The exported fixed-locability maps have complete artifact bundles:

```text
output/stdloc/map_cambridge_spgs_trainonly_dense_lsf_locability_20260525/{ShopFacade,OldHospital}
```

q80 diagnostic, using the same native STDLoc poselib/SPGS-prior dense cfg as
the v5 baseline:

| Scene | Dense Median TE Delta | R@5cm/5deg Delta | Decision |
| --- | ---: | ---: | --- |
| ShopFacade | `-0.0066cm` | `+0.0000` | neutral |
| OldHospital | `-0.0143cm` | `+0.0125` | weak positive |

Artifacts:

```text
output/dense_lsf_native_trainonly_20260525/{ShopFacade,OldHospital}
output/stdloc_native/dense_lsf_locability_20260525/selected/q80/{ShopFacade,OldHospital}
```

Conclusion: fixed dense locability is safe and slightly positive on this q80
pair, but the effect size is still far below a SOTA-level method claim. The
next route is larger-budget selection with saturation and hard-negative
controls, not further dense-prior tuning.

## Next Experiment

The next concrete experiment should be:

```text
Stage 1: export saturated-coverage large-edit train-only maps
Stage 2: add hard-negative/conflict guard if an artifact is available
Stage 3: evaluate q80/q160 on ShopFacade and OldHospital first
Stage 4: only expand to five scenes if macro median improves by at least 0.5cm
```

This targets the current bottleneck: the safe fixed priors are now measurable
but too weak; a larger support-set change is required, with explicit controls
against v5/v6 non-monotonic regressions.
