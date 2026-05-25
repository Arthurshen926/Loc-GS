# LSF-Loc Canonical q80 Status, 2026-05-23

## Scope

- Dataset scope: Cambridge five scenes, `eval_split=train`, `max_test_cameras=80`.
- No Cambridge test split was used for training, tuning, feedback, or reporting.
- Evaluator path: native STDLoc wrapper, OpenCV PROSAC/MAGSAC PnP, STDLoc-style dense refinement.
- Baseline root: `output/stdloc_native/clean16384_q80_allscene_20260523_rerun1/baseline`.

## Implementation Status

Implemented in this round:

- Canonical `LocalizationSupportField` artifact with:
  - `support_score`
  - `support_for_selection`
  - `inlier_consensus`
  - `hard_query_support`
  - `pose_information_gain`
  - `ambiguity_risk`
  - `hard_negative_risk`
  - `dense_worsen_risk`
  - `visibility_stability`
- Proposal/export support for canonical LSF fields and optional set-level solver utility.
- Artifact audit bundles for support fields, proposals, and exported maps:
  `manifest.json`, `command.txt`, `metrics_summary.json`, `split_audit.json`, `git_status.txt`, `artifact_audit.json`.
- Fixed dense locability export path and config hook.

Main code paths:

- `loc_gs/diagnostics/localization_support_field.py`
- `loc_gs/scripts/build_localization_support_field.py`
- `loc_gs/diagnostics/lsf_v2_native_proposal.py`
- `loc_gs/scripts/export_lsf_solver_aware_map.py`
- `loc_gs/reporting/artifact_audit.py`

## q80 Results

All results below are train-dev diagnostics, not Cambridge test claims.

| Recipe | Dense macro TE delta | Dense macro RE delta | Dense R5 delta | Dense R2 delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| v4 canonical solver+dense | +0.171 cm | -0.0003 deg | +0.25 pp | +0.25 pp | fail: OldHospital regresses |
| v5 canonical without dense prior | +0.008 cm | -0.0053 deg | -0.25 pp | +0.25 pp | neutral/fail |
| **v6 support-only canonical, 32 edits** | **-0.427 cm** | **-0.0082 deg** | -0.50 pp | +0.25 pp | current best |
| v7 support-only, 16 edits | -0.169 cm | -0.0051 deg | -0.25 pp | 0.00 pp | weaker, OldHospital regresses |
| v8 support-only, 24 edits | -0.159 cm | -0.0053 deg | -0.75 pp | 0.00 pp | weaker, OldHospital regresses |
| v9 support-only, no sparse risk | worse than v6 | worse than v6 | mixed | mixed | rejected |
| v10 support-only, protected native source | worse than v6 | worse than v6 | mixed | mixed | rejected |
| v11 balanced solver-admissible scan128 | +0.165 cm vs baseline | +0.0041 deg | -0.25 pp | -0.25 pp | rejected globally; fixes one GreatCourt outlier |

Current best fixed recipe:

```text
v6 = support-only canonical LSF
native_weight=0.75
support_weight=0.20
support_score_mode=rank_observed
max_edits=32
no solver utility weight
no solver-admissibility sparse filter
no dense locability prior
```

v6 per-scene dense deltas:

| Scene | TE delta | RE delta | R5 delta | R2 delta |
| --- | ---: | ---: | ---: | ---: |
| GreatCourt | +0.111 cm | +0.0007 deg | -1.25 pp | +1.25 pp |
| KingsCollege | -0.356 cm | +0.0009 deg | 0.00 pp | 0.00 pp |
| OldHospital | -1.258 cm | -0.0058 deg | -1.25 pp | -1.25 pp |
| ShopFacade | -0.028 cm | -0.0053 deg | 0.00 pp | 0.00 pp |
| StMarysChurch | -0.604 cm | -0.0314 deg | 0.00 pp | +1.25 pp |

## Interpretation

The current defensible claim is precision-primary:

> self-localization feedback can guide sparse landmark selection in a fixed STDLoc-compatible map, improving q80 train-dev dense pose precision on macro average and on four of five scenes.

The current non-claims are equally important:

- v6 is not a Cambridge test result.
- v6 is not an all-scene strict recall improvement.
- Dense locability prior and solver-admissibility constraints are implemented but not validated as the main method; in v4/v5 they hurt OldHospital.
- GreatCourt remains a failure boundary for the current support-only recipe, so the paper-safe fixed recipe keeps its native clean map there.
- v11 solver-admissibility is a diagnostic repair, not the main recipe: balanced query feedback fixed the known GreatCourt q42 outlier, but full-scene q80 macro TE regressed.

## Precision-Primary Scene-Level Gate

The current fixed train-dev gate is scene-level reconstruction-time acceptance,
not per-query branch selection:

- Candidate order: v6 support-only first, then v11 solver-admissible as fallback.
- GreatCourt is neutral and may keep the native clean map.
- KingsCollege, OldHospital, ShopFacade, and StMarysChurch must improve dense median TE.
- R5/R2 drops are warnings under the precision-primary policy, not reject reasons.

q80 gate artifact:

```text
output/lsf_scene_level_gate_20260523/precision_primary_q80_report.json
```

q80 selected recipe:

| Scene | Selected map | Dense TE delta | R5 delta | R2 delta | Gate note |
| --- | --- | ---: | ---: | ---: | --- |
| GreatCourt | native clean | +0.000 cm | +0.00 pp | +0.00 pp | neutral |
| KingsCollege | v6 | -0.356 cm | +0.00 pp | +0.00 pp | pass |
| OldHospital | v6 | -1.258 cm | -1.25 pp | -1.25 pp | recall warning |
| ShopFacade | v6 | -0.028 cm | +0.00 pp | +0.00 pp | pass |
| StMarysChurch | v6 | -0.604 cm | +0.00 pp | +1.25 pp | pass |
| Macro | scene-level fixed | -0.449 cm | -0.25 pp | +0.00 pp | pass with R5 warning |

q160 validation was run after fixing the gate report audit fields, still on
`eval_split=train` and without Cambridge test data:

```text
output/lsf_scene_level_gate_20260523/precision_primary_q160_report.json
output/stdloc_native/clean16384_q160_precision_primary_gate_20260523_rerun1
```

Frozen recipe manifest:

```text
output/lsf_scene_level_gate_20260523/frozen_recipe_manifest.json
```

Superseded safety status, 2026-05-24: after adding map-camera split audit, the
manifest now reports `paper_facing_ready=false`. The metric gates below remain
diagnostic, but KingsCollege, OldHospital, and ShopFacade selected maps inherit
`cameras.json` entries from official Cambridge test images, so they are not
paper-facing until train-only maps are rebuilt and re-evaluated.

q160 selected recipe:

| Scene | Baseline TE | Selected TE | Dense TE delta | R10 delta | R5 delta | R2 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 5.965 cm | 5.965 cm | +0.000 cm | +0.00 pp | +0.00 pp | +0.00 pp |
| KingsCollege | 12.152 cm | 12.127 cm | -0.025 cm | -0.63 pp | +0.00 pp | +0.00 pp |
| OldHospital | 6.709 cm | 6.113 cm | -0.595 cm | +4.38 pp | +2.50 pp | -2.50 pp |
| ShopFacade | 1.804 cm | 1.738 cm | -0.066 cm | +0.00 pp | +0.00 pp | +0.00 pp |
| StMarysChurch | 3.874 cm | 3.709 cm | -0.165 cm | +0.63 pp | +1.25 pp | +3.13 pp |
| Macro | 6.101 cm | 5.930 cm | -0.170 cm | +0.88 pp | +0.75 pp | +0.13 pp |

The q160 gate passes. The only warning is OldHospital R2, while median TE and
R5 improve there. This supports the precision-primary framing more strongly
than the strict all-recall framing.

## ChatGPT-new.md Completion

- Module 1 Solver-Consensus Support Estimation: implemented as canonical support field with full support-vector fields and audit.
- Module 2 Solvability-Aware Landmark Sampling: implemented as STDLoc-compatible same-budget map export. The support-only variant is currently effective; set-level solver utility/CVaR machinery exists but is not the validated main recipe.
- Module 3 Support-Consistent Sparse-to-Dense Verification: interface implemented through dense locability export/config. Current evidence says dense support should remain residual/dense-verification teacher, not a standalone sparse selector.
- Paper-safety/audit: implemented for new artifacts and native eval outputs.
- STDLoc/ULFLoc alignment: STDLoc path is maintained; ULFLoc is treated as paper-reported reference, not reproduced locally.

## Next Gate

Before any Cambridge test/full paper-facing table:

- Rebuild or locate train-only native clean maps for KingsCollege, OldHospital,
  and ShopFacade; then rebuild LSF selected maps from those train-only sources.
- Re-run q80/q160 train-dev and Cambridge test only after every run has
  `map_cameras_vs_cambridge_test_split` audit passing.
- Investigate GreatCourt and OldHospital recall drops without per-query branch selection.
- Rework dense support as residual filtering, not as global sparse candidate risk or global dense prior.

Active remediation started 2026-05-24:

```text
output/stdloc/map_cambridge_spgs_trainonly_20260524/{KingsCollege,OldHospital,ShopFacade}
output/logs/stdloc_trainonly_20260524/
```

These STDLoc maps are being rebuilt with `--train_only_cameras`; initial
`cameras.json` audit shows zero test overlap and zero outside-train cameras for
all three scenes. A background monitor will run q80/q160 train-dev and test
baseline evals after training completes.

## Frozen Test Evaluation, 2026-05-24

Held-out Cambridge test was evaluated after the recipe was frozen. These results
must not be used for recipe reselection or tuning. After the 2026-05-24
map-camera audit, they also must not be used as paper-facing results.

Artifacts:

```text
output/stdloc_native/frozen_recipe_test_20260524_rerun1
output/lsf_scene_level_gate_20260523/frozen_recipe_test_comparison_20260524.json
output/lsf_scene_level_gate_20260523/map_camera_split_audit_20260524.json
```

Frozen selected test recipe:

- GreatCourt: native clean map, same as baseline.
- KingsCollege, OldHospital, ShopFacade, StMarysChurch: v6 support-only maps.

Dense test results:

| Scene | Baseline TE | Selected TE | TE delta | RE delta | R10 delta | R5 delta | R2 delta | Failure delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 11.800 cm | 11.800 cm | +0.000 cm | +0.0000 deg | +0.00 pp | +0.00 pp | +0.00 pp | +0 |
| KingsCollege | 17.964 cm | 17.964 cm | -0.000 cm | -0.0015 deg | +0.87 pp | +0.00 pp | -0.29 pp | +0 |
| OldHospital | 13.416 cm | 12.073 cm | -1.344 cm | +0.0153 deg | +2.75 pp | -1.65 pp | +0.55 pp | +0 |
| ShopFacade | 2.758 cm | 2.808 cm | +0.050 cm | -0.0047 deg | +0.97 pp | +0.97 pp | +0.00 pp | -1 |
| StMarysChurch | 3.887 cm | 3.861 cm | -0.026 cm | -0.0009 deg | +0.57 pp | +1.13 pp | +1.13 pp | +0 |
| Macro | 9.965 cm | 9.701 cm | -0.264 cm | +0.0017 deg | +1.03 pp | +0.09 pp | +0.28 pp | -1 |

Interpretation:

- Frozen test is macro-positive on dense translation and recall, but this is
  diagnostic only because three scene maps include official test cameras.
- OldHospital transfers well on translation precision and R10, but has an R5 drop.
- ShopFacade is the main precision boundary on test: translation worsens by
  0.050 cm while rotation, recall, and failure count improve.
- GreatCourt remains neutral because the frozen recipe deliberately keeps the
  native map there.
- Map-camera split audit status: GreatCourt and StMarysChurch pass; KingsCollege
  fails with 343 official test camera ids, OldHospital fails with 182, and
  ShopFacade fails with 103. The v6 selected maps previously passed upstream
  feedback-bank audits, but that audit did not cover source-map `cameras.json`
  leakage.
- The current working tree has dirty files under `third_party/stdloc`
  (`stdloc.py`, `utils/pose_utils.py`, and one vendored test). Treat the
  numbers as diagnostic validation, not final paper-facing results, until
  train-only maps pass the new audit and native STDLoc parity is re-established
  or those changes are isolated with a parity report.
