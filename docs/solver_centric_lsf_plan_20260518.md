# Solver-Centric LSF Plan, 2026-05-18

## Main Problem

Loc-GS is now scoped as a solver-centric localization support field (LSF), not
as descriptor replacement and not as test-time branch selection.

The core problem is:

```text
3DGS localization feature selection should optimize PnP pose solvability,
hard-query tail risk, and dense residual reliability, not unary matchability
or support-count preservation alone.
```

## Paper-Facing Pipeline

The deployed path stays STDLoc-compatible and single-path:

```text
native STDLoc descriptor map
  -> self-localization feedback bank
  -> solver-aware sparse support field
  -> sampled_idx.pkl + sampled_scores.pkl + locability
  -> descriptor matching + OpenCV PROSAC/RANSAC PnP
  -> STDLoc-style dense refinement
```

No test-time branch selection, no per-query multi-path localization, and no
official test-query pose is used for training, calibration, self-map labels, or
candidate selection.

## Representation

Sparse LSF:

- per-Gaussian / per-landmark support for PnP;
- query-family support diagnostics from self-localization episodes;
- safe-core source landmarks and admissible replacements.

Dense LSF, later stage:

- rendered per-ray/per-pixel reliability mask;
- dense correspondence/residual utility;
- dense refinement filtering only after sparse Stage A passes.

## Stage Gates

No full Cambridge or official test promotion until all of these pass:

```text
1. train-q80 hard-scene pass
2. shifted train-s3/q75 pass
3. split audit pass
4. query-level lost/recovered audit pass
5. at least one solver-level diagnostic improves
```

Accuracy candidate pass line:

```text
macro median_te <= native - 0.15 cm
macro R5 >= native + 0.75 pp
macro R2 >= native + 0.25 pp
no scene R5 loss > 1.0 pp
no scene median_te regression > 0.4 cm
OldHospital and StMarysChurch not worse than native on both median and R5
```

Mechanism candidate pass line:

```text
hard-query viable tuple quality improves
hard-query logdet(H) does not drop
ambiguity risk does not increase
support count is not used as the only positive evidence
```

## Immediate Implementation

Added first diagnostic layer:

```text
loc_gs/diagnostics/pose_information.py
loc_gs/diagnostics/ambiguity_metrics.py
loc_gs/diagnostics/solver_tuple_bank.py
loc_gs/scripts/build_solver_tuple_bank.py
loc_gs/scripts/compare_solver_tuple_banks.py
tests/test_pose_information_metrics.py
tests/test_ambiguity_metrics.py
tests/test_solver_tuple_bank.py
tests/test_solver_tuple_bank_cli.py
```

The current audited pair cache lacks per-row `query_id`, so the first hard-scene
diagnostic uses `synthetic_chunks_512`. This is not paper-facing; the next
feedback-cache rebuild must store real query/image ids and preferably per-query
PnP/dense outcomes.

## First Diagnostic Result

Supportguardall-ret90 was rejected by Stage A accuracy, even though support
audit showed zero query support losses. The new solver diagnostic compares
native12288 against supportguardall-ret90 on hard scenes:

| Scene | Support delta | Viable mass delta | Mean logdet(H) delta | Query grouping |
| --- | ---: | ---: | ---: | --- |
| OldHospital | +58 | +5.092 | -0.584 | synthetic_chunks_512 |
| StMarysChurch | +81 | +5.057 | -2.081 | synthetic_chunks_512 |

This does not yet prove tuple mass explains the failure. It does show a sharper
mechanism: support-preserving edits can add more positive cached matches while
lowering the geometric conditioning of sampled minimal sets. Therefore the next
resampling objective must constrain replacement by logdet(H), hard-query CVaR,
and ambiguity, not only support counts.

Artifacts:

```text
output/solver_diagnostics/20260518_supportguardall_ret90_quick/OldHospital/solver_tuple_compare.json
output/solver_diagnostics/20260518_supportguardall_ret90_quick/StMarysChurch/solver_tuple_compare.json
docs/solver_selection_failure_oldhospital_20260518.md
docs/solver_selection_failure_stmaryschurch_20260518.md
```

## Next Code Step

Implement solver-aware replacement admissibility:

```text
loc_gs/stdloc_native/solver_admissibility.py
loc_gs/stdloc_native/solver_aware_resampling.py
loc_gs/scripts/export_solver_aware_map.py
tests/test_solver_admissibility.py
tests/test_solver_aware_resampling.py
```

Initial rule:

```text
start from native sampled set S0
build safe core from high-logdet/high-support hard-query tuples
only replace source landmark j with candidate i if:
  hard-query support does not drop,
  mean/worst logdet(H) does not drop beyond tolerance,
  ambiguity risk does not increase,
  candidate has positive evidence and low hard-negative risk,
  same-budget constraint is preserved
```

