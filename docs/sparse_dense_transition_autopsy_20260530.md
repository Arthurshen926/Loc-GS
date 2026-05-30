# Sparse-Dense Transition Autopsy - 2026-05-30

## Scope

This note audits the current sparse-conditioned dense transition stack:

- sparse landmark conditioned dense preflight / render control
- sparse-ray Gaussian gating
- guided render pose candidates
- patch dense matching
- anchor-conditioned dense refinement acceptance

All official Cambridge test examples mentioned here are diagnostic-only. They
must not be used for training, selector calibration, threshold tuning, or
paper-facing route selection.

## Confirmed Implementation Issue

`assess_anchor_conditioned_update(None)` treated missing local-refinement
diagnostics as `accept_anchor_conditioned_update`.

This was unsafe because a path that failed to write before/after reprojection
diagnostics could still accept the anchor-conditioned dense update. The fixed
behavior is:

- decision: `unknown_noop_keep_reference`
- `accept_update: false`
- failed check: `missing_reference_refinement_diagnostics`

This keeps the reference pose unless the no-GT local objective is actually
available and passes the acceptance checks.

## Local Invariants Added

Focused tests now cover three transition invariants:

- ACPD missing diagnostics are no-op/unknown, not accepted.
- ACPD sparse-anchor projection uses explicit `(width, height)` image-size
  ordering on a non-square image.
- SLCDP feature sampling can use explicit `ImageGeometry` to map image/depth
  coordinates into downsampled feature-map coordinates.
- `refine_pose_with_reference_prior` consumes `match_weights`; increasing
  sparse-anchor weights changes the refined pose toward the anchor-consistent
  solution.
- Soft-SDCG diagnostic transition uses a continuous dense reliability alpha:
  bad dense updates keep sparse, reliable dense updates accept dense, and
  intermediate preflight reliability gives an interpolated pose.
- Patch residual group weighting downweights local-only ambiguous patch groups
  and scales dense patch residual weights without changing sparse-anchor
  weights.

These tests do not prove the full pipeline is correct, but they remove one
confirmed logic bug and lock two high-risk assumptions.

## Current Diagnostic Interpretation

The hard cases do not reduce to one failure mode.

- KingsCollege #239: semantic-mask map improves the raw dense failure compared
  with the earlier map, but dense render/feature reliability can still be too
  weak. The guarded fused path correctly keeps sparse when dense repair quality
  is low. Patch matching did not improve this case.
- KingsCollege #33: patch dense has a positive diagnostic signal in an older
  occlusion-focused run, but the current fused semantic preset rejects the patch
  update and falls back to base dense. This suggests patch matching is not yet
  a reliable main-path component.
- Guided pose and gating can improve preflight/render diagnostics while still
  producing a dense pose worse than sparse. A better render view is therefore
  not equivalent to a better localization update.

## Design Conclusion

The present implementation still behaves like a concatenation of corrections:

```text
guided pose + gating + patch dense + sparse anchors + hard acceptance
```

The target design should be a single sparse-anchored dense transition
controller:

```text
dense reliability alpha in [0, 1]
patch residual weights r_p in [0, 1]
sparse-anchor trust strength mu >= 0
```

Dense refinement should become a reliability-weighted update around the sparse
pose, not an unconditional replacement and not a chain of independent hard
switches. A first diagnostic API was added as
`select_soft_sparse_conditioned_dense_transition`; it is exposed only through
explicit diagnostic flags and is not wired into the main evaluator path.

## Executable Autopsy Tool

Added:

```text
python -m loc_gs.scripts.build_sparse_dense_transition_autopsy
```

It reads one or more `label=path` variants containing `pgsh_summary.json`
artifacts and writes:

- `autopsy_summary.json`
- `variant_cases.csv`
- `report.md`
- standard artifact audit files

The tool rejects `split_name=test` unless `--allow_test_diagnostic` is passed,
and even then marks the artifact diagnostic-only.

Diagnostic smoke artifact:

```text
output/diagnostics/sparse_dense_transition_autopsy_20260530_diag
```

This smoke uses existing official-test visualization artifacts only to verify
report generation. It is not a tuning or route-selection result.

## Required Next Autopsy

Run the following ablation matrix on a train/self-map hard subset plus a normal
subset before changing official test behavior:

| Variant | Purpose |
| --- | --- |
| native sparse+dense | reference |
| sparse-only | measure dense harm ceiling |
| diagnostics-only preflight | calibration without behavior change |
| guided only | render-pose effect |
| gating only | map-side artifact effect |
| patch only | query-side occlusion effect |
| sparse anchors only | anchor residual effect |
| acceptance only | step acceptance effect |
| guided + gating | map-side combined effect |
| patch + anchors | query-side combined effect |
| full fused module | current stack |

Report dense-worsened count, sparse-good-to-dense-bad count, Reg20/Reg50, P90,
P95, false dense accept, false dense reject, runtime, and split audit.
