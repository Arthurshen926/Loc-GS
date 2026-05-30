# Clean Render Phase2/Phase3 Status 20260530

This note records the diagnostic implementation for Phase2 and Phase3 of the
Sparse-Anchored Clean Dense Rendering track.

These modules do not modify the STDLoc evaluator and do not select final poses.
They are intended to make the sparse-to-dense transition testable before any
future integration into dense refinement.

## Phase2: Patch Dense Candidate Generator

Implemented:

- `loc_gs/dense_support/patch_dense_candidates.py`
- `tests/test_patch_dense_candidates.py`

Main entry point:

```python
generate_patch_dense_candidates(
    query_features,
    rendered_features,
    rendered_depth,
    sparse_anchors,
    policy=PatchDenseCandidatePolicy(),
)
```

Outputs:

- dense candidate arrays: `xy`, `render_xy`, `depth`, `weights`,
  `match_scores`, `patch_ids`
- patch diagnostics:
  - `local_match_quality`
  - `anchor_consistency`
  - `off_patch_consistency`
  - `patch_pose_cluster_consistency`
  - `depth_spread_score`
  - `ambiguity_score`
  - `patch_weight`

Design constraints:

- `diagnostic_only=true`
- `does_not_select_final_pose=true`
- `uses_gt=false`
- no branch/oracle selection

Validation:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_patch_dense_candidates.py -q
```

Result:

```text
3 passed
```

The tests verify:

- candidates and patch scores are emitted without pose selection;
- repeated local features increase ambiguity and reduce patch weight;
- anchor/off-patch consistency follows sparse-anchor offset.

## Phase3: Sparse Anchor Residual Diagnostics

Implemented:

- `loc_gs/dense_support/sparse_anchor_residual.py`
- `tests/test_sparse_anchor_residual.py`

Main entry points:

```python
compute_sparse_anchor_residual_group(
    sparse_anchors,
    pose_w2c,
    intrinsic,
    image_size,
    policy=SparseAnchorResidualPolicy(),
)
```

```python
summarize_sparse_anchor_residual_modes(
    dense_errors_px,
    anchor_errors_px,
    policy=SparseAnchorResidualPolicy(),
)
```

The first function computes a robust sparse-anchor reprojection residual group
under a provided pose. The second compares:

- `no_anchor`
- `concat_matches`
- `separate_residual_group`

This explicitly tests the earlier concern that simply concatenating sparse
anchors into many dense matches can dilute sparse-anchor influence, while a
separate residual group preserves an auditable contribution.

Design constraints:

- `diagnostic_only=true`
- `does_not_select_final_pose=true`
- `uses_gt=false`
- no dense optimizer modification

Validation:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_sparse_anchor_residual.py -q
```

Result:

```text
3 passed
```

The tests verify:

- sparse anchors are projected and invalid anchors are excluded;
- robust residual loss is reported;
- separate anchor residual group avoids dense-match-count dilution;
- no mode selects a final pose.

## Combined Verification

The current Phase2/Phase3 implementation was also verified together with the
existing clean-render, SLCDP, patch-guided sparse, ACPD, visualization, and
artifact CLI tests:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_patch_dense_candidates.py \
  tests/test_sparse_anchor_residual.py \
  tests/test_clean_render_generator.py \
  tests/test_clean_render_phase0_benchmark.py \
  tests/test_clean_render_phase1_validation.py \
  tests/test_sparse_conditioned_dense_preflight.py \
  tests/test_patch_guided_sparse.py \
  tests/test_anchor_conditioned_patch_dense.py \
  tests/test_match_visualization.py \
  tests/test_method_artifact_clis.py \
  -q
```

## Current Limit

Phase2 and Phase3 are implemented as standalone diagnostic modules. They have
not yet been wired into a Cambridge full split dense refinement run. That should
remain a separate integration step after the diagnostics are used to define a
single no-GT transition safety score.
