# APD-Dense Phase0-Phase4 Status - 2026-05-30

This note records the first-principles reset of the dense-stage work.

## Goal

Replace only the STDLoc dense stage with a minimal module:

```text
STDLoc sparse pose + sparse PnP inlier anchors
-> clean/global/patch dense correspondence candidates
-> sparse-anchor verified robust dense refinement
-> final pose
```

The module does not change the STDLoc sparse detector, sparse matching, sparse
PnP, Cambridge splits, ground truth, or vendored evaluator.

## Core Judgements

APD-Dense keeps only three dense-candidate judgements:

- anchor consistency: candidate 3D points should remain compatible with the
  sparse pose/anchors;
- candidate geometry: candidates should cover image regions and have camera
  depth diversity under the sparse pose;
- query-render match quality: candidate match scores should be high.

The dense candidate weight is the geometric mean of these three signals.

## Implemented Phases

### Phase0: APD API

Implemented in `loc_gs/dense_support/apd_dense.py`.

Main entry point:

```python
run_anchor_patch_dense_refinement(...)
```

Output includes `final_pose`, candidate source counts, anchor residual count,
refinement success, and APD diagnostics.

### Phase1: Clean Render Candidate

Existing clean-render candidates can be passed as `clean_render_dense_candidates`.
APD treats clean render as one dense-candidate source only; the clean render pose
is not accepted as a final localization pose.

### Phase2: Patch Dense Candidate

Implemented in `loc_gs/dense_support/patch_dense_candidates.py`.

`generate_patch_dense_candidates(...)` now optionally lifts rendered depth to
world-space `p3d` when given `render_pose_w2c` and `intrinsic`, so patch
candidates can enter the same APD correspondence pool as global dense matches.

### Phase3: Anchor-Verified Robust Refinement

Implemented in `refine_pose_with_sparse_anchors_and_dense_candidates(...)`.

Dense candidates and sparse anchors are group-normalized before robust
refinement, so sparse anchors are not diluted by dense match count and do not
need large ad-hoc per-match weights.

### Phase4: Combination

`run_anchor_patch_dense_refinement(...)` combines:

- native/global dense candidates;
- clean-render dense candidates;
- patch dense candidates;
- sparse anchor residuals.

The hard accept/reject and soft pose-interpolation controllers are not part of
this APD path.

## Diagnostic CLI Integration

`loc_gs/scripts/visualize_stdloc_hard_matches.py` has APD flags:

```bash
--apd_dense
--apd_include_patch_candidates
--apd_dense_group_weight
--apd_anchor_group_weight
--apd_max_translation_delta_m
--apd_max_rotation_delta_deg
```

When `--apd_dense` is enabled, legacy sparse-conditioned transition controllers
are disabled in the resolved CLI options to avoid stacking methods.

## Verification

Fresh checks:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_apd_dense.py -q
```

Result:

```text
4 passed
```

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_visualize_stdloc_apd_options.py -q
```

Result:

```text
1 passed
```

Earlier combined module check:

```text
18 passed
```

## Current Limit

This is an implementation and diagnostic-entry milestone, not a paper-facing
Cambridge result. `locgsctl list-scenes` currently reports default sampled maps
as 8192 while the expected native count is 16384, so full metric claims must wait
for a paper-safe train/self-map evaluation setup.
