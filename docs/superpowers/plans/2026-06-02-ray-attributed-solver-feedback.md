# Ray-Attributed Solver Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 of cross-view/ray-attributed solver feedback: convert matched ray observations with top-K Gaussian contributors into audited Gaussian-level support, risk, and artifact tensors.

**Architecture:** Add a compact feedback core under `loc_gs/feedback` and a CLI under `loc_gs/scripts`. Phase 1 consumes JSON/JSONL observation artifacts; later phases will create those observations from ULF/3DGS cross-view matching and rendered-view matching.

**Tech Stack:** Python, PyTorch tensors, JSONL artifact IO, pytest synthetic tests, Loc-GS audit conventions.

---

### Task 1: Core Soft Ray Attribution

**Files:**
- Create: `loc_gs/feedback/ray_attributed_solver_feedback.py`
- Test: `tests/test_ray_attributed_solver_feedback.py`

- [ ] **Step 1: Write failing tests**

Test that a PnP-inlier match with contributors `(0, 0.7)` and `(1, 0.3)` assigns stronger support to Gaussian 0 while still crediting Gaussian 1. Test that non-inlier matches do not become strong hard negatives by default.

- [ ] **Step 2: Run tests and confirm failure**

Run: `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_ray_attributed_solver_feedback.py -q`

- [ ] **Step 3: Implement core functions**

Implement:
- `normalize_contributors`
- `soft_point_from_contributors`
- `artifact_score_from_ray`
- `accumulate_ray_feedback`

- [ ] **Step 4: Run tests and confirm pass**

Run the same pytest command and confirm all tests pass.

### Task 2: Artifact CLI

**Files:**
- Create: `loc_gs/scripts/build_ray_attributed_solver_feedback.py`
- Test: `tests/test_build_ray_attributed_solver_feedback.py`

- [ ] **Step 1: Write failing CLI tests**

Test that the CLI reads JSONL observations, rejects `split_name=test`, writes `ray_solver_feedback.pt`, `manifest.json`, `metrics_summary.json`, `split_audit.json`, `command.txt`, and `git_status.txt`.

- [ ] **Step 2: Run tests and confirm failure**

Run: `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_build_ray_attributed_solver_feedback.py -q`

- [ ] **Step 3: Implement CLI**

Use the core accumulator, write tensors plus metadata, and include paper-safety audit fields.

- [ ] **Step 4: Run CLI tests and core tests**

Run both new test files.

### Task 3: Documentation And Smoke Verification

**Files:**
- Create: `docs/ray_attributed_solver_feedback_phase1_20260602.md`

- [ ] **Step 1: Document data contract**

Document the observation JSONL schema, output tensor names, and constraints: no test split, non-inliers are weak evidence, rendered artifacts should be gated or rejected in later phases.

- [ ] **Step 2: Run final verification**

Run:
- `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_ray_attributed_solver_feedback.py tests/test_build_ray_attributed_solver_feedback.py -q`
- `/root/miniconda3/envs/cybersim_agent/bin/python -m py_compile loc_gs/feedback/ray_attributed_solver_feedback.py loc_gs/scripts/build_ray_attributed_solver_feedback.py`
- `git diff --check -- loc_gs/feedback/ray_attributed_solver_feedback.py loc_gs/scripts/build_ray_attributed_solver_feedback.py tests/test_ray_attributed_solver_feedback.py tests/test_build_ray_attributed_solver_feedback.py docs/ray_attributed_solver_feedback_phase1_20260602.md`

### Later Phases

- Phase 2: Build real-image cross-view observation extractor from train-fold pseudo-query/source pairs.
- Phase 3: Add rendered-view observation extractor with guided-pose/gating artifact control.
- Phase 4: Feed ray feedback into ULF overcomplete landmark selection and solver-weighted feature fusion.

### Phase 2A: Precomputed Cross-View Match To Ray Observation Extractor

**Files:**
- Create: `loc_gs/feedback/cross_view_ray_observations.py`
- Create: `loc_gs/scripts/build_cross_view_ray_observations.py`
- Test: `tests/test_cross_view_ray_observations.py`
- Test: `tests/test_build_cross_view_ray_observations.py`
- Document: `docs/cross_view_ray_observations_phase2a_20260602.md`

- [ ] **Step 1: Write failing core tests**

Test that precomputed cross-view matches are joined with source-view ray
contributors by `source_view_id` and source pixel. Test exact lookup, radius
lookup, dropped match counts, and test-split rejection.

- [ ] **Step 2: Implement core extractor**

Implement:
- `pixel_key`
- `build_ray_contributor_index`
- `find_ray_contributors`
- `build_cross_view_ray_observations`

- [x] **Step 3: Write failing CLI tests**

Test that the CLI reads match JSONL and ray JSONL, writes `observations.jsonl`,
`manifest.json`, `metrics_summary.json`, `split_audit.json`, `command.txt`, and
`git_status.txt`, and rejects `split_name=test`.

- [x] **Step 4: Implement CLI**

Implement `loc_gs.scripts.build_cross_view_ray_observations` as a pure artifact
builder. It must not run feature extraction, PnP, or Cambridge official test.

- [x] **Step 5: Verify and document**

Run focused tests, `py_compile`, `git diff --check`, and document the Phase 2A
schema. This produces observation JSONL suitable for
`build_ray_attributed_solver_feedback`.

Verification commands:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_cross_view_ray_observations.py tests/test_build_cross_view_ray_observations.py -q
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_ray_attributed_solver_feedback.py tests/test_build_ray_attributed_solver_feedback.py tests/test_cross_view_ray_observations.py tests/test_build_cross_view_ray_observations.py -q
/root/miniconda3/envs/cybersim_agent/bin/python -m py_compile loc_gs/feedback/ray_attributed_solver_feedback.py loc_gs/scripts/build_ray_attributed_solver_feedback.py loc_gs/feedback/cross_view_ray_observations.py loc_gs/scripts/build_cross_view_ray_observations.py
git diff --check -- loc_gs/feedback/ray_attributed_solver_feedback.py loc_gs/scripts/build_ray_attributed_solver_feedback.py loc_gs/feedback/cross_view_ray_observations.py loc_gs/scripts/build_cross_view_ray_observations.py tests/test_ray_attributed_solver_feedback.py tests/test_build_ray_attributed_solver_feedback.py tests/test_cross_view_ray_observations.py tests/test_build_cross_view_ray_observations.py docs/ray_attributed_solver_feedback_phase1_20260602.md docs/cross_view_ray_observations_phase2a_20260602.md docs/superpowers/plans/2026-06-02-ray-attributed-solver-feedback.md
```

## Phase 2B: Raster Intersection To Source-Ray Contributor Adapter

Goal: define the artifact layer that lets ULF-Loc/gsplat emit multiple
Gaussian contributors per rendered source pixel. This keeps the feedback
assignment soft over the ray instead of assigning each match to one depth or one
top Gaussian.

Files:
- Create `loc_gs/feedback/raster_ray_contributors.py`
- Create `loc_gs/scripts/build_raster_ray_contributors.py`
- Test `tests/test_raster_ray_contributors.py`
- Test `tests/test_build_raster_ray_contributors.py`
- Document `docs/raster_ray_contributors_phase2b_20260602.md`

- [x] **Step 1: Write failing core tests**

Test pixel id/xy parsing, grouping by pixel, top-k contributor retention,
opacity proxy weights, expected/rendered depth summaries, and test split
rejection.

- [x] **Step 2: Implement core adapter**

Implement `raster_intersections_to_ray_records` and `infer_pixel_xy` without
depending on ULF-Loc imports.

- [x] **Step 3: Write failing CLI tests**

Test `source_rays.jsonl`, `manifest.json`, `metrics_summary.json`,
`split_audit.json`, `command.txt`, and `git_status.txt`.

- [x] **Step 4: Implement CLI**

Implement `loc_gs.scripts.build_raster_ray_contributors`.

- [x] **Step 5: Verify and document**

Verification commands:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_raster_ray_contributors.py tests/test_build_raster_ray_contributors.py -q
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_ray_attributed_solver_feedback.py tests/test_build_ray_attributed_solver_feedback.py tests/test_cross_view_ray_observations.py tests/test_build_cross_view_ray_observations.py tests/test_raster_ray_contributors.py tests/test_build_raster_ray_contributors.py -q
/root/miniconda3/envs/cybersim_agent/bin/python -m py_compile loc_gs/feedback/ray_attributed_solver_feedback.py loc_gs/scripts/build_ray_attributed_solver_feedback.py loc_gs/feedback/cross_view_ray_observations.py loc_gs/scripts/build_cross_view_ray_observations.py loc_gs/feedback/raster_ray_contributors.py loc_gs/scripts/build_raster_ray_contributors.py
git diff --check -- loc_gs/feedback/ray_attributed_solver_feedback.py loc_gs/scripts/build_ray_attributed_solver_feedback.py loc_gs/feedback/cross_view_ray_observations.py loc_gs/scripts/build_cross_view_ray_observations.py loc_gs/feedback/raster_ray_contributors.py loc_gs/scripts/build_raster_ray_contributors.py tests/test_ray_attributed_solver_feedback.py tests/test_build_ray_attributed_solver_feedback.py tests/test_cross_view_ray_observations.py tests/test_build_cross_view_ray_observations.py tests/test_raster_ray_contributors.py tests/test_build_raster_ray_contributors.py docs/ray_attributed_solver_feedback_phase1_20260602.md docs/cross_view_ray_observations_phase2a_20260602.md docs/raster_ray_contributors_phase2b_20260602.md docs/superpowers/plans/2026-06-02-ray-attributed-solver-feedback.md
```

Outcome:

- The new pipeline has three audited artifact builders:
  `raster intersections -> source rays -> observations -> ray feedback`.
- ULF-Loc integration can now focus on emitting packed intersections from
  train/self-map source views without changing evaluator logic.

## Phase 3: ULF Sparse Self-Map Matches To Phase 2A Match Rows

Goal: reuse native ULF-Loc sparse PnP traces as the cross-view match input for
Phase 2A. This stays on train/self-map views and does not run official test
mining.

Files:
- Update `loc_gs/scripts/export_ulfloc_sparse_feedback.py`
- Update `tests/test_export_ulfloc_sparse_feedback.py`
- Document `docs/ulfloc_sparse_matches_phase3_20260602.md`

- [x] **Step 1: Write failing converter test**

Test that ULF sparse match details become Phase 2A-compatible rows with
`query_xy`, `source_xy`, `source_view_id`, `descriptor_score`, `pnp_inlier`, and
`matched_gaussian_id`.

- [x] **Step 2: Implement converter**

Add `cross_view_matches_from_ulfloc_sparse_details`.

- [x] **Step 3: Write failing writer test**

Test `matches.jsonl` manifest and match row format.

- [x] **Step 4: Implement writer and exporter integration**

Add `save_cross_view_matches_jsonl` and make the existing ULF sparse feedback
exporter write `matches.jsonl` alongside `feedback_bank.jsonl`.

- [x] **Step 5: Verify and document**

Verification command:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_export_ulfloc_sparse_feedback.py -q
```

Outcome:

- ULF train/self-map sparse feedback now has a direct bridge into the new ray
  attribution path.
- The remaining missing piece is emitting source-view raster intersections from
  ULF/gsplat for those same train/self-map views.

## Phase 4: Projected Gaussian Proxy Contributors At Sparse Source Pixels

Goal: avoid full-image per-pixel contributor export for the first integration
round. Given ULF sparse `matches.jsonl` and projected Gaussian metadata from the
same train/self-map source views, extract top-k proxy contributors around each
sparse source pixel.

Files:
- Create `loc_gs/feedback/projected_gaussian_rays.py`
- Create `loc_gs/scripts/build_projected_gaussian_rays.py`
- Test `tests/test_projected_gaussian_rays.py`
- Test `tests/test_build_projected_gaussian_rays.py`
- Document `docs/projected_gaussian_rays_phase4_20260602.md`

- [x] **Step 1: Write failing core tests**

Test proxy contribution decay, top-k extraction, uncovered sparse pixel drops,
and split rejection.

- [x] **Step 2: Implement core extractor**

Add `proxy_contribution` and `projected_gaussians_to_sparse_source_rays`.

- [x] **Step 3: Write failing CLI tests**

Test audited `source_rays.jsonl` output from `matches.jsonl` and
`projected_gaussians.jsonl`.

- [x] **Step 4: Implement CLI**

Add `loc_gs.scripts.build_projected_gaussian_rays`.

- [x] **Step 5: Verify and document**

Verification command:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_projected_gaussian_rays.py tests/test_build_projected_gaussian_rays.py -q
```

Outcome:

- The pipeline can now use projected Gaussian metadata to build sparse-pixel
  source rays without exact packed raster contributors.
- This is a proxy path; exact alpha-composition contributors remain the better
  target once ULF/gsplat rendering exposes them.

## Phase 5: Correct Projection Source To Full Raw Gaussians

Goal: fix the architecture wording and artifact contract so projected Gaussian
metadata is explicitly built from the full raw Gaussian map, not ULF sampled
landmarks.

Files:
- Create `loc_gs/feedback/full_raw_gaussian_projection.py`
- Create `loc_gs/scripts/build_full_raw_gaussian_projections.py`
- Create `loc_gs/scripts/export_ulfloc_full_raw_projections.py`
- Test `tests/test_full_raw_gaussian_projection.py`
- Test `tests/test_build_full_raw_gaussian_projections.py`
- Test `tests/test_export_ulfloc_full_raw_projections.py`
- Update `loc_gs/feedback/projected_gaussian_rays.py`
- Update `loc_gs/scripts/export_ulfloc_sparse_feedback.py`
- Document `docs/full_raw_gaussian_projection_phase5_20260603.md`

- [x] **Step 1: Write failing full raw projection tests**

Test all-source count audit, positive-depth/in-frame projection, rejection of
sampled projection source, and test split rejection.

- [x] **Step 2: Implement full raw projection core**

Add `project_full_raw_gaussians_to_view` and
`validate_full_raw_projection_bundle`.

- [x] **Step 3: Write failing CLI tests**

Test audited `projected_gaussians.jsonl`, `manifest.json`,
`metrics_summary.json`, `split_audit.json`, `command.txt`, and `git_status.txt`.

- [x] **Step 4: Implement CLI**

Add `loc_gs.scripts.build_full_raw_gaussian_projections`.

- [x] **Step 5: Enforce downstream full raw projection source**

Make `projected_gaussians_to_sparse_source_rays` reject sampled-subset
projection rows by default.

- [x] **Step 6: Mark ULF sparse matches as bootstrap/diagnostic**

Add `teacher_source=ulfloc_sampled_landmarks`,
`feedback_role=bootstrap_sparse_teacher`, and
`paper_safe_role=diagnostic_not_main_full_raw_feedback`.

Outcome:

- The main feedback artifact can now be generated over the full raw Gaussian
  id space.
- ULF sampled sparse matching remains useful as bootstrap solver evidence, but
  is no longer described as the main full-raw feedback source.

- [x] **Step 7: Add native ULF full raw projection exporter**

Add `loc_gs.scripts.export_ulfloc_full_raw_projections`, which reads
`gaussians.get_xyz` and `Scene.getTrainCameras()` from ULF-Loc and writes the
same `projected_gaussians.jsonl` contract. The helper tests verify that it
exports camera `world_to_camera`, intrinsics, and full Gaussian count without
`sampled_idx`.
