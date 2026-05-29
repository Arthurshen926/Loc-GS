# SLCDP-R Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sparse-conditioned local dense pose repair search before dense refinement diagnostics.

**Architecture:** Extend the existing SLCDP diagnostic module with deterministic camera-frame pose candidates, SLCDP candidate scoring, and best-candidate selection. Wire the repair search into `visualize_stdloc_hard_matches.py` behind an explicit flag so the vendored STDLoc evaluator and paper-facing path remain unchanged.

**Tech Stack:** Python, NumPy, PyTorch renderer calls through the existing STDLoc diagnostic wrapper, pytest.

---

### Task 1: Core Pose Search

**Files:**
- Modify: `loc_gs/dense_support/sparse_conditioned_dense_preflight.py`
- Test: `tests/test_sparse_conditioned_dense_preflight.py`

- [ ] Add tests for camera-frame pose offsets, candidate scoring, and best candidate selection.
- [ ] Run the new tests and verify they fail because the new functions do not exist.
- [ ] Implement `SLCDPRepairSearchConfig`, `apply_camera_frame_delta`, `generate_pose_repair_candidates`, `score_slcdp_preflight_result`, and `select_repaired_pose_candidate`.
- [ ] Run the tests and verify they pass.

### Task 2: Renderer Diagnostic Integration

**Files:**
- Modify: `loc_gs/scripts/visualize_stdloc_hard_matches.py`
- Test: `tests/test_sparse_conditioned_dense_preflight.py`

- [ ] Add CLI flags `--slcdp_repair_search` and `--slcdp_repair_skip_if_no_accept`.
- [ ] In `_capture_dense`, render and score candidate poses near sparse pose before dense matching.
- [ ] If a candidate passes SLCDP, use its render/pose for dense matching; if no candidate passes and skip is enabled, return sparse pose with skipped-dense metadata.
- [ ] Persist `slcdp_repair_search` metadata in match summaries.

### Task 3: KingsCollege #239 Verification

**Files:**
- Output: `output/diagnostics/slcdp/kings239_repair_search_20260528`

- [ ] Run the diagnostic on KingsCollege #239 with repair search enabled.
- [ ] Record whether any nearby pose makes sparse landmarks visible.
- [ ] Compare accepted final TE against the original dense TE.
- [ ] Keep the artifact diagnostic-only because it uses official test visualization.
