# LSF v6 Saturated Coverage Coreset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a saturated hard-query coverage objective that prevents large edit budgets from over-serving already covered queries.

**Architecture:** Extend `solver_coverage_coreset` with optional saturation and tail-query weighting while preserving v5 defaults. Wire the new controls into `export_lsf_solver_aware_map.py` and record all policy fields in map manifests.

**Tech Stack:** Python 3.9, PyTorch tensors, JSON solver admissibility artifacts, pytest, existing Loc-GS STDLoc export/eval scripts.

---

## File Structure

- Modify `loc_gs/stdloc_native/solver_coverage_coreset.py`
  - Add saturation target computation.
  - Add saturated candidate gain scoring.
  - Add metadata fields for v6 policy and query tail.

- Modify `loc_gs/scripts/export_lsf_solver_aware_map.py`
  - Add v6 CLI arguments.
  - Pass v6 arguments to `solver_coverage_local_edit`.
  - Record v6 fields in `hyperparameters` and `solver_aware`.

- Modify `tests/test_solver_coverage_coreset.py`
  - Add synthetic tests for saturation suppressing already-covered easy-query gain.
  - Keep existing v5 behavior tests passing with defaults.

- Modify `tests/test_export_lsf_solver_aware_map.py`
  - Add manifest/CLI coverage for v6 saturation arguments.

## Task 1: Core Saturation Objective

- [x] Write a failing test in `tests/test_solver_coverage_coreset.py` where a high-utility candidate only improves an already saturated query, while a lower-utility candidate improves an under-covered hard query. Expected: v6 selects the under-covered hard-query candidate.
- [x] Run `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_solver_coverage_coreset.py`.
- [x] Implement saturation target and saturated gain scoring in `loc_gs/stdloc_native/solver_coverage_coreset.py`.
- [x] Re-run the same test file and confirm all tests pass.

## Task 2: Exporter Wiring

- [x] Write a failing test in `tests/test_export_lsf_solver_aware_map.py` that runs `--selection_policy coverage --coverage_saturation_mode native_percentile` and asserts manifest fields are present.
- [x] Run `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_export_lsf_solver_aware_map.py`.
- [x] Add parser arguments and pass them through to `solver_coverage_local_edit`.
- [x] Re-run targeted tests for coreset and exporter.

## Task 3: v6 Map Export

- [x] Run `locgsctl status` and `locgsctl list-scenes`.
- [x] Export five v6 maps under `output/stdloc/map_cambridge_spgs_lsf_v6_saturated512_20260525`.
- [x] Export guarded drop-protection follow-up under `output/stdloc/map_cambridge_spgs_lsf_v6_guarded512_20260525`.
- [x] Verify each map has `sampled_count=16384`, same-budget true, split audit passed, explicit candidate pool, and `coverage_saturation_mode=native_percentile`.

## Task 4: q80/q160 Evaluation

- [x] Evaluate v6 selected maps on Cambridge train-dev q80/q160 with `third_party/stdloc/configs/stdloc_spgs_cambridge_dense1.yaml`.
- [x] Reuse the existing v5 baseline runs when protocol matches; otherwise regenerate baselines with the same wrapper.
- [x] Write precision-primary reports under `output/reports/lsf_v6_saturated512_spgs_prior_20260525`.
- [x] Write guarded precision-primary reports under `output/reports/lsf_v6_guarded512_spgs_prior_20260525`.
- [x] Write `docs/lsf_v6_saturated_coverage_status_20260525.md` comparing v5-128, v5-512, v6-saturated-512, and v6-guarded-512.

## Self-Review

- Spec coverage: covers objective, exporter, maps, eval, and report.
- Placeholder scan: no TODO/TBD placeholders.
- Type consistency: CLI names match planned function argument names.
