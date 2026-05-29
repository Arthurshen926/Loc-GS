# Review1 Method Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the seven `review1.md` method-upgrade directions as tested Loc-GS components.

**Architecture:** Add focused offline modules for support banks, descriptor fusion, dense verification, detector targets, negative memory, and large-edit resampling policy. Keep LSF v6 as the existing saturation controller and avoid changing the vendored evaluator or Cambridge split logic.

**Tech Stack:** Python 3.9, PyTorch tensors, pytest, existing Loc-GS feedback/STDLoc-native modules.

---

## File Structure

- Create `loc_gs/stdloc_native/support_banks.py`
  - Deterministic query grouping, bank support aggregation, and router metadata.
- Create `loc_gs/stdloc_native/solver_weighted_feature_fusion.py`
  - Solver/geometry/visibility weighted landmark descriptor fusion.
- Create `loc_gs/dense_support/dense_match_verifier.py`
  - Multi-signal dense correspondence scoring and filtering.
- Create `loc_gs/stdloc_native/detector_target_refinement.py`
  - LSF-weighted detector heatmap and supervision-weight target builder.
- Create `loc_gs/stdloc_native/negative_support_memory.py`
  - Pairwise hard-negative conflict graph and set penalty helpers.
- Create `loc_gs/stdloc_native/bank_level_resampling.py`
  - Safe large-edit budget and recipe validation helpers.
- Add tests:
  - `tests/test_support_banks.py`
  - `tests/test_solver_weighted_feature_fusion.py`
  - `tests/test_dense_match_verifier.py`
  - `tests/test_detector_target_refinement.py`
  - `tests/test_negative_support_memory.py`
  - `tests/test_bank_level_resampling.py`
- Add status doc:
  - `docs/review1_method_upgrades_status_20260525.md`

## Tasks

- [x] Add failing tests for the six new method components.
- [x] Verify the tests fail because modules/functions are missing.
- [x] Implement minimal component modules.
- [x] Run targeted tests and fix failures.
- [x] Write status doc mapping all seven `review1.md` directions to code and next experiment gates.
- [x] Run final verification: targeted tests, `git diff --check`, and `git diff --quiet -- third_party/stdloc`.

## Self-Review

- Spec coverage: covers all seven directions, with direction 4 mapped to existing v6.
- Placeholder scan: no TODO/TBD placeholders.
- Type consistency: all module names match the planned tests.
