# STDLoc Native Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add clean STDLoc-native training/evaluation entrypoints to the project so STDLoc can be reproduced before Loc-GS improvements are compared.

**Architecture:** Keep the source-of-truth implementation in `third_party/stdloc`, and expose it through typed `loc_gs/stdloc_native` command builders plus project-level scripts. This prevents current Loc-GS experimental code from contaminating the STDLoc control path.

**Tech Stack:** Python CLI scripts, `subprocess`, JSON result normalization, existing pytest suite.

---

### Task 1: Command Builders

**Files:**
- Create: `loc_gs/stdloc_native/__init__.py`
- Create: `loc_gs/stdloc_native/commands.py`
- Test: `tests/test_stdloc_native_commands.py`

- [ ] Write failing tests for eval/train command generation and GPU environment assignment.
- [ ] Implement dataclasses and command builders.
- [ ] Run `pytest tests/test_stdloc_native_commands.py -q`.

### Task 2: Result Normalization

**Files:**
- Create: `loc_gs/stdloc_native/results.py`
- Test: `tests/test_stdloc_native_results.py`

- [ ] Write failing tests for STDLoc `summary.json` and `results.json` loading.
- [ ] Implement normalized sparse/dense metric extraction.
- [ ] Run `pytest tests/test_stdloc_native_results.py -q`.

### Task 3: Project CLI Wrappers

**Files:**
- Create: `loc_gs/scripts/eval_stdloc_native.py`
- Create: `loc_gs/scripts/train_stdloc_native.py`
- Create: `loc_gs/scripts/launch_stdloc_native_cambridge.py`
- Test: `tests/test_stdloc_native_cli.py`

- [ ] Write failing parser/dry-run tests.
- [ ] Implement dry-run and execution paths.
- [ ] Run `pytest tests/test_stdloc_native_cli.py -q`.

### Task 4: Audit Wiring

**Files:**
- Create: `loc_gs/scripts/audit_stdloc_native_parity.py`
- Modify: `docs/loc_gs_lff_scenematch_mainline_20260512.md`
- Test: `tests/test_audit_cambridge_parity.py`

- [ ] Add a thin audit wrapper around the existing Cambridge parity audit.
- [ ] Document that Loc-GS improvements must compare against STDLoc native output.
- [ ] Run `pytest tests/test_audit_cambridge_parity.py -q`.

### Task 5: Verification

**Files:**
- No production edits.

- [ ] Run focused STDLoc-native tests.
- [ ] Run full project tests.
- [ ] Run `git diff --check`.
