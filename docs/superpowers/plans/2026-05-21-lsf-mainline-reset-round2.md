# LSF Mainline Reset Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert audited `feedback_bank_v2` traces into compact LSF training/sampling artifacts and add hard-query CVaR sampling utilities without running full Cambridge experiments.

**Architecture:** Round 2 stays offline and map-building only. It adds solver-consensus support estimation, CVaR hard-query scoring, and dense-support smoke validation while preserving native STDLoc inference as the only real-query path.

**Tech Stack:** Python, PyTorch, pytest, existing `loc_gs.feedback`, `loc_gs.diagnostics`, `loc_gs.stdloc_native`, and `loc_gs.dense_support` modules.

---

## Task 2A: Solver-Consensus Support Artifact

**Files:**
- Create: `loc_gs/diagnostics/solver_consensus_support.py`
- Create: `loc_gs/scripts/build_solver_consensus_support.py`
- Create: `tests/test_solver_consensus_support.py`

- [ ] Build a per-Gaussian support artifact from `feedback_bank_v2` records.
- [ ] Reject banks that fail `audit_feedback_bank_v2`.
- [ ] Output tensors: `support_score`, `inlier_consensus`, `weighted_support`, `hard_negative_risk`, `dense_worsen_risk`, `observed_count`.
- [ ] Output metadata: schema, split, image group count, selected threshold, source feedback bank.
- [ ] Add a CLI that writes `solver_consensus_support.pt` and `manifest.json`.
- [ ] Verify on synthetic v2 banks with real `image_id` groups.

## Task 2B: CVaR Hard-Query Sampling Utility

**Files:**
- Modify: `loc_gs/stdloc_native/solver_aware_resampling.py`
- Modify: `tests/test_solver_aware_resampling.py`

- [ ] Add a small utility function that scores query-level solver deltas using mean utility plus lower-tail CVaR.
- [ ] Inputs should accept string query ids.
- [ ] Include support, viable tuple mass, logdet, and ambiguity weights.
- [ ] Add a test where mean utility is positive but worst-tail CVaR fails, so the score rejects the candidate.
- [ ] Do not change existing export behavior until Round 3 wires the objective into map export.

## Task 2C: Dense LSF Dense-Only Guard

**Files:**
- Modify: `docs/mainline_lsf_20260521.md`
- Modify: `tests/test_dense_lsf_distillation.py` if needed.

- [ ] Keep dense LSF explicitly dense-only in docs and tests.
- [ ] Verify dense LSF target generation does not claim sparse selector safety.
- [ ] No changes to STDLoc evaluator or dense backend in this round.

## Verification

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_solver_consensus_support.py \
  tests/test_solver_aware_resampling.py \
  tests/test_dense_lsf_distillation.py \
  tests/test_feedback_bank_v2_audit.py

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke --scene ShopFacade --dry-run
```

Expected: pytest passes. The dry-run must not launch a long experiment; missing checkpoint is an acceptable clear status.
