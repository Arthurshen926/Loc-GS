# LSF Mainline Reset Round 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire solver-consensus support into the same-budget export dry-run path and produce compact real-grouped validation artifacts without launching full Cambridge evaluation.

**Architecture:** `solver_consensus_support.pt` remains an offline mapping artifact. Export may consume it to derive selector/support/risk tensors, but real-query inference still uses one native STDLoc-compatible map and OpenCV PROSAC/RANSAC PnP.

**Tech Stack:** Python, PyTorch, pytest, `loc_gs.scripts.export_lsf_solver_aware_map`, `loc_gs.scripts.build_solver_consensus_support`, existing output artifacts.

---

## Task 3A: Export Solver-Consensus Input

**Files:**
- Modify: `loc_gs/scripts/export_lsf_solver_aware_map.py`
- Modify: `tests/test_export_lsf_solver_aware_map.py`

- [ ] Add `--solver_consensus_support_path`.
- [ ] When provided, load `support_score` as selector/utility support input unless explicit `--selector_path` overrides it.
- [ ] Use `hard_negative_risk` from the artifact when `--hard_negative_risk_path` is absent.
- [ ] Treat `dense_worsen_risk` as an additional risk term in the evidence gate or utility metadata without changing evaluator behavior.
- [ ] Manifest must record `solver_consensus_support_path` and support artifact metadata.
- [ ] Add a dry-run/export test with synthetic support artifact where a high-support candidate is added and a dense-worsened candidate is rejected.

## Task 3B: Real-Grouped Artifact Validation

**Files:**
- Create or update generated artifacts under `output/feedback_bank_v2_20260521/solver_consensus_support_real_grouped/{scene}`.
- Update: `docs/mainline_lsf_20260521.md`

- [ ] Run `build_solver_consensus_support` on available real-grouped feedback banks for OldHospital, ShopFacade, and StMarysChurch.
- [ ] Write one compact markdown/json summary with observed count, selected count, mean support, hard-negative risk, dense-worsen risk, and image group count.
- [ ] State the validation correctly: positive data-chain/mechanism validation, not pose accuracy or SOTA.

## Verification

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_export_lsf_solver_aware_map.py \
  tests/test_solver_consensus_support.py \
  tests/test_solver_aware_resampling.py

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke --scene ShopFacade --dry-run
```
