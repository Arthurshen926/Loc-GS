# LSF Mainline Reset Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reset the repository entry points to the LSF-Loc mainline and fix the first feedback_bank_v2-to-solver integration bug without changing STDLoc evaluator behavior.

**Architecture:** Keep native STDLoc descriptors, matching, PROSAC/RANSAC PnP, and STDLoc-style dense refinement as the fixed backend. Treat LSF as mapping-time solver-support estimation that exports STDLoc-compatible sampled indices, scores, locability, and dense support masks. Archive residual descriptor, quality gate, SceneMatchNet, LoFTR, oracle ordering, and branch-selection stories as diagnostics.

**Tech Stack:** Python, PyTorch, pytest, existing `loc_gs.feedback`, `loc_gs.diagnostics`, `loc_gs.stdloc_native`, README/docs Markdown, and `locgsctl`.

---

## File Structure

- Modify `README.md`: make LSF-Loc the repository mainline and shorten old Cambridge hybrid commands to protected references.
- Modify `docs/research_contract.md`: update the core claim and evidence standard around solvability-guided support, not residual descriptors or quality gates.
- Modify `docs/experiment_protocol.md`: align allowed/disallowed experiment categories with LSF-Loc gates.
- Modify `docs/mainline_lsf_20260521.md`: keep it as the authoritative active note and add explicit Round 1 status.
- Modify `docs/loc_gs_lff_scenematch_mainline_20260512.md`: mark as archived diagnostic history, not an active paper-facing mainline.
- Modify `loc_gs/stdloc_native/solver_admissibility.py`: accept string query ids from feedback_bank_v2 tuple banks.
- Modify `tests/test_solver_admissibility.py`: add regression coverage for string query ids.

## Task 1: Document Mainline Reset

**Files:**
- Modify: `README.md`
- Modify: `docs/research_contract.md`
- Modify: `docs/experiment_protocol.md`
- Modify: `docs/mainline_lsf_20260521.md`
- Modify: `docs/loc_gs_lff_scenematch_mainline_20260512.md`

- [ ] **Step 1: Replace README Cambridge mainline text**

Write the README Cambridge section around this pipeline:

```text
STDLoc native feature Gaussian map
  -> train/rendered self-localization feedback_bank_v2
  -> solver-consensus support estimation
  -> solvability-aware landmark sampling
  -> support-consistent dense residual verification
  -> one STDLoc-compatible sparse-to-dense inference path
```

The README must state that residual descriptors, quality gates, branch selection, SceneMatchNet, LoFTR, oracle ordering, qcov/churn/ret sweeps, and raw dense locability priors are diagnostics unless future full-split evidence promotes them.

- [ ] **Step 2: Update the research contract**

Make the defensible claim:

```text
Feature selection for 3DGS localization is a solver-level support selection problem. LSF-Loc estimates localization support from self-localization episodes and exports STDLoc-compatible sparse landmark and dense residual support while preserving one fixed evaluator path.
```

Keep the paper-safety constraints: no test-query leakage, no evaluator drift, no per-query branch selection, and no paper-facing run without manifest, command, metrics summary, split audit, and git state.

- [ ] **Step 3: Update experiment protocol categories**

The protocol must distinguish:

```text
main_candidate: fixed LSF recipe that passes v2 audit, train-dev gates, and single-path evaluation
ablation: residual descriptors, dense masks, solver metrics, selector variants under audited splits
diagnostic: quality gates, oracle ordering, SceneMatchNet, LoFTR, scalar sweeps, failed support guards
rejected: any method that uses test query signal or modifies evaluator behavior
```

- [ ] **Step 4: Archive the old LFF/SceneMatchNet note**

At the top of `docs/loc_gs_lff_scenematch_mainline_20260512.md`, add an archive banner explaining it is retained for diagnostic history and superseded by `docs/mainline_lsf_20260521.md`.

- [ ] **Step 5: Verify doc cleanup**

Run:

```bash
rg -n "strongest paper-facing|current empirical default|paper should keep the quality gate as the main|active 2026-05-16 paper story|selector005_native_desc" README.md docs/research_contract.md docs/experiment_protocol.md docs/mainline_lsf_20260521.md docs/loc_gs_lff_scenematch_mainline_20260512.md
```

Expected: no active-mainline wording remains. Archive-only mentions are allowed only inside the archived LFF note.

## Task 2: Fix feedback_bank_v2 String Query Ids In Solver Safe Core

**Files:**
- Modify: `loc_gs/stdloc_native/solver_admissibility.py`
- Modify: `tests/test_solver_admissibility.py`

- [ ] **Step 1: Write the failing regression test**

Add a test:

```python
def test_build_safe_core_from_tuple_bank_accepts_feedback_bank_v2_string_query_ids():
    tuple_bank = {
        "queries": [
            {
                "query_id": "seq1/frame00001.png",
                "tuples": [
                    {"landmark_ids": [10, 11, 12, 13], "viable": True, "geometry_logdet": 4.0},
                ],
            },
        ]
    }

    safe_core, meta = build_safe_core_from_tuple_bank(
        tuple_bank,
        hard_query_ids={"seq1/frame00001.png"},
        source_idx=torch.arange(20),
        top_tuples_per_query=1,
    )

    assert safe_core.tolist() == [10, 11, 12, 13]
    assert meta["hard_query_count"] == 1
    assert meta["used_query_count"] == 1
```

- [ ] **Step 2: Run the test and confirm RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_solver_admissibility.py::test_build_safe_core_from_tuple_bank_accepts_feedback_bank_v2_string_query_ids
```

Expected before the fix: fail with a `ValueError` from integer conversion.

- [ ] **Step 3: Implement minimal query-id normalization**

Change `build_safe_core_from_tuple_bank` so `hard_query_ids` and tuple-bank `query_id` are compared as strings while preserving integer query ids for existing tests.

- [ ] **Step 4: Run solver admissibility tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_solver_admissibility.py
```

Expected: all tests pass.

## Task 3: Round 1 Verification

**Files:**
- No new production files.

- [ ] **Step 1: Run focused LSF tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_feedback_bank_v2_audit.py \
  tests/test_feedback_solver_tuple_bank_v2.py \
  tests/test_dense_lsf_distillation.py \
  tests/test_support_failure_mechanism.py \
  tests/test_solver_admissibility.py \
  tests/test_feedback_bank_export.py \
  tests/test_stage_transitions.py
```

Expected: all tests pass.

- [ ] **Step 2: Run locgsctl safe checks**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl status
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl list-scenes
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke --scene ShopFacade --dry-run
```

Expected: commands complete without launching a full Cambridge experiment.

- [ ] **Step 3: Assign Round 2**

If Round 1 verifies, assign the next implementation slice:

```text
Round 2A: solver-consensus support estimation from feedback_bank_v2.
Round 2B: solvability-aware same-budget sampling objective with CVaR hard-query aggregation.
Round 2C: support-consistent dense verification integration point and smoke-only validation.
```
