# LSF v2 Native Proposal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-safe proposal artifact that can produce real same-budget `sampled_idx` edits by combining STDLoc reconstruction-time native scores with audited v2 LSF support/risk.

**Architecture:** Keep the native STDLoc evaluator unchanged. A new diagnostic module reads `solver_consensus_support.pt` plus source-map native `score_avg` or point-cloud `locability_logit`, writes selector/candidate/safe-core tensors, then existing `export_lsf_solver_aware_map` materializes the map.

**Tech Stack:** Python, PyTorch tensors, existing STDLoc map payloads, `plyfile` for point-cloud locability fallback, pytest.

---

### Task 1: v2 Native Proposal Artifacts

**Files:**
- Create: `loc_gs/diagnostics/lsf_v2_native_proposal.py`
- Create: `loc_gs/scripts/build_lsf_v2_native_proposal.py`
- Test: `tests/test_lsf_v2_native_proposal.py`

- [x] **Step 1: Write failing tests**

Test that the builder:
- rejects a support artifact whose metadata split is `test`;
- loads full `score_avg` from a source map;
- writes `selector_lsf_v2_native_proposal.pt`, `candidate_pool_lsf_v2_native_topK.pt`, `safe_core_lsf_v2_source.pt`, risk tensors, `summary.json`, and `manifest.json`;
- ranks non-source candidates by fused native score plus LSF support minus risk;
- keeps supported low-risk source landmarks in safe core.

- [x] **Step 2: Verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_lsf_v2_native_proposal.py -q
```

Expected: import/module failure before implementation.

- [x] **Step 3: Implement module and CLI**

Implement a small module with:
- `_load_source_idx(source_map)`;
- `_load_native_score(source_map, size)` using `detector/sampled_scores.pkl` first and `point_cloud/iteration_30000/point_cloud.ply` `locability_logit` fallback;
- `build_lsf_v2_native_proposal(...)` writing tensors and manifest.

The CLI must expose fixed thresholds and output paths, and must reject test split artifacts.

- [x] **Step 4: Verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_lsf_v2_native_proposal.py -q
```

Expected: all tests pass.

### Task 2: Real Artifact Dry Run And Map Export

**Files:**
- No source edits expected.
- Outputs under `output/lsf_v2_native_proposal_20260521/`.

- [x] **Step 1: Build proposal artifacts for OldHospital, ShopFacade, StMarysChurch**

Use each scene's native source map and matching real-grouped `solver_consensus_support.pt`.

- [x] **Step 2: Export same-budget maps**

Use existing `loc_gs.scripts.export_lsf_solver_aware_map` with `selector_path`, `candidate_pool_path`, `safe_core_path`, and merged risk path from the proposal artifact.

- [x] **Step 3: Verify edits**

Read each output `manifest.json`; require `sampled_idx_changed=true` and nonzero `added_non_native_count` before running long eval.

### Task 3: Train-Dev Evaluation

**Files:**
- Outputs under `output/stdloc_native/train_dev_q80_20260521/`.

- [x] **Step 1: Dry-run native launcher commands**

Use `launch_stdloc_native_cambridge --phase eval --eval_split train --max_test_cameras 80`.

- [x] **Step 2: Run native baseline and candidate maps**

Run only train split. Do not use test for selection.

- [x] **Step 3: Summarize and compare**

Use `locgsctl summarize` and `locgsctl compare`. If the result is positive, freeze this recipe before any test-split eval.

Round outcome:

- ShopFacade train-dev q20/q80 is positive for dense and sparse metrics under the single OpenCV STDLoc path.
- OldHospital is mixed: q20 positive, q80 sparse positive, q80 dense regresses.
- StMarysChurch is a failure boundary: rotation-only changes do not translate into useful TE/recall gains.
- q160 train-dev confirmation runs were started only on `eval_split=train`; they are confirmation/robustness runs, not model-selection on test.
