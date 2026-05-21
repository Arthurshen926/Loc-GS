# Query-Conditioned LSF Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first query-conditioned solver-aware support generator for Loc-GS, producing `solver_admissibility_path` constraints from self-map feedback banks with real `query_id`.

**Architecture:** Keep STDLoc inference unchanged. Add a diagnostics module that reads a self-map episode cache, identifies hard queries, estimates candidate gain and source loss per query, and writes the JSON schema already consumed by `export_lsf_solver_aware_map.py`.

**Tech Stack:** Python, PyTorch tensors, existing `loc_gs.diagnostics` and `loc_gs.stdloc_native` modules, pytest with synthetic caches.

---

### Task 1: Query-Conditioned Constraint Builder

**Files:**
- Create: `loc_gs/stdloc_native/query_conditioned_support.py`
- Test: `tests/test_query_conditioned_support.py`

- [ ] **Step 1: Write failing tests**

Create tests that build a synthetic `selfmap_episode_v1` payload with two `query_id`s. The expected behavior is:

```python
constraints = build_query_conditioned_solver_constraints(
    payload,
    source_idx=torch.tensor([0, 1]),
    num_gaussians=4,
    hard_query_ids=[10],
    min_candidate_positive=0.5,
    source_loss_default=0.0,
)
assert constraints["hard_query_ids"] == [10]
assert constraints["candidate_gain"]["2"]["10"]["support"] > 0
assert constraints["candidate_gain"]["3"]["10"]["support"] == 0
assert constraints["source_loss"]["0"]["10"]["support"] > 0
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_query_conditioned_support.py
```

Expected: import failure because the module does not exist yet.

- [ ] **Step 3: Implement minimal module**

Implement:

```python
def build_query_conditioned_solver_constraints(...):
    # validate query_id exists; no synthetic fallback here
    # map candidate ids through base_gaussian_id when present
    # positive support = pair_label or reprojection threshold + visibility + pnp inlier
    # source_loss records positive source landmarks per hard query
    # candidate_gain records positive non-source landmarks per hard query
    # logdet proxy is per-landmark query support quality for now
```

- [ ] **Step 4: Run tests and keep existing admissibility tests green**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_query_conditioned_support.py tests/test_solver_admissibility.py tests/test_export_lsf_solver_aware_map.py
```

### Task 2: CLI Exporter

**Files:**
- Create: `loc_gs/scripts/build_query_conditioned_solver_constraints.py`
- Test: `tests/test_build_query_conditioned_solver_constraints_cli.py`

- [ ] **Step 1: Write failing CLI test**

The test saves a synthetic `.pt` cache and `source_idx.pkl`, invokes the module, and asserts a JSON file with `hard_query_ids`, `candidate_gain`, `source_loss`, and `thresholds` exists.

- [ ] **Step 2: Implement CLI**

Arguments:

```text
--episode_cache
--source_idx
--output_json
--num_gaussians
--hard_query_ids optional comma/list
--reprojection_threshold_px
--score_threshold
--min_candidate_positive
--min_logdet_delta
--min_viable_tuple_delta
```

- [ ] **Step 3: Run CLI tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_build_query_conditioned_solver_constraints_cli.py
```

### Task 3: Documentation and Verification

**Files:**
- Modify: `docs/lsf_implementation_status_20260518.md`

- [ ] **Step 1: Document new path**

Add a section explaining that the next export should use:

```bash
python -m loc_gs.scripts.build_query_conditioned_solver_constraints ...
python -m loc_gs.scripts.export_lsf_solver_aware_map --solver_admissibility_path ...
```

- [ ] **Step 2: Verify full targeted suite**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_query_conditioned_support.py tests/test_build_query_conditioned_solver_constraints_cli.py tests/test_export_lsf_solver_aware_map.py tests/test_solver_admissibility.py tests/test_selfmap_episode_cache.py
/root/miniconda3/envs/cybersim_agent/bin/python -m compileall -q loc_gs
git diff --check
```
