# LSF v5 Solver-Coverage Coreset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LSF v5 solver-coverage coreset selector that optimizes query-level solver support rather than only unary landmark utility.

**Architecture:** Implement a pure coverage-aware same-budget edit module, then wire it into `export_lsf_solver_aware_map` behind `--selection_policy coverage`. The exporter remains backward compatible but v5 mainline runs must pass `--require_candidate_pool` so candidate generation cannot silently fall back to every Gaussian.

**Tech Stack:** Python 3.9, PyTorch tensors, JSON solver admissibility artifacts, pytest, existing Loc-GS STDLoc export/eval scripts.

---

## File Structure

- Create `loc_gs/stdloc_native/solver_coverage_coreset.py`
  - Parse solver admissibility JSON payloads into per-landmark query metrics.
  - Compute query coverage from source landmarks.
  - Select same-budget coverage-aware replacements.

- Modify `loc_gs/scripts/export_lsf_solver_aware_map.py`
  - Add `--selection_policy local|coverage`.
  - Add `--require_candidate_pool`.
  - Record `candidate_pool_path`, `candidate_pool_required`, and coverage metadata.
  - Dispatch to the new coverage selector when requested.

- Create `tests/test_solver_coverage_coreset.py`
  - Synthetic unit tests for coverage novelty, source protection, dense risk rejection, and same-budget output.

- Modify `tests/test_export_lsf_solver_aware_map.py`
  - Test candidate pool manifest recording.
  - Test `--require_candidate_pool` rejects missing pool.
  - Test `--selection_policy coverage` selects a candidate that covers an uncovered hard query.

---

### Task 1: Coverage Coreset Pure Module

**Files:**
- Create: `loc_gs/stdloc_native/solver_coverage_coreset.py`
- Test: `tests/test_solver_coverage_coreset.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_solver_coverage_coreset.py` with tests equivalent to:

```python
import torch

from loc_gs.stdloc_native.solver_coverage_coreset import (
    build_solver_coverage_tables,
    solver_coverage_local_edit,
)


def _constraints():
    return {
        "hard_query_ids": ["q0", "q1"],
        "candidate_gain": {
            "3": {"q0": {"support": 2.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "4": {"q1": {"support": 2.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "5": {"q0": {"support": 1.0, "viable_tuple_mass": 0.1, "dense_worsen_risk": 3.0}},
        },
        "source_loss": {
            "0": {"q0": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "1": {"q1": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "2": {},
        },
        "thresholds": {
            "cvar_weights": {
                "support": 1.0,
                "viable_tuple_mass": 1.0,
                "logdet_H": 1.0,
                "min_eigenvalue": 1.0,
                "dense_worsen_risk": -1.0,
                "ambiguity": -1.0,
            }
        },
    }


def test_build_solver_coverage_tables_parses_string_query_ids():
    tables = build_solver_coverage_tables(_constraints())

    assert tables.hard_query_ids == ("q0", "q1")
    assert tables.candidate_gain[3]["q0"]["support"] == 2.0
    assert tables.source_loss[1]["q1"]["viable_tuple_mass"] == 1.0


def test_solver_coverage_local_edit_prefers_uncovered_queries_and_preserves_budget():
    tables = build_solver_coverage_tables(_constraints())
    result = solver_coverage_local_edit(
        source_idx=torch.tensor([0, 1, 2]),
        candidate_pool=torch.tensor([3, 4, 5]),
        utility=torch.tensor([0.2, 0.2, 0.0, 0.9, 0.8, 0.95]),
        evidence_mask=torch.ones(6, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=2,
        min_coverage_gain=0.1,
    )

    sampled = set(result["sampled_idx"].tolist())
    assert len(sampled) == 3
    assert {3, 4}.issubset(sampled)
    assert 5 not in sampled
    assert result["metadata"]["coverage_policy"] == "solver_coverage_coreset"


def test_solver_coverage_local_edit_protects_source_support_for_low_coverage_query():
    tables = build_solver_coverage_tables(_constraints())
    result = solver_coverage_local_edit(
        source_idx=torch.tensor([0, 1, 2]),
        candidate_pool=torch.tensor([3]),
        utility=torch.tensor([0.0, 0.0, 0.1, 0.9]),
        evidence_mask=torch.ones(4, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=1,
        min_coverage_gain=0.1,
    )

    sampled = set(result["sampled_idx"].tolist())
    assert 1 in sampled
    assert 3 in sampled
    assert result["metadata"]["coverage_protected_drop_count"] >= 1
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_solver_coverage_coreset.py
```

Expected: import failure for `loc_gs.stdloc_native.solver_coverage_coreset`.

- [ ] **Step 3: Implement module**

Implement `CoverageTables`, `build_solver_coverage_tables`, and
`solver_coverage_local_edit`.

Required behavior:

- Normalize candidate/source JSON keys to `int -> str -> dict[str, float]`.
- Compute metric utility with default weights:
  `support=1`, `viable_tuple_mass=1`, `logdet_H=1`, `min_eigenvalue=1`,
  `dense_worsen_risk=-1`, `ambiguity=-1`.
- Candidate score:
  `base_utility + coverage_bonus`, where coverage bonus is larger for hard
  queries with low current coverage.
- Drop cost:
  source utility plus a protection penalty when the source contributes to a hard
  query with current coverage at or below `min_query_coverage`.
- Never accept candidates with coverage gain below `min_coverage_gain`.
- Preserve output count and source order style used by
  `solver_aware_local_edit`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_solver_coverage_coreset.py
```

Expected: all tests pass.

---

### Task 2: Exporter Integration And Guardrails

**Files:**
- Modify: `loc_gs/scripts/export_lsf_solver_aware_map.py`
- Modify: `tests/test_export_lsf_solver_aware_map.py`

- [ ] **Step 1: Write failing exporter tests**

Append tests that verify:

```python
def test_export_lsf_solver_aware_map_records_candidate_pool_path(tmp_path):
    # Build source, selector, positive support, and candidate_pool_path.
    # Run export with --candidate_pool_path.
    # Assert manifest["candidate_pool_path"] equals the path and
    # manifest["solver_aware"]["candidate_pool_count"] equals the pool length.


def test_export_lsf_solver_aware_map_require_candidate_pool_rejects_missing_pool(tmp_path):
    # Build source and selector but do not pass --candidate_pool_path.
    # Run export with --require_candidate_pool.
    # Assert nonzero return and stderr contains "candidate_pool_path is required".


def test_export_lsf_solver_aware_map_uses_coverage_policy(tmp_path):
    # Build source [0,1,2], candidate pool [3,4,5], utility favoring 5.
    # Build solver constraints where 3 and 4 cover q0/q1 and 5 has dense risk.
    # Run with --selection_policy coverage and --solver_admissibility_path.
    # Assert sampled contains 3 and 4, not 5.
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_export_lsf_solver_aware_map.py
```

Expected: failures for missing args / missing manifest field / unsupported policy.

- [ ] **Step 3: Implement exporter wiring**

Update `export_lsf_solver_aware_map.py`:

- Add parser args:
  - `--selection_policy`, choices `local` and `coverage`, default `local`.
  - `--require_candidate_pool`, action `store_true`.
  - `--min_coverage_gain`, float default `0.0`.
  - `--min_query_coverage`, float default `1.0`.
- Track `candidate_pool_path` in manifest.
- If `--require_candidate_pool` and no pool path, raise:
  `ValueError("candidate_pool_path is required when require_candidate_pool is enabled")`.
- If policy is `coverage`, require `solver_admissibility_path` and call
  `build_solver_coverage_tables` plus `solver_coverage_local_edit`.
- Keep existing local path unchanged.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_solver_coverage_coreset.py tests/test_export_lsf_solver_aware_map.py
```

Expected: all pass.

---

### Task 3: v5 Map Export Smoke

**Files:**
- No production code expected.
- Write docs/results after experiment.

- [ ] **Step 1: Dry-run export for one scene**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.export_lsf_solver_aware_map \
  --source_map output/stdloc/map_cambridge_spgs_trainonly_20260524/OldHospital \
  --localization_support_field_path output/lsf_trainonly_native_20260524/support_field/OldHospital/localization_support_field.pt \
  --selector_path output/lsf_trainonly_native_20260524/proposal_fullsize/OldHospital/selector_lsf_v2_native_proposal.pt \
  --hard_negative_risk_path output/lsf_trainonly_native_20260524/proposal_fullsize/OldHospital/merged_risk_lsf_v2.pt \
  --safe_core_path output/lsf_trainonly_native_20260524/proposal_fullsize/OldHospital/safe_core_lsf_v2_source.pt \
  --candidate_pool_path output/lsf_trainonly_native_20260524/proposal_fullsize/OldHospital/candidate_pool_lsf_v2_native_top4096.pt \
  --solver_admissibility_path output/lsf_v4_candidate_probe_native_20260524/solver_admissibility/OldHospital/solver_constraints.json \
  --require_candidate_gain \
  --require_candidate_pool \
  --selection_policy coverage \
  --output_map output/stdloc/map_cambridge_spgs_lsf_v5_coverage_20260524/OldHospital \
  --max_edits 128 \
  --admissibility_drop_scan 32 \
  --write_dense_locability \
  --dry_run
```

Expected: JSON manifest includes `selection_policy=coverage`,
`candidate_pool_path`, and same-budget metadata.

- [ ] **Step 2: Export five v5 maps**

Run one export per scene using the same source/proposal paths already used by
the v4 recipe, with:

```text
--require_candidate_pool
--selection_policy coverage
--max_edits 128
--admissibility_drop_scan 32
--write_dense_locability
```

Output root:

```text
output/stdloc/map_cambridge_spgs_lsf_v5_coverage_20260524
```

- [ ] **Step 3: Verify map audits**

Run a Python check that each scene has:

- `sampled_idx=16384`;
- `same_budget=true`;
- `split_audit.audit_status=passed`;
- `solver_aware.coverage_policy=solver_coverage_coreset`;
- `candidate_pool_path` is non-empty and exists.

---

### Task 4: q80/q160 Train-Dev Evaluation

**Files:**
- Add result note under `docs/`.

- [ ] **Step 1: Launch fixed poselib SPGS prior eval**

Use:

```text
third_party/stdloc/configs/stdloc_spgs_cambridge_dense1.yaml
```

Compare baseline maps vs v5 maps on q80 and q160 for all five scenes.

Output root:

```text
output/stdloc_native/lsf_v5_coverage_spgs_prior_20260524
```

Every eval command must include:

```text
--expected_sampled_count 16384
--require_paper_safe_poselib_evaluator
```

- [ ] **Step 2: Summarize raw results**

Compute dense median TE, R@10, R@5, R@2 deltas for q80/q160.

- [ ] **Step 3: Write precision-primary acceptance reports**

Use `loc_gs.scripts.write_fixed_recipe_gate_report` with:

```text
--recall-policy warn
--max-median-te-regression-cm 0.0
```

Output:

```text
output/reports/lsf_v5_coverage_spgs_prior_20260524/q80_precision_acceptance.json
output/reports/lsf_v5_coverage_spgs_prior_20260524/q160_precision_acceptance.json
```

- [ ] **Step 4: Document result**

Create:

```text
docs/lsf_v5_coverage_status_20260524.md
```

Include raw all-selected deltas, scene-level accepted recipe, audit status, and
whether the first acceptance target was met.

---

## Self-Review

- Spec coverage: covers coverage coreset, candidate pool guardrail, map export,
  train-dev eval, and paper-safety constraints.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: uses existing `sampled_idx`, `candidate_pool_path`,
  `solver_admissibility_path`, and `metrics_summary.json` names.

