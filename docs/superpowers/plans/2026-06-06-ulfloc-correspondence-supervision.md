# ULF-Loc Reproduction And Correspondence-Supervision Plan

> **For agentic workers:** Use `superpowers:executing-plans` and keep each
> implementation step auditable. Do not use Cambridge official test images,
> poses, descriptors, or evaluation results for feedback mining or model
> selection.

**Goal:** First explain why the local ULF-Loc run is below the paper table, then
upgrade solver feedback from landmark ranking into correspondence-level
supervision for descriptor fusion, detector targets, pair/match scoring,
PnP-aware ranking, and repeated-structure conflict graphs.

**Current diagnosis:** Loc-GS already has sparse feedback exporters and
full-Gaussian set selection, but current gains remain small. Existing feedback
mostly becomes unary support or per-query scalar support. It does not yet expose
the exact correspondence labels needed to decide whether failures come from
detector starvation, descriptor ambiguity, bad ranking, degenerate PnP geometry,
or render/map artifacts.

## Task 1: ULF Reproduction Gap Audit

Files:
- Create `loc_gs/reporting/ulfloc_reproduction_audit.py`
- Create `loc_gs/scripts/audit_ulfloc_reproduction.py`
- Test `tests/test_ulfloc_reproduction_audit.py`

Steps:
- [ ] Parse local ULF checkout state, config files, requirement versions,
      result manifests, split names, and sparse/dense metric summaries.
- [ ] Classify reproduction blockers: incomplete checkout, non-native local
      modifications, missing five-scene paper-test outputs, train-dev-only
      outputs, environment/version mismatch, missing processed/mask usage, and
      missing native config parity.
- [ ] Emit `report.json`, `report.md`, `manifest.json`, `command.txt`, and
      `git_status.txt` under `output/reports/ulfloc_reproduction_audit_*`.

## Task 2: Sparse Failure Decomposition

Files:
- Create `loc_gs/diagnostics/sparse_failure_decomposition.py`
- Test `tests/test_sparse_failure_decomposition.py`

Steps:
- [ ] Convert query-level match/PnP/render summaries into mechanism scores:
      detector starvation, descriptor ambiguity, ranking failure, geometry
      degeneracy, and map/render artifact risk.
- [ ] Return a primary cause plus all active flags, so hard cases are not forced
      into a single false category.
- [ ] Keep the module synthetic-testable and paper-safe; it is diagnostic only.

## Task 3: Correspondence Supervision Artifact

Files:
- Create `loc_gs/feedback/correspondence_supervision.py`
- Test `tests/test_correspondence_supervision.py`

Steps:
- [ ] Build records from existing `FeedbackMatchRecord` rows.
- [ ] Export targets for:
      descriptor fusion weights,
      detector heatmap weights,
      pair/match scorer labels,
      PnP-aware ranking labels,
      repeated-structure conflict edges.
- [ ] Preserve split safety: reject `split_name=test` artifacts.

## Task 4: First Smoke Diagnostics

Steps:
- [ ] Run the audit on `/root/ULF-Loc` and current local outputs.
- [ ] Run synthetic tests and `py_compile`.
- [ ] Summarize which evidence is still missing before launching long ULF
      reproduction or Cambridge full-split experiments.

Verification:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_ulfloc_reproduction_audit.py \
  tests/test_sparse_failure_decomposition.py \
  tests/test_correspondence_supervision.py -q
/root/miniconda3/envs/cybersim_agent/bin/python -m py_compile \
  loc_gs/reporting/ulfloc_reproduction_audit.py \
  loc_gs/scripts/audit_ulfloc_reproduction.py \
  loc_gs/diagnostics/sparse_failure_decomposition.py \
  loc_gs/feedback/correspondence_supervision.py
git diff --check -- \
  docs/superpowers/plans/2026-06-06-ulfloc-correspondence-supervision.md \
  loc_gs/reporting/ulfloc_reproduction_audit.py \
  loc_gs/scripts/audit_ulfloc_reproduction.py \
  loc_gs/diagnostics/sparse_failure_decomposition.py \
  loc_gs/feedback/correspondence_supervision.py \
  tests/test_ulfloc_reproduction_audit.py \
  tests/test_sparse_failure_decomposition.py \
  tests/test_correspondence_supervision.py
```

