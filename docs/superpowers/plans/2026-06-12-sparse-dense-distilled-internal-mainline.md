# Sparse-Dense Distilled Internal Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first internal Loc-GS slice for sparse-dense distilled, solver-aware sparse localization without runtime dependence on `/root/ULF-Loc` or vendored STDLoc execution scripts.

**Architecture:** Add thin internal APIs first: split/external-runtime guards, sparse candidate schemas, teacher-label schemas, deterministic reranking, and a sparse-only dry-run CLI that writes auditable manifests. Historical scripts remain for reproduction, but the new mainline entry points import only `loc_gs` modules.

**Tech Stack:** Python 3.9, dataclasses, NumPy, PyTorch optional for tensors, pytest, existing Loc-GS test conventions.

---

## File Structure

- Create `loc_gs/sparse/audit.py`: split rejection and forbidden external-runtime scan.
- Create `loc_gs/sparse/correspondences.py`: sparse query/candidate dataclasses and validation helpers.
- Create `loc_gs/teacher/labels.py`: teacher label dataclasses and dense-teacher split guards.
- Create `loc_gs/sparse/rerank.py`: deterministic correspondence reranking and oracle/candidate-availability summaries.
- Create `loc_gs/scripts/eval_sparse_distilled_cambridge.py`: sparse-only new-mainline dry-run/eval entry point with manifest output.
- Create `loc_gs/scripts/audit_internal_mainline.py`: CLI wrapper around the dependency audit.
- Modify `loc_gs/sparse/__init__.py` and `loc_gs/teacher/__init__.py`: export stable APIs.
- Add tests:
  - `tests/test_internal_mainline_audit.py`
  - `tests/test_sparse_distilled_schemas.py`
  - `tests/test_sparse_distilled_rerank.py`
  - `tests/test_eval_sparse_distilled_cambridge.py`

## Task 1: Internal Mainline Audit Guards

**Files:**
- Create: `loc_gs/sparse/audit.py`
- Create: `tests/test_internal_mainline_audit.py`
- Modify: `loc_gs/sparse/__init__.py`

- [ ] **Step 1: Write failing tests**

Add tests that reject test splits and catch forbidden external runtime imports:

```python
from pathlib import Path

import pytest

from loc_gs.sparse.audit import (
    ForbiddenRuntimeDependency,
    assert_internal_mainline_sources,
    reject_test_split,
)


def test_reject_test_split_blocks_training_labels():
    for split in ("test", "official_test", "cambridge_test"):
        with pytest.raises(ValueError, match="test split"):
            reject_test_split(split, purpose="teacher labels")


def test_reject_test_split_allows_non_test_and_unknown():
    assert reject_test_split("train_selfmap", purpose="teacher labels") == "train_selfmap"
    assert reject_test_split("", purpose="eval manifest") == "unknown"


def test_internal_mainline_audit_finds_external_ulfloc_import(tmp_path: Path):
    path = tmp_path / "bad.py"
    path.write_text("import sys\nsys.path.insert(0, '/root/ULF-Loc')\nfrom ulfloc import ULFLoc\n")

    with pytest.raises(ForbiddenRuntimeDependency) as exc:
        assert_internal_mainline_sources([path])

    assert "/root/ULF-Loc" in str(exc.value)
    assert str(path) in str(exc.value)


def test_internal_mainline_audit_finds_vendored_stdloc_runtime(tmp_path: Path):
    path = tmp_path / "bad_stdloc.py"
    path.write_text("STDLOC_ROOT = 'third_party/stdloc'\nsubprocess.run(['python', 'third_party/stdloc/stdloc.py'])\n")

    with pytest.raises(ForbiddenRuntimeDependency, match="third_party/stdloc/stdloc.py"):
        assert_internal_mainline_sources([path])


def test_internal_mainline_audit_allows_reference_text_when_disabled(tmp_path: Path):
    path = tmp_path / "doc.py"
    path.write_text("REFERENCE = 'third_party/stdloc/configs/stdloc_cambridge.yaml'\n")

    assert_internal_mainline_sources([path])
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_internal_mainline_audit.py
```

Expected: import failure for `loc_gs.sparse.audit`.

- [ ] **Step 3: Implement guard module**

Implement `loc_gs/sparse/audit.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORBIDDEN_RUNTIME_PATTERNS: tuple[str, ...] = (
    "/root/ULF-Loc",
    "from ulfloc import",
    "import ulfloc",
    "third_party/stdloc/stdloc.py",
    "third_party/stdloc/train.py",
)


class ForbiddenRuntimeDependency(RuntimeError):
    """Raised when a new-mainline source depends on external method runtime code."""


@dataclass(frozen=True)
class ForbiddenRuntimeHit:
    path: Path
    pattern: str
    line_number: int
    line: str


def reject_test_split(split_name: str | None, *, purpose: str) -> str:
    split = str(split_name or "unknown").strip() or "unknown"
    lowered = split.lower()
    if lowered == "test" or lowered == "official_test" or lowered.endswith("_test"):
        raise ValueError(f"test split is not allowed for {purpose}: {split}")
    return split


def scan_forbidden_runtime_dependencies(paths: Iterable[str | Path]) -> list[ForbiddenRuntimeHit]:
    hits: list[ForbiddenRuntimeHit] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            for pattern in FORBIDDEN_RUNTIME_PATTERNS:
                if pattern in line:
                    hits.append(ForbiddenRuntimeHit(path=path, pattern=pattern, line_number=line_number, line=line.strip()))
    return hits


def assert_internal_mainline_sources(paths: Iterable[str | Path]) -> None:
    hits = scan_forbidden_runtime_dependencies(paths)
    if not hits:
        return
    detail = "; ".join(f"{hit.path}:{hit.line_number}: {hit.pattern}" for hit in hits)
    raise ForbiddenRuntimeDependency(f"forbidden external runtime dependency found: {detail}")
```

Update `loc_gs/sparse/__init__.py` to export the new functions.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_internal_mainline_audit.py
```

Expected: all tests pass.

## Task 2: Sparse Candidate And Teacher Label Schemas

**Files:**
- Create: `loc_gs/sparse/correspondences.py`
- Create: `loc_gs/teacher/labels.py`
- Create: `loc_gs/teacher/__init__.py`
- Create: `tests/test_sparse_distilled_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Add tests for top-K candidate validation and dense-teacher split rejection:

```python
import pytest

from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.teacher.labels import TeacherLabelBatch, build_teacher_labels


def test_sparse_candidate_batch_requires_matching_topk_shapes():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train_selfmap",
        query_id="q1",
        keypoint_xy=[[10.0, 20.0], [30.0, 40.0]],
        candidate_landmark_ids=[[1, 2], [3]],
        candidate_scores=[[0.9, 0.8], [0.7, 0.6]],
    )

    with pytest.raises(ValueError, match="same top-k length"):
        batch.validate()


def test_teacher_labels_reject_dense_teacher_on_test_split():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="test",
        query_id="q1",
        keypoint_xy=[[10.0, 20.0]],
        candidate_landmark_ids=[[1, 2]],
        candidate_scores=[[0.9, 0.8]],
    )

    with pytest.raises(ValueError, match="test split"):
        build_teacher_labels(batch, geometric_correct=[[True, False]], dense_consistent=[[True, False]])


def test_teacher_labels_preserve_per_candidate_roles():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train_selfmap",
        query_id="q1",
        keypoint_xy=[[10.0, 20.0]],
        candidate_landmark_ids=[[1, 2]],
        candidate_scores=[[0.9, 0.8]],
    )

    labels = build_teacher_labels(
        batch,
        geometric_correct=[[True, False]],
        dense_consistent=[[True, False]],
        sparse_inlier=[[True, False]],
        reprojection_error_px=[[1.0, 18.0]],
    )

    assert isinstance(labels, TeacherLabelBatch)
    assert labels.label_roles == [["protected_support", "hard_negative"]]
    assert labels.positive_count == 1
    assert labels.hard_negative_count == 1
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_sparse_distilled_schemas.py
```

Expected: import failure for the new schema modules.

- [ ] **Step 3: Implement schemas**

Implement focused dataclasses with pure-Python lists so tests do not require
GPU or Torch tensors:

```python
@dataclass(frozen=True)
class SparseCandidateBatch:
    scene: str
    split_name: str
    query_id: str
    keypoint_xy: Sequence[Sequence[float]]
    candidate_landmark_ids: Sequence[Sequence[int]]
    candidate_scores: Sequence[Sequence[float]]

    def validate(self) -> None:
        if not self.scene:
            raise ValueError("scene is required")
        if not self.query_id:
            raise ValueError("query_id is required")
        if len(self.keypoint_xy) != len(self.candidate_landmark_ids):
            raise ValueError("keypoint_xy and candidate_landmark_ids must have the same keypoint count")
        if len(self.candidate_scores) != len(self.candidate_landmark_ids):
            raise ValueError("candidate_scores and candidate_landmark_ids must have the same keypoint count")
        for row_idx, (ids, scores) in enumerate(zip(self.candidate_landmark_ids, self.candidate_scores)):
            if len(ids) != len(scores):
                raise ValueError(f"candidate ids and scores must have the same top-k length at row {row_idx}")

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoint_xy)
```

`build_teacher_labels` should call `reject_test_split` before accepting dense
labels and should assign:

- `protected_support` when geometric, dense-consistent, and sparse-inlier are true;
- `positive_inlier` when geometric and sparse-inlier are true;
- `hard_negative` when not geometric and reprojection error is at least 8 px;
- `neutral` otherwise.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_sparse_distilled_schemas.py
```

Expected: all tests pass.

## Task 3: Solver-Aware Reranker

**Files:**
- Create: `loc_gs/sparse/rerank.py`
- Create: `tests/test_sparse_distilled_rerank.py`

- [ ] **Step 1: Write failing reranker tests**

Add tests that verify PROSAC prefix preservation and oracle/candidate summaries:

```python
from loc_gs.sparse.rerank import CandidateRerankConfig, rerank_candidate_rows, summarize_candidate_availability


def test_reranker_preserves_high_score_prefix_and_reranks_tail():
    rows = [
        {"candidate_id": "a", "native_score": 0.99, "solver_score": 0.1},
        {"candidate_id": "b", "native_score": 0.98, "solver_score": 0.2},
        {"candidate_id": "c", "native_score": 0.70, "solver_score": 0.3},
        {"candidate_id": "d", "native_score": 0.60, "solver_score": 0.9},
    ]

    reranked = rerank_candidate_rows(rows, CandidateRerankConfig(prefix_fraction=0.5, solver_weight=1.0))

    assert [row["candidate_id"] for row in reranked] == ["a", "b", "d", "c"]


def test_candidate_availability_reports_oracle_gap():
    summary = summarize_candidate_availability(
        [
            [{"geometric_correct": False}, {"geometric_correct": True}],
            [{"geometric_correct": False}, {"geometric_correct": False}],
        ]
    )

    assert summary["query_count"] == 2
    assert summary["top1_correct"] == 0
    assert summary["topk_available"] == 1
    assert summary["oracle_gap"] == 1
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_sparse_distilled_rerank.py
```

Expected: import failure for `loc_gs.sparse.rerank`.

- [ ] **Step 3: Implement reranker**

Implement stable sorting:

```python
@dataclass(frozen=True)
class CandidateRerankConfig:
    prefix_fraction: float = 0.0
    solver_weight: float = 1.0
    native_weight: float = 1.0


def rerank_candidate_rows(rows: Sequence[Mapping[str, Any]], cfg: CandidateRerankConfig) -> list[dict[str, Any]]:
    ordered = sorted(enumerate(rows), key=lambda item: (-float(item[1].get("native_score", 0.0)), item[0]))
    prefix_count = int(round(len(ordered) * max(0.0, min(1.0, cfg.prefix_fraction))))
    prefix = [dict(row) for _idx, row in ordered[:prefix_count]]
    tail = sorted(
        ordered[prefix_count:],
        key=lambda item: (
            -(cfg.native_weight * float(item[1].get("native_score", 0.0)) + cfg.solver_weight * float(item[1].get("solver_score", 0.0))),
            item[0],
        ),
    )
    return prefix + [dict(row) for _idx, row in tail]
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_sparse_distilled_rerank.py
```

Expected: all tests pass.

## Task 4: Sparse-Only New-Mainline CLI

**Files:**
- Create: `loc_gs/scripts/eval_sparse_distilled_cambridge.py`
- Create: `tests/test_eval_sparse_distilled_cambridge.py`

- [ ] **Step 1: Write failing CLI tests**

Add tests for split rejection, dense-inference rejection, and manifest output:

```python
import json

import pytest

from loc_gs.scripts.eval_sparse_distilled_cambridge import build_argparser, build_sparse_distilled_manifest


def test_sparse_distilled_cli_rejects_dense_inference_flag():
    parser = build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--scene", "GreatCourt", "--enable_dense_inference"])


def test_sparse_distilled_manifest_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_distilled_manifest(scene="GreatCourt", split_name="test", command=["cmd"])


def test_sparse_distilled_manifest_records_sparse_only_teacher_boundary():
    manifest = build_sparse_distilled_manifest(
        scene="GreatCourt",
        split_name="train_dev",
        command=["python", "-m", "loc_gs.scripts.eval_sparse_distilled_cambridge"],
        dense_teacher_enabled=False,
    )

    assert manifest["method"] == "sparse_dense_distilled_internal"
    assert manifest["inference_stage"] == "sparse_only"
    assert manifest["dense_teacher_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    json.dumps(manifest)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_eval_sparse_distilled_cambridge.py
```

Expected: import failure for the new script.

- [ ] **Step 3: Implement dry-run CLI**

Implement:

- `build_argparser()` with required `--scene`, default `--split_name train_dev`,
  `--dry_run`, and no accepted `--enable_dense_inference`;
- `build_sparse_distilled_manifest(scene: str, split_name: str, command: list[str], dense_teacher_enabled: bool = False)`;
- `main()` that writes `manifest.json` when `--output_dir` is supplied and
  exits without Cambridge data when `--dry_run` is true.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_eval_sparse_distilled_cambridge.py
```

Expected: all tests pass.

## Task 5: Mainline Audit CLI And Targeted Verification

**Files:**
- Create: `loc_gs/scripts/audit_internal_mainline.py`
- Extend: `tests/test_internal_mainline_audit.py`

- [ ] **Step 1: Write failing CLI test**

Add a test that calls the audit function on the new mainline files and expects
no forbidden runtime dependencies.

```python
from pathlib import Path

from loc_gs.scripts.audit_internal_mainline import internal_mainline_source_paths, run_internal_mainline_audit


def test_internal_mainline_audit_cli_source_set_is_clean():
    paths = internal_mainline_source_paths(Path("loc_gs"))
    assert any(str(path).endswith("eval_sparse_distilled_cambridge.py") for path in paths)

    assert run_internal_mainline_audit(paths) == {"checked_file_count": len(paths), "forbidden_hit_count": 0}
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_internal_mainline_audit.py::test_internal_mainline_audit_cli_source_set_is_clean
```

Expected: import failure for `loc_gs.scripts.audit_internal_mainline`.

- [ ] **Step 3: Implement CLI wrapper**

Implement `internal_mainline_source_paths(root)` to include only new-mainline
files:

```python
INTERNAL_MAINLINE_RELATIVE_PATHS = (
    "sparse/audit.py",
    "sparse/correspondences.py",
    "sparse/rerank.py",
    "teacher/labels.py",
    "scripts/eval_sparse_distilled_cambridge.py",
)
```

`run_internal_mainline_audit(paths)` calls
`assert_internal_mainline_sources(paths)` and returns counts.

- [ ] **Step 4: Run targeted suite**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_internal_mainline_audit.py \
  tests/test_sparse_distilled_schemas.py \
  tests/test_sparse_distilled_rerank.py \
  tests/test_eval_sparse_distilled_cambridge.py
```

Expected: all tests pass.

- [ ] **Step 5: Run diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Status may still include pre-existing dirty
files from the inherited branch; new files should be easy to identify.
