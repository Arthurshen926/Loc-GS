# Sparse Solver-Feedback Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete initial ULF/STDLoc-style sparse localization system, run self-map sparse PnP through that complete system, then use Feedback v4 in an outer solver-feedback loop to improve descriptor aggregation/view selection, the scene-specific detector, and query-conditioned activation.

**Architecture:** Stage 1 uses ULF K.C./visibility/mask sampling. Stage 2 uses ULF geometry-weighted multi-view descriptor aggregation. Stage 3 trains an initial STDLoc-style sampled-landmark projection detector online. Stage 4 runs train/self-map sparse PnP with the initial sampled landmarks, descriptors, and detector. Stage 5 alternates solver-feedback updates to aggregation/view selection, detector residuals, and activation. Stage 6 freezes the recipe for disjoint train-dev sparse-only evaluation. Offline descriptor-fusion/feature-log rewriting and SuperPoint-teacher distillation are diagnostic only. GreatCourt disjoint train-dev sparse-only is the first large-scene gate; five-scene expansion is blocked until that gate passes.

**Tech Stack:** Python, PyTorch, ULF-Loc sparse pipeline, OpenCV/PoseLib PnP traces, SuperPoint features, Cambridge train/self-map artifacts, `/root/miniconda3/envs/cybersim_agent/bin/python`.

---

## File Structure

- Create `loc_gs/feedback/sparse_feedback_v4.py`: schema validation, label assignment, query utility, protected/harmful attribution.
- Create `loc_gs/scripts/build_sparse_feedback_v4.py`: CLI converting sparse trace payloads into Feedback v4 artifacts with audit files.
- Modify `loc_gs/scripts/eval_ulfloc_sparse_only.py`: trace export must include query-level pose outcome and descriptor margin.
- Modify `/root/ULF-Loc/ulfloc.py`: sparse match trace must expose enough raw correspondence data for Feedback v4.
- Keep offline descriptor aggregation/fusion scripts diagnostic only; the main launcher must not call feature-log rewriting.
- Modify `loc_gs/training/ulfloc_scene_detector.py`: support direct heatmap detector training with sampled-landmark visibility + bounded solver residual targets.
- Modify `loc_gs/localization/ulfloc_scene_detector.py`: ensure direct heatmap mode is the main path and rerank mode is diagnostic only.
- Create `loc_gs/training/query_landmark_activation_v2.py`: query-token/landmark-token overlap activation model.
- Create `loc_gs/scripts/train_query_landmark_activation_v2.py`: train activation v2 from Feedback v4 and geometry visibility masks.
- Modify `loc_gs/scripts/eval_ulfloc_sparse_only.py`: add activation v2 checkpoint/config audit and direct-detector audit.
- Add focused tests under `tests/`.

## Hard Constraints

- Main method must not call `fuse_landmark_descriptors`, `fuse_landmark_descriptors_contrastive`, `build_ulfloc_native_fusion_feedback`, or `build_ulfloc_native_feature_fusion_log`.
- Main detector training must not use SuperPoint teacher distillation unless an explicit diagnostic flag is passed.
- Main detector mode must be `grid`/direct heatmap, not `rerank_superpoint`.
- Main activation model must not use only `mean_descriptor` query features.
- The first self-map trace used for Feedback v4 must include the initial scene-specific detector.
- Static detector target caches and activation caches are smoke/debug artifacts, not the main training strategy.
- Feedback artifacts with `split_name=test` or `official_test_used=true` must raise `ValueError`.
- GreatCourt large-scene gate must run before any five-scene train-dev expansion.

### Task 1: Feedback v4 Schema And Labels

**Files:**
- Create: `loc_gs/feedback/sparse_feedback_v4.py`
- Test: `tests/test_sparse_feedback_v4.py`

- [ ] **Step 1: Write failing tests for test-split rejection and strong labels**

Create `tests/test_sparse_feedback_v4.py`:

```python
import pytest

from loc_gs.feedback.sparse_feedback_v4 import build_sparse_feedback_v4


def test_feedback_v4_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_feedback_v4({"split_name": "test", "queries": [], "correspondences": []})


def test_feedback_v4_assigns_positive_harmful_and_protected_labels():
    payload = {
        "split_name": "selfmap_train",
        "queries": [
            {
                "query_id": "q_good",
                "sparse_te_cm": 3.0,
                "sparse_re_deg": 0.1,
                "pose_success": True,
                "inlier_count": 80,
                "match_count": 120,
                "inlier_image_cell_count": 12,
                "inlier_depth_bin_count": 4,
            },
            {
                "query_id": "q_bad",
                "sparse_te_cm": 80.0,
                "sparse_re_deg": 2.0,
                "pose_success": False,
                "candidate_minus_baseline_te_cm": 40.0,
                "inlier_count": 5,
                "match_count": 100,
            },
        ],
        "correspondences": [
            {
                "query_id": "q_good",
                "gaussian_id": 10,
                "sampled_row": 0,
                "pnp_inlier": True,
                "reprojection_error_px": 1.0,
                "descriptor_score": 0.9,
                "descriptor_margin": 0.3,
                "keypoint_xy": [10.0, 20.0],
                "query_descriptor": [1.0, 0.0],
                "landmark_descriptor": [1.0, 0.0],
            },
            {
                "query_id": "q_bad",
                "gaussian_id": 11,
                "sampled_row": 1,
                "pnp_inlier": False,
                "reprojection_error_px": 24.0,
                "descriptor_score": 0.95,
                "descriptor_margin": 0.02,
                "keypoint_xy": [50.0, 60.0],
                "query_descriptor": [0.0, 1.0],
                "landmark_descriptor": [1.0, 0.0],
            },
        ],
    }
    feedback = build_sparse_feedback_v4(payload, protected_te_cm=10.0, hard_te_cm=20.0, harmful_regression_cm=20.0)
    labels = {row["gaussian_id"]: row["label_role"] for row in feedback["correspondences"]}
    assert labels[10] == "protected_support"
    assert labels[11] == "harmful_negative"
    assert feedback["query_index"]["q_good"]["protected_good_query"] is True
    assert feedback["query_index"]["q_bad"]["hard_query"] is True
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_sparse_feedback_v4.py -q
```

Expected: import failure because `sparse_feedback_v4.py` does not exist.

- [ ] **Step 3: Implement Feedback v4 builder**

Create `loc_gs/feedback/sparse_feedback_v4.py` with:

```python
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "sparse_solver_feedback_v4"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _split_name(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        raise ValueError("split_name is required")
    audit = payload.get("split_audit", {})
    if split.lower() == "test" or (isinstance(audit, Mapping) and bool(audit.get("test_split_used", False))):
        raise ValueError("test split is not allowed for sparse solver feedback v4")
    return split


def _query_state(row: Mapping[str, Any], *, protected_te_cm: float, hard_te_cm: float) -> dict[str, Any]:
    te = _finite_float(row.get("sparse_te_cm"), 1e9)
    success = _bool(row.get("pose_success")) and te <= float(hard_te_cm)
    protected = success and te <= float(protected_te_cm)
    hard = (not success) or te > float(hard_te_cm)
    inliers = max(0.0, _finite_float(row.get("inlier_count"), 0.0))
    matches = max(1.0, _finite_float(row.get("match_count"), 1.0))
    return {
        **dict(row),
        "pose_success": bool(success),
        "protected_good_query": bool(protected),
        "hard_query": bool(hard),
        "inlier_ratio": float(inliers / matches),
    }


def _label_correspondence(
    row: Mapping[str, Any],
    query: Mapping[str, Any],
    *,
    low_reprojection_px: float,
    high_score_threshold: float,
    harmful_regression_cm: float,
) -> str:
    inlier = _bool(row.get("pnp_inlier"))
    reproj = _finite_float(row.get("reprojection_error_px"), 1e9)
    score = _finite_float(row.get("descriptor_score"), 0.0)
    regression = _finite_float(query.get("candidate_minus_baseline_te_cm"), 0.0)
    if inlier and bool(query.get("protected_good_query", False)) and reproj <= float(low_reprojection_px):
        return "protected_support"
    if inlier and bool(query.get("pose_success", False)) and reproj <= float(low_reprojection_px):
        return "positive_inlier"
    if (not inlier) and bool(query.get("hard_query", False)) and score >= float(high_score_threshold):
        return "harmful_negative"
    if (not inlier) and regression >= float(harmful_regression_cm) and score >= float(high_score_threshold):
        return "harmful_negative"
    return "neutral_outlier"


def build_sparse_feedback_v4(
    payload: Mapping[str, Any],
    *,
    protected_te_cm: float = 10.0,
    hard_te_cm: float = 20.0,
    low_reprojection_px: float = 4.0,
    high_score_threshold: float = 0.8,
    harmful_regression_cm: float = 20.0,
) -> dict[str, Any]:
    split = _split_name(payload)
    queries = [dict(row) for row in payload.get("queries", []) if isinstance(row, Mapping)]
    query_index = {
        str(row.get("query_id")): _query_state(row, protected_te_cm=protected_te_cm, hard_te_cm=hard_te_cm)
        for row in queries
    }
    correspondences = []
    counts = {"protected_support": 0, "positive_inlier": 0, "harmful_negative": 0, "neutral_outlier": 0}
    for raw in payload.get("correspondences", []):
        if not isinstance(raw, Mapping):
            continue
        query_id = str(raw.get("query_id", ""))
        query = query_index.get(query_id, {})
        label = _label_correspondence(
            raw,
            query,
            low_reprojection_px=low_reprojection_px,
            high_score_threshold=high_score_threshold,
            harmful_regression_cm=harmful_regression_cm,
        )
        row = dict(raw)
        row["label_role"] = label
        row["query_protected_good"] = bool(query.get("protected_good_query", False))
        row["query_hard"] = bool(query.get("hard_query", False))
        correspondences.append(row)
        counts[label] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "split_name": split,
        "query_index": query_index,
        "correspondences": correspondences,
        "metrics": {
            "query_count": len(query_index),
            "correspondence_count": len(correspondences),
            **{f"{key}_count": value for key, value in counts.items()},
        },
        "split_audit": {
            "split_name": split,
            "test_split_used": False,
            "official_test_used": False,
            "role": "sparse_solver_feedback_v4",
        },
    }
```

- [ ] **Step 4: Run test and verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_sparse_feedback_v4.py -q
```

Expected: 2 tests pass.

### Task 2: Export Real Query-Level Sparse Trace Fields

**Files:**
- Modify: `loc_gs/scripts/eval_ulfloc_sparse_only.py`
- Modify: `/root/ULF-Loc/ulfloc.py`
- Test: `tests/test_eval_ulfloc_sparse_only.py`

- [ ] **Step 1: Add failing test for query-level trace fields**

Extend `tests/test_eval_ulfloc_sparse_only.py` with a unit test around `serialize_sparse_match_attributions` that asserts exported rows include `descriptor_margin`, and add a helper test for a new `build_sparse_feedback_trace_payload` function that includes query `sparse_te_cm`, `sparse_re_deg`, `pose_success`, `inlier_count`, and `match_count`.

- [ ] **Step 2: Run focused test and verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_eval_ulfloc_sparse_only.py -q
```

Expected: failure because query-level builder and descriptor margin are missing.

- [ ] **Step 3: Implement trace payload builder**

Add a pure helper in `eval_ulfloc_sparse_only.py`:

```python
def build_sparse_feedback_trace_payload(*, scene, split_name, query_rows, correspondence_rows, query_features, query_match_descriptors, split_audit):
    return {
        "schema_version": "ulfloc_sparse_feedback_trace_v2",
        "scene": scene,
        "split_name": split_name,
        "queries": query_rows,
        "correspondences": correspondence_rows,
        "query_features": query_features,
        "query_match_descriptors": query_match_descriptors,
        "record_count": len(correspondence_rows),
        "query_count": len(query_rows),
        "split_audit": split_audit,
    }
```

In the evaluation loop, append a query row for every evaluated camera containing TE/RE, success, inlier count, match count, inlier coverage fields when available. Update `serialize_sparse_match_attributions` to compute descriptor margin from top-2 match scores if ULF exposes them; otherwise write `descriptor_margin=0.0` and mark `descriptor_margin_source="missing_top2_default_zero"`.

- [ ] **Step 4: Run tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_eval_ulfloc_sparse_only.py -q
```

Expected: pass.

### Task 3: Feedback v4 CLI

**Files:**
- Create: `loc_gs/scripts/build_sparse_feedback_v4.py`
- Test: `tests/test_build_sparse_feedback_v4_cli.py`

- [ ] **Step 1: Write failing CLI test**

Create a test that writes a small trace payload, runs:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_sparse_feedback_v4 --trace TRACE.pt --output_dir OUT
```

and asserts `feedback_v4.pt`, `metrics_summary.json`, `manifest.json`, `split_audit.json`, `command.txt`, and `git_status.txt` exist.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_build_sparse_feedback_v4_cli.py -q
```

Expected: module missing.

- [ ] **Step 3: Implement CLI**

Load `.pt` or `.json`, call `build_sparse_feedback_v4`, write audit files, reject test split.

- [ ] **Step 4: Run and verify GREEN**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_build_sparse_feedback_v4_cli.py -q
```

Expected: pass.

### Task 4: Aggregation-Stage Descriptor Training

**Files:**
- Create: `loc_gs/training/ulfloc_aggregation_samples.py`
- Create: `loc_gs/training/ulfloc_descriptor_aggregation.py`
- Create: `loc_gs/scripts/train_ulfloc_descriptor_aggregation.py`
- Test: `tests/test_ulfloc_descriptor_aggregation.py`
- Test: `tests/test_train_ulfloc_descriptor_aggregation_cli.py`

- [ ] **Step 1: Write failing tests for sample labels**

Create tests asserting Feedback v4 rows become aggregation samples:

```python
positive_inlier -> positive view sample
harmful_negative -> negative view sample
protected_support -> protected positive sample
neutral_outlier -> ignored by default
```

- [ ] **Step 2: Write failing tests for aggregation loss**

Use synthetic descriptors:

```text
native=[1,0]
positive=[1,0]
negative=[0,1]
```

Assert training one small aggregation step increases positive cosine and decreases negative cosine while respecting a native cosine floor.

- [ ] **Step 3: Implement aggregation sample builder**

Create functions:

```python
build_aggregation_samples(feedback_v4, *, include_render_aug=True)
group_samples_by_landmark(samples)
```

The builder must reject test split and store `label_role`, `source_role`, `view_id`, `descriptor`, `gaussian_id`, `sampled_row`.

- [ ] **Step 4: Implement aggregation model**

Create:

```python
class LandmarkAggregationModel(nn.Module):
    view_score = MLP([descriptor, label_features, reliability_features])
```

For the first implementation, keep it deterministic and small: learn scalar view logits and aggregate weighted descriptors. Loss combines positive attraction, harmful-negative repulsion, protected retention, and native anchor.

- [ ] **Step 5: Implement CLI export**

The CLI consumes:

- source ULF log dir;
- Feedback v4 artifact;
- optional render-aug observations;
- output log dir.

It writes a ULF-compatible log dir while recording that descriptors were rebuilt from aggregation samples, not post-hoc mean-shift.

- [ ] **Step 6: Run aggregation tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_ulfloc_descriptor_aggregation.py tests/test_train_ulfloc_descriptor_aggregation_cli.py -q
```

Expected: pass.

### Task 5: Direct Scene-Specific Detector

**Files:**
- Modify: `loc_gs/training/ulfloc_scene_detector.py`
- Modify: `loc_gs/localization/ulfloc_scene_detector.py`
- Modify: `loc_gs/scripts/train_ulfloc_scene_detector.py`
- Modify: `/root/ULF-Loc/ulfloc.py`
- Test: `tests/test_train_ulfloc_scene_detector.py`
- Test: `tests/test_eval_ulfloc_sparse_only.py`

- [ ] **Step 1: Write failing test that main mode is direct heatmap**

Add test asserting a checkpoint configured with `scene_specific_detector.mode=grid` uses direct detector keypoints and metrics record `scene_detector_mode="direct_heatmap"`. Add a separate test that `rerank_superpoint` metrics are marked `diagnostic_only=true`.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_train_ulfloc_scene_detector.py tests/test_eval_ulfloc_sparse_only.py -q
```

- [ ] **Step 3: Update detector target/loss composition**

Train with four explicit terms:

```text
sp_teacher_loss
visibility_loss
solver_positive_residual_loss
solver_negative_suppression_loss
```

Do not multiply all positive targets by solver validity.

- [ ] **Step 4: Force main eval config to direct heatmap**

`eval_ulfloc_sparse_only.py` should set:

```python
config["scene_specific_detector"]["mode"] = "grid"
```

for main direct-detector recipes. Keep rerank available only if CLI explicitly requests diagnostic mode.

- [ ] **Step 5: Run tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_train_ulfloc_scene_detector.py tests/test_eval_ulfloc_sparse_only.py -q
```

Expected: pass.

### Task 6: Query-Conditioned Landmark Activation v2

**Files:**
- Create: `loc_gs/training/query_landmark_activation_v2.py`
- Create: `loc_gs/scripts/train_query_landmark_activation_v2.py`
- Modify: `/root/ULF-Loc/ulfloc.py`
- Test: `tests/test_query_landmark_activation_v2.py`
- Test: `tests/test_train_query_landmark_activation_v2_cli.py`

- [ ] **Step 1: Write failing architecture test**

Create a test that feeds:

- query tokens `[num_keypoints, descriptor_dim + xy + score]`;
- landmark tokens `[num_landmarks, descriptor_dim + xyz + priors]`;
- labels `[num_landmarks]`.

Assert the model returns `[num_landmarks]` logits and does not accept a single mean query descriptor as the only input.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_query_landmark_activation_v2.py -q
```

- [ ] **Step 3: Implement activation v2 model**

Create:

```python
class QueryLandmarkActivationV2(nn.Module):
    query_encoder
    landmark_encoder
    cross_attention
    landmark_head
```

Use bounded top-k attention for memory control. Output one logit per landmark.

- [ ] **Step 4: Implement labels and loss**

Loss:

```text
BCE geometry visibility
+ BCE solver-positive
+ BCE harmful-negative suppression
+ protected-support retention
+ coverage regularizer
```

- [ ] **Step 5: Implement CLI**

The CLI consumes Feedback v4, geometry visibility masks, source ULF log dir, and query token caches. It writes checkpoint, metrics, split audit, and manifest.

- [ ] **Step 6: Integrate inference hook**

Modify ULF activation hook to support `activation_model_type="v2_tokens"`. The old `mean_descriptor` hook must be marked diagnostic and disabled in main recipes.

- [ ] **Step 7: Run tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_query_landmark_activation_v2.py tests/test_train_query_landmark_activation_v2_cli.py tests/test_eval_ulfloc_sparse_only.py -q
```

Expected: pass.

### Task 7: GreatCourt Sparse Gate

**Files:**
- Create: `docs/ulfloc_sparse_solver_feedback_reset_status_20260609.md`

- [ ] **Step 1: Run baseline sparse trace on GreatCourt self-map**

Use the existing ULF sparse eval wrapper with `--dump_sparse_match_attributions`. The split must be self-map/train, not official test.

- [ ] **Step 2: Build Feedback v4**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_sparse_feedback_v4 \
  --trace SELF_MAP_TRACE.pt \
  --output_dir output/reports/ulfloc_feedback_v4_greatcourt_selfmap_20260609
```

- [ ] **Step 3: Verify native descriptor path**

The main recipe keeps the native ULF descriptor log. Do not run offline fusion/feature-log rewriting in this gate. Descriptor observation audits may be generated only as diagnostics.

- [ ] **Step 4: Train direct scene detector**

Use sampled-landmark projection visibility + Feedback v4 residuals. SuperPoint teacher must be disabled unless this is explicitly marked as a diagnostic ablation. Main mode must be direct heatmap.

- [ ] **Step 5: Train activation v2**

Use tokenized query/landmark inputs and Feedback v4 labels.

- [ ] **Step 6: Eval candidates separately**

Run disjoint train-dev sparse-only:

```text
baseline geomw
direct detector only
activation v2 only
direct detector + activation v2
```

- [ ] **Step 7: Apply gate**

Pass only if:

```text
median TE <= baseline median TE
R@10 or R@5 improves
the other recall does not decrease
activation selected count is meaningfully below full landmark count
```

- [ ] **Step 8: Write status doc**

Write exact metrics, artifact paths, split audit status, and module verdicts to:

```text
docs/ulfloc_sparse_solver_feedback_reset_status_20260609.md
```

### Task 8: Five-Scene Expansion Only After Gate

**Files:**
- Update: `docs/ulfloc_sparse_solver_feedback_reset_status_20260609.md`

- [ ] **Step 1: Check GreatCourt gate status**

If the gate failed, stop. Do not run five scenes.

- [ ] **Step 2: Run train-dev sparse-only on five scenes**

Run only the passing fixed recipe.

- [ ] **Step 3: Summarize five-scene table**

Report median TE/RE, R@10/5, R@5/5, selected landmark count, and module audit.

- [ ] **Step 4: Decide frozen recipe**

Only after five-scene train-dev is non-negative can a frozen recipe be considered for official Cambridge test.

## Plan Self-Review

- Spec coverage: Feedback v4, direct detector, activation v2, and GreatCourt gate all have tasks.
- Placeholder scan: no task uses open-ended implementation placeholders; each task defines files, tests, and expected commands.
- Type consistency: Feedback v4 uses `queries` and `correspondences`; downstream tasks consume that schema.
- Scope control: dense stage is excluded; official test is excluded until a frozen recipe passes train-dev.
