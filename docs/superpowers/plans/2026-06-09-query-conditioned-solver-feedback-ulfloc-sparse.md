# DEPRECATED: Query-Conditioned Solver Feedback ULF Sparse Implementation Plan

This plan is retained only as historical context. It is not the current sparse
mainline.

Current sparse mainline is:

```text
fixed high-quality ULF sampled landmarks
+ native ULF landmark descriptors
+ STDLoc-style sampled-landmark projection detector initialization
+ Feedback v4 solver residuals
+ GreatCourt disjoint train-dev sparse-only gate
```

The current mainline must not call offline descriptor fusion / feature-log
rewriting scripts, and must not use SuperPoint-teacher distillation unless an
experiment is explicitly labeled diagnostic. Use
`docs/superpowers/specs/2026-06-09-sparse-solver-feedback-reset-design.md` and
`docs/superpowers/plans/2026-06-09-sparse-solver-feedback-reset.md` instead.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sparse-only ULF-Loc mainline where self-localization solver feedback trains map feature fusion, STDLoc-style scene-specific detection, and query-conditioned landmark activation on a fixed high-quality ULF landmark set.

**Architecture:** Keep the best sampled ULF landmark set fixed and stop using Full-Gaussian SparseSet as the main path. Convert sparse PnP traces into positive/negative impact attribution, then consume that attribution in three places: map-side landmark descriptor fusion, query-side scene-specific detector training, and pre-matching query-conditioned 3D landmark activation. Inference remains one path: detector, active landmark subset, descriptor matching, OpenCV/PoseLib PnP.

**Tech Stack:** Python, PyTorch, ULF-Loc sparse pipeline, Loc-GS feedback exporters, SuperPoint features, Cambridge train/self-map split artifacts. Use `/root/miniconda3/envs/cybersim_agent/bin/python`.

---

## Scope And Non-Goals

This plan intentionally replaces the previous sparse mainline shape:

- Do not use `full_gaussian_sparse_set` guard2 as the main result.
- Do not use correspondence supervision as a post-matching reranker main path.
- Do not tune on Cambridge official test.
- Do not change official metric definitions.

The fixed ULF landmark set is the current high-precision sampled set used by the best sparse baseline, normally the ULF `keypoints_sampled_idx.pkl` and corresponding log dir for each scene.

## File Structure

- Create `loc_gs/feedback/solver_feedback_impact.py`: split-safe positive/negative attribution from sparse PnP traces.
- Create `loc_gs/scripts/build_solver_feedback_impact.py`: CLI that writes `impact_attribution.pt/json`, audit files, and manifest.
- Create `loc_gs/training/ulfloc_visibility_teacher.py`: STDLoc-style projected sampled-landmark visibility teacher.
- Create `loc_gs/scripts/build_ulfloc_visibility_teacher.py`: CLI that projects fixed sampled landmarks into train images and writes detector teacher targets.
- Modify `loc_gs/training/ulfloc_scene_detector.py`: train with teacher positive heatmap plus negative suppression heatmap.
- Create `loc_gs/stdloc_native/solver_feedback_feature_fusion_v2.py`: positive/negative view selection and contrastive anchored landmark descriptor fusion.
- Create `loc_gs/scripts/build_ulfloc_solver_feedback_feature_log.py`: export ULF log with fused `keypoints_features.pkl` while preserving sampled ids.
- Create `loc_gs/localization/query_landmark_activation.py`: model, data classes, and top-N active landmark selection.
- Create `loc_gs/scripts/build_ulfloc_landmark_activation_cache.py`: build training examples from feedback traces and fixed ULF landmarks.
- Create `loc_gs/scripts/train_ulfloc_landmark_activation.py`: train query-conditioned activation network.
- Modify `/root/ULF-Loc/ulfloc.py`: add pre-matching active landmark subset hook, separate from `scene_matcher`.
- Modify `loc_gs/scripts/eval_ulfloc_sparse_only.py`: pass activation checkpoint/config to ULF-Loc and audit activation use.
- Add tests listed per task below.

## Task 1: Solver Feedback Impact Attribution

**Files:**
- Create: `loc_gs/feedback/solver_feedback_impact.py`
- Create: `loc_gs/scripts/build_solver_feedback_impact.py`
- Test: `tests/test_solver_feedback_impact.py`
- Test: `tests/test_build_solver_feedback_impact_cli.py`

- [ ] **Step 1: Write synthetic attribution tests**

Create `tests/test_solver_feedback_impact.py` with:

```python
from loc_gs.feedback.solver_feedback_impact import build_solver_feedback_impact


def test_rejects_test_split():
    payload = {"split_name": "test", "records": []}
    try:
        build_solver_feedback_impact(payload)
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("test split feedback must be rejected")


def test_positive_and_negative_landmark_impact_are_separated():
    payload = {
        "split_name": "train_selfmap",
        "records": [
            {
                "query_id": "q1",
                "image_id": "im1",
                "gaussian_id": 10,
                "landmark_id": 0,
                "label": 1,
                "pnp_inlier": True,
                "pose_success": True,
                "reprojection_error_px": 1.0,
                "descriptor_margin": 0.4,
                "keypoint_xy": [20.0, 30.0],
                "query_xy_norm": [0.1, 0.2],
                "source_role": "baseline_trace",
            },
            {
                "query_id": "q1",
                "image_id": "im1",
                "gaussian_id": 11,
                "landmark_id": 1,
                "label": 0,
                "pnp_inlier": False,
                "pose_success": False,
                "reprojection_error_px": 24.0,
                "descriptor_score": 0.92,
                "descriptor_margin": 0.02,
                "query_regression_delta_cm": 35.0,
                "keypoint_xy": [80.0, 90.0],
                "query_xy_norm": [0.4, 0.5],
                "source_role": "candidate_trace",
            },
        ],
    }
    impact = build_solver_feedback_impact(payload)
    assert impact["split_name"] == "train_selfmap"
    assert impact["landmark_positive"][10]["support"] > 0.0
    assert impact["landmark_negative"][11]["risk"] > 0.0
    assert impact["view_positive"][("10", "im1")]["weight"] > 0.0
    assert impact["view_negative"][("11", "im1")]["weight"] > 0.0
    assert impact["detector_negative"]["q1"][0]["gaussian_id"] == 11
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_solver_feedback_impact.py -q
```

Expected: fails with `ModuleNotFoundError` or missing `build_solver_feedback_impact`.

- [ ] **Step 3: Implement attribution builder**

Create `loc_gs/feedback/solver_feedback_impact.py`:

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _positive_weight(row: Mapping[str, Any]) -> float:
    if int(row.get("label", 0)) != 1 or not bool(row.get("pnp_inlier", False)):
        return 0.0
    reproj = max(0.0, _f(row.get("reprojection_error_px"), 999.0))
    margin = max(0.0, _f(row.get("descriptor_margin"), 0.0))
    pose = 1.0 if bool(row.get("pose_success", False)) else 0.5
    return pose * (1.0 / (1.0 + reproj / 4.0)) * (0.5 + min(0.5, margin))


def _negative_weight(row: Mapping[str, Any]) -> float:
    if int(row.get("label", 0)) != 0 or bool(row.get("pnp_inlier", False)):
        return 0.0
    reproj = max(0.0, _f(row.get("reprojection_error_px"), 0.0))
    score = max(0.0, _f(row.get("descriptor_score"), 0.0))
    regression = max(0.0, _f(row.get("query_regression_delta_cm"), 0.0))
    margin = max(0.0, _f(row.get("descriptor_margin"), 0.0))
    low_margin = 1.0 / (1.0 + margin / 0.15)
    return (0.25 + score) * min(2.0, reproj / 8.0) * (1.0 + min(2.0, regression / 20.0)) * low_margin


def build_solver_feedback_impact(payload: Mapping[str, Any]) -> dict[str, Any]:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        raise ValueError("split_name is required")
    if split.lower() == "test":
        raise ValueError("refusing to build solver feedback impact from test split")
    records = [row for row in payload.get("records", []) if isinstance(row, Mapping)]
    landmark_positive: dict[int, dict[str, float]] = defaultdict(lambda: {"support": 0.0, "count": 0.0})
    landmark_negative: dict[int, dict[str, float]] = defaultdict(lambda: {"risk": 0.0, "count": 0.0})
    view_positive: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "count": 0.0})
    view_negative: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "count": 0.0})
    detector_positive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detector_negative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        gid = int(row.get("gaussian_id", -1))
        if gid < 0:
            continue
        query_id = str(row.get("query_id", ""))
        image_id = str(row.get("image_id", ""))
        pos = _positive_weight(row)
        neg = _negative_weight(row)
        if pos > 0.0:
            landmark_positive[gid]["support"] += float(pos)
            landmark_positive[gid]["count"] += 1.0
            view_positive[(str(gid), image_id)]["weight"] += float(pos)
            view_positive[(str(gid), image_id)]["count"] += 1.0
            detector_positive[query_id].append({"gaussian_id": gid, "keypoint_xy": row.get("keypoint_xy"), "weight": float(pos)})
        if neg > 0.0:
            landmark_negative[gid]["risk"] += float(neg)
            landmark_negative[gid]["count"] += 1.0
            view_negative[(str(gid), image_id)]["weight"] += float(neg)
            view_negative[(str(gid), image_id)]["count"] += 1.0
            detector_negative[query_id].append({"gaussian_id": gid, "keypoint_xy": row.get("keypoint_xy"), "weight": float(neg)})
    return {
        "schema_version": "solver_feedback_impact_v1",
        "split_name": split,
        "record_count": len(records),
        "landmark_positive": dict(landmark_positive),
        "landmark_negative": dict(landmark_negative),
        "view_positive": dict(view_positive),
        "view_negative": dict(view_negative),
        "detector_positive": dict(detector_positive),
        "detector_negative": dict(detector_negative),
        "metadata": {
            "positive_landmark_count": len(landmark_positive),
            "negative_landmark_count": len(landmark_negative),
        },
    }
```

- [ ] **Step 4: Add CLI with audit files**

Create `loc_gs/scripts/build_solver_feedback_impact.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from loc_gs.feedback.solver_feedback_impact import build_solver_feedback_impact


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()


def _git_status(root: Path) -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)


def _load_payload(path: Path) -> dict:
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"feedback payload must be a dict: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    impact = build_solver_feedback_impact(_load_payload(Path(args.feedback)))
    torch.save(impact, out / "impact_attribution.pt")
    (out / "impact_attribution.json").write_text(json.dumps(impact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "solver_feedback_impact_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "command": sys.argv,
        "feedback": str(Path(args.feedback).resolve()),
        "outputs": {
            "impact_attribution_pt": str(out / "impact_attribution.pt"),
            "impact_attribution_json": str(out / "impact_attribution.json"),
        },
        "metrics": impact["metadata"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "command.txt").write_text(" ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8")
    (out / "git_status.txt").write_text(_git_status(repo), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_solver_feedback_impact.py -q
```

Expected: passes.

## Task 2: STDLoc-Style ULF Visibility Teacher

**Files:**
- Create: `loc_gs/training/ulfloc_visibility_teacher.py`
- Create: `loc_gs/scripts/build_ulfloc_visibility_teacher.py`
- Test: `tests/test_ulfloc_visibility_teacher.py`

- [ ] **Step 1: Write teacher target tests**

Create `tests/test_ulfloc_visibility_teacher.py`:

```python
import torch

from loc_gs.training.ulfloc_visibility_teacher import build_visibility_teacher_targets


def test_visibility_teacher_keeps_stable_visible_points():
    projections = {
        "im1.png": [
            {"gaussian_id": 1, "keypoint_yx": [10.0, 20.0], "visible": True, "mask_valid": True},
            {"gaussian_id": 2, "keypoint_yx": [30.0, 40.0], "visible": True, "mask_valid": False},
        ]
    }
    targets = build_visibility_teacher_targets(projections, height=64, width=80)
    entry = targets["im1.png"]
    assert entry["positive_count"] == 1
    assert entry["gaussian_ids"].tolist() == [1]
    assert torch.allclose(entry["support_weights"], torch.ones(1))
```

- [ ] **Step 2: Implement compact target builder**

Create `loc_gs/training/ulfloc_visibility_teacher.py`:

```python
from __future__ import annotations

from typing import Any, Mapping

import torch


def build_visibility_teacher_targets(
    projections: Mapping[str, list[Mapping[str, Any]]],
    *,
    height: int,
    width: int,
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for image_id, rows in projections.items():
        gids: list[int] = []
        yx: list[list[float]] = []
        weights: list[float] = []
        for row in rows:
            if not bool(row.get("visible", False)):
                continue
            if not bool(row.get("mask_valid", True)):
                continue
            point = row.get("keypoint_yx")
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            y = float(point[0])
            x = float(point[1])
            if not (0.0 <= y < float(height) and 0.0 <= x < float(width)):
                continue
            gids.append(int(row["gaussian_id"]))
            yx.append([y, x])
            weights.append(float(row.get("weight", 1.0)))
        targets[str(image_id)] = {
            "gaussian_ids": torch.tensor(gids, dtype=torch.long),
            "keypoint_yx": torch.tensor(yx, dtype=torch.float32).reshape(-1, 2),
            "support_weights": torch.tensor(weights, dtype=torch.float32),
            "positive_count": len(gids),
            "height": int(height),
            "width": int(width),
            "target_role": "stdloc_visibility_teacher",
        }
    return targets
```

- [ ] **Step 3: Add projection CLI**

Create `loc_gs/scripts/build_ulfloc_visibility_teacher.py` with a CLI that:

```text
loads fixed ULF log dir
loads keypoints_sampled_idx.pkl
projects sampled Gaussian xyz into train images
applies processed masks when present
writes visibility_teacher.pt, split_audit.json, manifest.json, command.txt, git_status.txt
rejects --split_name test
```

The script should reuse existing ULF projection helpers in `loc_gs/scripts/resample_ulfloc_with_solver_feedback.py` where practical, especially mask loading and projection conventions.

- [ ] **Step 4: Run unit test**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_ulfloc_visibility_teacher.py -q
```

Expected: passes.

## Task 3: Negative-Aware Scene Detector Targets

**Files:**
- Modify: `loc_gs/training/ulfloc_scene_detector.py`
- Create: `loc_gs/training/ulfloc_solver_feedback_detector_targets.py`
- Create: `loc_gs/scripts/build_ulfloc_solver_feedback_detector_targets.py`
- Test: `tests/test_ulfloc_solver_feedback_detector_targets.py`
- Test: `tests/test_train_ulfloc_scene_detector.py`

- [ ] **Step 1: Add target composition tests**

Create `tests/test_ulfloc_solver_feedback_detector_targets.py`:

```python
from loc_gs.training.ulfloc_solver_feedback_detector_targets import compose_detector_targets


def test_teacher_positive_and_negative_suppression_are_separate():
    teacher = {
        "q1": {
            "gaussian_ids": [1],
            "keypoint_yx": [[10.0, 20.0]],
            "support_weights": [1.0],
        }
    }
    impact = {
        "split_name": "train_selfmap",
        "detector_positive": {"q1": [{"gaussian_id": 1, "keypoint_xy": [20.0, 10.0], "weight": 2.0}]},
        "detector_negative": {"q1": [{"gaussian_id": 2, "keypoint_xy": [40.0, 30.0], "weight": 3.0}]},
    }
    out = compose_detector_targets(teacher, impact, height=64, width=80)
    assert out["q1"]["positive_count"] == 1
    assert out["q1"]["negative_count"] == 1
    assert out["q1"]["teacher_support_weights"][0] == 1.0
```

- [ ] **Step 2: Implement target composition**

Create `loc_gs/training/ulfloc_solver_feedback_detector_targets.py`:

```python
from __future__ import annotations

from typing import Any, Mapping

import torch


def _xy_to_yx(xy: Any) -> list[float] | None:
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None
    return [float(xy[1]), float(xy[0])]


def compose_detector_targets(
    teacher_targets: Mapping[str, Mapping[str, Any]],
    impact: Mapping[str, Any],
    *,
    height: int,
    width: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    detector_positive = impact.get("detector_positive", {})
    detector_negative = impact.get("detector_negative", {})
    for query_id in sorted(set(teacher_targets) | set(detector_positive) | set(detector_negative)):
        teacher = teacher_targets.get(query_id, {})
        gids = torch.as_tensor(teacher.get("gaussian_ids", []), dtype=torch.long).reshape(-1)
        yx = torch.as_tensor(teacher.get("keypoint_yx", []), dtype=torch.float32).reshape(-1, 2)
        weights = torch.as_tensor(teacher.get("support_weights", []), dtype=torch.float32).reshape(-1)
        if weights.numel() != gids.numel():
            weights = torch.ones_like(gids, dtype=torch.float32)
        neg_yx: list[list[float]] = []
        neg_weights: list[float] = []
        neg_gids: list[int] = []
        for row in detector_negative.get(query_id, []):
            point = _xy_to_yx(row.get("keypoint_xy"))
            if point is None:
                continue
            y, x = point
            if 0.0 <= y < float(height) and 0.0 <= x < float(width):
                neg_yx.append([y, x])
                neg_weights.append(float(row.get("weight", 1.0)))
                neg_gids.append(int(row.get("gaussian_id", -1)))
        out[str(query_id)] = {
            "gaussian_ids": gids,
            "keypoint_yx": yx,
            "support_weights": weights,
            "teacher_support_weights": weights.clone(),
            "negative_gaussian_ids": torch.tensor(neg_gids, dtype=torch.long),
            "negative_keypoint_yx": torch.tensor(neg_yx, dtype=torch.float32).reshape(-1, 2),
            "negative_weights": torch.tensor(neg_weights, dtype=torch.float32),
            "positive_count": int(gids.numel()),
            "negative_count": int(len(neg_gids)),
        }
    return out
```

- [ ] **Step 3: Update detector loss to use suppression**

Modify `scene_detector_loss` in `loc_gs/training/ulfloc_scene_detector.py` so positive target and suppression target are not collapsed into the same scalar weight. Add a new function:

```python
def scene_detector_loss_with_suppression(
    prediction: torch.Tensor,
    positive_target: torch.Tensor,
    positive_weight: torch.Tensor,
    suppression_target: torch.Tensor,
    *,
    positive_weight_scale: float = 6.0,
    suppression_weight_scale: float = 2.0,
    background_weight: float = 0.25,
) -> torch.Tensor:
    pred = torch.as_tensor(prediction, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    pos = torch.as_tensor(positive_target, dtype=torch.float32).to(pred.device).clamp(0.0, 1.0)
    pos_w = torch.as_tensor(positive_weight, dtype=torch.float32).to(pred.device).clamp_min(0.0)
    suppress = torch.as_tensor(suppression_target, dtype=torch.float32).to(pred.device).clamp(0.0, 1.0)
    bce_pos = torch.nn.functional.binary_cross_entropy(pred, pos, reduction="none")
    bce_suppress = torch.nn.functional.binary_cross_entropy(pred, torch.zeros_like(pred), reduction="none")
    loss = bce_pos * (background_weight + positive_weight_scale * pos + pos_w)
    loss = loss + suppression_weight_scale * suppress * bce_suppress
    return loss.mean()
```

- [ ] **Step 4: Keep geomw teacher as positive anchor**

Modify `train_ulfloc_scene_detector.py` so it loads composed detector targets and rasterizes:

```text
positive teacher heatmap from sampled landmark projection
negative suppression heatmap from solver feedback impact
```

Do not use `teacher_weights * residual_scale` in this new mainline.

- [ ] **Step 5: Run detector tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_ulfloc_solver_feedback_detector_targets.py \
  tests/test_train_ulfloc_scene_detector.py -q
```

Expected: passes.

## Task 4: Solver-Feedback Feature Fusion V2

**Files:**
- Create: `loc_gs/stdloc_native/solver_feedback_feature_fusion_v2.py`
- Create: `loc_gs/scripts/build_ulfloc_solver_feedback_feature_log.py`
- Test: `tests/test_solver_feedback_feature_fusion_v2.py`

- [ ] **Step 1: Write fusion tests**

Create `tests/test_solver_feedback_feature_fusion_v2.py`:

```python
import torch

from loc_gs.stdloc_native.solver_feedback_feature_fusion_v2 import fuse_landmark_descriptors


def test_negative_views_are_excluded_and_native_anchor_kept():
    native = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    observations = {
        0: [
            {"descriptor": torch.tensor([1.0, 0.0]), "positive_weight": 1.0, "negative_weight": 0.0},
            {"descriptor": torch.tensor([0.0, 1.0]), "positive_weight": 0.0, "negative_weight": 5.0},
        ],
        1: [
            {"descriptor": torch.tensor([0.0, 1.0]), "positive_weight": 1.0, "negative_weight": 0.0},
        ],
    }
    fused, meta = fuse_landmark_descriptors(native, observations, trust_alpha=0.1, min_native_cosine=0.95)
    assert fused.shape == native.shape
    assert float((fused[0] * native[0]).sum()) >= 0.95
    assert meta["negative_view_excluded_count"] == 1
```

- [ ] **Step 2: Implement fusion**

Create `loc_gs/stdloc_native/solver_feedback_feature_fusion_v2.py`:

```python
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def fuse_landmark_descriptors(
    native_descriptors: torch.Tensor,
    observations: Mapping[int, list[Mapping[str, Any]]],
    *,
    trust_alpha: float = 0.1,
    min_native_cosine: float = 0.95,
    min_positive_weight: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    native = F.normalize(torch.as_tensor(native_descriptors, dtype=torch.float32), p=2, dim=1)
    fused = native.clone()
    used = 0
    excluded = 0
    fallback = 0
    alpha = max(0.0, min(1.0, float(trust_alpha)))
    for gid, rows in observations.items():
        idx = int(gid)
        if idx < 0 or idx >= native.shape[0]:
            continue
        weighted: list[torch.Tensor] = []
        weights: list[float] = []
        for row in rows:
            pos = max(0.0, float(row.get("positive_weight", 0.0)))
            neg = max(0.0, float(row.get("negative_weight", 0.0)))
            if neg > pos:
                excluded += 1
                continue
            if pos <= min_positive_weight:
                continue
            desc = F.normalize(torch.as_tensor(row["descriptor"], dtype=torch.float32).reshape(-1), p=2, dim=0)
            weighted.append(desc)
            weights.append(pos)
        if not weighted:
            fallback += 1
            continue
        stack = torch.stack(weighted, dim=0)
        w = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)
        mean = F.normalize((stack * w).sum(dim=0), p=2, dim=0)
        candidate = F.normalize((1.0 - alpha) * native[idx] + alpha * mean, p=2, dim=0)
        if float((candidate * native[idx]).sum().item()) < float(min_native_cosine):
            fallback += 1
            continue
        fused[idx] = candidate
        used += 1
    return fused, {
        "fused_landmark_count": int(used),
        "fallback_count": int(fallback),
        "negative_view_excluded_count": int(excluded),
        "trust_alpha": float(alpha),
        "min_native_cosine": float(min_native_cosine),
    }
```

- [ ] **Step 3: Add ULF log export CLI**

Create `loc_gs/scripts/build_ulfloc_solver_feedback_feature_log.py` that:

```text
loads fixed ULF input_log_dir
loads existing keypoints_features.pkl and keypoints_sampled_idx.pkl
loads impact_attribution.pt
loads observation descriptors from the existing active fusion / pair-cache artifact
calls fuse_landmark_descriptors
writes a new ULF log dir with the same sampled_idx and fused features
writes manifest.json, command.txt, metrics_summary.json, split_audit.json, git_status.txt
```

The CLI must fail if impact split is `test` or sampled ids mismatch the observation cache.

- [ ] **Step 4: Run fusion tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_solver_feedback_feature_fusion_v2.py -q
```

Expected: passes.

## Task 5: Query-Conditioned Landmark Activation

**Files:**
- Create: `loc_gs/localization/query_landmark_activation.py`
- Create: `loc_gs/scripts/build_ulfloc_landmark_activation_cache.py`
- Create: `loc_gs/scripts/train_ulfloc_landmark_activation.py`
- Test: `tests/test_query_landmark_activation.py`
- Test: `tests/test_train_ulfloc_landmark_activation.py`

- [ ] **Step 1: Write activation tests**

Create `tests/test_query_landmark_activation.py`:

```python
import torch

from loc_gs.localization.query_landmark_activation import select_active_landmarks


def test_select_active_landmarks_keeps_safe_core_and_top_scores():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    safe_core = torch.tensor([0])
    selected = select_active_landmarks(scores, top_n=2, safe_core=safe_core)
    assert selected.tolist() == [0, 1, 3]
```

- [ ] **Step 2: Implement selector and model skeleton**

Create `loc_gs/localization/query_landmark_activation.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class QueryLandmarkActivationNet(nn.Module):
    def __init__(self, query_dim: int, landmark_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.landmark_proj = nn.Linear(landmark_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, query_feature: torch.Tensor, landmark_features: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query_feature).reshape(1, -1)
        l = self.landmark_proj(landmark_features)
        h = torch.relu(l + q)
        return self.score(h).reshape(-1)


def select_active_landmarks(
    scores: torch.Tensor,
    *,
    top_n: int,
    safe_core: torch.Tensor | None = None,
) -> torch.Tensor:
    score = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    k = min(max(0, int(top_n)), int(score.numel()))
    top = torch.topk(score, k=k).indices if k > 0 else torch.empty(0, dtype=torch.long)
    if safe_core is None:
        merged = top
    else:
        merged = torch.cat([torch.as_tensor(safe_core, dtype=torch.long).reshape(-1), top])
    return torch.unique(merged, sorted=True)
```

- [ ] **Step 3: Build activation cache**

Create `loc_gs/scripts/build_ulfloc_landmark_activation_cache.py` that writes per-query training rows:

```text
query_id
query_global_feature
landmark_features for fixed sampled landmarks
positive landmark ids from PnP inliers
negative landmark ids from high-score PnP outliers / regression competitors
safe core ids from high positive support and low negative risk
split audit
```

Reject `split_name=test`.

- [ ] **Step 4: Train with contrastive activation objective**

Create `loc_gs/scripts/train_ulfloc_landmark_activation.py` with loss:

```python
def activation_loss(scores, positive_ids, negative_ids, margin=0.2):
    pos = scores[positive_ids]
    neg = scores[negative_ids]
    bce_target = torch.zeros_like(scores)
    bce_target[positive_ids] = 1.0
    bce = torch.nn.functional.binary_cross_entropy_with_logits(scores, bce_target)
    if pos.numel() == 0 or neg.numel() == 0:
        return bce
    pair = torch.relu(float(margin) - pos.reshape(-1, 1) + neg.reshape(1, -1)).mean()
    return bce + pair
```

The checkpoint must store:

```text
state_dict
query_dim
landmark_dim
top_n default
safe_core ids
source cache path
split_name
```

- [ ] **Step 5: Run tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_query_landmark_activation.py \
  tests/test_train_ulfloc_landmark_activation.py -q
```

Expected: passes.

## Task 6: ULF Sparse Inference Integration

**Files:**
- Modify: `/root/ULF-Loc/ulfloc.py`
- Modify: `loc_gs/scripts/eval_ulfloc_sparse_only.py`
- Test: `tests/test_eval_ulfloc_sparse_only.py`

- [ ] **Step 1: Add eval CLI arguments**

Modify `eval_ulfloc_sparse_only.py` to accept:

```python
parser.add_argument("--landmark_activation_checkpoint", default=None, type=Path)
parser.add_argument("--landmark_activation_top_n", default=0, type=int)
parser.add_argument("--landmark_activation_safe_core", default=None, type=Path)
```

Set ULF config:

```python
config["landmark_activation"] = {
    "enabled": args.landmark_activation_checkpoint is not None,
    "checkpoint": None if args.landmark_activation_checkpoint is None else str(Path(args.landmark_activation_checkpoint).resolve()),
    "top_n": int(args.landmark_activation_top_n),
    "safe_core": None if args.landmark_activation_safe_core is None else str(Path(args.landmark_activation_safe_core).resolve()),
}
```

- [ ] **Step 2: Add ULF hook before descriptor matching**

Modify `/root/ULF-Loc/ulfloc.py` so `loc_coarse` applies activation before matching:

```text
load activation checkpoint in __init__
after query sparse descriptors are computed and before corr_matrix construction:
  compute query feature summary
  score fixed sampled landmarks
  keep active sampled rows
  subset landmark_features, point3D, and sampled ids consistently
  record landmark_activation metadata
```

The hook must not use GT pose, test labels, or query evaluation error.

- [ ] **Step 3: Add runtime audit**

`eval_ulfloc_sparse_only.py` must write:

```text
landmark_activation.enabled
landmark_activation_checkpoint
landmark_activation_enabled_query_count
landmark_activation_mean_active_count
```

If checkpoint is configured but no query reports activation metadata, raise `RuntimeError`.

- [ ] **Step 4: Run integration tests**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_eval_ulfloc_sparse_only.py -q
```

Expected: passes.

## Task 7: Smoke Experiments On ShopFacade

**Files:**
- No new code files.
- Outputs under `output/ulfloc_solver_feedback_mainline_v2/ShopFacade/...`.

- [ ] **Step 1: Build impact attribution**

Run with non-test feedback:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_solver_feedback_impact \
  --feedback output/feedback_banks/ShopFacade/train_selfmap_sparse_feedback.json \
  --output_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/impact
```

Expected output:

```text
impact_attribution.pt
impact_attribution.json
manifest.json
command.txt
git_status.txt
```

- [ ] **Step 2: Build visibility teacher**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_ulfloc_visibility_teacher \
  --scene ShopFacade \
  --split_name train_selfmap \
  --ulf_root /root/ULF-Loc \
  --input_log_dir output/ulfloc_best/ShopFacade/log \
  --source_path /root/data/Cambridge/ShopFacade \
  --images processed \
  --output_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/visibility_teacher
```

Expected: `visibility_teacher.pt` and audit files.

- [ ] **Step 3: Train detector**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_ulfloc_solver_feedback_detector_targets \
  --visibility_teacher output/ulfloc_solver_feedback_mainline_v2/ShopFacade/visibility_teacher/visibility_teacher.pt \
  --impact output/ulfloc_solver_feedback_mainline_v2/ShopFacade/impact/impact_attribution.pt \
  --output_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/detector_targets

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.train_ulfloc_scene_detector \
  --detector_targets output/ulfloc_solver_feedback_mainline_v2/ShopFacade/detector_targets/detector_targets.pt \
  --source_path /root/data/Cambridge/ShopFacade \
  --images processed \
  --epochs 3 \
  --output_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/detector
```

Expected: `scene_detector.pth` and training metrics.

- [ ] **Step 4: Build fused feature log**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_ulfloc_solver_feedback_feature_log \
  --input_log_dir output/ulfloc_best/ShopFacade/log \
  --impact output/ulfloc_solver_feedback_mainline_v2/ShopFacade/impact/impact_attribution.pt \
  --observation_cache output/ulfloc_solver_feedback_mainline_v2/ShopFacade/active_fusion_observations.pt \
  --trust_alpha 0.1 \
  --min_native_cosine 0.95 \
  --output_log_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/fused_log
```

Expected: same sampled ids, fused `keypoints_features.pkl`, metrics.

- [ ] **Step 5: Train activation**

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_ulfloc_landmark_activation_cache \
  --impact output/ulfloc_solver_feedback_mainline_v2/ShopFacade/impact/impact_attribution.pt \
  --input_log_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/fused_log \
  --output_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/activation_cache

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.train_ulfloc_landmark_activation \
  --cache output/ulfloc_solver_feedback_mainline_v2/ShopFacade/activation_cache/cache.pt \
  --epochs 5 \
  --top_n 12000 \
  --output_dir output/ulfloc_solver_feedback_mainline_v2/ShopFacade/activation
```

Expected: `landmark_activation.pth`.

- [ ] **Step 6: Sparse-only eval ablation table**

Run the same train-dev split for:

```text
A: ULF best fixed sampled set
B: A + solver-feedback fused descriptor
C: B + negative-aware scene detector
D: C + query-conditioned landmark activation
```

Each run must write manifest, command, metrics, split audit, and git status. Primary metrics:

```text
median TE
median RE
R10
R5
```

Gate:

```text
D median TE must be <= B median TE
D R10 and R5 must not both fall below B
activation mean active count must be below full landmark count
```

## Task 8: Expand To Five Cambridge Scenes

**Files:**
- Create: `loc_gs/scripts/launch_ulfloc_solver_feedback_mainline.py`
- Test: `tests/test_launch_ulfloc_solver_feedback_mainline.py`

- [ ] **Step 1: Add launcher dry-run test**

Create a launcher that emits commands for:

```text
GreatCourt
KingsCollege
OldHospital
ShopFacade
StMarysChurch
```

The dry-run test asserts:

```python
def test_launcher_uses_non_test_feedback_for_training():
    commands = build_commands(scenes=["ShopFacade"], split_name="train_selfmap", dry_run=True)
    text = "\n".join(" ".join(cmd) for cmd in commands)
    assert "split_name test" not in text
    assert "--split_name train_selfmap" in text
```

- [ ] **Step 2: Run five-scene sparse train-dev eval**

Use three GPUs by scheduling independent scene jobs. Do not start official test runs in this task.

Gate:

```text
macro median TE improves over fixed ULF best baseline
at least 3/5 scenes improve median TE
ShopFacade does not regress
OldHospital and StMarysChurch do not both regress
```

- [ ] **Step 3: Write status report**

Create `docs/ulfloc_solver_feedback_query_conditioned_status_20260609.md` with:

```text
method summary
per-scene sparse table
which modules passed gates
negative results
next action
```

## Verification Commands

Run before claiming implementation complete:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_solver_feedback_impact.py \
  tests/test_build_solver_feedback_impact_cli.py \
  tests/test_ulfloc_visibility_teacher.py \
  tests/test_ulfloc_solver_feedback_detector_targets.py \
  tests/test_solver_feedback_feature_fusion_v2.py \
  tests/test_query_landmark_activation.py \
  tests/test_train_ulfloc_landmark_activation.py \
  tests/test_eval_ulfloc_sparse_only.py \
  tests/test_launch_ulfloc_solver_feedback_mainline.py -q

/root/miniconda3/envs/cybersim_agent/bin/python -m compileall \
  loc_gs/feedback \
  loc_gs/training \
  loc_gs/localization \
  loc_gs/stdloc_native \
  loc_gs/scripts
```

## Self-Review Checklist

- The plan keeps official Cambridge test out of feedback and model selection.
- The plan keeps fixed ULF sampled ids for the new mainline.
- The plan removes `teacher_weights * residual_scale` from the new detector main path.
- The plan treats negative correspondence evidence as attribution for detector/fusion/activation, not as a post-matching reranker.
- The plan requires query-conditioned landmark activation before matching and before PnP.
- The plan uses median TE, median RE, R10, and R5 as the first sparse metrics.
