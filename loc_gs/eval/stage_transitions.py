from __future__ import annotations

import math
from statistics import median
from typing import Any

from loc_gs.eval.query_partitions import _as_float, _stage_metric


def _query_id(row: dict[str, Any], index: int) -> str:
    value = row.get("query_id")
    if value is not None:
        return str(value)
    return f"query_{index:06d}"


def _valid_pose(row: dict[str, Any], stage: str) -> tuple[float, float] | None:
    te = _stage_metric(row, stage, "te_cm")
    re = _stage_metric(row, stage, "re_deg")
    if te is None:
        te = _as_float(row.get(f"{stage}_TE"))
    if re is None:
        re = _as_float(row.get(f"{stage}_AE"))
    if te is None or re is None:
        return None
    if not math.isfinite(te) or not math.isfinite(re):
        return None
    return float(te), float(re)


def _passes(row: dict[str, Any], stage: str, *, te_cm: float, re_deg: float) -> bool:
    pose = _valid_pose(row, stage)
    if pose is None:
        return False
    te, re = pose
    return te <= te_cm and re <= re_deg


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "stable_ok"
    if before and not after:
        return "lost"
    if not before and after:
        return "rescued"
    return "stable_fail"


def dense_transition_labels(
    rows: list[dict[str, Any]],
    *,
    te_epsilon_cm: float = 0.0,
    hard_tail_te_cm: float = 50.0,
) -> list[dict[str, Any]]:
    """Build per-query labels for dense-stage self-map risk distillation."""

    labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sparse_pose = _valid_pose(row, "sparse")
        dense_pose = _valid_pose(row, "dense")
        if sparse_pose is None or dense_pose is None:
            continue
        sparse_te, sparse_re = sparse_pose
        dense_te, dense_re = dense_pose
        delta = dense_te - sparse_te
        if delta < -float(te_epsilon_cm):
            pose_transition = "improved"
        elif delta > float(te_epsilon_cm):
            pose_transition = "worsened"
        else:
            pose_transition = "unchanged"

        sparse_r5 = _passes(row, "sparse", te_cm=5.0, re_deg=5.0)
        dense_r5 = _passes(row, "dense", te_cm=5.0, re_deg=5.0)
        sparse_r2 = _passes(row, "sparse", te_cm=2.0, re_deg=2.0)
        dense_r2 = _passes(row, "dense", te_cm=2.0, re_deg=2.0)
        labels.append(
            {
                "query_id": _query_id(row, index),
                "sparse_te_cm": float(sparse_te),
                "sparse_re_deg": float(sparse_re),
                "dense_te_cm": float(dense_te),
                "dense_re_deg": float(dense_re),
                "dense_minus_sparse_te_cm": float(delta),
                "pose_transition": pose_transition,
                "r5_transition": _transition(sparse_r5, dense_r5),
                "r2_transition": _transition(sparse_r2, dense_r2),
                "hard_tail_before_dense": bool(sparse_te >= float(hard_tail_te_cm)),
                "hard_tail_after_dense": bool(dense_te >= float(hard_tail_te_cm)),
            }
        )
    return labels


def _empty_summary() -> dict[str, Any]:
    return {
        "query_count": 0,
        "improved_count": 0,
        "worsened_count": 0,
        "unchanged_count": 0,
        "mean_dense_minus_sparse_te_cm": 0.0,
        "median_dense_minus_sparse_te_cm": 0.0,
        "r5_rescued_count": 0,
        "r5_lost_count": 0,
        "r2_rescued_count": 0,
        "r2_lost_count": 0,
        "worst_dense_worsened_queries": [],
        "best_dense_improved_queries": [],
    }


def stage_transition_summary(
    rows: list[dict[str, Any]],
    *,
    te_epsilon_cm: float = 0.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarize how dense refinement changes sparse pose errors for one run."""

    deltas: list[float] = []
    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []
    unchanged = 0
    r5_rescued = 0
    r5_lost = 0
    r2_rescued = 0
    r2_lost = 0

    for index, row in enumerate(rows):
        sparse_pose = _valid_pose(row, "sparse")
        dense_pose = _valid_pose(row, "dense")
        if sparse_pose is None or dense_pose is None:
            continue
        sparse_te, _ = sparse_pose
        dense_te, _ = dense_pose
        delta = dense_te - sparse_te
        deltas.append(delta)
        item = {
            "query_id": _query_id(row, index),
            "sparse_te_cm": sparse_te,
            "dense_te_cm": dense_te,
            "dense_minus_sparse_te_cm": delta,
        }
        if delta < -te_epsilon_cm:
            improved.append(item)
        elif delta > te_epsilon_cm:
            worsened.append(item)
        else:
            unchanged += 1

        sparse_r5 = _passes(row, "sparse", te_cm=5.0, re_deg=5.0)
        dense_r5 = _passes(row, "dense", te_cm=5.0, re_deg=5.0)
        sparse_r2 = _passes(row, "sparse", te_cm=2.0, re_deg=2.0)
        dense_r2 = _passes(row, "dense", te_cm=2.0, re_deg=2.0)
        r5_rescued += int(not sparse_r5 and dense_r5)
        r5_lost += int(sparse_r5 and not dense_r5)
        r2_rescued += int(not sparse_r2 and dense_r2)
        r2_lost += int(sparse_r2 and not dense_r2)

    if not deltas:
        return _empty_summary()

    worsened.sort(key=lambda item: item["dense_minus_sparse_te_cm"], reverse=True)
    improved.sort(key=lambda item: item["dense_minus_sparse_te_cm"])
    return {
        "query_count": int(len(deltas)),
        "improved_count": int(len(improved)),
        "worsened_count": int(len(worsened)),
        "unchanged_count": int(unchanged),
        "mean_dense_minus_sparse_te_cm": float(sum(deltas) / len(deltas)),
        "median_dense_minus_sparse_te_cm": float(median(deltas)),
        "r5_rescued_count": int(r5_rescued),
        "r5_lost_count": int(r5_lost),
        "r2_rescued_count": int(r2_rescued),
        "r2_lost_count": int(r2_lost),
        "worst_dense_worsened_queries": worsened[:top_k],
        "best_dense_improved_queries": improved[:top_k],
    }
