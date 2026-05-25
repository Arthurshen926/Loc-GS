from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import load_feedback_bank


_GOOD_TRANSITIONS = {"improved", "rescued", "stable_ok"}
_BAD_TRANSITIONS = {"worsened", "lost"}
_NEUTRAL_TRANSITIONS = {"unchanged", "stable_fail", ""}


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _gaussian_id(record: dict[str, Any]) -> int | None:
    value = record.get("matched_gaussian_id", record.get("matched_landmark_id"))
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _transition_vote(record: dict[str, Any], *, delta_scale_cm: float) -> tuple[float, float]:
    transition = str(record.get("dense_transition", "")).strip().lower()
    delta = _as_float(record.get("dense_delta_te_cm"))
    if delta is not None and abs(delta) > 1e-9:
        weight = 1.0 + min(abs(float(delta)) / max(float(delta_scale_cm), 1e-6), 5.0)
        if delta < 0.0:
            return weight, 0.0
        return 0.0, weight
    if transition in _GOOD_TRANSITIONS:
        return 1.0, 0.0
    if transition in _BAD_TRANSITIONS:
        return 0.0, 1.0
    if transition in _NEUTRAL_TRANSITIONS:
        return 0.5, 0.5
    return 0.5, 0.5


def distill_dense_lsf_targets(
    feedback_bank: str | Path,
    *,
    num_gaussians: int | None = None,
    smoothing: float = 1.0,
    keep_threshold: float = 0.5,
    delta_scale_cm: float = 10.0,
) -> dict[str, Any]:
    """Distill dense-stage self-map feedback into a per-Gaussian support target."""

    audit = audit_feedback_bank_v2(feedback_bank)
    if audit["audit_status"] != "passed":
        raise ValueError(f"feedback bank v2 audit failed: {audit['reasons']}")
    bank = load_feedback_bank(feedback_bank)
    records = [dict(record) for record in bank.get("records", [])]
    ids = [gid for gid in (_gaussian_id(record) for record in records) if gid is not None]
    if num_gaussians is None:
        num_gaussians = max(ids) + 1 if ids else 0
    num_gaussians = int(num_gaussians)
    if ids and max(ids) >= num_gaussians:
        raise IndexError("num_gaussians is smaller than observed matched_gaussian_id")

    smooth = max(float(smoothing), 0.0)
    positive = torch.full((num_gaussians,), smooth, dtype=torch.float32)
    negative = torch.full((num_gaussians,), smooth, dtype=torch.float32)
    observed = torch.zeros((num_gaussians,), dtype=torch.long)
    for record in records:
        gid = _gaussian_id(record)
        if gid is None or gid >= num_gaussians:
            continue
        pos, neg = _transition_vote(record, delta_scale_cm=float(delta_scale_cm))
        positive[gid] += float(pos)
        negative[gid] += float(neg)
        observed[gid] += 1

    denom = (positive + negative).clamp_min(1e-6)
    target = (positive / denom).clamp(0.0, 1.0)
    confidence = ((denom - 2.0 * smooth) / denom.clamp_min(1e-6)).clamp(0.0, 1.0)
    selected_idx = torch.where((observed > 0) & (target >= float(keep_threshold)))[0].long()
    manifest = dict(bank.get("manifest", {}))
    return {
        "dense_lsf_target": target,
        "dense_lsf_confidence": confidence,
        "positive_weight": positive - smooth,
        "negative_weight": negative - smooth,
        "observed_count": observed,
        "selected_idx": selected_idx,
        "metadata": {
            "dense_lsf_distillation": "v1_transition_beta",
            "usage_scope": "dense_residual_teacher_only",
            "sparse_selector_safe": False,
            "feedback_bank": str(feedback_bank),
            "feedback_bank_schema": manifest.get("schema_version", ""),
            "split_name": manifest.get("split_name", ""),
            "smoothing": float(smooth),
            "keep_threshold": float(keep_threshold),
            "delta_scale_cm": float(delta_scale_cm),
            "observed_gaussian_count": int((observed > 0).sum().item()),
            "selected_count": int(selected_idx.numel()),
        },
    }
