from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import load_feedback_bank
from loc_gs.feedback.labels import derive_hard_negative_labels


SCHEMA = "solver_consensus_support_v1"
_DENSE_WORSENED = {"worsened", "lost"}


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _gaussian_id(record: dict[str, Any], row: int) -> int:
    value = record.get("matched_gaussian_id", record.get("matched_landmark_id"))
    try:
        gaussian_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {row} has non-integer matched_gaussian_id: {value!r}") from exc
    if gaussian_id < 0:
        raise ValueError(f"record {row} has negative matched_gaussian_id: {gaussian_id}")
    return gaussian_id


def _descriptor_quality(record: dict[str, Any]) -> float:
    value = _as_float(record.get("descriptor_score"))
    return 0.0 if value is None else _clamp01(value)


def _reprojection_quality(record: dict[str, Any], threshold_px: float) -> float:
    error = _as_float(record.get("reprojection_error_px"))
    if error is None:
        return 0.0
    return _clamp01(1.0 - (error / max(float(threshold_px), 1e-6)))


def _dense_worsen_risk(record: dict[str, Any], dense_delta_bad_cm: float) -> float:
    transition = str(record.get("dense_transition", "")).strip().lower()
    risk = 1.0 if transition in _DENSE_WORSENED else 0.0
    delta = _as_float(record.get("dense_delta_te_cm"))
    if delta is not None and delta > 0.0:
        risk = max(risk, _clamp01(delta / max(float(dense_delta_bad_cm), 1e-6)))
    return risk


def _base_support(record: dict[str, Any], reprojection_quality_threshold_px: float) -> float:
    pnp_quality = 1.0 if bool(record.get("pnp_inlier", False)) else 0.0
    reprojection_quality = _reprojection_quality(record, reprojection_quality_threshold_px)
    descriptor_quality = _descriptor_quality(record)
    return _clamp01((0.50 * pnp_quality) + (0.30 * reprojection_quality) + (0.20 * descriptor_quality))


def build_solver_consensus_support(
    feedback_bank: str | Path,
    *,
    num_gaussians: int | None = None,
    support_threshold: float = 0.5,
    reprojection_quality_threshold_px: float = 8.0,
    hard_negative_descriptor_score_min: float = 0.5,
    hard_negative_reprojection_error_px_min: float = 8.0,
    hard_negative_penalty: float = 0.75,
    dense_worsen_penalty: float = 0.60,
    dense_delta_bad_cm: float = 5.0,
) -> dict[str, Any]:
    """Build compact per-Gaussian solver-consensus support from feedback_bank_v2."""

    source_path = Path(feedback_bank)
    audit = audit_feedback_bank_v2(source_path)
    if audit["audit_status"] != "passed":
        raise ValueError(f"feedback bank v2 audit failed: {audit['reasons']}")

    bank = load_feedback_bank(source_path)
    manifest = dict(bank.get("manifest", {}))
    records = [dict(record) for record in bank.get("records", [])]
    gaussian_ids = [_gaussian_id(record, row) for row, record in enumerate(records)]
    inferred_count = (max(gaussian_ids) + 1) if gaussian_ids else 0
    gaussian_count = inferred_count if num_gaussians is None else int(num_gaussians)
    if gaussian_count < inferred_count:
        raise ValueError(
            f"num_gaussians={gaussian_count} is smaller than observed max gaussian id {inferred_count - 1}"
        )

    observed_count = torch.zeros((gaussian_count,), dtype=torch.long)
    inlier_sum = torch.zeros((gaussian_count,), dtype=torch.float32)
    weighted_support = torch.zeros((gaussian_count,), dtype=torch.float32)
    hard_negative_sum = torch.zeros((gaussian_count,), dtype=torch.float32)
    dense_worsen_sum = torch.zeros((gaussian_count,), dtype=torch.float32)

    hard_negative_labels = derive_hard_negative_labels(
        records,
        descriptor_score_min=float(hard_negative_descriptor_score_min),
        reprojection_error_px_min=float(hard_negative_reprojection_error_px_min),
    )
    for row, (record, gaussian_id) in enumerate(zip(records, gaussian_ids)):
        hard_negative = float(hard_negative_labels[row])
        dense_risk = _dense_worsen_risk(record, float(dense_delta_bad_cm))
        base_support = _base_support(record, float(reprojection_quality_threshold_px))
        penalty = _clamp01((float(hard_negative_penalty) * hard_negative) + (float(dense_worsen_penalty) * dense_risk))
        support_vote = base_support * (1.0 - penalty)

        observed_count[gaussian_id] += 1
        inlier_sum[gaussian_id] += 1.0 if bool(record.get("pnp_inlier", False)) else 0.0
        weighted_support[gaussian_id] += float(support_vote)
        hard_negative_sum[gaussian_id] += hard_negative
        dense_worsen_sum[gaussian_id] += dense_risk

    denom = observed_count.clamp_min(1).to(torch.float32)
    support_score = weighted_support / denom
    inlier_consensus = inlier_sum / denom
    hard_negative_risk = hard_negative_sum / denom
    dense_worsen_risk = dense_worsen_sum / denom
    selected_mask = (observed_count > 0) & (support_score >= float(support_threshold))

    metadata = {
        "schema": SCHEMA,
        "split": str(audit.get("split_name", manifest.get("split_name", manifest.get("split", "")))),
        "scene": str(manifest.get("scene", "")),
        "image_group_count": int(audit.get("image_group_count", 0)),
        "record_count": int(audit.get("record_count", len(records))),
        "source_feedback_bank": str(source_path),
        "selected_count": int(selected_mask.sum().item()),
        "support_threshold": float(support_threshold),
        "gaussian_count": int(gaussian_count),
        "feedback_bank_schema": str(audit.get("schema_version", manifest.get("schema_version", ""))),
        "query_id_source": str(audit.get("query_id_source", manifest.get("query_id_source", ""))),
        "hyperparameters": {
            "reprojection_quality_threshold_px": float(reprojection_quality_threshold_px),
            "hard_negative_descriptor_score_min": float(hard_negative_descriptor_score_min),
            "hard_negative_reprojection_error_px_min": float(hard_negative_reprojection_error_px_min),
            "hard_negative_penalty": float(hard_negative_penalty),
            "dense_worsen_penalty": float(dense_worsen_penalty),
            "dense_delta_bad_cm": float(dense_delta_bad_cm),
        },
        "audit": audit,
    }
    return {
        "support_score": support_score.to(torch.float32),
        "inlier_consensus": inlier_consensus.to(torch.float32),
        "weighted_support": weighted_support.to(torch.float32),
        "hard_negative_risk": hard_negative_risk.to(torch.float32),
        "dense_worsen_risk": dense_worsen_risk.to(torch.float32),
        "observed_count": observed_count,
        "metadata": metadata,
    }
