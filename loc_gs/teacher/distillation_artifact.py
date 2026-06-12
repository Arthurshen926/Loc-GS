from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class DistillationArtifactConfig:
    dense_consistency_reprojection_px: float = 4.0
    sparse_inlier_reprojection_px: float = 8.0
    hard_negative_reprojection_px: float = 8.0
    max_solver_weight: float = 4.0


@dataclass(frozen=True)
class SolverFeedbackRow:
    scene: str
    split_name: str
    query_id: str
    dense_helped: bool
    distill_weight: float


def load_solver_feedback_label_rows(path: str) -> list[SolverFeedbackRow]:
    import json
    from pathlib import Path

    rows: list[SolverFeedbackRow] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"solver feedback label line {line_number} must be an object")
        split = reject_test_split(str(payload.get("split_name", "unknown")), purpose="distillation artifact labels")
        query_id = str(payload.get("query_id") or "")
        if not query_id:
            raise ValueError(f"solver feedback label line {line_number} is missing query_id")
        rows.append(
            SolverFeedbackRow(
                scene=str(payload.get("scene", "unknown")),
                split_name=split,
                query_id=query_id,
                dense_helped=bool(payload.get("dense_helped", False)),
                distill_weight=float(payload.get("distill_weight", 0.0)),
            )
        )
    return rows


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    meta = payload.get("metadata", {})
    return dict(meta) if isinstance(meta, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"candidate artifact is missing required field: {key}")
    return payload[key]


def _role(*, valid: bool, geometric: bool, dense: bool, sparse_inlier: bool, reprojection_px: float, cfg: DistillationArtifactConfig) -> str:
    if not valid:
        return "neutral"
    if geometric and dense and sparse_inlier:
        return "protected_support"
    if geometric and sparse_inlier:
        return "positive_inlier"
    if (not geometric) and reprojection_px >= float(cfg.hard_negative_reprojection_px):
        return "hard_negative"
    return "neutral"


def build_distillation_payload(
    payload: Mapping[str, Any],
    feedback_rows: Sequence[SolverFeedbackRow],
    *,
    scene: str,
    split_name: str,
    cfg: DistillationArtifactConfig | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    if cfg is None:
        cfg = DistillationArtifactConfig()
    split = reject_test_split(split_name, purpose="distillation artifact generation")
    labels_by_query = {row.query_id: row for row in feedback_rows}
    for row in feedback_rows:
        reject_test_split(row.split_name, purpose="distillation artifact generation")
        if row.scene not in {"unknown", str(scene)}:
            raise ValueError(f"feedback label scene mismatch: expected {scene}, got {row.scene}")

    landmark_ids = _as_list(_required(payload, "landmark_id"))
    query_ids = [str(item) for item in _as_list(_required(payload, "query_id"))]
    image_ids = [str(item) for item in _as_list(payload.get("image_id", [qid.split("::", 1)[0] for qid in query_ids]))]
    labels = [int(item) for item in _as_list(_required(payload, "label"))]
    candidate_mask = _as_list(payload.get("candidate_mask", [[True for _ in row] for row in landmark_ids]))
    reprojection_error = _as_list(
        payload.get("reprojection_error", [[float("inf") for _ in row] for row in landmark_ids])
    )
    row_count = len(landmark_ids)
    lengths = {
        "query_id": len(query_ids),
        "image_id": len(image_ids),
        "label": len(labels),
        "candidate_mask": len(candidate_mask),
        "reprojection_error": len(reprojection_error),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"candidate artifact field length mismatch: landmark_id={row_count}, lengths={lengths}")

    dense_consistent_rows: list[list[bool]] = []
    sparse_inlier_rows: list[list[bool]] = []
    solver_weight_rows: list[list[float]] = []
    label_role_rows: list[list[str]] = []
    matched_feedback = 0
    dense_helped_queries: set[str] = set()
    for row_idx, ids in enumerate(landmark_ids):
        image_id = image_ids[row_idx]
        feedback = labels_by_query.get(image_id)
        if feedback is not None:
            matched_feedback += 1
            if feedback.dense_helped:
                dense_helped_queries.add(image_id)
        teacher_label = labels[row_idx]
        dense_row: list[bool] = []
        inlier_row: list[bool] = []
        weight_row: list[float] = []
        role_row: list[str] = []
        for rank in range(len(ids)):
            valid = bool(candidate_mask[row_idx][rank])
            geometric = 0 <= teacher_label < len(ids) and rank == teacher_label
            reproj = float(reprojection_error[row_idx][rank])
            if feedback is not None and feedback.dense_helped:
                dense = bool(geometric and reproj <= float(cfg.dense_consistency_reprojection_px))
            else:
                dense = bool(geometric)
            sparse_inlier = bool(geometric and reproj <= float(cfg.sparse_inlier_reprojection_px))
            role = _role(valid=valid, geometric=geometric, dense=dense, sparse_inlier=sparse_inlier, reprojection_px=reproj, cfg=cfg)
            distill_boost = float(feedback.distill_weight) if feedback is not None else 0.0
            weight = min(float(cfg.max_solver_weight), 1.0 + max(0.0, distill_boost))
            dense_row.append(dense)
            inlier_row.append(sparse_inlier)
            weight_row.append(float(weight))
            role_row.append(role)
        dense_consistent_rows.append(dense_row)
        sparse_inlier_rows.append(inlier_row)
        solver_weight_rows.append(weight_row)
        label_role_rows.append(role_row)

    distilled = dict(payload)
    meta = _metadata(payload)
    meta.update(
        {
            "format": str(meta.get("format", "listwise")),
            "scene": str(scene),
            "source_split_name": split,
            "distillation_teacher": "sparse_dense_solver_feedback",
            "dense_teacher_enabled": True,
        }
    )
    distilled["metadata"] = meta
    distilled["dense_consistent"] = torch.tensor(dense_consistent_rows, dtype=torch.bool)
    distilled["sparse_inlier"] = torch.tensor(sparse_inlier_rows, dtype=torch.bool)
    distilled["solver_weight"] = torch.tensor(solver_weight_rows, dtype=torch.float32)
    distilled["label_roles"] = label_role_rows
    summary = {
        "schema_version": "internal_distillation_artifact_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "candidate_row_count": int(row_count),
        "candidate_sample_count": int(sum(len(row) for row in landmark_ids)),
        "feedback_query_count": int(len(labels_by_query)),
        "matched_feedback_row_count": int(matched_feedback),
        "dense_helped_query_count": int(len(dense_helped_queries)),
        "protected_support_count": int(sum(role == "protected_support" for row in label_role_rows for role in row)),
        "hard_negative_count": int(sum(role == "hard_negative" for row in label_role_rows for role in row)),
    }
    return distilled, summary
