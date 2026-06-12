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


@dataclass(frozen=True)
class TeacherObservationRow:
    scene: str
    split_name: str
    image_id: str
    keypoint_id: str | None
    candidate_rank: int | None
    landmark_id: int | None
    geometric_correct: bool
    dense_consistent: bool
    sparse_inlier: bool
    reprojection_error_px: float
    solver_weight: float
    label_role: str | None = None


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


def load_teacher_observation_rows(path: str) -> list[TeacherObservationRow]:
    import json
    from pathlib import Path

    rows: list[TeacherObservationRow] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"teacher observation line {line_number} must be an object")
        split = reject_test_split(str(payload.get("split_name", "unknown")), purpose="teacher observation labels")
        image_id = str(payload.get("image_id") or payload.get("query_id") or "")
        if not image_id:
            raise ValueError(f"teacher observation line {line_number} is missing image_id/query_id")
        if payload.get("candidate_rank") is None and payload.get("landmark_id") is None:
            raise ValueError(f"teacher observation line {line_number} requires candidate_rank or landmark_id")
        rows.append(
            TeacherObservationRow(
                scene=str(payload.get("scene", "unknown")),
                split_name=split,
                image_id=image_id,
                keypoint_id=None if payload.get("keypoint_id") is None else str(payload.get("keypoint_id")),
                candidate_rank=None if payload.get("candidate_rank") is None else int(payload.get("candidate_rank")),
                landmark_id=None if payload.get("landmark_id") is None else int(payload.get("landmark_id")),
                geometric_correct=bool(payload.get("geometric_correct", False)),
                dense_consistent=bool(payload.get("dense_consistent", False)),
                sparse_inlier=bool(payload.get("sparse_inlier", False)),
                reprojection_error_px=float(payload.get("reprojection_error_px", payload.get("reprojection_error", float("inf")))),
                solver_weight=float(payload.get("solver_weight", payload.get("distill_weight", 1.0))),
                label_role=None if payload.get("label_role") is None else str(payload.get("label_role")),
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


def build_distillation_payload_from_observations(
    payload: Mapping[str, Any],
    observations: Sequence[TeacherObservationRow],
    *,
    scene: str,
    split_name: str,
    cfg: DistillationArtifactConfig | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    if cfg is None:
        cfg = DistillationArtifactConfig()
    split = reject_test_split(split_name, purpose="teacher observation distillation artifact generation")
    for row in observations:
        reject_test_split(row.split_name, purpose="teacher observation distillation artifact generation")
        if row.scene not in {"unknown", str(scene)}:
            raise ValueError(f"teacher observation scene mismatch: expected {scene}, got {row.scene}")

    landmark_ids = _as_list(_required(payload, "landmark_id"))
    query_ids = [str(item) for item in _as_list(_required(payload, "query_id"))]
    image_ids = [str(item) for item in _as_list(payload.get("image_id", [qid.split("::", 1)[0] for qid in query_ids]))]
    keypoint_ids = [str(item) for item in _as_list(payload.get("keypoint_id", [qid.rsplit("::", 1)[-1] for qid in query_ids]))]
    candidate_mask = _as_list(payload.get("candidate_mask", [[True for _ in row] for row in landmark_ids]))
    row_count = len(landmark_ids)
    lengths = {
        "query_id": len(query_ids),
        "image_id": len(image_ids),
        "keypoint_id": len(keypoint_ids),
        "candidate_mask": len(candidate_mask),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"candidate artifact field length mismatch: landmark_id={row_count}, lengths={lengths}")

    topk_lengths = [len(row) for row in landmark_ids]
    dense_consistent = [[False for _ in range(length)] for length in topk_lengths]
    sparse_inlier = [[False for _ in range(length)] for length in topk_lengths]
    reprojection_error = [[float("inf") for _ in range(length)] for length in topk_lengths]
    solver_weight = [[1.0 for _ in range(length)] for length in topk_lengths]
    geometric_correct = [[False for _ in range(length)] for length in topk_lengths]
    label_roles = [["neutral" for _ in range(length)] for length in topk_lengths]
    labels = [-1 for _ in range(row_count)]
    row_lookup: dict[tuple[str, str | None], list[int]] = {}
    for row_idx, (image_id, keypoint_id) in enumerate(zip(image_ids, keypoint_ids)):
        row_lookup.setdefault((image_id, keypoint_id), []).append(row_idx)
        row_lookup.setdefault((image_id, None), []).append(row_idx)

    matched_observations = 0
    for observation in observations:
        matched = _match_observation_to_candidate_row(
            observation,
            row_lookup=row_lookup,
            landmark_ids=landmark_ids,
        )
        if matched is None:
            continue
        row_idx, rank = matched
        if not bool(candidate_mask[row_idx][rank]):
            continue
        matched_observations += 1
        dense_consistent[row_idx][rank] = bool(observation.dense_consistent)
        sparse_inlier[row_idx][rank] = bool(observation.sparse_inlier)
        reprojection_error[row_idx][rank] = float(observation.reprojection_error_px)
        solver_weight[row_idx][rank] = min(float(cfg.max_solver_weight), max(0.0, float(observation.solver_weight)))
        geometric_correct[row_idx][rank] = bool(observation.geometric_correct)
        if observation.geometric_correct and labels[row_idx] < 0:
            labels[row_idx] = int(rank)
        label_roles[row_idx][rank] = (
            observation.label_role
            if observation.label_role is not None
            else _role(
                valid=True,
                geometric=bool(observation.geometric_correct),
                dense=bool(observation.dense_consistent),
                sparse_inlier=bool(observation.sparse_inlier),
                reprojection_px=float(observation.reprojection_error_px),
                cfg=cfg,
            )
        )

    distilled = dict(payload)
    meta = _metadata(payload)
    meta.update(
        {
            "format": str(meta.get("format", "listwise")),
            "scene": str(scene),
            "source_split_name": split,
            "distillation_teacher": "per_candidate_sparse_dense_observations",
            "dense_teacher_enabled": True,
        }
    )
    distilled["metadata"] = meta
    distilled["label"] = torch.tensor(labels, dtype=torch.long)
    distilled["dense_consistent"] = torch.tensor(dense_consistent, dtype=torch.bool)
    distilled["sparse_inlier"] = torch.tensor(sparse_inlier, dtype=torch.bool)
    distilled["reprojection_error"] = torch.tensor(reprojection_error, dtype=torch.float32)
    distilled["solver_weight"] = torch.tensor(solver_weight, dtype=torch.float32)
    distilled["label_roles"] = label_roles
    summary = {
        "schema_version": "internal_observation_distillation_artifact_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "candidate_row_count": int(row_count),
        "candidate_sample_count": int(sum(topk_lengths)),
        "observation_count": int(len(observations)),
        "matched_observation_count": int(matched_observations),
        "geometric_positive_row_count": int(sum(1 for label in labels if label >= 0)),
        "dense_consistent_count": int(sum(value for row in dense_consistent for value in row)),
        "sparse_inlier_count": int(sum(value for row in sparse_inlier for value in row)),
        "protected_support_count": int(sum(role == "protected_support" for row in label_roles for role in row)),
        "hard_negative_count": int(sum(role == "hard_negative" for row in label_roles for role in row)),
    }
    return distilled, summary


def _match_observation_to_candidate_row(
    observation: TeacherObservationRow,
    *,
    row_lookup: Mapping[tuple[str, str | None], Sequence[int]],
    landmark_ids: Sequence[Sequence[int]],
) -> tuple[int, int] | None:
    candidate_rows = row_lookup.get((observation.image_id, observation.keypoint_id), [])
    if not candidate_rows:
        return None
    for row_idx in candidate_rows:
        ids = [int(value) for value in landmark_ids[row_idx]]
        if observation.candidate_rank is not None:
            rank = int(observation.candidate_rank)
            if 0 <= rank < len(ids):
                return row_idx, rank
        if observation.landmark_id is not None:
            landmark_id = int(observation.landmark_id)
            for rank, candidate_landmark_id in enumerate(ids):
                if int(candidate_landmark_id) == landmark_id:
                    return row_idx, rank
    return None
