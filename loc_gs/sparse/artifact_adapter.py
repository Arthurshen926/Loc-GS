from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.sparse.rerank import summarize_candidate_availability


@dataclass(frozen=True)
class CachedCandidateArtifact:
    scene: str
    split_name: str
    source_path: str
    artifact_format: str
    topk: int
    batches: Sequence[SparseCandidateBatch]
    metadata: Mapping[str, object]

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    @property
    def keypoint_count(self) -> int:
        return sum(batch.keypoint_count for batch in self.batches)

    def summarize_candidate_availability(self) -> dict[str, int | float | str]:
        rows_by_keypoint: list[list[dict[str, object]]] = []
        for batch in self.batches:
            correct_rows = batch.candidate_geometric_correct
            if correct_rows is None:
                continue
            for row in correct_rows:
                rows_by_keypoint.append([{"geometric_correct": bool(value)} for value in row])
        summary = summarize_candidate_availability(rows_by_keypoint)
        topk_lengths = [length for batch in self.batches for length in batch.topk_lengths()]
        return {
            "scene": self.scene,
            "split_name": self.split_name,
            "artifact_format": self.artifact_format,
            "batch_count": int(self.batch_count),
            "keypoint_count": int(self.keypoint_count),
            "query_count": int(summary["query_count"]),
            "topk": int(self.topk),
            "mean_topk": float(sum(topk_lengths) / max(1, len(topk_lengths))),
            "top1_correct": int(summary["top1_correct"]),
            "topk_available": int(summary["topk_available"]),
            "oracle_gap": int(summary["oracle_gap"]),
            "positive_ratio": float(summary["topk_available"] / max(1, summary["query_count"])),
        }


def _load_torch_payload(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is present in the project environment.
        raise RuntimeError("torch is required to read cached candidate .pt artifacts") from exc
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate artifact must contain a mapping payload: {path}")
    return payload


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    meta = payload.get("metadata", {})
    if not isinstance(meta, Mapping):
        return {}
    return dict(meta)


def _field(payload: Mapping[str, Any], key: str, *, required: bool = True) -> Any:
    if key not in payload:
        if required:
            raise ValueError(f"candidate artifact is missing required field: {key}")
        return None
    return payload[key]


def _first_field(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _rows(value: Any, *, max_rows: int | None = None, row_indices: Sequence[int] | None = None) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if row_indices is not None:
        if hasattr(value, "index_select"):
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - torch is present in the project environment.
                raise RuntimeError("torch is required to row-filter cached candidate tensors") from exc
            value = value.index_select(0, torch.as_tensor(list(row_indices), dtype=torch.long))
        else:
            value = [value[int(idx)] for idx in row_indices]
    elif max_rows is not None:
        value = value[: int(max_rows)]
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _safe_metadata_value(value: Any) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe_metadata_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {"count": len(value), "preview": [_safe_metadata_value(v) for v in value[:5]]}
        return [_safe_metadata_value(v) for v in value]
    return str(value)


def _candidate_feature_row(value: Any, *, topk: int, default: float) -> list[float]:
    if isinstance(value, (str, bytes)):
        return [float(default)] * int(topk)
    if isinstance(value, Sequence):
        values = list(value)
        if len(values) == int(topk):
            return [float(item) for item in values]
        if len(values) == 1:
            return [float(values[0])] * int(topk)
        raise ValueError(f"candidate feature row length must be 1 or topk={topk}, got {len(values)}")
    return [float(value)] * int(topk)


def _sanitized_metadata(meta: Mapping[str, Any]) -> dict[str, object]:
    return {str(key): _safe_metadata_value(value) for key, value in meta.items()}


def _reject_unsafe_split(payload: Mapping[str, Any], meta: Mapping[str, Any], *, purpose: str) -> str:
    candidate_splits = [
        payload.get("split_name"),
        meta.get("source_split_name"),
        meta.get("split_name"),
        meta.get("feedback_bank_split_name"),
    ]
    split = next((str(item) for item in candidate_splits if item), "unknown")
    for item in candidate_splits:
        if item:
            reject_test_split(str(item), purpose=purpose)

    audits = [payload.get("split_audit"), meta.get("split_audit")]
    for audit in audits:
        if not isinstance(audit, Mapping):
            continue
        if bool(audit.get("official_test_used")) or bool(audit.get("test_split_used")):
            raise ValueError(f"test split is not allowed for {purpose}: split_audit marks test usage")
        if str(audit.get("audit_status", "")).lower() == "failed":
            raise ValueError(f"candidate artifact split audit failed for {purpose}")
        checks = audit.get("checks", {})
        if isinstance(checks, Mapping):
            feedback_bank = checks.get("feedback_bank_split", {})
            if isinstance(feedback_bank, Mapping) and feedback_bank.get("split_name"):
                reject_test_split(str(feedback_bank["split_name"]), purpose=purpose)
    return reject_test_split(split, purpose=purpose)


def _infer_image_id(query_id: str) -> str:
    if "::" in query_id:
        return query_id.split("::", 1)[0]
    return query_id


def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _selected_row_indices(
    *,
    full_query_ids: Sequence[str],
    full_image_ids: Sequence[str],
    requested_query_ids: Sequence[str] | None,
    max_batches: int | None,
    max_rows: int | None,
) -> list[int] | None:
    if len(full_query_ids) != len(full_image_ids):
        raise ValueError("query_id and image_id must have the same row count before filtering")
    if requested_query_ids is None and max_batches is None and max_rows is None:
        return None
    if requested_query_ids is not None:
        requested = [str(query_id) for query_id in requested_query_ids]
        allowed_query_ids = set(requested)
        requested_images = _unique_ordered(_infer_image_id(query_id) for query_id in requested)
        if max_batches is not None:
            requested_images = requested_images[: max(0, int(max_batches))]
            allowed_query_ids = {query_id for query_id in allowed_query_ids if _infer_image_id(query_id) in requested_images}
        allowed_images = set(requested_images)
        indices = [
            row_idx
            for row_idx, (query_id, image_id) in enumerate(zip(full_query_ids, full_image_ids))
            if str(query_id) in allowed_query_ids or str(image_id) in allowed_images
        ]
    elif max_batches is not None:
        allowed_images = set(_unique_ordered(full_image_ids)[: max(0, int(max_batches))])
        indices = [row_idx for row_idx, image_id in enumerate(full_image_ids) if str(image_id) in allowed_images]
    else:
        indices = list(range(len(full_query_ids)))
    if max_rows is not None:
        indices = indices[: max(0, int(max_rows))]
    return indices


def load_listwise_candidate_artifact(
    path: str | Path,
    *,
    max_rows: int | None = None,
    query_ids: Sequence[str] | None = None,
    max_batches: int | None = None,
) -> CachedCandidateArtifact:
    source_path = Path(path)
    payload = _load_torch_payload(source_path)
    meta = _metadata(payload)
    artifact_format = str(meta.get("format") or payload.get("format") or "listwise")
    if artifact_format != "listwise":
        raise ValueError(f"unsupported candidate artifact format: {artifact_format}")
    scene = str(payload.get("scene") or meta.get("scene") or "unknown")
    split_name = _reject_unsafe_split(payload, meta, purpose="internal sparse candidate adapter")

    image_ids_field = _field(payload, "image_id", required=False)
    full_query_ids = [str(v) for v in _rows(_field(payload, "query_id"))]
    full_image_ids = (
        [str(v) for v in _rows(image_ids_field)]
        if image_ids_field is not None
        else [_infer_image_id(qid) for qid in full_query_ids]
    )
    row_indices = _selected_row_indices(
        full_query_ids=full_query_ids,
        full_image_ids=full_image_ids,
        requested_query_ids=query_ids,
        max_batches=max_batches,
        max_rows=max_rows,
    )

    query_yx = _rows(_field(payload, "query_yx"), row_indices=row_indices)
    landmark_ids = _rows(_field(payload, "landmark_id"), row_indices=row_indices)
    scores = _rows(_field(payload, "cosine"), row_indices=row_indices)
    labels = [int(v) for v in _rows(_field(payload, "label"), row_indices=row_indices)]
    artifact_query_ids = [str(v) for v in _rows(_field(payload, "query_id"), row_indices=row_indices)]
    keypoint_ids_field = _field(payload, "keypoint_id", required=False)
    phase_field = _field(payload, "source_phase", required=False)
    masks_field = _field(payload, "candidate_mask", required=False)
    query_desc_field = _first_field(payload, ("query_desc", "query_descriptor", "candidate_query_desc"))
    landmark_desc_field = _first_field(payload, ("landmark_desc", "landmark_descriptor", "candidate_landmark_desc"))
    dense_consistent_field = _first_field(payload, ("dense_consistent", "candidate_dense_consistent"))
    sparse_inlier_field = _first_field(payload, ("sparse_inlier", "candidate_sparse_inlier"))
    reprojection_error_field = _first_field(
        payload,
        ("reprojection_error", "reprojection_error_px", "candidate_reprojection_error_px"),
    )
    solver_weight_field = _first_field(payload, ("solver_weight", "distill_weight", "candidate_solver_weight"))
    label_roles_field = _first_field(payload, ("label_roles", "label_role", "candidate_label_roles"))
    margin_field = _first_field(payload, ("margin", "descriptor_margin", "candidate_margin"))
    query_score_field = _first_field(payload, ("query_score", "detector_score", "candidate_query_score"))
    landmark_prior_field = _first_field(payload, ("landmark_prior", "candidate_landmark_prior"))
    image_ids = (
        [str(v) for v in _rows(image_ids_field, row_indices=row_indices)]
        if image_ids_field is not None
        else [_infer_image_id(qid) for qid in artifact_query_ids]
    )
    keypoint_ids = (
        [str(v) for v in _rows(keypoint_ids_field, row_indices=row_indices)]
        if keypoint_ids_field is not None
        else [qid.rsplit("::", 1)[-1] for qid in artifact_query_ids]
    )
    phases = (
        [str(v) for v in _rows(phase_field, row_indices=row_indices)]
        if phase_field is not None
        else [split_name] * len(artifact_query_ids)
    )
    masks = _rows(masks_field, row_indices=row_indices) if masks_field is not None else None
    query_descriptors = _rows(query_desc_field, row_indices=row_indices) if query_desc_field is not None else None
    landmark_descriptors = _rows(landmark_desc_field, row_indices=row_indices) if landmark_desc_field is not None else None
    dense_consistent = _rows(dense_consistent_field, row_indices=row_indices) if dense_consistent_field is not None else None
    sparse_inlier = _rows(sparse_inlier_field, row_indices=row_indices) if sparse_inlier_field is not None else None
    reprojection_error = (
        _rows(reprojection_error_field, row_indices=row_indices) if reprojection_error_field is not None else None
    )
    solver_weight = _rows(solver_weight_field, row_indices=row_indices) if solver_weight_field is not None else None
    label_roles = _rows(label_roles_field, row_indices=row_indices) if label_roles_field is not None else None
    margin = _rows(margin_field, row_indices=row_indices) if margin_field is not None else None
    query_score = _rows(query_score_field, row_indices=row_indices) if query_score_field is not None else None
    landmark_prior = _rows(landmark_prior_field, row_indices=row_indices) if landmark_prior_field is not None else None

    row_count = len(query_yx)
    lengths = {
        "landmark_id": len(landmark_ids),
        "cosine": len(scores),
        "label": len(labels),
        "query_id": len(artifact_query_ids),
        "image_id": len(image_ids),
        "keypoint_id": len(keypoint_ids),
        "source_phase": len(phases),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"candidate artifact field length mismatch: query_yx={row_count}, lengths={lengths}")
    optional_row_fields = {
        "candidate_mask": masks,
        "query_desc": query_descriptors,
        "landmark_desc": landmark_descriptors,
        "dense_consistent": dense_consistent,
        "sparse_inlier": sparse_inlier,
        "reprojection_error": reprojection_error,
        "solver_weight": solver_weight,
        "label_roles": label_roles,
        "margin": margin,
        "query_score": query_score,
        "landmark_prior": landmark_prior,
    }
    for name, value in optional_row_fields.items():
        if value is not None and len(value) != row_count:
            raise ValueError(f"{name} must have the same row count as query_yx")

    grouped: "OrderedDict[str, dict[str, list[Any]]]" = OrderedDict()
    for row_idx in range(row_count):
        image_id = image_ids[row_idx]
        ids = [int(v) for v in landmark_ids[row_idx]]
        row_scores = [float(v) for v in scores[row_idx]]
        if len(ids) != len(row_scores):
            raise ValueError(f"candidate id/score length mismatch at row {row_idx}")
        label = int(labels[row_idx])
        teacher_label = label if 0 <= label < len(ids) else None
        valid_mask = [bool(v) for v in masks[row_idx]] if masks is not None else [True] * len(ids)
        correct = [idx == teacher_label for idx in range(len(ids))]
        yx = query_yx[row_idx]
        if len(yx) < 2:
            raise ValueError(f"query_yx row {row_idx} must contain y and x")
        bucket = grouped.setdefault(
            image_id,
            {
                "keypoint_xy": [],
                "query_descriptors": [],
                "candidate_landmark_ids": [],
                "candidate_scores": [],
                "candidate_landmark_descriptors": [],
                "teacher_labels": [],
                "candidate_valid_mask": [],
                "candidate_geometric_correct": [],
                "candidate_dense_consistent": [],
                "candidate_sparse_inlier": [],
                "candidate_reprojection_error_px": [],
                "candidate_solver_weight": [],
                "candidate_label_roles": [],
                "candidate_margin": [],
                "candidate_query_score": [],
                "candidate_landmark_prior": [],
                "source_keypoint_ids": [],
                "source_phases": [],
            },
        )
        bucket["keypoint_xy"].append([float(yx[1]), float(yx[0])])
        if query_descriptors is not None:
            bucket["query_descriptors"].append([float(v) for v in query_descriptors[row_idx]])
        bucket["candidate_landmark_ids"].append(ids)
        bucket["candidate_scores"].append(row_scores)
        if landmark_descriptors is not None:
            bucket["candidate_landmark_descriptors"].append(
                [[float(v) for v in descriptor] for descriptor in landmark_descriptors[row_idx]]
            )
        bucket["teacher_labels"].append(teacher_label)
        bucket["candidate_valid_mask"].append(valid_mask)
        bucket["candidate_geometric_correct"].append(correct)
        if dense_consistent is not None:
            bucket["candidate_dense_consistent"].append([bool(v) for v in dense_consistent[row_idx]])
        if sparse_inlier is not None:
            bucket["candidate_sparse_inlier"].append([bool(v) for v in sparse_inlier[row_idx]])
        if reprojection_error is not None:
            bucket["candidate_reprojection_error_px"].append([float(v) for v in reprojection_error[row_idx]])
        if solver_weight is not None:
            bucket["candidate_solver_weight"].append([float(v) for v in solver_weight[row_idx]])
        if label_roles is not None:
            bucket["candidate_label_roles"].append([str(v) for v in label_roles[row_idx]])
        if margin is not None:
            bucket["candidate_margin"].append(_candidate_feature_row(margin[row_idx], topk=len(ids), default=0.0))
        if query_score is not None:
            bucket["candidate_query_score"].append(
                _candidate_feature_row(query_score[row_idx], topk=len(ids), default=1.0)
            )
        if landmark_prior is not None:
            bucket["candidate_landmark_prior"].append(
                _candidate_feature_row(landmark_prior[row_idx], topk=len(ids), default=0.0)
            )
        bucket["source_keypoint_ids"].append(keypoint_ids[row_idx])
        bucket["source_phases"].append(phases[row_idx])

    topk = int(meta.get("topk") or (len(landmark_ids[0]) if landmark_ids else 0))
    batches = [
        SparseCandidateBatch(
            scene=scene,
            split_name=split_name,
            query_id=image_id,
            keypoint_xy=bucket["keypoint_xy"],
            query_descriptors=bucket["query_descriptors"] or None,
            candidate_landmark_ids=bucket["candidate_landmark_ids"],
            candidate_scores=bucket["candidate_scores"],
            candidate_landmark_descriptors=bucket["candidate_landmark_descriptors"] or None,
            teacher_labels=bucket["teacher_labels"],
            candidate_valid_mask=bucket["candidate_valid_mask"],
            candidate_geometric_correct=bucket["candidate_geometric_correct"],
            candidate_dense_consistent=bucket["candidate_dense_consistent"] or None,
            candidate_sparse_inlier=bucket["candidate_sparse_inlier"] or None,
            candidate_reprojection_error_px=bucket["candidate_reprojection_error_px"] or None,
            candidate_solver_weight=bucket["candidate_solver_weight"] or None,
            candidate_label_roles=bucket["candidate_label_roles"] or None,
            candidate_margin=bucket["candidate_margin"] or None,
            candidate_query_score=bucket["candidate_query_score"] or None,
            candidate_landmark_prior=bucket["candidate_landmark_prior"] or None,
            source_keypoint_ids=bucket["source_keypoint_ids"],
            source_phases=bucket["source_phases"],
            metadata={"source_path": str(source_path), "artifact_format": artifact_format},
        )
        for image_id, bucket in grouped.items()
    ]
    for batch in batches:
        batch.validate()
    return CachedCandidateArtifact(
        scene=scene,
        split_name=split_name,
        source_path=str(source_path),
        artifact_format=artifact_format,
        topk=topk,
        batches=batches,
        metadata=_sanitized_metadata(meta),
    )


def _batch_to_json_dict(batch: SparseCandidateBatch) -> dict[str, object]:
    batch.validate()
    return {
        "scene": batch.scene,
        "split_name": batch.split_name,
        "query_id": batch.query_id,
        "keypoint_xy": [[float(v) for v in xy[:2]] for xy in batch.keypoint_xy],
        "query_descriptors": (
            [[float(v) for v in desc] for desc in batch.query_descriptors]
            if batch.query_descriptors is not None
            else None
        ),
        "candidate_landmark_ids": [[int(v) for v in row] for row in batch.candidate_landmark_ids],
        "candidate_scores": [[float(v) for v in row] for row in batch.candidate_scores],
        "candidate_landmark_descriptors": (
            [
                [[float(v) for v in descriptor] for descriptor in row]
                for row in batch.candidate_landmark_descriptors
            ]
            if batch.candidate_landmark_descriptors is not None
            else None
        ),
        "teacher_labels": list(batch.teacher_labels) if batch.teacher_labels is not None else None,
        "candidate_valid_mask": batch.candidate_valid_mask,
        "candidate_geometric_correct": batch.candidate_geometric_correct,
        "candidate_dense_consistent": batch.candidate_dense_consistent,
        "candidate_sparse_inlier": batch.candidate_sparse_inlier,
        "candidate_reprojection_error_px": batch.candidate_reprojection_error_px,
        "candidate_solver_weight": batch.candidate_solver_weight,
        "candidate_label_roles": batch.candidate_label_roles,
        "candidate_margin": batch.candidate_margin,
        "candidate_query_score": batch.candidate_query_score,
        "candidate_landmark_prior": batch.candidate_landmark_prior,
        "source_keypoint_ids": list(batch.source_keypoint_ids) if batch.source_keypoint_ids is not None else None,
        "source_phases": list(batch.source_phases) if batch.source_phases is not None else None,
        "metadata": dict(batch.metadata or {}),
    }


def write_candidate_batches_jsonl(
    batches: Iterable[SparseCandidateBatch],
    output_path: str | Path,
    *,
    max_batches: int | None = None,
) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            if max_batches is not None and written >= int(max_batches):
                break
            handle.write(json.dumps(_batch_to_json_dict(batch), sort_keys=True) + "\n")
            written += 1
    return written
