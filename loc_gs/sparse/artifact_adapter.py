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


def _rows(value: Any, *, max_rows: int | None = None) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if max_rows is not None:
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


def load_listwise_candidate_artifact(path: str | Path, *, max_rows: int | None = None) -> CachedCandidateArtifact:
    source_path = Path(path)
    payload = _load_torch_payload(source_path)
    meta = _metadata(payload)
    artifact_format = str(meta.get("format") or payload.get("format") or "listwise")
    if artifact_format != "listwise":
        raise ValueError(f"unsupported candidate artifact format: {artifact_format}")
    scene = str(payload.get("scene") or meta.get("scene") or "unknown")
    split_name = _reject_unsafe_split(payload, meta, purpose="internal sparse candidate adapter")

    query_yx = _rows(_field(payload, "query_yx"), max_rows=max_rows)
    landmark_ids = _rows(_field(payload, "landmark_id"), max_rows=max_rows)
    scores = _rows(_field(payload, "cosine"), max_rows=max_rows)
    labels = [int(v) for v in _rows(_field(payload, "label"), max_rows=max_rows)]
    query_ids = [str(v) for v in _rows(_field(payload, "query_id"), max_rows=max_rows)]
    image_ids_field = _field(payload, "image_id", required=False)
    keypoint_ids_field = _field(payload, "keypoint_id", required=False)
    phase_field = _field(payload, "source_phase", required=False)
    masks_field = _field(payload, "candidate_mask", required=False)
    image_ids = (
        [str(v) for v in _rows(image_ids_field, max_rows=max_rows)]
        if image_ids_field is not None
        else [_infer_image_id(qid) for qid in query_ids]
    )
    keypoint_ids = (
        [str(v) for v in _rows(keypoint_ids_field, max_rows=max_rows)]
        if keypoint_ids_field is not None
        else [qid.rsplit("::", 1)[-1] for qid in query_ids]
    )
    phases = (
        [str(v) for v in _rows(phase_field, max_rows=max_rows)] if phase_field is not None else [split_name] * len(query_ids)
    )
    masks = _rows(masks_field, max_rows=max_rows) if masks_field is not None else None

    row_count = len(query_yx)
    lengths = {
        "landmark_id": len(landmark_ids),
        "cosine": len(scores),
        "label": len(labels),
        "query_id": len(query_ids),
        "image_id": len(image_ids),
        "keypoint_id": len(keypoint_ids),
        "source_phase": len(phases),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"candidate artifact field length mismatch: query_yx={row_count}, lengths={lengths}")
    if masks is not None and len(masks) != row_count:
        raise ValueError("candidate_mask must have the same row count as query_yx")

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
                "candidate_landmark_ids": [],
                "candidate_scores": [],
                "teacher_labels": [],
                "candidate_valid_mask": [],
                "candidate_geometric_correct": [],
                "source_keypoint_ids": [],
                "source_phases": [],
            },
        )
        bucket["keypoint_xy"].append([float(yx[1]), float(yx[0])])
        bucket["candidate_landmark_ids"].append(ids)
        bucket["candidate_scores"].append(row_scores)
        bucket["teacher_labels"].append(teacher_label)
        bucket["candidate_valid_mask"].append(valid_mask)
        bucket["candidate_geometric_correct"].append(correct)
        bucket["source_keypoint_ids"].append(keypoint_ids[row_idx])
        bucket["source_phases"].append(phases[row_idx])

    topk = int(meta.get("topk") or (len(landmark_ids[0]) if landmark_ids else 0))
    batches = [
        SparseCandidateBatch(
            scene=scene,
            split_name=split_name,
            query_id=image_id,
            keypoint_xy=bucket["keypoint_xy"],
            candidate_landmark_ids=bucket["candidate_landmark_ids"],
            candidate_scores=bucket["candidate_scores"],
            teacher_labels=bucket["teacher_labels"],
            candidate_valid_mask=bucket["candidate_valid_mask"],
            candidate_geometric_correct=bucket["candidate_geometric_correct"],
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
        "candidate_landmark_ids": [[int(v) for v in row] for row in batch.candidate_landmark_ids],
        "candidate_scores": [[float(v) for v in row] for row in batch.candidate_scores],
        "teacher_labels": list(batch.teacher_labels) if batch.teacher_labels is not None else None,
        "candidate_valid_mask": batch.candidate_valid_mask,
        "candidate_geometric_correct": batch.candidate_geometric_correct,
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
