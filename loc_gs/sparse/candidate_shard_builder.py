from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split


def build_candidate_shard_artifact(
    *,
    completion_shard: Mapping[str, Any],
    query_feature_cache: str | Path,
    base_candidate_artifact: str | Path,
    output_artifact: str | Path,
    scene: str,
    split_name: str,
    topk: int,
    max_landmarks: int | None = None,
    landmark_chunk_size: int | None = None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate shard artifact")
    shard_split = reject_test_split(
        str(completion_shard.get("split_name") or split),
        purpose="internal candidate shard artifact",
    )
    if shard_split != split:
        raise ValueError(f"completion shard split mismatch: expected {split}, got {shard_split}")
    shard_scene = str(completion_shard.get("scene") or scene)
    if shard_scene != str(scene):
        raise ValueError(f"completion shard scene mismatch: expected {scene}, got {shard_scene}")
    requested_query_ids = [str(value) for value in completion_shard.get("query_ids", [])]
    if not requested_query_ids:
        raise ValueError("completion shard must contain at least one query id")

    query_cache = _load_torch_mapping(query_feature_cache)
    base_payload = _load_torch_mapping(base_candidate_artifact)
    base_artifact = load_listwise_candidate_artifact(base_candidate_artifact, max_rows=1)
    base_split_audit = base_artifact.metadata.get("split_audit")
    if not isinstance(base_split_audit, Mapping):
        base_split_audit = {"audit_status": "unknown"}
    query_rows = _select_query_feature_rows(query_cache, requested_query_ids)
    missing_queries = [query_id for query_id in requested_query_ids if query_id not in {row["image_id"] for row in query_rows}]
    if not query_rows:
        raise ValueError("query feature cache does not contain any requested completion shard queries")

    base_desc = torch.as_tensor(_required(base_payload, "base_landmark_desc"), dtype=torch.float32)
    base_gaussian_id = torch.as_tensor(_required(base_payload, "base_gaussian_id"), dtype=torch.long).cpu()
    if max_landmarks is not None:
        limit = min(int(max_landmarks), int(base_desc.shape[0]))
        base_desc = base_desc[:limit].clone()
        base_gaussian_id = base_gaussian_id[:limit].clone()
    if base_desc.ndim != 2:
        raise ValueError("base_landmark_desc must have shape [num_landmarks, descriptor_dim]")
    query_desc = torch.stack([row["query_desc"] for row in query_rows], dim=0).float()
    if query_desc.ndim != 2 or query_desc.shape[1] != base_desc.shape[1]:
        raise ValueError("query_desc and base_landmark_desc descriptor dimensions must match")
    k = min(int(topk), int(base_desc.shape[0]))
    if k <= 0:
        raise ValueError("topk must be positive and base_landmark_desc must be non-empty")
    top_scores, top_indices = _compute_topk_scores(
        query_desc=query_desc,
        base_desc=base_desc,
        topk=k,
        landmark_chunk_size=landmark_chunk_size,
    )
    output = _build_listwise_payload(
        query_rows=query_rows,
        top_indices=top_indices.cpu(),
        top_scores=top_scores.cpu(),
        base_gaussian_id=base_gaussian_id,
        base_landmark_desc=base_desc.cpu(),
        scene=str(scene),
        split_name=split,
        completion_shard=completion_shard,
        base_candidate_artifact=base_candidate_artifact,
        query_feature_cache=query_feature_cache,
        landmark_chunk_size=landmark_chunk_size,
        split_audit=_split_audit(split, base_split_audit),
    )
    output_path = Path(output_artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    artifact = load_listwise_candidate_artifact(output_path)
    return {
        "schema_version": "internal_candidate_shard_artifact_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "completion_shard_id": str(completion_shard.get("shard_id") or "unknown"),
        "query_count": int(artifact.batch_count),
        "requested_query_count": int(len(requested_query_ids)),
        "missing_feature_query_count": int(len(missing_queries)),
        "missing_feature_query_ids_preview": missing_queries[:10],
        "keypoint_count": int(artifact.keypoint_count),
        "topk": int(k),
        "base_landmark_count": int(base_desc.shape[0]),
        "landmark_chunk_size": None if landmark_chunk_size is None else int(landmark_chunk_size),
        "output_artifact": str(output_path),
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }


def load_completion_shard(path: str | Path, *, shard_id: str | None = None) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"completion shard file is empty: {path}")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"completion shard JSON must be an object: {path}")
        return dict(payload)
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    objects = [dict(row) for row in rows if isinstance(row, Mapping)]
    if shard_id is None:
        if len(objects) != 1:
            raise ValueError("completion plan JSONL requires --shard_id when it contains multiple rows")
        return objects[0]
    for row in objects:
        if str(row.get("shard_id")) == str(shard_id):
            return row
    raise ValueError(f"completion shard id not found: {shard_id}")


def _load_torch_mapping(path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"torch artifact must contain a mapping: {path}")
    return payload


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"artifact is missing required field: {key}")
    return payload[key]


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _select_query_feature_rows(payload: Mapping[str, Any], requested_query_ids: Sequence[str]) -> list[dict[str, Any]]:
    requested = set(str(query_id) for query_id in requested_query_ids)
    image_ids = [str(value) for value in _as_list(_required(payload, "image_id"))]
    keypoint_ids = [str(value) for value in _as_list(payload.get("keypoint_id", [f"kp_{idx:06d}" for idx in range(len(image_ids))]))]
    query_yx = torch.as_tensor(_required(payload, "query_yx"), dtype=torch.float32)
    query_desc = torch.as_tensor(_required(payload, "query_desc"), dtype=torch.float32)
    query_score = torch.as_tensor(payload.get("query_score", torch.ones(len(image_ids))), dtype=torch.float32)
    row_count = len(image_ids)
    lengths = {
        "keypoint_id": len(keypoint_ids),
        "query_yx": int(query_yx.shape[0]),
        "query_desc": int(query_desc.shape[0]),
        "query_score": int(query_score.shape[0]),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"query feature cache field length mismatch: image_id={row_count}, lengths={lengths}")
    rows: list[dict[str, Any]] = []
    for row_idx, image_id in enumerate(image_ids):
        if image_id not in requested:
            continue
        rows.append(
            {
                "image_id": image_id,
                "keypoint_id": keypoint_ids[row_idx],
                "query_yx": query_yx[row_idx].cpu(),
                "query_desc": query_desc[row_idx].cpu(),
                "query_score": float(query_score[row_idx]),
            }
        )
    return rows


def _compute_topk_scores(
    *,
    query_desc: torch.Tensor,
    base_desc: torch.Tensor,
    topk: int,
    landmark_chunk_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = F.normalize(query_desc, p=2, dim=-1)
    landmark_count = int(base_desc.shape[0])
    k = min(int(topk), landmark_count)
    chunk_size = landmark_count if landmark_chunk_size is None else int(landmark_chunk_size)
    if chunk_size <= 0:
        raise ValueError("landmark_chunk_size must be positive when provided")
    if chunk_size >= landmark_count:
        scores = query @ F.normalize(base_desc, p=2, dim=-1).T
        return torch.topk(scores, k=k, dim=1)

    best_scores: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    for start in range(0, landmark_count, chunk_size):
        end = min(start + chunk_size, landmark_count)
        chunk = F.normalize(base_desc[start:end], p=2, dim=-1)
        scores = query @ chunk.T
        local_k = min(k, int(scores.shape[1]))
        chunk_scores, chunk_indices = torch.topk(scores, k=local_k, dim=1)
        chunk_indices = chunk_indices + int(start)
        if best_scores is None or best_indices is None:
            best_scores = chunk_scores
            best_indices = chunk_indices
            continue
        combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
        combined_indices = torch.cat([best_indices, chunk_indices], dim=1)
        best_scores, order = torch.topk(combined_scores, k=k, dim=1)
        best_indices = combined_indices.gather(1, order)
    if best_scores is None or best_indices is None:
        raise ValueError("base_landmark_desc must be non-empty")
    return best_scores, best_indices


def _build_listwise_payload(
    *,
    query_rows: Sequence[Mapping[str, Any]],
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    base_gaussian_id: torch.Tensor,
    base_landmark_desc: torch.Tensor,
    scene: str,
    split_name: str,
    completion_shard: Mapping[str, Any],
    base_candidate_artifact: str | Path,
    query_feature_cache: str | Path,
    landmark_chunk_size: int | None,
    split_audit: Mapping[str, object],
) -> dict[str, Any]:
    image_ids = [str(row["image_id"]) for row in query_rows]
    keypoint_ids = [str(row["keypoint_id"]) for row in query_rows]
    query_ids = [f"{image_id}::{keypoint_id}" for image_id, keypoint_id in zip(image_ids, keypoint_ids)]
    query_yx = torch.stack([torch.as_tensor(row["query_yx"], dtype=torch.float32) for row in query_rows], dim=0)
    query_desc = torch.stack([torch.as_tensor(row["query_desc"], dtype=torch.float32) for row in query_rows], dim=0)
    top_indices = top_indices.long()
    top_scores = top_scores.float()
    landmark_desc = base_landmark_desc[top_indices].float()
    margin = top_scores[:, 0] - top_scores[:, 1] if top_scores.shape[1] > 1 else top_scores[:, 0]
    row_count, topk = top_indices.shape
    return {
        "metadata": {
            "format": "listwise",
            "scene": scene,
            "source_split_name": split_name,
            "topk": int(topk),
            "source": "internal_candidate_shard_builder",
            "completion_shard_id": str(completion_shard.get("shard_id") or "unknown"),
            "base_candidate_artifact": str(base_candidate_artifact),
            "query_feature_cache": str(query_feature_cache),
            "landmark_chunk_size": None if landmark_chunk_size is None else int(landmark_chunk_size),
            "split_audit": dict(split_audit),
        },
        "base_gaussian_id": base_gaussian_id.long(),
        "base_landmark_desc": base_landmark_desc.float(),
        "query_desc": query_desc.float(),
        "landmark_desc": landmark_desc.float(),
        "cosine": top_scores,
        "margin": margin.float(),
        "query_score": torch.tensor([float(row["query_score"]) for row in query_rows], dtype=torch.float32),
        "landmark_prior": torch.zeros((row_count, topk), dtype=torch.float32),
        "candidate_mask": torch.ones((row_count, topk), dtype=torch.bool),
        "reprojection_error": torch.full((row_count, topk), float("inf"), dtype=torch.float32),
        "query_yx": query_yx.float(),
        "landmark_id": top_indices,
        "label": torch.full((row_count,), -1, dtype=torch.long),
        "query_id": query_ids,
        "image_id": image_ids,
        "keypoint_id": keypoint_ids,
        "source_phase": [split_name] * row_count,
    }


def _split_audit(split_name: str, base_split_audit: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed" if base_split_audit.get("audit_status") == "passed" else "unknown",
        "split_name": split_name,
        "official_test_used": False,
        "test_split_used": False,
        "source": "internal_candidate_shard_builder",
        "base_candidate_split_audit_status": str(base_split_audit.get("audit_status", "unknown")),
    }
