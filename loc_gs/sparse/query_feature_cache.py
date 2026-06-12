from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split


def build_query_feature_cache_from_listwise_artifact(
    *,
    source_artifact: str | Path,
    output_cache: str | Path,
    scene: str,
    split_name: str,
    query_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal query feature cache")
    artifact = load_listwise_candidate_artifact(source_artifact)
    if artifact.scene not in {"unknown", str(scene)}:
        raise ValueError(f"source artifact scene mismatch: expected {scene}, got {artifact.scene}")
    payload = _load_torch_mapping(source_artifact)
    image_ids = [str(value) for value in _as_list(_required(payload, "image_id"))]
    keypoint_ids = [str(value) for value in _as_list(payload.get("keypoint_id", [f"kp_{idx:06d}" for idx in range(len(image_ids))]))]
    query_desc = torch.as_tensor(_required(payload, "query_desc"), dtype=torch.float32)
    query_yx = torch.as_tensor(_required(payload, "query_yx"), dtype=torch.float32)
    query_score = torch.as_tensor(payload.get("query_score", torch.ones(len(image_ids))), dtype=torch.float32)
    row_count = len(image_ids)
    lengths = {
        "keypoint_id": len(keypoint_ids),
        "query_desc": int(query_desc.shape[0]),
        "query_yx": int(query_yx.shape[0]),
        "query_score": int(query_score.shape[0]),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"source artifact field length mismatch: image_id={row_count}, lengths={lengths}")
    requested = [str(query_id) for query_id in query_ids] if query_ids is not None else None
    requested_set = set(requested) if requested is not None else None
    selected_indices = [
        row_idx for row_idx, image_id in enumerate(image_ids) if requested_set is None or image_id in requested_set
    ]
    covered_queries = _dedupe(image_ids[row_idx] for row_idx in selected_indices)
    missing_queries = [query_id for query_id in requested or [] if query_id not in set(covered_queries)]
    selected = torch.as_tensor(selected_indices, dtype=torch.long)
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, Mapping):
        split_audit = {"audit_status": "unknown"}
    output = {
        "metadata": {
            "schema_version": "internal_query_feature_cache_v1",
            "scene": str(scene),
            "split_name": split,
            "source": "internal_query_feature_cache_builder",
            "source_artifact": str(source_artifact),
            "split_audit": dict(split_audit),
        },
        "image_id": [image_ids[idx] for idx in selected_indices],
        "keypoint_id": [keypoint_ids[idx] for idx in selected_indices],
        "query_yx": query_yx.index_select(0, selected) if selected.numel() else query_yx[:0],
        "query_desc": query_desc.index_select(0, selected) if selected.numel() else query_desc[:0],
        "query_score": query_score.index_select(0, selected) if selected.numel() else query_score[:0],
    }
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return {
        "schema_version": "internal_query_feature_cache_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "source_artifact": str(source_artifact),
        "output_cache": str(output_path),
        "requested_query_count": None if requested is None else int(len(requested)),
        "query_count": int(len(covered_queries)),
        "keypoint_count": int(len(selected_indices)),
        "missing_query_count": None if requested is None else int(len(missing_queries)),
        "missing_query_ids_preview": missing_queries[:10],
        "descriptor_dim": int(query_desc.shape[1]) if query_desc.ndim == 2 else 0,
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }


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


def _dedupe(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = str(raw_value)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
