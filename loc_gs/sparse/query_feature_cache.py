from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from loc_gs.data.superpoint_cache import superpoint_score_map_from_logits
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


def build_query_feature_cache_from_feature_map_cache(
    *,
    feature_map_cache: str | Path,
    output_cache: str | Path,
    scene: str,
    split_name: str,
    query_ids: Sequence[str] | None = None,
    max_keypoints: int = 2048,
    score_threshold: float | None = None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal query feature cache from feature maps")
    payload = _load_torch_mapping(feature_map_cache)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    source_scene = str(metadata.get("scene") or payload.get("scene") or "unknown")
    if source_scene not in {"unknown", str(scene)}:
        raise ValueError(f"feature map cache scene mismatch: expected {scene}, got {source_scene}")
    source_split = str(metadata.get("split_name") or metadata.get("source_split_name") or payload.get("split_name") or "unknown")
    reject_test_split(source_split, purpose="internal query feature map cache")
    if source_split not in {"unknown", split}:
        raise ValueError(f"feature map cache split mismatch: expected {split}, got {source_split}")
    requested = [str(query_id) for query_id in query_ids] if query_ids is not None else None
    requested_set = set(requested) if requested is not None else None
    max_k = int(max_keypoints)
    if max_k <= 0:
        raise ValueError("max_keypoints must be positive")

    rows: list[dict[str, Any]] = []
    descriptor_dim = 0
    for entry in _feature_map_entries(payload):
        image_id = str(_required(entry, "image_id"))
        if requested_set is not None and image_id not in requested_set:
            continue
        descriptor_map = _descriptor_map(entry)
        descriptor_dim = descriptor_dim or int(descriptor_map.shape[0])
        keypoints_yx, keypoint_scores = _select_feature_map_keypoints(
            entry=entry,
            max_keypoints=max_k,
            score_threshold=score_threshold,
        )
        if keypoints_yx.numel() == 0:
            continue
        query_desc = _sample_descriptors_bilinear(descriptor_map, keypoints_yx)
        for row_idx in range(int(keypoints_yx.shape[0])):
            rows.append(
                {
                    "image_id": image_id,
                    "keypoint_id": f"kp_{row_idx:06d}",
                    "query_yx": keypoints_yx[row_idx].cpu(),
                    "query_desc": query_desc[row_idx].cpu(),
                    "query_score": float(keypoint_scores[row_idx]),
                }
            )

    covered_queries = _dedupe(row["image_id"] for row in rows)
    missing_queries = [query_id for query_id in requested or [] if query_id not in set(covered_queries)]
    query_yx = (
        torch.stack([torch.as_tensor(row["query_yx"], dtype=torch.float32) for row in rows], dim=0)
        if rows
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    query_desc = (
        torch.stack([torch.as_tensor(row["query_desc"], dtype=torch.float32) for row in rows], dim=0)
        if rows
        else torch.zeros((0, int(descriptor_dim)), dtype=torch.float32)
    )
    query_score = (
        torch.tensor([float(row["query_score"]) for row in rows], dtype=torch.float32)
        if rows
        else torch.zeros((0,), dtype=torch.float32)
    )
    split_audit = metadata.get("split_audit")
    if not isinstance(split_audit, Mapping):
        split_audit = {"audit_status": "unknown"}
    output = {
        "metadata": {
            "schema_version": "internal_query_feature_cache_v1",
            "scene": str(scene),
            "split_name": split,
            "source": "internal_feature_map_cache_sampler",
            "source_feature_map_cache": str(feature_map_cache),
            "split_audit": dict(split_audit),
        },
        "image_id": [str(row["image_id"]) for row in rows],
        "keypoint_id": [str(row["keypoint_id"]) for row in rows],
        "query_yx": query_yx.float(),
        "query_desc": query_desc.float(),
        "query_score": query_score.float(),
    }
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return {
        "schema_version": "internal_query_feature_cache_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "source_feature_map_cache": str(feature_map_cache),
        "output_cache": str(output_path),
        "requested_query_count": None if requested is None else int(len(requested)),
        "query_count": int(len(covered_queries)),
        "keypoint_count": int(len(rows)),
        "missing_query_count": None if requested is None else int(len(missing_queries)),
        "missing_query_ids_preview": missing_queries[:10],
        "descriptor_dim": int(descriptor_dim),
        "max_keypoints": int(max_k),
        "score_threshold": None if score_threshold is None else float(score_threshold),
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }


def build_query_feature_cache_from_superpoint_dir(
    *,
    superpoint_dir: str | Path,
    output_cache: str | Path,
    scene: str,
    split_name: str,
    query_ids: Sequence[str] | None = None,
    image_id_prefix: str = "",
    max_keypoints: int = 2048,
    score_threshold: float | None = None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal query feature cache from SuperPoint dir")
    source_dir = Path(superpoint_dir)
    manifest = _load_superpoint_frame_manifest(source_dir)
    source_scene = str(manifest.get("scene") or "unknown")
    if source_scene not in {"unknown", str(scene)}:
        raise ValueError(f"SuperPoint feature dir scene mismatch: expected {scene}, got {source_scene}")
    frames = manifest.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("SuperPoint frame_manifest.json is missing frames")
    requested = [str(query_id) for query_id in query_ids] if query_ids is not None else None
    requested_set = set(requested) if requested is not None else None
    max_k = int(max_keypoints)
    if max_k <= 0:
        raise ValueError("max_keypoints must be positive")

    rows: list[dict[str, Any]] = []
    descriptor_dim = int(manifest.get("descriptor_dim") or 0)
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("SuperPoint frame entries must be mappings")
        image_id = _superpoint_image_id(frame, image_id_prefix=image_id_prefix)
        if requested_set is not None and image_id not in requested_set:
            continue
        saved_stem = str(_required(frame, "saved_stem"))
        descriptor_map = _load_superpoint_descriptor_map(source_dir / "descriptor" / f"{saved_stem}.pt")
        detector_logits = _load_superpoint_detector_logits(source_dir / "detector" / f"{saved_stem}.pt")
        descriptor_dim = descriptor_dim or int(descriptor_map.shape[0])
        score_map = torch.as_tensor(superpoint_score_map_from_logits(detector_logits), dtype=torch.float32)
        keypoints_yx, keypoint_scores = _select_feature_map_keypoints(
            entry={"score_map": score_map},
            max_keypoints=max_k,
            score_threshold=score_threshold,
        )
        if keypoints_yx.numel() == 0:
            continue
        descriptor_grid_yx = keypoints_yx / 8.0
        query_desc = _sample_descriptors_bilinear(descriptor_map, descriptor_grid_yx)
        query_desc = F.normalize(query_desc, p=2, dim=1)
        for row_idx in range(int(keypoints_yx.shape[0])):
            rows.append(
                {
                    "image_id": image_id,
                    "keypoint_id": f"kp_{row_idx:06d}",
                    "query_yx": keypoints_yx[row_idx].cpu(),
                    "query_desc": query_desc[row_idx].cpu(),
                    "query_score": float(keypoint_scores[row_idx]),
                }
            )

    covered_queries = _dedupe(row["image_id"] for row in rows)
    missing_queries = [query_id for query_id in requested or [] if query_id not in set(covered_queries)]
    query_yx = (
        torch.stack([torch.as_tensor(row["query_yx"], dtype=torch.float32) for row in rows], dim=0)
        if rows
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    query_desc = (
        torch.stack([torch.as_tensor(row["query_desc"], dtype=torch.float32) for row in rows], dim=0)
        if rows
        else torch.zeros((0, int(descriptor_dim)), dtype=torch.float32)
    )
    query_score = (
        torch.tensor([float(row["query_score"]) for row in rows], dtype=torch.float32)
        if rows
        else torch.zeros((0,), dtype=torch.float32)
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    output = {
        "metadata": {
            "schema_version": "internal_query_feature_cache_v1",
            "scene": str(scene),
            "split_name": split,
            "source": "internal_superpoint_feature_dir_sampler",
            "source_superpoint_dir": str(source_dir),
            "source_frame_manifest": str(source_dir / "frame_manifest.json"),
            "image_id_prefix": str(image_id_prefix),
            "split_audit": dict(split_audit),
        },
        "image_id": [str(row["image_id"]) for row in rows],
        "keypoint_id": [str(row["keypoint_id"]) for row in rows],
        "query_yx": query_yx.float(),
        "query_desc": query_desc.float(),
        "query_score": query_score.float(),
    }
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return {
        "schema_version": "internal_query_feature_cache_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "source_superpoint_dir": str(source_dir),
        "source_frame_manifest": str(source_dir / "frame_manifest.json"),
        "output_cache": str(output_path),
        "requested_query_count": None if requested is None else int(len(requested)),
        "query_count": int(len(covered_queries)),
        "keypoint_count": int(len(rows)),
        "missing_query_count": None if requested is None else int(len(missing_queries)),
        "missing_query_ids_preview": missing_queries[:10],
        "descriptor_dim": int(descriptor_dim),
        "max_keypoints": int(max_k),
        "score_threshold": None if score_threshold is None else float(score_threshold),
        "image_id_prefix": str(image_id_prefix),
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


def _feature_map_entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = payload.get("entries")
    if entries is None:
        entries = payload.get("feature_maps")
    if entries is None:
        raise ValueError("feature map cache is missing entries")
    if isinstance(entries, Mapping):
        return [dict(entry, image_id=str(image_id)) for image_id, entry in entries.items() if isinstance(entry, Mapping)]
    result: list[Mapping[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("feature map cache entries must be mappings")
        result.append(entry)
    return result


def _load_superpoint_frame_manifest(source_dir: Path) -> Mapping[str, Any]:
    manifest_path = source_dir / "frame_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"SuperPoint feature dir is missing frame_manifest.json: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"SuperPoint frame manifest must contain a mapping: {manifest_path}")
    return payload


def _superpoint_image_id(frame: Mapping[str, Any], *, image_id_prefix: str) -> str:
    source_file = str(_required(frame, "source_file"))
    prefix = str(image_id_prefix).strip("/")
    return f"{prefix}/{source_file}" if prefix else source_file


def _load_superpoint_descriptor_map(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"SuperPoint descriptor file is missing: {path}")
    descriptor = torch.as_tensor(torch.load(path, map_location="cpu"), dtype=torch.float32)
    if descriptor.ndim == 4 and descriptor.shape[0] == 1:
        descriptor = descriptor[0]
    if descriptor.ndim != 3:
        raise ValueError("SuperPoint descriptor must have shape [C,H,W] or [1,C,H,W]")
    return descriptor.cpu()


def _load_superpoint_detector_logits(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"SuperPoint detector file is missing: {path}")
    detector = torch.as_tensor(torch.load(path, map_location="cpu"), dtype=torch.float32)
    if detector.ndim == 4 and detector.shape[0] == 1:
        detector = detector[0]
    if detector.ndim != 3 or int(detector.shape[0]) != 65:
        raise ValueError("SuperPoint detector logits must have shape [65,H,W] or [1,65,H,W]")
    return detector.cpu()


def _descriptor_map(entry: Mapping[str, Any]) -> torch.Tensor:
    raw = entry.get("descriptor_map", entry.get("query_descriptor_map"))
    if raw is None:
        raise ValueError("feature map entry is missing descriptor_map")
    descriptor = torch.as_tensor(raw, dtype=torch.float32)
    if descriptor.ndim == 4 and descriptor.shape[0] == 1:
        descriptor = descriptor[0]
    if descriptor.ndim != 3:
        raise ValueError("descriptor_map must have shape [C,H,W] or [1,C,H,W]")
    return descriptor.cpu()


def _select_feature_map_keypoints(
    *,
    entry: Mapping[str, Any],
    max_keypoints: int,
    score_threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "keypoints_yx" in entry or "query_yx" in entry:
        keypoints = torch.as_tensor(entry.get("keypoints_yx", entry.get("query_yx")), dtype=torch.float32).reshape(-1, 2)
        scores = _explicit_keypoint_scores(entry, keypoints)
        if score_threshold is not None:
            keep = scores >= float(score_threshold)
            keypoints = keypoints[keep]
            scores = scores[keep]
        if keypoints.numel() == 0:
            return keypoints.reshape(0, 2), scores.reshape(0)
        order = torch.argsort(scores, descending=True)
        order = order[: int(max_keypoints)]
        return keypoints.index_select(0, order).cpu(), scores.index_select(0, order).cpu()

    score_map = _score_map(entry)
    if score_map is None:
        raise ValueError("feature map entry must contain keypoints_yx/query_yx or score_map")
    flat = score_map.reshape(-1)
    if score_threshold is not None:
        candidate_indices = torch.nonzero(flat >= float(score_threshold), as_tuple=False).reshape(-1)
        if candidate_indices.numel() == 0:
            return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
        candidate_scores = flat.index_select(0, candidate_indices)
        k = min(int(max_keypoints), int(candidate_scores.numel()))
        top_scores, local_indices = torch.topk(candidate_scores, k=k, largest=True, sorted=True)
        top_indices = candidate_indices.index_select(0, local_indices)
    else:
        k = min(int(max_keypoints), int(flat.numel()))
        top_scores, top_indices = torch.topk(flat, k=k, largest=True, sorted=True)
    width = int(score_map.shape[-1])
    y = torch.div(top_indices, width, rounding_mode="floor").float()
    x = (top_indices % width).float()
    return torch.stack([y, x], dim=1).cpu(), top_scores.float().cpu()


def _explicit_keypoint_scores(entry: Mapping[str, Any], keypoints_yx: torch.Tensor) -> torch.Tensor:
    raw_scores = entry.get("keypoint_scores", entry.get("query_score"))
    if raw_scores is not None:
        scores = torch.as_tensor(raw_scores, dtype=torch.float32).reshape(-1)
        if scores.shape[0] != keypoints_yx.shape[0]:
            raise ValueError("keypoint_scores must have one value per keypoint")
        return scores
    score_map = _score_map(entry)
    if score_map is not None:
        return _sample_score_map_nearest(score_map, keypoints_yx)
    return torch.ones((int(keypoints_yx.shape[0]),), dtype=torch.float32)


def _score_map(entry: Mapping[str, Any]) -> torch.Tensor | None:
    raw_score = entry.get("score_map", entry.get("detector_score_map"))
    if raw_score is None:
        return None
    score = torch.as_tensor(raw_score, dtype=torch.float32)
    if score.ndim == 3 and score.shape[0] == 1:
        score = score[0]
    if score.ndim != 2:
        raise ValueError("score_map must have shape [H,W] or [1,H,W]")
    return score.cpu()


def _sample_score_map_nearest(score_map: torch.Tensor, keypoints_yx: torch.Tensor) -> torch.Tensor:
    if keypoints_yx.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    height, width = int(score_map.shape[-2]), int(score_map.shape[-1])
    y = keypoints_yx[:, 0].round().long().clamp(0, height - 1)
    x = keypoints_yx[:, 1].round().long().clamp(0, width - 1)
    return score_map[y, x].float()


def _sample_descriptors_bilinear(descriptor_map: torch.Tensor, keypoints_yx: torch.Tensor) -> torch.Tensor:
    if keypoints_yx.numel() == 0:
        return descriptor_map.new_zeros((0, int(descriptor_map.shape[0])))
    channels, height, width = int(descriptor_map.shape[0]), int(descriptor_map.shape[1]), int(descriptor_map.shape[2])
    points = torch.as_tensor(keypoints_yx, dtype=torch.float32)
    y = points[:, 0].clamp(0, height - 1)
    x = points[:, 1].clamp(0, width - 1)
    grid_x = (x / max(1, width - 1)) * 2.0 - 1.0 if width > 1 else torch.zeros_like(x)
    grid_y = (y / max(1, height - 1)) * 2.0 - 1.0 if height > 1 else torch.zeros_like(y)
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        descriptor_map.view(1, channels, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous().float()
