from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.landmarks import CacheLandmarkResolver, load_gaussian_landmark_map


def build_teacher_observations_from_geometry(
    *,
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    scene: str,
    split_name: str,
    target_width: int | None = None,
    target_height: int | None = None,
    missing_principal_point: str = "pixel_center",
    dense_consistency_reprojection_px: float = 4.0,
    sparse_inlier_reprojection_px: float = 8.0,
    hard_negative_reprojection_px: float = 8.0,
    max_solver_weight: float = 4.0,
    max_rows: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    split = reject_test_split(split_name, purpose="geometry teacher observations")
    payload = _load_torch_mapping(candidate_artifact)
    meta = payload.get("metadata", {})
    if not isinstance(meta, Mapping):
        meta = {}
    source_scene = str(meta.get("scene") or payload.get("scene") or "unknown")
    if source_scene not in {"unknown", str(scene)}:
        raise ValueError(f"candidate artifact scene mismatch: expected {scene}, got {source_scene}")
    for candidate_split in (
        payload.get("split_name"),
        meta.get("source_split_name"),
        meta.get("split_name"),
        meta.get("feedback_bank_split_name"),
    ):
        if candidate_split:
            reject_test_split(str(candidate_split), purpose="geometry teacher observations")

    if (target_width is None) != (target_height is None):
        raise ValueError("target_width and target_height must be provided together")
    cameras = load_camera_records(
        cameras_json,
        target_width=target_width,
        target_height=target_height,
        missing_principal_point=str(missing_principal_point),
    )
    landmark_map = load_gaussian_landmark_map(point_cloud)
    resolver = CacheLandmarkResolver.from_pair_cache(candidate_artifact, landmark_map)

    query_yx = _as_float_array(_required(payload, "query_yx"))
    landmark_id = _as_int_array(_required(payload, "landmark_id"))
    candidate_mask = (
        _as_bool_array(payload["candidate_mask"])
        if "candidate_mask" in payload
        else np.ones(landmark_id.shape, dtype=bool)
    )
    image_ids = [str(value) for value in _as_list(payload.get("image_id", [_image_id(qid) for qid in _as_list(_required(payload, "query_id"))]))]
    keypoint_ids = [str(value) for value in _as_list(payload.get("keypoint_id", [f"kp_{idx:06d}" for idx in range(query_yx.shape[0])]))]
    row_count = int(query_yx.shape[0])
    if landmark_id.shape[0] != row_count or candidate_mask.shape != landmark_id.shape:
        raise ValueError("landmark_id/candidate_mask must match query_yx rows")
    if len(image_ids) != row_count or len(keypoint_ids) != row_count:
        raise ValueError("image_id/keypoint_id must match query_yx rows")
    row_limit = row_count if max_rows is None else min(row_count, max(0, int(max_rows)))

    observations: list[dict[str, object]] = []
    missing_camera_count = 0
    invalid_projection_count = 0
    for row_idx in range(row_limit):
        image_id = image_ids[row_idx]
        camera = cameras.get(image_id)
        if camera is None or camera.pose_w2c is None:
            missing_camera_count += 1
            continue
        query_xy = np.array([float(query_yx[row_idx, 1]), float(query_yx[row_idx, 0])], dtype=np.float64)
        cache_ids = landmark_id[row_idx]
        gaussian_ids = resolver.resolve_gaussian_ids([cache_ids])[0]
        points_xyz = landmark_map.lookup_xyz(gaussian_ids)
        projected_xy, valid = project_points_w2c(points_xyz, camera.pose_w2c, camera.intrinsics)
        for rank, cache_landmark_id in enumerate(cache_ids):
            if not bool(candidate_mask[row_idx, rank]):
                continue
            is_valid = bool(valid[rank]) and np.all(np.isfinite(projected_xy[rank]))
            if not is_valid:
                invalid_projection_count += 1
                error_px = float("inf")
            else:
                error_px = float(np.linalg.norm(projected_xy[rank] - query_xy))
            dense_consistent = bool(is_valid and error_px <= float(dense_consistency_reprojection_px))
            sparse_inlier = bool(is_valid and error_px <= float(sparse_inlier_reprojection_px))
            role = _label_role(
                geometric_correct=dense_consistent,
                dense_consistent=dense_consistent,
                sparse_inlier=sparse_inlier,
                reprojection_error_px=error_px,
                hard_negative_reprojection_px=float(hard_negative_reprojection_px),
            )
            observations.append(
                {
                    "schema_version": "internal_geometry_teacher_observation_v1",
                    "scene": str(scene),
                    "split_name": split,
                    "image_id": image_id,
                    "keypoint_id": keypoint_ids[row_idx],
                    "query_id": f"{image_id}::{keypoint_ids[row_idx]}",
                    "candidate_rank": int(rank),
                    "landmark_id": int(cache_landmark_id),
                    "gaussian_id": int(gaussian_ids[rank]),
                    "geometric_correct": bool(dense_consistent),
                    "dense_consistent": bool(dense_consistent),
                    "sparse_inlier": bool(sparse_inlier),
                    "reprojection_error_px": float(error_px),
                    "solver_weight": float(_solver_weight(error_px, sparse_inlier_reprojection_px, max_solver_weight)),
                    "label_role": role,
                    "source": "internal_geometry_projection_teacher",
                }
            )

    dense_count = sum(1 for row in observations if bool(row["dense_consistent"]))
    sparse_inlier_count = sum(1 for row in observations if bool(row["sparse_inlier"]))
    hard_negative_count = sum(1 for row in observations if row["label_role"] == "hard_negative")
    summary = {
        "schema_version": "internal_geometry_teacher_observation_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "candidate_artifact": str(candidate_artifact),
        "point_cloud": str(point_cloud),
        "cameras_json": str(cameras_json),
        "candidate_row_count": int(row_count),
        "scanned_row_count": int(row_limit),
        "observation_count": int(len(observations)),
        "dense_consistent_count": int(dense_count),
        "sparse_inlier_count": int(sparse_inlier_count),
        "hard_negative_count": int(hard_negative_count),
        "missing_camera_count": int(missing_camera_count),
        "invalid_projection_count": int(invalid_projection_count),
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }
    return observations, summary


def _load_torch_mapping(path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate artifact must contain a mapping: {path}")
    return payload


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"candidate artifact is missing required field: {key}")
    return payload[key]


def _as_float_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _as_int_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.int64, copy=False)
    return np.asarray(value, dtype=np.int64)


def _as_bool_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(bool, copy=False)
    return np.asarray(value, dtype=bool)


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _image_id(query_id: Any) -> str:
    value = str(query_id)
    return value.split("::", 1)[0] if "::" in value else value


def _label_role(
    *,
    geometric_correct: bool,
    dense_consistent: bool,
    sparse_inlier: bool,
    reprojection_error_px: float,
    hard_negative_reprojection_px: float,
) -> str:
    if geometric_correct and dense_consistent and sparse_inlier:
        return "protected_support"
    if geometric_correct and sparse_inlier:
        return "positive_inlier"
    if (not geometric_correct) and float(reprojection_error_px) >= float(hard_negative_reprojection_px):
        return "hard_negative"
    return "neutral"


def _solver_weight(error_px: float, sparse_inlier_reprojection_px: float, max_solver_weight: float) -> float:
    if not np.isfinite(error_px):
        return 1.0
    threshold = max(1.0e-9, float(sparse_inlier_reprojection_px))
    value = 1.0 + max(0.0, (threshold - float(error_px)) / threshold)
    return min(float(max_solver_weight), max(1.0e-6, value))
