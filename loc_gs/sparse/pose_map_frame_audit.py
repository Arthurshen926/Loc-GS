from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from loc_gs.core.camera import CameraRecord, load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.landmarks import CacheLandmarkResolver, load_gaussian_landmark_map


@dataclass(frozen=True)
class PoseMapFrameAuditConfig:
    target_width: int | None = None
    target_height: int | None = None
    max_rows: int = 1024
    agreement_threshold_px: float = 0.05
    min_positive_count: int = 1


def audit_pose_map_frame(
    *,
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    target_width: int | None = None,
    target_height: int | None = None,
    max_rows: int = 1024,
    agreement_threshold_px: float = 0.05,
    min_positive_count: int = 1,
) -> dict[str, Any]:
    cfg = PoseMapFrameAuditConfig(
        target_width=target_width,
        target_height=target_height,
        max_rows=int(max_rows),
        agreement_threshold_px=float(agreement_threshold_px),
        min_positive_count=int(min_positive_count),
    )
    return _audit_pose_map_frame(
        Path(candidate_artifact),
        Path(point_cloud),
        Path(cameras_json),
        cfg,
    )


def _audit_pose_map_frame(
    candidate_artifact: Path,
    point_cloud: Path,
    cameras_json: Path,
    cfg: PoseMapFrameAuditConfig,
) -> dict[str, Any]:
    payload = _load_torch_payload(candidate_artifact)
    meta = _metadata(payload)
    split_name = _safe_split_name(payload, meta)
    landmark_map = load_gaussian_landmark_map(point_cloud)
    resolver = CacheLandmarkResolver.from_pair_cache(candidate_artifact, landmark_map)
    auto_resize_candidates = _auto_resize_candidates(cameras_json) if cfg.target_width is None else []
    frame_records = _camera_frame_hypotheses(
        cameras_json,
        target_width=cfg.target_width,
        target_height=cfg.target_height,
        auto_resize_candidates=auto_resize_candidates,
    )

    rows = _positive_rows(payload, resolver, max_rows=cfg.max_rows)
    frame_errors: OrderedDict[str, list[float]] = OrderedDict((name, []) for name in frame_records)
    frame_stored: OrderedDict[str, list[float]] = OrderedDict((name, []) for name in frame_records)
    missing_camera_count = 0
    for row in rows:
        any_camera = False
        for frame_name, records in frame_records.items():
            camera = records.get(row.image_id)
            if camera is None or camera.pose_w2c is None:
                continue
            any_camera = True
            projected_xy, _valid = project_points_w2c(row.point_xyz.reshape(1, 3), camera.pose_w2c, camera.intrinsics)
            if projected_xy.shape[0] != 1 or not np.all(np.isfinite(projected_xy[0])):
                continue
            error_px = float(np.linalg.norm(projected_xy[0] - row.query_xy))
            frame_errors[frame_name].append(error_px)
            frame_stored[frame_name].append(float(row.stored_reprojection_error_px))
        if not any_camera:
            missing_camera_count += 1

    frame_hypotheses = OrderedDict(
        (
            name,
            _frame_stats(
                errors,
                frame_stored[name],
                records=frame_records[name],
            ),
        )
        for name, errors in frame_errors.items()
    )
    best_frame = _best_frame(frame_hypotheses)
    best_stats = frame_hypotheses.get(best_frame, {}) if best_frame else {}
    best_delta = best_stats.get("median_abs_delta_to_stored_error_px")
    passed = (
        best_frame is not None
        and best_delta is not None
        and float(best_delta) <= float(cfg.agreement_threshold_px)
        and len(rows) >= int(cfg.min_positive_count)
    )
    return {
        "schema_version": "internal_pose_map_frame_audit_metrics_v1",
        "status": "passed" if passed else "failed",
        "split_name": split_name,
        "candidate_artifact": str(candidate_artifact),
        "point_cloud": str(point_cloud),
        "cameras_json": str(cameras_json),
        "target_width": cfg.target_width,
        "target_height": cfg.target_height,
        "max_rows": int(cfg.max_rows),
        "agreement_threshold_px": float(cfg.agreement_threshold_px),
        "scanned_row_count": int(min(_row_count(payload), max(0, int(cfg.max_rows)))),
        "evaluated_positive_count": int(len(rows)),
        "missing_camera_count": int(missing_camera_count),
        "best_frame": best_frame,
        "best_frame_width": best_stats.get("width"),
        "best_frame_height": best_stats.get("height"),
        "auto_resize_candidates": [[int(width), int(height)] for width, height in auto_resize_candidates],
        "frame_hypotheses": frame_hypotheses,
    }


@dataclass(frozen=True)
class _AuditRow:
    image_id: str
    query_xy: np.ndarray
    point_xyz: np.ndarray
    stored_reprojection_error_px: float


def _load_torch_payload(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is available in the project env.
        raise RuntimeError("torch is required to read cached candidate artifacts") from exc
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate artifact must contain a mapping payload: {path}")
    return payload


def _metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = payload.get("metadata", {})
    return meta if isinstance(meta, Mapping) else {}


def _safe_split_name(payload: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    split = str(
        payload.get("split_name")
        or meta.get("source_split_name")
        or meta.get("split_name")
        or meta.get("feedback_bank_split_name")
        or "unknown"
    )
    for candidate in (
        payload.get("split_name"),
        meta.get("source_split_name"),
        meta.get("split_name"),
        meta.get("feedback_bank_split_name"),
    ):
        if candidate:
            reject_test_split(str(candidate), purpose="pose/map frame calibration audit")
    split_audit = meta.get("split_audit")
    if isinstance(split_audit, Mapping):
        if bool(split_audit.get("official_test_used")) or bool(split_audit.get("test_split_used")):
            raise ValueError("test split is not allowed for pose/map frame calibration audit")
        if str(split_audit.get("audit_status", "")).lower() == "failed":
            raise ValueError("candidate artifact split audit failed for pose/map frame calibration audit")
        checks = split_audit.get("checks", {})
        if isinstance(checks, Mapping):
            feedback_bank = checks.get("feedback_bank_split", {})
            if isinstance(feedback_bank, Mapping) and feedback_bank.get("split_name"):
                reject_test_split(str(feedback_bank["split_name"]), purpose="pose/map frame calibration audit")
    return reject_test_split(split, purpose="pose/map frame calibration audit")


def _camera_frame_hypotheses(
    cameras_json: Path,
    *,
    target_width: int | None,
    target_height: int | None,
    auto_resize_candidates: Sequence[tuple[int, int]] = (),
) -> OrderedDict[str, dict[str, CameraRecord]]:
    frames: OrderedDict[str, dict[str, CameraRecord]] = OrderedDict()
    frames["native_half_extent"] = load_camera_records(cameras_json, missing_principal_point="half_extent")
    frames["native_pixel_center"] = load_camera_records(cameras_json, missing_principal_point="pixel_center")
    if target_width is not None or target_height is not None:
        frames["resized_half_extent"] = load_camera_records(
            cameras_json,
            target_width=target_width,
            target_height=target_height,
            missing_principal_point="half_extent",
        )
        frames["resized_pixel_center"] = load_camera_records(
            cameras_json,
            target_width=target_width,
            target_height=target_height,
            missing_principal_point="pixel_center",
        )
    else:
        for width, height in auto_resize_candidates:
            frames[f"auto_{int(width)}x{int(height)}_half_extent"] = load_camera_records(
                cameras_json,
                target_width=int(width),
                target_height=int(height),
                missing_principal_point="half_extent",
            )
            frames[f"auto_{int(width)}x{int(height)}_pixel_center"] = load_camera_records(
                cameras_json,
                target_width=int(width),
                target_height=int(height),
                missing_principal_point="pixel_center",
            )
    return frames


def _auto_resize_candidates(cameras_json: Path) -> list[tuple[int, int]]:
    native_records = load_camera_records(cameras_json, missing_principal_point="half_extent")
    source_sizes: OrderedDict[tuple[int, int], None] = OrderedDict()
    for record in native_records.values():
        source_sizes[(int(record.intrinsics.width), int(record.intrinsics.height))] = None

    candidates: OrderedDict[tuple[int, int], None] = OrderedDict()
    for source_width, source_height in source_sizes:
        for divisor in (2, 3, 4, 6, 8):
            if source_width % divisor != 0 or source_height % divisor != 0:
                continue
            width = source_width // divisor
            height = source_height // divisor
            if width < 16 or height < 16 or (width, height) == (source_width, source_height):
                continue
            candidates[(int(width), int(height))] = None
    return list(candidates.keys())


def _positive_rows(payload: Mapping[str, Any], resolver: CacheLandmarkResolver, *, max_rows: int) -> list[_AuditRow]:
    query_yx = _to_numpy_float(payload["query_yx"])
    landmark_id = _to_numpy_int(payload["landmark_id"])
    labels = _to_numpy_int(payload["label"]).reshape(-1)
    masks = _to_numpy_bool(payload.get("candidate_mask")) if "candidate_mask" in payload else None
    stored_errors = _stored_reprojection_errors(payload)
    image_ids = _image_ids(payload, query_yx.shape[0])
    row_limit = min(int(query_yx.shape[0]), max(0, int(max_rows)))
    rows: list[_AuditRow] = []
    for idx in range(row_limit):
        row_ids = landmark_id[idx]
        label = int(labels[idx])
        if label < 0 or label >= int(row_ids.shape[0]):
            continue
        if masks is not None and not bool(masks[idx, label]):
            continue
        cache_landmark_id = int(row_ids[label])
        gaussian_id = int(resolver.resolve_gaussian_ids([[cache_landmark_id]])[0, 0])
        point_xyz = resolver.landmark_map.lookup_xyz([gaussian_id])[0]
        if not np.all(np.isfinite(point_xyz)):
            continue
        stored_error = float(stored_errors[idx, label]) if stored_errors is not None else float("nan")
        rows.append(
            _AuditRow(
                image_id=image_ids[idx],
                query_xy=np.array([float(query_yx[idx, 1]), float(query_yx[idx, 0])], dtype=np.float64),
                point_xyz=np.asarray(point_xyz, dtype=np.float64),
                stored_reprojection_error_px=stored_error,
            )
        )
    return rows


def _stored_reprojection_errors(payload: Mapping[str, Any]) -> np.ndarray | None:
    if "reprojection_error" not in payload:
        return None
    errors = _to_numpy_float(payload["reprojection_error"])
    if errors.ndim == 1:
        return errors.reshape(-1, 1)
    return errors


def _row_count(payload: Mapping[str, Any]) -> int:
    return int(_to_numpy_float(payload["query_yx"]).shape[0])


def _to_numpy_float(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _to_numpy_int(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.int64, copy=False)
    return np.asarray(value, dtype=np.int64)


def _to_numpy_bool(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(bool, copy=False)
    return np.asarray(value, dtype=bool)


def _rows(value: Any, *, max_rows: int | None = None) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if max_rows is not None:
        value = value[: int(max_rows)]
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _infer_image_id(query_id: str) -> str:
    if "::" in query_id:
        return query_id.split("::", 1)[0]
    return query_id


def _image_ids(payload: Mapping[str, Any], row_count: int) -> list[str]:
    if "image_id" in payload:
        values = [str(value) for value in _rows(payload["image_id"], max_rows=row_count)]
    else:
        values = [_infer_image_id(str(value)) for value in _rows(payload["query_id"], max_rows=row_count)]
    if len(values) != row_count:
        raise ValueError("image_id/query_id length must match query_yx rows")
    return values


def _frame_stats(
    errors: Sequence[float],
    stored_errors: Sequence[float],
    *,
    records: Mapping[str, CameraRecord],
) -> dict[str, Any]:
    finite_errors = np.asarray([value for value in errors if np.isfinite(value)], dtype=np.float64)
    stored = np.asarray(stored_errors, dtype=np.float64)
    if finite_errors.size and stored.size == finite_errors.size:
        finite_delta_mask = np.isfinite(stored)
        deltas = np.abs(finite_errors[finite_delta_mask] - stored[finite_delta_mask])
    else:
        deltas = np.empty((0,), dtype=np.float64)
    sample_intrinsics = next(iter(records.values())).intrinsics if records else None
    return {
        "camera_count": int(len(records)),
        "width": None if sample_intrinsics is None else int(sample_intrinsics.width),
        "height": None if sample_intrinsics is None else int(sample_intrinsics.height),
        "evaluated_count": int(finite_errors.size),
        "mean_reprojection_error_px": _mean_or_none(finite_errors),
        "median_reprojection_error_px": _percentile_or_none(finite_errors, 50.0),
        "p90_reprojection_error_px": _percentile_or_none(finite_errors, 90.0),
        "max_reprojection_error_px": _max_or_none(finite_errors),
        "mean_abs_delta_to_stored_error_px": _mean_or_none(deltas),
        "median_abs_delta_to_stored_error_px": _percentile_or_none(deltas, 50.0),
        "p90_abs_delta_to_stored_error_px": _percentile_or_none(deltas, 90.0),
    }


def _best_frame(frame_hypotheses: Mapping[str, Mapping[str, Any]]) -> str | None:
    candidates: list[tuple[float, float, str]] = []
    for name, stats in frame_hypotheses.items():
        if int(stats.get("evaluated_count", 0)) <= 0:
            continue
        delta = stats.get("median_abs_delta_to_stored_error_px")
        reproj = stats.get("median_reprojection_error_px")
        if delta is None or reproj is None:
            continue
        candidates.append((float(delta), float(reproj), str(name)))
    if not candidates:
        return None
    return sorted(candidates)[0][2]


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _max_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values))
