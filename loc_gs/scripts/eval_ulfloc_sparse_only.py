#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sparse_trace_payload_for_json(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe sparse trace without descriptor matrices."""

    out = {str(key): value for key, value in payload.items() if key != "query_match_descriptors"}
    correspondences = payload.get("correspondences", [])
    compact_correspondences: list[dict[str, Any]] = []
    correspondence_descriptor_count = 0
    query_descriptor_dim = 0
    landmark_descriptor_dim = 0
    if isinstance(correspondences, Sequence) and not isinstance(correspondences, (str, bytes)):
        for row in correspondences:
            if not isinstance(row, Mapping):
                compact_correspondences.append(row)
                continue
            compact_row = {
                str(key): value
                for key, value in row.items()
                if str(key) not in {"query_descriptor", "landmark_descriptor"}
            }
            if "query_descriptor" in row:
                correspondence_descriptor_count += 1
                if query_descriptor_dim <= 0:
                    query_descriptor_dim = int(np.asarray(row.get("query_descriptor", [])).reshape(-1).shape[0])
            if "landmark_descriptor" in row and landmark_descriptor_dim <= 0:
                landmark_descriptor_dim = int(np.asarray(row.get("landmark_descriptor", [])).reshape(-1).shape[0])
            compact_correspondences.append(compact_row)
        out["correspondences"] = compact_correspondences
    descriptors = payload.get("query_match_descriptors", {})
    descriptor_count = 0
    descriptor_dim = 0
    query_count = 0
    if isinstance(descriptors, Mapping):
        query_count = len(descriptors)
        for value in descriptors.values():
            tensor = torch.as_tensor(value)
            if tensor.numel() == 0:
                continue
            if tensor.dim() == 1:
                descriptor_count += 1
                descriptor_dim = int(tensor.numel())
            elif tensor.dim() >= 2:
                descriptor_count += int(tensor.shape[0])
                descriptor_dim = int(tensor.shape[-1])
    out["query_match_descriptor_count"] = int(descriptor_count)
    out["query_match_descriptor_dim"] = int(descriptor_dim)
    out["query_match_descriptor_query_count"] = int(query_count)
    out["query_match_descriptors_omitted_from_json"] = True
    out["correspondence_descriptor_count"] = int(correspondence_descriptor_count)
    out["query_descriptor_dim"] = int(query_descriptor_dim)
    out["landmark_descriptor_dim"] = int(landmark_descriptor_dim)
    out["correspondence_descriptors_omitted_from_json"] = True
    return out


def compact_sparse_attributions_for_result_json(attributions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-match attribution rows without embedding descriptor vectors in results.json."""
    rows = list(attributions)
    query_descriptor_dim = 0
    landmark_descriptor_dim = 0
    for row in rows:
        if query_descriptor_dim <= 0 and "query_descriptor" in row:
            query_descriptor_dim = int(np.asarray(row.get("query_descriptor", [])).reshape(-1).shape[0])
        if landmark_descriptor_dim <= 0 and "landmark_descriptor" in row:
            landmark_descriptor_dim = int(np.asarray(row.get("landmark_descriptor", [])).reshape(-1).shape[0])
        if query_descriptor_dim > 0 and landmark_descriptor_dim > 0:
            break
    return {
        "count": int(len(rows)),
        "pnp_inlier_count": int(sum(1 for row in rows if bool(row.get("pnp_inlier", False)))),
        "query_descriptor_dim": int(query_descriptor_dim),
        "landmark_descriptor_dim": int(landmark_descriptor_dim),
        "descriptors_omitted_from_results_json": True,
    }


def build_sparse_feedback_trace_payload(
    *,
    scene: str | None,
    split_name: str,
    query_rows: list[dict[str, Any]],
    correspondence_rows: list[dict[str, Any]],
    query_features: Mapping[str, Any],
    query_match_descriptors: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    descriptor_mode: str = "full",
) -> dict[str, Any]:
    return {
        "schema_version": "ulfloc_sparse_feedback_trace_v2",
        "scene": scene,
        "split_name": str(split_name),
        "source_role": "baseline_trace",
        "descriptor_mode": str(descriptor_mode),
        "queries": list(query_rows),
        "correspondences": list(correspondence_rows),
        "query_features": dict(query_features),
        "query_match_descriptors": dict(query_match_descriptors),
        "query_match_descriptor_source": "matched_sparse_query_descriptors",
        "record_count": int(len(correspondence_rows)),
        "query_count": int(len(query_rows)),
        "query_feature_count": int(len(query_features)),
        "query_match_descriptor_query_count": int(len(query_match_descriptors)),
        "split_audit": dict(split_audit),
    }


def load_landmark_prior_weights(path: Path) -> torch.Tensor:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        try:
            payload = torch.load(source, map_location="cpu")
        except Exception:
            with source.open("rb") as handle:
                payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"landmark prior artifact must be a JSON object: {path}")
    split_name = str(payload.get("split_name", payload.get("split", ""))).strip().lower()
    split_audit = payload.get("split_audit", {})
    audit_test = (
        bool(split_audit.get("test_split_used", False) or split_audit.get("official_test_used", False))
        if isinstance(split_audit, Mapping)
        else False
    )
    if split_name == "test" or bool(payload.get("official_test_used", False)) or bool(payload.get("test_split_used", False)) or audit_test:
        raise ValueError("refusing to use landmark prior weights from test split")
    if "descriptor_fusion" in payload and isinstance(payload["descriptor_fusion"], Mapping):
        raw_weights = payload["descriptor_fusion"].get("landmark_weights", {})
    else:
        raw_weights = payload.get("landmark_weights", {})
    if isinstance(raw_weights, torch.Tensor):
        return torch.as_tensor(raw_weights, dtype=torch.float32).reshape(-1).clamp_min(0.0).cpu()
    if not isinstance(raw_weights, Mapping):
        raise TypeError("landmark prior artifact must contain a landmark_weights mapping")
    parsed: dict[int, float] = {}
    for raw_gid, raw_weight in raw_weights.items():
        try:
            gid = int(raw_gid)
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if gid < 0:
            continue
        parsed[gid] = max(0.0, weight)
    if not parsed:
        return torch.empty((0,), dtype=torch.float32)
    weights = torch.zeros((max(parsed) + 1,), dtype=torch.float32)
    for gid, weight in parsed.items():
        weights[gid] = float(weight)
    return weights


def load_landmark_activation_audit(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"enabled": False}
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"landmark activation checkpoint must contain a dict: {source}")
    split_name = str(payload.get("split_name", payload.get("split", ""))).strip()
    split_audit = payload.get("split_audit", {})
    audit = split_audit if isinstance(split_audit, Mapping) else {}
    if split_name.lower() == "test" or bool(audit.get("test_split_used")) or bool(audit.get("official_test_used")):
        raise ValueError("refusing to use landmark activation checkpoint trained from test split")
    landmark_ids = torch.as_tensor(payload.get("landmark_ids", []), dtype=torch.long).reshape(-1)
    safe_core = torch.as_tensor(payload.get("safe_core_indices", []), dtype=torch.long).reshape(-1)
    activation_model_type = str(payload.get("activation_model_type", "v1_mean_descriptor"))
    query_feature_mode = (
        "token_cross_attention"
        if activation_model_type == "v2_tokens"
        else str(payload.get("query_feature_mode", "mean_descriptor"))
    )
    return {
        "enabled": True,
        "checkpoint": str(source),
        "checkpoint_split_name": split_name or "unknown",
        "activation_model_type": activation_model_type,
        "query_feature_mode": query_feature_mode,
        "diagnostic_only": activation_model_type != "v2_tokens",
        "query_dim": int(payload.get("query_dim", 0)),
        "query_token_dim": int(payload.get("query_token_dim", 0)),
        "landmark_dim": int(payload.get("landmark_dim", 0)),
        "landmark_token_dim": int(payload.get("landmark_token_dim", 0)),
        "hidden_dim": int(payload.get("hidden_dim", 0)),
        "attention_top_k": int(payload.get("attention_top_k", 0)),
        "checkpoint_top_n": int(payload.get("top_n", 0)),
        "safe_core_count": int(safe_core.numel()),
        "checkpoint_landmark_count": int(landmark_ids.numel()),
        "source_cache_path": payload.get("source_cache_path"),
        "split_audit": dict(audit),
    }


def build_scene_detector_eval_config(
    *,
    checkpoint: str | Path,
    mode: str = "stdloc_fullres",
    sparse_k: int,
    candidate_top_k: int = 0,
    blend_alpha: float = 0.5,
    native_keep_fraction: float = 0.0,
    nms_radius: int = 4,
    score_threshold: float = 0.0,
    fusion_rule: str = "geometric",
    descriptor_stride: int = 8,
) -> dict[str, Any]:
    raw_mode = str(mode).strip().lower()
    if raw_mode in {"direct", "direct_heatmap", "grid"}:
        resolved_mode = "grid"
        detector_mode = "direct_heatmap"
        diagnostic_only = True
    elif raw_mode in {"rerank", "rerank_superpoint", "superpoint_rerank"}:
        resolved_mode = "rerank_superpoint"
        detector_mode = "rerank_superpoint"
        diagnostic_only = True
    elif raw_mode in {"score_fusion", "superpoint_score_fusion", "sp_score_fusion"}:
        resolved_mode = "score_fusion"
        detector_mode = "score_fusion"
        diagnostic_only = True
    elif raw_mode in {"stdloc_fullres", "fullres", "full_resolution"}:
        resolved_mode = "stdloc_fullres"
        detector_mode = "stdloc_fullres"
        diagnostic_only = False
    else:
        raise ValueError(f"unsupported scene detector eval mode: {mode}")
    sparse_count = int(sparse_k)
    effective_native_keep_fraction = (
        float(native_keep_fraction) if resolved_mode == "rerank_superpoint" else 0.0
    )
    return {
        "enabled": True,
        "checkpoint": str(Path(checkpoint).resolve()),
        "mode": resolved_mode,
        "scene_detector_mode": detector_mode,
        "diagnostic_only": bool(diagnostic_only),
        "candidate_top_k": int(candidate_top_k) if int(candidate_top_k) > 0 else sparse_count * 2,
        "blend_alpha": float(blend_alpha),
        "native_keep_fraction": float(effective_native_keep_fraction),
        "fusion_rule": str(fusion_rule),
        "descriptor_stride": int(descriptor_stride),
        "nms_radius": int(nms_radius),
        "score_threshold": float(score_threshold),
    }


def summarize_sparse_metrics(
    *,
    translation_cm: Sequence[float],
    rotation_deg: Sequence[float],
    inliers: Sequence[int],
) -> dict[str, Any]:
    te = np.asarray(list(translation_cm), dtype=np.float64)
    re = np.asarray(list(rotation_deg), dtype=np.float64)
    inlier_array = np.asarray(list(inliers), dtype=np.float64)
    if te.size == 0:
        return {
            "query_count": 0,
            "median_te_cm": None,
            "median_re_deg": None,
            "p90_te_cm": None,
            "p95_te_cm": None,
            "mean_inliers": None,
            "recall_50cm_5d": None,
            "recall_15cm_5d": None,
            "recall_10cm_5d": None,
            "recall_5cm_5d": None,
            "recall_2cm_2d": None,
        }
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "median_re_deg": float(np.median(re)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "p95_te_cm": float(np.percentile(te, 95)),
        "mean_inliers": float(inlier_array.mean()) if inlier_array.size else 0.0,
        "recall_50cm_5d": float(((te <= 50.0) & (re <= 5.0)).sum() / te.size),
        "recall_15cm_5d": float(((te <= 15.0) & (re <= 5.0)).sum() / te.size),
        "recall_10cm_5d": float(((te <= 10.0) & (re <= 5.0)).sum() / te.size),
        "recall_5cm_5d": float(((te <= 5.0) & (re <= 5.0)).sum() / te.size),
        "recall_2cm_2d": float(((te <= 2.0) & (re <= 2.0)).sum() / te.size),
    }


def build_sparse_only_split_audit(split_name: str) -> dict[str, Any]:
    split = str(split_name).strip()
    is_test = split.lower() == "test"
    is_known = bool(split) and split.lower() not in {"unknown", "none", "null"}
    return {
        "schema_version": "ulfloc_sparse_only_split_audit_v1",
        "split_name": split,
        "official_test_eval": bool(is_test),
        "official_test_used": bool(is_test),
        "test_split_used": bool(is_test),
        "paper_safe_for_tuning": bool(is_known and not is_test),
        "role": "evaluation_only",
    }


def _finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.mean())


def _finite_median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def summarize_sparse_match_diagnostics(matches: Mapping[str, Any]) -> dict[str, Any]:
    """Compact per-query sparse correspondence diagnostics.

    This is evaluation-only instrumentation. It does not modify match ranking,
    PnP, or metric definitions.
    """
    descriptor = np.asarray(matches.get("descriptor_score", []), dtype=np.float64).reshape(-1)
    detector = np.asarray(matches.get("detector_score", []), dtype=np.float64).reshape(-1)
    reproj = np.asarray(matches.get("reprojection_error_px", []), dtype=np.float64).reshape(-1)
    inlier_mask = np.asarray(matches.get("pnp_inlier_mask", []), dtype=bool).reshape(-1)
    landmark_ids = np.asarray(matches.get("matched_gaussian_ids", []), dtype=np.int64).reshape(-1)
    query_xy = np.asarray(matches.get("query_xy", []), dtype=np.float64)

    total = int(max(descriptor.size, detector.size, reproj.size, inlier_mask.size, landmark_ids.size))
    if inlier_mask.size != total:
        padded = np.zeros((total,), dtype=bool)
        padded[: min(total, inlier_mask.size)] = inlier_mask[: min(total, inlier_mask.size)]
        inlier_mask = padded

    def selected(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != total:
            padded = np.full((total,), np.nan, dtype=np.float64)
            padded[: min(total, values.size)] = values[: min(total, values.size)]
            values = padded
        return values[mask]

    if landmark_ids.size:
        unique_landmarks = int(np.unique(landmark_ids).shape[0])
    else:
        unique_landmarks = 0

    bbox_area_norm: float | None = None
    if query_xy.ndim == 2 and query_xy.shape[0] > 0 and query_xy.shape[1] >= 2:
        image_size = matches.get("image_size", None)
        if image_size is not None and len(image_size) >= 2:
            width = max(float(image_size[0]), 1.0)
            height = max(float(image_size[1]), 1.0)
            xy = query_xy[:, :2]
            span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 0.0)
            bbox_area_norm = float((span[0] * span[1]) / max(width * height, 1.0))

    return {
        "total_match_count": int(total),
        "unique_landmark_count": unique_landmarks,
        "pnp_inlier_count": int(inlier_mask.sum()),
        "pnp_inlier_ratio": float(inlier_mask.sum() / total) if total > 0 else 0.0,
        "descriptor_score_mean": _finite_mean(descriptor),
        "descriptor_score_median": _finite_median(descriptor),
        "descriptor_score_inlier_mean": _finite_mean(selected(descriptor, inlier_mask)),
        "descriptor_score_outlier_mean": _finite_mean(selected(descriptor, ~inlier_mask)),
        "detector_score_mean": _finite_mean(detector),
        "detector_score_inlier_mean": _finite_mean(selected(detector, inlier_mask)),
        "detector_score_outlier_mean": _finite_mean(selected(detector, ~inlier_mask)),
        "reprojection_error_median_px": _finite_median(reproj),
        "reprojection_error_inlier_median_px": _finite_median(selected(reproj, inlier_mask)),
        "reprojection_error_outlier_median_px": _finite_median(selected(reproj, ~inlier_mask)),
        "query_bbox_area_norm": bbox_area_norm,
    }


def _array_1d(matches: Mapping[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    return np.asarray(matches.get(key, []), dtype=dtype).reshape(-1)


def _array_2d(matches: Mapping[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    value = np.asarray(matches.get(key, []), dtype=dtype)
    if value.size == 0:
        return np.empty((0, 0), dtype=dtype)
    if value.ndim == 1:
        return value.reshape(1, -1)
    return value.reshape(value.shape[0], -1)


def _query_xy_array(matches: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(matches.get("query_xy", []), dtype=np.float64)
    if value.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if value.ndim == 1:
        if value.size == 2:
            return value.reshape(1, 2)
        if value.size % 2 == 0:
            return value.reshape(-1, 2)
        return value.reshape(1, -1)
    return value.reshape(value.shape[0], -1)


def _value_at(array: np.ndarray, index: int, default: Any = None) -> Any:
    if index < int(array.shape[0]):
        return array[index]
    return default


def _query_xy_norm_and_cell(xy: np.ndarray, image_size: Any, *, grid_size: int = 8) -> tuple[list[float] | None, int | None]:
    if image_size is None or len(image_size) < 2:
        return None, None
    width = max(float(image_size[0]), 1.0)
    height = max(float(image_size[1]), 1.0)
    x_norm = min(max(float(xy[0]) / width, 0.0), 1.0)
    y_norm = min(max(float(xy[1]) / height, 0.0), 1.0)
    grid = max(1, int(grid_size))
    x_cell = min(grid - 1, int(np.floor(x_norm * grid)))
    y_cell = min(grid - 1, int(np.floor(y_norm * grid)))
    return [float(x_norm), float(y_norm)], int(y_cell * grid + x_cell)


def _descriptor_margin_values(matches: Mapping[str, Any], total: int) -> tuple[np.ndarray, str]:
    if "descriptor_margin" in matches:
        margin = _array_1d(matches, "descriptor_margin", dtype=np.float64)
        source = str(matches.get("descriptor_margin_source", "matches.descriptor_margin"))
    elif "second_descriptor_score" in matches:
        descriptor = _array_1d(matches, "descriptor_score", dtype=np.float64)
        second = _array_1d(matches, "second_descriptor_score", dtype=np.float64)
        margin = descriptor[: min(descriptor.size, second.size)] - second[: min(descriptor.size, second.size)]
        source = "descriptor_score_minus_second_descriptor_score"
    else:
        margin = np.zeros((total,), dtype=np.float64)
        source = "missing_top2_default_zero"
    if margin.size != total:
        padded = np.zeros((total,), dtype=np.float64)
        padded[: min(total, margin.size)] = margin[: min(total, margin.size)]
        margin = padded
    return margin, source


def _inlier_mask(matches: Mapping[str, Any], total: int) -> np.ndarray:
    mask = _array_1d(matches, "pnp_inlier_mask", dtype=bool)
    if mask.size == total:
        return mask
    padded = np.zeros((total,), dtype=bool)
    padded[: min(total, mask.size)] = mask[: min(total, mask.size)]
    return padded


def _spread(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(finite.max() - finite.min())


def _bearing_spread(bearing: np.ndarray) -> float:
    if bearing.ndim != 2 or bearing.shape[0] < 2:
        return 0.0
    finite = bearing[np.isfinite(bearing).all(axis=1)]
    if finite.shape[0] < 2:
        return 0.0
    normalized = finite / np.linalg.norm(finite, axis=1, keepdims=True).clip(min=1e-12)
    center = normalized.mean(axis=0)
    center = center / max(float(np.linalg.norm(center)), 1e-12)
    cos = np.clip(normalized @ center, -1.0, 1.0)
    return float(np.max(np.arccos(cos)))


def build_sparse_feedback_query_row(
    *,
    scene: str | None,
    split_name: str,
    query_id: str,
    image_id: str,
    sparse_te_cm: float,
    sparse_re_deg: float,
    matches: Mapping[str, Any],
    image_grid_size: int = 8,
    depth_bin_count: int = 8,
) -> dict[str, Any]:
    match_count = int(matches.get("total_match_count", 0))
    inlier_count = int(matches.get("sparse_inlier_count", matches.get("inliers", 0)))
    mask = _inlier_mask(matches, match_count)
    query_xy = _query_xy_array(matches)
    image_cells: set[int] = set()
    if query_xy.ndim == 2 and query_xy.shape[0] > 0:
        for xy in query_xy[: min(query_xy.shape[0], mask.size)][mask[: min(query_xy.shape[0], mask.size)]]:
            _norm, cell = _query_xy_norm_and_cell(xy[:2], matches.get("image_size"), grid_size=int(image_grid_size))
            if cell is not None:
                image_cells.add(int(cell))
    depth = _array_1d(matches, "depth_m", dtype=np.float64)
    if depth.size == 0 and "p3d" in matches:
        p3d = np.asarray(matches.get("p3d", []), dtype=np.float64)
        if p3d.ndim == 2 and p3d.shape[1] >= 3:
            depth = p3d[:, 2]
    inlier_depth = depth[: min(depth.size, mask.size)][mask[: min(depth.size, mask.size)]] if depth.size else np.empty(0)
    depth_bins: set[int] = set()
    if inlier_depth.size:
        min_depth = float(np.nanmin(inlier_depth))
        max_depth = float(np.nanmax(inlier_depth))
        span = max(max_depth - min_depth, 1e-6)
        bins = np.floor((inlier_depth - min_depth) / span * max(1, int(depth_bin_count) - 1)).astype(np.int64)
        depth_bins = {int(item) for item in bins.tolist()}
    bearing = np.asarray(matches.get("bearing", []), dtype=np.float64)
    if bearing.ndim == 2 and bearing.shape[0] and mask.size:
        inlier_bearing = bearing[: min(bearing.shape[0], mask.size)][mask[: min(bearing.shape[0], mask.size)]]
    else:
        inlier_bearing = np.empty((0, 3), dtype=np.float64)
    return {
        "scene": scene,
        "split_name": str(split_name),
        "query_id": str(query_id),
        "image_id": str(image_id),
        "pose_success": bool(matches.get("pnp_success", False)),
        "sparse_te_cm": float(sparse_te_cm),
        "sparse_re_deg": float(sparse_re_deg),
        "catastrophic_failure": bool(float(sparse_te_cm) > 500.0),
        "inlier_count": int(inlier_count),
        "match_count": int(match_count),
        "inlier_image_cell_count": int(len(image_cells)),
        "inlier_depth_bin_count": int(len(depth_bins)),
        "bearing_spread": float(_bearing_spread(inlier_bearing)),
        "depth_spread": float(_spread(inlier_depth)),
    }


def serialize_sparse_match_attributions(
    matches: Mapping[str, Any],
    *,
    scene: str | None = None,
    split_name: str | None = None,
    query_id: str | None = None,
    image_id: str | None = None,
    source_role: str = "baseline_trace",
    include_descriptors: bool = True,
    max_correspondences: int = 0,
) -> list[dict[str, Any]]:
    """Export per-correspondence sparse solver traces for offline pruning."""

    landmark_ids = _array_1d(matches, "matched_gaussian_ids", dtype=np.int64)
    sampled_rows = _array_1d(matches, "matched_sampled_indices", dtype=np.int64)
    active_rows = _array_1d(matches, "matched_active_indices", dtype=np.int64)
    query_keypoint_indices = _array_1d(matches, "query_keypoint_indices", dtype=np.int64)
    inlier_mask = _array_1d(matches, "pnp_inlier_mask", dtype=bool)
    descriptor = _array_1d(matches, "descriptor_score", dtype=np.float64)
    detector = _array_1d(matches, "detector_score", dtype=np.float64)
    reproj = _array_1d(matches, "reprojection_error_px", dtype=np.float64)
    query_xy = _query_xy_array(matches)
    camera_xyz = _array_2d(matches, "camera_xyz", dtype=np.float64)
    depth = _array_1d(matches, "depth_m", dtype=np.float64)
    bearing = _array_2d(matches, "bearing", dtype=np.float64)
    query_descriptor = (
        _array_2d(matches, "query_descriptor", dtype=np.float64)
        if bool(include_descriptors)
        else np.empty((0, 0), dtype=np.float64)
    )
    landmark_descriptor = (
        _array_2d(matches, "landmark_descriptor", dtype=np.float64)
        if bool(include_descriptors)
        else np.empty((0, 0), dtype=np.float64)
    )
    query_xy_count = int(query_xy.shape[0]) if query_xy.ndim == 2 else 0
    total_base = int(
        max(
            landmark_ids.size,
            sampled_rows.size,
            active_rows.size,
            query_keypoint_indices.size,
            inlier_mask.size,
            descriptor.size,
            detector.size,
            reproj.size,
            query_xy_count,
            camera_xyz.shape[0],
            depth.size,
            bearing.shape[0],
            query_descriptor.shape[0],
            landmark_descriptor.shape[0],
        )
    )
    descriptor_margin, descriptor_margin_source = _descriptor_margin_values(matches, total_base)
    total = int(max(total_base, descriptor_margin.size))
    if int(max_correspondences) > 0 and total > int(max_correspondences):
        inlier_indices = [int(idx) for idx in np.flatnonzero(inlier_mask[: min(inlier_mask.size, total)])]
        selected = set(inlier_indices)
        remaining = max(0, int(max_correspondences) - len(selected))
        if remaining > 0:
            score = np.full((total,), -np.inf, dtype=np.float64)
            score[: min(total, descriptor.size)] = descriptor[: min(total, descriptor.size)]
            outlier_order = [
                int(idx)
                for idx in np.argsort(-score)
                if int(idx) not in selected and np.isfinite(score[int(idx)])
            ]
            for idx in outlier_order[:remaining]:
                selected.add(int(idx))
        selected_indices = sorted(selected)
    else:
        selected_indices = list(range(total))
    out: list[dict[str, Any]] = []
    for index in selected_indices:
        row: dict[str, Any] = {
            "match_row_index": int(index),
            "source_role": str(source_role),
            "active_index_space": "active_landmark_rows",
            "sampled_row_space": "sampled_landmark_rows",
            "gaussian_id_space": "full_gaussian_ids",
        }
        if scene is not None:
            row["scene"] = str(scene)
        if split_name is not None:
            row["split_name"] = str(split_name)
        if index < landmark_ids.size:
            row["matched_gaussian_id"] = int(landmark_ids[index])
            row["gaussian_id"] = int(landmark_ids[index])
            row["landmark_id"] = int(landmark_ids[index])
        if index < sampled_rows.size:
            row["matched_sampled_index"] = int(sampled_rows[index])
            row["sampled_row"] = int(sampled_rows[index])
        if index < active_rows.size:
            row["matched_active_index"] = int(active_rows[index])
        if index < query_keypoint_indices.size:
            row["query_keypoint_index"] = int(query_keypoint_indices[index])
        if index < inlier_mask.size:
            row["pnp_inlier"] = bool(inlier_mask[index])
            row["label"] = 1 if bool(inlier_mask[index]) else 0
        if index < descriptor.size and np.isfinite(descriptor[index]):
            row["descriptor_score"] = float(descriptor[index])
        if index < descriptor_margin.size and np.isfinite(descriptor_margin[index]):
            row["descriptor_margin"] = float(descriptor_margin[index])
            row["descriptor_margin_source"] = descriptor_margin_source
        if index < detector.size and np.isfinite(detector[index]):
            row["detector_score"] = float(detector[index])
        if index < reproj.size and np.isfinite(reproj[index]):
            row["reprojection_error_px"] = float(reproj[index])
        if query_xy.ndim == 2 and index < query_xy.shape[0] and query_xy.shape[1] >= 2:
            xy = query_xy[index, :2]
            if np.isfinite(xy).all():
                row["query_xy"] = [float(xy[0]), float(xy[1])]
                row["keypoint_xy"] = [float(xy[0]), float(xy[1])]
                norm, cell = _query_xy_norm_and_cell(xy, matches.get("image_size"))
                if norm is not None:
                    row["query_xy_norm"] = norm
                if cell is not None:
                    row["image_cell"] = int(cell)
        xyz = _value_at(camera_xyz, index)
        if xyz is not None and np.asarray(xyz).size >= 3 and np.isfinite(xyz[:3]).all():
            row["camera_xyz"] = [float(item) for item in xyz[:3]]
        if index < depth.size and np.isfinite(depth[index]):
            row["depth_m"] = float(depth[index])
        bearing_row = _value_at(bearing, index)
        if bearing_row is not None and np.asarray(bearing_row).size >= 3 and np.isfinite(bearing_row[:3]).all():
            row["bearing"] = [float(item) for item in bearing_row[:3]]
        if bool(include_descriptors):
            query_desc = _value_at(query_descriptor, index)
            if query_desc is not None and np.asarray(query_desc).size > 0 and np.isfinite(query_desc).all():
                row["query_descriptor"] = [float(item) for item in np.asarray(query_desc).reshape(-1).tolist()]
            landmark_desc = _value_at(landmark_descriptor, index)
            if landmark_desc is not None and np.asarray(landmark_desc).size > 0 and np.isfinite(landmark_desc).all():
                row["landmark_descriptor"] = [float(item) for item in np.asarray(landmark_desc).reshape(-1).tolist()]
        if query_id is not None:
            row["query_id"] = str(query_id)
        if image_id is not None:
            row["image_id"] = str(image_id)
        if row:
            out.append(row)
    return out


def _import_ulfloc(ulf_root: Path) -> dict[str, Any]:
    root = str(Path(ulf_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore
    from ulfloc import ULFLoc  # type: ignore
    from utils.pose_utils import cal_pose_error  # type: ignore

    return {
        "Scene": Scene,
        "GaussianModel": GaussianModel,
        "GaussianModel_2dgs": GaussianModel_2dgs,
        "ULFLoc": ULFLoc,
        "cal_pose_error": cal_pose_error,
    }


def _dataset_namespace(args: argparse.Namespace, config: Mapping[str, Any]) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=str(Path(args.source_path).resolve()),
        feature_type=str(args.feature_type),
        gaussian_type=str(args.gaussian_type),
        model_path=str(Path(args.model_path).resolve()),
        images=str(args.images),
        resolution=-1,
        white_background=True,
        longest_edge=int(args.longest_edge),
        data_device=str(args.data_device),
        eval=True,
        speedup=False,
        norm_before_render=bool(config.get("dense", {}).get("norm_before_render", True))
        if isinstance(config.get("dense", {}), Mapping)
        else True,
        render_items=["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
    )


def _load_masks(source_path: Path, images: str) -> Any:
    path = Path(source_path) / images / "masks.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def resize_mask_to_image(mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(mask)
    original_dtype = tensor.dtype
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError("mask must have shape [H,W] or [1,H,W]")
    if tensor.shape[-2:] == (int(height), int(width)):
        return tensor.to(dtype=original_dtype)
    resized = F.interpolate(
        tensor.float().unsqueeze(0),
        size=(int(height), int(width)),
        mode="nearest",
    ).squeeze(0)
    if original_dtype == torch.bool:
        return resized > 0.5
    return resized.to(dtype=original_dtype)


def _apply_masks(query_image: torch.Tensor, masks: Any, image_name: str) -> torch.Tensor:
    if masks is None:
        return query_image
    with torch.no_grad():
        height, width = int(query_image.shape[-2]), int(query_image.shape[-1])
        obj_mask = resize_mask_to_image(masks[image_name][0], height=height, width=width).to(query_image.device)
        sky_mask = resize_mask_to_image(masks[image_name][1], height=height, width=width).to(query_image.device)
        distort_mask = resize_mask_to_image(masks[image_name][2], height=height, width=width).to(query_image.device)
        mask = obj_mask & distort_mask
        query_image = query_image * mask
        query_image[sky_mask.repeat(3, 1, 1) == False] = 0
    return query_image


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate only the ULF-Loc sparse PnP stage.")
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_log_dir", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene_detector_checkpoint", default=None, type=Path)
    parser.add_argument(
        "--scene_detector_mode",
        default="stdloc_fullres",
        choices=("grid", "direct_heatmap", "rerank_superpoint", "score_fusion", "stdloc_fullres"),
    )
    parser.add_argument("--scene_detector_blend_alpha", default=0.5, type=float)
    parser.add_argument(
        "--scene_detector_fusion_rule",
        default="geometric",
        choices=("geometric", "residual_boost"),
    )
    parser.add_argument("--scene_detector_candidate_top_k", default=0, type=int)
    parser.add_argument("--scene_detector_nms_radius", default=None, type=int)
    parser.add_argument("--scene_detector_score_threshold", default=0.0, type=float)
    parser.add_argument("--scene_detector_native_keep_fraction", default=0.0, type=float)
    parser.add_argument("--scene_matcher_checkpoint", default=None, type=Path)
    parser.add_argument("--scene_matcher_topk", default=4, type=int)
    parser.add_argument("--scene_matcher_score_weight", default=0.0, type=float)
    parser.add_argument(
        "--scene_matcher_score_mode",
        choices=["candidate_minus_dustbin", "candidate"],
        default="candidate_minus_dustbin",
    )
    parser.add_argument("--scene_matcher_max_descriptor_margin", default=float("inf"), type=float)
    parser.add_argument("--scene_matcher_drop_dustbin", action="store_true")
    parser.add_argument("--scene_matcher_min_keep", default=0, type=int)
    parser.add_argument("--landmark_activation_checkpoint", default=None, type=Path)
    parser.add_argument("--landmark_activation_top_n", default=0, type=int)
    parser.add_argument("--landmark_activation_disable_empty_fallback", action="store_true")
    parser.add_argument("--sparse_match_filter_mode", default="none")
    parser.add_argument("--sparse_match_filter_min_query_margin", default=0.0, type=float)
    parser.add_argument("--sparse_match_filter_min_descriptor_score", default=-float("inf"), type=float)
    parser.add_argument("--sparse_match_filter_top_m", default=0, type=int)
    parser.add_argument("--sparse_match_filter_min_keep", default=0, type=int)
    parser.add_argument("--sparse_match_filter_margin_weight", default=1.0, type=float)
    parser.add_argument("--sparse_match_filter_detector_score_weight", default=0.0, type=float)
    parser.add_argument("--sparse_match_filter_landmark_prior", default=None, type=Path)
    parser.add_argument("--sparse_match_filter_landmark_prior_weight", default=0.0, type=float)
    parser.add_argument("--sparse_match_filter_landmark_prior_pre_topk_weight", default=0.0, type=float)
    parser.add_argument("--sparse_match_filter_landmark_prior_center", default=0.0, type=float)
    parser.add_argument("--sparse_match_filter_landmark_prior_scale", default=1.0, type=float)
    parser.add_argument("--sparse_match_filter_landmark_prior_clip", default=0.0, type=float)
    parser.add_argument("--sparse_match_filter_image_grid_size", default=0, type=int)
    parser.add_argument("--sparse_match_filter_max_per_image_cell", default=0, type=int)
    parser.add_argument("--sparse_match_filter_unique_landmark", action="store_true")
    parser.add_argument("--max_queries", default=0, type=int)
    parser.add_argument("--query_id", action="append", default=None)
    parser.add_argument("--dump_sparse_match_diagnostics", action="store_true")
    parser.add_argument("--dump_sparse_match_attributions", action="store_true")
    parser.add_argument(
        "--sparse_trace_descriptor_mode",
        choices=["full", "none"],
        default="full",
        help=(
            "Descriptor payload stored in sparse PnP trace. Use 'none' for full-scene solver-feedback "
            "attribution; descriptor contrastive caches should be built with dedicated observation exporters."
        ),
    )
    parser.add_argument(
        "--sparse_trace_max_correspondences_per_query",
        default=0,
        type=int,
        help="If >0, keep all PnP inliers and only the top-scoring outliers up to this per-query trace cap.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imported = _import_ulfloc(Path(args.ulf_root))
    config = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("ULF config must be a YAML mapping")
    config["longest_edge"] = int(args.longest_edge)
    config["model_path"] = str(Path(args.model_path).resolve())
    if args.scene_detector_checkpoint is not None:
        sparse_k = int(config.get("sparse", {}).get("kpts_num", 2048))
        config["scene_specific_detector"] = build_scene_detector_eval_config(
            checkpoint=Path(args.scene_detector_checkpoint),
            mode=str(args.scene_detector_mode),
            sparse_k=sparse_k,
            candidate_top_k=int(args.scene_detector_candidate_top_k),
            blend_alpha=float(args.scene_detector_blend_alpha),
            fusion_rule=str(args.scene_detector_fusion_rule),
            native_keep_fraction=float(args.scene_detector_native_keep_fraction),
            nms_radius=int(args.scene_detector_nms_radius)
            if args.scene_detector_nms_radius is not None
            else int(config.get("sparse", {}).get("nms", 4)),
            score_threshold=float(args.scene_detector_score_threshold),
        )
    else:
        config["scene_specific_detector"] = {"enabled": False}
    config["scene_matcher"] = {
        "enabled": args.scene_matcher_checkpoint is not None,
        "checkpoint": None if args.scene_matcher_checkpoint is None else str(Path(args.scene_matcher_checkpoint).resolve()),
        "topk": int(args.scene_matcher_topk),
        "score_weight": float(args.scene_matcher_score_weight),
        "score_mode": str(args.scene_matcher_score_mode),
        "max_descriptor_margin": float(args.scene_matcher_max_descriptor_margin),
        "drop_dustbin": bool(args.scene_matcher_drop_dustbin),
        "min_keep": int(args.scene_matcher_min_keep),
    }
    config["landmark_activation"] = {
        "enabled": args.landmark_activation_checkpoint is not None,
        "checkpoint": None
        if args.landmark_activation_checkpoint is None
        else str(Path(args.landmark_activation_checkpoint).resolve()),
        "top_n": int(args.landmark_activation_top_n),
        "fallback_to_all_if_empty": not bool(args.landmark_activation_disable_empty_fallback),
        "query_feature_mode": "mean_descriptor",
    }
    landmark_activation_audit = load_landmark_activation_audit(args.landmark_activation_checkpoint)
    sparse_filter_mode = str(args.sparse_match_filter_mode).strip().lower()
    landmark_prior_weights = (
        load_landmark_prior_weights(Path(args.sparse_match_filter_landmark_prior))
        if args.sparse_match_filter_landmark_prior is not None
        else None
    )
    config["sparse_match_filter"] = {
        "enabled": sparse_filter_mode not in {"", "none", "off", "false", "0"},
        "mode": sparse_filter_mode,
        "min_query_margin": float(args.sparse_match_filter_min_query_margin),
        "min_descriptor_score": float(args.sparse_match_filter_min_descriptor_score),
        "top_m": int(args.sparse_match_filter_top_m),
        "min_keep": int(args.sparse_match_filter_min_keep),
        "margin_weight": float(args.sparse_match_filter_margin_weight),
        "detector_score_weight": float(args.sparse_match_filter_detector_score_weight),
        "landmark_prior_weights": landmark_prior_weights,
        "landmark_prior_weight": float(args.sparse_match_filter_landmark_prior_weight),
        "landmark_prior_pre_topk_weight": float(args.sparse_match_filter_landmark_prior_pre_topk_weight),
        "landmark_prior_center": float(args.sparse_match_filter_landmark_prior_center),
        "landmark_prior_scale": float(args.sparse_match_filter_landmark_prior_scale),
        "landmark_prior_clip": float(args.sparse_match_filter_landmark_prior_clip),
        "landmark_prior_path": None
        if args.sparse_match_filter_landmark_prior is None
        else str(Path(args.sparse_match_filter_landmark_prior)),
        "landmark_prior_loaded_count": int(landmark_prior_weights.numel())
        if landmark_prior_weights is not None
        else 0,
        "image_grid_size": int(args.sparse_match_filter_image_grid_size),
        "max_per_image_cell": int(args.sparse_match_filter_max_per_image_cell),
        "unique_landmark": bool(args.sparse_match_filter_unique_landmark),
    }
    sparse_match_filter_metrics = dict(config["sparse_match_filter"])
    sparse_match_filter_metrics.pop("landmark_prior_weights", None)

    dataset = _dataset_namespace(args, config)
    if dataset.gaussian_type == "3dgs":
        gaussians = imported["GaussianModel"](dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = imported["GaussianModel_2dgs"](dataset.sh_degree)
    else:
        raise ValueError(f"unsupported gaussian_type: {dataset.gaussian_type}")
    scene = imported["Scene"](
        dataset,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        preload_cameras=True,
    )
    masks = _load_masks(Path(args.source_path), str(args.images))
    ulfloc = imported["ULFLoc"](scene, gaussians, masks, str(Path(args.input_log_dir)), config)
    if landmark_activation_audit.get("enabled"):
        sampled_count = int(torch.as_tensor(ulfloc.sampled_idx).reshape(-1).numel())
        landmark_activation_audit["sampled_idx_count"] = sampled_count
        landmark_activation_audit["landmark_ids_match_sampled_idx"] = (
            int(landmark_activation_audit.get("checkpoint_landmark_count", -1)) == sampled_count
        )
    cameras = list(scene.getTestCameras())
    requested = set(str(item) for item in (args.query_id or []))
    if requested:
        cameras = [camera for camera in cameras if str(camera.image_name) in requested]
    if int(args.max_queries) > 0:
        cameras = cameras[: int(args.max_queries)]

    sparse_tes: list[float] = []
    sparse_res: list[float] = []
    sparse_inliers: list[int] = []
    sparse_filter_inputs: list[int] = []
    sparse_filter_outputs: list[int] = []
    scene_matcher_outputs: list[int] = []
    scene_detector_modes: list[str] = []
    scene_detector_diagnostic_flags: list[bool] = []
    landmark_activation_counts: list[int] = []
    landmark_activation_fallbacks: list[str] = []
    sparse_feedback_query_rows: list[dict[str, Any]] = []
    sparse_feedback_records: list[dict[str, Any]] = []
    query_features: dict[str, Any] = {}
    query_match_descriptors: dict[str, Any] = {}
    include_trace_descriptors = str(args.sparse_trace_descriptor_mode) == "full"
    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for camera in tqdm(cameras, desc="ULF sparse-only"):
        gt_w2c = camera.world_view_transform.transpose(0, 1).cpu().numpy()
        query_image = camera.original_image.to(device)
        query_image = _apply_masks(query_image, masks, camera.image_name)
        result = ulfloc.loc_coarse(
            camera,
            query_image,
            camera.FoVx,
            camera.FoVy,
            return_matches=bool(args.dump_sparse_match_diagnostics or args.dump_sparse_match_attributions),
        )
        re_deg, te_cm = imported["cal_pose_error"](result["pose_w2c"], gt_w2c)
        sparse_tes.append(float(te_cm))
        sparse_res.append(float(re_deg))
        sparse_inliers.append(int(result["inliers"]))
        filter_metadata = result.get("sparse_match_filter", {})
        if isinstance(filter_metadata, Mapping):
            sparse_filter_inputs.append(int(filter_metadata.get("input_match_count", 0)))
            sparse_filter_outputs.append(int(filter_metadata.get("output_match_count", 0)))
        scene_matcher_metadata = result.get("scene_matcher", {})
        if isinstance(scene_matcher_metadata, Mapping) and bool(scene_matcher_metadata.get("scene_matcher_enabled", False)):
            scene_matcher_outputs.append(int(scene_matcher_metadata.get("scene_matcher_selected_count", 0)))
        scene_detector_metadata = result.get("scene_detector", {})
        if isinstance(scene_detector_metadata, Mapping) and bool(scene_detector_metadata.get("enabled", False)):
            scene_detector_modes.append(str(scene_detector_metadata.get("scene_detector_mode", scene_detector_metadata.get("mode", ""))))
            scene_detector_diagnostic_flags.append(bool(scene_detector_metadata.get("diagnostic_only", False)))
        landmark_activation_metadata = result.get("landmark_activation", {})
        if isinstance(landmark_activation_metadata, Mapping) and bool(landmark_activation_metadata.get("enabled", False)):
            landmark_activation_counts.append(int(landmark_activation_metadata.get("selected_count", 0)))
            fallback = landmark_activation_metadata.get("fallback")
            if fallback:
                landmark_activation_fallbacks.append(str(fallback))
        row = {
            "image_name": str(camera.image_name),
            "sparse_te_cm": float(te_cm),
            "sparse_re_deg": float(re_deg),
            "sparse_inliers": int(result["inliers"]),
            "sparse_inlier_rate": float(result.get("inlier_rate", 0.0)),
            "sparse_match_filter": filter_metadata,
            "scene_detector": scene_detector_metadata,
            "scene_matcher": scene_matcher_metadata,
            "landmark_activation": landmark_activation_metadata,
        }
        if bool(args.dump_sparse_match_diagnostics):
            row["sparse_match_diagnostics"] = summarize_sparse_match_diagnostics(result.get("matches", {}))
        if bool(args.dump_sparse_match_attributions):
            matches = result.get("matches", {})
            sparse_feedback_query_rows.append(
                build_sparse_feedback_query_row(
                    scene=args.scene,
                    split_name=str(args.split_name),
                    query_id=str(camera.image_name),
                    image_id=str(camera.image_name),
                    sparse_te_cm=float(te_cm),
                    sparse_re_deg=float(re_deg),
                    matches=matches,
                )
            )
            attributions = serialize_sparse_match_attributions(
                matches,
                scene=args.scene,
                split_name=str(args.split_name),
                query_id=str(camera.image_name),
                image_id=str(camera.image_name),
                source_role="baseline_trace",
                include_descriptors=include_trace_descriptors,
                max_correspondences=int(args.sparse_trace_max_correspondences_per_query),
            )
            row["sparse_match_attribution_summary"] = compact_sparse_attributions_for_result_json(attributions)
            sparse_feedback_records.extend(attributions)
            if isinstance(matches, Mapping) and "query_global_feature" in matches:
                query_features[str(camera.image_name)] = matches["query_global_feature"]
            if include_trace_descriptors and isinstance(matches, Mapping) and "query_descriptor" in matches:
                query_match_descriptors[str(camera.image_name)] = matches["query_descriptor"]
        rows.append(row)
    if bool(config["scene_matcher"].get("enabled", False)) and cameras and not scene_matcher_outputs:
        raise RuntimeError(
            "scene_matcher checkpoint was configured, but no query reported scene_matcher_enabled=true; "
            "correspondence supervision did not enter inference"
        )
    if bool(config["landmark_activation"].get("enabled", False)) and cameras and not landmark_activation_counts:
        raise RuntimeError(
            "landmark_activation checkpoint was configured, but no query reported landmark_activation.enabled=true"
        )

    summary = summarize_sparse_metrics(
        translation_cm=sparse_tes,
        rotation_deg=sparse_res,
        inliers=sparse_inliers,
    )
    metrics = {
        "schema_version": "ulfloc_sparse_only_metrics_v1",
        "scene": args.scene,
        "split_name": str(args.split_name),
        "scene_detector_enabled": args.scene_detector_checkpoint is not None,
        "scene_detector_checkpoint": None if args.scene_detector_checkpoint is None else str(args.scene_detector_checkpoint),
        "sparse_match_diagnostics_enabled": bool(args.dump_sparse_match_diagnostics),
        "sparse_match_attributions_enabled": bool(args.dump_sparse_match_attributions),
        "sparse_trace_descriptor_mode": str(args.sparse_trace_descriptor_mode),
        "sparse_trace_max_correspondences_per_query": int(args.sparse_trace_max_correspondences_per_query),
        "scene_matcher": dict(config["scene_matcher"]),
        "scene_matcher_enabled_query_count": int(len(scene_matcher_outputs)),
        "scene_matcher_mean_selected_count": float(np.mean(scene_matcher_outputs))
        if scene_matcher_outputs
        else None,
        "scene_detector": {
            **dict(config.get("scene_specific_detector", {})),
            "enabled_query_count": int(len(scene_detector_modes)),
            "observed_modes": sorted(set(scene_detector_modes)),
            "diagnostic_query_count": int(sum(1 for item in scene_detector_diagnostic_flags if item)),
        },
        "landmark_activation": {
            **dict(config["landmark_activation"]),
            **dict(landmark_activation_audit),
            "active_index_space": "sampled_landmark_rows",
            "matched_index_mapping": "active_local_to_sampled_rows_to_gaussian_ids",
        },
        "landmark_activation_enabled_query_count": int(len(landmark_activation_counts)),
        "landmark_activation_mean_selected_count": float(np.mean(landmark_activation_counts))
        if landmark_activation_counts
        else None,
        "landmark_activation_min_selected_count": int(np.min(landmark_activation_counts))
        if landmark_activation_counts
        else None,
        "landmark_activation_max_selected_count": int(np.max(landmark_activation_counts))
        if landmark_activation_counts
        else None,
        "landmark_activation_fallback_query_count": int(len(landmark_activation_fallbacks)),
        "sparse_match_filter": sparse_match_filter_metrics,
        "sparse_match_filter_mean_input_matches": float(np.mean(sparse_filter_inputs))
        if sparse_filter_inputs
        else None,
        "sparse_match_filter_mean_output_matches": float(np.mean(sparse_filter_outputs))
        if sparse_filter_outputs
        else None,
        **summary,
    }
    split_audit = build_sparse_only_split_audit(str(args.split_name))
    sparse_pnp_trace_path = None
    if bool(args.dump_sparse_match_attributions):
        sparse_trace_payload = build_sparse_feedback_trace_payload(
            scene=args.scene,
            split_name=str(args.split_name),
            query_rows=sparse_feedback_query_rows,
            correspondence_rows=sparse_feedback_records,
            query_features=query_features,
            query_match_descriptors=query_match_descriptors,
            split_audit=split_audit,
            descriptor_mode=str(args.sparse_trace_descriptor_mode),
        )
        sparse_pnp_trace_path = output_dir / "sparse_pnp_trace_payload.pt"
        torch.save(sparse_trace_payload, sparse_pnp_trace_path)
        _write_json(output_dir / "sparse_pnp_trace_payload.json", sparse_trace_payload_for_json(sparse_trace_payload))
    metrics["sparse_pnp_trace_payload"] = None if sparse_pnp_trace_path is None else str(sparse_pnp_trace_path)
    metrics["sparse_pnp_trace_record_count"] = int(len(sparse_feedback_records))
    metrics["sparse_pnp_trace_query_feature_count"] = int(len(query_features))
    metrics["sparse_pnp_trace_query_match_descriptor_query_count"] = int(len(query_match_descriptors))
    manifest = {
        "schema_version": "ulfloc_sparse_only_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "source_path": str(args.source_path),
        "model_path": str(args.model_path),
        "input_log_dir": str(args.input_log_dir),
        "cfg": str(args.cfg),
        "output_dir": str(output_dir),
        "landmark_activation": metrics["landmark_activation"],
        "sparse_pnp_trace_payload": None if sparse_pnp_trace_path is None else str(sparse_pnp_trace_path),
        "metrics": metrics,
        "split_audit": split_audit,
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "results.json", {"rows": rows})
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
