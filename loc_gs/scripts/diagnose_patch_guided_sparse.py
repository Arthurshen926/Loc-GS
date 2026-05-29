#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from loc_gs.diagnostics.match_visualization import (
    draw_match_canvas,
    pose_error_cm_deg,
    project_points,
    summarize_match_quality,
    write_json,
)
from loc_gs.dense_support.sparse_conditioned_dense_preflight import (
    DenseTransitionPolicy,
    select_sparse_conditioned_dense_transition,
)
from loc_gs.reporting.artifact_audit import write_artifact_audit_bundle
from loc_gs.scripts.visualize_stdloc_hard_matches import (
    _build_context,
    _capture_dense,
    _capture_sparse,
    _inlier_mask,
    _resolve_slcdp_effective_options,
    _tensor_to_image,
    _write_matches_csv,
)
from loc_gs.stdloc_native.patch_guided_sparse import (
    Patch,
    PatchHypothesis,
    evaluate_pose_global_consistency,
    filter_matches_by_reference_reprojection,
    filter_matches_by_patch,
    generate_patch_grid,
    merge_group_matches,
    refine_pose_with_reference_prior,
    score_patch_hypothesis,
    select_pose_consistent_patch_group,
)


DIAGNOSTIC_ONLY = True


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _resolve_diagnostic_slcdp_options(args: argparse.Namespace) -> dict[str, Any]:
    options = _resolve_slcdp_effective_options(args)
    if bool(getattr(args, "slcdp_sparse_conditioned_legacy_repair_selection", False)):
        if str(getattr(args, "slcdp_render_control", "")) != "sparse_conditioned":
            raise ValueError("--slcdp_sparse_conditioned_legacy_repair_selection requires --slcdp_render_control sparse_conditioned")
        options.update(
            {
                "slcdp_repair_skip_if_no_accept": False,
                "slcdp_repair_require_accept": False,
                "slcdp_repair_min_score_gain": 0.05,
                "slcdp_repair_translation_penalty_per_m": 0.12,
            }
        )
    return options


def _capture_sparse_with_overrides(
    ctx: Mapping[str, Any],
    query_image: torch.Tensor,
    fovx: float,
    fovy: float,
    *,
    solver: str | None,
    max_iterations: int | None,
    min_iterations: int | None,
) -> dict[str, Any]:
    loc = ctx["localizer"]
    sparse_cfg = loc.config["sparse"]
    old_values = {
        "solver": sparse_cfg.get("solver"),
        "max_iterations": sparse_cfg.get("max_iterations"),
        "min_iterations": sparse_cfg.get("min_iterations"),
    }
    try:
        if solver:
            sparse_cfg["solver"] = str(solver)
        if max_iterations is not None:
            sparse_cfg["max_iterations"] = int(max_iterations)
        if min_iterations is not None:
            sparse_cfg["min_iterations"] = int(min_iterations)
        return _capture_sparse(dict(ctx), query_image, fovx, fovy)
    finally:
        sparse_cfg.update(old_values)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in str(value).split(",") if item.strip())


def _camera_center_rotation(pose_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    center = -rotation.T @ pose[:3, 3]
    return center.astype(np.float64), rotation.astype(np.float64)


def _rotation_delta_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3) @ np.asarray(rotation_b, dtype=np.float64).reshape(3, 3).T
    trace = float(np.trace(relative))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _dense_pgsh_base_trust_acceptance(
    *,
    base_dense_pose_w2c: np.ndarray,
    candidate_pose_w2c: np.ndarray,
    max_translation_delta_m: float,
    max_rotation_delta_deg: float,
) -> dict[str, Any]:
    base_center, base_rotation = _camera_center_rotation(base_dense_pose_w2c)
    candidate_center, candidate_rotation = _camera_center_rotation(candidate_pose_w2c)
    translation_delta = float(np.linalg.norm(base_center - candidate_center))
    rotation_delta = _rotation_delta_deg(base_rotation, candidate_rotation)
    translation_tol = max(1.0e-6, abs(float(max_translation_delta_m)) * 1.0e-6)
    rotation_tol = max(1.0e-6, abs(float(max_rotation_delta_deg)) * 1.0e-6)
    failed_checks = {
        "translation_from_base_dense": bool(translation_delta > float(max_translation_delta_m) + translation_tol),
        "rotation_from_base_dense": bool(rotation_delta > float(max_rotation_delta_deg) + rotation_tol),
    }
    return {
        "schema": "loc_gs_dense_pgsh_base_trust_v1",
        "decision": "accept_dense_pgsh_pose" if not any(failed_checks.values()) else "reject_dense_pgsh_keep_base_dense",
        "translation_delta_m": translation_delta,
        "rotation_delta_deg": rotation_delta,
        "max_translation_delta_m": float(max_translation_delta_m),
        "max_rotation_delta_deg": float(max_rotation_delta_deg),
        "failed_checks": failed_checks,
        "diagnostic_only": True,
    }


def _dense_pgsh_report_aliases(dense_pgsh_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Expose dense-only PGSH metrics at dense-summary level with unambiguous names."""
    if not bool(dense_pgsh_summary.get("enabled", False)):
        return {
            "dense_pgsh_effective_te_cm": None,
            "dense_pgsh_effective_re_deg": None,
            "dense_pgsh_raw_candidate_te_cm": None,
            "dense_pgsh_raw_candidate_re_deg": None,
            "dense_pgsh_decision": "disabled",
        }
    return {
        "dense_pgsh_effective_te_cm": dense_pgsh_summary.get("dense_pgsh_te_cm"),
        "dense_pgsh_effective_re_deg": dense_pgsh_summary.get("dense_pgsh_re_deg"),
        "dense_pgsh_raw_candidate_te_cm": dense_pgsh_summary.get("dense_pgsh_raw_te_cm"),
        "dense_pgsh_raw_candidate_re_deg": dense_pgsh_summary.get("dense_pgsh_raw_re_deg"),
        "dense_pgsh_decision": dense_pgsh_summary.get("decision"),
    }


def _camera_depths(points_world: np.ndarray, pose_w2c: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (pose @ homog.T).T[:, 2].astype(np.float64)


def _reprojection_errors(
    *,
    query_xy: np.ndarray,
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    projected, valid = project_points(points_world, pose_w2c, intrinsic, width=width, height=height)
    errors = np.linalg.norm(np.asarray(projected, dtype=np.float64) - np.asarray(query_xy, dtype=np.float64), axis=1)
    errors[~valid] = np.inf
    return errors.astype(np.float64)


def _build_dense_pgsh_refinement_matches(
    dense_merged: Mapping[str, Any],
    sparse: Mapping[str, Any],
    *,
    reference_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    sparse_anchor_weight: float,
    sparse_anchor_max_count: int,
    sparse_anchor_max_reprojection_px: float,
) -> dict[str, Any]:
    dense_xy = np.asarray(dense_merged.get("xy", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    dense_xyz = np.asarray(dense_merged.get("xyz", np.empty((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    dense_weights = np.ones((dense_xy.shape[0],), dtype=np.float32)
    diagnostics: dict[str, Any] = {
        "schema": "loc_gs_dense_pgsh_sparse_anchor_refinement_v1",
        "enabled": bool(float(sparse_anchor_weight) > 0.0),
        "dense_match_count": int(dense_xy.shape[0]),
        "candidate_anchor_count": 0,
        "selected_anchor_count": 0,
        "sparse_anchor_weight": float(sparse_anchor_weight),
        "sparse_anchor_max_count": int(sparse_anchor_max_count),
        "sparse_anchor_max_reprojection_px": float(sparse_anchor_max_reprojection_px),
        "diagnostic_only": True,
    }
    if float(sparse_anchor_weight) <= 0.0:
        return {"xy": dense_xy, "xyz": dense_xyz, "weights": dense_weights, "diagnostics": diagnostics}

    sparse_xy_all = np.asarray(sparse.get("query_xy", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    sparse_xyz_all = np.asarray(sparse.get("p3d", np.empty((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    sparse_inliers = np.asarray(sparse.get("inliers", np.empty((0,), dtype=np.int32)), dtype=np.int64).reshape(-1)
    max_index = min(int(sparse_xy_all.shape[0]), int(sparse_xyz_all.shape[0]))
    valid_inliers = sparse_inliers[(sparse_inliers >= 0) & (sparse_inliers < max_index)]
    diagnostics["candidate_anchor_count"] = int(valid_inliers.shape[0])
    if valid_inliers.shape[0] == 0:
        return {"xy": dense_xy, "xyz": dense_xyz, "weights": dense_weights, "diagnostics": diagnostics}

    anchor_xy = sparse_xy_all[valid_inliers]
    anchor_xyz = sparse_xyz_all[valid_inliers]
    errors = _reprojection_errors(
        query_xy=anchor_xy,
        points_world=anchor_xyz,
        pose_w2c=np.asarray(reference_pose_w2c, dtype=np.float32).reshape(4, 4),
        intrinsic=np.asarray(intrinsic, dtype=np.float32).reshape(3, 3),
        width=int(width),
        height=int(height),
    )
    keep = np.isfinite(errors) & (errors <= float(sparse_anchor_max_reprojection_px))
    kept_indices = np.flatnonzero(keep).astype(np.int64)
    if kept_indices.shape[0] > int(sparse_anchor_max_count) > 0:
        order = np.argsort(errors[kept_indices], kind="mergesort")
        kept_indices = kept_indices[order[: int(sparse_anchor_max_count)]]
    selected_xy = anchor_xy[kept_indices]
    selected_xyz = anchor_xyz[kept_indices]
    diagnostics["selected_anchor_count"] = int(selected_xy.shape[0])
    finite_errors = errors[np.isfinite(errors)]
    diagnostics["candidate_median_reprojection_error_px"] = float(np.median(finite_errors)) if finite_errors.size else None
    diagnostics["selected_median_reprojection_error_px"] = (
        float(np.median(errors[kept_indices])) if kept_indices.shape[0] else None
    )
    if selected_xy.shape[0] == 0:
        return {"xy": dense_xy, "xyz": dense_xyz, "weights": dense_weights, "diagnostics": diagnostics}

    xy = np.concatenate([dense_xy, selected_xy.astype(np.float32)], axis=0)
    xyz = np.concatenate([dense_xyz, selected_xyz.astype(np.float32)], axis=0)
    weights = np.concatenate(
        [dense_weights, np.full((selected_xy.shape[0],), float(sparse_anchor_weight), dtype=np.float32)],
        axis=0,
    )
    return {"xy": xy, "xyz": xyz, "weights": weights, "diagnostics": diagnostics}


def _solve_sparse_pose(
    ctx: Mapping[str, Any],
    query_xy: np.ndarray,
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    *,
    solver: str | None = None,
    max_iterations: int | None = None,
    min_iterations: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loc = ctx["localizer"]
    stdloc = ctx["stdloc"]
    if int(np.asarray(query_xy).shape[0]) < 4:
        return np.eye(4, dtype=np.float32), np.empty((0,), dtype=np.int32)
    pose, inliers = stdloc.solve_pose(
        np.asarray(query_xy, dtype=np.float32),
        np.asarray(points_world, dtype=np.float32),
        intrinsic,
        str(solver if solver is not None else loc.config["sparse"]["solver"]),
        loc.config["sparse"]["reprojection_error"],
        loc.config["sparse"]["confidence"],
        int(max_iterations if max_iterations is not None else loc.config["sparse"]["max_iterations"]),
        int(min_iterations if min_iterations is not None else loc.config["sparse"]["min_iterations"]),
    )
    return np.asarray(pose, dtype=np.float32).reshape(4, 4), np.asarray(inliers).reshape(-1).astype(np.int32)


def _solve_stage_pose(
    ctx: Mapping[str, Any],
    query_xy: np.ndarray,
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    *,
    stage: str,
    solver: str | None = None,
    max_iterations: int | None = None,
    min_iterations: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loc = ctx["localizer"]
    stdloc = ctx["stdloc"]
    cfg = loc.config[str(stage)]
    if int(np.asarray(query_xy).shape[0]) < 4:
        return np.eye(4, dtype=np.float32), np.empty((0,), dtype=np.int32)
    pose, inliers = stdloc.solve_pose(
        np.asarray(query_xy, dtype=np.float32),
        np.asarray(points_world, dtype=np.float32),
        intrinsic,
        str(solver if solver is not None else cfg["solver"]),
        cfg["reprojection_error"],
        cfg["confidence"],
        int(max_iterations if max_iterations is not None else cfg["max_iterations"]),
        int(min_iterations if min_iterations is not None else cfg["min_iterations"]),
    )
    return np.asarray(pose, dtype=np.float32).reshape(4, 4), np.asarray(inliers).reshape(-1).astype(np.int32)


def _unique_patches(patches: Sequence[Patch]) -> list[Patch]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[Patch] = []
    for patch in patches:
        key = (int(patch.x0), int(patch.y0), int(patch.x1), int(patch.y1))
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            Patch(
                patch_id=len(unique),
                x0=patch.x0,
                y0=patch.y0,
                x1=patch.x1,
                y1=patch.y1,
                image_width=patch.image_width,
                image_height=patch.image_height,
            )
        )
    return unique


def build_multiscale_patches(
    *,
    image_size: tuple[int, int],
    grids: Sequence[int],
    overlap_ratio: float,
) -> list[Patch]:
    """Build deterministic 2x2/3x3-style overlapping patches without GT."""

    width, height = int(image_size[0]), int(image_size[1])
    ratio = float(np.clip(float(overlap_ratio), 0.0, 0.95))
    patches: list[Patch] = []
    for grid in grids:
        grid = max(1, int(grid))
        denom = 1.0 + float(grid - 1) * (1.0 - ratio)
        patch_w = max(4, int(round(float(width) / denom)))
        patch_h = max(4, int(round(float(height) / denom)))
        overlap_w = min(patch_w - 1, int(round(patch_w * ratio)))
        overlap_h = min(patch_h - 1, int(round(patch_h * ratio)))
        patches.extend(
            generate_patch_grid(
                image_size=(width, height),
                patch_size=(patch_w, patch_h),
                overlap=(overlap_w, overlap_h),
            )
        )
    return _unique_patches(patches)


def _matches_from_sparse(sparse: Mapping[str, Any]) -> dict[str, np.ndarray]:
    xy = np.asarray(sparse["query_xy"], dtype=np.float32).reshape(-1, 2)
    matches = {
        "xy": np.asarray(sparse["query_xy"], dtype=np.float32).reshape(-1, 2),
        "xyz": np.asarray(sparse["p3d"], dtype=np.float32).reshape(-1, 3),
        "gs_ids": np.asarray(sparse["gs_ids"], dtype=np.int64).reshape(-1),
        "score": np.asarray(sparse.get("scores", np.ones((int(xy.shape[0]),), dtype=np.float32)), dtype=np.float32).reshape(-1),
    }
    detector_scores = sparse.get("detector_scores")
    if detector_scores is not None:
        detector_scores = np.asarray(detector_scores, dtype=np.float32).reshape(-1)
        if detector_scores.shape[0] == xy.shape[0]:
            matches["detector_score"] = detector_scores
    return matches


def _matches_from_dense(dense: Mapping[str, Any]) -> dict[str, np.ndarray]:
    query_xy = np.asarray(dense["query_xy"], dtype=np.float32).reshape(-1, 2)
    rendered_xy = np.asarray(dense.get("rendered_xy", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    if rendered_xy.shape[0] != query_xy.shape[0]:
        rendered_xy = np.zeros_like(query_xy)
    matches = {
        "xy": query_xy,
        "xyz": np.asarray(dense["p3d"], dtype=np.float32).reshape(-1, 3),
        "rendered_xy": rendered_xy,
        "score": np.ones((int(query_xy.shape[0]),), dtype=np.float32),
        "indices": np.arange(int(query_xy.shape[0]), dtype=np.int64),
    }
    detector_scores = dense.get("detector_scores")
    if detector_scores is not None:
        detector_scores = np.asarray(detector_scores, dtype=np.float32).reshape(-1)
        if detector_scores.shape[0] == query_xy.shape[0]:
            matches["detector_score"] = detector_scores
    return matches


@torch.no_grad()
def _scene_detector_scores_for_xy(ctx: Mapping[str, Any], fine_map: torch.Tensor, xy: Any) -> np.ndarray:
    """Sample STDLoc's scene-specific detector heatmap at query xy locations."""
    coords = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if coords.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    stdloc = ctx["stdloc"]
    loc = ctx["localizer"]
    height, width = int(fine_map.shape[-2]), int(fine_map.shape[-1])
    heat = loc.detector(fine_map)
    kp_scores = stdloc.simple_nms(heat, loc.config["sparse"].get("nms", 4)).reshape(-1)
    x = np.clip(np.floor(coords[:, 0]).astype(np.int64), 0, max(0, width - 1))
    y = np.clip(np.floor(coords[:, 1]).astype(np.int64), 0, max(0, height - 1))
    ids = torch.as_tensor(y * width + x, device=kp_scores.device, dtype=torch.long)
    return kp_scores[ids].detach().cpu().numpy().astype(np.float32)


def _empty_matches() -> dict[str, np.ndarray]:
    return {
        "xy": np.empty((0, 2), dtype=np.float32),
        "xyz": np.empty((0, 3), dtype=np.float32),
        "gs_ids": np.empty((0,), dtype=np.int64),
        "score": np.empty((0,), dtype=np.float32),
        "detector_score": np.empty((0,), dtype=np.float32),
        "indices": np.empty((0,), dtype=np.int64),
    }


def _empty_match_pool() -> dict[str, list[np.ndarray]]:
    return {"xy": [], "xyz": [], "gs_ids": [], "score": [], "detector_score": []}


def _empty_dense_match_pool() -> dict[str, list[np.ndarray]]:
    return {"xy": [], "xyz": [], "rendered_xy": [], "score": [], "detector_score": []}


def _finalize_match_pool(pool: Mapping[str, Sequence[np.ndarray]]) -> dict[str, np.ndarray]:
    if not pool.get("xy"):
        return _empty_matches()
    merged = {
        "xy": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, 2) for v in pool["xy"]], axis=0),
        "xyz": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, 3) for v in pool["xyz"]], axis=0),
        "gs_ids": np.concatenate([np.asarray(v, dtype=np.int64).reshape(-1) for v in pool["gs_ids"]], axis=0),
        "score": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in pool["score"]], axis=0),
        "detector_score": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in pool["detector_score"]], axis=0),
    }
    merged["indices"] = np.arange(int(merged["xy"].shape[0]), dtype=np.int64)
    return merged


def _finalize_dense_match_pool(pool: Mapping[str, Sequence[np.ndarray]]) -> dict[str, np.ndarray]:
    if not pool.get("xy"):
        empty = _empty_matches()
        empty["rendered_xy"] = np.empty((0, 2), dtype=np.float32)
        return empty
    merged = {
        "xy": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, 2) for v in pool["xy"]], axis=0),
        "xyz": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, 3) for v in pool["xyz"]], axis=0),
        "rendered_xy": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, 2) for v in pool["rendered_xy"]], axis=0),
        "score": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in pool["score"]], axis=0),
        "detector_score": np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in pool["detector_score"]], axis=0),
    }
    merged["indices"] = np.arange(int(merged["xy"].shape[0]), dtype=np.int64)
    return merged


def _append_match_subset(
    pool: dict[str, list[np.ndarray]],
    matches: Mapping[str, Any],
    selected_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
    count_before = int(sum(np.asarray(part).reshape(-1, 2).shape[0] for part in pool["xy"]))
    selected_count = int(mask.sum())
    if selected_count == 0:
        return np.empty((0,), dtype=np.int64)
    pool["xy"].append(np.asarray(matches["xy"], dtype=np.float32).reshape(-1, 2)[mask])
    pool["xyz"].append(np.asarray(matches["xyz"], dtype=np.float32).reshape(-1, 3)[mask])
    pool["gs_ids"].append(np.asarray(matches.get("gs_ids", np.zeros(mask.shape[0], dtype=np.int64)), dtype=np.int64).reshape(-1)[mask])
    pool["score"].append(np.asarray(matches.get("score", np.ones(mask.shape[0], dtype=np.float32)), dtype=np.float32).reshape(-1)[mask])
    pool["detector_score"].append(
        np.asarray(matches.get("detector_score", np.full(mask.shape[0], 0.5, dtype=np.float32)), dtype=np.float32).reshape(-1)[mask]
    )
    return np.arange(count_before, count_before + selected_count, dtype=np.int64)


def _append_dense_match_subset(
    pool: dict[str, list[np.ndarray]],
    matches: Mapping[str, Any],
    selected_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
    count_before = int(sum(np.asarray(part).reshape(-1, 2).shape[0] for part in pool["xy"]))
    selected_count = int(mask.sum())
    if selected_count == 0:
        return np.empty((0,), dtype=np.int64)
    pool["xy"].append(np.asarray(matches["xy"], dtype=np.float32).reshape(-1, 2)[mask])
    pool["xyz"].append(np.asarray(matches["xyz"], dtype=np.float32).reshape(-1, 3)[mask])
    pool["rendered_xy"].append(np.asarray(matches["rendered_xy"], dtype=np.float32).reshape(-1, 2)[mask])
    pool["score"].append(np.asarray(matches.get("score", np.ones(mask.shape[0], dtype=np.float32)), dtype=np.float32).reshape(-1)[mask])
    pool["detector_score"].append(
        np.asarray(matches.get("detector_score", np.full(mask.shape[0], 0.5, dtype=np.float32)), dtype=np.float32).reshape(-1)[mask]
    )
    return np.arange(count_before, count_before + selected_count, dtype=np.int64)


@torch.no_grad()
def _match_patch_locally(
    ctx: Mapping[str, Any],
    sparse: Mapping[str, Any],
    patch: Patch,
    *,
    patch_detect_num: int,
    patch_dual_softmax: str,
    patch_mnn_match: str,
    patch_match_threshold: float | None,
) -> dict[str, np.ndarray]:
    """Run STDLoc-style sparse matching on patch-local query keypoints only."""

    stdloc = ctx["stdloc"]
    loc = ctx["localizer"]
    fine_map = sparse["fine_map"]
    height, width = int(fine_map.shape[-2]), int(fine_map.shape[-1])
    heat = loc.detector(fine_map)
    kp_scores = stdloc.simple_nms(heat, loc.config["sparse"].get("nms", 4)).flatten()

    xs = torch.arange(width, device=kp_scores.device).repeat(height)
    ys = torch.arange(height, device=kp_scores.device).repeat_interleave(width)
    patch_mask = (
        (xs >= int(patch.x0))
        & (xs < int(patch.x1))
        & (ys >= int(patch.y0))
        & (ys < int(patch.y1))
        & (kp_scores > 0)
    )
    candidate_ids = torch.nonzero(patch_mask, as_tuple=False).reshape(-1)
    if candidate_ids.numel() == 0:
        return _empty_matches()
    detect_num = min(int(max(1, patch_detect_num)), int(candidate_ids.numel()))
    _scores, order = torch.topk(kp_scores[candidate_ids], detect_num)
    kp_ids = candidate_ids[order]

    sampled_features = fine_map.reshape(fine_map.shape[0], -1)[:, kp_ids]
    landmark_features = F.normalize(loc.landmarks.get_loc_feature.squeeze(), dim=-1)
    corr = torch.matmul(sampled_features.T, landmark_features.T)
    corr = stdloc.apply_landmark_prior(
        corr,
        loc.landmark_prior,
        weight=loc.config["sparse"].get("landmark_prior_weight", 0.0),
    )
    use_dual_softmax = bool(loc.config["sparse"]["dual_softmax"])
    if str(patch_dual_softmax) == "on":
        use_dual_softmax = True
    elif str(patch_dual_softmax) == "off":
        use_dual_softmax = False
    use_mnn_match = bool(loc.config["sparse"]["mnn_match"])
    if str(patch_mnn_match) == "on":
        use_mnn_match = True
    elif str(patch_mnn_match) == "off":
        use_mnn_match = False
    threshold = loc.config["sparse"]["threshold"] if patch_match_threshold is None else float(patch_match_threshold)
    if use_dual_softmax:
        corr = stdloc.dual_softmax(corr, temp=loc.config["sparse"]["dual_softmax_temp"])
    if use_mnn_match:
        _b, im_idx, gs_ids = stdloc.mnn_match(corr[None], thr=threshold)
        im_idx = im_idx.reshape(-1)
        gs_ids = gs_ids.reshape(-1)
        scores = corr[im_idx, gs_ids] if im_idx.numel() else corr.new_empty(0)
    else:
        im_idx, gs_ids, scores = stdloc.topk_match(
            corr[None],
            loc.config["sparse"]["topk"],
            thr=threshold,
        )
        im_idx = im_idx.reshape(-1)
        gs_ids = gs_ids.reshape(-1)
        scores = scores.reshape(-1)
    if im_idx.numel() == 0:
        return _empty_matches()
    grid = torch.stack([torch.arange(height * width) % width, torch.arange(height * width) // width], dim=1)
    p2d = grid[kp_ids.detach().cpu()][im_idx.detach().cpu()].numpy().astype(np.float32) + 0.5
    p3d = loc.landmarks.get_xyz[gs_ids].detach().cpu().numpy().astype(np.float32)
    detector_scores = kp_scores[kp_ids][im_idx].detach().cpu().numpy().astype(np.float32)
    return {
        "xy": p2d,
        "xyz": p3d,
        "gs_ids": gs_ids.detach().cpu().numpy().astype(np.int64),
        "score": scores.detach().cpu().numpy().astype(np.float32),
        "detector_score": detector_scores,
        "indices": np.arange(int(p2d.shape[0]), dtype=np.int64),
    }


def _patch_to_dict(patch: Patch) -> dict[str, Any]:
    return {
        "patch_id": int(patch.patch_id),
        "x0": int(patch.x0),
        "y0": int(patch.y0),
        "x1": int(patch.x1),
        "y1": int(patch.y1),
        "width": int(patch.width),
        "height": int(patch.height),
        "area": int(patch.area),
    }


def _hypothesis_to_dict(item: PatchHypothesis, patch: Patch, *, match_count: int) -> dict[str, Any]:
    return {
        "patch": _patch_to_dict(patch),
        "score": float(item.score),
        "inlier_count": int(item.inlier_count),
        "match_count": int(match_count),
        "match_indices": [int(v) for v in item.match_indices.tolist()],
        "components": {key: float(value) for key, value in item.components.items()},
    }


def _pose_from_center_rotation(center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = -pose[:3, :3] @ np.asarray(center, dtype=np.float64).reshape(3)
    return pose


def build_patch_guided_sparse_hypotheses(
    ctx: Mapping[str, Any],
    sparse: Mapping[str, Any],
    *,
    patches: Sequence[Patch],
    patch_match_mode: str,
    patch_detect_num: int,
    patch_dual_softmax: str,
    patch_mnn_match: str,
    patch_match_threshold: float | None,
    min_patch_matches: int,
    min_patch_inliers: int,
    patch_pnp_max_iterations: int,
    patch_pnp_min_iterations: int,
    patch_pnp_solver: str,
    detector_score_weight: float = 0.0,
    global_consistency: bool = True,
    global_sparse_query_xy: np.ndarray | None = None,
    global_sparse_points_world: np.ndarray | None = None,
    global_sparse_inlier_indices: np.ndarray | None = None,
    sparse_retention_min_ratio: float = 0.70,
    sparse_retention_max_error_px: float = 8.0,
    reference_pose_w2c: np.ndarray | None = None,
    reference_max_translation_delta_m: float | None = None,
    reference_max_rotation_delta_deg: float | None = None,
    reference_match_max_reprojection_px: float = 0.0,
    reference_match_min_keep: int = 4,
    diagnostic_gt_w2c: np.ndarray | None = None,
    diagnostic_good_px: float = 5.0,
) -> tuple[list[PatchHypothesis], list[dict[str, Any]], dict[str, np.ndarray]]:
    global_matches = _matches_from_sparse(sparse)
    candidate_pool = _empty_match_pool()
    hypotheses: list[PatchHypothesis] = []
    rows: list[dict[str, Any]] = []
    for patch in patches:
        if str(patch_match_mode) == "local":
            patch_matches = _match_patch_locally(
                ctx,
                sparse,
                patch,
                patch_detect_num=int(patch_detect_num),
                patch_dual_softmax=str(patch_dual_softmax),
                patch_mnn_match=str(patch_mnn_match),
                patch_match_threshold=patch_match_threshold,
            )
        else:
            patch_matches = filter_matches_by_patch(global_matches, patch)
        reference_filter = None
        original_match_count = int(np.asarray(patch_matches["xy"]).shape[0])
        if reference_pose_w2c is not None and float(reference_match_max_reprojection_px) > 0.0 and original_match_count > 0:
            patch_matches, reference_filter = filter_matches_by_reference_reprojection(
                patch_matches,
                pose_w2c=reference_pose_w2c,
                intrinsic=sparse["K"],
                image_size=(int(sparse["width"]), int(sparse["height"])),
                max_reprojection_error_px=float(reference_match_max_reprojection_px),
            )
        match_count = int(np.asarray(patch_matches["xy"]).shape[0])
        if reference_filter is not None and match_count < int(reference_match_min_keep):
            rows.append(
                {
                    "patch": _patch_to_dict(patch),
                    "match_mode": str(patch_match_mode),
                    "match_count": match_count,
                    "original_match_count": original_match_count,
                    "skipped": "too_few_reference_consistent_matches",
                    "reference_filter": reference_filter,
                }
            )
            continue
        if match_count < int(min_patch_matches):
            rows.append(
                {
                    "patch": _patch_to_dict(patch),
                    "match_mode": str(patch_match_mode),
                    "match_count": match_count,
                    "original_match_count": original_match_count,
                    "reference_filter": reference_filter,
                    "skipped": "too_few_matches",
                }
            )
            continue
        pose, inliers = _solve_sparse_pose(
            ctx,
            patch_matches["xy"],
            patch_matches["xyz"],
            sparse["K"],
            solver=str(patch_pnp_solver),
            max_iterations=int(patch_pnp_max_iterations),
            min_iterations=int(patch_pnp_min_iterations),
        )
        inlier_mask = _inlier_mask(match_count, inliers)
        if int(inlier_mask.sum()) < int(min_patch_inliers):
            rows.append(
                {
                    "patch": _patch_to_dict(patch),
                    "match_mode": str(patch_match_mode),
                    "match_count": match_count,
                    "original_match_count": original_match_count,
                    "inlier_count": int(inlier_mask.sum()),
                    "reference_filter": reference_filter,
                    "skipped": "too_few_inliers",
                }
            )
            continue
        errors = _reprojection_errors(
            query_xy=patch_matches["xy"],
            points_world=patch_matches["xyz"],
            pose_w2c=pose,
            intrinsic=sparse["K"],
            width=int(sparse["width"]),
            height=int(sparse["height"]),
        )
        depths = _camera_depths(patch_matches["xyz"], pose)
        center, rotation = _camera_center_rotation(pose)
        consistency = None
        if bool(global_consistency):
            consistency = evaluate_pose_global_consistency(
                pose_w2c=pose,
                intrinsic=sparse["K"],
                image_size=(int(sparse["width"]), int(sparse["height"])),
                sparse_query_xy=global_sparse_query_xy,
                sparse_points_world=global_sparse_points_world,
                sparse_inlier_indices=global_sparse_inlier_indices,
                sparse_retention_max_error_px=float(sparse_retention_max_error_px),
                sparse_retention_min_ratio=float(sparse_retention_min_ratio),
                reference_pose_w2c=reference_pose_w2c,
                reference_max_translation_delta_m=reference_max_translation_delta_m,
                reference_max_rotation_delta_deg=reference_max_rotation_delta_deg,
            )
            if str(consistency.get("decision")) != "accept_patch_pose":
                rows.append(
                    {
                        "patch": _patch_to_dict(patch),
                        "match_mode": str(patch_match_mode),
                        "match_count": match_count,
                        "original_match_count": original_match_count,
                        "inlier_count": int(inlier_mask.sum()),
                        "skipped": "global_consistency_reject",
                        "reference_filter": reference_filter,
                        "global_consistency": consistency,
                    }
                )
                continue
        pool_indices = _append_match_subset(candidate_pool, patch_matches, inlier_mask)
        hypothesis = score_patch_hypothesis(
            patch,
            match_xy=patch_matches["xy"],
            inlier_mask=inlier_mask,
            reproj_errors=errors,
            depths=depths,
            bearings_2d=patch_matches["xy"],
            camera_center=center,
            rotation=rotation,
            match_indices=pool_indices,
            detector_scores=patch_matches.get("detector_score"),
            detector_score_weight=float(detector_score_weight),
        )
        hypotheses.append(hypothesis)
        row = _hypothesis_to_dict(hypothesis, patch, match_count=match_count)
        row["match_mode"] = str(patch_match_mode)
        row["original_match_count"] = original_match_count
        if reference_filter is not None:
            row["reference_filter"] = reference_filter
        if consistency is not None:
            row["global_consistency"] = consistency
        if diagnostic_gt_w2c is not None:
            gt_errors = _reprojection_errors(
                query_xy=patch_matches["xy"],
                points_world=patch_matches["xyz"],
                pose_w2c=np.asarray(diagnostic_gt_w2c, dtype=np.float64).reshape(4, 4),
                intrinsic=sparse["K"],
                width=int(sparse["width"]),
                height=int(sparse["height"]),
            )
            gt_good = np.isfinite(gt_errors) & (gt_errors <= float(diagnostic_good_px))
            patch_pose = _pose_from_center_rotation(hypothesis.camera_center, hypothesis.rotation)
            patch_te, patch_re = pose_error_cm_deg(patch_pose, np.asarray(diagnostic_gt_w2c, dtype=np.float64).reshape(4, 4))
            row["diagnostic_gt_good_count"] = int(gt_good.sum())
            row["diagnostic_gt_good_ratio"] = float(gt_good.sum() / max(int(gt_good.shape[0]), 1))
            row["diagnostic_solver_inlier_gt_good_count"] = int((gt_good & inlier_mask).sum())
            row["diagnostic_solver_inlier_gt_bad_count"] = int((~gt_good & inlier_mask).sum())
            row["diagnostic_pose_te_cm"] = float(patch_te)
            row["diagnostic_pose_re_deg"] = float(patch_re)
            row["diagnostic_median_gt_reprojection_px"] = float(np.median(gt_errors[np.isfinite(gt_errors)])) if np.isfinite(gt_errors).any() else None
        rows.append(row)
    return hypotheses, rows, _finalize_match_pool(candidate_pool)


def build_patch_guided_dense_hypotheses(
    ctx: Mapping[str, Any],
    dense: Mapping[str, Any],
    *,
    patches: Sequence[Patch],
    image_size: tuple[int, int],
    min_patch_matches: int,
    min_patch_inliers: int,
    patch_pnp_max_iterations: int,
    patch_pnp_min_iterations: int,
    patch_pnp_solver: str | None,
    detector_score_weight: float = 0.0,
    global_consistency: bool = True,
    global_sparse_query_xy: np.ndarray | None = None,
    global_sparse_points_world: np.ndarray | None = None,
    global_sparse_inlier_indices: np.ndarray | None = None,
    sparse_retention_min_ratio: float = 0.70,
    sparse_retention_max_error_px: float = 8.0,
    reference_pose_w2c: np.ndarray | None = None,
    reference_max_translation_delta_m: float | None = None,
    reference_max_rotation_delta_deg: float | None = None,
    reference_match_max_reprojection_px: float = 0.0,
    reference_match_min_keep: int = 4,
    diagnostic_gt_w2c: np.ndarray | None = None,
    diagnostic_good_px: float = 5.0,
) -> tuple[list[PatchHypothesis], list[dict[str, Any]], dict[str, np.ndarray]]:
    dense_matches = _matches_from_dense(dense)
    candidate_pool = _empty_dense_match_pool()
    hypotheses: list[PatchHypothesis] = []
    rows: list[dict[str, Any]] = []
    width, height = int(image_size[0]), int(image_size[1])
    for patch in patches:
        patch_matches = filter_matches_by_patch(dense_matches, patch)
        reference_filter = None
        original_match_count = int(np.asarray(patch_matches["xy"]).shape[0])
        if reference_pose_w2c is not None and float(reference_match_max_reprojection_px) > 0.0 and original_match_count > 0:
            patch_matches, reference_filter = filter_matches_by_reference_reprojection(
                patch_matches,
                pose_w2c=reference_pose_w2c,
                intrinsic=dense["K"],
                image_size=(width, height),
                max_reprojection_error_px=float(reference_match_max_reprojection_px),
            )
        match_count = int(np.asarray(patch_matches["xy"]).shape[0])
        if reference_filter is not None and match_count < int(reference_match_min_keep):
            rows.append(
                {
                    "patch": _patch_to_dict(patch),
                    "match_mode": "dense_filtered",
                    "match_count": match_count,
                    "original_match_count": original_match_count,
                    "skipped": "too_few_reference_consistent_matches",
                    "reference_filter": reference_filter,
                }
            )
            continue
        if match_count < int(min_patch_matches):
            rows.append(
                {
                    "patch": _patch_to_dict(patch),
                    "match_mode": "dense_filtered",
                    "match_count": match_count,
                    "original_match_count": original_match_count,
                    "reference_filter": reference_filter,
                    "skipped": "too_few_matches",
                }
            )
            continue
        pose, inliers = _solve_stage_pose(
            ctx,
            patch_matches["xy"],
            patch_matches["xyz"],
            dense["K"],
            stage="dense",
            solver=patch_pnp_solver,
            max_iterations=int(patch_pnp_max_iterations),
            min_iterations=int(patch_pnp_min_iterations),
        )
        inlier_mask = _inlier_mask(match_count, inliers)
        if int(inlier_mask.sum()) < int(min_patch_inliers):
            rows.append(
                {
                    "patch": _patch_to_dict(patch),
                    "match_mode": "dense_filtered",
                    "match_count": match_count,
                    "original_match_count": original_match_count,
                    "inlier_count": int(inlier_mask.sum()),
                    "reference_filter": reference_filter,
                    "skipped": "too_few_inliers",
                }
            )
            continue
        errors = _reprojection_errors(
            query_xy=patch_matches["xy"],
            points_world=patch_matches["xyz"],
            pose_w2c=pose,
            intrinsic=dense["K"],
            width=width,
            height=height,
        )
        depths = _camera_depths(patch_matches["xyz"], pose)
        center, rotation = _camera_center_rotation(pose)
        consistency = None
        if bool(global_consistency):
            consistency = evaluate_pose_global_consistency(
                pose_w2c=pose,
                intrinsic=dense["K"],
                image_size=(width, height),
                sparse_query_xy=global_sparse_query_xy,
                sparse_points_world=global_sparse_points_world,
                sparse_inlier_indices=global_sparse_inlier_indices,
                sparse_retention_max_error_px=float(sparse_retention_max_error_px),
                sparse_retention_min_ratio=float(sparse_retention_min_ratio),
                reference_pose_w2c=reference_pose_w2c,
                reference_max_translation_delta_m=reference_max_translation_delta_m,
                reference_max_rotation_delta_deg=reference_max_rotation_delta_deg,
            )
            if str(consistency.get("decision")) != "accept_patch_pose":
                rows.append(
                    {
                        "patch": _patch_to_dict(patch),
                        "match_mode": "dense_filtered",
                        "match_count": match_count,
                        "original_match_count": original_match_count,
                        "inlier_count": int(inlier_mask.sum()),
                        "skipped": "global_consistency_reject",
                        "reference_filter": reference_filter,
                        "global_consistency": consistency,
                    }
                )
                continue
        pool_indices = _append_dense_match_subset(candidate_pool, patch_matches, inlier_mask)
        hypothesis = score_patch_hypothesis(
            patch,
            match_xy=patch_matches["xy"],
            inlier_mask=inlier_mask,
            reproj_errors=errors,
            depths=depths,
            bearings_2d=patch_matches["xy"],
            camera_center=center,
            rotation=rotation,
            match_indices=pool_indices,
            detector_scores=patch_matches.get("detector_score"),
            detector_score_weight=float(detector_score_weight),
        )
        hypotheses.append(hypothesis)
        row = _hypothesis_to_dict(hypothesis, patch, match_count=match_count)
        row["match_mode"] = "dense_filtered"
        row["original_match_count"] = original_match_count
        if reference_filter is not None:
            row["reference_filter"] = reference_filter
        if consistency is not None:
            row["global_consistency"] = consistency
        if diagnostic_gt_w2c is not None:
            gt_errors = _reprojection_errors(
                query_xy=patch_matches["xy"],
                points_world=patch_matches["xyz"],
                pose_w2c=np.asarray(diagnostic_gt_w2c, dtype=np.float64).reshape(4, 4),
                intrinsic=dense["K"],
                width=width,
                height=height,
            )
            gt_good = np.isfinite(gt_errors) & (gt_errors <= float(diagnostic_good_px))
            patch_pose = _pose_from_center_rotation(hypothesis.camera_center, hypothesis.rotation)
            patch_te, patch_re = pose_error_cm_deg(patch_pose, np.asarray(diagnostic_gt_w2c, dtype=np.float64).reshape(4, 4))
            row["diagnostic_gt_good_count"] = int(gt_good.sum())
            row["diagnostic_gt_good_ratio"] = float(gt_good.sum() / max(int(gt_good.shape[0]), 1))
            row["diagnostic_solver_inlier_gt_good_count"] = int((gt_good & inlier_mask).sum())
            row["diagnostic_solver_inlier_gt_bad_count"] = int((~gt_good & inlier_mask).sum())
            row["diagnostic_pose_te_cm"] = float(patch_te)
            row["diagnostic_pose_re_deg"] = float(patch_re)
            row["diagnostic_median_gt_reprojection_px"] = float(np.median(gt_errors[np.isfinite(gt_errors)])) if np.isfinite(gt_errors).any() else None
        rows.append(row)
    return hypotheses, rows, _finalize_dense_match_pool(candidate_pool)


def _sparse_capture_from_matches(
    sparse: Mapping[str, Any],
    merged_matches: Mapping[str, Any],
    pose_w2c: np.ndarray,
    inliers: np.ndarray,
) -> dict[str, Any]:
    return {
        "fine_map": sparse["fine_map"],
        "coarse_map": sparse["coarse_map"],
        "query_xy": np.asarray(merged_matches["xy"], dtype=np.float32).reshape(-1, 2),
        "p3d": np.asarray(merged_matches["xyz"], dtype=np.float32).reshape(-1, 3),
        "gs_ids": np.asarray(merged_matches.get("gs_ids", []), dtype=np.int64).reshape(-1),
        "scores": np.asarray(merged_matches.get("score", np.ones((len(merged_matches["xy"]),), dtype=np.float32)), dtype=np.float32).reshape(-1),
        "detector_scores": np.asarray(
            merged_matches.get("detector_score", np.full((len(merged_matches["xy"]),), 0.5, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1),
        "pose_w2c": np.asarray(pose_w2c, dtype=np.float32).reshape(4, 4),
        "inliers": np.asarray(inliers, dtype=np.int32).reshape(-1),
        "K": sparse["K"],
        "width": int(sparse["width"]),
        "height": int(sparse["height"]),
    }


def _dense_capture_from_matches(
    base_dense: Mapping[str, Any],
    merged_matches: Mapping[str, Any],
    pose_w2c: np.ndarray,
    inliers: np.ndarray,
    *,
    width: int,
    height: int,
    dense_pgsh_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rendered_xy = np.asarray(
        merged_matches.get("rendered_xy", np.zeros_like(np.asarray(merged_matches["xy"], dtype=np.float32).reshape(-1, 2))),
        dtype=np.float32,
    ).reshape(-1, 2)
    return {
        "query_xy": np.asarray(merged_matches["xy"], dtype=np.float32).reshape(-1, 2),
        "rendered_xy": rendered_xy,
        "p3d": np.asarray(merged_matches["xyz"], dtype=np.float32).reshape(-1, 3),
        "detector_scores": np.asarray(
            merged_matches.get("detector_score", np.full((len(merged_matches["xy"]),), 0.5, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1),
        "pose_w2c": np.asarray(pose_w2c, dtype=np.float32).reshape(4, 4),
        "raw_pose_w2c": np.asarray(pose_w2c, dtype=np.float32).reshape(4, 4),
        "inliers": np.asarray(inliers, dtype=np.int32).reshape(-1),
        "K": np.asarray(base_dense["K"], dtype=np.float32).reshape(3, 3),
        "render": base_dense.get("render"),
        "width": int(width),
        "height": int(height),
        "ray_depth_diagnostics": base_dense.get("ray_depth_diagnostics"),
        "slcdp_render_control": base_dense.get("slcdp_render_control"),
        "slcdp_preflight": base_dense.get("slcdp_preflight"),
        "slcdp_repair_search": base_dense.get("slcdp_repair_search"),
        "slcdp_transition_control": base_dense.get("slcdp_transition_control"),
        "dense_pgsh": dict(dense_pgsh_metadata or {}),
    }


def _draw_patch_overlay(
    query: Image.Image,
    patches: Sequence[Patch],
    selected_patch_ids: set[int],
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = query.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    row_by_patch = {int(row.get("patch", {}).get("patch_id", -1)): row for row in rows}
    for patch in patches:
        selected = int(patch.patch_id) in selected_patch_ids
        color = (0, 190, 0, 210) if selected else (255, 180, 0, 120)
        width = 4 if selected else 2
        draw.rectangle((patch.x0, patch.y0, patch.x1 - 1, patch.y1 - 1), outline=color, width=width)
        row = row_by_patch.get(int(patch.patch_id), {})
        score = row.get("score")
        inliers = row.get("inlier_count", 0)
        matches = row.get("match_count", 0)
        label = f"{patch.patch_id}: {inliers}/{matches}"
        if score is not None:
            label += f" s={float(score):.2f}"
        draw.text((patch.x0 + 4, patch.y0 + 4), label, fill=(0, 0, 0, 230))
        draw.text((patch.x0 + 3, patch.y0 + 3), label, fill=(255, 255, 255, 230))
    canvas.save(output_path, quality=92)


def _write_hypotheses_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patch_id",
        "match_mode",
        "x0",
        "y0",
        "x1",
        "y1",
        "match_count",
        "inlier_count",
        "score",
        "inlier_ratio",
        "reproj_median",
        "spatial_extent",
        "depth_spread",
        "ambiguity_risk",
        "diagnostic_gt_good_count",
        "diagnostic_gt_good_ratio",
        "diagnostic_solver_inlier_gt_good_count",
        "diagnostic_solver_inlier_gt_bad_count",
        "diagnostic_pose_te_cm",
        "diagnostic_pose_re_deg",
        "diagnostic_median_gt_reprojection_px",
        "skipped",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            patch = row.get("patch", {})
            comps = row.get("components", {})
            writer.writerow(
                {
                    "patch_id": patch.get("patch_id"),
                    "match_mode": row.get("match_mode", ""),
                    "x0": patch.get("x0"),
                    "y0": patch.get("y0"),
                    "x1": patch.get("x1"),
                    "y1": patch.get("y1"),
                    "match_count": row.get("match_count"),
                    "inlier_count": row.get("inlier_count"),
                    "score": row.get("score"),
                    "inlier_ratio": comps.get("inlier_ratio"),
                    "reproj_median": comps.get("reproj_median"),
                    "spatial_extent": comps.get("spatial_extent"),
                    "depth_spread": comps.get("depth_spread"),
                    "ambiguity_risk": comps.get("ambiguity_risk"),
                    "diagnostic_gt_good_count": row.get("diagnostic_gt_good_count"),
                    "diagnostic_gt_good_ratio": row.get("diagnostic_gt_good_ratio"),
                    "diagnostic_solver_inlier_gt_good_count": row.get("diagnostic_solver_inlier_gt_good_count"),
                    "diagnostic_solver_inlier_gt_bad_count": row.get("diagnostic_solver_inlier_gt_bad_count"),
                    "diagnostic_pose_te_cm": row.get("diagnostic_pose_te_cm"),
                    "diagnostic_pose_re_deg": row.get("diagnostic_pose_re_deg"),
                    "diagnostic_median_gt_reprojection_px": row.get("diagnostic_median_gt_reprojection_px"),
                    "skipped": row.get("skipped", ""),
                }
            )


def _summarize_sparse_pose(
    sparse_capture: Mapping[str, Any],
    *,
    gt_w2c: np.ndarray,
    good_px: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_xy, visible = project_points(
        sparse_capture["p3d"],
        sparse_capture["pose_w2c"],
        sparse_capture["K"],
        width=int(sparse_capture["width"]),
        height=int(sparse_capture["height"]),
    )
    gt_xy, gt_valid = project_points(
        sparse_capture["p3d"],
        gt_w2c,
        sparse_capture["K"],
        width=int(sparse_capture["width"]),
        height=int(sparse_capture["height"]),
    )
    errors = (
        np.linalg.norm(gt_xy - np.asarray(sparse_capture["query_xy"], dtype=np.float64), axis=1)
        if len(sparse_capture["query_xy"])
        else np.empty((0,), dtype=np.float64)
    )
    errors[~(visible & gt_valid)] = np.inf
    inlier_mask = _inlier_mask(len(errors), sparse_capture["inliers"])
    good = np.isfinite(errors) & (errors <= float(good_px))
    summary = summarize_match_quality(
        reprojection_errors_px=errors,
        solver_inlier_mask=inlier_mask,
        good_px_threshold=good_px,
    )
    return summary, ref_xy, errors, inlier_mask, good


def _summarize_dense_matches(
    dense_capture: Mapping[str, Any],
    *,
    gt_w2c: np.ndarray,
    good_px: float,
    width: int,
    height: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    gt_xy, gt_valid = project_points(
        dense_capture["p3d"],
        gt_w2c,
        dense_capture["K"],
        width=int(width),
        height=int(height),
    )
    query_xy = np.asarray(dense_capture["query_xy"], dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(gt_xy - query_xy, axis=1) if query_xy.size else np.empty((0,), dtype=np.float64)
    errors[~gt_valid] = np.inf
    inlier_mask = _inlier_mask(len(errors), dense_capture["inliers"])
    good = np.isfinite(errors) & (errors <= float(good_px))
    summary = summarize_match_quality(
        reprojection_errors_px=errors,
        solver_inlier_mask=inlier_mask,
        good_px_threshold=good_px,
    )
    return summary, errors, inlier_mask, good


def _make_contact_sheet(items: Sequence[tuple[str, Path]], output_path: Path) -> None:
    loaded = [(label, Image.open(path).convert("RGB")) for label, path in items if path.exists()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not loaded:
        Image.new("RGB", (32, 32), "white").save(output_path)
        return
    width = max(image.width for _label, image in loaded)
    height = sum(image.height + 24 for _label, image in loaded)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, image in loaded:
        draw.text((8, y + 5), label, fill=(0, 0, 0))
        y += 24
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output_path, quality=92)


def _resolve_camera(ctx: Mapping[str, Any], *, image_name: str, query_index: int | None) -> Any:
    if image_name:
        camera = ctx["camera_by_name"].get(image_name)
        if camera is None:
            raise KeyError(f"camera not found for image_name={image_name}")
        return camera
    if query_index is None:
        raise ValueError("provide --image_name or --query_index")
    cameras = list(ctx["cameras"])
    if query_index < 0 or query_index >= len(cameras):
        raise IndexError(f"query_index {query_index} is outside split camera count {len(cameras)}")
    return cameras[int(query_index)]


def analyze_case(
    *,
    candidate_root: Path,
    scene: str,
    image_name: str,
    query_index: int | None,
    split: str,
    output_dir: Path,
    patch_grids: Sequence[int],
    overlap_ratio: float,
    patch_match_mode: str,
    patch_detect_num: int,
    patch_dual_softmax: str,
    patch_mnn_match: str,
    patch_match_threshold: float | None,
    pgsh_detector_score_weight: float,
    min_patch_matches: int,
    min_patch_inliers: int,
    base_sparse_solver: str | None,
    base_sparse_max_iterations: int | None,
    base_sparse_min_iterations: int | None,
    patch_pnp_max_iterations: int,
    patch_pnp_min_iterations: int,
    patch_pnp_solver: str,
    pgsh_global_consistency: bool,
    pgsh_sparse_retention_min_ratio: float,
    pgsh_sparse_retention_max_error_px: float,
    pgsh_reference_match_max_reprojection_px: float,
    pgsh_reference_match_min_keep: int,
    cluster_center_thresh_m: float,
    cluster_rotation_thresh_deg: float,
    min_group_patches: int,
    min_final_inliers: int,
    pgsh_max_base_inliers_for_override: int,
    pgsh_fallback_to_base: bool,
    disable_sparse_pgsh: bool,
    run_dense: bool,
    dense_pgsh: bool,
    dense_pgsh_min_patch_matches: int,
    dense_pgsh_min_patch_inliers: int,
    dense_pgsh_min_group_patches: int,
    dense_pgsh_min_final_inliers: int,
    dense_pgsh_pnp_solver: str | None,
    dense_pgsh_pnp_max_iterations: int,
    dense_pgsh_pnp_min_iterations: int,
    dense_pgsh_reference_local_refine: bool,
    dense_pgsh_local_refine_max_iterations: int,
    dense_pgsh_local_refine_loss_scale_px: float,
    dense_pgsh_local_refine_translation_prior_weight: float,
    dense_pgsh_local_refine_rotation_prior_weight: float,
    dense_pgsh_sparse_anchor_weight: float,
    dense_pgsh_sparse_anchor_max_count: int,
    dense_pgsh_sparse_anchor_max_reprojection_px: float,
    dense_pgsh_max_translation_from_base_dense_m: float,
    dense_pgsh_max_rotation_from_base_dense_deg: float,
    good_px: float,
    max_draw: int,
    slcdp_options: Mapping[str, Any],
) -> dict[str, Any]:
    ctx = _build_context(candidate_root / scene, split_override=split)
    camera = _resolve_camera(ctx, image_name=image_name, query_index=query_index)
    query_image = camera.original_image.to("cuda")
    gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    with torch.no_grad():
        base_sparse = _capture_sparse_with_overrides(
            ctx,
            query_image,
            camera.FoVx,
            camera.FoVy,
            solver=base_sparse_solver,
            max_iterations=base_sparse_max_iterations,
            min_iterations=base_sparse_min_iterations,
        )
        base_sparse["detector_scores"] = _scene_detector_scores_for_xy(ctx, base_sparse["fine_map"], base_sparse["query_xy"])

    patches = build_multiscale_patches(
        image_size=(int(base_sparse["width"]), int(base_sparse["height"])),
        grids=patch_grids,
        overlap_ratio=float(overlap_ratio),
    )
    base_sparse_inlier_count = int(np.asarray(base_sparse.get("inliers", [])).reshape(-1).shape[0])
    skip_pgsh_search = (
        bool(disable_sparse_pgsh)
        or (
        bool(pgsh_fallback_to_base)
        and int(pgsh_max_base_inliers_for_override) > 0
        and base_sparse_inlier_count >= int(pgsh_max_base_inliers_for_override)
        )
    )
    if skip_pgsh_search:
        hypotheses, hypothesis_rows, candidate_matches = [], [], _empty_matches()
    else:
        hypotheses, hypothesis_rows, candidate_matches = build_patch_guided_sparse_hypotheses(
            ctx,
            base_sparse,
            patches=patches,
            patch_match_mode=str(patch_match_mode),
            patch_detect_num=int(patch_detect_num),
            patch_dual_softmax=str(patch_dual_softmax),
            patch_mnn_match=str(patch_mnn_match),
            patch_match_threshold=patch_match_threshold,
            min_patch_matches=int(min_patch_matches),
            min_patch_inliers=int(min_patch_inliers),
            patch_pnp_max_iterations=int(patch_pnp_max_iterations),
            patch_pnp_min_iterations=int(patch_pnp_min_iterations),
            patch_pnp_solver=str(patch_pnp_solver),
            detector_score_weight=float(pgsh_detector_score_weight),
            global_consistency=bool(pgsh_global_consistency),
            global_sparse_query_xy=base_sparse["query_xy"],
            global_sparse_points_world=base_sparse["p3d"],
            global_sparse_inlier_indices=base_sparse["inliers"],
            sparse_retention_min_ratio=float(pgsh_sparse_retention_min_ratio),
            sparse_retention_max_error_px=float(pgsh_sparse_retention_max_error_px),
            reference_match_max_reprojection_px=float(pgsh_reference_match_max_reprojection_px),
            reference_match_min_keep=int(pgsh_reference_match_min_keep),
            diagnostic_gt_w2c=gt_w2c,
            diagnostic_good_px=float(good_px),
        )
    group = select_pose_consistent_patch_group(
        hypotheses,
        center_thresh=float(cluster_center_thresh_m),
        rotation_thresh_deg=float(cluster_rotation_thresh_deg),
        min_patch_count=int(min_group_patches),
    )
    merged = merge_group_matches(candidate_matches, group)
    selected_merged_match_count = int(np.asarray(merged.get("xy", [])).shape[0])
    if bool(disable_sparse_pgsh):
        pgsh_decision = "sparse_pgsh_disabled"
    elif skip_pgsh_search:
        pgsh_decision = "fallback_base_strong_base_sparse"
    else:
        pgsh_decision = "use_patch_guided_sparse"
    candidate_pgsh_pose = np.asarray(base_sparse["pose_w2c"], dtype=np.float32).reshape(4, 4)
    candidate_pgsh_inliers = np.empty((0,), dtype=np.int32)
    if pgsh_decision == "use_patch_guided_sparse" and int(np.asarray(merged.get("xy", [])).shape[0]) >= 4:
        candidate_pgsh_pose, candidate_pgsh_inliers = _solve_sparse_pose(
            ctx,
            merged["xy"],
            merged["xyz"],
            base_sparse["K"],
            solver=str(patch_pnp_solver),
            max_iterations=int(max(patch_pnp_max_iterations, 20000)),
            min_iterations=int(max(patch_pnp_min_iterations, 100)),
        )
    elif pgsh_decision == "use_patch_guided_sparse":
        pgsh_decision = "fallback_base_no_pose_consistent_patch_group"
    final_inlier_count = int(np.asarray(candidate_pgsh_inliers).reshape(-1).shape[0])
    if pgsh_decision == "use_patch_guided_sparse" and final_inlier_count < int(min_final_inliers):
        pgsh_decision = "fallback_base_too_few_final_inliers"
    if pgsh_decision != "use_patch_guided_sparse" and (bool(pgsh_fallback_to_base) or bool(disable_sparse_pgsh)):
        pgsh_sparse = dict(base_sparse)
        pgsh_sparse["pgsh_fallback"] = True
        pgsh_sparse["pgsh_decision"] = pgsh_decision
        merged = _matches_from_sparse(base_sparse)
        candidate_pgsh_sparse = _sparse_capture_from_matches(base_sparse, merged, candidate_pgsh_pose, candidate_pgsh_inliers)
    else:
        pgsh_sparse = _sparse_capture_from_matches(base_sparse, merged, candidate_pgsh_pose, candidate_pgsh_inliers)
        pgsh_sparse["pgsh_fallback"] = False
        pgsh_sparse["pgsh_decision"] = pgsh_decision
        candidate_pgsh_sparse = dict(pgsh_sparse)

    case_name = f"{scene}_{int(query_index if query_index is not None else 0):05d}_{Path(str(camera.image_name)).stem}"
    case_dir = output_dir / scene / case_name
    query_pil = _tensor_to_image(query_image, size=(int(base_sparse["width"]), int(base_sparse["height"])))
    base_render_pkg = ctx["stdloc"].render_from_pose_gsplat(
        ctx["localizer"].gaussians,
        torch.tensor(base_sparse["pose_w2c"], device="cuda"),
        camera.FoVx,
        camera.FoVy,
        int(base_sparse["width"]),
        int(base_sparse["height"]),
        render_mode="RGB+ED",
        rgb_only=True,
        norm_feat_bf_render=ctx["config"]["dense"]["norm_before_render"],
        rasterize_mode="antialiased",
    )
    pgsh_render_pkg = ctx["stdloc"].render_from_pose_gsplat(
        ctx["localizer"].gaussians,
        torch.tensor(pgsh_sparse["pose_w2c"], device="cuda"),
        camera.FoVx,
        camera.FoVy,
        int(base_sparse["width"]),
        int(base_sparse["height"]),
        render_mode="RGB+ED",
        rgb_only=True,
        norm_feat_bf_render=ctx["config"]["dense"]["norm_before_render"],
        rasterize_mode="antialiased",
    )
    base_render = _tensor_to_image(base_render_pkg["render"], size=query_pil.size)
    pgsh_render = _tensor_to_image(pgsh_render_pkg["render"], size=query_pil.size)

    base_summary, base_ref_xy, base_errors, base_inlier_mask, base_good = _summarize_sparse_pose(
        base_sparse,
        gt_w2c=gt_w2c,
        good_px=float(good_px),
    )
    pgsh_summary, pgsh_ref_xy, pgsh_errors, pgsh_inlier_mask, pgsh_good = _summarize_sparse_pose(
        pgsh_sparse,
        gt_w2c=gt_w2c,
        good_px=float(good_px),
    )
    base_sparse_te, base_sparse_re = pose_error_cm_deg(base_sparse["pose_w2c"], gt_w2c)
    pgsh_sparse_te, pgsh_sparse_re = pose_error_cm_deg(pgsh_sparse["pose_w2c"], gt_w2c)
    candidate_pgsh_sparse_te, candidate_pgsh_sparse_re = pose_error_cm_deg(candidate_pgsh_sparse["pose_w2c"], gt_w2c)

    base_sparse_path = case_dir / "base_sparse_matches.jpg"
    pgsh_sparse_path = case_dir / "pgsh_sparse_matches.jpg"
    patch_overlay_path = case_dir / "pgsh_patch_overlay.jpg"
    draw_match_canvas(
        query_pil,
        base_render,
        query_xy=base_sparse["query_xy"],
        reference_xy=base_ref_xy,
        gt_good_mask=base_good,
        solver_inlier_mask=base_inlier_mask,
        output_path=base_sparse_path,
        max_draw=int(max_draw),
    )
    draw_match_canvas(
        query_pil,
        pgsh_render,
        query_xy=pgsh_sparse["query_xy"],
        reference_xy=pgsh_ref_xy,
        gt_good_mask=pgsh_good,
        solver_inlier_mask=pgsh_inlier_mask,
        output_path=pgsh_sparse_path,
        max_draw=int(max_draw),
    )
    _draw_patch_overlay(query_pil, patches, set(group.patch_ids), hypothesis_rows, patch_overlay_path)
    _write_matches_csv(case_dir / "base_sparse_matches.csv", base_sparse["query_xy"], base_ref_xy, base_errors, base_inlier_mask, base_good)
    _write_matches_csv(case_dir / "pgsh_sparse_matches.csv", pgsh_sparse["query_xy"], pgsh_ref_xy, pgsh_errors, pgsh_inlier_mask, pgsh_good)
    _write_hypotheses_csv(case_dir / "patch_hypotheses.csv", hypothesis_rows)

    dense_summary: dict[str, Any] = {}
    sheet_items: list[tuple[str, Path]] = [
        ("patch overlay", patch_overlay_path),
        ("base sparse", base_sparse_path),
        ("PGSH sparse", pgsh_sparse_path),
    ]
    if bool(run_dense):
        with torch.no_grad():
            base_dense = _capture_dense(
                ctx,
                base_sparse["coarse_map"],
                base_sparse["fine_map"],
                base_sparse["pose_w2c"],
                camera.FoVx,
                camera.FoVy,
                sparse_capture=base_sparse,
                **dict(slcdp_options),
            )
            if pgsh_sparse.get("pgsh_decision") == "sparse_pgsh_disabled" or np.allclose(
                np.asarray(pgsh_sparse["pose_w2c"], dtype=np.float32),
                np.asarray(base_sparse["pose_w2c"], dtype=np.float32),
            ):
                pgsh_dense = dict(base_dense)
            else:
                pgsh_dense = _capture_dense(
                    ctx,
                    pgsh_sparse["coarse_map"],
                    pgsh_sparse["fine_map"],
                    pgsh_sparse["pose_w2c"],
                    camera.FoVx,
                    camera.FoVy,
                    sparse_capture=pgsh_sparse,
                    **dict(slcdp_options),
                )
        base_dense = dict(base_dense)
        pgsh_dense = dict(pgsh_dense)
        base_dense["width"] = int(base_sparse["width"])
        base_dense["height"] = int(base_sparse["height"])
        pgsh_dense["width"] = int(base_sparse["width"])
        pgsh_dense["height"] = int(base_sparse["height"])
        base_dense["detector_scores"] = _scene_detector_scores_for_xy(ctx, base_sparse["fine_map"], base_dense["query_xy"])
        pgsh_dense["detector_scores"] = _scene_detector_scores_for_xy(ctx, base_sparse["fine_map"], pgsh_dense["query_xy"])
        base_dense_te, base_dense_re = pose_error_cm_deg(base_dense["pose_w2c"], gt_w2c)
        pgsh_dense_te, pgsh_dense_re = pose_error_cm_deg(pgsh_dense["pose_w2c"], gt_w2c)
        base_dense_raw_te, base_dense_raw_re = pose_error_cm_deg(base_dense.get("raw_pose_w2c", base_dense["pose_w2c"]), gt_w2c)
        pgsh_dense_raw_te, pgsh_dense_raw_re = pose_error_cm_deg(pgsh_dense.get("raw_pose_w2c", pgsh_dense["pose_w2c"]), gt_w2c)

        base_dense_summary, base_dense_errors, base_dense_inlier_mask, base_dense_good = _summarize_dense_matches(
            base_dense,
            gt_w2c=gt_w2c,
            good_px=float(good_px),
            width=int(base_sparse["width"]),
            height=int(base_sparse["height"]),
        )
        pgsh_dense_summary, pgsh_dense_errors, pgsh_dense_inlier_mask, pgsh_dense_good = _summarize_dense_matches(
            pgsh_dense,
            gt_w2c=gt_w2c,
            good_px=float(good_px),
            width=int(base_sparse["width"]),
            height=int(base_sparse["height"]),
        )
        base_dense_path = case_dir / "base_dense_matches.jpg"
        pgsh_dense_path = case_dir / "pgsh_sparse_then_dense_matches.jpg"
        draw_match_canvas(
            query_pil,
            _tensor_to_image(base_dense["render"], size=query_pil.size),
            query_xy=base_dense["query_xy"],
            reference_xy=base_dense.get("rendered_xy", np.empty((0, 2), dtype=np.float32)),
            gt_good_mask=base_dense_good,
            solver_inlier_mask=base_dense_inlier_mask,
            output_path=base_dense_path,
            max_draw=int(max_draw),
        )
        draw_match_canvas(
            query_pil,
            _tensor_to_image(pgsh_dense["render"], size=query_pil.size),
            query_xy=pgsh_dense["query_xy"],
            reference_xy=pgsh_dense.get("rendered_xy", np.empty((0, 2), dtype=np.float32)),
            gt_good_mask=pgsh_dense_good,
            solver_inlier_mask=pgsh_dense_inlier_mask,
            output_path=pgsh_dense_path,
            max_draw=int(max_draw),
        )
        _write_matches_csv(
            case_dir / "base_dense_matches.csv",
            base_dense["query_xy"],
            base_dense.get("rendered_xy", np.empty((0, 2), dtype=np.float32)),
            base_dense_errors,
            base_dense_inlier_mask,
            base_dense_good,
        )
        _write_matches_csv(
            case_dir / "pgsh_sparse_then_dense_matches.csv",
            pgsh_dense["query_xy"],
            pgsh_dense.get("rendered_xy", np.empty((0, 2), dtype=np.float32)),
            pgsh_dense_errors,
            pgsh_dense_inlier_mask,
            pgsh_dense_good,
        )
        sheet_items.extend(
            [
                ("base sparse-conditioned dense", base_dense_path),
                ("PGSH sparse then dense", pgsh_dense_path),
            ]
        )

        dense_pgsh_summary: dict[str, Any] = {"enabled": bool(dense_pgsh)}
        if bool(dense_pgsh):
            dense_hypotheses, dense_rows, dense_candidate_matches = build_patch_guided_dense_hypotheses(
                ctx,
                base_dense,
                patches=patches,
                image_size=(int(base_sparse["width"]), int(base_sparse["height"])),
                min_patch_matches=int(dense_pgsh_min_patch_matches),
                min_patch_inliers=int(dense_pgsh_min_patch_inliers),
                patch_pnp_max_iterations=int(dense_pgsh_pnp_max_iterations),
                patch_pnp_min_iterations=int(dense_pgsh_pnp_min_iterations),
                patch_pnp_solver=dense_pgsh_pnp_solver,
                detector_score_weight=float(pgsh_detector_score_weight),
                global_consistency=bool(pgsh_global_consistency),
                global_sparse_query_xy=base_sparse["query_xy"],
                global_sparse_points_world=base_sparse["p3d"],
                global_sparse_inlier_indices=base_sparse["inliers"],
                sparse_retention_min_ratio=float(pgsh_sparse_retention_min_ratio),
                sparse_retention_max_error_px=float(pgsh_sparse_retention_max_error_px),
                reference_pose_w2c=base_dense["pose_w2c"],
                reference_max_translation_delta_m=float(dense_pgsh_max_translation_from_base_dense_m),
                reference_max_rotation_delta_deg=float(dense_pgsh_max_rotation_from_base_dense_deg),
                reference_match_max_reprojection_px=float(pgsh_reference_match_max_reprojection_px),
                reference_match_min_keep=int(pgsh_reference_match_min_keep),
                diagnostic_gt_w2c=gt_w2c,
                diagnostic_good_px=float(good_px),
            )
            dense_group = select_pose_consistent_patch_group(
                dense_hypotheses,
                center_thresh=float(cluster_center_thresh_m),
                rotation_thresh_deg=float(cluster_rotation_thresh_deg),
                min_patch_count=int(dense_pgsh_min_group_patches),
            )
            dense_merged = merge_group_matches(dense_candidate_matches, dense_group)
            dense_pgsh_decision = "use_dense_pgsh"
            dense_pgsh_raw_pose = np.asarray(base_dense["pose_w2c"], dtype=np.float32).reshape(4, 4)
            dense_pgsh_inliers = np.empty((0,), dtype=np.int32)
            dense_pgsh_local_refine = None
            dense_pgsh_refinement_anchor = None
            if int(np.asarray(dense_merged.get("xy", [])).shape[0]) >= 4:
                if bool(dense_pgsh_reference_local_refine):
                    refinement_matches = _build_dense_pgsh_refinement_matches(
                        dense_merged,
                        base_sparse,
                        reference_pose_w2c=base_dense["pose_w2c"],
                        intrinsic=base_dense["K"],
                        width=int(base_sparse["width"]),
                        height=int(base_sparse["height"]),
                        sparse_anchor_weight=float(dense_pgsh_sparse_anchor_weight),
                        sparse_anchor_max_count=int(dense_pgsh_sparse_anchor_max_count),
                        sparse_anchor_max_reprojection_px=float(dense_pgsh_sparse_anchor_max_reprojection_px),
                    )
                    dense_pgsh_refinement_anchor = refinement_matches["diagnostics"]
                    dense_pgsh_raw_pose, dense_pgsh_local_refine = refine_pose_with_reference_prior(
                        reference_pose_w2c=base_dense["pose_w2c"],
                        match_xy=refinement_matches["xy"],
                        points_world=refinement_matches["xyz"],
                        intrinsic=base_dense["K"],
                        image_size=(int(base_sparse["width"]), int(base_sparse["height"])),
                        max_iterations=int(dense_pgsh_local_refine_max_iterations),
                        reprojection_loss_scale_px=float(dense_pgsh_local_refine_loss_scale_px),
                        translation_prior_weight=float(dense_pgsh_local_refine_translation_prior_weight),
                        rotation_prior_weight=float(dense_pgsh_local_refine_rotation_prior_weight),
                        max_translation_delta_m=float(dense_pgsh_max_translation_from_base_dense_m),
                        max_rotation_delta_deg=float(dense_pgsh_max_rotation_from_base_dense_deg),
                        match_weights=refinement_matches["weights"],
                    )
                    dense_errors_for_inliers = _reprojection_errors(
                        query_xy=dense_merged["xy"],
                        points_world=dense_merged["xyz"],
                        pose_w2c=dense_pgsh_raw_pose,
                        intrinsic=base_dense["K"],
                        width=int(base_sparse["width"]),
                        height=int(base_sparse["height"]),
                    )
                    dense_pgsh_inliers = np.flatnonzero(np.isfinite(dense_errors_for_inliers) & (dense_errors_for_inliers <= 8.0)).astype(np.int32)
                else:
                    dense_pgsh_raw_pose, dense_pgsh_inliers = _solve_stage_pose(
                        ctx,
                        dense_merged["xy"],
                        dense_merged["xyz"],
                        base_dense["K"],
                        stage="dense",
                        solver=dense_pgsh_pnp_solver,
                        max_iterations=int(max(dense_pgsh_pnp_max_iterations, 20000)),
                        min_iterations=int(max(dense_pgsh_pnp_min_iterations, 100)),
                    )
            else:
                dense_pgsh_decision = "fallback_base_dense_no_pose_consistent_patch_group"
            if dense_pgsh_decision == "use_dense_pgsh" and int(np.asarray(dense_pgsh_inliers).reshape(-1).shape[0]) < int(dense_pgsh_min_final_inliers):
                dense_pgsh_decision = "fallback_base_dense_too_few_final_inliers"

            dense_pgsh_transition = None
            dense_pgsh_base_trust = None
            dense_pgsh_final_pose = np.asarray(dense_pgsh_raw_pose, dtype=np.float32).reshape(4, 4)
            if dense_pgsh_decision == "use_dense_pgsh" and bool(dict(slcdp_options).get("slcdp_transition_control", False)):
                dense_pgsh_transition = select_sparse_conditioned_dense_transition(
                    sparse_query_xy=base_sparse["query_xy"],
                    sparse_points_world=base_sparse["p3d"],
                    sparse_pose_w2c=base_sparse["pose_w2c"],
                    dense_pose_w2c=dense_pgsh_raw_pose,
                    intrinsic=base_dense["K"],
                    sparse_inlier_indices=base_sparse["inliers"],
                    policy=DenseTransitionPolicy(
                        max_reprojection_error_px=float(dict(slcdp_options).get("slcdp_transition_max_reprojection_error_px", 8.0)),
                        min_retained_ratio=float(dict(slcdp_options).get("slcdp_transition_min_retained_ratio", 0.90)),
                        max_translation_delta_m=float(dict(slcdp_options).get("slcdp_transition_max_translation_delta_m", 0.35)),
                        max_rotation_delta_deg=float(dict(slcdp_options).get("slcdp_transition_max_rotation_delta_deg", 5.0)),
                        line_search_fractions=tuple(float(v) for v in dict(slcdp_options).get("slcdp_transition_line_search_fractions", (1.0, 0.75, 0.5, 0.25, 0.0))),
                    ),
                    image_size=(int(base_sparse["width"]), int(base_sparse["height"])),
                )
                if str(dense_pgsh_transition.get("decision")) == "reject_dense_keep_sparse":
                    dense_pgsh_decision = "fallback_base_dense_transition_reject"
                else:
                    dense_pgsh_final_pose = np.asarray(dense_pgsh_transition["selected_pose_w2c"], dtype=np.float32).reshape(4, 4)

            if dense_pgsh_decision == "use_dense_pgsh":
                dense_pgsh_base_trust = _dense_pgsh_base_trust_acceptance(
                    base_dense_pose_w2c=base_dense["pose_w2c"],
                    candidate_pose_w2c=dense_pgsh_final_pose,
                    max_translation_delta_m=float(dense_pgsh_max_translation_from_base_dense_m),
                    max_rotation_delta_deg=float(dense_pgsh_max_rotation_from_base_dense_deg),
                )
                if str(dense_pgsh_base_trust.get("decision")) != "accept_dense_pgsh_pose":
                    dense_pgsh_decision = "fallback_base_dense_base_trust_reject"

            if dense_pgsh_decision == "use_dense_pgsh":
                dense_pgsh_capture = _dense_capture_from_matches(
                    base_dense,
                    dense_merged,
                    dense_pgsh_final_pose,
                    dense_pgsh_inliers,
                    width=int(base_sparse["width"]),
                    height=int(base_sparse["height"]),
                    dense_pgsh_metadata={
                        "decision": dense_pgsh_decision,
                        "selected_group": {
                            "patch_ids": [int(v) for v in dense_group.patch_ids],
                            "hypothesis_indices": [int(v) for v in dense_group.hypothesis_indices],
                            "score": float(dense_group.score),
                            "inlier_count": int(dense_group.inlier_count),
                            "merged_match_count": int(np.asarray(dense_merged.get("xy", [])).shape[0]),
                            "candidate_match_count": int(np.asarray(dense_candidate_matches.get("xy", [])).shape[0]),
                        },
                        "transition_control": dense_pgsh_transition,
                        "base_trust": dense_pgsh_base_trust,
                        "reference_local_refine": dense_pgsh_local_refine,
                        "refinement_sparse_anchor": dense_pgsh_refinement_anchor,
                    },
                )
            else:
                dense_pgsh_capture = dict(base_dense)
                dense_pgsh_capture["dense_pgsh"] = {
                    "decision": dense_pgsh_decision,
                    "fallback_to_base_dense": True,
                    "transition_control": dense_pgsh_transition,
                    "base_trust": dense_pgsh_base_trust,
                    "reference_local_refine": dense_pgsh_local_refine,
                    "refinement_sparse_anchor": dense_pgsh_refinement_anchor,
                }

            dense_pgsh_te, dense_pgsh_re = pose_error_cm_deg(dense_pgsh_capture["pose_w2c"], gt_w2c)
            dense_pgsh_raw_te, dense_pgsh_raw_re = pose_error_cm_deg(dense_pgsh_raw_pose, gt_w2c)
            dense_pgsh_match_summary, dense_pgsh_errors, dense_pgsh_inlier_mask, dense_pgsh_good = _summarize_dense_matches(
                dense_pgsh_capture,
                gt_w2c=gt_w2c,
                good_px=float(good_px),
                width=int(base_sparse["width"]),
                height=int(base_sparse["height"]),
            )
            dense_pgsh_path = case_dir / "dense_pgsh_matches.jpg"
            dense_patch_overlay_path = case_dir / "dense_pgsh_patch_overlay.jpg"
            draw_match_canvas(
                query_pil,
                _tensor_to_image(dense_pgsh_capture["render"], size=query_pil.size),
                query_xy=dense_pgsh_capture["query_xy"],
                reference_xy=dense_pgsh_capture.get("rendered_xy", np.empty((0, 2), dtype=np.float32)),
                gt_good_mask=dense_pgsh_good,
                solver_inlier_mask=dense_pgsh_inlier_mask,
                output_path=dense_pgsh_path,
                max_draw=int(max_draw),
            )
            _draw_patch_overlay(query_pil, patches, set(dense_group.patch_ids), dense_rows, dense_patch_overlay_path)
            _write_matches_csv(
                case_dir / "dense_pgsh_matches.csv",
                dense_pgsh_capture["query_xy"],
                dense_pgsh_capture.get("rendered_xy", np.empty((0, 2), dtype=np.float32)),
                dense_pgsh_errors,
                dense_pgsh_inlier_mask,
                dense_pgsh_good,
            )
            _write_hypotheses_csv(case_dir / "dense_patch_hypotheses.csv", dense_rows)
            sheet_items.extend(
                [
                    ("dense PGSH patch overlay", dense_patch_overlay_path),
                    ("dense-only PGSH", dense_pgsh_path),
                ]
            )
            dense_pgsh_summary = {
                "enabled": True,
                "decision": dense_pgsh_decision,
                "dense_pgsh_te_cm": dense_pgsh_te,
                "dense_pgsh_re_deg": dense_pgsh_re,
                "dense_pgsh_raw_te_cm": dense_pgsh_raw_te,
                "dense_pgsh_raw_re_deg": dense_pgsh_raw_re,
                "dense_pgsh_inliers": int(np.asarray(dense_pgsh_capture.get("inliers", [])).reshape(-1).shape[0]),
                "selected_group": {
                    "patch_ids": [int(v) for v in dense_group.patch_ids],
                    "hypothesis_indices": [int(v) for v in dense_group.hypothesis_indices],
                    "score": float(dense_group.score),
                    "inlier_count": int(dense_group.inlier_count),
                    "merged_match_count": int(np.asarray(dense_merged.get("xy", [])).shape[0]),
                    "candidate_match_count": int(np.asarray(dense_candidate_matches.get("xy", [])).shape[0]),
                },
                "match_quality": dense_pgsh_match_summary,
                "transition_control": dense_pgsh_transition,
                "base_trust": dense_pgsh_base_trust,
                "reference_local_refine": dense_pgsh_local_refine,
                "refinement_sparse_anchor": dense_pgsh_refinement_anchor,
                "paths": {
                    "matches": str(dense_pgsh_path),
                    "patch_overlay": str(dense_patch_overlay_path),
                    "patch_hypotheses_csv": str(case_dir / "dense_patch_hypotheses.csv"),
                    "matches_csv": str(case_dir / "dense_pgsh_matches.csv"),
                },
                "hypotheses": dense_rows,
            }
        dense_summary = {
            "base_dense_te_cm": base_dense_te,
            "base_dense_re_deg": base_dense_re,
            "base_dense_raw_te_cm": base_dense_raw_te,
            "base_dense_raw_re_deg": base_dense_raw_re,
            "pgsh_dense_te_cm": pgsh_dense_te,
            "pgsh_dense_re_deg": pgsh_dense_re,
            "pgsh_dense_raw_te_cm": pgsh_dense_raw_te,
            "pgsh_dense_raw_re_deg": pgsh_dense_raw_re,
            "base_dense_inliers": int(np.asarray(base_dense.get("inliers", [])).reshape(-1).shape[0]),
            "pgsh_dense_inliers": int(np.asarray(pgsh_dense.get("inliers", [])).reshape(-1).shape[0]),
            "base_dense_match_quality": base_dense_summary,
            "pgsh_dense_match_quality": pgsh_dense_summary,
            "base_slcdp": base_dense.get("slcdp_repair_search"),
            "pgsh_slcdp": pgsh_dense.get("slcdp_repair_search"),
            "dense_pgsh": dense_pgsh_summary,
            **_dense_pgsh_report_aliases(dense_pgsh_summary),
            "paths": {
                "base_dense_matches": str(base_dense_path),
                "base_dense_matches_csv": str(case_dir / "base_dense_matches.csv"),
                "pgsh_sparse_then_dense_matches": str(pgsh_dense_path),
                "pgsh_sparse_then_dense_matches_csv": str(case_dir / "pgsh_sparse_then_dense_matches.csv"),
            },
        }
    sheet_path = case_dir / "pgsh_contact_sheet.jpg"
    _make_contact_sheet(sheet_items, sheet_path)

    summary = {
        "schema": "loc_gs_patch_guided_sparse_diagnostic_v1",
        "diagnostic_only": True,
        "paper_safe_for_tuning": False,
        "uses_gt_pose_for_visual_diagnostic": True,
        "selection_uses_gt": False,
        "scene": scene,
        "split": split,
        "image_name": str(camera.image_name),
        "query_index": int(query_index) if query_index is not None else None,
        "patch_config": {
            "patch_grids": [int(v) for v in patch_grids],
            "overlap_ratio": float(overlap_ratio),
            "patch_match_mode": str(patch_match_mode),
            "patch_detect_num": int(patch_detect_num),
            "patch_dual_softmax": str(patch_dual_softmax),
            "patch_mnn_match": str(patch_mnn_match),
            "patch_match_threshold": patch_match_threshold,
            "pgsh_detector_score_weight": float(pgsh_detector_score_weight),
            "min_patch_matches": int(min_patch_matches),
            "min_patch_inliers": int(min_patch_inliers),
            "base_sparse_solver": str(base_sparse_solver) if base_sparse_solver else "native_config",
            "base_sparse_max_iterations": base_sparse_max_iterations,
            "base_sparse_min_iterations": base_sparse_min_iterations,
            "patch_pnp_max_iterations": int(patch_pnp_max_iterations),
            "patch_pnp_min_iterations": int(patch_pnp_min_iterations),
            "patch_pnp_solver": str(patch_pnp_solver),
            "pgsh_global_consistency": bool(pgsh_global_consistency),
            "pgsh_sparse_retention_min_ratio": float(pgsh_sparse_retention_min_ratio),
            "pgsh_sparse_retention_max_error_px": float(pgsh_sparse_retention_max_error_px),
            "pgsh_reference_match_max_reprojection_px": float(pgsh_reference_match_max_reprojection_px),
            "pgsh_reference_match_min_keep": int(pgsh_reference_match_min_keep),
            "cluster_center_thresh_m": float(cluster_center_thresh_m),
            "cluster_rotation_thresh_deg": float(cluster_rotation_thresh_deg),
            "min_group_patches": int(min_group_patches),
            "min_final_inliers": int(min_final_inliers),
            "pgsh_max_base_inliers_for_override": int(pgsh_max_base_inliers_for_override),
            "pgsh_fallback_to_base": bool(pgsh_fallback_to_base),
            "disable_sparse_pgsh": bool(disable_sparse_pgsh),
            "dense_pgsh": bool(dense_pgsh),
            "dense_pgsh_min_patch_matches": int(dense_pgsh_min_patch_matches),
            "dense_pgsh_min_patch_inliers": int(dense_pgsh_min_patch_inliers),
            "dense_pgsh_min_group_patches": int(dense_pgsh_min_group_patches),
            "dense_pgsh_min_final_inliers": int(dense_pgsh_min_final_inliers),
            "dense_pgsh_pnp_solver": str(dense_pgsh_pnp_solver) if dense_pgsh_pnp_solver else "native_dense_config",
            "dense_pgsh_pnp_max_iterations": int(dense_pgsh_pnp_max_iterations),
            "dense_pgsh_pnp_min_iterations": int(dense_pgsh_pnp_min_iterations),
            "dense_pgsh_reference_local_refine": bool(dense_pgsh_reference_local_refine),
            "dense_pgsh_local_refine_max_iterations": int(dense_pgsh_local_refine_max_iterations),
            "dense_pgsh_local_refine_loss_scale_px": float(dense_pgsh_local_refine_loss_scale_px),
            "dense_pgsh_local_refine_translation_prior_weight": float(dense_pgsh_local_refine_translation_prior_weight),
            "dense_pgsh_local_refine_rotation_prior_weight": float(dense_pgsh_local_refine_rotation_prior_weight),
            "dense_pgsh_sparse_anchor_weight": float(dense_pgsh_sparse_anchor_weight),
            "dense_pgsh_sparse_anchor_max_count": int(dense_pgsh_sparse_anchor_max_count),
            "dense_pgsh_sparse_anchor_max_reprojection_px": float(dense_pgsh_sparse_anchor_max_reprojection_px),
            "dense_pgsh_max_translation_from_base_dense_m": float(dense_pgsh_max_translation_from_base_dense_m),
            "dense_pgsh_max_rotation_from_base_dense_deg": float(dense_pgsh_max_rotation_from_base_dense_deg),
        },
        "base_sparse_te_cm": base_sparse_te,
        "base_sparse_re_deg": base_sparse_re,
        "pgsh_sparse_te_cm": pgsh_sparse_te,
        "pgsh_sparse_re_deg": pgsh_sparse_re,
        "candidate_pgsh_sparse_te_cm": candidate_pgsh_sparse_te,
        "candidate_pgsh_sparse_re_deg": candidate_pgsh_sparse_re,
        "pgsh_decision": pgsh_decision,
        "pgsh_fallback": bool(pgsh_sparse.get("pgsh_fallback", False)),
        "base_sparse_inlier_count_for_pgsh_gate": int(base_sparse_inlier_count),
        "base_sparse": base_summary,
        "pgsh_sparse": pgsh_summary,
        "selected_group": {
            "patch_ids": [int(v) for v in group.patch_ids],
            "hypothesis_indices": [int(v) for v in group.hypothesis_indices],
            "score": float(group.score),
            "inlier_count": int(group.inlier_count),
            "merged_match_count": int(selected_merged_match_count),
            "candidate_match_count": int(np.asarray(candidate_matches.get("xy", [])).shape[0]),
        },
        "hypotheses": hypothesis_rows,
        "dense": dense_summary,
        "paths": {
            "contact_sheet": str(sheet_path),
            "patch_overlay": str(patch_overlay_path),
            "base_sparse_matches": str(base_sparse_path),
            "pgsh_sparse_matches": str(pgsh_sparse_path),
            "patch_hypotheses_csv": str(case_dir / "patch_hypotheses.csv"),
        },
    }
    safe_summary = _json_safe(summary)
    write_json(case_dir / "pgsh_summary.json", safe_summary)
    write_artifact_audit_bundle(
        output_dir,
        manifest=_json_safe({
            "method": "patch_guided_sparse_hypothesis_generation",
            "diagnostic_only": True,
            "paper_safe_for_tuning": False,
            "candidate_root": str(candidate_root),
            "scene": scene,
            "split_name": split,
            "image_name": str(camera.image_name),
            "output_dir": str(output_dir),
            "patch_config": summary["patch_config"],
            "slcdp_options": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in dict(slcdp_options).items()
            },
        }),
        command=_command(),
        metrics_summary=_json_safe({
            "base_sparse_te_cm": base_sparse_te,
            "pgsh_sparse_te_cm": pgsh_sparse_te,
            **dense_summary,
        }),
        split_audit={
            "audit_status": "failed" if split == "test" else "unknown",
            "reason": "PGSH hard-case diagnostic may inspect official test visualization; not valid for tuning or method selection",
        },
    )
    print(json.dumps({"summary": str(case_dir / "pgsh_summary.json"), "contact_sheet": str(sheet_path)}, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch-guided sparse hypothesis diagnostic for occluded Cambridge queries.")
    parser.add_argument("--candidate_root", default="output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected")
    parser.add_argument("--scene", default="KingsCollege")
    parser.add_argument("--image_name", default="")
    parser.add_argument("--query_index", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", default="output/diagnostics/patch_guided_sparse/kings33_pgsh")
    parser.add_argument("--patch_grids", default="2,3")
    parser.add_argument("--patch_overlap_ratio", type=float, default=0.50)
    parser.add_argument("--patch_match_mode", default="local", choices=["local", "filtered"])
    parser.add_argument("--patch_detect_num", type=int, default=512)
    parser.add_argument("--patch_dual_softmax", default="native", choices=["native", "on", "off"])
    parser.add_argument("--patch_mnn_match", default="native", choices=["native", "on", "off"])
    parser.add_argument("--patch_match_threshold", type=float, default=None)
    parser.add_argument("--pgsh_detector_score_weight", type=float, default=0.0)
    parser.add_argument("--min_patch_matches", type=int, default=24)
    parser.add_argument("--min_patch_inliers", type=int, default=6)
    parser.add_argument("--base_sparse_solver", default="native", choices=["native", "opencv", "poselib"])
    parser.add_argument("--base_sparse_max_iterations", type=int, default=0)
    parser.add_argument("--base_sparse_min_iterations", type=int, default=0)
    parser.add_argument("--patch_pnp_max_iterations", type=int, default=5000)
    parser.add_argument("--patch_pnp_min_iterations", type=int, default=50)
    parser.add_argument("--patch_pnp_solver", default="opencv", choices=["opencv", "poselib"])
    parser.add_argument("--disable_pgsh_global_consistency", action="store_true")
    parser.add_argument("--pgsh_sparse_retention_min_ratio", type=float, default=0.70)
    parser.add_argument("--pgsh_sparse_retention_max_error_px", type=float, default=8.0)
    parser.add_argument("--pgsh_reference_match_max_reprojection_px", type=float, default=0.0)
    parser.add_argument("--pgsh_reference_match_min_keep", type=int, default=4)
    parser.add_argument("--cluster_center_thresh_m", type=float, default=2.0)
    parser.add_argument("--cluster_rotation_thresh_deg", type=float, default=12.0)
    parser.add_argument("--min_group_patches", type=int, default=2)
    parser.add_argument("--min_final_inliers", type=int, default=12)
    parser.add_argument("--pgsh_max_base_inliers_for_override", type=int, default=80)
    parser.add_argument("--disable_pgsh_fallback", action="store_true")
    parser.add_argument("--disable_sparse_pgsh", action="store_true")
    parser.add_argument("--run_dense", action="store_true")
    parser.add_argument("--dense_pgsh", action="store_true")
    parser.add_argument("--slcdp_sparse_conditioned_legacy_repair_selection", action="store_true")
    parser.add_argument("--dense_pgsh_min_patch_matches", type=int, default=16)
    parser.add_argument("--dense_pgsh_min_patch_inliers", type=int, default=5)
    parser.add_argument("--dense_pgsh_min_group_patches", type=int, default=2)
    parser.add_argument("--dense_pgsh_min_final_inliers", type=int, default=12)
    parser.add_argument("--dense_pgsh_pnp_solver", default="native", choices=["native", "opencv", "poselib"])
    parser.add_argument("--dense_pgsh_pnp_max_iterations", type=int, default=5000)
    parser.add_argument("--dense_pgsh_pnp_min_iterations", type=int, default=50)
    parser.add_argument("--dense_pgsh_reference_local_refine", action="store_true")
    parser.add_argument("--dense_pgsh_local_refine_max_iterations", type=int, default=50)
    parser.add_argument("--dense_pgsh_local_refine_loss_scale_px", type=float, default=4.0)
    parser.add_argument("--dense_pgsh_local_refine_translation_prior_weight", type=float, default=0.05)
    parser.add_argument("--dense_pgsh_local_refine_rotation_prior_weight", type=float, default=0.05)
    parser.add_argument("--dense_pgsh_sparse_anchor_weight", type=float, default=0.0)
    parser.add_argument("--dense_pgsh_sparse_anchor_max_count", type=int, default=64)
    parser.add_argument("--dense_pgsh_sparse_anchor_max_reprojection_px", type=float, default=8.0)
    parser.add_argument("--dense_pgsh_max_translation_from_base_dense_m", type=float, default=0.75)
    parser.add_argument("--dense_pgsh_max_rotation_from_base_dense_deg", type=float, default=3.0)
    parser.add_argument("--good_px", type=float, default=5.0)
    parser.add_argument("--max_draw", type=int, default=350)
    from loc_gs.scripts.visualize_stdloc_hard_matches import build_argparser as build_match_argparser

    match_parser = build_match_argparser()
    for action in match_parser._actions:
        if not action.option_strings or not any(option.startswith("--slcdp_") for option in action.option_strings):
            continue
        if any(option in parser._option_string_actions for option in action.option_strings):
            continue
        kwargs = {
            "default": action.default,
            "help": argparse.SUPPRESS,
        }
        if isinstance(action, argparse._StoreTrueAction):
            kwargs["action"] = "store_true"
        elif action.choices is not None:
            kwargs["choices"] = action.choices
        elif action.type is not None:
            kwargs["type"] = action.type
        parser.add_argument(*action.option_strings, **kwargs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.image_name and args.query_index is None:
        args.image_name = "seq2/frame00034.png"
        args.query_index = 33
    slcdp_options = _resolve_diagnostic_slcdp_options(args)
    analyze_case(
        candidate_root=Path(args.candidate_root),
        scene=str(args.scene),
        image_name=str(args.image_name),
        query_index=args.query_index,
        split=str(args.split),
        output_dir=Path(args.output_dir),
        patch_grids=_parse_ints(args.patch_grids),
        overlap_ratio=float(args.patch_overlap_ratio),
        patch_match_mode=str(args.patch_match_mode),
        patch_detect_num=int(args.patch_detect_num),
        patch_dual_softmax=str(args.patch_dual_softmax),
        patch_mnn_match=str(args.patch_mnn_match),
        patch_match_threshold=args.patch_match_threshold,
        pgsh_detector_score_weight=float(args.pgsh_detector_score_weight),
        min_patch_matches=int(args.min_patch_matches),
        min_patch_inliers=int(args.min_patch_inliers),
        base_sparse_solver=None if str(args.base_sparse_solver) == "native" else str(args.base_sparse_solver),
        base_sparse_max_iterations=int(args.base_sparse_max_iterations) if int(args.base_sparse_max_iterations) > 0 else None,
        base_sparse_min_iterations=int(args.base_sparse_min_iterations) if int(args.base_sparse_min_iterations) > 0 else None,
        patch_pnp_max_iterations=int(args.patch_pnp_max_iterations),
        patch_pnp_min_iterations=int(args.patch_pnp_min_iterations),
        patch_pnp_solver=str(args.patch_pnp_solver),
        pgsh_global_consistency=not bool(args.disable_pgsh_global_consistency),
        pgsh_sparse_retention_min_ratio=float(args.pgsh_sparse_retention_min_ratio),
        pgsh_sparse_retention_max_error_px=float(args.pgsh_sparse_retention_max_error_px),
        pgsh_reference_match_max_reprojection_px=float(args.pgsh_reference_match_max_reprojection_px),
        pgsh_reference_match_min_keep=int(args.pgsh_reference_match_min_keep),
        cluster_center_thresh_m=float(args.cluster_center_thresh_m),
        cluster_rotation_thresh_deg=float(args.cluster_rotation_thresh_deg),
        min_group_patches=int(args.min_group_patches),
        min_final_inliers=int(args.min_final_inliers),
        pgsh_max_base_inliers_for_override=int(args.pgsh_max_base_inliers_for_override),
        pgsh_fallback_to_base=not bool(args.disable_pgsh_fallback),
        disable_sparse_pgsh=bool(args.disable_sparse_pgsh),
        run_dense=bool(args.run_dense),
        dense_pgsh=bool(args.dense_pgsh),
        dense_pgsh_min_patch_matches=int(args.dense_pgsh_min_patch_matches),
        dense_pgsh_min_patch_inliers=int(args.dense_pgsh_min_patch_inliers),
        dense_pgsh_min_group_patches=int(args.dense_pgsh_min_group_patches),
        dense_pgsh_min_final_inliers=int(args.dense_pgsh_min_final_inliers),
        dense_pgsh_pnp_solver=None if str(args.dense_pgsh_pnp_solver) == "native" else str(args.dense_pgsh_pnp_solver),
        dense_pgsh_pnp_max_iterations=int(args.dense_pgsh_pnp_max_iterations),
        dense_pgsh_pnp_min_iterations=int(args.dense_pgsh_pnp_min_iterations),
        dense_pgsh_reference_local_refine=bool(args.dense_pgsh_reference_local_refine),
        dense_pgsh_local_refine_max_iterations=int(args.dense_pgsh_local_refine_max_iterations),
        dense_pgsh_local_refine_loss_scale_px=float(args.dense_pgsh_local_refine_loss_scale_px),
        dense_pgsh_local_refine_translation_prior_weight=float(args.dense_pgsh_local_refine_translation_prior_weight),
        dense_pgsh_local_refine_rotation_prior_weight=float(args.dense_pgsh_local_refine_rotation_prior_weight),
        dense_pgsh_sparse_anchor_weight=float(args.dense_pgsh_sparse_anchor_weight),
        dense_pgsh_sparse_anchor_max_count=int(args.dense_pgsh_sparse_anchor_max_count),
        dense_pgsh_sparse_anchor_max_reprojection_px=float(args.dense_pgsh_sparse_anchor_max_reprojection_px),
        dense_pgsh_max_translation_from_base_dense_m=float(args.dense_pgsh_max_translation_from_base_dense_m),
        dense_pgsh_max_rotation_from_base_dense_deg=float(args.dense_pgsh_max_rotation_from_base_dense_deg),
        good_px=float(args.good_px),
        max_draw=int(args.max_draw),
        slcdp_options=slcdp_options,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
