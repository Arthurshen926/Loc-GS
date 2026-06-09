#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))

from loc_gs.dense_support.sparse_anchor_residual import (
    SparseAnchorResidualPolicy,
    compute_sparse_anchor_residual_group,
)
from loc_gs.scripts.visualize_stdloc_hard_matches import (
    _build_context,
    _capture_dense,
    _capture_sparse,
    _resolve_slcdp_effective_options,
    pose_error_cm_deg,
)


CAMBRIDGE_SCENES = ("GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch")
RECALL_THRESHOLDS = {
    "R50cm5deg": (50.0, 5.0),
    "R15cm5deg": (15.0, 5.0),
    "R10cm5deg": (10.0, 5.0),
    "R5cm5deg": (5.0, 5.0),
    "R2cm2deg": (2.0, 2.0),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True).strip()
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=_repo_root(), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _preload_render_backend() -> bool:
    """Load gsplat's CUDA extension before large scene objects inflate process memory."""

    try:
        import gsplat.cuda._backend  # noqa: F401
    except Exception:
        return False
    return True


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _cvar_tail(values: Sequence[float], fraction: float = 0.10) -> float | None:
    if not values:
        return None
    ordered = sorted((float(v) for v in values), reverse=True)
    count = max(1, int(np.ceil(len(ordered) * float(fraction))))
    return float(sum(ordered[:count]) / count)


def summarize_pose_rows(rows: Sequence[Mapping[str, Any]], *, te_key: str, re_key: str) -> dict[str, Any]:
    te_values = [float(row[te_key]) for row in rows if row.get(te_key) is not None and np.isfinite(float(row[te_key]))]
    re_values = [float(row[re_key]) for row in rows if row.get(re_key) is not None and np.isfinite(float(row[re_key]))]
    summary: dict[str, Any] = {
        "query_count": int(len(rows)),
        "valid_count": int(min(len(te_values), len(re_values))),
        "median_te_cm": float(np.median(te_values)) if te_values else None,
        "median_re_deg": float(np.median(re_values)) if re_values else None,
        "p90_te_cm": _percentile(te_values, 90.0),
        "p95_te_cm": _percentile(te_values, 95.0),
        "cvar10_te_cm": _cvar_tail(te_values, 0.10),
        "severe_rate_1m": float(sum(v > 100.0 for v in te_values) / len(te_values)) if te_values else None,
        "catastrophic_rate_5m": float(sum(v > 500.0 for v in te_values) / len(te_values)) if te_values else None,
    }
    for name, (te_thr, re_thr) in RECALL_THRESHOLDS.items():
        hits = [
            float(row[te_key]) <= te_thr and float(row[re_key]) <= re_thr
            for row in rows
            if row.get(te_key) is not None and row.get(re_key) is not None
        ]
        summary[name] = float(sum(hits) / len(hits)) if hits else None
    return summary


def summarize_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [
        float(row["sparse_conditioned_dense_te_cm"]) - float(row["base_dense_te_cm"])
        for row in rows
        if row.get("sparse_conditioned_dense_te_cm") is not None and row.get("base_dense_te_cm") is not None
    ]
    return {
        "query_count": int(len(rows)),
        "median_delta_te_cm": float(np.median(deltas)) if deltas else None,
        "mean_delta_te_cm": float(np.mean(deltas)) if deltas else None,
        "p90_delta_te_cm": _percentile(deltas, 90.0),
        "regression_20cm_count": int(sum(v >= 20.0 for v in deltas)),
        "regression_50cm_count": int(sum(v >= 50.0 for v in deltas)),
        "improvement_20cm_count": int(sum(v <= -20.0 for v in deltas)),
        "worse_gt_0_1cm_count": int(sum(v > 0.1 for v in deltas)),
        "worse_gt_0_5cm_count": int(sum(v > 0.5 for v in deltas)),
        "improved_gt_0_5cm_count": int(sum(v < -0.5 for v in deltas)),
        "non_base_render_count": int(sum(str(row.get("sparse_conditioned_label")) != "base" for row in rows)),
        "low_confidence_gated_base_count": int(sum(bool(row.get("low_confidence_gated_base")) for row in rows)),
        "guided_pose_count": int(sum(str(row.get("sparse_conditioned_label", "")).startswith("gated_ray_") for row in rows)),
        "base_dense_reused_count": int(sum(bool(row.get("base_dense_reused")) for row in rows)),
        "base_dense_high_confidence_reuse_count": int(
            sum(str(row.get("base_reuse_reason", "")) == "base_dense_high_confidence" for row in rows)
        ),
        "base_preflight_safe_reuse_count": int(
            sum(str(row.get("base_reuse_reason", "")) == "base_preflight_safe" for row in rows)
        ),
        "base_dense_worsened_count": int(
            sum(
                float(row["base_dense_te_cm"]) - float(row["sparse_te_cm"]) >= 5.0
                for row in rows
                if row.get("base_dense_te_cm") is not None and row.get("sparse_te_cm") is not None
            )
        ),
        "sparse_conditioned_dense_worsened_count": int(
            sum(
                float(row["sparse_conditioned_dense_te_cm"]) - float(row["sparse_te_cm"]) >= 5.0
                for row in rows
                if row.get("sparse_conditioned_dense_te_cm") is not None and row.get("sparse_te_cm") is not None
            )
        ),
    }


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return -rotation.T @ translation


def _rotation_delta_deg(reference_w2c: np.ndarray, candidate_w2c: np.ndarray) -> float:
    reference = np.asarray(reference_w2c, dtype=np.float64).reshape(4, 4)
    candidate = np.asarray(candidate_w2c, dtype=np.float64).reshape(4, 4)
    relative = candidate[:3, :3] @ reference[:3, :3].T
    cos_angle = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _pose_delta_fields(prefix: str, reference_w2c: np.ndarray, candidate_w2c: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_translation_delta_m": float(np.linalg.norm(_camera_center(candidate_w2c) - _camera_center(reference_w2c))),
        f"{prefix}_rotation_delta_deg": _rotation_delta_deg(reference_w2c, candidate_w2c),
    }


def _sparse_anchor_residual_fields(
    prefix: str,
    *,
    pose_w2c: np.ndarray,
    sparse_capture: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    anchors = {
        "query_xy": sparse_capture.get("query_xy", np.empty((0, 2), dtype=np.float32)),
        "p3d": sparse_capture.get("p3d", np.empty((0, 3), dtype=np.float32)),
        "inliers": sparse_capture.get("inliers", np.empty((0,), dtype=np.int32)),
    }
    residual = compute_sparse_anchor_residual_group(
        anchors,
        pose_w2c=np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
        intrinsic=np.asarray(sparse_capture.get("K"), dtype=np.float64).reshape(3, 3),
        image_size=(int(image_size[0]), int(image_size[1])),
        policy=SparseAnchorResidualPolicy(max_anchor_count=256),
    )
    return {
        f"{prefix}_anchor_valid_count": residual.get("valid_anchor_count"),
        f"{prefix}_anchor_median_px": residual.get("median_error_px"),
        f"{prefix}_anchor_p90_px": residual.get("p90_error_px"),
        f"{prefix}_anchor_group_loss": residual.get("group_loss"),
    }


def _fixed_options(
    render_control: str,
    *,
    apd_dense: bool = False,
    apd_include_patch_candidates: bool = False,
    apd_dense_group_weight: float = 1.0,
    apd_anchor_group_weight: float = 1.0,
    apd_max_translation_delta_m: float = 0.0,
    apd_max_rotation_delta_deg: float = 0.0,
    apd_anchor_monotonic: bool = False,
    apd_anchor_monotonic_epsilon_px: float = 1.0,
    apd_use_refined_pose: bool = False,
    apd_risk_weighted: bool = False,
    apd_risk_max_beta: float = 1.0,
    apd_no_regression_gate: bool = True,
    apd_gate_min_inlier_ratio_delta: float = 0.02,
    apd_gate_max_median_reproj_increase_px: float = 1.0,
    apd_gate_max_p90_reproj_increase_px: float = 3.0,
    apd_gate_min_inlier_count_ratio: float = 0.90,
    sadc_dense: bool = False,
    sadc_include_patch_candidates: bool = False,
    sadc_mode: str = "conflict_filter_patch_append",
    sadc_max_candidates: int = 4096,
    sadc_anchor_flow_scale_px: float = 8.0,
    sadc_native_drop_percentile: float = 95.0,
    sadc_min_native_keep_ratio: float = 0.90,
    sadc_min_native_conflict_score: float = 0.75,
    sadc_patch_add_percentile: float = 90.0,
    sadc_max_patch_fraction: float = 0.15,
    sadc_min_patch_anchor_consistency: float = 0.75,
    sadc_anchor_monotonic: bool = False,
    sadc_anchor_monotonic_epsilon_px: float = 1.0,
    sadc_activation_mode: str = "always",
    sadc_activation_min_risk: float = 0.5,
    sadc_activation_min_sparse_confidence: float = 0.0,
) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    from loc_gs.scripts.visualize_stdloc_hard_matches import build_argparser as build_visualizer_argparser

    parser = build_visualizer_argparser()
    argv = ["--slcdp_render_control", render_control]
    if bool(apd_dense):
        argv.append("--apd_dense")
    if bool(apd_include_patch_candidates):
        argv.append("--apd_include_patch_candidates")
    if bool(apd_anchor_monotonic):
        argv.append("--apd_anchor_monotonic")
    if bool(apd_use_refined_pose):
        argv.append("--apd_use_refined_pose")
    if bool(apd_risk_weighted):
        argv.append("--apd_risk_weighted")
    if bool(sadc_dense):
        argv.append("--sadc_dense")
    if bool(sadc_include_patch_candidates):
        argv.append("--sadc_include_patch_candidates")
    if bool(sadc_anchor_monotonic):
        argv.append("--sadc_anchor_monotonic")
    if not bool(apd_no_regression_gate):
        argv.append("--no_apd_no_regression_gate")
    argv.extend(
        [
            "--apd_dense_group_weight",
            str(float(apd_dense_group_weight)),
            "--apd_anchor_group_weight",
            str(float(apd_anchor_group_weight)),
            "--apd_max_translation_delta_m",
            str(float(apd_max_translation_delta_m)),
            "--apd_max_rotation_delta_deg",
            str(float(apd_max_rotation_delta_deg)),
            "--apd_anchor_monotonic_epsilon_px",
            str(float(apd_anchor_monotonic_epsilon_px)),
            "--apd_risk_max_beta",
            str(float(apd_risk_max_beta)),
            "--apd_gate_min_inlier_ratio_delta",
            str(float(apd_gate_min_inlier_ratio_delta)),
            "--apd_gate_max_median_reproj_increase_px",
            str(float(apd_gate_max_median_reproj_increase_px)),
            "--apd_gate_max_p90_reproj_increase_px",
            str(float(apd_gate_max_p90_reproj_increase_px)),
            "--apd_gate_min_inlier_count_ratio",
            str(float(apd_gate_min_inlier_count_ratio)),
            "--sadc_max_candidates",
            str(int(sadc_max_candidates)),
            "--sadc_anchor_flow_scale_px",
            str(float(sadc_anchor_flow_scale_px)),
            "--sadc_mode",
            str(sadc_mode),
            "--sadc_native_drop_percentile",
            str(float(sadc_native_drop_percentile)),
            "--sadc_min_native_keep_ratio",
            str(float(sadc_min_native_keep_ratio)),
            "--sadc_min_native_conflict_score",
            str(float(sadc_min_native_conflict_score)),
            "--sadc_patch_add_percentile",
            str(float(sadc_patch_add_percentile)),
            "--sadc_max_patch_fraction",
            str(float(sadc_max_patch_fraction)),
            "--sadc_min_patch_anchor_consistency",
            str(float(sadc_min_patch_anchor_consistency)),
            "--sadc_anchor_monotonic_epsilon_px",
            str(float(sadc_anchor_monotonic_epsilon_px)),
            "--sadc_activation_mode",
            str(sadc_activation_mode),
            "--sadc_activation_min_risk",
            str(float(sadc_activation_min_risk)),
            "--sadc_activation_min_sparse_confidence",
            str(float(sadc_activation_min_sparse_confidence)),
        ]
    )
    args = parser.parse_args(argv)
    return _resolve_slcdp_effective_options(args)


def _scene_output_dir(output_dir: Path, scene: str) -> Path:
    return output_dir / scene


def _read_case_rows(
    paths: Sequence[str],
    *,
    case_types: Sequence[str],
    phase0_split: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    selected_types = {str(item) for item in case_types if str(item)}
    selected_phase0 = str(phase0_split)
    by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if selected_types and str(row.get("case_type", "")) not in selected_types:
                    continue
                if selected_phase0 != "all" and str(row.get("phase0_split", "")) != selected_phase0:
                    continue
                if str(row.get("source_split", "unknown")) == "test":
                    raise ValueError(
                        "case_csv contains official test rows; this evaluator is for train/self-map diagnostics"
                    )
                scene = str(row.get("scene", ""))
                image_name = str(row.get("image_name", ""))
                if not scene or not image_name:
                    continue
                by_scene.setdefault(scene, {})[image_name] = dict(row)
    return by_scene


def _resume_rows(scene_dir: Path, cameras: Sequence[Any], *, resume_partial: bool) -> list[dict[str, Any]]:
    if not bool(resume_partial):
        return []
    partial_path = scene_dir / "results.partial.json"
    if not partial_path.exists():
        return []
    rows = json.loads(partial_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Partial results are not a row list: {partial_path}")
    if len(rows) > len(cameras):
        raise ValueError(f"Partial row count exceeds camera count: {len(rows)} > {len(cameras)}")
    for index, row in enumerate(rows):
        expected_name = str(getattr(cameras[index], "image_name"))
        actual_name = str(row.get("image_name"))
        if actual_name != expected_name:
            raise ValueError(
                f"Partial row {index} image_name={actual_name!r} does not match camera prefix {expected_name!r}"
            )
    return [dict(row) for row in rows]


def _resume_partial_rows(scene_dir: Path, *, resume_partial: bool) -> list[dict[str, Any]]:
    if not bool(resume_partial):
        return []
    partial_path = scene_dir / "results.partial.json"
    if not partial_path.exists():
        return []
    rows = json.loads(partial_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Partial results are not a row list: {partial_path}")
    return [dict(row) for row in rows]


def _selected_camera_count(camera_iterable: Any, *, query_stride: int, max_queries: int) -> int | None:
    try:
        total = int(len(camera_iterable))
    except TypeError:
        return int(max_queries) if int(max_queries) > 0 else None
    stride = max(1, int(query_stride))
    selected = (total + stride - 1) // stride
    if int(max_queries) > 0:
        selected = min(selected, int(max_queries))
    return int(selected)


def evaluate_scene(
    *,
    candidate_root: Path,
    output_dir: Path,
    scene: str,
    eval_split: str,
    max_queries: int,
    query_stride: int,
    progress_interval: int,
    resume_partial: bool = False,
    candidate_render_control: str = "sparse_conditioned",
    apd_dense: bool = False,
    apd_include_patch_candidates: bool = False,
    apd_dense_group_weight: float = 1.0,
    apd_anchor_group_weight: float = 1.0,
    apd_max_translation_delta_m: float = 0.0,
    apd_max_rotation_delta_deg: float = 0.0,
    apd_anchor_monotonic: bool = False,
    apd_anchor_monotonic_epsilon_px: float = 1.0,
    apd_use_refined_pose: bool = False,
    apd_risk_weighted: bool = False,
    apd_risk_max_beta: float = 1.0,
    apd_no_regression_gate: bool = True,
    apd_gate_min_inlier_ratio_delta: float = 0.02,
    apd_gate_max_median_reproj_increase_px: float = 1.0,
    apd_gate_max_p90_reproj_increase_px: float = 3.0,
    apd_gate_min_inlier_count_ratio: float = 0.90,
    sadc_dense: bool = False,
    sadc_include_patch_candidates: bool = False,
    sadc_mode: str = "conflict_filter_patch_append",
    sadc_max_candidates: int = 4096,
    sadc_anchor_flow_scale_px: float = 8.0,
    sadc_native_drop_percentile: float = 95.0,
    sadc_min_native_keep_ratio: float = 0.90,
    sadc_min_native_conflict_score: float = 0.75,
    sadc_patch_add_percentile: float = 90.0,
    sadc_max_patch_fraction: float = 0.15,
    sadc_min_patch_anchor_consistency: float = 0.75,
    sadc_anchor_monotonic: bool = False,
    sadc_anchor_monotonic_epsilon_px: float = 1.0,
    sadc_activation_mode: str = "always",
    sadc_activation_min_risk: float = 0.5,
    sadc_activation_min_sparse_confidence: float = 0.0,
    case_rows_by_image: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _preload_render_backend()
    run_dir = candidate_root / scene
    ctx = _build_context(run_dir, split_override=eval_split)
    base_options = _fixed_options("none")
    candidate_options = _fixed_options(
        candidate_render_control,
        apd_dense=bool(apd_dense),
        apd_include_patch_candidates=bool(apd_include_patch_candidates),
        apd_dense_group_weight=float(apd_dense_group_weight),
        apd_anchor_group_weight=float(apd_anchor_group_weight),
        apd_max_translation_delta_m=float(apd_max_translation_delta_m),
        apd_max_rotation_delta_deg=float(apd_max_rotation_delta_deg),
        apd_anchor_monotonic=bool(apd_anchor_monotonic),
        apd_anchor_monotonic_epsilon_px=float(apd_anchor_monotonic_epsilon_px),
        apd_use_refined_pose=bool(apd_use_refined_pose),
        apd_risk_weighted=bool(apd_risk_weighted),
        apd_risk_max_beta=float(apd_risk_max_beta),
        apd_no_regression_gate=bool(apd_no_regression_gate),
        apd_gate_min_inlier_ratio_delta=float(apd_gate_min_inlier_ratio_delta),
        apd_gate_max_median_reproj_increase_px=float(apd_gate_max_median_reproj_increase_px),
        apd_gate_max_p90_reproj_increase_px=float(apd_gate_max_p90_reproj_increase_px),
        apd_gate_min_inlier_count_ratio=float(apd_gate_min_inlier_count_ratio),
        sadc_dense=bool(sadc_dense),
        sadc_include_patch_candidates=bool(sadc_include_patch_candidates),
        sadc_mode=str(sadc_mode),
        sadc_max_candidates=int(sadc_max_candidates),
        sadc_anchor_flow_scale_px=float(sadc_anchor_flow_scale_px),
        sadc_native_drop_percentile=float(sadc_native_drop_percentile),
        sadc_min_native_keep_ratio=float(sadc_min_native_keep_ratio),
        sadc_min_native_conflict_score=float(sadc_min_native_conflict_score),
        sadc_patch_add_percentile=float(sadc_patch_add_percentile),
        sadc_max_patch_fraction=float(sadc_max_patch_fraction),
        sadc_min_patch_anchor_consistency=float(sadc_min_patch_anchor_consistency),
        sadc_anchor_monotonic=bool(sadc_anchor_monotonic),
        sadc_anchor_monotonic_epsilon_px=float(sadc_anchor_monotonic_epsilon_px),
        sadc_activation_mode=str(sadc_activation_mode),
        sadc_activation_min_risk=float(sadc_activation_min_risk),
        sadc_activation_min_sparse_confidence=float(sadc_activation_min_sparse_confidence),
    )
    cameras = ctx["cameras"]
    total_selected = _selected_camera_count(
        cameras,
        query_stride=int(query_stride),
        max_queries=int(max_queries),
    )
    scene_dir = _scene_output_dir(output_dir, scene)
    rows: list[dict[str, Any]] = _resume_partial_rows(scene_dir, resume_partial=bool(resume_partial))
    resume_start_index = int(len(rows))
    start = time.time()
    selected_index = 0
    case_rows_by_image = dict(case_rows_by_image or {})
    use_case_filter = bool(case_rows_by_image)
    for raw_index, camera in enumerate(cameras):
        if int(query_stride) > 1 and raw_index % int(query_stride) != 0:
            continue
        image_name = str(getattr(camera, "image_name"))
        if use_case_filter and image_name not in case_rows_by_image:
            continue
        if int(max_queries) > 0 and selected_index >= int(max_queries):
            break
        if selected_index < resume_start_index:
            expected_name = str(rows[selected_index].get("image_name"))
            actual_name = image_name
            if actual_name != expected_name:
                raise ValueError(
                    f"Partial row {selected_index} image_name={expected_name!r} "
                    f"does not match camera prefix {actual_name!r}"
                )
            selected_index += 1
            continue
        index = raw_index if use_case_filter else selected_index
        case_row = case_rows_by_image.get(image_name, {})
        query_image = camera.original_image.to("cuda")
        gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        with torch.no_grad():
            sparse = _capture_sparse(ctx, query_image, camera.FoVx, camera.FoVy)
            base_dense = _capture_dense(
                ctx,
                sparse["coarse_map"],
                sparse["fine_map"],
                sparse["pose_w2c"],
                camera.FoVx,
                camera.FoVy,
                sparse_capture=sparse,
                **base_options,
            )
            controlled_dense = _capture_dense(
                ctx,
                sparse["coarse_map"],
                sparse["fine_map"],
                sparse["pose_w2c"],
                camera.FoVx,
                camera.FoVy,
                sparse_capture=sparse,
                slcdp_base_dense_capture=base_dense,
                **candidate_options,
            )
        sparse_te, sparse_re = pose_error_cm_deg(sparse["pose_w2c"], gt_w2c)
        base_te, base_re = pose_error_cm_deg(base_dense["pose_w2c"], gt_w2c)
        controlled_te, controlled_re = pose_error_cm_deg(controlled_dense["pose_w2c"], gt_w2c)
        selection = (controlled_dense.get("slcdp_repair_search") or {}).get("selection") or {}
        clean_render_selection = (controlled_dense.get("slcdp_repair_search") or {}).get("clean_render_generation") or {}
        render_control = controlled_dense.get("slcdp_render_control") or {}
        dense_quality = controlled_dense.get("dense_pose_quality") or {}
        base_dense_quality = base_dense.get("dense_pose_quality") or {}
        apd_info = controlled_dense.get("apd_dense") or {}
        apd_monotonic = ((apd_info.get("diagnostics") or {}).get("anchor_monotonic") or {})
        apd_switch = controlled_dense.get("apd_pose_switch") or {}
        sadc_info = controlled_dense.get("sadc_dense") or {}
        sadc_activation = controlled_dense.get("sadc_activation") or {}
        sadc_anchor_monotonic_diag = controlled_dense.get("sadc_anchor_monotonic") or {}
        sadc_score_summary = sadc_info.get("score_summary") or {}
        selected_label = render_control.get("candidate_label")
        if selected_label is None:
            selected_label = clean_render_selection.get("selected_label", "base")
        image_size = (int(sparse.get("width")), int(sparse.get("height")))
        sparse_anchor_fields = _sparse_anchor_residual_fields(
            "sparse_pose",
            pose_w2c=sparse["pose_w2c"],
            sparse_capture=sparse,
            image_size=image_size,
        )
        base_anchor_fields = _sparse_anchor_residual_fields(
            "base_dense_pose",
            pose_w2c=base_dense["pose_w2c"],
            sparse_capture=sparse,
            image_size=image_size,
        )
        controlled_anchor_fields = _sparse_anchor_residual_fields(
            "controlled_dense_pose",
            pose_w2c=controlled_dense["pose_w2c"],
            sparse_capture=sparse,
            image_size=image_size,
        )
        rows.append(
            {
                "scene": scene,
                "split": eval_split,
                "candidate_method": "sadc_dense" if bool(sadc_dense) else "apd_dense" if bool(apd_dense) else str(candidate_render_control),
                "query_index": int(index),
                "image_name": image_name,
                "case_type": case_row.get("case_type"),
                "phase0_split": case_row.get("phase0_split"),
                "sparse_te_cm": sparse_te,
                "sparse_re_deg": sparse_re,
                "sparse_inlier_count": int(np.asarray(sparse.get("inliers", [])).reshape(-1).shape[0]),
                "base_dense_te_cm": base_te,
                "base_dense_re_deg": base_re,
                "base_dense_inlier_count": int(np.asarray(base_dense.get("inliers", [])).reshape(-1).shape[0]),
                "base_dense_pose_match_count": base_dense_quality.get("match_count"),
                "base_dense_pose_solver_inlier_count": base_dense_quality.get("solver_inlier_count"),
                "base_dense_pose_solver_inlier_ratio": base_dense_quality.get("solver_inlier_ratio"),
                "base_dense_pose_median_reprojection_error_px": base_dense_quality.get("median_reprojection_error_px"),
                "base_dense_pose_p90_reprojection_error_px": base_dense_quality.get("p90_reprojection_error_px"),
                "sparse_conditioned_dense_te_cm": controlled_te,
                "sparse_conditioned_dense_re_deg": controlled_re,
                "sparse_conditioned_dense_inlier_count": int(np.asarray(controlled_dense.get("inliers", [])).reshape(-1).shape[0]),
                "sparse_conditioned_label": selected_label,
                "sparse_conditioned_decision": selection.get("decision"),
                "sparse_conditioned_best_label": selection.get("best_label"),
                "sparse_conditioned_score_gain": selection.get("score_gain"),
                "clean_render_decision": clean_render_selection.get("decision"),
                "clean_render_selected_label": clean_render_selection.get("selected_label"),
                "clean_render_score_gain": clean_render_selection.get("score_gain"),
                "low_confidence_gated_base": bool(selection.get("low_confidence_gated_base", False)),
                "transition_decision": (controlled_dense.get("slcdp_transition_control") or {}).get("decision"),
                "base_dense_reused": bool(render_control.get("base_dense_reused", False)),
                "base_reuse_reason": render_control.get("base_reuse_reason"),
                "apd_dense_refine_success": apd_info.get("dense_refine_success"),
                "apd_num_global_candidates": apd_info.get("num_global_candidates"),
                "apd_num_clean_render_candidates": apd_info.get("num_clean_render_candidates"),
                "apd_num_patch_candidates": apd_info.get("num_patch_candidates"),
                "apd_num_anchor_residuals": apd_info.get("num_anchor_residuals"),
                "apd_anchor_monotonic_selected_alpha": apd_monotonic.get("selected_alpha"),
                "apd_anchor_monotonic_reason": apd_monotonic.get("reason"),
                "apd_anchor_monotonic_sparse_anchor_median_px": apd_monotonic.get("sparse_anchor_median_px"),
                "apd_anchor_monotonic_selected_anchor_median_px": apd_monotonic.get("selected_anchor_median_px"),
                "apd_pose_switch_mode": apd_switch.get("mode"),
                "apd_pose_switch_beta": apd_switch.get("beta"),
                "apd_pose_switch_risk": apd_switch.get("risk"),
                "apd_pose_switch_accept_apd_pose": apd_switch.get("accept_apd_pose"),
                "sadc_mode": sadc_info.get("mode"),
                "sadc_candidate_count": sadc_info.get("candidate_count"),
                "sadc_kept_count": sadc_info.get("kept_count"),
                "sadc_dropped_count": sadc_info.get("dropped_count"),
                "sadc_native_candidate_count": sadc_info.get("native_candidate_count"),
                "sadc_native_kept_count": sadc_info.get("native_kept_count"),
                "sadc_patch_candidate_count": sadc_info.get("patch_candidate_count"),
                "sadc_patch_kept_count": sadc_info.get("patch_kept_count"),
                "sadc_conflict_score_mean": sadc_info.get("conflict_score_mean"),
                "sadc_conflict_score_p95": sadc_info.get("conflict_score_p95"),
                "sadc_weight_median": sadc_info.get("weight_median"),
                "sadc_weight_mean": sadc_info.get("weight_mean"),
                "sadc_anchor_flow_median": sadc_score_summary.get("anchor_flow_median"),
                "sadc_match_quality_median": sadc_score_summary.get("match_quality_median"),
                "sadc_geometry_score": sadc_score_summary.get("geometry_score"),
                "sadc_activation_mode": sadc_activation.get("mode"),
                "sadc_activation_active": sadc_activation.get("active"),
                "sadc_activation_reason": sadc_activation.get("reason"),
                "sadc_activation_risk": sadc_activation.get("risk"),
                "sadc_activation_sparse_confidence": sadc_activation.get("sparse_confidence"),
                "sadc_anchor_monotonic_selected_alpha": sadc_anchor_monotonic_diag.get("selected_alpha"),
                "sadc_anchor_monotonic_reason": sadc_anchor_monotonic_diag.get("reason"),
                "sadc_anchor_monotonic_sparse_anchor_median_px": sadc_anchor_monotonic_diag.get("sparse_anchor_median_px"),
                "sadc_anchor_monotonic_selected_anchor_median_px": sadc_anchor_monotonic_diag.get("selected_anchor_median_px"),
                "dense_pose_match_count": dense_quality.get("match_count"),
                "dense_pose_solver_inlier_count": dense_quality.get("solver_inlier_count"),
                "dense_pose_solver_inlier_ratio": dense_quality.get("solver_inlier_ratio"),
                "dense_pose_median_reprojection_error_px": dense_quality.get("median_reprojection_error_px"),
                "dense_pose_p90_reprojection_error_px": dense_quality.get("p90_reprojection_error_px"),
                **_pose_delta_fields("base_dense_vs_sparse", sparse["pose_w2c"], base_dense["pose_w2c"]),
                **_pose_delta_fields("controlled_dense_vs_sparse", sparse["pose_w2c"], controlled_dense["pose_w2c"]),
                **sparse_anchor_fields,
                **base_anchor_fields,
                **controlled_anchor_fields,
            }
        )
        if (index + 1) % max(1, int(progress_interval)) == 0 or index == 0:
            _write_json(scene_dir / "results.partial.json", rows)
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "split": eval_split,
                        "processed": index + 1,
                        "total": total_selected,
                        "elapsed_sec": round(float(time.time() - start), 3),
                    }
                ),
                flush=True,
            )
        selected_index += 1
    summary = {
        "scene": scene,
        "split": eval_split,
        "elapsed_sec": float(time.time() - start),
        "base_dense": summarize_pose_rows(rows, te_key="base_dense_te_cm", re_key="base_dense_re_deg"),
        "sparse_conditioned_dense": summarize_pose_rows(
            rows,
            te_key="sparse_conditioned_dense_te_cm",
            re_key="sparse_conditioned_dense_re_deg",
        ),
        "sparse": summarize_pose_rows(rows, te_key="sparse_te_cm", re_key="sparse_re_deg"),
        "comparison": summarize_comparison(rows),
    }
    _write_json(scene_dir / "results.json", rows)
    _write_json(scene_dir / "summary.json", summary)
    _write_json(
        scene_dir / "manifest.json",
        {
            "schema": "loc_gs_sparse_conditioned_dense_control_eval_v1",
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "scene": scene,
            "split": eval_split,
            "candidate_root": str(candidate_root),
            "run_dir": str(run_dir),
            "map_path": ctx.get("map_path"),
            "data_root": ctx.get("scene_root"),
            "render_control": str(candidate_render_control),
            "base_render_control": "none",
            "apd_dense": bool(apd_dense),
            "apd_include_patch_candidates": bool(apd_include_patch_candidates),
            "sadc_dense": bool(sadc_dense),
            "sadc_include_patch_candidates": bool(sadc_include_patch_candidates),
            "sadc_anchor_monotonic": bool(sadc_anchor_monotonic),
            "sadc_anchor_monotonic_epsilon_px": float(sadc_anchor_monotonic_epsilon_px),
            "sadc_activation_mode": str(sadc_activation_mode),
            "sadc_activation_min_risk": float(sadc_activation_min_risk),
            "sadc_activation_min_sparse_confidence": float(sadc_activation_min_sparse_confidence),
            "selection_policy": candidate_options,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "resume_partial": bool(resume_partial),
            "resume_start_index": int(resume_start_index),
            "diagnostic_only": True,
            "case_filter_enabled": bool(use_case_filter),
            "case_filter_count": int(len(case_rows_by_image)),
        },
    )
    (scene_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    (scene_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    _write_json(scene_dir / "split_audit.json", {"scene": scene, "split": eval_split, "diagnostic_only": True})
    return summary


def _macro_summary(scene_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"scene_count": int(len(scene_summaries))}
    for method in ("base_dense", "sparse_conditioned_dense"):
        method_rows = [summary[method] for summary in scene_summaries]
        out[method] = {
            key: float(np.mean([float(row[key]) for row in method_rows if row.get(key) is not None]))
            for key in ("median_te_cm", "median_re_deg", "R50cm5deg", "R15cm5deg", "R10cm5deg", "R5cm5deg", "R2cm2deg", "p90_te_cm", "p95_te_cm", "cvar10_te_cm")
        }
    comparisons = [summary["comparison"] for summary in scene_summaries]
    out["comparison"] = {
        "mean_median_delta_te_cm": float(np.mean([float(row["median_delta_te_cm"]) for row in comparisons if row.get("median_delta_te_cm") is not None])),
        "total_regression_20cm_count": int(sum(int(row.get("regression_20cm_count", 0)) for row in comparisons)),
        "total_regression_50cm_count": int(sum(int(row.get("regression_50cm_count", 0)) for row in comparisons)),
        "total_improvement_20cm_count": int(sum(int(row.get("improvement_20cm_count", 0)) for row in comparisons)),
        "total_non_base_render_count": int(sum(int(row.get("non_base_render_count", 0)) for row in comparisons)),
        "total_low_confidence_gated_base_count": int(sum(int(row.get("low_confidence_gated_base_count", 0)) for row in comparisons)),
        "total_guided_pose_count": int(sum(int(row.get("guided_pose_count", 0)) for row in comparisons)),
        "total_base_dense_reused_count": int(sum(int(row.get("base_dense_reused_count", 0)) for row in comparisons)),
        "total_base_dense_high_confidence_reuse_count": int(
            sum(int(row.get("base_dense_high_confidence_reuse_count", 0)) for row in comparisons)
        ),
        "total_base_preflight_safe_reuse_count": int(
            sum(int(row.get("base_preflight_safe_reuse_count", 0)) for row in comparisons)
        ),
        "base_dense_worsened_count": int(sum(int(row.get("base_dense_worsened_count", 0)) for row in comparisons)),
        "sparse_conditioned_dense_worsened_count": int(sum(int(row.get("sparse_conditioned_dense_worsened_count", 0)) for row in comparisons)),
    }
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fixed sparse-conditioned dense render control on Cambridge splits.")
    parser.add_argument("--candidate_root", default="output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--eval_split", choices=["train", "test"], default="train")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--resume_partial", action="store_true")
    parser.add_argument(
        "--candidate_render_control",
        choices=[
            "none",
            "gaussian_gating",
            "guided_pose",
            "gating_guided_pose",
            "fast_guided_pose",
            "fast_gating_guided_pose",
            "sparse_conditioned",
        ],
        default="sparse_conditioned",
    )
    parser.add_argument("--apd_dense", action="store_true")
    parser.add_argument("--apd_include_patch_candidates", action="store_true")
    parser.add_argument("--apd_dense_group_weight", type=float, default=1.0)
    parser.add_argument("--apd_anchor_group_weight", type=float, default=1.0)
    parser.add_argument("--apd_max_translation_delta_m", type=float, default=0.0)
    parser.add_argument("--apd_max_rotation_delta_deg", type=float, default=0.0)
    parser.add_argument("--apd_anchor_monotonic", action="store_true")
    parser.add_argument("--apd_anchor_monotonic_epsilon_px", type=float, default=1.0)
    parser.add_argument("--apd_use_refined_pose", action="store_true")
    parser.add_argument("--apd_risk_weighted", action="store_true")
    parser.add_argument("--apd_risk_max_beta", type=float, default=1.0)
    parser.add_argument("--apd_no_regression_gate", action="store_true", default=True)
    parser.add_argument("--no_apd_no_regression_gate", action="store_false", dest="apd_no_regression_gate")
    parser.add_argument("--apd_gate_min_inlier_ratio_delta", type=float, default=0.02)
    parser.add_argument("--apd_gate_max_median_reproj_increase_px", type=float, default=1.0)
    parser.add_argument("--apd_gate_max_p90_reproj_increase_px", type=float, default=3.0)
    parser.add_argument("--apd_gate_min_inlier_count_ratio", type=float, default=0.90)
    parser.add_argument("--sadc_dense", action="store_true")
    parser.add_argument("--sadc_include_patch_candidates", action="store_true")
    parser.add_argument(
        "--sadc_mode",
        choices=[
            "topk",
            "passthrough",
            "score_only",
            "conflict_filter",
            "patch_append",
            "conflict_filter_patch_append",
        ],
        default="conflict_filter_patch_append",
    )
    parser.add_argument("--sadc_max_candidates", type=int, default=4096)
    parser.add_argument("--sadc_anchor_flow_scale_px", type=float, default=8.0)
    parser.add_argument("--sadc_native_drop_percentile", type=float, default=95.0)
    parser.add_argument("--sadc_min_native_keep_ratio", type=float, default=0.90)
    parser.add_argument("--sadc_min_native_conflict_score", type=float, default=0.75)
    parser.add_argument("--sadc_patch_add_percentile", type=float, default=90.0)
    parser.add_argument("--sadc_max_patch_fraction", type=float, default=0.15)
    parser.add_argument("--sadc_min_patch_anchor_consistency", type=float, default=0.75)
    parser.add_argument("--sadc_anchor_monotonic", action="store_true")
    parser.add_argument("--sadc_anchor_monotonic_epsilon_px", type=float, default=1.0)
    parser.add_argument("--sadc_activation_mode", choices=["always", "dense_damage_risk"], default="always")
    parser.add_argument("--sadc_activation_min_risk", type=float, default=0.5)
    parser.add_argument("--sadc_activation_min_sparse_confidence", type=float, default=0.0)
    parser.add_argument("--case_csv", action="append", default=[])
    parser.add_argument("--case_type", action="append", default=[])
    parser.add_argument("--phase0_split", choices=["train", "val", "all"], default="all")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    ns = build_argparser().parse_args() if args is None else args
    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows = _read_case_rows(
        list(ns.case_csv),
        case_types=list(ns.case_type),
        phase0_split=str(ns.phase0_split),
    ) if ns.case_csv else {}
    scenes = tuple(ns.scene) if ns.scene else tuple(sorted(case_rows.keys())) if case_rows else CAMBRIDGE_SCENES
    summaries = [
        evaluate_scene(
            candidate_root=Path(ns.candidate_root),
            output_dir=output_dir,
            scene=scene,
            eval_split=str(ns.eval_split),
            max_queries=int(ns.max_queries),
            query_stride=int(ns.query_stride),
            progress_interval=int(ns.progress_interval),
            resume_partial=bool(ns.resume_partial),
            candidate_render_control=str(ns.candidate_render_control),
            apd_dense=bool(ns.apd_dense),
            apd_include_patch_candidates=bool(ns.apd_include_patch_candidates),
            apd_dense_group_weight=float(ns.apd_dense_group_weight),
            apd_anchor_group_weight=float(ns.apd_anchor_group_weight),
            apd_max_translation_delta_m=float(ns.apd_max_translation_delta_m),
            apd_max_rotation_delta_deg=float(ns.apd_max_rotation_delta_deg),
            apd_anchor_monotonic=bool(ns.apd_anchor_monotonic),
            apd_anchor_monotonic_epsilon_px=float(ns.apd_anchor_monotonic_epsilon_px),
            apd_use_refined_pose=bool(ns.apd_use_refined_pose),
            apd_risk_weighted=bool(ns.apd_risk_weighted),
            apd_risk_max_beta=float(ns.apd_risk_max_beta),
            apd_no_regression_gate=bool(ns.apd_no_regression_gate),
            apd_gate_min_inlier_ratio_delta=float(ns.apd_gate_min_inlier_ratio_delta),
            apd_gate_max_median_reproj_increase_px=float(ns.apd_gate_max_median_reproj_increase_px),
            apd_gate_max_p90_reproj_increase_px=float(ns.apd_gate_max_p90_reproj_increase_px),
            apd_gate_min_inlier_count_ratio=float(ns.apd_gate_min_inlier_count_ratio),
            sadc_dense=bool(ns.sadc_dense),
            sadc_include_patch_candidates=bool(ns.sadc_include_patch_candidates),
            sadc_mode=str(ns.sadc_mode),
            sadc_max_candidates=int(ns.sadc_max_candidates),
            sadc_anchor_flow_scale_px=float(ns.sadc_anchor_flow_scale_px),
            sadc_native_drop_percentile=float(ns.sadc_native_drop_percentile),
            sadc_min_native_keep_ratio=float(ns.sadc_min_native_keep_ratio),
            sadc_min_native_conflict_score=float(ns.sadc_min_native_conflict_score),
            sadc_patch_add_percentile=float(ns.sadc_patch_add_percentile),
            sadc_max_patch_fraction=float(ns.sadc_max_patch_fraction),
            sadc_min_patch_anchor_consistency=float(ns.sadc_min_patch_anchor_consistency),
            sadc_anchor_monotonic=bool(ns.sadc_anchor_monotonic),
            sadc_anchor_monotonic_epsilon_px=float(ns.sadc_anchor_monotonic_epsilon_px),
            sadc_activation_mode=str(ns.sadc_activation_mode),
            sadc_activation_min_risk=float(ns.sadc_activation_min_risk),
            sadc_activation_min_sparse_confidence=float(ns.sadc_activation_min_sparse_confidence),
            case_rows_by_image=case_rows.get(scene, {}),
        )
        for scene in scenes
    ]
    macro = _macro_summary(summaries)
    _write_json(output_dir / "summary.json", {"scenes": summaries, "macro": macro})
    print(json.dumps({"output_dir": str(output_dir), "macro": macro}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
