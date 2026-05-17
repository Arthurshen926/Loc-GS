#!/usr/bin/env python3
from __future__ import annotations

import datetime
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from tqdm import tqdm

from loc_gs.stdloc_native.timing_profile import aggregate_timing_profile, finalize_query_timing, sync_cuda


REPO_ROOT = Path(__file__).resolve().parents[2]
STDLOC_ROOT = REPO_ROOT / "third_party" / "stdloc"


def _prepare_stdloc_imports() -> Any:
    stdloc_path = str(STDLOC_ROOT)
    if stdloc_path not in sys.path:
        sys.path.insert(0, stdloc_path)
    import stdloc as stdloc_module

    return stdloc_module


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _count_sampled_landmarks(model_path: str | Path, config: dict[str, Any]) -> int | None:
    landmark_path = config.get("sparse", {}).get("landmark_path")
    if not landmark_path:
        return None
    path = Path(model_path) / str(landmark_path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        sampled_idx = pickle.load(handle)
    return int(torch.as_tensor(sampled_idx).reshape(-1).numel())


def _resolve_cfg_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    vendored = STDLOC_ROOT / path
    if vendored.exists():
        return vendored
    return path


def _utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _git_text(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return f"git unavailable: {exc}"
    return result.stdout.strip()


def _git_commit() -> str:
    commit = _git_text(["rev-parse", "HEAD"])
    return commit or "unknown"


def _command_for_manifest() -> list[str]:
    return [sys.executable, "-m", "loc_gs.scripts.profile_stdloc_native", *sys.argv[1:]]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selector_manifest_for_model(model_path: str | Path) -> dict[str, Any] | None:
    return _read_json_if_exists(Path(model_path) / "selector_resampling_manifest.json")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64), q))


def _inlier_indices(inliers: Any, match_count: int) -> np.ndarray:
    inlier_array = np.asarray(inliers).reshape(-1)
    if inlier_array.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if inlier_array.dtype == np.bool_ and inlier_array.size == match_count:
        return np.where(inlier_array)[0].astype(np.int64)
    indices = inlier_array.astype(np.int64, copy=False)
    return indices[(indices >= 0) & (indices < match_count)]


def _reprojection_errors_px(
    p2d: Any,
    p3d: Any,
    intrinsic: Any,
    pose_w2c: Any,
) -> np.ndarray:
    points_2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    points_3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    if points_2d.shape[0] != points_3d.shape[0] or points_2d.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    k_mat = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points_3d_h = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    camera_points = (w2c[:3, :] @ points_3d_h.T).T
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (np.abs(z) > 1e-12)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float64)

    projected_h = (k_mat @ camera_points[valid].T).T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    return np.linalg.norm(points_2d[valid] - projected, axis=1)


def compute_pose_reliability_stats(
    p2d: Any,
    p3d: Any,
    intrinsic: Any,
    pose_w2c: Any,
    inliers: Any,
    *,
    solver: str,
    reprojection_error: float,
    confidence: float,
    max_iterations: int,
    min_iterations: int,
) -> dict[str, Any]:
    """Compute profiler-only pose reliability fields without changing STDLoc output."""

    points_2d = np.asarray(p2d).reshape(-1, 2)
    match_count = int(points_2d.shape[0])
    indices = _inlier_indices(inliers, match_count)
    errors = _reprojection_errors_px(p2d, p3d, intrinsic, pose_w2c)
    inlier_errors = errors[indices[indices < errors.shape[0]]] if errors.size else errors
    inlier_count = int(indices.size)
    return {
        "solver": str(solver),
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / match_count) if match_count > 0 else 0.0,
        "success": bool(inlier_count > 0),
        "reprojection_error_px": float(reprojection_error),
        "confidence": float(confidence),
        "max_iterations": int(max_iterations),
        "min_iterations": int(min_iterations),
        "all_reprojection_mean_px": float(errors.mean()) if errors.size else None,
        "all_reprojection_median_px": _percentile(errors, 50.0),
        "all_reprojection_p90_px": _percentile(errors, 90.0),
        "inlier_reprojection_mean_px": float(inlier_errors.mean()) if inlier_errors.size else None,
        "inlier_reprojection_median_px": _percentile(inlier_errors, 50.0),
        "inlier_reprojection_p90_px": _percentile(inlier_errors, 90.0),
    }


def apply_config_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    """Apply explicit profiler-only STDLoc config overrides."""

    sparse_cfg = config.setdefault("sparse", {})
    dense_cfg = config.setdefault("dense", {})
    overrides = {
        "sparse": {
            "max_iterations": _optional_int(getattr(args, "sparse_max_iterations", None)),
            "min_iterations": _optional_int(getattr(args, "sparse_min_iterations", None)),
            "confidence": _optional_float(getattr(args, "sparse_confidence", None)),
            "reprojection_error": _optional_float(getattr(args, "sparse_reprojection_error", None)),
            "landmark_prior_weight": _optional_float(getattr(args, "sparse_landmark_prior_weight", None)),
        },
        "dense": {
            "iters": _optional_int(getattr(args, "dense_iters", None)),
            "max_iterations": _optional_int(getattr(args, "dense_max_iterations", None)),
            "min_iterations": _optional_int(getattr(args, "dense_min_iterations", None)),
            "confidence": _optional_float(getattr(args, "dense_confidence", None)),
            "reprojection_error": _optional_float(getattr(args, "dense_reprojection_error", None)),
            "locability_prior_weight": _optional_float(getattr(args, "dense_locability_prior_weight", None)),
        },
    }
    for key, value in overrides["sparse"].items():
        if value is not None:
            sparse_cfg[key] = value
    for key, value in overrides["dense"].items():
        if value is not None:
            dense_cfg[key] = value
    return overrides


def build_profile_manifest(
    *,
    args: Any,
    dataset: Any,
    config: Mapping[str, Any],
    scene_name: str,
    method_name: str,
    cfg_path: str | Path,
    output_path: str | Path,
    command: list[str],
    selector_manifest: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build the required audit manifest for a STDLoc profiling run."""

    dense_cfg = config.get("dense", {}) if isinstance(config, Mapping) else {}
    sparse_cfg = config.get("sparse", {}) if isinstance(config, Mapping) else {}
    selector_payload = dict(selector_manifest or {})
    checkpoint_path = selector_payload.get("checkpoint_path")
    return {
        "git_commit": git_commit or _git_commit(),
        "command": list(command),
        "scene": str(scene_name),
        "split": str(getattr(args, "eval_split", "unknown")),
        "checkpoint_path": checkpoint_path,
        "map_path": str(dataset.model_path),
        "data_roots": [str(dataset.source_path)],
        "hyperparameters": {
            "method": str(method_name),
            "cfg_path": str(cfg_path),
            "output_path": str(output_path),
            "iteration": int(getattr(args, "iteration", -1)),
            "images": str(getattr(dataset, "images", "")),
            "data_device": str(getattr(args, "data_device", "")),
            "feature_type": str(getattr(dataset, "feature_type", "")),
            "gaussian_type": str(getattr(dataset, "gaussian_type", "")),
            "longest_edge": getattr(dataset, "longest_edge", None),
            "norm_before_render": bool(getattr(dataset, "norm_before_render", False)),
            "landmark_path": sparse_cfg.get("landmark_path"),
            "dense_iters": dense_cfg.get("iters"),
            "sparse_max_iterations": sparse_cfg.get("max_iterations"),
            "sparse_min_iterations": sparse_cfg.get("min_iterations"),
            "sparse_confidence": sparse_cfg.get("confidence"),
            "sparse_reprojection_error": sparse_cfg.get("reprojection_error"),
            "sparse_landmark_prior_weight": sparse_cfg.get("landmark_prior_weight"),
            "dense_max_iterations": dense_cfg.get("max_iterations"),
            "dense_min_iterations": dense_cfg.get("min_iterations"),
            "dense_confidence": dense_cfg.get("confidence"),
            "dense_reprojection_error": dense_cfg.get("reprojection_error"),
            "dense_locability_prior_weight": dense_cfg.get("locability_prior_weight"),
            "test_stride": int(getattr(args, "test_stride", 1)),
            "max_test_cameras": getattr(args, "max_test_cameras", None),
            "warmup_cameras": int(getattr(args, "warmup_cameras", 0) or 0),
            "config_overrides": getattr(args, "_config_overrides", None),
        },
        "timestamp": timestamp or _utc_timestamp(),
        "feedback": {
            "residual": False,
            "selector": False,
            "rho": False,
        },
        "selector_resampling": selector_payload,
    }


def _profile_split_audit(
    *,
    scene_name: str,
    method_name: str,
    split: str,
    query_count: int,
    warmup_cameras: int,
) -> dict[str, Any]:
    return {
        "scene": str(scene_name),
        "method": str(method_name),
        "split": str(split),
        "query_count": int(query_count),
        "warmup_cameras": int(warmup_cameras),
        "status": "unknown",
        "paper_safe": False,
        "reason": (
            "Profiling output records the evaluator split but does not audit "
            "self-map/calibration image disjointness. Treat as diagnostic until "
            "a full split audit is attached."
        ),
    }


def _write_profile_audit_artifacts(
    *,
    output_path: str | Path,
    manifest: Mapping[str, Any],
    metrics_summary: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> None:
    out = Path(output_path)
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=4, default=_json_default)
    with (out / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, indent=4, default=_json_default)
    with (out / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(split_audit, handle, indent=4, default=_json_default)
    command = manifest.get("command", [])
    command_text = " ".join(shlex.quote(str(part)) for part in command)
    (out / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    (out / "git_status.txt").write_text(_git_text(["status", "--short"]) + "\n", encoding="utf-8")


def _timed_call(row: dict[str, Any], key: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    sync_cuda()
    start = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        sync_cuda()
        row[key] = float(row.get(key, 0.0)) + (time.perf_counter() - start) * 1000.0


def _attach_profiler(stdloc_module: Any, stdloc: Any) -> tuple[list[dict[str, Any]], Any]:
    rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {"row": None, "stage_stack": []}
    original_solve_pose = stdloc_module.solve_pose
    original_get_feature_map = stdloc.get_feature_map
    original_loc_sparse = stdloc.loc_sparse
    original_loc_dense = stdloc.loc_dense

    def timed_solve_pose(*args: Any, **kwargs: Any) -> Any:
        row = state.get("row")
        stage = state["stage_stack"][-1] if state["stage_stack"] else "unknown"
        key = f"{stage}_pose_ms" if stage in {"sparse", "dense"} else "pose_ms"
        if row is None:
            return original_solve_pose(*args, **kwargs)
        result = _timed_call(row, key, original_solve_pose, *args, **kwargs)
        if stage in {"sparse", "dense"}:
            try:
                pose_w2c, inliers = result
                solver = kwargs.get("solver", args[3] if len(args) > 3 else "unknown")
                reprojection_error = kwargs.get("reprojection_error", args[4] if len(args) > 4 else 0.0)
                confidence = kwargs.get("confidence", args[5] if len(args) > 5 else 0.0)
                max_iterations = kwargs.get("max_iterations", args[6] if len(args) > 6 else 0)
                min_iterations = kwargs.get("min_iterations", args[7] if len(args) > 7 else 0)
                row[f"{stage}_pose_reliability"] = compute_pose_reliability_stats(
                    args[0],
                    args[1],
                    args[2],
                    pose_w2c,
                    inliers,
                    solver=str(solver),
                    reprojection_error=float(reprojection_error),
                    confidence=float(confidence),
                    max_iterations=int(max_iterations),
                    min_iterations=int(min_iterations),
                )
            except Exception as exc:  # pragma: no cover - profiler metadata must not affect localization.
                row[f"{stage}_pose_reliability_error"] = str(exc)
        return result

    def timed_get_feature_map(*args: Any, **kwargs: Any) -> Any:
        row = state.get("row")
        if row is None:
            return original_get_feature_map(*args, **kwargs)
        return _timed_call(row, "feature_ms", original_get_feature_map, *args, **kwargs)

    def timed_loc_sparse(*args: Any, **kwargs: Any) -> Any:
        row = state.get("row")
        if row is None:
            return original_loc_sparse(*args, **kwargs)
        state["stage_stack"].append("sparse")
        try:
            return _timed_call(row, "sparse_total_ms", original_loc_sparse, *args, **kwargs)
        finally:
            state["stage_stack"].pop()

    def timed_loc_dense(*args: Any, **kwargs: Any) -> Any:
        row = state.get("row")
        if row is None:
            return original_loc_dense(*args, **kwargs)
        row["dense_iterations"] = int(row.get("dense_iterations", 0)) + 1
        state["stage_stack"].append("dense")
        try:
            return _timed_call(row, "dense_total_ms", original_loc_dense, *args, **kwargs)
        finally:
            state["stage_stack"].pop()

    stdloc_module.solve_pose = timed_solve_pose
    stdloc.get_feature_map = timed_get_feature_map
    stdloc.loc_sparse = timed_loc_sparse
    stdloc.loc_dense = timed_loc_dense

    def localize_with_profile(image_name: str, query_image: torch.Tensor, fovx: float, fovy: float) -> Any:
        row: dict[str, Any] = {
            "image_name": str(image_name),
            "feature_ms": 0.0,
            "sparse_total_ms": 0.0,
            "sparse_pose_ms": 0.0,
            "dense_total_ms": 0.0,
            "dense_pose_ms": 0.0,
            "dense_iterations": 0,
        }
        state["row"] = row
        sync_cuda()
        start = time.perf_counter()
        try:
            return stdloc.localize(query_image, fovx, fovy)
        finally:
            sync_cuda()
            row["total_ms"] = (time.perf_counter() - start) * 1000.0
            rows.append(finalize_query_timing(row))
            state["row"] = None

    def restore() -> None:
        stdloc_module.solve_pose = original_solve_pose
        stdloc.get_feature_map = original_get_feature_map
        stdloc.loc_sparse = original_loc_sparse
        stdloc.loc_dense = original_loc_dense

    return rows, (localize_with_profile, restore)


def _pose_summary(aes: list[float], tes: list[float], inliers: list[int]) -> dict[str, Any]:
    aes_np = np.array(aes)
    tes_np = np.array(tes)
    return {
        "median_ae": np.median(aes_np),
        "median_te": np.median(tes_np),
        "recall_5m_10d": ((aes_np <= 10) & (tes_np <= 500)).sum() / len(aes_np),
        "recall_2m_5d": ((aes_np <= 5) & (tes_np <= 200)).sum() / len(aes_np),
        "recall_10cm_5d": ((aes_np <= 5) & (tes_np <= 10)).sum() / len(aes_np),
        "recall_5cm_5d": ((aes_np <= 5) & (tes_np <= 5)).sum() / len(aes_np),
        "recall_2cm_2d": ((aes_np <= 2) & (tes_np <= 2)).sum() / len(aes_np),
        "avg_inliers": np.array(inliers).mean(),
    }


def build_argparser() -> ArgumentParser:
    stdloc_module = _prepare_stdloc_imports()
    parser = ArgumentParser(description="Profile native STDLoc evaluation without editing vendored evaluator files.")
    stdloc_module.ModelParams(parser, sentinel=True)
    stdloc_module.PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--prefix", default=None, type=str)
    parser.add_argument("--max_test_cameras", default=None, type=int)
    parser.add_argument("--test_stride", default=1, type=int)
    parser.add_argument("--eval_split", choices=["test", "train"], default="test")
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--scene_name", default="", type=str)
    parser.add_argument("--method_name", default="", type=str)
    parser.add_argument("--warmup_cameras", default=0, type=int)
    parser.add_argument("--sparse_max_iterations", default=None, type=int)
    parser.add_argument("--sparse_min_iterations", default=None, type=int)
    parser.add_argument("--sparse_confidence", default=None, type=float)
    parser.add_argument("--sparse_reprojection_error", default=None, type=float)
    parser.add_argument("--sparse_landmark_prior_weight", default=None, type=float)
    parser.add_argument("--dense_iters", default=None, type=int)
    parser.add_argument("--dense_max_iterations", default=None, type=int)
    parser.add_argument("--dense_min_iterations", default=None, type=int)
    parser.add_argument("--dense_confidence", default=None, type=float)
    parser.add_argument("--dense_reprojection_error", default=None, type=float)
    parser.add_argument("--dense_locability_prior_weight", default=None, type=float)
    return parser


def _extract_model_params(stdloc_module: Any, args: Any) -> Any:
    parser = ArgumentParser(add_help=False)
    model = stdloc_module.ModelParams(parser, sentinel=True)
    return model.extract(args)


def main() -> int:
    stdloc_module = _prepare_stdloc_imports()
    parser = build_argparser()
    args = stdloc_module.get_combined_args(parser)
    args.eval = args.eval_split == "test"

    if args.output_path:
        output_path = args.output_path
    elif hasattr(args, "prefix") and args.prefix:
        output_path = (
            f"results/{args.prefix}-{args.model_path.replace('/', '_')}-"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        output_path = f"results/{args.model_path.replace('/', '_')}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("Output path:", output_path)
    os.makedirs(output_path, exist_ok=True)

    dataset = _extract_model_params(stdloc_module, args)
    if dataset.gaussian_type == "3dgs":
        gaussians = stdloc_module.GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = stdloc_module.GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError("Gaussian type not supported")

    scene = stdloc_module.Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
        dataloader_num_workers=0,
        pin_memory=False,
    )

    cfg_path = _resolve_cfg_path(args.cfg)
    config = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
    args._config_overrides = apply_config_overrides(config, args)
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    yaml.dump(config, open(os.path.join(output_path, cfg_path.name), "w"))

    stdloc = stdloc_module.STDLoc(gaussians, config)
    timing_rows, profiler = _attach_profiler(stdloc_module, stdloc)
    localize_with_profile, restore_profiler = profiler

    eval_cameras = scene.getTrainCameras() if args.eval_split == "train" else scene.getTestCameras()
    warmup_cameras = max(0, int(getattr(args, "warmup_cameras", 0) or 0))
    requested_max = getattr(args, "max_test_cameras", None)
    select_max = None
    if requested_max is not None and int(requested_max) > 0:
        select_max = int(requested_max) + warmup_cameras
    test_cameras = stdloc_module.select_eval_cameras(
        eval_cameras,
        max_cameras=select_max,
        stride=getattr(args, "test_stride", 1),
    )

    results = []
    sparse_aes: list[float] = []
    sparse_tes: list[float] = []
    sparse_inliers: list[int] = []
    dense_aes: list[float] = []
    dense_tes: list[float] = []
    dense_inliers: list[int] = []

    try:
        for idx, camera_info in enumerate(tqdm(test_cameras, desc="STDLoc profile")):
            print("\nLocalize image:", camera_info.image_name)
            gt_w2c = camera_info.world_view_transform.transpose(0, 1).cpu().numpy()
            query_image = camera_info.original_image.to("cuda")
            fovx = camera_info.FoVx
            fovy = camera_info.FoVy

            if idx < warmup_cameras:
                _ = stdloc.localize(query_image, fovx, fovy)
                continue

            loc_res = localize_with_profile(camera_info.image_name, query_image, fovx, fovy)

            sparse_ae, sparse_te = stdloc_module.cal_pose_error(loc_res["sparse"]["pose_w2c"], gt_w2c)
            sparse_aes.append(float(sparse_ae))
            sparse_tes.append(float(sparse_te))
            sparse_inliers.append(int(loc_res["sparse"]["inliers"]))
            loc_res["sparse_AE"] = sparse_ae
            loc_res["sparse_TE"] = sparse_te

            dense_ae, dense_te = stdloc_module.cal_pose_error(loc_res["dense"][-1]["pose_w2c"], gt_w2c)
            dense_aes.append(float(dense_ae))
            dense_tes.append(float(dense_te))
            dense_inliers.append(int(loc_res["dense"][-1]["inliers"]))
            print(f"AE: {dense_ae:.3f}deg, TE: {dense_te:.3f}cm, inliers: {loc_res['dense'][-1]['inliers']}")

            loc_res["gt_pose_w2c"] = gt_w2c.tolist()
            loc_res["dense_AE"] = dense_ae
            loc_res["dense_TE"] = dense_te
            loc_res["image_name"] = camera_info.image_name
            results.append(loc_res)
    finally:
        restore_profiler()

    scene_name = args.scene_name or Path(dataset.source_path).name
    method_name = args.method_name or Path(dataset.model_path).name
    landmark_count = _count_sampled_landmarks(dataset.model_path, config)
    timing_profile = aggregate_timing_profile(
        timing_rows,
        scene=scene_name,
        method=method_name,
        landmark_count=landmark_count,
        dense_iterations=int(config.get("dense", {}).get("iters", 0)),
    )
    timing_profile["eval_split"] = args.eval_split
    timing_profile["test_stride"] = int(getattr(args, "test_stride", 1))
    timing_profile["max_test_cameras"] = getattr(args, "max_test_cameras", None)
    timing_profile["warmup_cameras"] = warmup_cameras
    selector_manifest = _selector_manifest_for_model(dataset.model_path)

    results_summary = {
        "model_path": dataset.model_path,
        "scene": scene_name,
        "method": method_name,
        "eval_split": args.eval_split,
        "test_stride": int(getattr(args, "test_stride", 1)),
        "max_test_cameras": getattr(args, "max_test_cameras", None),
        "warmup_cameras": warmup_cameras,
        "landmark_count": landmark_count,
        "sparse": _pose_summary(sparse_aes, sparse_tes, sparse_inliers),
        "dense": _pose_summary(dense_aes, dense_tes, dense_inliers),
        "timing_profile": {
            "latency_ms": timing_profile["latency_ms"],
            "fps": timing_profile["fps"],
            "queries": timing_profile["queries"],
        },
    }
    manifest = build_profile_manifest(
        args=args,
        dataset=dataset,
        config=config,
        scene_name=scene_name,
        method_name=method_name,
        cfg_path=cfg_path,
        output_path=output_path,
        command=_command_for_manifest(),
        selector_manifest=selector_manifest,
    )
    split_audit = _profile_split_audit(
        scene_name=scene_name,
        method_name=method_name,
        split=args.eval_split,
        query_count=timing_profile["queries"],
        warmup_cameras=warmup_cameras,
    )

    print("Result Summary:")
    print(json.dumps(results_summary, indent=4, default=_json_default))

    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(results_summary, handle, indent=4, default=_json_default)
    with open(os.path.join(output_path, "timing_profile.json"), "w", encoding="utf-8") as handle:
        json.dump(timing_profile, handle, indent=4, default=_json_default)
    _write_profile_audit_artifacts(
        output_path=output_path,
        manifest=manifest,
        metrics_summary=results_summary,
        split_audit=split_audit,
    )

    for item in results:
        item["sparse"]["pose_w2c"] = item["sparse"]["pose_w2c"].tolist()
        for dense_item in item["dense"]:
            dense_item["pose_w2c"] = dense_item["pose_w2c"].tolist()
    with open(os.path.join(output_path, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4, default=_json_default)

    print("Result and timing profile are saved in", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
