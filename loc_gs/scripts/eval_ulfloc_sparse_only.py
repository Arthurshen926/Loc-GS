#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
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
    if split_name == "test":
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
    return {
        "schema_version": "ulfloc_sparse_only_split_audit_v1",
        "split_name": split,
        "official_test_eval": bool(is_test),
        "official_test_used": bool(is_test),
        "test_split_used": bool(is_test),
        "paper_safe_for_tuning": not bool(is_test),
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


def serialize_sparse_match_attributions(matches: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Export per-correspondence sparse solver traces for offline pruning."""

    landmark_ids = _array_1d(matches, "matched_gaussian_ids", dtype=np.int64)
    inlier_mask = _array_1d(matches, "pnp_inlier_mask", dtype=bool)
    descriptor = _array_1d(matches, "descriptor_score", dtype=np.float64)
    detector = _array_1d(matches, "detector_score", dtype=np.float64)
    reproj = _array_1d(matches, "reprojection_error_px", dtype=np.float64)
    query_xy = np.asarray(matches.get("query_xy", []), dtype=np.float64)
    total = int(max(landmark_ids.size, inlier_mask.size, descriptor.size, detector.size, reproj.size))
    out: list[dict[str, Any]] = []
    for index in range(total):
        row: dict[str, Any] = {}
        if index < landmark_ids.size:
            row["matched_gaussian_id"] = int(landmark_ids[index])
        if index < inlier_mask.size:
            row["pnp_inlier"] = bool(inlier_mask[index])
        if index < descriptor.size and np.isfinite(descriptor[index]):
            row["descriptor_score"] = float(descriptor[index])
        if index < detector.size and np.isfinite(detector[index]):
            row["detector_score"] = float(detector[index])
        if index < reproj.size and np.isfinite(reproj[index]):
            row["reprojection_error_px"] = float(reproj[index])
        if query_xy.ndim == 2 and index < query_xy.shape[0] and query_xy.shape[1] >= 2:
            xy = query_xy[index, :2]
            if np.isfinite(xy).all():
                row["query_xy"] = [float(xy[0]), float(xy[1])]
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
    parser.add_argument("--scene_detector_blend_alpha", default=0.5, type=float)
    parser.add_argument("--scene_detector_candidate_top_k", default=0, type=int)
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
        config["scene_specific_detector"] = {
            "enabled": True,
            "checkpoint": str(Path(args.scene_detector_checkpoint).resolve()),
            "mode": "rerank_superpoint",
            "candidate_top_k": int(args.scene_detector_candidate_top_k)
            if int(args.scene_detector_candidate_top_k) > 0
            else sparse_k * 2,
            "blend_alpha": float(args.scene_detector_blend_alpha),
            "native_keep_fraction": float(args.scene_detector_native_keep_fraction),
            "descriptor_stride": 8,
            "nms_radius": int(config.get("sparse", {}).get("nms", 4)),
            "score_threshold": 0.0,
        }
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
        row = {
            "image_name": str(camera.image_name),
            "sparse_te_cm": float(te_cm),
            "sparse_re_deg": float(re_deg),
            "sparse_inliers": int(result["inliers"]),
            "sparse_inlier_rate": float(result.get("inlier_rate", 0.0)),
            "sparse_match_filter": filter_metadata,
            "scene_matcher": scene_matcher_metadata,
        }
        if bool(args.dump_sparse_match_diagnostics):
            row["sparse_match_diagnostics"] = summarize_sparse_match_diagnostics(result.get("matches", {}))
        if bool(args.dump_sparse_match_attributions):
            row["sparse_match_attributions"] = serialize_sparse_match_attributions(result.get("matches", {}))
        rows.append(row)
    if bool(config["scene_matcher"].get("enabled", False)) and cameras and not scene_matcher_outputs:
        raise RuntimeError(
            "scene_matcher checkpoint was configured, but no query reported scene_matcher_enabled=true; "
            "correspondence supervision did not enter inference"
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
        "scene_matcher": dict(config["scene_matcher"]),
        "scene_matcher_enabled_query_count": int(len(scene_matcher_outputs)),
        "scene_matcher_mean_selected_count": float(np.mean(scene_matcher_outputs))
        if scene_matcher_outputs
        else None,
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
