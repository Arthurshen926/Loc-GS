#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from loc_gs.localization.ulfloc_scene_match_cache import build_ulfloc_scene_matcher_pair_cache
from loc_gs.localization.ulfloc_sparse_match_filter import prepare_selected_landmark_prior_weights
from loc_gs.scripts.eval_ulfloc_sparse_only import (
    _apply_masks,
    _import_ulfloc,
    _load_masks,
    load_landmark_prior_weights,
)
from loc_gs.scripts.export_ulfloc_sparse_feedback import resolve_ulfloc_source_path_for_loader


def validate_split_name(split_name: str) -> str:
    split = str(split_name).strip()
    if not split:
        raise ValueError("split_name is required")
    if split.lower() == "test":
        raise ValueError("refusing to build scene matcher cache from test split")
    return split


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repeat_row_source_view_ids(image_row_counts: list[tuple[str, int]]) -> list[str]:
    row_ids: list[str] = []
    for image_name, row_count in image_row_counts:
        count = int(row_count)
        if count < 0:
            raise ValueError("row_count must be non-negative")
        row_ids.extend([str(image_name)] * count)
    return row_ids


def _dataset_namespace_for_pair_cache(args: argparse.Namespace, config: Mapping[str, Any]) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=str(args.source_path),
        feature_type=str(args.feature_type),
        gaussian_type=str(args.gaussian_type),
        model_path=str(Path(args.model_path).resolve()),
        images=str(args.images),
        resolution=-1,
        white_background=True,
        longest_edge=int(args.longest_edge),
        data_device=str(args.data_device),
        eval=str(args.camera_split) != "train",
        speedup=False,
        norm_before_render=bool(config.get("dense", {}).get("norm_before_render", True))
        if isinstance(config.get("dense", {}), Mapping)
        else True,
        render_items=["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
    )


def _project_reprojection_errors(
    *,
    p3d: np.ndarray,
    keypoint_xy: np.ndarray,
    camera: Any,
    K: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = p3d.reshape(-1, 3)
    gt_R = camera.R
    gt_t = camera.T
    projected = cv2.projectPoints(flat, gt_R.T, gt_t.T, K, None)[0].reshape(*p3d.shape[:2], 2)
    target = keypoint_xy[:, None, :] + 0.5
    errors = np.linalg.norm(projected - target, axis=-1)
    cam_xyz = (gt_R.T @ flat.T).T + gt_t.reshape(1, 3)
    depth = cam_xyz[:, 2].reshape(p3d.shape[:2])
    visible = (
        np.isfinite(errors)
        & (depth > 0.0)
        & (projected[..., 0] >= 0.0)
        & (projected[..., 0] < float(image_width))
        & (projected[..., 1] >= 0.0)
        & (projected[..., 1] < float(image_height))
    )
    return torch.as_tensor(errors, dtype=torch.float32), torch.as_tensor(visible, dtype=torch.bool)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ULF-Loc listwise scene matcher top-K pair cache.")
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_log_dir", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera_split", choices=["test", "train"], default="test")
    parser.add_argument("--topk", default=4, type=int)
    parser.add_argument("--reprojection_threshold_px", default=4.0, type=float)
    parser.add_argument("--scene_detector_checkpoint", default=None, type=Path)
    parser.add_argument("--scene_detector_blend_alpha", default=0.5, type=float)
    parser.add_argument("--scene_detector_candidate_top_k", default=0, type=int)
    parser.add_argument("--scene_detector_native_keep_fraction", default=0.0, type=float)
    parser.add_argument("--landmark_prior", default=None, type=Path)
    parser.add_argument("--max_queries", default=0, type=int)
    parser.add_argument("--query_id", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = validate_split_name(str(args.split_name))
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imported = _import_ulfloc(Path(args.ulf_root))
    config = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("ULF config must be a YAML mapping")
    config["longest_edge"] = int(args.longest_edge)
    config["model_path"] = str(Path(args.model_path).resolve())
    config["scene_matcher"] = {"enabled": False}
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

    original_source_path = Path(args.source_path)
    loader_source_path = resolve_ulfloc_source_path_for_loader(
        original_source_path,
        output_dir / "_ulf_loader_links",
    )
    args = argparse.Namespace(**vars(args))
    args.source_path = loader_source_path
    dataset = _dataset_namespace_for_pair_cache(args, config)
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    landmark_features = F.normalize(ulfloc.landmarks._loc_feature.squeeze().to(device).float(), dim=-1)
    sampled_idx = torch.as_tensor(ulfloc.sampled_idx, dtype=torch.long).reshape(-1)
    selected_prior = None
    if args.landmark_prior is not None:
        selected_prior = prepare_selected_landmark_prior_weights(
            load_landmark_prior_weights(Path(args.landmark_prior)),
            sampled_idx=sampled_idx,
            selected_count=int(landmark_features.shape[0]),
        )

    if str(args.camera_split) == "train":
        cameras = list(scene.getTrainCameras())
    else:
        cameras = list(scene.getTestCameras())
    requested = set(str(item) for item in (args.query_id or []))
    if requested:
        cameras = [camera for camera in cameras if str(camera.image_name) in requested]
    if int(args.max_queries) > 0:
        cameras = cameras[: int(args.max_queries)]

    chunks: dict[str, list[torch.Tensor]] = {
        "query_desc": [],
        "landmark_desc": [],
        "cosine": [],
        "margin": [],
        "query_score": [],
        "landmark_prior": [],
        "landmark_id": [],
        "candidate_mask": [],
        "reprojection_error": [],
        "label": [],
    }
    processed = 0
    positive = 0
    total_keypoints = 0
    processed_image_names: list[str] = []
    image_row_counts: list[tuple[str, int]] = []
    topk = max(1, int(args.topk))
    for camera in tqdm(cameras, desc="ULF scene matcher cache"):
        query_image = camera.original_image.to(device)
        query_image = _apply_masks(query_image, masks, camera.image_name)
        keypoints, keypoint_score, query_features = ulfloc.detect_sparse_keypoints(query_image)
        if int(query_features.numel()) == 0:
            continue
        query_features = F.normalize(query_features.to(device).float(), dim=-1)
        corr = torch.matmul(query_features, landmark_features.T)
        if config["sparse"]["dual_softmax"] is True:
            from utils.loc_utils import dual_softmax  # type: ignore

            corr = dual_softmax(corr_matrix=corr, temp=config["sparse"]["dual_softmax_temp"])
        k = min(topk, int(corr.shape[1]))
        top_values, top_ids = torch.topk(corr, k=k, dim=1)
        p3d = ulfloc.landmarks.get_xyz[top_ids.reshape(-1)].detach().cpu().numpy().reshape(int(top_ids.shape[0]), k, 3)
        H, W = query_image.shape[-2:]
        K = np.asarray(
            [
                [float(W) / (2.0 * np.tan(float(camera.FoVx) / 2.0)), 0.0, float(W) / 2.0],
                [0.0, float(H) / (2.0 * np.tan(float(camera.FoVy) / 2.0)), float(H) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        reproj, visible = _project_reprojection_errors(
            p3d=p3d,
            keypoint_xy=keypoints[:, :2].detach().cpu().numpy(),
            camera=camera,
            K=K,
            image_width=int(W),
            image_height=int(H),
        )
        prior = None
        if selected_prior is not None:
            prior = selected_prior[top_ids.detach().cpu()]
        cache = build_ulfloc_scene_matcher_pair_cache(
            query_desc=query_features.detach().cpu(),
            landmark_desc=landmark_features.detach().cpu(),
            candidate_landmark_ids=top_ids.detach().cpu(),
            candidate_cosine=top_values.detach().cpu(),
            keypoint_xy=keypoints[:, :2].detach().cpu(),
            candidate_reprojection_error=reproj,
            candidate_visible=visible & (top_values.detach().cpu() > float(config["sparse"]["threshold"])),
            query_score=keypoint_score.detach().cpu(),
            landmark_prior=prior,
            base_gaussian_ids=sampled_idx.detach().cpu(),
            reprojection_threshold_px=float(args.reprojection_threshold_px),
        )
        for key in chunks:
            chunks[key].append(torch.as_tensor(cache[key]))
        query_count = int(cache["metadata"]["query_count"])
        processed += 1
        total_keypoints += query_count
        positive += int(cache["metadata"]["positive_query_count"])
        processed_image_names.append(str(camera.image_name))
        image_row_counts.append((str(camera.image_name), query_count))

    if processed == 0:
        raise ValueError("no cameras produced scene matcher cache entries")
    payload = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    payload["base_landmark_desc"] = landmark_features.detach().cpu()
    payload["base_gaussian_id"] = sampled_idx.detach().cpu()
    payload["processed_image_names"] = list(processed_image_names)
    payload["row_source_view_id"] = _repeat_row_source_view_ids(image_row_counts)
    payload["metadata"] = {
        "format": "ulfloc_scene_matcher_listwise_pair_cache_v1",
        "scene": str(args.scene),
        "split_name": split,
        "processed_images": int(processed),
        "query_keypoint_count": int(total_keypoints),
        "positive_query_count": int(positive),
        "dustbin_query_count": int(total_keypoints - positive),
        "topk": int(topk),
        "reprojection_threshold_px": float(args.reprojection_threshold_px),
        "scene_detector_enabled": args.scene_detector_checkpoint is not None,
        "landmark_prior_path": None if args.landmark_prior is None else str(Path(args.landmark_prior)),
        "row_source_view_id_available": True,
        "camera_split": str(args.camera_split),
    }
    torch.save(payload, output_dir / "pair_cache.pt")
    split_audit = {
        "schema_version": "ulfloc_scene_matcher_pair_cache_split_audit_v1",
        "split_name": split,
        "test_split_used": False,
        "official_test_used": False,
        "source": "ULF-Loc self-map/train-dev overlay cameras",
        "camera_split": str(args.camera_split),
    }
    manifest = {
        "schema_version": "ulfloc_scene_matcher_pair_cache_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "source_path": str(args.source_path),
        "original_source_path": str(original_source_path),
        "model_path": str(args.model_path),
        "input_log_dir": str(args.input_log_dir),
        "cfg": str(args.cfg),
        "output_dir": str(output_dir),
        "metrics": payload["metadata"],
        "split_audit": split_audit,
    }
    _write_json(output_dir / "metrics_summary.json", payload["metadata"])
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps(payload["metadata"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
