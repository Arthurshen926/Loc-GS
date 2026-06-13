"""Diagnostic sparse oracle experiments for ULF-Loc.

This script is intentionally diagnostic-only: it can use official test poses to
measure oracle upper bounds, but those outputs must not be used for training,
model selection, or paper-facing tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


@dataclass(frozen=True)
class PoseResult:
    pose_w2c: np.ndarray
    inliers: int
    ae_deg: float
    te_cm: float
    match_count: int


def _insert_ulfloc_path(ulfloc_root: Path) -> None:
    root = str(ulfloc_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_intrinsic(fovx: float, fovy: float, width: int, height: int) -> np.ndarray:
    from utils.graphics_utils import fov2focal  # type: ignore

    return np.array(
        [
            [fov2focal(fovx, width), 0.0, width / 2.0],
            [0.0, fov2focal(fovy, height), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _project(points_3d: np.ndarray, pose_w2c: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)], axis=1)
    cam = (pose_w2c[:3, :] @ xyz_h.T).T
    z = cam[:, 2]
    uv = (intrinsic @ cam.T).T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-8)
    return uv.astype(np.float32), z.astype(np.float32)


def _pose_from_rt(camera: Any) -> np.ndarray:
    return camera.world_view_transform.transpose(0, 1).cpu().numpy().astype(np.float32)


def _solve_and_score(
    p2d: np.ndarray,
    p3d: np.ndarray,
    intrinsic: np.ndarray,
    gt_w2c: np.ndarray,
    config: dict[str, Any],
) -> PoseResult:
    from utils.pose_utils import cal_pose_error, solve_pose  # type: ignore

    if p2d.shape[0] < 4:
        pose = np.eye(4, dtype=np.float32)
        ae, te = cal_pose_error(pose, gt_w2c)
        return PoseResult(pose, 0, float(ae), float(te), int(p2d.shape[0]))
    pose, inliers = solve_pose(
        p2d + 0.5,
        p3d,
        intrinsic,
        config["sparse"]["solver"],
        config["sparse"]["reprojection_error"],
        config["sparse"]["confidence"],
        config["sparse"]["max_iterations"],
        config["sparse"]["min_iterations"],
    )
    ae, te = cal_pose_error(pose, gt_w2c)
    return PoseResult(pose, int(np.asarray(inliers).reshape(-1).shape[0]), float(ae), float(te), int(p2d.shape[0]))


def _summarize(rows: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"query_count": len(rows), "variants": {}}
    for variant in variants:
        tes = np.array([float(row[variant]["te_cm"]) for row in rows], dtype=np.float64)
        aes = np.array([float(row[variant]["ae_deg"]) for row in rows], dtype=np.float64)
        counts = np.array([int(row[variant]["match_count"]) for row in rows], dtype=np.float64)
        inliers = np.array([int(row[variant]["inliers"]) for row in rows], dtype=np.float64)
        summary["variants"][variant] = {
            "median_te_cm": float(np.median(tes)) if tes.size else math.nan,
            "median_re_deg": float(np.median(aes)) if aes.size else math.nan,
            "R50cm5deg": float(np.mean((tes <= 50.0) & (aes <= 5.0))) if tes.size else math.nan,
            "R10cm5deg": float(np.mean((tes <= 10.0) & (aes <= 5.0))) if tes.size else math.nan,
            "R5cm5deg": float(np.mean((tes <= 5.0) & (aes <= 5.0))) if tes.size else math.nan,
            "avg_match_count": float(np.mean(counts)) if counts.size else math.nan,
            "avg_inliers": float(np.mean(inliers)) if inliers.size else math.nan,
        }
    return summary


def _load_dense_pose_rows(path: Path | None) -> list[np.ndarray]:
    if path is None or not path.exists():
        return []
    rows = json.loads(path.read_text())
    poses: list[np.ndarray] = []
    for row in rows:
        dense = row.get("dense")
        if isinstance(dense, list) and dense:
            poses.append(np.asarray(dense[-1]["pose_w2c"], dtype=np.float32))
    return poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulfloc-root", type=Path, default=Path("/root/ULF-Loc-clean"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--artifact-log-name", default="test")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--gt-threshold-px", type=float, default=5.0)
    parser.add_argument("--pose-threshold-px", type=float, default=8.0)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--max-per-cell", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dense-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _insert_ulfloc_path(args.ulfloc_root)
    os.chdir(args.ulfloc_root)

    from argparse import ArgumentParser

    from arguments import ModelParams, PipelineParams, get_combined_args  # type: ignore
    from encoders.feature_extractor import FeatureExtractor  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore
    from utils.mask_utils import mask_to_image_size  # type: ignore

    ulf_parser = ArgumentParser()
    model_params = ModelParams(ulf_parser, sentinel=True)
    PipelineParams(ulf_parser)
    ulf_parser.add_argument("--iteration", default=-1, type=int)
    ulf_parser.add_argument("--cfg", default=None, type=str)
    old_argv = sys.argv
    sys.argv = [
        "diagnose_ulfloc_sparse_oracles",
        "-s",
        str(args.source),
        "-m",
        str(args.model),
        "--images",
        args.images,
        "--data_device",
        "cpu",
        "--iteration",
        str(args.iteration),
        "--cfg",
        str(args.cfg),
    ]
    try:
        ulf_args = get_combined_args(ulf_parser)
    finally:
        sys.argv = old_argv
    ulf_args.eval = True

    dataset = model_params.extract(ulf_args)
    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError(f"unsupported Gaussian type: {dataset.gaussian_type}")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, preload_cameras=False)

    config = yaml.safe_load(args.cfg.read_text())
    config["longest_edge"] = dataset.longest_edge
    artifact_dir = args.model / args.artifact_log_name
    sampled_idx = pickle.load(open(artifact_dir / config["sample"]["landmark_file_name"], "rb"))
    landmark_features = pickle.load(open(artifact_dir / config["sample"]["kpts_feature_file_name"], "rb"))
    sampled_idx_t = torch.as_tensor(sampled_idx, dtype=torch.long, device="cuda")
    landmark_xyz = gaussians.get_xyz[sampled_idx_t].detach()
    landmark_features = F.normalize(torch.as_tensor(landmark_features, dtype=torch.float32, device="cuda").squeeze(), dim=-1)

    feature_extractor = FeatureExtractor(config["feature_type"]).cuda().eval()

    masks = None
    masks_path = Path(dataset.images) / "masks.pkl"
    if masks_path.exists():
        masks = pickle.load(open(masks_path, "rb"))

    dense_pose_rows = _load_dense_pose_rows(args.dense_results)
    cameras = scene.getTestCameras()
    dense_pose_map: dict[str, np.ndarray] = {}
    if dense_pose_rows:
        for camera, pose in zip(cameras, dense_pose_rows):
            dense_pose_map[str(camera.image_name)] = pose
    selected = [cam for i, cam in enumerate(cameras) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        selected = selected[: args.limit]

    rows: list[dict[str, Any]] = []
    variants = [
        "native_top1",
        "gt_top1_good",
        "gt_topk_good",
        "dense_pose_topk_consistent",
        "sparse_pose_topk_consistent",
        "sparse_pose_grid_consistent",
    ]

    for camera in selected:
        query_image = camera.original_image.to("cuda")
        if masks is not None:
            with torch.no_grad():
                mask_hw = (query_image.shape[1], query_image.shape[2])
                obj_mask = mask_to_image_size(masks[camera.image_name][0], mask_hw, query_image.device)[None]
                sky_mask = mask_to_image_size(masks[camera.image_name][1], mask_hw, query_image.device)[None]
                distort_mask = mask_to_image_size(masks[camera.image_name][2], mask_hw, query_image.device)[None]
                mask = obj_mask & distort_mask
                query_image = query_image * mask
                query_image[sky_mask.repeat(3, 1, 1) == False] = 0

        with torch.no_grad():
            kpts, scores, qfeat = feature_extractor.detectAndCompute(
                query_image[None],
                top_k=int(config["sparse"]["kpts_num"]),
            )[0].values()
            qfeat = F.normalize(qfeat, dim=-1)
            corr = torch.matmul(qfeat, landmark_features.T)
            vals, idx = torch.topk(corr, int(args.topk), dim=1)
            n_keypoints = int(kpts.shape[0])
            im_idx_topk = torch.arange(n_keypoints, device="cuda").repeat_interleave(int(args.topk))
            gs_idx_topk = idx.reshape(-1)
            score_topk = vals.reshape(-1)
            p2d_topk = kpts[im_idx_topk, :2].detach().cpu().numpy().astype(np.float32)
            p3d_topk = landmark_xyz[gs_idx_topk].detach().cpu().numpy().astype(np.float32)
            im_idx_topk_np = im_idx_topk.detach().cpu().numpy()
            score_topk_np = score_topk.detach().cpu().numpy()

        h, w = query_image.shape[-2:]
        intrinsic = _get_intrinsic(camera.FoVx, camera.FoVy, w, h)
        gt_w2c = _pose_from_rt(camera)
        gt_uv, gt_z = _project(p3d_topk, gt_w2c, intrinsic)
        gt_err = np.linalg.norm(gt_uv - p2d_topk, axis=1)
        in_img = (
            (gt_uv[:, 0] >= 0)
            & (gt_uv[:, 0] < w)
            & (gt_uv[:, 1] >= 0)
            & (gt_uv[:, 1] < h)
            & (gt_z > 0)
        )
        gt_good = (gt_err <= float(args.gt_threshold_px)) & in_img
        top1 = (np.arange(p2d_topk.shape[0]) % int(args.topk)) == 0

        row: dict[str, Any] = {
            "image_name": camera.image_name,
            "candidate_count": int(p2d_topk.shape[0]),
            "gt_good_top1_count": int(np.sum(gt_good & top1)),
            "gt_good_topk_count": int(np.sum(gt_good)),
        }

        def run_variant(name: str, keep: np.ndarray) -> PoseResult:
            keep = np.asarray(keep, dtype=bool)
            result = _solve_and_score(p2d_topk[keep], p3d_topk[keep], intrinsic, gt_w2c, config)
            row[name] = {
                "te_cm": result.te_cm,
                "ae_deg": result.ae_deg,
                "inliers": result.inliers,
                "match_count": result.match_count,
            }
            return result

        native = run_variant("native_top1", top1)
        run_variant("gt_top1_good", gt_good & top1)
        run_variant("gt_topk_good", gt_good)

        dense_pose = dense_pose_map.get(camera.image_name)
        if dense_pose is not None:
            dense_uv, dense_z = _project(p3d_topk, dense_pose, intrinsic)
            dense_err = np.linalg.norm(dense_uv - p2d_topk, axis=1)
            dense_keep = (
                (dense_err <= float(args.pose_threshold_px))
                & (dense_z > 0)
                & (dense_uv[:, 0] >= 0)
                & (dense_uv[:, 0] < w)
                & (dense_uv[:, 1] >= 0)
                & (dense_uv[:, 1] < h)
            )
        else:
            dense_keep = np.zeros_like(gt_good)
        run_variant("dense_pose_topk_consistent", dense_keep)

        sparse_uv, sparse_z = _project(p3d_topk, native.pose_w2c, intrinsic)
        sparse_err = np.linalg.norm(sparse_uv - p2d_topk, axis=1)
        sparse_keep = (
            (sparse_err <= float(args.pose_threshold_px))
            & (sparse_z > 0)
            & (sparse_uv[:, 0] >= 0)
            & (sparse_uv[:, 0] < w)
            & (sparse_uv[:, 1] >= 0)
            & (sparse_uv[:, 1] < h)
        )
        run_variant("sparse_pose_topk_consistent", sparse_keep)

        grid_keep = np.zeros_like(sparse_keep)
        if np.any(sparse_keep):
            cell_x = np.clip((p2d_topk[:, 0] / max(w, 1) * int(args.grid)).astype(np.int32), 0, int(args.grid) - 1)
            cell_y = np.clip((p2d_topk[:, 1] / max(h, 1) * int(args.grid)).astype(np.int32), 0, int(args.grid) - 1)
            for cy in range(int(args.grid)):
                for cx in range(int(args.grid)):
                    candidates = np.where(sparse_keep & (cell_x == cx) & (cell_y == cy))[0]
                    if candidates.size == 0:
                        continue
                    order = candidates[np.argsort(-score_topk_np[candidates])]
                    grid_keep[order[: int(args.max_per_cell)]] = True
        run_variant("sparse_pose_grid_consistent", grid_keep)

        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rows_shard{args.shard_index:02d}.json").write_text(json.dumps(_jsonable(rows), indent=2))
    summary = _summarize(rows, variants)
    summary.update(
        {
            "diagnostic_only": True,
            "paper_safe_for_tuning": False,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "topk": args.topk,
            "gt_threshold_px": args.gt_threshold_px,
            "pose_threshold_px": args.pose_threshold_px,
        }
    )
    (args.output / f"summary_shard{args.shard_index:02d}.json").write_text(json.dumps(_jsonable(summary), indent=2))
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
