#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loc_gs.data.cambridge_dataset import CambridgeHybridDataset
from loc_gs.data.external_match_cache import ExternalMatchCache
from loc_gs.models.hybrid_gaussian import HybridFeatureGaussian, SuperPointOutputHead
from loc_gs.scripts.train_cambridge_hybrid import render_hybrid_superpoint


def _sample_depth(depth: torch.Tensor, keypoints_xy: np.ndarray) -> torch.Tensor:
    if keypoints_xy.size == 0:
        return depth.new_empty(0)
    pts = torch.as_tensor(keypoints_xy, device=depth.device, dtype=depth.dtype)
    H, W = depth.shape
    grid = torch.stack(
        [
            2.0 * pts[:, 0] / max(W - 1, 1) - 1.0,
            2.0 * pts[:, 1] / max(H - 1, 1) - 1.0,
        ],
        dim=-1,
    ).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        depth.view(1, 1, H, W).float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, 0, :, 0]


def _unproject_xy(keypoints_xy: np.ndarray, depth: torch.Tensor, pose_w2c: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    pts = torch.as_tensor(keypoints_xy, device=depth.device, dtype=depth.dtype)
    d = _sample_depth(depth, keypoints_xy)
    x = pts[:, 0]
    y = pts[:, 1]
    x_cam = (x - K[0, 2]) / K[0, 0].clamp_min(1e-8) * d
    y_cam = (y - K[1, 2]) / K[1, 1].clamp_min(1e-8) * d
    pts_cam = torch.stack([x_cam, y_cam, d, torch.ones_like(d)], dim=-1)
    c2w = torch.linalg.inv(pose_w2c.float())
    return (c2w @ pts_cam.T).T[:, :3]


def _project_world_xy(points_world: torch.Tensor, pose_w2c: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pts_h = torch.cat([points_world, torch.ones_like(points_world[:, :1])], dim=-1)
    pts_cam = (pose_w2c.float() @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    x = K[0, 0] * pts_cam[:, 0] / z.clamp_min(1e-8) + K[0, 2]
    y = K[1, 1] * pts_cam[:, 1] / z.clamp_min(1e-8) + K[1, 2]
    return torch.stack([x, y], dim=-1), z


def geometric_inlier_mask(
    kpts_a_xy: np.ndarray,
    kpts_b_xy: np.ndarray,
    depth_a: torch.Tensor,
    depth_b: torch.Tensor,
    pose_a_w2c: torch.Tensor,
    pose_b_w2c: torch.Tensor,
    K: torch.Tensor,
    reprojection_threshold_px: float = 3.0,
    depth_tolerance: float = 0.25,
) -> np.ndarray:
    if len(kpts_a_xy) == 0:
        return np.zeros(0, dtype=np.bool_)
    world = _unproject_xy(kpts_a_xy, depth_a, pose_a_w2c, K)
    reproj_b, z_b = _project_world_xy(world, pose_b_w2c, K)
    target_b = torch.as_tensor(kpts_b_xy, device=depth_a.device, dtype=depth_a.dtype)
    err = torch.linalg.norm(reproj_b - target_b, dim=-1)
    sampled_b = _sample_depth(depth_b, kpts_b_xy)
    H, W = depth_b.shape
    in_frame = (
        (reproj_b[:, 0] >= 0)
        & (reproj_b[:, 0] <= W - 1)
        & (reproj_b[:, 1] >= 0)
        & (reproj_b[:, 1] <= H - 1)
    )
    valid = (
        torch.isfinite(err)
        & in_frame
        & (err <= float(reprojection_threshold_px))
        & torch.isfinite(sampled_b)
        & (sampled_b > 0.05)
        & ((sampled_b - z_b).abs() <= float(depth_tolerance))
    )
    return valid.detach().cpu().numpy().astype(np.bool_)


def _rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


@torch.no_grad()
def _loftr_matches(rgb_a: torch.Tensor, rgb_b: torch.Tensor, matcher) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = matcher({"image0": _rgb_to_gray(rgb_a), "image1": _rgb_to_gray(rgb_b)})
    k0 = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
    k1 = out["keypoints1"].detach().cpu().numpy().astype(np.float32)
    conf = out["confidence"].detach().cpu().numpy().astype(np.float32)
    return k0, k1, conf


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate detector-free external match supervision cache.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default="ShopFacade")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--pipeline", choices=["loftr"], default="loftr")
    parser.add_argument("--max_pairs", type=int, default=5)
    parser.add_argument("--pair_stride", type=int, default=1)
    parser.add_argument("--reprojection_threshold_px", type=float, default=3.0)
    parser.add_argument("--depth_tolerance", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    scene_root = Path(args.data_root) / args.scene
    dataset = CambridgeHybridDataset(
        scene_root=scene_root,
        cameras_json=train_args["cameras_json"],
        split="train",
        image_subdir=train_args.get("image_subdir", "processed"),
        image_height=train_args["image_height"],
        image_width=train_args["image_width"],
        max_frames=max(args.max_pairs + args.pair_stride, 2),
    )

    from loc_gs.rendering.feature_renderer import FeatureFieldRenderer

    renderer = FeatureFieldRenderer(
        image_height=train_args["image_height"],
        image_width=train_args["image_width"],
        fx=train_args["feature_intrinsics"]["fx"] * 8.0,
        fy=train_args["feature_intrinsics"]["fy"] * 8.0,
        cx=train_args["feature_intrinsics"]["cx"] * 8.0,
        cy=train_args["feature_intrinsics"]["cy"] * 8.0,
        max_channels_per_chunk=32,
        far_plane=10000.0,
        packed=False,
        rasterize_mode="antialiased",
    ).to(device)
    model = HybridFeatureGaussian(
        latent_dim=train_args["latent_dim"],
        hash_output_dim=train_args["hash_output_dim"],
        fine_dim=train_args["fine_dim"],
        coarse_dim=train_args["coarse_dim"],
        output_dim=train_args["hybrid_output_dim"],
    )
    model.load_from_ply(train_args["ply_path"])
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    sp_head = SuperPointOutputHead(
        fused_dim=train_args["hybrid_output_dim"],
        descriptor_dim=256,
        detector_dim=65,
        hidden_dim=256,
        num_res_blocks=2,
        use_3x3=True,
    ).to(device)
    sp_head.load_state_dict(ckpt["sp_head_state_dict"])
    sp_head.eval()
    cache = ExternalMatchCache(args.output_root, scene=args.scene, pipeline=args.pipeline, split="train")
    from kornia.feature import LoFTR

    loftr = LoFTR(pretrained="outdoor").to(device).eval()

    stats = []
    max_start = min(args.max_pairs, max(0, len(dataset) - args.pair_stride))
    for i in tqdm(range(max_start), desc=f"Generating {args.pipeline} cache", dynamic_ncols=True):
        item_a = dataset[i]
        item_b = dataset[i + args.pair_stride]
        rgb_a = item_a["rgb"].unsqueeze(0).to(device)
        rgb_b = item_b["rgb"].unsqueeze(0).to(device)
        k0, k1, scores = _loftr_matches(rgb_a, rgb_b, loftr)
        pose_a = item_a["pose_w2c"].to(device).unsqueeze(0)
        pose_b = item_b["pose_w2c"].to(device).unsqueeze(0)
        depth_a = render_hybrid_superpoint(model, sp_head, renderer, pose_a)["depth"][0].float()
        depth_b = render_hybrid_superpoint(model, sp_head, renderer, pose_b)["depth"][0].float()
        K = item_a["K"].to(device)
        inliers = geometric_inlier_mask(
            k0,
            k1,
            depth_a,
            depth_b,
            pose_a[0],
            pose_b[0],
            K,
            reprojection_threshold_px=args.reprojection_threshold_px,
            depth_tolerance=args.depth_tolerance,
        )
        cache.save_matches(
            str(item_a["image_name"]),
            str(item_b["image_name"]),
            kpts_a_xy=k0,
            kpts_b_xy=k1,
            scores=scores,
            geom_inlier_mask=inliers,
            stats={
                "num_raw_matches": int(len(scores)),
                "num_inliers": int(inliers.sum()),
                "inlier_ratio": float(inliers.mean()) if len(inliers) else 0.0,
            },
        )
        stats.append({"pair": [str(item_a["image_name"]), str(item_b["image_name"])], "raw": int(len(scores)), "inliers": int(inliers.sum())})
    summary_path = cache.match_root / "generation_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(stats), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
