#!/usr/bin/env python3
"""Train a standard 3D Gaussian Splatting model from COLMAP sparse reconstruction.

Initializes Gaussians from COLMAP points3D.ply, trains with gsplat rasterization
and adaptive densification (clone + split), and saves a standard 3DGS PLY file.

Usage:
    python -m loc_gs.scripts.train_colmap_gs \
        --scene_root /mnt/pool/sqy/lerf_ovs/figurines \
        --output_dir output/3dgs_models/figurines \
        --iters 30000 --device cuda
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gsplat import DefaultStrategy, rasterization
from loc_gs.data.lerf_dataset import (
    _camera_params_to_intrinsics,
    _parse_colmap_sparse,
    _qvec_to_rotmat,
    _read_cameras_binary,
    _read_images_binary,
)

# SH basis constant for degree-0
C0 = 0.28209479177387814


# ──────────────────────────────────────────────────────────────────
# COLMAP points3D.ply loader
# ──────────────────────────────────────────────────────────────────

def load_colmap_points(scene_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load COLMAP sparse points from points3D.ply.

    Returns:
        xyz: (N, 3) float32 positions
        rgb: (N, 3) float32 colours in [0, 1]
    """
    ply_path = scene_root / "sparse" / "0" / "points3D.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"COLMAP points3D.ply not found: {ply_path}")

    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata.elements[0]
    xyz = np.stack(
        [np.asarray(vertex["x"], dtype=np.float32),
         np.asarray(vertex["y"], dtype=np.float32),
         np.asarray(vertex["z"], dtype=np.float32)],
        axis=1,
    )
    rgb = np.stack(
        [np.asarray(vertex["red"], dtype=np.float32),
         np.asarray(vertex["green"], dtype=np.float32),
         np.asarray(vertex["blue"], dtype=np.float32)],
        axis=1,
    ) / 255.0
    return xyz, rgb


# ──────────────────────────────────────────────────────────────────
# Scene data loading
# ──────────────────────────────────────────────────────────────────

def load_scene(scene_root: str, device: torch.device):
    """Load images, w2c matrices, and intrinsics from a COLMAP scene.

    Images are kept on CPU to save GPU memory; only the active frame is
    moved to *device* during training.

    Returns:
        images: list of [H, W, 3] float32 tensors on **CPU**
        w2cs:   list of [4, 4] float32 w2c tensors on *device*
        K:      [3, 3] intrinsics tensor on *device*
        W, H:   image width and height
        camera_extent: float, NeRF++ style camera radius for LR scaling
    """
    scene_root = Path(scene_root)
    colmap = _parse_colmap_sparse(scene_root)

    W, H = colmap["w"], colmap["h"]
    fx, fy, cx, cy = colmap["fl_x"], colmap["fl_y"], colmap["cx"], colmap["cy"]
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )

    # c2w → w2c
    c2w_list = colmap["c2w_list"]
    w2cs = []
    for c2w in c2w_list:
        w2c = np.linalg.inv(c2w).astype(np.float32)
        w2cs.append(torch.from_numpy(w2c).to(device))

    # Load images (kept on CPU)
    file_paths = colmap["file_paths"]
    images = []
    for fp in file_paths:
        full = scene_root / fp
        if not full.exists():
            raise FileNotFoundError(f"Image not found: {full}")
        img = np.array(Image.open(str(full)).convert("RGB"), dtype=np.float32) / 255.0
        images.append(torch.from_numpy(img))  # CPU

    print(f"Loaded {len(images)} images at {W}×{H}, "
          f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    camera_extent = compute_camera_extent(colmap["c2w_list"])
    print(f"  Camera extent (NeRF++ radius): {camera_extent:.2f}")
    return images, w2cs, K, W, H, camera_extent


# ──────────────────────────────────────────────────────────────────
# Gaussian initialisation
# ──────────────────────────────────────────────────────────────────

def compute_scene_scale(xyz: np.ndarray) -> float:
    """Compute scene scale as the extent of the point cloud bounding box."""
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    return float(np.linalg.norm(pmax - pmin))


def compute_camera_extent(c2w_list: list) -> float:
    """Compute camera extent (NeRF++ style) for position LR scaling.

    Returns the radius of the smallest sphere centred at the camera centroid
    that contains all cameras, times 1.1 (standard 3DGS padding).
    """
    cam_centers = np.array([c2w[:3, 3] for c2w in c2w_list])
    centroid = cam_centers.mean(axis=0)
    dists = np.linalg.norm(cam_centers - centroid, axis=1)
    return float(np.max(dists) * 1.1)


def init_gaussians(scene_root: str, device: torch.device):
    """Create initial Gaussian parameters from COLMAP points3D.ply.

    Returns:
        params: dict of nn.Parameter with keys
            means, scales, quats, opacities, sh0, shN
        scene_scale: float
    """
    xyz, rgb = load_colmap_points(Path(scene_root))
    N = xyz.shape[0]
    scene_scale = compute_scene_scale(xyz)
    print(f"Initialising {N:,} Gaussians from COLMAP points "
          f"(scene scale={scene_scale:.2f})")

    means = torch.from_numpy(xyz).float().to(device)

    # SH DC from colours: colour = C0 * sh_dc + 0.5  →  sh_dc = (colour - 0.5) / C0
    sh_dc = (torch.from_numpy(rgb).float().to(device) - 0.5) / C0  # [N, 3]
    sh0 = sh_dc.unsqueeze(1)  # [N, 1, 3]

    # Higher-order SH (degree 3 → 15 extra coefficients per channel)
    shN = torch.zeros(N, 15, 3, device=device)

    # Initial log-scales: use a fraction of the mean nearest-neighbour distance
    log_scales = _estimate_initial_scales(means, scene_scale)

    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # identity rotation

    # Logit of initial opacity 0.1 → logit(0.1) ≈ −2.197
    opacities = torch.full((N, 1), math.log(0.1 / 0.9), device=device)

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(log_scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }
    return params, scene_scale


def _estimate_initial_scales(means: Tensor, scene_scale: float) -> Tensor:
    """Estimate per-Gaussian initial log-scale from nearest-neighbour distances."""
    N = means.shape[0]
    device = means.device
    cpu_means = means.detach().cpu()

    # For large point clouds, subsample to estimate median NN distance
    max_sample = min(N, 50_000)
    idx = torch.randperm(N)[:max_sample]
    subset = cpu_means[idx]

    # Pairwise distances on CPU (chunked to avoid GPU OOM during init)
    chunk = 1024
    nn_dists = []
    for i in range(0, len(subset), chunk):
        end = min(i + chunk, len(subset))
        dists = torch.cdist(subset[i:end], subset)  # [chunk_sz, max_sample]
        # Mask self-distances on the diagonal (local row j → global col i+j)
        rows = torch.arange(end - i)
        dists[rows, i + rows] = float("inf")
        nn_dist, _ = dists.min(dim=1)
        nn_dists.append(nn_dist)

    nn_dists = torch.cat(nn_dists)
    median_nn = nn_dists.median().item()
    init_scale = max(median_nn * 0.5, scene_scale * 1e-4)
    log_scale = math.log(init_scale)
    print(f"  Median NN distance: {median_nn:.4f}, "
          f"init scale: {init_scale:.4f} (log={log_scale:.3f})")
    return torch.full((N, 3), log_scale, device=device)


# ──────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────

def l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred - gt).abs().mean()


_ssim_window_cache: dict = {}


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     dtype: torch.dtype, device: torch.device) -> Tensor:
    """Create a 2-D Gaussian window for SSIM (cached across calls)."""
    key = (window_size, sigma, channels, dtype, device)
    if key not in _ssim_window_cache:
        coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        _ssim_window_cache[key] = window_2d.unsqueeze(0).unsqueeze(0).expand(
            channels, 1, -1, -1
        ).contiguous()
    return _ssim_window_cache[key]


def ssim_loss(pred: Tensor, gt: Tensor, window_size: int = 11) -> Tensor:
    """Structural similarity loss: 1 − SSIM (Gaussian-weighted window)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    p = pred.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    g = gt.permute(2, 0, 1).unsqueeze(0)
    C = p.shape[1]
    pad = window_size // 2

    # Reflect-pad to avoid border bias from zero-padding
    p = F.pad(p, [pad, pad, pad, pad], mode="reflect")
    g = F.pad(g, [pad, pad, pad, pad], mode="reflect")

    window = _gaussian_window(window_size, 1.5, C, p.dtype, p.device)

    mu_p = F.conv2d(p, window, groups=C)
    mu_g = F.conv2d(g, window, groups=C)
    mu_pp = mu_p * mu_p
    mu_gg = mu_g * mu_g
    mu_pg = mu_p * mu_g

    sigma_pp = F.conv2d(p * p, window, groups=C) - mu_pp
    sigma_gg = F.conv2d(g * g, window, groups=C) - mu_gg
    sigma_pg = F.conv2d(p * g, window, groups=C) - mu_pg

    # Clamp variances to avoid numerical artifacts from E[x²] − E[x]²
    sigma_pp = sigma_pp.clamp(min=0.0)
    sigma_gg = sigma_gg.clamp(min=0.0)

    ssim = ((2.0 * mu_pg + C1) * (2.0 * sigma_pg + C2)) / \
           ((mu_pp + mu_gg + C1) * (sigma_pp + sigma_gg + C2))
    return 1.0 - ssim.mean()


# ──────────────────────────────────────────────────────────────────
# PLY export (standard 3DGS format)
# ──────────────────────────────────────────────────────────────────

def save_ply(path: str, params: dict, sh_degree: int = 3) -> None:
    """Save Gaussians to standard 3DGS PLY format.

    The PLY contains per-vertex properties:
        x, y, z, nx, ny, nz,
        f_dc_0..2, f_rest_0..44,
        opacity, scale_0..2, rot_0..3
    """
    from plyfile import PlyData, PlyElement

    means = params["means"].detach().cpu().numpy()         # [N, 3]
    scales = params["scales"].detach().cpu().numpy()        # [N, 3]
    quats = params["quats"].detach().cpu().numpy()          # [N, 4]
    opacities = params["opacities"].detach().cpu().numpy()  # [N, 1]
    sh0 = params["sh0"].detach().cpu().numpy()              # [N, 1, 3]
    shN = params["shN"].detach().cpu().numpy()              # [N, K, 3]

    N = means.shape[0]
    n_rest = shN.shape[1]  # 15 for degree 3

    # Build structured numpy array
    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    for i in range(3):
        dtype_list.append((f"f_dc_{i}", "f4"))
    # f_rest: stored as [ch0_c0, ch0_c1, ..., ch0_cK, ch1_c0, ..., ch2_cK]
    for i in range(n_rest * 3):
        dtype_list.append((f"f_rest_{i}", "f4"))
    dtype_list.append(("opacity", "f4"))
    for i in range(3):
        dtype_list.append((f"scale_{i}", "f4"))
    for i in range(4):
        dtype_list.append((f"rot_{i}", "f4"))

    arr = np.empty(N, dtype=dtype_list)
    arr["x"] = means[:, 0]
    arr["y"] = means[:, 1]
    arr["z"] = means[:, 2]
    arr["nx"] = 0.0
    arr["ny"] = 0.0
    arr["nz"] = 0.0

    # SH DC
    for i in range(3):
        arr[f"f_dc_{i}"] = sh0[:, 0, i]

    # SH rest — original 3DGS stores as transpose(1,2).flatten():
    #   [ch0_c0, ch0_c1, ..., ch0_cK, ch1_c0, ..., ch2_cK]
    # shN is [N, K, 3], so we need [N, 3, K] flattened to [N, 3*K]
    sh_rest_flat = shN.transpose(0, 2, 1).reshape(N, -1)  # [N, 3*K]
    for i in range(n_rest * 3):
        arr[f"f_rest_{i}"] = sh_rest_flat[:, i]

    arr["opacity"] = opacities[:, 0]
    for i in range(3):
        arr[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))
    print(f"Saved {N:,} Gaussians to {path}")


# ──────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load scene data
    images, w2cs, K, W, H, camera_extent = load_scene(args.scene_root, device)
    n_frames = len(images)

    # Initialise Gaussians from COLMAP points
    params, scene_scale = init_gaussians(args.scene_root, device)
    n_init = params["means"].shape[0]

    # Background colour
    if args.white_bg:
        bg = torch.ones(3, device=device)
    else:
        bg = torch.zeros(3, device=device)

    # Optimiser with per-parameter learning rates
    # Scale position LR by camera extent (standard 3DGS practice — NeRF++ radius)
    lr_map = {
        "means": args.lr_means * camera_extent,
        "scales": args.lr_scale,
        "quats": args.lr_quat,
        "opacities": args.lr_opacity,
        "sh0": args.lr_sh,
        "shN": args.lr_sh,
    }
    optimizers = {}
    for name, param in params.items():
        optimizers[name] = torch.optim.Adam(
            [param], lr=lr_map[name], eps=1e-15,
        )

    # LR schedulers for positions: warmup (1% → 100% over 1000 steps) + exponential decay
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizers["means"], start_factor=0.01, total_iters=1000,
    )
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"],
        gamma=(args.lr_means_final_factor) ** (1.0 / max(args.iters, 1)),
    )
    pos_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [warmup_scheduler, decay_scheduler]
    )

    # Densification strategy
    strategy = DefaultStrategy(
        grow_grad2d=args.densify_grad_thresh,
        refine_start_iter=args.densify_from,
        refine_stop_iter=args.densify_until,
        refine_every=args.densify_every,
        reset_every=args.opacity_reset_every,
        prune_opa=0.005,
        absgrad=True,
        verbose=True,
    )
    state = strategy.initialize_state(scene_scale=scene_scale)

    # SH degree schedule: start at 0, increase every 1000 iters up to sh_degree
    max_sh_degree = args.sh_degree

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "point_cloud" / f"iteration_{args.iters}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training 3DGS: {n_init:,} initial Gaussians, {args.iters} iterations")
    print(f"Scene: {args.scene_root}")
    print(f"Output: {out_dir}")
    print(f"Scene scale (pts bbox): {scene_scale:.2f}, Camera extent: {camera_extent:.2f}")
    print(f"Position LR: {args.lr_means} × {camera_extent:.2f} = {args.lr_means * camera_extent:.4e}")
    print(f"{'='*60}\n")

    for step in range(args.iters):
        # Current SH degree
        cur_sh_degree = min(step // 1000, max_sh_degree)

        # Random training frame
        idx = random.randint(0, n_frames - 1)
        gt_img = images[idx].to(device)  # [H, W, 3] — CPU → GPU
        viewmat = w2cs[idx]              # [4, 4]

        # Build SH colour tensor: [N, K_cur, 3]
        n_cur_sh = (cur_sh_degree + 1) ** 2
        if cur_sh_degree == 0:
            colors = params["sh0"]  # [N, 1, 3]
        else:
            colors = torch.cat(
                [params["sh0"], params["shN"][:, : n_cur_sh - 1, :]],
                dim=1,
            )  # [N, K_cur, 3]

        scales = torch.exp(params["scales"])
        opacities = torch.sigmoid(params["opacities"]).squeeze(-1)  # [N]
        quats = F.normalize(params["quats"], p=2, dim=-1)

        # Render
        renders, alphas, info = rasterization(
            means=params["means"],
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=1000.0,
            backgrounds=bg.unsqueeze(0),
            sh_degree=cur_sh_degree,
            packed=False,
            absgrad=True,
        )
        pred_rgb = renders[0].clamp(0.0, 1.0)  # [H, W, 3]

        # Loss: L1 + SSIM
        loss_l1 = l1_loss(pred_rgb, gt_img)
        loss_ssim = ssim_loss(pred_rgb, gt_img)
        loss = (1.0 - args.lambda_ssim) * loss_l1 + args.lambda_ssim * loss_ssim

        # Densification: pre-backward
        strategy.step_pre_backward(
            params=params, optimizers=optimizers, state=state,
            step=step, info=info,
        )

        loss.backward()

        # Densification: post-backward
        strategy.step_post_backward(
            params=params, optimizers=optimizers, state=state,
            step=step, info=info, packed=False,
        )

        # Optimiser step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        pos_scheduler.step()

        # Logging
        if step % args.log_every == 0 or step == args.iters - 1:
            with torch.no_grad():
                mse = ((pred_rgb - gt_img) ** 2).mean()
                psnr = -10.0 * math.log10(mse.item() + 1e-10)
            n_gs = params["means"].shape[0]
            lr_pos = optimizers["means"].param_groups[0]["lr"]
            print(
                f"[Iter {step:>5d}/{args.iters}] "
                f"loss={loss.item():.4f}  PSNR={psnr:.2f} dB  "
                f"#GS={n_gs:,}  SH={cur_sh_degree}  lr_pos={lr_pos:.2e}"
            )

        # Intermediate checkpoint
        if args.save_every > 0 and (step + 1) % args.save_every == 0 and step > 0:
            mid_dir = out_dir / "point_cloud" / f"iteration_{step + 1}"
            mid_dir.mkdir(parents=True, exist_ok=True)
            save_ply(str(mid_dir / "point_cloud.ply"), params, max_sh_degree)

    # Final save
    save_ply(str(ply_dir / "point_cloud.ply"), params, max_sh_degree)
    print(f"\nTraining complete. Final #GS: {params['means'].shape[0]:,}")
    print(f"Output saved to {ply_dir / 'point_cloud.ply'}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian Splatting from COLMAP sparse reconstruction",
    )
    # I/O
    parser.add_argument("--scene_root", required=True,
                        help="Path to COLMAP scene (with sparse/0/ and images/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for trained model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--lambda_ssim", type=float, default=0.2,
                        help="SSIM weight in combined loss (L1 weight = 1 − this)")

    # Learning rates
    parser.add_argument("--lr_means", type=float, default=1.6e-4)
    parser.add_argument("--lr_means_final_factor", type=float, default=0.01,
                        help="Final LR = lr_means × this factor (exponential decay)")
    parser.add_argument("--lr_sh", type=float, default=2.5e-3)
    parser.add_argument("--lr_scale", type=float, default=5e-3)
    parser.add_argument("--lr_quat", type=float, default=1e-3)
    parser.add_argument("--lr_opacity", type=float, default=0.05)

    # Densification
    parser.add_argument("--densify_from", type=int, default=500)
    parser.add_argument("--densify_until", type=int, default=15000)
    parser.add_argument("--densify_every", type=int, default=100)
    parser.add_argument("--densify_grad_thresh", type=float, default=0.0008)
    parser.add_argument("--opacity_reset_every", type=int, default=3000)

    # Logging / checkpoints
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save intermediate checkpoint every N iters (0=off)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
