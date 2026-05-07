import json
import os
import random
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from arguments import ModelParams
from gaussian_renderer import render_from_pose_gsplat
from scene import Scene
from scene.dataset_readers import fetchPly
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from utils.geometry_metrics import chamfer_distance_stats, projected_depth_consistency
from utils.graphics_utils import fov2focal, getWorld2View2


def select_items(items, max_items=None, stride=1):
    stride = max(1, int(stride or 1))
    selected = list(items)[::stride]
    if max_items is not None and max_items > 0:
        selected = selected[:max_items]
    return selected


def subsample_points(points, max_points, seed):
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))
    ids = torch.randperm(points.shape[0], device=points.device, generator=generator)[:max_points]
    return points[ids]


def intrinsic_from_camera_info(cam_info, device):
    width = int(cam_info.width)
    height = int(cam_info.height)
    return torch.tensor(
        [
            [fov2focal(cam_info.FovX, width), 0.0, width / 2.0],
            [0.0, fov2focal(cam_info.FovY, height), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def project_points(points_world, pose_w2c, K, width, height):
    ones = torch.ones((points_world.shape[0], 1), device=points_world.device)
    points_h = torch.cat([points_world, ones], dim=-1)
    points_cam = (pose_w2c @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    valid = z > 1e-6
    pixels = (K @ (points_cam / z.clamp_min(1e-6)[:, None]).T).T
    xy = pixels[:, :2]
    valid = (
        valid
        & torch.isfinite(xy).all(dim=-1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= height - 1)
    )
    return xy[valid], z[valid]


def summarize_depth_stats(per_camera):
    keys = ["count", "abs_error_mean", "abs_error_median", "abs_error_p90", "rel_error_median"]
    valid = [item for item in per_camera if item["count"] > 0]
    if not valid:
        return {key: 0.0 for key in keys}

    total = sum(item["count"] for item in valid)
    summary = {"count": int(total)}
    for key in keys[1:]:
        summary[key] = sum(item[key] * item["count"] for item in valid) / total
    return summary


def main():
    parser = ArgumentParser(description="Evaluate Gaussian geometry consistency against COLMAP sparse geometry.")
    lp = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--max_cameras", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_reference_points", type=int, default=200000)
    parser.add_argument("--max_gaussian_points", type=int, default=200000)
    parser.add_argument("--chamfer_chunk", type=int, default=32768)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = lp.extract(args)
    if dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
        dataloader_num_workers=0,
        pin_memory=False,
    )

    reference_ply = os.path.join(dataset.source_path, "sparse/0/points3D.ply")
    reference = torch.as_tensor(fetchPly(reference_ply).points, dtype=torch.float32, device=device)
    gaussian_xyz = gaussians.get_xyz.detach().to(device=device, dtype=torch.float32)

    chamfer = chamfer_distance_stats(
        subsample_points(gaussian_xyz, args.max_gaussian_points, args.seed),
        subsample_points(reference, args.max_reference_points, args.seed + 1),
        chunk_size=args.chamfer_chunk,
    )

    if args.split == "train":
        camera_infos = scene.scene_info.train_cameras
    elif args.split == "test":
        camera_infos = scene.scene_info.test_cameras
    else:
        camera_infos = scene.scene_info.train_cameras + scene.scene_info.test_cameras
    camera_infos = select_items(camera_infos, max_items=args.max_cameras, stride=args.stride)

    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    per_camera = []
    for cam_info in tqdm(camera_infos, desc="Geometry consistency"):
        width = int(cam_info.width)
        height = int(cam_info.height)
        pose = torch.tensor(getWorld2View2(cam_info.R, cam_info.T), dtype=torch.float32, device=device)
        K = intrinsic_from_camera_info(cam_info, device)
        with torch.no_grad():
            render_pkg = render_from_pose_gsplat(
                gaussians,
                pose,
                cam_info.FovX,
                cam_info.FovY,
                width,
                height,
                bg,
                rgb_only=True,
                render_mode="RGB+ED",
                rasterize_mode="antialiased",
            )
        depth = render_pkg["depth"]
        if depth is None:
            continue
        points_xy, point_depth = project_points(reference, pose, K, width, height)
        stats = projected_depth_consistency(depth.squeeze(), points_xy, point_depth)
        stats["image_name"] = cam_info.image_name
        per_camera.append(stats)

    summary = {
        "model_path": dataset.model_path,
        "source_path": dataset.source_path,
        "iteration": scene.loaded_iter,
        "split": args.split,
        "num_cameras": len(camera_infos),
        "chamfer": chamfer,
        "depth_consistency": summarize_depth_stats(per_camera),
        "per_camera": per_camera,
    }

    output = args.output or os.path.join(dataset.model_path, "geometry_consistency.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["iteration", "chamfer", "depth_consistency"]}, indent=2))
    print(f"Saved geometry consistency report to {output}")


if __name__ == "__main__":
    main()
