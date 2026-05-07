#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
from itertools import chain, islice
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import get_render_visible_mask, render_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state, seed_everything
from utils.graphics_utils import focal2fov, fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.loss_utils import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import pickle

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor
from scene.kpdetector import KpDetector


def get_next_train_camera(scene, viewpoint_stack=None, viewpoint_iter=None):
    if getattr(scene, "preload_cameras", True):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        return viewpoint_cam, viewpoint_stack, viewpoint_iter

    if viewpoint_iter is None:
        viewpoint_iter = iter(scene.getTrainCameras())
    try:
        viewpoint_cam = next(viewpoint_iter)
    except StopIteration:
        viewpoint_iter = iter(scene.getTrainCameras())
        viewpoint_cam = next(viewpoint_iter)
    return viewpoint_cam, viewpoint_stack, viewpoint_iter


def select_report_train_cameras(scene, count=5):
    train_cameras = scene.getTrainCameras()
    if getattr(scene, "preload_cameras", True):
        if len(train_cameras) == 0:
            return []
        return [
            train_cameras[idx % len(train_cameras)]
            for idx in range(5, 5 * (count + 1), 5)
        ]
    return list(islice(iter(train_cameras), count))


def get_train_image_names(source_path, images):
    test_images = set()
    for split_name in ("dataset_test.txt", os.path.join("sparse", "0", "list_test.txt")):
        split_path = os.path.join(source_path, split_name)
        if os.path.exists(split_path):
            with open(split_path) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        test_images.add(line.split(" ")[0])

    image_root = os.path.join(source_path, images)
    names = []
    for root, _dirs, files in os.walk(image_root):
        for filename in files:
            rel_path = os.path.relpath(os.path.join(root, filename), image_root)
            rel_path = rel_path.replace(os.sep, "/")
            if rel_path not in test_images:
                names.append(rel_path)
    return sorted(names)


def first_camera_and_iter(cameras):
    if len(cameras) == 0:
        return None, iter(())
    if hasattr(cameras, "__getitem__"):
        return cameras[0], iter(cameras)
    iterator = iter(cameras)
    first = next(iterator, None)
    if first is None:
        return None, iter(())
    return first, chain([first], iterator)


def get_sampled_gaussian(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(gaussians.max_sh_degree)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians._loc_feature[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians


@torch.no_grad()
def calculate_match_score(
    gaussians: GaussianModel,
    gt_feature_map,
    pose,
    K,
    render_visible_mask=None,
    img_mask=None,
):
    xyz = gaussians.get_xyz
    feat = gaussians.get_loc_feature.squeeze()

    # project gaussians to image space
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_homo = xyz_cam / depths

    xy = (K @ xyz_cam_homo)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    if render_visible_mask is not None:
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask

    if img_mask is not None:
        visible_xy = xy[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_xy[1], visible_xy[0]]
        visible_mask = visible_mask & img_mask_expand

    xy = xy[:, visible_mask]
    depths = depths[visible_mask]
    feat = feat[visible_mask]

    gs_feats = F.normalize(feat, p=2, dim=1)
    im_feats = gt_feature_map[:, xy[1], xy[0]].T
    score = (gs_feats * im_feats).sum(-1)
    return score, visible_mask


def generate_gt_map(
    gaussians: GaussianModel,
    gt_feature_map,
    idx_sampled,
    pose,
    K,
    render_visible_mask=None,
):
    if render_visible_mask is not None:
        render_visible_mask = render_visible_mask[idx_sampled]
        idx_sampled = idx_sampled[render_visible_mask]
    sampled_xyz = gaussians.get_xyz[idx_sampled]

    gt_map = torch.zeros(
        (1, gt_feature_map.shape[1], gt_feature_map.shape[2]),
        device=gt_feature_map.device,
    )
    
    xyz_homo = torch.cat(
        [sampled_xyz, torch.ones(sampled_xyz.shape[0], 1, device=sampled_xyz.device)],
        dim=-1,
    )
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths

    xy = (K @ xyz_cam_norm)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    xy_pos = xy[:, in_mask]

    gt_map[:, xy_pos[1], xy_pos[0]] = 1

    return gt_map


def random_knn_score(points, npoints, score, k=32):
    sampled_idx = torch.randperm(points.shape[0])[:npoints]
    sampled_points = points[sampled_idx]
    points = points.cpu()
    sampled_points = sampled_points.cpu()
    dist = torch.cdist(sampled_points, points)
    knn_idx = torch.topk(dist, k, largest=False, dim=-1)[1]
    knn_idx = knn_idx.cuda()

    # knn select
    knn_score = score[knn_idx]  # (npoints, k)
    score_knn_sort_idx = torch.argsort(
        knn_score, descending=True, dim=-1
    )  # (npoints, k)

    final_sampled_idx = set()

    for i in range(npoints):
        for j in score_knn_sort_idx[i]:
            idx = knn_idx[i, j].item()  
            if idx not in final_sampled_idx: 
                final_sampled_idx.add(idx)  
                break  

    return torch.tensor(list(final_sampled_idx)).cuda()


def normalize_match_scores(score_avg, score_num):
    score = score_avg.float().clamp(0.0, 1.0)
    observations = score_num.float().clamp_min(0.0)
    max_observations = observations.max().clamp_min(1.0)
    repeatability = torch.log1p(observations) / torch.log1p(max_observations)
    return (score * repeatability).clamp(0.0, 1.0)


def matching_oriented_sample(
    scene,
    gaussians,
    feature_extractor,
    render_visible_masks,
    masks=None,
    num=16384,
    k=32,
):
    train_cameras = scene.getTrainCameras()
    if getattr(scene, "preload_cameras", True):
        viewpoint_iter = train_cameras.copy()
        first_viewpoint = viewpoint_iter[0]
    else:
        raw_iter = iter(train_cameras)
        first_viewpoint = next(raw_iter)
        viewpoint_iter = chain([first_viewpoint], raw_iter)
    score_sum = torch.zeros(
        gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda"
    )
    score_num = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.int, device="cuda")
    fine_resolution = (
        first_viewpoint.original_image.shape[1],
        first_viewpoint.original_image.shape[2],
    )

    for viewpoint_cam in tqdm(viewpoint_iter, desc="Match Score"):
        gt_image = viewpoint_cam.original_image.cuda()
        gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        # print("focal:", focalX, focalY)
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        if render_visible_masks.get(viewpoint_cam.image_name, None) is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            render_visible_masks[viewpoint_cam.image_name] = render_visible_mask
        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            img_mask = (
                F.interpolate(
                    mask[None].float(),
                    size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                > 0.5
            )
        else:
            img_mask = None

        score, mask = calculate_match_score(
            gaussians,
            gt_feature_map,
            viewmat,
            K,
            render_visible_mask=render_visible_masks[viewpoint_cam.image_name],
            img_mask=img_mask,
        )
        score_num[mask] += 1
        score_sum[mask] += score

    score_num[score_num == 0] = 1  # avoid divide by zero
    score_avg = score_sum / score_num

    sampled_idx = random_knn_score(gaussians.get_xyz, num, score_avg, k=k)
    sampled_idx = torch.unique(sampled_idx)
    return sampled_idx, score_avg, score_num


def evaluate_detector(
    detector,
    feature_extractor,
    gaussians,
    sampled_idx,
    scene,
    masks=None,
    render_visible_masks=None,
    tb_writer=None,
    iteration=0,
):
    torch.cuda.empty_cache()

    landmarks = get_sampled_gaussian(gaussians, sampled_idx)

    bg_color = [1, 1, 1] if scene.args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": select_report_train_cameras(scene),
        },
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            first_viewpoint, camera_iter = first_camera_and_iter(config["cameras"])
            if first_viewpoint is None:
                continue
            fine_resolution = get_resolution_from_longest_edge(
                first_viewpoint.original_image.shape[1],
                first_viewpoint.original_image.shape[2],
                scene.longest_edge,
            )
            loss_sum = 0.0
            camera_count = 0
            for idx, viewpoint_cam in enumerate(camera_iter):
                camera_count += 1
                gt_image = viewpoint_cam.original_image.cuda()
                gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
                gt_feature_map = F.interpolate(
                    gt_feature_map,
                    size=(fine_resolution[0], fine_resolution[1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

                viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
                focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
                focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
                # print("focal:", focalX, focalY)
                K = torch.tensor(
                    [
                        [focalX, 0.0, gt_feature_map.shape[2] / 2],
                        [0.0, focalY, gt_feature_map.shape[1] / 2],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                    device="cuda",
                )
                visible_mask = render_visible_masks.get(
                    viewpoint_cam.image_name, None
                )
                if visible_mask is None:
                    visible_mask = get_render_visible_mask(
                        gaussians,
                        viewpoint_cam,
                        gt_feature_map.shape[2],
                        gt_feature_map.shape[1],
                    )
                    render_visible_masks[viewpoint_cam.image_name] = visible_mask
                else:
                    visible_mask = render_visible_masks[viewpoint_cam.image_name]

                gt_map = generate_gt_map(
                    gaussians,
                    gt_feature_map,
                    sampled_idx,
                    viewmat,
                    K,
                    visible_mask,
                )

                if masks is not None:
                    object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
                    # sky_mask = masks[viewpoint_cam.image_name][1].cuda()[None]
                    distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

                    mask = object_mask & distort_mask
                    gt_map_mask = (
                        F.interpolate(
                            mask[None].float(),
                            size=(gt_map.shape[1], gt_map.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        > 0.5
                    )
                    gt_map *= gt_map_mask

                # Loss
                heat_map = detector(gt_feature_map)
                loss = score_map_bce_loss(heat_map, gt_map)

                loss_sum += loss.item()
                if tb_writer and idx < 5:
                    render = render_gsplat(
                        viewpoint_cam, gaussians, background, rgb_only=True
                    )["render"]
                    sampled_render = render_gsplat(
                        viewpoint_cam, landmarks, background, rgb_only=True
                    )["render"]
                    heat_map = (heat_map - heat_map.min()) / (
                        heat_map.max() - heat_map.min()
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/gt_map_{idx}",
                        gt_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/heat_map{idx}",
                        heat_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/render_{idx}",
                        render[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/sampled_render_{idx}",
                        sampled_render[None],
                        iteration,
                    )

            loss_sum /= max(camera_count, 1)
            print(
                f"\n[ITER {iteration}] Evaluating detector: {config['name']} loss {loss_sum}"
            )
            if tb_writer:
                tb_writer.add_scalar(
                    f"detector_loss_patches/{config['name']}_loss",
                    loss_sum,
                    iteration,
                )


def training_detector(
    gaussians,
    scene: Scene,
    masks,
    testing_iterations,
    saving_iterations,
    tb_writer,
    train_iteration=30000,
    detector_folder="",
    landmark_num=16384,
    landmark_k=32,
    locability_save_iteration=None,
):
    viewpoint_stack = None
    viewpoint_iter = None
    viewpoint_cam, viewpoint_stack, viewpoint_iter = get_next_train_camera(
        scene, viewpoint_stack, viewpoint_iter
    )
    feature_extractor = FeatureExtractor(scene.feature_type)

    render_visible_masks = {}

    # M.O. sampling
    print("Matching oriented sampling...")
    sampled_idx, score_avg, score_num = matching_oriented_sample(
        scene,
        gaussians,
        feature_extractor,
        render_visible_masks,
        masks=masks,
        num=landmark_num,
        k=landmark_k,
    )
    save_path = os.path.join(scene.model_path, detector_folder)
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(sampled_idx, open(os.path.join(save_path, "sampled_idx.pkl"), "wb"))
    locability = normalize_match_scores(score_avg, score_num)
    if hasattr(gaussians, "_locability_logit"):
        gaussians._locability_logit.data = torch.logit(
            locability[:, None].clamp(1e-4, 1.0 - 1e-4)
        )
    pickle.dump(
        {
            "sampled_scores": locability[sampled_idx].detach().cpu(),
            "score_avg": locability.detach().cpu(),
            "score_num": score_num.detach().cpu(),
        },
        open(os.path.join(save_path, "sampled_scores.pkl"), "wb"),
    )
    if locability_save_iteration is not None:
        scene.save(locability_save_iteration)

    # training scene-specific detector
    print("Training detector...")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    viewpoint_iter = None
    progress_bar = tqdm(range(0, train_iteration), desc="Scene-Specific Detector")
    first_iter = 1

    detector = KpDetector(feature_extractor.feature_dim).cuda().train()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)
    grad_accum = 8
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_iteration // grad_accum, eta_min=0.0005
    )

    for iteration in range(first_iter, train_iteration + 1):
        iter_start.record()
        viewpoint_cam, viewpoint_stack, viewpoint_iter = get_next_train_camera(
            scene, viewpoint_stack, viewpoint_iter
        )
        fine_resolution = get_resolution_from_longest_edge(
            viewpoint_cam.original_image.shape[1],
            viewpoint_cam.original_image.shape[2],
            scene.longest_edge,
        )

        # generate gt_feature_map
        gt_image = viewpoint_cam.original_image.cuda()
        gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

        # get viewmat and K
        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        # get render visible mask
        render_visible_mask = render_visible_masks.get(viewpoint_cam.image_name, None)
        if render_visible_mask is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            render_visible_masks[viewpoint_cam.image_name] = render_visible_mask
        else:
            render_visible_mask = render_visible_masks[viewpoint_cam.image_name]

        # generate gt_map
        gt_map = generate_gt_map(
            gaussians, gt_feature_map, sampled_idx, viewmat, K, render_visible_mask
        )

        # use mask to filter out object
        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            gt_map_mask = (
                F.interpolate(
                    mask[None].float(),
                    size=(gt_map.shape[1], gt_map.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                > 0.5
            )
            gt_map *= gt_map_mask

        # Loss
        heat_map = detector(gt_feature_map)
        loss = score_map_bce_loss(heat_map, gt_map)

        loss.backward()
        if iteration % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            loss_val = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss_val:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == train_iteration:
                progress_bar.close()
            if tb_writer:
                tb_writer.add_scalar(
                    "detector_loss_patches/training_loss", loss_val, iteration
                )
                tb_writer.add_scalar(
                    "detector_loss_patches/lr",
                    optimizer.param_groups[0]["lr"],
                    iteration,
                )

        if iteration in testing_iterations:
            print("\n[ITER {}] Evaluating detector".format(iteration))
            detector.eval()
            evaluate_detector(
                detector,
                feature_extractor,
                gaussians,
                sampled_idx,
                scene,
                masks,
                render_visible_masks,
                tb_writer,
                iteration,
            )
            detector.train()

        if iteration in saving_iterations:
            print("\n[ITER {}] Saving detector".format(iteration))
            torch.save(detector.state_dict(), save_path + f"/{iteration}_detector.pth")


def prepare_output_and_logger(args, folder=None):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    if folder:
        output_path = os.path.join(args.model_path, folder)
    else:
        output_path = args.model_path
    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    seed_everything(2025)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)
    parser.add_argument("--train_detector_iterations", type=int, default=30000)
    parser.add_argument("--stream_cameras", action="store_true", default=False)
    parser.add_argument("--train_only_cameras", action="store_true", default=False)

    args = get_combined_args(parser)
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    if dataset.gaussian_type == "3dgs":
        from scene.gaussian_model import GaussianModel
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        from scene.gaussian_model import GaussianModel_2dgs
        gaussians = GaussianModel_2dgs(dataset.sh_degree)

    masks = None
    if os.path.exists(os.path.join(dataset.source_path, "masks.pkl")):
        import pickle
        masks = pickle.load(open(os.path.join(dataset.source_path, "masks.pkl"), "rb"))

    images_to_read = (
        get_train_image_names(dataset.source_path, dataset.images)
        if args.train_only_cameras
        else None
    )
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        preload_cameras=not args.stream_cameras,
        images_to_read=images_to_read,
        dataloader_num_workers=0 if args.stream_cameras else 4,
        pin_memory=not args.stream_cameras,
    )

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(
            os.path.join(dataset.model_path, args.detector_folder)
        )
    else:
        tb_writer = None

    training_detector(
        gaussians,
        scene,
        masks,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        tb_writer=tb_writer,
        train_iteration=args.train_detector_iterations,
        detector_folder=args.detector_folder,
        landmark_num=args.landmark_num,
        landmark_k=args.landmark_k,
        locability_save_iteration=scene.loaded_iter or args.iteration,
    )

    # All done
    print("\n Scene-specific detector training complete.")
