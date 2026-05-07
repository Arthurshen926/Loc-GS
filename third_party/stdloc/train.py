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
import math
from itertools import islice
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import pickle
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from gaussian_renderer import render_from_pose_gsplat, render_gsplat
from scene import Scene
from utils.general_utils import safe_state, seed_everything
from utils.image_utils import psnr
from utils.localization_loss import (
    keypoint_reprojection_loss,
    locability_weighted_feature_loss,
    pose_guided_reprojection_loss,
)
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor
from train_detector import training_detector


def get_intrinsic_from_fov(fovx, fovy, width, height, device="cuda"):
    focal_x = width / (2 * math.tan(fovx * 0.5))
    focal_y = height / (2 * math.tan(fovy * 0.5))
    return torch.tensor(
        [
            [focal_x, 0.0, width / 2.0],
            [0.0, focal_y, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def perturb_pose_w2c(pose_w2c, trans_std=0.05, rot_deg=3.0):
    if trans_std <= 0 and rot_deg <= 0:
        return pose_w2c

    device = pose_w2c.device
    dtype = pose_w2c.dtype
    axis = torch.randn(3, device=device, dtype=dtype)
    axis = axis / axis.norm().clamp_min(1e-6)
    angle = torch.randn((), device=device, dtype=dtype) * (float(rot_deg) * torch.pi / 180.0)
    skew = torch.zeros((3, 3), device=device, dtype=dtype)
    skew[0, 1] = -axis[2]
    skew[0, 2] = axis[1]
    skew[1, 0] = axis[2]
    skew[1, 2] = -axis[0]
    skew[2, 0] = -axis[1]
    skew[2, 1] = axis[0]
    eye = torch.eye(3, device=device, dtype=dtype)
    delta_R = eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)
    delta = torch.eye(4, device=device, dtype=dtype)
    delta[:3, :3] = delta_R
    delta[:3, 3] = torch.randn(3, device=device, dtype=dtype) * float(trans_std)
    return delta @ pose_w2c


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


def training(
    dataset,
    opt,
    testing_iterations,
    saving_iterations,
    checkpoint,
    train_detector=True,
    test_detector_iterations=[10_000, 20_000, 30_000],
    save_detector_iterations=[10_000, 20_000, 30_000],
    detector_folder="detector",
    train_detector_iterations=30000,
    landmark_num=16384,
    landmark_k=32,
    stream_cameras=False,
    train_only_cameras=False,
    load_iteration=None,
):
    print(opt)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    print("Feature type:", dataset.feature_type)
    print("Gaussian type:", dataset.gaussian_type)
    if dataset.gaussian_type == "3dgs":
        from scene.gaussian_model import GaussianModel

        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        from scene.gaussian_model import GaussianModel_2dgs

        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError("Gaussian type not supported")

    images_to_read = (
        get_train_image_names(dataset.source_path, dataset.images)
        if train_only_cameras
        else None
    )
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=load_iteration,
        preload_cameras=not stream_cameras,
        images_to_read=images_to_read,
        dataloader_num_workers=0 if stream_cameras else 4,
        pin_memory=not stream_cameras,
    )
    # scene = Scene(dataset, gaussians, num=10)

    # load masks
    masks = None
    if os.path.exists(os.path.join(dataset.source_path, dataset.images, "masks.pkl")):
        print(
            "Loading masks from",
            os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        )
        masks = pickle.load(
            open(os.path.join(dataset.source_path, dataset.images, "masks.pkl"), "rb")
        )

    viewpoint_stack = None
    viewpoint_iter = None
    viewpoint_cam, viewpoint_stack, viewpoint_iter = get_next_train_camera(
        scene, viewpoint_stack, viewpoint_iter
    )

    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    initial_xyz = gaussians.get_xyz.detach().clone()
    initial_scaling = gaussians._scaling.detach().clone()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    viewpoint_iter = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Feature Gaussian")
    first_iter += 1

    # Training loop
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        viewpoint_cam, viewpoint_stack, viewpoint_iter = get_next_train_camera(
            scene, viewpoint_stack, viewpoint_iter
        )

        # Render
        render_pkg = render_gsplat(
            viewpoint_cam,
            gaussians,
            background,
            rgb_only=False,
            norm_feat_bf_render=dataset.norm_before_render,
            longest_edge=dataset.longest_edge,
            rasterize_mode="antialiased",
        )

        feature_map, image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["feature_map"],
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        # make sure ground truth image is the same size as the rendered image
        original_image = viewpoint_cam.original_image.cuda()
        gt_image = F.interpolate(
            original_image.unsqueeze(0),
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if masks is not None:
            # use mask
            obj_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            sky_mask = masks[viewpoint_cam.image_name][1].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

            # mask obj and distort border
            mask = obj_mask & distort_mask

            image = image * mask
            gt_image = gt_image * mask
            # mask sky
            gt_image[sky_mask.repeat(3, 1, 1) == False] = 1  # 全白

        Ll1 = l1_loss(image, gt_image)

        if feature_map is not None:
            gt_feature_out = feature_extractor(original_image[None])
            gt_feature_map = gt_feature_out["feature_map"][0]
            selective_stats = None

            # resize feature map to gt size
            gt_feature_map = F.interpolate(
                gt_feature_map.unsqueeze(0),
                size=(feature_map.shape[1], feature_map.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

            # mask feature map
            if masks is not None:
                feature_map_mask = (
                    F.interpolate(
                        mask[None].float(),
                        size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    > 0.5
                )
                feature_map *= feature_map_mask
                gt_feature_map *= feature_map_mask

            Ll1_feature = l1_loss(feature_map, gt_feature_map)
            Lselective_recon = torch.tensor(0.0, device="cuda")
            if (
                getattr(opt, "selective_recon_weight", 0.0) > 0
                and render_pkg.get("locability_map") is not None
            ):
                Lselective_recon, _selective_stats = locability_weighted_feature_loss(
                    feature_map,
                    gt_feature_map,
                    render_pkg["locability_map"],
                    min_weight=getattr(opt, "selective_recon_min_weight", 0.1),
                    gamma=getattr(opt, "selective_recon_gamma", 1.0),
                    top_ratio=getattr(opt, "selective_recon_top_ratio", 0.0),
                    mask=feature_map_mask if masks is not None else None,
                )
                selective_stats = _selective_stats
            Lkeypoint_loc = torch.tensor(0.0, device="cuda")
            Lpose_loc = torch.tensor(0.0, device="cuda")
            if (
                getattr(opt, "lambda_keypoint_loc", 0.0) > 0
                and "scores" in gt_feature_out
            ):
                score_map = gt_feature_out["scores"][0]
                if masks is not None:
                    score_mask = F.interpolate(
                        feature_map_mask.float()[None],
                        size=score_map.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                    score_map = score_map * score_mask
                Lkeypoint_loc, _loc_stats = keypoint_reprojection_loss(
                    feature_map,
                    gt_feature_map,
                    score_map,
                    max_keypoints=getattr(opt, "keypoint_loc_max", 64),
                    temperature=getattr(opt, "keypoint_loc_temperature", 0.07),
                    min_score=getattr(opt, "keypoint_loc_min_score", 0.0),
                )
            if (
                getattr(opt, "lambda_pose_loc", 0.0) > 0
                and "scores" in gt_feature_out
            ):
                score_map = gt_feature_out["scores"][0]
                pose_w2c = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()
                render_pose_w2c = perturb_pose_w2c(
                    pose_w2c,
                    trans_std=getattr(opt, "pose_loc_noise_trans", 0.05),
                    rot_deg=getattr(opt, "pose_loc_noise_rot_deg", 3.0),
                )
                pose_render = render_from_pose_gsplat(
                    gaussians,
                    render_pose_w2c,
                    viewpoint_cam.FoVx,
                    viewpoint_cam.FoVy,
                    image.shape[2],
                    image.shape[1],
                    background,
                    rgb_only=False,
                    norm_feat_bf_render=dataset.norm_before_render,
                    render_mode="RGB+ED",
                    rasterize_mode="antialiased",
                )
                if pose_render["feature_map"] is not None and pose_render["depth"] is not None:
                    K_pose = get_intrinsic_from_fov(
                        viewpoint_cam.FoVx,
                        viewpoint_cam.FoVy,
                        image.shape[2],
                        image.shape[1],
                    )
                    Lpose_loc, _pose_stats = pose_guided_reprojection_loss(
                        pose_render["feature_map"],
                        gt_feature_map,
                        score_map,
                        pose_render["depth"],
                        render_pose_w2c,
                        pose_w2c,
                        K_pose,
                        locability_map=pose_render.get("locability_map"),
                        locability_weight=getattr(opt, "pose_loc_locability_weight", 0.0),
                        max_keypoints=getattr(opt, "pose_loc_max", 64),
                        temperature=getattr(opt, "pose_loc_temperature", 0.07),
                        target_sigma_px=getattr(opt, "pose_loc_target_sigma_px", 2.0),
                        min_score=getattr(opt, "keypoint_loc_min_score", 0.0),
                        min_depth=getattr(opt, "pose_loc_min_depth", 0.05),
                        max_depth=getattr(opt, "pose_loc_max_depth", 100.0),
                    )
        else:
            Ll1_feature = torch.tensor(0.0, device="cuda")
            Lselective_recon = torch.tensor(0.0, device="cuda")
            Lkeypoint_loc = torch.tensor(0.0, device="cuda")
            Lpose_loc = torch.tensor(0.0, device="cuda")
            selective_stats = None

        # overall loss
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            + 1.0 * Ll1_feature
            + getattr(opt, "selective_recon_weight", 0.0) * Lselective_recon
            + getattr(opt, "lambda_keypoint_loc", 0.0) * Lkeypoint_loc
            + getattr(opt, "lambda_pose_loc", 0.0) * Lpose_loc
        )
        if getattr(opt, "geometry_anchor_weight", 0.0) > 0:
            xyz_anchor = (gaussians.get_xyz - initial_xyz).abs().mean()
            loss = loss + float(opt.geometry_anchor_weight) * xyz_anchor
        if getattr(opt, "geometry_scale_anchor_weight", 0.0) > 0:
            scale_anchor = (gaussians._scaling - initial_scaling).abs().mean()
            loss = loss + float(opt.geometry_scale_anchor_weight) * scale_anchor

        # regularization for 2DGS
        if dataset.gaussian_type == "2dgs":
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            rend_alpha = render_pkg['rend_alpha']
            # ------------------------------
            surf_normal *= rend_alpha.squeeze(0).detach()
            rend_normal = rend_normal.squeeze(0).permute(2, 0, 1)
            if len(surf_normal.shape) == 4:
                surf_normal = surf_normal.squeeze(0)
            surf_normal = surf_normal.permute(2, 0, 1)
            # ------------------------------
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            if masks is not None:
                normal_error *= mask
                rend_dist = rend_dist.squeeze(-1)
                rend_dist *= mask
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            loss = loss + dist_loss + normal_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                Ll1_feature,
                Lkeypoint_loc,
                Lpose_loc,
                Lselective_recon,
                selective_stats,
                loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                background,
                dataset,
                feature_extractor=feature_extractor,
                masks=masks,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration), flush=True)
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats_gsplat(
                    viewspace_point_tensor,
                    visibility_filter,
                    image.shape[2],
                    image.shape[1],
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    if train_detector:
        if tb_writer is not None:
            tb_writer.close()
            tb_writer = SummaryWriter(
                os.path.join(dataset.model_path, args.detector_folder)
            )
        training_detector(
            gaussians,
            scene,
            masks,
            testing_iterations=test_detector_iterations,
            saving_iterations=save_detector_iterations,
            tb_writer=tb_writer,
            train_iteration=train_detector_iterations,
            detector_folder=detector_folder,
            landmark_num=landmark_num,
            landmark_k=landmark_k,
            locability_save_iteration=opt.iterations,
        )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    Ll1_feature,
    Lkeypoint_loc,
    Lpose_loc,
    Lselective_recon,
    selective_stats,
    loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    background,
    dataset,
    feature_extractor,
    masks=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(
            "train_loss_patches/l1_loss_feature", Ll1_feature.item(), iteration
        )
        tb_writer.add_scalar(
            "train_loss_patches/keypoint_loc_loss", Lkeypoint_loc.item(), iteration
        )
        tb_writer.add_scalar(
            "train_loss_patches/pose_loc_loss", Lpose_loc.item(), iteration
        )
        tb_writer.add_scalar(
            "train_loss_patches/selective_recon_loss",
            Lselective_recon.item(),
            iteration,
        )
        if selective_stats is not None:
            tb_writer.add_scalar(
                "train_loss_patches/selective_recon_selected_fraction",
                selective_stats["selected_fraction"].item(),
                iteration,
            )
            for key, value in selective_stats.items():
                tb_writer.add_scalar(
                    f"train_loss_patches/selective_recon_{key}",
                    value.item(),
                    iteration,
                )
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": select_report_train_cameras(scene),
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                l1_feature_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = render_gsplat(
                        viewpoint,
                        scene.gaussians,
                        background,
                        rgb_only=False,
                        norm_feat_bf_render=dataset.norm_before_render,
                        longest_edge=dataset.longest_edge,
                        rasterize_mode="antialiased",
                    )

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    feature_map = render_pkg["feature_map"]

                    original_image = viewpoint.original_image.cuda()

                    gt_image = F.interpolate(
                        original_image.unsqueeze(0),
                        size=(image.shape[1], image.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    gt_feature_map = feature_extractor(original_image[None])[
                        "feature_map"
                    ][0]

                    if masks is not None:
                        mask = masks[viewpoint.image_name][0].cuda()[None]
                        sky_mask = masks[viewpoint.image_name][1].cuda()[None]
                        distort_mask = masks[viewpoint.image_name][2].cuda()[None]
                        mask = mask & distort_mask

                    if feature_map is not None:
                        gt_feature_map = F.interpolate(
                            gt_feature_map.unsqueeze(0),
                            size=(feature_map.shape[1], feature_map.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
                        if masks is not None:
                            feature_map_mask_float = F.interpolate(
                                mask[None].float(),
                                size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)
                            feature_map_mask = feature_map_mask_float > 0.5
                            feature_map_loss = feature_map * feature_map_mask
                            gt_feature_map_loss = gt_feature_map * feature_map_mask
                        else:
                            feature_map_loss = feature_map
                            gt_feature_map_loss = gt_feature_map
                        l1_feature_test += (
                            l1_loss(feature_map_loss, gt_feature_map_loss)
                            .mean()
                            .double()
                        )

                    if masks is not None:
                        image_loss = image * mask
                        gt_image_loss = gt_image * mask
                        # mask sky
                        gt_image_loss[sky_mask.repeat(3, 1, 1) == False] = 1  # 全白
                    else:
                        image_loss = image
                        gt_image_loss = gt_image

                    l1_test += l1_loss(image_loss, gt_image_loss).mean().double()
                    psnr_test += psnr(image_loss, gt_image_loss).mean().double()

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                            if masks is not None:
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}/mask".format(viewpoint.image_name),
                                    mask[None],
                                    global_step=iteration,
                                )
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}/sky_mask".format(viewpoint.image_name),
                                    sky_mask[None],
                                    global_step=iteration,
                                )

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                l1_feature_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} feature L1 {}".format(
                        iteration, config["name"], l1_test, psnr_test, l1_feature_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss_feature",
                        l1_feature_test,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    seed_everything(2025)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--train_detector", action="store_true", default=False)
    parser.add_argument(
        "--test_detector_iterations", nargs="+", type=int, default=[7000, 30000]
    )
    parser.add_argument(
        "--save_detector_iterations", nargs="+", type=int, default=[7000, 30000]
    )
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--train_detector_iterations", type=int, default=30000)
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--stream_cameras", action="store_true", default=False)
    parser.add_argument("--train_only_cameras", action="store_true", default=False)
    parser.add_argument("--load_iteration", type=int, default=None)

    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.start_checkpoint,
        args.train_detector,
        args.test_detector_iterations,
        args.save_detector_iterations,
        args.detector_folder,
        args.train_detector_iterations,
        args.landmark_num,
        args.landmark_k,
        args.stream_cameras,
        args.train_only_cameras,
        args.load_iteration,
    )

    # All done
    print("\nTraining complete.")
