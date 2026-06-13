#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from loc_gs.localization.stdloc_detector import StdlocKeypointDetector
from loc_gs.scripts.train_ulfloc_scene_detector import (
    _import_ulfloc_feature_extractor,
    _load_image,
    _seed_everything,
)
from loc_gs.training.ulfloc_online_scene_detector import (
    attach_solver_residuals_to_entry,
    attach_superpoint_teacher_to_entry,
    build_online_stdloc_projection_entry,
)
from loc_gs.training.ulfloc_scene_detector import (
    build_stdloc_fullres_detector_loss_terms,
    make_trainable_detector_input,
    rasterize_fullres_scene_detector_components,
)


def _command() -> str:
    return " ".join(shlex.quote(str(part)) for part in sys.argv)


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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _reject_test_split(split_name: str) -> str:
    split = str(split_name).strip()
    lowered = split.lower()
    if lowered == "test" or lowered == "official_test" or lowered.endswith("_test"):
        raise ValueError(f"refusing to train online scene detector from test split: {split}")
    return split or "unknown"


def _load_impact(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"solver impact artifact must be a dict: {path}")
    split = str(payload.get("split_name", payload.get("split", ""))).strip().lower()
    if split == "test" or split == "official_test" or split.endswith("_test"):
        raise ValueError(f"refusing to use solver impact from test split: {path}")
    return payload


def _detector_state_dict_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("detector checkpoint must be a mapping")
    for key in ("state_dict", "detector_state_dict", "model_state_dict"):
        state = payload.get(key)
        if isinstance(state, Mapping):
            return dict(state)
    if all(isinstance(value, torch.Tensor) for value in payload.values()):
        return dict(payload)
    raise KeyError("detector checkpoint must contain state_dict, detector_state_dict, or model_state_dict")


def _read_image_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    names: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        if first.lower() == "visual":
            continue
        names.append(first)
    return names


def _mask_entry(masks: Any, image_name: str) -> Any | None:
    if masks is None or not isinstance(masks, Mapping):
        return None
    candidates = [str(image_name), Path(str(image_name)).name, Path(str(image_name)).stem]
    for candidate in candidates:
        if candidate in masks:
            return masks[candidate]
    return None


def _resize_mask(mask_tensor: Any, *, height: int, width: int) -> torch.Tensor:
    mask = torch.as_tensor(mask_tensor).detach().cpu().float()
    if mask.dim() == 2:
        mask = mask[None, None]
    elif mask.dim() == 3:
        mask = mask[:1][None]
    else:
        raise ValueError(f"unsupported processed mask shape: {tuple(mask.shape)}")
    if tuple(mask.shape[-2:]) != (int(height), int(width)):
        mask = F.interpolate(mask, size=(int(height), int(width)), mode="nearest")
    return (mask[0, 0] > 0.5).bool()


def _load_processed_masks(source_path: Path, images: str) -> tuple[Any | None, Path | None]:
    candidates = [Path(source_path) / str(images) / "masks.pkl", Path(source_path) / "masks.pkl"]
    for path in candidates:
        if path.exists():
            import pickle

            with path.open("rb") as handle:
                return pickle.load(handle), path
    return None, None


def _stable_mask(
    masks: Any,
    image_name: str,
    *,
    height: int,
    width: int,
    mode: str,
) -> tuple[torch.Tensor | None, str]:
    if str(mode) == "none":
        return None, "disabled"
    entry = _mask_entry(masks, image_name)
    if entry is None:
        return None, "missing_image_mask" if masks is not None else "no_masks"
    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
        raise ValueError(f"processed mask entry for {image_name!r} must contain obj, sky, distort channels")
    obj = _resize_mask(entry[0], height=height, width=width)
    sky = _resize_mask(entry[1], height=height, width=width)
    distort = _resize_mask(entry[2], height=height, width=width)
    if str(mode) == "stdloc_object_distort":
        return obj & distort, "object_distort"
    if str(mode) == "object_sky_distort":
        return obj & sky & distort, "object_sky_distort"
    raise ValueError(f"unsupported mask mode: {mode}")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _load_sampled_idx(input_log_dir: Path, config: Mapping[str, Any], sampled_idx_path: Path | None) -> tuple[torch.Tensor, Path]:
    import pickle

    path = sampled_idx_path
    if path is None:
        sample_cfg = config.get("sample", {})
        if not isinstance(sample_cfg, Mapping):
            sample_cfg = {}
        path = Path(input_log_dir) / str(sample_cfg.get("landmark_file_name", "keypoints_sampled_idx.pkl"))
    if not Path(path).exists():
        raise FileNotFoundError(f"sampled idx not found: {path}")
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    sampled = torch.as_tensor(payload, dtype=torch.long).reshape(-1).cpu()
    if int(sampled.numel()) == 0:
        raise ValueError(f"sampled idx is empty: {path}")
    return sampled, Path(path)


def _import_ulfloc_scene_modules(ulf_root: Path) -> dict[str, Any]:
    root = str(Path(ulf_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from gaussian_renderer import get_render_visible_mask  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore
    from utils.graphics_utils import getWorld2View2  # type: ignore

    return {
        "Scene": Scene,
        "GaussianModel": GaussianModel,
        "GaussianModel_2dgs": GaussianModel_2dgs,
        "getWorld2View2": getWorld2View2,
        "get_render_visible_mask": get_render_visible_mask,
    }


def _dataset_namespace(args: argparse.Namespace, config: Mapping[str, Any], source_path: Path) -> argparse.Namespace:
    dense_cfg = config.get("dense", {})
    if not isinstance(dense_cfg, Mapping):
        dense_cfg = {}
    return argparse.Namespace(
        sh_degree=int(args.sh_degree),
        source_path=str(source_path),
        feature_type=str(args.feature_type or config.get("feature_type", "sp")),
        gaussian_type=str(args.gaussian_type or config.get("gaussian_type", "3dgs")),
        model_path=str(Path(args.model_path).resolve()),
        images=str(args.images),
        resolution=int(args.resolution),
        white_background=True,
        longest_edge=int(args.longest_edge),
        data_device=str(args.data_device),
        eval=False,
        speedup=False,
        norm_before_render=bool(dense_cfg.get("norm_before_render", True)),
        render_items=["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
    )


def _resize_image_to_longest_edge(image: torch.Tensor, longest_edge: int) -> torch.Tensor:
    max_edge = int(longest_edge)
    if max_edge <= 0:
        return image
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    current = max(height, width)
    if current <= max_edge:
        return image
    scale = float(max_edge) / float(current)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return F.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )[0]


def _intrinsic_from_camera(camera: Any, *, width: int | None = None, height: int | None = None) -> torch.Tensor:
    width = int(camera.image_width if width is None else width)
    height = int(camera.image_height if height is None else height)
    focal_x = float(width) / (2.0 * torch.tan(torch.tensor(float(camera.FoVx) * 0.5)).item())
    focal_y = float(height) / (2.0 * torch.tan(torch.tensor(float(camera.FoVy) * 0.5)).item())
    return torch.tensor([[focal_x, 0.0, width / 2.0], [0.0, focal_y, height / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _camera_from_info(cam_info: Any, get_world_to_view2: Any) -> argparse.Namespace:
    return argparse.Namespace(
        image_name=str(cam_info.image_name),
        image_width=int(cam_info.width),
        image_height=int(cam_info.height),
        FoVx=float(cam_info.FovX),
        FoVy=float(cam_info.FovY),
        world_view_transform=torch.tensor(
            get_world_to_view2(cam_info.R, cam_info.T, np.array([0.0, 0.0, 0.0]), 1.0),
            dtype=torch.float32,
        ).transpose(0, 1),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online STDLoc-style ULF scene detector training.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_log_dir", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--image_list", default=None, type=Path)
    parser.add_argument("--sampled_idx", default=None, type=Path)
    parser.add_argument("--solver_impact", default=None, type=Path)
    parser.add_argument("--init_checkpoint", default=None, type=Path)
    parser.add_argument("--iterations", default=200, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask_mode", default="stdloc_object_distort", choices=("stdloc_object_distort", "object_sky_distort", "none"))
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--no_render_visible_mask", action="store_true")
    parser.add_argument("--max_points_per_image", default=0, type=int)
    parser.add_argument("--target_sigma_px", default=0.0, type=float)
    parser.add_argument("--background_weight", default=1.0, type=float)
    parser.add_argument("--positive_weight", default=0.0, type=float)
    parser.add_argument("--use_superpoint_teacher", action="store_true")
    parser.add_argument("--no_superpoint_teacher", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--superpoint_teacher_top_k", default=0, type=int)
    parser.add_argument("--solver_feedback_residual_alpha", default=0.0, type=float)
    parser.add_argument("--suppression_weight", default=2.0, type=float)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--gaussian_type", default=None)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--longest_edge", default=1600, type=int)
    parser.add_argument("--data_device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    split_name = _reject_test_split(str(args.split_name))
    _seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("online detector training requires CUDA for ULF FeatureExtractor")
    device = torch.device(args.device)
    config = _load_yaml(Path(args.cfg))
    sampled_idx, sampled_idx_path = _load_sampled_idx(Path(args.input_log_dir), config, args.sampled_idx)
    impact = _load_impact(args.solver_impact)

    from loc_gs.scripts.export_ulfloc_sparse_feedback import resolve_ulfloc_source_path_for_loader

    loader_source = resolve_ulfloc_source_path_for_loader(Path(args.source_path), output_dir / "_ulf_loader_links")
    image_names = _read_image_list(args.image_list)
    imports = _import_ulfloc_scene_modules(Path(args.ulf_root))
    dataset = _dataset_namespace(args, config, loader_source)
    if dataset.gaussian_type == "3dgs":
        gaussians = imports["GaussianModel"](dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = imports["GaussianModel_2dgs"](dataset.sh_degree)
    else:
        raise ValueError(f"unsupported gaussian_type: {dataset.gaussian_type}")
    scene_obj = imports["Scene"](
        dataset,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        images_to_read=image_names,
        preload_cameras=False,
    )
    camera_infos = list(scene_obj.scene_info.train_cameras)
    if int(args.max_views) > 0:
        camera_infos = camera_infos[: int(args.max_views)]
    if not camera_infos:
        raise ValueError("no train cameras available for online detector training")

    FeatureExtractor = _import_ulfloc_feature_extractor(Path(args.ulf_root))
    feature_extractor = FeatureExtractor(str(args.feature_type)).cuda().eval()
    detector = StdlocKeypointDetector(in_dim=256).to(device)
    if args.init_checkpoint is not None:
        init_payload = torch.load(Path(args.init_checkpoint), map_location=device)
        detector.load_state_dict(_detector_state_dict_from_checkpoint(init_payload), strict=True)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=float(args.lr), weight_decay=1e-4)
    gaussian_xyz = torch.as_tensor(gaussians.get_xyz, dtype=torch.float32).detach().cpu()
    opacity = None
    if hasattr(gaussians, "get_opacity"):
        opacity = torch.as_tensor(gaussians.get_opacity, dtype=torch.float32).detach().cpu().reshape(-1).clamp(0.0, 1.0)
    masks, mask_path = (None, None) if bool(args.no_masks) else _load_processed_masks(Path(args.source_path), str(args.images))

    metrics_accum = {
        "loss_sum": 0.0,
        "kept_projection_count": 0,
        "sampled_projection_count": 0,
        "dropped_render_invisible_count": 0,
        "mask_invalid_projection_count": 0,
        "solver_positive_count": 0,
        "negative_count": 0,
        "sp_teacher_count": 0,
    }
    sparse_cfg = config.get("sparse", {})
    if not isinstance(sparse_cfg, Mapping):
        sparse_cfg = {}
    sp_teacher_top_k = int(args.superpoint_teacher_top_k)
    if sp_teacher_top_k <= 0:
        sp_teacher_top_k = int(sparse_cfg.get("kpts_num", 2048))
    use_superpoint_teacher = bool(args.use_superpoint_teacher) and not bool(args.no_superpoint_teacher)
    for step in range(int(args.iterations)):
        cam_info = random.choice(camera_infos)
        camera = _camera_from_info(cam_info, imports["getWorld2View2"])
        image_id = str(camera.image_name)
        image_path = Path(args.source_path) / str(args.images) / image_id
        image = _resize_image_to_longest_edge(_load_image(image_path, device), int(args.longest_edge))
        canvas_height = int(image.shape[-2])
        canvas_width = int(image.shape[-1])
        with torch.no_grad():
            feature_map, _scores = feature_extractor.detectAndComputeDense(image[None])
            feature_map = torch.as_tensor(feature_map[0], dtype=torch.float32, device=device)
            feature_map = F.interpolate(
                feature_map.unsqueeze(0),
                size=(canvas_height, canvas_width),
                mode="bilinear",
                align_corners=False,
            )[0]
            feature_map = F.normalize(feature_map, p=2, dim=0)
            feature_map = make_trainable_detector_input(feature_map).to(device=device)
            sp_teacher = None
            if use_superpoint_teacher and str(args.feature_type) == "sp":
                sp_teacher = feature_extractor.detectAndCompute(image[None], top_k=sp_teacher_top_k)[0]
        stable_mask, mask_status = _stable_mask(
            masks,
            image_id,
            height=canvas_height,
            width=canvas_width,
            mode=str(args.mask_mode),
        )
        render_visible = None
        if not bool(args.no_render_visible_mask):
            render_visible = imports["get_render_visible_mask"](
                gaussians,
                camera,
                canvas_width,
                canvas_height,
            ).detach().cpu()
        entry, projection_metrics = build_online_stdloc_projection_entry(
            gaussian_xyz=gaussian_xyz,
            sampled_idx=sampled_idx,
            world_to_camera=torch.as_tensor(camera.world_view_transform, dtype=torch.float32).transpose(0, 1),
            intrinsic=_intrinsic_from_camera(camera, width=canvas_width, height=canvas_height),
            height=canvas_height,
            width=canvas_width,
            image_id=image_id,
            render_visible_mask=render_visible,
            stable_mask=stable_mask,
            opacity=opacity,
            max_points_per_image=int(args.max_points_per_image),
        )
        entry, sp_metrics = attach_superpoint_teacher_to_entry(
            entry,
            sp_teacher,
            height=canvas_height,
            width=canvas_width,
        )
        entry, residual_metrics = attach_solver_residuals_to_entry(entry, impact, image_id=image_id)
        target, solver_positive, suppression, _target_meta = rasterize_fullres_scene_detector_components(
            entry,
            target_height=int(feature_map.shape[-2]),
            target_width=int(feature_map.shape[-1]),
            sigma_px=float(args.target_sigma_px),
        )
        prediction = detector(feature_map)
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)
        terms = build_stdloc_fullres_detector_loss_terms(
            prediction,
            base_target=target.to(device=device),
            solver_positive_target=solver_positive.to(device=device),
            solver_negative_target=suppression.to(device=device),
            residual_alpha=float(args.solver_feedback_residual_alpha),
            solver_negative_weight=float(args.suppression_weight),
            background_weight=float(args.background_weight),
            positive_weight=float(args.positive_weight),
        )
        loss = terms["total_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metrics_accum["loss_sum"] += float(loss.detach().item())
        for key in ("kept_projection_count", "sampled_projection_count", "dropped_render_invisible_count", "mask_invalid_projection_count"):
            metrics_accum[key] += int(projection_metrics.get(key, 0))
        for key in ("solver_positive_count", "negative_count"):
            metrics_accum[key] += int(residual_metrics.get(key, 0))
        metrics_accum["sp_teacher_count"] += int(sp_metrics.get("sp_teacher_count", 0))
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == int(args.iterations):
            _append_jsonl(
                progress_path,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "step": int(step + 1),
                    "loss": float(loss.detach().item()),
                    "image_id": image_id,
                    "mask_status": mask_status,
                    **{f"projection_{k}": int(v) for k, v in projection_metrics.items() if isinstance(v, int)},
                    **{f"residual_{k}": int(v) for k, v in residual_metrics.items()},
                    **{f"sp_teacher_{k}": int(v) for k, v in sp_metrics.items()},
                },
            )

    checkpoint = {
        "schema_version": "ulfloc_online_scene_detector_checkpoint_v1",
        "state_dict": detector.state_dict(),
        "metadata": {
            "scene_detector_mode": "stdloc_fullres",
            "training_mode": "online_stdloc_projection",
            "split_name": split_name,
            "solver_feedback_residual_alpha": float(args.solver_feedback_residual_alpha),
            "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
            "superpoint_teacher_enabled": bool(use_superpoint_teacher),
            "superpoint_teacher_top_k": int(sp_teacher_top_k),
        },
    }
    checkpoint_path = output_dir / "scene_detector.pth"
    torch.save(checkpoint, checkpoint_path)
    total_steps = max(1, int(args.iterations))
    metrics = {
        "schema_version": "ulfloc_online_scene_detector_training_metrics_v1",
        "scene": str(args.scene),
        "split_name": split_name,
        "training_mode": "online_stdloc_projection",
        "checkpoint": str(checkpoint_path),
        "iterations": int(args.iterations),
        "train_camera_count": int(len(camera_infos)),
        "sampled_landmark_count": int(sampled_idx.numel()),
        "final_mean_loss": float(metrics_accum["loss_sum"] / total_steps),
        "mean_kept_projection_count": float(metrics_accum["kept_projection_count"] / total_steps),
        "mean_sampled_projection_count": float(metrics_accum["sampled_projection_count"] / total_steps),
        "mean_dropped_render_invisible_count": float(metrics_accum["dropped_render_invisible_count"] / total_steps),
        "mean_mask_invalid_projection_count": float(metrics_accum["mask_invalid_projection_count"] / total_steps),
        "mean_solver_positive_count": float(metrics_accum["solver_positive_count"] / total_steps),
        "mean_negative_count": float(metrics_accum["negative_count"] / total_steps),
        "mean_sp_teacher_count": float(metrics_accum["sp_teacher_count"] / total_steps),
        "mask_mode": str(args.mask_mode),
        "mask_path": None if mask_path is None else str(mask_path),
        "render_visible_enabled": not bool(args.no_render_visible_mask),
        "solver_impact": None if args.solver_impact is None else str(args.solver_impact),
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
    }
    split_audit = {
        "schema_version": "ulfloc_online_scene_detector_split_audit_v1",
        "audit_status": "passed",
        "split_name": split_name,
        "test_split_used": False,
        "official_test_used": False,
    }
    manifest = {
        "schema_version": "ulfloc_online_scene_detector_manifest_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": _command(),
        "scene": str(args.scene),
        "split_name": split_name,
        "checkpoint_path": str(args.model_path),
        "map_path": str(args.model_path),
        "data_roots": [str(args.source_path)],
        "sampled_idx_path": str(sampled_idx_path),
        "hyperparameters": {
            "iterations": int(args.iterations),
            "lr": float(args.lr),
            "mask_mode": str(args.mask_mode),
            "solver_feedback_residual_alpha": float(args.solver_feedback_residual_alpha),
            "max_points_per_image": int(args.max_points_per_image),
            "target_sigma_px": float(args.target_sigma_px),
            "background_weight": float(args.background_weight),
            "positive_weight": float(args.positive_weight),
            "superpoint_teacher_enabled": bool(use_superpoint_teacher),
            "superpoint_teacher_top_k": int(sp_teacher_top_k),
            "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        },
        "feedback_enabled": {
            "residual": args.solver_impact is not None and float(args.solver_feedback_residual_alpha) > 0.0,
            "selector": False,
            "rho": False,
        },
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "metrics_summary": str(output_dir / "metrics_summary.json"),
            "split_audit": str(output_dir / "split_audit.json"),
        },
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
