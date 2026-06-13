#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from loc_gs.training.ulfloc_visibility_teacher import (
    SCHEMA_VERSION,
    build_visibility_teacher_targets_with_metrics,
    validate_visibility_teacher_split_name,
)

REAL_PROJECTION_SOURCE = "ulfloc_train_camera_sampled_projection"


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _normalize_projections(payload: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    projections = payload.get("projections")
    if isinstance(projections, Mapping):
        normalized: dict[str, list[Mapping[str, Any]]] = {}
        for image_id, rows in projections.items():
            if not isinstance(rows, list):
                raise TypeError(f"projection rows for {image_id!r} must be a list")
            normalized[str(image_id)] = [row for row in rows if isinstance(row, Mapping)]
        return normalized

    records = payload.get("records")
    if isinstance(records, list):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in records:
            if not isinstance(row, Mapping):
                continue
            image_id = str(row.get("image_id", row.get("source_view_id", row.get("view_id", "")))).strip()
            if not image_id:
                raise ValueError("records projection JSON rows must include image_id, source_view_id, or view_id")
            grouped.setdefault(image_id, []).append(row)
        return grouped

    raise KeyError("input projections JSON must contain a projections mapping or records list")


def _payload_split_name(payload: Mapping[str, Any]) -> str:
    raw = str(payload.get("split_name", payload.get("split", ""))).strip()
    return raw if raw else "unknown"


def _resolve_split_name(cli_split_name: str | None, payload: Mapping[str, Any]) -> tuple[str, str]:
    source_split = _payload_split_name(payload)
    if source_split.lower() == "test":
        raise ValueError("refusing to build ULF-Loc visibility teacher from test split projections")
    if cli_split_name is not None and str(cli_split_name).strip().lower() == "test":
        raise ValueError("test split is not allowed for ULF-Loc visibility teacher targets")
    split = str(cli_split_name).strip() if cli_split_name is not None and str(cli_split_name).strip() else source_split
    split = validate_visibility_teacher_split_name(split)
    if source_split != "unknown" and split != source_split:
        raise ValueError(f"split_name mismatch: CLI {split!r} != projections JSON {source_split!r}")
    return split, source_split


def _resolve_int_dimension(name: str, cli_value: int | None, payload: Mapping[str, Any]) -> int:
    value = cli_value if cli_value is not None else payload.get(name)
    if value is None:
        raise ValueError(f"{name} is required, either as --{name} or in the input projections JSON")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _data_roots(args: argparse.Namespace, payload: Mapping[str, Any]) -> list[str]:
    roots = [str(path) for path in (args.data_root or [])]
    if roots:
        return roots
    raw_roots = payload.get("data_roots", payload.get("data_root", []))
    if isinstance(raw_roots, (str, Path)):
        return [str(raw_roots)]
    if isinstance(raw_roots, list):
        return [str(item) for item in raw_roots]
    return []


def _path_or_payload(args: argparse.Namespace, attr: str, payload: Mapping[str, Any]) -> str:
    value = getattr(args, attr)
    if value is not None:
        return str(value)
    payload_value = payload.get(attr)
    return "unknown" if payload_value is None else str(payload_value)


def _split_audit(*, split_name: str, source_split_name: str, projection_source: str) -> dict[str, Any]:
    status = "passed" if source_split_name != "unknown" else "unknown"
    return {
        "schema_version": "ulfloc_visibility_teacher_split_audit_v1",
        "audit_status": status,
        "split_name": split_name,
        "source_split_name": source_split_name,
        "test_split_used": False,
        "official_test_used": False,
        "projection_source": projection_source,
        "notes": "Visibility teacher builder rejects test split inputs and stores compact point targets.",
    }


def _real_projection_split_audit(*, split_name: str, train_camera_count: int) -> dict[str, Any]:
    return {
        "schema_version": "ulfloc_visibility_teacher_split_audit_v1",
        "audit_status": "passed",
        "split_name": split_name,
        "source_split_name": split_name,
        "test_split_used": False,
        "official_test_used": False,
        "projection_source": REAL_PROJECTION_SOURCE,
        "camera_source": "ULF-Loc Scene.getTrainCameras",
        "dataset_eval": False,
        "train_camera_count": int(train_camera_count),
        "notes": "Real projection mode forces dataset.eval=False, iterates train cameras only, and rejects split_name=test.",
    }


def _manifest(
    *,
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    repo_root: Path,
    split_name: str,
    source_split_name: str,
    scene: str,
    height: int,
    width: int,
    metrics: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    projection_source = "input_projections_json"
    return {
        "schema_version": "ulfloc_visibility_teacher_manifest_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": _command(),
        "scene": scene,
        "split": split_name,
        "split_name": split_name,
        "source_split_name": source_split_name,
        "checkpoint_path": _path_or_payload(args, "checkpoint_path", payload),
        "map_path": _path_or_payload(args, "map_path", payload),
        "data_roots": _data_roots(args, payload),
        "hyperparameters": {
            "height": int(height),
            "width": int(width),
            "projection_source": projection_source,
            "target_storage": "points",
            "max_points_per_image": int(args.max_points_per_image),
        },
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": False,
        "rho_feedback_enabled": False,
        "feedback_enabled": {
            "residual": False,
            "selector": False,
            "rho": False,
        },
        "input_projections_json": None if args.input_projections_json is None else str(args.input_projections_json),
        "output_dir": str(args.output_dir),
        "metrics": dict(metrics),
        "split_audit": dict(split_audit),
        "outputs": {
            "visibility_teacher": str(Path(args.output_dir) / "visibility_teacher.pt"),
            "metrics_summary": str(Path(args.output_dir) / "metrics_summary.json"),
            "split_audit": str(Path(args.output_dir) / "split_audit.json"),
        },
    }


def _real_projection_manifest(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    ulf_root: Path,
    split_name: str,
    scene: str,
    metrics: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    sampled_idx_path: Path,
    loader_source_path: Path,
    original_source_path: Path,
    mask_path: Path | None,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    return {
        "schema_version": "ulfloc_visibility_teacher_manifest_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "ulf_git_commit": _git_commit(ulf_root),
        "command": _command(),
        "scene": scene,
        "split": split_name,
        "split_name": split_name,
        "source_split_name": split_name,
        "checkpoint_path": str(Path(args.model_path)),
        "map_path": str(Path(args.model_path)),
        "data_roots": [str(original_source_path)],
        "hyperparameters": {
            "projection_source": REAL_PROJECTION_SOURCE,
            "target_storage": "points",
            "camera_source": "ULF-Loc Scene.getTrainCameras",
            "dataset_eval": False,
            "iteration": int(args.iteration),
            "images": str(args.images),
            "feature_type": str(args.feature_type),
            "gaussian_type": str(args.gaussian_type),
            "sh_degree": int(args.sh_degree),
            "resolution": int(args.resolution),
            "longest_edge": int(args.longest_edge),
            "data_device": str(args.data_device),
            "max_views": int(args.max_views),
            "image_list": None if args.image_list is None else str(args.image_list),
            "max_points_per_image": int(args.max_points_per_image),
            "sampled_idx_count": int(metrics.get("sampled_idx_count", 0)),
            "mask_path": None if mask_path is None else str(mask_path),
            "processed_masks_enabled": not bool(args.no_masks),
        },
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": False,
        "rho_feedback_enabled": False,
        "feedback_enabled": {
            "residual": False,
            "selector": False,
            "rho": False,
        },
        "input_projections_json": None,
        "cfg": str(Path(args.cfg)),
        "input_log_dir": str(Path(args.input_log_dir)),
        "sampled_idx_path": str(sampled_idx_path),
        "source_path": str(loader_source_path),
        "original_source_path": str(original_source_path),
        "model_path": str(Path(args.model_path)),
        "ulf_root": str(ulf_root),
        "output_dir": str(output_dir),
        "metrics": dict(metrics),
        "split_audit": dict(split_audit),
        "outputs": {
            "visibility_teacher": str(output_dir / "visibility_teacher.pt"),
            "metrics_summary": str(output_dir / "metrics_summary.json"),
            "split_audit": str(output_dir / "split_audit.json"),
        },
    }


def _build_from_input_projections_json(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    payload = _load_json(Path(args.input_projections_json))
    split_name, source_split_name = _resolve_split_name(args.split_name, payload)
    height = _resolve_int_dimension("height", args.height, payload)
    width = _resolve_int_dimension("width", args.width, payload)
    scene = str(args.scene or payload.get("scene", "unknown"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projections = _normalize_projections(payload)
    targets, metrics = build_visibility_teacher_targets_with_metrics(
        projections,
        height=height,
        width=width,
        max_points_per_image=int(args.max_points_per_image),
    )
    metrics = dict(metrics)
    metrics.update(
        {
            "projection_source": "input_projections_json",
            "scene": scene,
            "split_name": split_name,
            "source_split_name": source_split_name,
        }
    )
    split_audit = _split_audit(
        split_name=split_name,
        source_split_name=source_split_name,
        projection_source="input_projections_json",
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "split_name": split_name,
        "targets": targets,
        "metadata": metrics,
        "split_audit": split_audit,
    }

    torch.save(artifact, output_dir / "visibility_teacher.pt")
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(
        output_dir / "manifest.json",
        _manifest(
            args=args,
            payload=payload,
            repo_root=repo_root,
            split_name=split_name,
            source_split_name=source_split_name,
            scene=scene,
            height=height,
            width=width,
            metrics=metrics,
            split_audit=split_audit,
        ),
    )
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")

    print(
        json.dumps(
            {
                "visibility_teacher": str(output_dir / "visibility_teacher.pt"),
                "metrics_path": str(output_dir / "metrics_summary.json"),
                "kept_projection_count": int(metrics["kept_projection_count"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _require_real_projection_args(args: argparse.Namespace) -> None:
    if getattr(args, "cfg", None) is None and getattr(args, "config", None) is not None:
        args.cfg = args.config
    if getattr(args, "config", None) is None and getattr(args, "cfg", None) is not None:
        args.config = args.cfg
    missing = [
        name
        for name in ("scene", "ulf_root", "model_path", "source_path", "images", "input_log_dir", "cfg", "output_dir", "split_name")
        if getattr(args, name) is None or str(getattr(args, name)).strip() == ""
    ]
    if missing:
        formatted = ", ".join(f"--{name}" for name in missing)
        raise ValueError(f"real ULF projection mode requires {formatted}")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ULF config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ULF config must contain a YAML mapping: {path}")
    return payload


def _sampled_idx_path(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    if args.sampled_idx is not None:
        return Path(args.sampled_idx)
    sample_cfg = config.get("sample", {})
    if not isinstance(sample_cfg, Mapping):
        sample_cfg = {}
    name = str(sample_cfg.get("landmark_file_name", "keypoints_sampled_idx.pkl"))
    return Path(args.input_log_dir) / name


def _load_sampled_idx(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[torch.Tensor, Path]:
    input_log_dir = Path(args.input_log_dir)
    if not input_log_dir.exists():
        raise FileNotFoundError(f"input_log_dir not found: {input_log_dir}")
    path = _sampled_idx_path(args, config)
    if not path.exists():
        raise FileNotFoundError(f"sampled Gaussian index file not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    sampled_idx = torch.as_tensor(payload, dtype=torch.long).reshape(-1).cpu()
    if int(sampled_idx.numel()) == 0:
        raise ValueError(f"sampled Gaussian index file is empty: {path}")
    if bool((sampled_idx < 0).any().item()):
        raise ValueError(f"sampled Gaussian index file contains negative ids: {path}")
    return sampled_idx, path


def _import_ulfloc_projection_modules(ulf_root: Path) -> dict[str, Any]:
    if not ulf_root.exists():
        raise FileNotFoundError(f"ULF-Loc root not found: {ulf_root}")
    if str(ulf_root) not in sys.path:
        sys.path.insert(0, str(ulf_root))
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore
    from gaussian_renderer import get_render_visible_mask  # type: ignore

    return {
        "Scene": Scene,
        "GaussianModel": GaussianModel,
        "GaussianModel_2dgs": GaussianModel_2dgs,
        "get_render_visible_mask": get_render_visible_mask,
    }


def _dataset_namespace(args: argparse.Namespace, config: Mapping[str, Any]) -> argparse.Namespace:
    dense_cfg = config.get("dense", {})
    if not isinstance(dense_cfg, Mapping):
        dense_cfg = {}
    return argparse.Namespace(
        sh_degree=int(args.sh_degree),
        source_path=str(Path(args.source_path)),
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


def _intrinsic_from_camera(camera: Any) -> torch.Tensor:
    width = int(camera.image_width)
    height = int(camera.image_height)
    focal_x = float(width) / (2.0 * torch.tan(torch.tensor(float(camera.FoVx) * 0.5)).item())
    focal_y = float(height) / (2.0 * torch.tan(torch.tensor(float(camera.FoVy) * 0.5)).item())
    return torch.tensor(
        [
            [float(focal_x), 0.0, float(width) / 2.0],
            [0.0, float(focal_y), float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _lightweight_camera_from_cam_info(cam_info: Any, *, get_world_to_view2: Any) -> argparse.Namespace:
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


def _resize_mask_channel_cpu(mask_tensor: Any, *, height: int, width: int) -> torch.Tensor:
    mask = torch.as_tensor(mask_tensor).detach().cpu().float()
    if mask.dim() == 2:
        mask = mask[None, None]
    elif mask.dim() == 3:
        if int(mask.shape[0]) == 1:
            mask = mask[None]
        else:
            mask = mask[:1][None]
    else:
        raise ValueError(f"unsupported processed mask shape: {tuple(mask.shape)}")
    if tuple(mask.shape[-2:]) != (int(height), int(width)):
        mask = F.interpolate(mask, size=(int(height), int(width)), mode="nearest")
    return (mask[0, 0] > 0.5).bool()


def _mask_candidates(dataset_source: Path, images: str) -> list[Path]:
    return [
        dataset_source / images / "masks.pkl",
        Path(images) / "masks.pkl",
    ]


def _load_processed_masks(dataset_source: Path, images: str) -> tuple[Any | None, Path | None]:
    for path in _mask_candidates(dataset_source, images):
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle), path
    return None, None


def _mask_entry(masks: Any, image_name: str) -> Any | None:
    if masks is None or not isinstance(masks, Mapping):
        return None
    candidates = [
        str(image_name),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    for candidate in candidates:
        if candidate in masks:
            return masks[candidate]
    return None


def _stable_mask_for_image(masks: Any, image_name: str, *, height: int, width: int) -> tuple[torch.Tensor | None, str]:
    entry = _mask_entry(masks, image_name)
    if entry is None:
        return None, "missing_image_mask" if masks is not None else "no_masks"
    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
        raise ValueError(f"processed mask entry for {image_name!r} must contain obj, sky, and distort channels")
    obj_mask = _resize_mask_channel_cpu(entry[0], height=height, width=width)
    sky_mask = _resize_mask_channel_cpu(entry[1], height=height, width=width)
    distort_mask = _resize_mask_channel_cpu(entry[2], height=height, width=width)
    return obj_mask & sky_mask & distort_mask, "applied"


def _project_sampled_gaussians_to_camera(
    *,
    gaussian_xyz: torch.Tensor,
    opacity: torch.Tensor | None,
    sampled_idx: torch.Tensor,
    camera: Any,
    masks: Any,
    near_depth: float = 1e-6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_id = str(camera.image_name)
    width = int(camera.image_width)
    height = int(camera.image_height)
    world_to_camera = torch.as_tensor(camera.world_view_transform, dtype=torch.float32).detach().cpu().transpose(0, 1)
    intrinsic = _intrinsic_from_camera(camera)
    sampled_xyz = gaussian_xyz.index_select(0, sampled_idx)
    homo = torch.cat([sampled_xyz, torch.ones((int(sampled_xyz.shape[0]), 1), dtype=torch.float32)], dim=1)
    camera_xyz = (world_to_camera @ homo.T).T[:, :3]
    depth = camera_xyz[:, 2]
    positive_depth = depth > float(near_depth)
    uvw = (intrinsic @ camera_xyz.T).T
    uv = uvw[:, :2] / depth.clamp_min(float(near_depth))[:, None]
    in_frame = (
        positive_depth
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    stable_mask, mask_status = _stable_mask_for_image(masks, image_id, height=height, width=width)

    rows: list[dict[str, Any]] = []
    projected = torch.where(in_frame)[0]
    projected_mask_valid_count = 0
    for local_idx in projected.tolist():
        x = float(uv[local_idx, 0].item())
        y = float(uv[local_idx, 1].item())
        mask_valid = True
        if stable_mask is not None:
            yi = max(0, min(height - 1, int(y)))
            xi = max(0, min(width - 1, int(x)))
            mask_valid = bool(stable_mask[yi, xi].item())
        if mask_valid:
            projected_mask_valid_count += 1
        gid = int(sampled_idx[local_idx].item())
        weight = 1.0 if opacity is None else float(opacity[gid].item())
        rows.append(
            {
                "gaussian_id": gid,
                "keypoint_yx": [float(y), float(x)],
                "visible": True,
                "mask_valid": bool(mask_valid),
                "weight": float(max(0.0, weight)),
                "depth": float(depth[local_idx].item()),
                "projection_source": REAL_PROJECTION_SOURCE,
            }
        )

    metrics = {
        "image_id": image_id,
        "width": width,
        "height": height,
        "sampled_projection_count": int(sampled_idx.numel()),
        "projected_in_frame_count": int(projected.numel()),
        "projected_mask_valid_count": int(projected_mask_valid_count),
        "dropped_behind_count": int((~positive_depth).sum().item()),
        "dropped_out_of_frame_count": int((positive_depth & ~in_frame).sum().item()),
        "mask_status": mask_status,
    }
    return rows, metrics


def _project_sampled_gaussians_to_target(
    *,
    gaussian_xyz: torch.Tensor,
    opacity: torch.Tensor | None,
    sampled_idx: torch.Tensor,
    camera: Any,
    masks: Any,
    max_points_per_image: int,
    render_visible_mask: torch.Tensor | Any | None = None,
    near_depth: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    image_id = str(camera.image_name)
    width = int(camera.image_width)
    height = int(camera.image_height)
    world_to_camera = torch.as_tensor(camera.world_view_transform, dtype=torch.float32).detach().cpu().transpose(0, 1)
    intrinsic = _intrinsic_from_camera(camera)
    sampled_idx = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
    visible_local_mask = torch.ones(int(sampled_idx.numel()), dtype=torch.bool)
    if render_visible_mask is not None and int(sampled_idx.numel()) > 0:
        raw_visible = torch.as_tensor(render_visible_mask).detach().cpu().reshape(-1).bool()
        if int(sampled_idx.max().item()) >= int(raw_visible.numel()):
            raise ValueError(
                f"render_visible_mask length {int(raw_visible.numel())} is smaller than sampled id {int(sampled_idx.max().item())}"
            )
        visible_local_mask = raw_visible.index_select(0, sampled_idx)
    visible_local = torch.where(visible_local_mask)[0]
    active_sampled_idx = sampled_idx.index_select(0, visible_local)
    sampled_xyz = gaussian_xyz.index_select(0, active_sampled_idx) if int(active_sampled_idx.numel()) else gaussian_xyz.new_empty((0, 3))
    homo = torch.cat([sampled_xyz, torch.ones((int(sampled_xyz.shape[0]), 1), dtype=torch.float32)], dim=1)
    camera_xyz = (world_to_camera @ homo.T).T[:, :3]
    depth = camera_xyz[:, 2]
    positive_depth = depth > float(near_depth)
    uvw = (intrinsic @ camera_xyz.T).T
    uv = uvw[:, :2] / depth.clamp_min(float(near_depth))[:, None]
    in_frame = (
        positive_depth
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    stable_mask, mask_status = _stable_mask_for_image(masks, image_id, height=height, width=width)
    projected = torch.where(in_frame)[0]
    mask_valid = torch.ones(int(projected.numel()), dtype=torch.bool)
    if stable_mask is not None and int(projected.numel()) > 0:
        xy = uv.index_select(0, projected)
        yi = xy[:, 1].long().clamp(0, height - 1)
        xi = xy[:, 0].long().clamp(0, width - 1)
        mask_valid = stable_mask[yi, xi].bool()
    valid_local = projected[mask_valid]

    gids = active_sampled_idx.index_select(0, valid_local).long()
    yx = torch.empty((int(valid_local.numel()), 2), dtype=torch.float32)
    if int(valid_local.numel()) > 0:
        valid_uv = uv.index_select(0, valid_local)
        yx[:, 0] = valid_uv[:, 1].float()
        yx[:, 1] = valid_uv[:, 0].float()
    if opacity is None:
        weights = torch.ones(int(valid_local.numel()), dtype=torch.float32)
    else:
        weights = opacity.index_select(0, gids).reshape(-1).float().clamp(0.0, 1.0)

    dropped_by_cap = 0
    cap = int(max_points_per_image)
    if cap > 0 and int(gids.numel()) > cap:
        order = torch.argsort(-weights, stable=True)[:cap]
        dropped_by_cap = int(gids.numel() - cap)
        gids = gids.index_select(0, order)
        yx = yx.index_select(0, order)
        weights = weights.index_select(0, order)

    entry = {
        "gaussian_ids": gids.cpu(),
        "keypoint_yx": yx.cpu(),
        "support_weights": weights.cpu(),
        "positive_count": int(gids.numel()),
        "height": int(height),
        "width": int(width),
        "target_role": "stdloc_visibility_teacher",
    }
    single_metrics = {
        "input_projection_count": int(projected.numel()),
        "kept_projection_count": int(gids.numel()),
        "invisible_projection_count": int(sampled_idx.numel() - active_sampled_idx.numel()),
        "mask_invalid_projection_count": int((~mask_valid).sum().item()),
        "out_of_bounds_projection_count": 0,
        "invalid_projection_count": 0,
        "empty_image_count": 1 if int(gids.numel()) == 0 else 0,
        "support_weight_sum": float(weights.sum().item()) if int(weights.numel()) > 0 else 0.0,
        "dropped_by_image_cap_count": int(dropped_by_cap),
    }
    projection_metrics = {
        "image_id": image_id,
        "width": width,
        "height": height,
        "sampled_projection_count": int(sampled_idx.numel()),
        "render_visible_projection_count": int(active_sampled_idx.numel()),
        "projected_in_frame_count": int(projected.numel()),
        "projected_mask_valid_count": int(mask_valid.sum().item()),
        "dropped_render_invisible_count": int(sampled_idx.numel() - active_sampled_idx.numel()),
        "dropped_behind_count": int((~positive_depth).sum().item()),
        "dropped_out_of_frame_count": int((positive_depth & ~in_frame).sum().item()),
        "mask_status": mask_status,
    }
    return entry, single_metrics, projection_metrics


def _build_variable_sized_targets(
    projections: Mapping[str, list[Mapping[str, Any]]],
    image_sizes: Mapping[str, tuple[int, int]],
    *,
    max_points_per_image: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    metrics = {
        "schema_version": "ulfloc_visibility_teacher_metrics_v1",
        "image_count": int(len(projections)),
        "input_projection_count": 0,
        "kept_projection_count": 0,
        "invisible_projection_count": 0,
        "mask_invalid_projection_count": 0,
        "out_of_bounds_projection_count": 0,
        "invalid_projection_count": 0,
        "empty_image_count": 0,
        "support_weight_sum": 0.0,
        "dropped_by_image_cap_count": 0,
        "max_points_per_image": int(max_points_per_image),
        "detector_target_storage": "points",
        "target_role": "stdloc_visibility_teacher",
    }
    for image_id, rows in projections.items():
        height, width = image_sizes[image_id]
        single_targets, single_metrics = build_visibility_teacher_targets_with_metrics(
            {image_id: rows},
            height=int(height),
            width=int(width),
            max_points_per_image=int(max_points_per_image),
        )
        targets[image_id] = single_targets[image_id]
        for key in (
            "input_projection_count",
            "kept_projection_count",
            "invisible_projection_count",
            "mask_invalid_projection_count",
            "out_of_bounds_projection_count",
            "invalid_projection_count",
            "empty_image_count",
            "dropped_by_image_cap_count",
        ):
            metrics[key] += int(single_metrics[key])
        metrics["support_weight_sum"] += float(single_metrics["support_weight_sum"])
    sizes = sorted({(int(height), int(width)) for height, width in image_sizes.values()})
    first_height, first_width = sizes[0] if sizes else (0, 0)
    metrics.update(
        {
            "height": int(first_height),
            "width": int(first_width),
            "variable_image_size": len(sizes) > 1,
            "image_sizes": {str(image_id): [int(size[0]), int(size[1])] for image_id, size in image_sizes.items()},
            "max_points_per_image": int(max_points_per_image),
        }
    )
    metrics["support_weight_sum"] = float(metrics["support_weight_sum"])
    return targets, metrics


def _build_from_real_projection(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    _require_real_projection_args(args)
    split_name = validate_visibility_teacher_split_name(str(args.split_name))
    scene_name = str(args.scene)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_yaml_config(Path(args.cfg))
    sampled_idx, sampled_idx_path = _load_sampled_idx(args, config)

    from loc_gs.scripts.export_ulfloc_sparse_feedback import (
        load_dataset_image_list,
        resolve_ulfloc_source_path_for_loader,
    )

    original_source_path = Path(args.source_path)
    loader_source_path = resolve_ulfloc_source_path_for_loader(
        original_source_path,
        output_dir / "_ulf_loader_links",
    )
    args = argparse.Namespace(**vars(args))
    args.source_path = loader_source_path
    dataset = _dataset_namespace(args, config)
    dataset.eval = False
    images_to_read = None
    image_list_path = None
    if args.image_list is not None:
        image_list_path = Path(args.image_list)
        images_to_read = load_dataset_image_list(image_list_path)
        if not images_to_read:
            raise ValueError(f"--image_list did not contain any image rows: {image_list_path}")

    ulf_root = Path(args.ulf_root)
    imports = _import_ulfloc_projection_modules(ulf_root)
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
        images_to_read=images_to_read,
        preload_cameras=False,
    )
    from utils.graphics_utils import getWorld2View2  # type: ignore

    train_camera_infos = list(scene_obj.scene_info.train_cameras)
    if int(args.max_views) > 0:
        train_camera_infos = train_camera_infos[: int(args.max_views)]

    gaussian_xyz = torch.as_tensor(gaussians.get_xyz, dtype=torch.float32).detach().cpu()
    if int(sampled_idx.max().item()) >= int(gaussian_xyz.shape[0]):
        raise ValueError(
            f"sampled Gaussian index {int(sampled_idx.max().item())} exceeds loaded gaussian count {gaussian_xyz.shape[0]}"
        )
    opacity = None
    if hasattr(gaussians, "get_opacity"):
        opacity = torch.as_tensor(gaussians.get_opacity, dtype=torch.float32).detach().cpu().reshape(-1).clamp(0.0, 1.0)

    masks, mask_path = (None, None) if args.no_masks else _load_processed_masks(Path(dataset.source_path), str(dataset.images))
    targets: dict[str, dict[str, Any]] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    target_metrics = {
        "schema_version": "ulfloc_visibility_teacher_metrics_v1",
        "image_count": 0,
        "input_projection_count": 0,
        "kept_projection_count": 0,
        "invisible_projection_count": 0,
        "mask_invalid_projection_count": 0,
        "out_of_bounds_projection_count": 0,
        "invalid_projection_count": 0,
        "empty_image_count": 0,
        "support_weight_sum": 0.0,
        "dropped_by_image_cap_count": 0,
        "max_points_per_image": int(args.max_points_per_image),
        "detector_target_storage": "points",
        "target_role": "stdloc_visibility_teacher",
    }
    per_image_projection_metrics: list[dict[str, Any]] = []
    sampled_projection_count = 0
    projected_in_frame_count = 0
    projected_mask_valid_count = 0
    dropped_behind_count = 0
    dropped_out_of_frame_count = 0
    mask_status_counts: dict[str, int] = {}
    render_visible_enabled = not bool(args.no_render_visible_mask)
    dropped_render_invisible_count = 0
    render_visible_projection_count = 0

    train_camera_count = 0
    for camera_info in train_camera_infos:
        train_camera_count += 1
        camera = _lightweight_camera_from_cam_info(camera_info, get_world_to_view2=getWorld2View2)
        image_id = str(camera.image_name)
        height = int(camera.image_height)
        width = int(camera.image_width)
        render_visible_mask = None
        if render_visible_enabled:
            render_visible_mask = imports["get_render_visible_mask"](gaussians, camera, width, height).detach().cpu()
        entry, single_metrics, projection_metrics = _project_sampled_gaussians_to_target(
            gaussian_xyz=gaussian_xyz,
            opacity=opacity,
            sampled_idx=sampled_idx,
            camera=camera,
            masks=masks,
            max_points_per_image=int(args.max_points_per_image),
            render_visible_mask=render_visible_mask,
        )
        image_sizes[image_id] = (height, width)
        targets[image_id] = entry
        target_metrics["image_count"] += 1
        for key in (
            "input_projection_count",
            "kept_projection_count",
            "invisible_projection_count",
            "mask_invalid_projection_count",
            "out_of_bounds_projection_count",
            "invalid_projection_count",
            "empty_image_count",
            "dropped_by_image_cap_count",
        ):
            target_metrics[key] += int(single_metrics[key])
        target_metrics["support_weight_sum"] += float(single_metrics["support_weight_sum"])
        per_image_projection_metrics.append(projection_metrics)
        sampled_projection_count += int(projection_metrics["sampled_projection_count"])
        render_visible_projection_count += int(projection_metrics.get("render_visible_projection_count", 0))
        projected_in_frame_count += int(projection_metrics["projected_in_frame_count"])
        projected_mask_valid_count += int(projection_metrics["projected_mask_valid_count"])
        dropped_render_invisible_count += int(projection_metrics.get("dropped_render_invisible_count", 0))
        dropped_behind_count += int(projection_metrics["dropped_behind_count"])
        dropped_out_of_frame_count += int(projection_metrics["dropped_out_of_frame_count"])
        mask_status = str(projection_metrics["mask_status"])
        mask_status_counts[mask_status] = mask_status_counts.get(mask_status, 0) + 1

    sizes = sorted({(int(height), int(width)) for height, width in image_sizes.values()})
    first_height, first_width = sizes[0] if sizes else (0, 0)
    target_metrics.update(
        {
            "height": int(first_height),
            "width": int(first_width),
            "variable_image_size": len(sizes) > 1,
            "image_sizes": {str(image_id): [int(size[0]), int(size[1])] for image_id, size in image_sizes.items()},
            "support_weight_sum": float(target_metrics["support_weight_sum"]),
            "max_points_per_image": int(args.max_points_per_image),
        }
    )
    metrics = dict(target_metrics)
    metrics.update(
        {
            "projection_source": REAL_PROJECTION_SOURCE,
            "scene": scene_name,
            "split_name": split_name,
            "source_split_name": split_name,
            "sampled_idx_count": int(sampled_idx.numel()),
            "sampled_idx_path": str(sampled_idx_path),
            "source_gaussian_count": int(gaussian_xyz.shape[0]),
            "train_camera_count": int(train_camera_count),
            "images_to_read_count": None if images_to_read is None else int(len(images_to_read)),
            "image_list_path": None if image_list_path is None else str(image_list_path),
            "sampled_projection_count": int(sampled_projection_count),
            "render_visible_enabled": bool(render_visible_enabled),
            "render_visible_projection_count": int(render_visible_projection_count),
            "projected_in_frame_count": int(projected_in_frame_count),
            "projected_mask_valid_count": int(projected_mask_valid_count),
            "dropped_render_invisible_count": int(dropped_render_invisible_count),
            "dropped_behind_count": int(dropped_behind_count),
            "dropped_out_of_frame_count": int(dropped_out_of_frame_count),
            "processed_masks_enabled": not bool(args.no_masks),
            "processed_mask_path": None if mask_path is None else str(mask_path),
            "mask_status_counts": dict(sorted(mask_status_counts.items())),
            "dropped_by_image_cap_count": int(target_metrics["dropped_by_image_cap_count"]),
            "per_image_projection_metrics": per_image_projection_metrics,
            "camera_source": "ULF-Loc Scene.getTrainCameras",
            "dataset_eval": False,
        }
    )
    if train_camera_count <= 0:
        raise RuntimeError("no train cameras loaded; refusing to build visibility teacher from an empty train split")
    split_audit = _real_projection_split_audit(split_name=split_name, train_camera_count=train_camera_count)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene_name,
        "split_name": split_name,
        "targets": targets,
        "metadata": metrics,
        "split_audit": split_audit,
    }

    torch.save(artifact, output_dir / "visibility_teacher.pt")
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(
        output_dir / "manifest.json",
        _real_projection_manifest(
            args=args,
            repo_root=repo_root,
            ulf_root=ulf_root,
            split_name=split_name,
            scene=scene_name,
            metrics=metrics,
            split_audit=split_audit,
            sampled_idx_path=sampled_idx_path,
            loader_source_path=loader_source_path,
            original_source_path=original_source_path,
            mask_path=mask_path,
        ),
    )
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    (output_dir / "ulf_git_status.txt").write_text(_git_status(ulf_root), encoding="utf-8")

    print(
        json.dumps(
            {
                "visibility_teacher": str(output_dir / "visibility_teacher.pt"),
                "metrics_path": str(output_dir / "metrics_summary.json"),
                "kept_projection_count": int(metrics["kept_projection_count"]),
                "train_camera_count": int(metrics["train_camera_count"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compact STDLoc-style ULF-Loc visibility teacher targets from sampled landmark projections."
    )
    parser.add_argument("--input_projections_json", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split_name", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--height", default=None, type=int)
    parser.add_argument("--width", default=None, type=int)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--map_path", default=None, type=Path)
    parser.add_argument("--data_root", action="append", default=[])
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--source_path", default=None, type=Path)
    parser.add_argument("--model_path", default=None, type=Path)
    parser.add_argument("--cfg", default=None, type=Path)
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--input_log_dir", default=None, type=Path)
    parser.add_argument("--sampled_idx", default=None, type=Path)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--image_list", default=None, type=Path)
    parser.add_argument("--max_points_per_image", default=0, type=int)
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--no_render_visible_mask", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.input_projections_json is not None:
        return _build_from_input_projections_json(args)
    return _build_from_real_projection(args)


if __name__ == "__main__":
    raise SystemExit(main())
