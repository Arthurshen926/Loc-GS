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

import torch
import torch.nn.functional as F

from loc_gs.training.ulfloc_multiview_feature_observations import (
    DESCRIPTOR_SOURCE,
    SCHEMA_VERSION,
    build_multiview_feature_observation_cache,
    validate_multiview_observation_split_name,
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict JSON payload: {path}")
    return payload


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict payload: {path}")
    return payload


def _load_sampled_idx(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    sampled = torch.as_tensor(payload, dtype=torch.long).reshape(-1).detach().cpu()
    if sampled.numel() == 0:
        raise ValueError(f"sampled idx is empty: {path}")
    return sampled


def _sampled_idx_path(args: argparse.Namespace) -> Path:
    if args.sampled_idx is not None:
        return Path(args.sampled_idx)
    return Path(args.source_log_dir) / "keypoints_sampled_idx.pkl"


def _observations_from_json(payload: Mapping[str, Any]) -> Mapping[str, Any] | list[Any]:
    raw = payload.get("observations", payload.get("multiview_observations"))
    if not isinstance(raw, (Mapping, list, tuple)):
        raise KeyError("input observations JSON must contain an observations mapping or list")
    return raw


def _is_test_split(value: str) -> bool:
    split = str(value).strip().lower()
    return split == "test" or split == "official_test" or split.endswith("_test")


def _split_from_payload(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        audit = payload.get("split_audit", {})
        if isinstance(audit, Mapping):
            split = str(audit.get("split_name", audit.get("split", ""))).strip()
    return split


def _reject_test_payload(payload: Mapping[str, Any], *, name: str) -> None:
    split = _split_from_payload(payload)
    if _is_test_split(split):
        raise ValueError(f"refusing to use test split {name}")
    audit = payload.get("split_audit", {})
    if isinstance(audit, Mapping):
        if bool(audit.get("test_split_used", False)) or bool(audit.get("official_test_used", False)):
            raise ValueError(f"refusing to use test split {name}")
        audit_split = str(audit.get("split_name", audit.get("split", ""))).strip()
        if _is_test_split(audit_split):
            raise ValueError(f"refusing to use test split {name}")


def _impact_pair_keys(table: Any) -> list[tuple[str, str]]:
    if not isinstance(table, Mapping):
        return []
    out: list[tuple[str, str]] = []
    for raw_key, raw_value in table.items():
        if isinstance(raw_key, tuple) and len(raw_key) >= 2:
            out.append((str(raw_key[0]), str(raw_key[1])))
            continue
        text = str(raw_key)
        for sep in ("|", ":", "/"):
            if sep in text:
                left, right = text.split(sep, 1)
                if left.strip() and right.strip():
                    out.append((left.strip(), right.strip()))
                    break
        else:
            if isinstance(raw_value, Mapping):
                for nested_key in raw_value.keys():
                    out.append((text, str(nested_key)))
    return out


def _target_view_gaussian_ids_from_impact(impact: Mapping[str, Any]) -> dict[str, set[int]]:
    """Return exact view/gaussian pairs supervised by split-safe solver impact."""

    _reject_test_payload(impact, name="impact_attribution")
    targets: dict[str, set[int]] = {}
    for table_name in ("view_positive", "view_negative"):
        for gid_text, view_id in _impact_pair_keys(impact.get(table_name, {})):
            try:
                gid = int(gid_text)
            except (TypeError, ValueError):
                continue
            if not view_id:
                continue
            targets.setdefault(str(view_id), set()).add(gid)
    return targets


def _import_ulfloc_modules(ulf_root: Path) -> dict[str, Any]:
    if not ulf_root.exists():
        raise FileNotFoundError(f"ULF-Loc root not found: {ulf_root}")
    if str(ulf_root) not in sys.path:
        sys.path.insert(0, str(ulf_root))
    from encoders.sp_encoder.export_image_embeddings import SuperPoint  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore

    return {
        "SuperPoint": SuperPoint,
        "Scene": Scene,
        "GaussianModel": GaussianModel,
        "GaussianModel_2dgs": GaussianModel_2dgs,
    }


def _require_real_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("ulf_root", "model_path", "source_path", "images", "source_log_dir", "cfg", "scene")
        if getattr(args, name) is None or str(getattr(args, name)).strip() == ""
    ]
    if missing:
        raise ValueError("real ULF multiview extraction requires " + ", ".join(f"--{name}" for name in missing))


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"ULF config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ULF config must contain a mapping: {path}")
    return payload


def _dataset_namespace(args: argparse.Namespace, config: Mapping[str, Any]) -> argparse.Namespace:
    dense_cfg = config.get("dense", {})
    if not isinstance(dense_cfg, Mapping):
        dense_cfg = {}
    return argparse.Namespace(
        sh_degree=int(args.sh_degree),
        source_path=str(Path(args.source_path).resolve()),
        feature_type="sp",
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


def _stable_mask_for_camera(dataset_source: Path, images: str, camera: Any) -> torch.Tensor | None:
    mask_path = dataset_source / images / "masks.pkl"
    if not mask_path.exists():
        return None
    with mask_path.open("rb") as handle:
        masks = pickle.load(handle)
    if not isinstance(masks, Mapping):
        return None
    entry = None
    for key in (str(camera.image_name), Path(str(camera.image_name)).name, Path(str(camera.image_name)).stem):
        if key in masks:
            entry = masks[key]
            break
    if entry is None or not isinstance(entry, (list, tuple)) or len(entry) < 3:
        return None
    channels: list[torch.Tensor] = []
    for raw in entry[:3]:
        mask = torch.as_tensor(raw).detach().cpu().float()
        if mask.dim() == 2:
            mask = mask[None, None]
        elif mask.dim() == 3:
            mask = mask[:1][None]
        else:
            continue
        if tuple(mask.shape[-2:]) != (int(camera.image_height), int(camera.image_width)):
            mask = F.interpolate(mask, size=(int(camera.image_height), int(camera.image_width)), mode="nearest")
        channels.append((mask[0, 0] > 0.5).bool())
    if len(channels) != 3:
        return None
    return channels[0] & channels[1] & channels[2]


def _intrinsic(camera: Any, *, device: torch.device) -> torch.Tensor:
    width = int(camera.image_width)
    height = int(camera.image_height)
    focal_x = float(width) / (2.0 * torch.tan(torch.tensor(float(camera.FoVx) * 0.5)).item())
    focal_y = float(height) / (2.0 * torch.tan(torch.tensor(float(camera.FoVy) * 0.5)).item())
    return torch.tensor(
        [[focal_x, 0.0, width / 2.0], [0.0, focal_y, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _extract_real_observations(
    args: argparse.Namespace,
    sampled_idx: torch.Tensor,
    *,
    target_view_gaussian_ids: Mapping[str, set[int]] | None = None,
) -> Mapping[str, list[Mapping[str, Any]]]:
    _require_real_args(args)
    config = _load_yaml_config(Path(args.cfg))
    from loc_gs.scripts.export_ulfloc_sparse_feedback import resolve_ulfloc_source_path_for_loader

    original_source_path = Path(args.source_path)
    loader_source_path = resolve_ulfloc_source_path_for_loader(
        original_source_path,
        Path(args.output_dir) / "_ulf_loader_links",
    )
    real_args = argparse.Namespace(**vars(args))
    real_args.source_path = loader_source_path
    dataset = _dataset_namespace(real_args, config)
    dataset.eval = False

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    imports = _import_ulfloc_modules(Path(args.ulf_root))
    gaussians = (
        imports["GaussianModel"](dataset.sh_degree)
        if dataset.gaussian_type == "3dgs"
        else imports["GaussianModel_2dgs"](dataset.sh_degree)
    )
    scene_obj = imports["Scene"](dataset, gaussians, load_iteration=int(args.iteration), shuffle=False, preload_cameras=True)
    cameras = list(scene_obj.getTrainCameras())
    if int(args.max_views) > 0:
        cameras = cameras[: int(args.max_views)]
    if not cameras:
        raise RuntimeError("no train cameras loaded for ULF multiview feature observations")

    model = imports["SuperPoint"]().to(device).eval()
    sampled = sampled_idx.to(device=device, dtype=torch.long)
    xyz = torch.as_tensor(gaussians.get_xyz, dtype=torch.float32, device=device).index_select(0, sampled)
    if hasattr(gaussians, "get_smallest_axis"):
        normals = F.normalize(torch.as_tensor(gaussians.get_smallest_axis(), dtype=torch.float32, device=device).index_select(0, sampled), dim=-1)
    else:
        normals = torch.zeros_like(xyz)
    observations: dict[str, list[Mapping[str, Any]]] = {}
    exact_filter = target_view_gaussian_ids if target_view_gaussian_ids else None
    with torch.inference_mode():
        for camera in cameras:
            camera_name = str(camera.image_name)
            target_gids = None if exact_filter is None else exact_filter.get(camera_name, set())
            if exact_filter is not None and not target_gids:
                observations[camera_name] = []
                continue
            image = camera.original_image[:3].to(device=device, dtype=torch.float32).unsqueeze(0)
            feature_map, _scores = model.detectAndComputeDense(image)
            width = int(camera.image_width)
            height = int(camera.image_height)
            world_to_camera = torch.as_tensor(camera.world_view_transform, dtype=torch.float32, device=device).transpose(0, 1)
            intrinsic = _intrinsic(camera, device=device)
            homo = torch.cat([xyz, torch.ones((int(xyz.shape[0]), 1), dtype=torch.float32, device=device)], dim=1)
            camera_xyz = (world_to_camera @ homo.T).T[:, :3]
            depth = camera_xyz[:, 2]
            uvw = (intrinsic @ camera_xyz.T).T
            uv = uvw[:, :2] / depth.clamp_min(1e-6)[:, None]
            valid = (
                (depth > 1e-6)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] < float(width))
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] < float(height))
            )
            if not bool(args.no_masks):
                stable = _stable_mask_for_camera(Path(dataset.source_path), str(dataset.images), camera)
                if stable is not None:
                    stable = stable.to(device=device)
                    xi = uv[:, 0].clamp(0, width - 1).long()
                    yi = uv[:, 1].clamp(0, height - 1).long()
                    valid = valid & stable[yi, xi]
            indices = torch.where(valid)[0]
            if target_gids is not None:
                target = torch.as_tensor(sorted(int(gid) for gid in target_gids), dtype=torch.long, device=device)
                if target.numel() == 0:
                    indices = indices[:0]
                else:
                    sampled_gids = sampled.index_select(0, indices)
                    try:
                        keep = torch.isin(sampled_gids, target)
                    except AttributeError:
                        keep = (sampled_gids[:, None] == target[None, :]).any(dim=1)
                    indices = indices[keep]
            if args.max_observations_per_view is not None and int(args.max_observations_per_view) > 0:
                indices = indices[: int(args.max_observations_per_view)]
            if indices.numel() == 0:
                observations[camera_name] = []
                continue
            uv_valid = uv.index_select(0, indices)
            grid = uv_valid.clone()
            grid[:, 0] = 2.0 * (grid[:, 0] + 0.5) / float(width) - 1.0
            grid[:, 1] = 2.0 * (grid[:, 1] + 0.5) / float(height) - 1.0
            sampled_desc = F.grid_sample(
                feature_map,
                grid.reshape(1, -1, 1, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0).squeeze(-1).T
            sampled_desc = F.normalize(sampled_desc, p=2, dim=-1).detach().cpu()
            cam_center = camera.camera_center.detach().to(device=device, dtype=torch.float32)
            view_dirs = F.normalize(cam_center[None, :] - xyz.index_select(0, indices), dim=-1)
            geo = torch.abs((normals.index_select(0, indices) * view_dirs).sum(dim=-1)).clamp_min(0.0).detach().cpu()
            depth_cpu = depth.index_select(0, indices).detach().cpu()
            uv_cpu = uv_valid.detach().cpu()
            gid_cpu = sampled_idx.index_select(0, indices.detach().cpu())
            rows: list[Mapping[str, Any]] = []
            for row_idx in range(int(indices.numel())):
                rows.append(
                    {
                        "gaussian_id": int(gid_cpu[row_idx].item()),
                        "descriptor": sampled_desc[row_idx].tolist(),
                        "geometry_weight": float(geo[row_idx].item()),
                        "keypoint_xy": [float(uv_cpu[row_idx, 0].item()), float(uv_cpu[row_idx, 1].item())],
                        "depth_m": float(depth_cpu[row_idx].item()),
                    }
                )
            observations[camera_name] = rows
    return observations


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ULF original multiview feature-observation cache.")
    parser.add_argument("--input_observations_json", default=None, type=Path)
    parser.add_argument("--source_log_dir", required=True, type=Path)
    parser.add_argument("--sampled_idx", default=None, type=Path)
    parser.add_argument(
        "--impact_attribution",
        default=None,
        type=Path,
        help="Optional split-safe solver impact artifact used to export only supervised view/landmark observations.",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--model_path", default=None, type=Path)
    parser.add_argument("--source_path", default=None, type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--cfg", default=None, type=Path)
    parser.add_argument("--gaussian_type", default=None)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--longest_edge", default=1600, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--max_observations_per_view", default=None, type=int)
    parser.add_argument("--no_masks", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    split_name = validate_multiview_observation_split_name(str(args.split_name))
    sampled_idx = _load_sampled_idx(_sampled_idx_path(args))
    if args.input_observations_json is not None:
        payload = _load_json(Path(args.input_observations_json))
        payload_split = str(payload.get("split_name", split_name))
        if _is_test_split(payload_split):
            raise ValueError("refusing to build ULF multiview feature observation cache from test split")
        if payload_split != split_name and payload_split.lower() != "unknown":
            raise ValueError(f"split_name mismatch: CLI {split_name!r} != observations {payload_split!r}")
        observations = _observations_from_json(payload)
        source = "input_observations_json"
    else:
        target_view_gaussian_ids = None
        impact_pair_count = 0
        impact_view_count = 0
        if args.impact_attribution is not None:
            impact = _load_payload(Path(args.impact_attribution))
            target_view_gaussian_ids = _target_view_gaussian_ids_from_impact(impact)
            impact_view_count = int(len(target_view_gaussian_ids))
            impact_pair_count = int(sum(len(values) for values in target_view_gaussian_ids.values()))
            if impact_pair_count <= 0:
                raise ValueError("impact_attribution has no view_positive/view_negative pairs to export")
        observations = _extract_real_observations(
            args,
            sampled_idx,
            target_view_gaussian_ids=target_view_gaussian_ids,
        )
        source = "real_ulf_multiview_feature_projection"
    cache, metrics = build_multiview_feature_observation_cache(
        observations,
        sampled_idx=sampled_idx,
        scene=str(args.scene),
        split_name=split_name,
    )
    metrics = dict(metrics)
    metrics["observation_source"] = source
    if args.impact_attribution is not None:
        metrics["impact_attribution"] = str(Path(args.impact_attribution))
        metrics["impact_filter_view_count"] = impact_view_count
        metrics["impact_filter_pair_count"] = impact_pair_count
    split_audit = {
        "schema_version": "ulfloc_multiview_feature_observation_cache_split_audit_v1",
        "split_name": split_name,
        "test_split_used": False,
        "official_test_used": False,
        "observation_source": source,
    }
    cache["split_audit"] = split_audit
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "observation_cache.pt"
    torch.save(cache, cache_path)
    manifest = {
        "schema_version": "ulfloc_multiview_feature_observation_cache_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv if argv is None else [sys.executable, "-m", "loc_gs.scripts.build_ulfloc_multiview_feature_observation_cache", *argv],
        "scene": str(args.scene),
        "split_name": split_name,
        "source_log_dir": str(Path(args.source_log_dir)),
        "sampled_idx": str(_sampled_idx_path(args)),
        "observation_cache": str(cache_path),
        "observation_descriptor_source": DESCRIPTOR_SOURCE,
        "observation_source": source,
        "impact_attribution": None if args.impact_attribution is None else str(Path(args.impact_attribution)),
        "metrics": metrics,
        "split_audit": split_audit,
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"observation_cache": str(cache_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
