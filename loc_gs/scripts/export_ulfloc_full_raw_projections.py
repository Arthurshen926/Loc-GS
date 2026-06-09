#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from loc_gs.feedback.full_raw_gaussian_projection import (
    PROJECTION_SOURCE,
    SCHEMA_VERSION,
    project_full_raw_gaussians_to_view,
    validate_full_raw_projection_bundle,
)
from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name
from loc_gs.scripts.export_ulfloc_sparse_feedback import (
    _dataset_namespace,
    _git_commit,
    _git_status,
    _import_ulfloc,
)


def intrinsic_from_fov(*, fovx: float, fovy: float, width: int, height: int) -> list[list[float]]:
    focal_x = float(width) / (2.0 * math.tan(float(fovx) * 0.5))
    focal_y = float(height) / (2.0 * math.tan(float(fovy) * 0.5))
    return [
        [float(focal_x), 0.0, float(width) / 2.0],
        [0.0, float(focal_y), float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ]


def _tensor_to_list(value: Any) -> list[Any]:
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu().tolist()


def view_record_from_ulfloc_camera(camera: Any, *, split_name: str) -> dict[str, Any]:
    split = validate_split_name(split_name)
    width = int(camera.image_width)
    height = int(camera.image_height)
    world_to_camera = torch.as_tensor(camera.world_view_transform, dtype=torch.float32).transpose(0, 1)
    return {
        "source_view_id": str(camera.image_name),
        "split_name": split,
        "width": width,
        "height": height,
        "world_to_camera": _tensor_to_list(world_to_camera),
        "intrinsic": intrinsic_from_fov(fovx=float(camera.FoVx), fovy=float(camera.FoVy), width=width, height=height),
    }


def gaussian_payload_from_ulfloc_model(gaussians: Any) -> dict[str, Any]:
    xyz = torch.as_tensor(gaussians.get_xyz, dtype=torch.float32).detach().cpu()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "projection_source": PROJECTION_SOURCE,
        "source_gaussian_count": int(xyz.shape[0]),
        "xyz": xyz,
    }
    if hasattr(gaussians, "get_opacity"):
        payload["opacity"] = torch.as_tensor(gaussians.get_opacity, dtype=torch.float32).detach().cpu().reshape(-1)
    return payload


def resolve_ulfloc_source_path_for_loader(source_path: Path, link_root: Path) -> Path:
    source = Path(source_path)
    text = str(source)
    if "7scenes" in text or "12scenes" in text or "cambridge" in text:
        return source
    if "Cambridge" not in text and "CAMBRIDGE" not in text:
        return source

    link_root = Path(link_root)
    link_root.mkdir(parents=True, exist_ok=True)
    link_name = "cambridge_" + source.name
    link_path = link_root / link_name
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() != source.resolve():
            raise FileExistsError(f"ULF loader compatibility link points elsewhere: {link_path}")
    else:
        link_path.symlink_to(source.resolve(), target_is_directory=True)
    return link_path


def write_projected_gaussians_jsonl(
    path: Path,
    manifest: dict[str, Any],
    projections: list[dict[str, Any]],
) -> None:
    rows = [{"type": "manifest", "manifest": manifest}]
    rows.extend({"type": "projection", "projection": row} for row in projections)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export full raw Gaussian projections from native ULF-Loc train views.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--split_name", default="selfmap_train")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--max_images", default=0, type=int)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    split_name = validate_split_name(str(args.split_name))
    ulf_root = Path(args.ulf_root)
    config = yaml.load(Path(args.config).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise ValueError(f"ULF config must contain a YAML mapping: {args.config}")
    imports = _import_ulfloc(ulf_root)
    loader_source_path = resolve_ulfloc_source_path_for_loader(
        Path(args.source_path),
        Path(args.output_dir) / "_ulf_loader_links",
    )
    original_source_path = Path(args.source_path)
    args = argparse.Namespace(**vars(args))
    args.source_path = loader_source_path
    dataset = _dataset_namespace(args, config)

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
        preload_cameras=True,
    )
    gaussian_payload = gaussian_payload_from_ulfloc_model(gaussians)

    projections: list[dict[str, Any]] = []
    view_count = 0
    projected_count = 0
    dropped_behind = 0
    dropped_out_of_frame = 0
    for camera in scene_obj.getTrainCameras():
        if int(args.max_images) > 0 and view_count >= int(args.max_images):
            break
        view = view_record_from_ulfloc_camera(camera, split_name=split_name)
        bundle = project_full_raw_gaussians_to_view(
            gaussian_xyz=gaussian_payload["xyz"],
            world_to_camera=view["world_to_camera"],
            intrinsic=view["intrinsic"],
            width=int(view["width"]),
            height=int(view["height"]),
            source_view_id=str(view["source_view_id"]),
            split_name=split_name,
            opacity=gaussian_payload.get("opacity"),
        )
        summary = validate_full_raw_projection_bundle(
            bundle,
            expected_source_gaussian_count=int(gaussian_payload["source_gaussian_count"]),
        )
        projections.extend(bundle["projections"])
        projected_count += int(summary["projected_gaussian_count"])
        dropped_behind += int(summary["dropped_behind_count"])
        dropped_out_of_frame += int(summary["dropped_out_of_frame_count"])
        view_count += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projections_path = output_dir / "projected_gaussians.jsonl"
    command = " ".join(shlex.quote(str(item)) for item in sys.argv)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(args.scene),
        "split_name": split_name,
        "projection_source": PROJECTION_SOURCE,
        "sampled_idx_used": False,
        "source_gaussian_count": int(gaussian_payload["source_gaussian_count"]),
        "view_count": int(view_count),
        "projected_gaussian_count": int(projected_count),
        "dropped_behind_count": int(dropped_behind),
        "dropped_out_of_frame_count": int(dropped_out_of_frame),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "ulf_git_commit": _git_commit(ulf_root),
        "command": command,
        "scene": str(args.scene),
        "split_name": split_name,
        "projection_source": PROJECTION_SOURCE,
        "sampled_idx_used": False,
        "source_path": str(args.source_path),
        "original_source_path": str(original_source_path),
        "model_path": str(args.model_path),
        "config": str(args.config),
        "iteration": int(args.iteration),
        "max_images": int(args.max_images),
        "projections_path": str(projections_path),
        "metrics_summary": metrics,
    }
    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "projection_source": PROJECTION_SOURCE,
        "sampled_idx_used": False,
        "test_split_used": False,
        "notes": "ULF full raw projection exporter uses Scene.getTrainCameras and full gaussians.get_xyz.",
    }
    write_projected_gaussians_jsonl(projections_path, manifest, projections)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")
    (output_dir / "ulf_git_status.txt").write_text(_git_status(ulf_root), encoding="utf-8")
    print(json.dumps({"projections_path": str(projections_path), "metrics_path": str(output_dir / "metrics_summary.json")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
