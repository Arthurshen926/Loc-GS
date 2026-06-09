#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from loc_gs.feedback.full_raw_gaussian_projection import (
    build_sparse_source_rays_from_full_raw_gaussians,
)
from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name
from loc_gs.scripts.build_cross_view_ray_observations import load_matches
from loc_gs.scripts.build_projected_gaussian_rays import write_source_rays_jsonl
from loc_gs.scripts.export_ulfloc_full_raw_projections import (
    gaussian_payload_from_ulfloc_model,
    view_record_from_ulfloc_camera,
)
from loc_gs.scripts.export_ulfloc_sparse_feedback import (
    _dataset_namespace,
    _git_commit,
    _git_status,
    _import_ulfloc,
    resolve_ulfloc_source_path_for_loader,
)


SCHEMA_VERSION = "ulfloc_full_raw_source_rays_v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _match_sources(matches: Sequence[Mapping[str, Any]]) -> set[str]:
    sources: set[str] = set()
    for match in matches:
        source = str(match.get("source_view_id", match.get("view_id", ""))).strip()
        if source:
            sources.add(source)
    return sources


def view_records_for_match_sources(
    cameras: Sequence[Any],
    matches: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    max_views: int = 0,
) -> dict[str, dict[str, Any]]:
    split = validate_split_name(split_name)
    needed = _match_sources(matches)
    views: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        source = str(camera.image_name)
        if source not in needed:
            continue
        views[source] = view_record_from_ulfloc_camera(camera, split_name=split)
        if int(max_views) > 0 and len(views) >= int(max_views):
            break
    return views


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build sparse source rays from ULF-Loc train matches by streaming full raw Gaussian projections per view."
        )
    )
    parser.add_argument("--matches", required=True, type=Path)
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
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--top_k", default=8, type=int)
    parser.add_argument("--max_radius_px", default=8.0, type=float)
    parser.add_argument("--radius_scale", default=2.0, type=float)
    parser.add_argument("--min_contribution", default=1e-6, type=float)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    split_name = validate_split_name(str(args.split_name))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_manifest, matches = load_matches(Path(args.matches))
    matches_split = validate_split_name(str(matches_manifest.get("split_name", split_name)))

    ulf_root = Path(args.ulf_root)
    config = yaml.load(Path(args.config).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise ValueError(f"ULF config must contain a YAML mapping: {args.config}")
    imports = _import_ulfloc(ulf_root)
    original_source_path = Path(args.source_path)
    loader_source_path = resolve_ulfloc_source_path_for_loader(
        original_source_path,
        output_dir / "_ulf_loader_links",
    )
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
        images_to_read=sorted(_match_sources(matches)),
        preload_cameras=True,
    )
    gaussian_payload = gaussian_payload_from_ulfloc_model(gaussians)
    views_by_source = view_records_for_match_sources(
        scene_obj.getTrainCameras(),
        matches,
        split_name=split_name,
        max_views=int(args.max_views),
    )
    bundle = build_sparse_source_rays_from_full_raw_gaussians(
        matches=matches,
        gaussian_xyz=gaussian_payload["xyz"],
        views_by_source=views_by_source,
        split_name=split_name,
        opacity=gaussian_payload.get("opacity"),
        top_k=int(args.top_k),
        max_radius_px=float(args.max_radius_px),
        radius_scale=float(args.radius_scale),
        min_contribution=float(args.min_contribution),
    )

    source_rays_path = output_dir / "source_rays.jsonl"
    command = " ".join(shlex.quote(str(item)) for item in sys.argv)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "ulf_git_commit": _git_commit(ulf_root),
        "command": command,
        "scene": str(args.scene),
        "split_name": split_name,
        "matches_split_name": matches_split,
        "matches": str(Path(args.matches)),
        "source_rays_path": str(source_rays_path),
        "source_path": str(args.source_path),
        "original_source_path": str(original_source_path),
        "model_path": str(args.model_path),
        "config": str(args.config),
        "iteration": int(args.iteration),
        "max_views": int(args.max_views),
        "images_to_read_count": int(len(_match_sources(matches))),
        "projection_source": "full_raw_gaussians",
        "sampled_idx_used": False,
        "materialized_projection_jsonl": False,
        "hyperparameters": {
            "top_k": int(args.top_k),
            "max_radius_px": float(args.max_radius_px),
            "radius_scale": float(args.radius_scale),
            "min_contribution": float(args.min_contribution),
        },
        "source_manifests": {
            "matches": matches_manifest,
        },
    }
    write_source_rays_jsonl(source_rays_path, manifest, bundle["rays"])
    metrics = dict(bundle["summary"])
    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "matches_split_name": matches_split,
        "test_split_used": False,
        "projection_source": "full_raw_gaussians",
        "sampled_idx_used": False,
        "notes": "Streaming ULF full-raw source-ray builder uses Scene.getTrainCameras and full gaussians.get_xyz.",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")
    (output_dir / "ulf_git_status.txt").write_text(_git_status(ulf_root), encoding="utf-8")

    print(
        json.dumps(
            {
                "source_rays_path": str(source_rays_path),
                "manifest_path": str(output_dir / "manifest.json"),
                "metrics_path": str(output_dir / "metrics_summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
