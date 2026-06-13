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
from PIL import Image, ImageDraw

from loc_gs.diagnostics.ulfloc_scene_detector_audit import (
    keypoint_overlap_fraction,
    summarize_detector_score_groups,
)
from loc_gs.localization.stdloc_detector import StdlocKeypointDetector, simple_nms
from loc_gs.localization.ulfloc_scene_detector import (
    extract_ulfloc_scene_detector_keypoints_with_superpoint_scores,
)
from loc_gs.training.ulfloc_scene_detector import (
    load_detector_target_artifact,
    make_trainable_detector_input,
    pixel_yx_to_coarse_yx,
    scale_pixel_yx,
)


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


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


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device=device, dtype=torch.float32)


def _load_masks(source_path: Path, images: str) -> Any:
    path = Path(source_path) / images / "masks.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resize_mask_to_image(mask: torch.Tensor | Any, *, height: int, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(mask)
    original_dtype = tensor.dtype
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError("mask must have shape [H,W] or [1,H,W]")
    if tensor.shape[-2:] == (int(height), int(width)):
        return tensor.to(dtype=original_dtype)
    resized = F.interpolate(
        tensor.float().unsqueeze(0),
        size=(int(height), int(width)),
        mode="nearest",
    ).squeeze(0)
    if original_dtype == torch.bool:
        return resized > 0.5
    return resized.to(dtype=original_dtype)


def _apply_audit_masks(query_image: torch.Tensor, masks: Any, image_name: str) -> torch.Tensor:
    """Apply the same object/distortion/sky masks used by sparse eval."""

    if masks is None:
        return query_image
    if image_name not in masks:
        return query_image
    with torch.no_grad():
        height, width = int(query_image.shape[-2]), int(query_image.shape[-1])
        obj_mask = _resize_mask_to_image(masks[image_name][0], height=height, width=width).to(query_image.device)
        sky_mask = _resize_mask_to_image(masks[image_name][1], height=height, width=width).to(query_image.device)
        distort_mask = _resize_mask_to_image(masks[image_name][2], height=height, width=width).to(query_image.device)
        valid_mask = obj_mask & distort_mask
        masked = query_image * valid_mask
        masked[sky_mask.repeat(3, 1, 1) == False] = 0
    return masked


def _import_ulfloc_feature_extractor(ulf_root: Path):
    root = str(Path(ulf_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from encoders.feature_extractor import FeatureExtractor  # type: ignore

    return FeatureExtractor


def _iter_target_items(
    targets: Mapping[str, Any],
    *,
    source_path: Path,
    images: str,
    max_images: int,
) -> list[tuple[str, Path, Mapping[str, Any]]]:
    items: list[tuple[str, Path, Mapping[str, Any]]] = []
    for query_id, raw_entry in sorted(targets.items()):
        if not isinstance(raw_entry, Mapping):
            continue
        image_path = source_path / images / str(query_id)
        if not image_path.exists():
            continue
        items.append((str(query_id), image_path, raw_entry))
        if int(max_images) > 0 and len(items) >= int(max_images):
            break
    return items


def _entry_points(entry: Mapping[str, Any], key: str) -> torch.Tensor:
    return torch.as_tensor(entry.get(key, []), dtype=torch.float32).reshape(-1, 2)


def _scale_points_to_heatmap(
    points_yx: torch.Tensor,
    *,
    entry: Mapping[str, Any],
    heat_height: int,
    heat_width: int,
    target_mode: str,
    descriptor_stride: int,
) -> torch.Tensor:
    if str(target_mode) == "stdloc_fullres":
        source_height = int(entry.get("height", heat_height))
        source_width = int(entry.get("width", heat_width))
        return scale_pixel_yx(
            points_yx,
            source_height=source_height,
            source_width=source_width,
            target_height=int(heat_height),
            target_width=int(heat_width),
        )
    return pixel_yx_to_coarse_yx(points_yx, stride=int(descriptor_stride))


def _draw_points(draw: ImageDraw.ImageDraw, points_xy: torch.Tensor, color: tuple[int, int, int], radius: int) -> None:
    for x, y in torch.as_tensor(points_xy, dtype=torch.float32).reshape(-1, 2).tolist():
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)


def _heatmap_image(heatmap: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    heat = torch.as_tensor(heatmap, dtype=torch.float32).detach().cpu()
    while heat.dim() > 2:
        heat = heat[0]
    heat = heat.clamp_min(0.0)
    if heat.numel() and float(heat.max().item()) > 0.0:
        heat = heat / heat.max().clamp_min(1e-12)
    array = (heat.numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(array, mode="L").resize(size, Image.Resampling.NEAREST).convert("RGB")
    red = Image.new("RGB", size, (255, 0, 0))
    return Image.blend(image, red, alpha=0.35)


def _topk_xy_from_heatmap(heatmap: torch.Tensor, *, max_keypoints: int, nms_radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    heat = torch.as_tensor(heatmap, dtype=torch.float32).detach().cpu()
    while heat.dim() > 2:
        heat = heat[0]
    if heat.dim() != 2:
        raise ValueError("heatmap must reduce to [H,W]")
    scores = simple_nms(heat, int(nms_radius)).reshape(-1)
    if scores.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
    topk = min(int(max_keypoints), int(scores.numel()))
    values, indices = torch.topk(scores, k=topk)
    keep = values > 0.0
    values = values[keep]
    indices = indices[keep]
    width = int(heat.shape[-1])
    y = torch.div(indices, width, rounding_mode="floor")
    x = indices % width
    xy = torch.stack([x.float(), y.float()], dim=-1)
    return xy, values


def _make_overlay(
    image_path: Path,
    *,
    native_xy: torch.Tensor,
    fused_xy: torch.Tensor,
    visibility_yx: torch.Tensor,
    solver_positive_yx: torch.Tensor,
    negative_yx: torch.Tensor,
    heatmap: torch.Tensor,
    output_path: Path,
    max_points: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    _draw_points(draw, native_xy[:max_points], (0, 210, 255), 2)
    _draw_points(draw, fused_xy[:max_points], (0, 255, 0), 3)
    vis_xy = torch.stack([visibility_yx[:, 1], visibility_yx[:, 0]], dim=-1) if visibility_yx.numel() else visibility_yx.reshape(0, 2)
    pos_xy = (
        torch.stack([solver_positive_yx[:, 1], solver_positive_yx[:, 0]], dim=-1)
        if solver_positive_yx.numel()
        else solver_positive_yx.reshape(0, 2)
    )
    neg_xy = torch.stack([negative_yx[:, 1], negative_yx[:, 0]], dim=-1) if negative_yx.numel() else negative_yx.reshape(0, 2)
    _draw_points(draw, vis_xy[:max_points], (255, 220, 0), 1)
    _draw_points(draw, pos_xy[:max_points], (40, 100, 255), 4)
    _draw_points(draw, neg_xy[:max_points], (255, 0, 0), 4)
    heat = _heatmap_image(heatmap, image.size)
    sheet = Image.new("RGB", (image.width * 2, image.height), (0, 0, 0))
    sheet.paste(overlay, (0, 0))
    sheet.paste(heat, (image.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ULF-Loc scene detector heatmaps against teacher/solver targets.")
    parser.add_argument("--detector_targets", required=True, type=Path)
    parser.add_argument("--scene_detector_checkpoint", required=True, type=Path)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--target_mode", default="auto", choices=("auto", "coarse_teacher_residual", "stdloc_fullres"))
    parser.add_argument("--descriptor_stride", default=8, type=int)
    parser.add_argument("--max_images", default=16, type=int)
    parser.add_argument("--max_visuals", default=8, type=int)
    parser.add_argument("--overlay_topk", default=200, type=int)
    parser.add_argument("--blend_alpha", default=0.1, type=float)
    parser.add_argument("--fusion_rule", default="residual_boost", choices=("geometric", "residual_boost"))
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_detector_target_artifact(args.detector_targets)
    targets = artifact["targets"]
    split_audit = dict(artifact.get("split_audit", {}))
    split_name = str(artifact.get("split_name", split_audit.get("split_name", "unknown")))
    checkpoint = torch.load(Path(args.scene_detector_checkpoint), map_location="cpu")
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, Mapping) else {}
    target_mode = str(args.target_mode)
    if target_mode == "auto":
        target_mode = str(metadata.get("target_mode", "coarse_teacher_residual"))

    device = torch.device(args.device)
    FeatureExtractor = _import_ulfloc_feature_extractor(Path(args.ulf_root))
    feature_extractor = FeatureExtractor(str(args.feature_type)).cuda().eval()
    detector = StdlocKeypointDetector(in_dim=256).to(device).eval()
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    detector.load_state_dict(state_dict, strict=True)

    items = _iter_target_items(
        targets,
        source_path=Path(args.source_path),
        images=str(args.images),
        max_images=int(args.max_images),
    )
    if not items:
        raise ValueError("no audit images found under source_path/images")
    masks = _load_masks(Path(args.source_path), str(args.images))

    per_image: list[dict[str, Any]] = []
    for index, (query_id, image_path, entry) in enumerate(items):
        image = _apply_audit_masks(_load_image(image_path, device), masks, query_id)
        with torch.no_grad():
            feature_map, superpoint_scores = feature_extractor.detectAndComputeDense(image[None])
            detector_feature_map = feature_map
            if target_mode == "stdloc_fullres":
                detector_feature_map = F.interpolate(
                    feature_map,
                    size=(int(image.shape[-2]), int(image.shape[-1])),
                    mode="bilinear",
                    align_corners=False,
                )
                detector_feature_map = F.normalize(detector_feature_map, p=2, dim=1)
            detector_input = make_trainable_detector_input(detector_feature_map[0]).to(device=device)
            heat = detector(detector_input)
            while heat.dim() > 2:
                heat = heat[0]
            heat_cpu = heat.detach().cpu()
            native_xy, _native_scores = extract_ulfloc_scene_detector_keypoints_with_superpoint_scores(
                feature_map[0],
                superpoint_scores[0],
                detector,
                max_keypoints=int(args.overlay_topk),
                nms_radius=4,
                descriptor_stride=int(args.descriptor_stride),
                blend_alpha=0.0,
                fusion_rule="residual_boost",
                remove_borders=4,
            )
            if target_mode == "stdloc_fullres":
                fused_xy, _fused_scores = _topk_xy_from_heatmap(
                    heat_cpu,
                    max_keypoints=int(args.overlay_topk),
                    nms_radius=4,
                )
            else:
                fused_xy, _fused_scores = extract_ulfloc_scene_detector_keypoints_with_superpoint_scores(
                    feature_map[0],
                    superpoint_scores[0],
                    detector,
                    max_keypoints=int(args.overlay_topk),
                    nms_radius=4,
                    descriptor_stride=int(args.descriptor_stride),
                    blend_alpha=float(args.blend_alpha),
                    fusion_rule=str(args.fusion_rule),
                    remove_borders=4,
                )

        groups = {}
        for group_name, entry_key in (
            ("visibility", "keypoint_yx"),
            ("sp_teacher", "sp_teacher_keypoint_yx"),
            ("solver_positive", "solver_positive_keypoint_yx"),
            ("negative", "negative_keypoint_yx"),
        ):
            points = _entry_points(entry, entry_key)
            groups[group_name] = _scale_points_to_heatmap(
                points,
                entry=entry,
                heat_height=int(heat_cpu.shape[-2]),
                heat_width=int(heat_cpu.shape[-1]),
                target_mode=target_mode,
                descriptor_stride=int(args.descriptor_stride),
            )
        summary = summarize_detector_score_groups(heat_cpu, groups)
        overlap = keypoint_overlap_fraction(native_xy.cpu(), fused_xy.cpu())
        record = {
            "query_id": query_id,
            "image_path": str(image_path),
            "target_mode": target_mode,
            **summary,
            **{f"keypoint_{k}": v for k, v in overlap.items()},
        }
        per_image.append(record)
        if index < int(args.max_visuals):
            safe_name = query_id.replace("/", "_").replace("\\", "_")
            _make_overlay(
                image_path,
                native_xy=native_xy.cpu(),
                fused_xy=fused_xy.cpu(),
                visibility_yx=_entry_points(entry, "keypoint_yx"),
                solver_positive_yx=_entry_points(entry, "solver_positive_keypoint_yx"),
                negative_yx=_entry_points(entry, "negative_keypoint_yx"),
                heatmap=heat_cpu,
                output_path=output_dir / "contact_sheets" / f"{safe_name}_detector_audit.jpg",
                max_points=int(args.overlay_topk),
            )

    aggregate: dict[str, Any] = {
        "schema_version": "ulfloc_scene_detector_audit_v1",
        "scene_detector_checkpoint": str(args.scene_detector_checkpoint),
        "detector_targets": str(args.detector_targets),
        "split_name": split_name,
        "target_mode": target_mode,
        "fusion_rule": str(args.fusion_rule),
        "blend_alpha": float(args.blend_alpha),
        "image_count": int(len(per_image)),
    }
    numeric_keys = sorted({key for row in per_image for key, value in row.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        values = [float(row[key]) for row in per_image if isinstance(row.get(key), (int, float))]
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
    manifest = {
        "schema_version": "ulfloc_scene_detector_audit_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "source_path": str(args.source_path),
        "images": str(args.images),
        "masks_applied": masks is not None,
        "metrics": aggregate,
        "split_audit": split_audit,
    }
    _write_json(output_dir / "metrics_summary.json", aggregate)
    _write_json(output_dir / "per_image_metrics.json", {"images": per_image})
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
