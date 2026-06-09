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
from PIL import Image

from loc_gs.localization.stdloc_detector import StdlocKeypointDetector
from loc_gs.training.ulfloc_scene_detector import (
    load_detector_target_artifact,
    make_trainable_detector_input,
    rasterize_compact_detector_target,
    scene_detector_loss,
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
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def _import_ulfloc_feature_extractor(ulf_root: Path):
    root = str(Path(ulf_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from encoders.feature_extractor import FeatureExtractor  # type: ignore

    return FeatureExtractor


def _seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ULF-Loc scene-specific sparse detector from solver-feedback targets.")
    parser.add_argument("--detector_targets", required=True, type=Path)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_images", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--positive_weight", default=6.0, type=float)
    parser.add_argument("--negative_weight", default=0.25, type=float)
    parser.add_argument("--sigma_cells", default=1.0, type=float)
    parser.add_argument("--solver_validity_power", default=0.0, type=float)
    parser.add_argument("--descriptor_stride", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    _seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_detector_target_artifact(args.detector_targets)
    split_audit = dict(artifact.get("split_audit", {}))
    split_name = str(artifact.get("split_name", split_audit.get("split_name", "unknown")))
    targets = artifact["targets"]
    if not isinstance(targets, Mapping):
        raise TypeError("detector target artifact targets must be a mapping")

    device = torch.device(args.device)
    detector = StdlocKeypointDetector(in_dim=256).to(device)
    history: list[dict[str, float | int]] = []
    target_items = _iter_target_items(
        targets,
        source_path=Path(args.source_path),
        images=str(args.images),
        max_images=int(args.max_images),
    )

    if int(args.epochs) > 0:
        if device.type != "cuda":
            raise ValueError("ULF-Loc FeatureExtractor currently requires CUDA for nonzero detector training epochs")
        if not target_items:
            raise ValueError("no detector target images found under source_path/images")
        FeatureExtractor = _import_ulfloc_feature_extractor(Path(args.ulf_root))
        feature_extractor = FeatureExtractor(str(args.feature_type)).cuda().eval()
        optimizer = torch.optim.AdamW(detector.parameters(), lr=float(args.lr), weight_decay=1e-4)
        for epoch in range(int(args.epochs)):
            detector.train()
            total_loss = 0.0
            used = 0
            for _query_id, image_path, entry in target_items:
                image = _load_image(image_path, device)
                with torch.no_grad():
                    feature_map, _scores = feature_extractor.detectAndComputeDense(image[None])
                    feature_map = make_trainable_detector_input(feature_map[0]).to(device=device)
                target, weight, _target_meta = rasterize_compact_detector_target(
                    entry,
                    coarse_height=int(feature_map.shape[-2]),
                    coarse_width=int(feature_map.shape[-1]),
                    stride=int(args.descriptor_stride),
                    sigma_cells=float(args.sigma_cells),
                    solver_validity_power=float(args.solver_validity_power),
                )
                target = target.to(device=device)
                weight = weight.to(device=device)
                prediction = detector(feature_map)
                if prediction.dim() == 3:
                    prediction = prediction.unsqueeze(0)
                loss = scene_detector_loss(
                    prediction,
                    target,
                    weight,
                    positive_weight=float(args.positive_weight),
                    negative_weight=float(args.negative_weight),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                used += 1
            history.append({"epoch": int(epoch), "mean_loss": float(total_loss / max(used, 1)), "image_count": int(used)})

    checkpoint = {
        "state_dict": detector.state_dict(),
        "metadata": {
            "schema_version": "ulfloc_scene_detector_checkpoint_v1",
            "split_name": split_name,
            "source_detector_targets": str(args.detector_targets),
            "feature_type": str(args.feature_type),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "solver_validity_power": float(args.solver_validity_power),
            "trained_image_count": int(len(target_items) if int(args.epochs) > 0 else 0),
            "available_target_image_count": int(len(target_items)),
            "history": history,
        },
    }
    checkpoint_path = output_dir / "scene_detector.pth"
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "schema_version": "ulfloc_scene_detector_training_metrics_v1",
        "split_name": split_name,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "solver_validity_power": float(args.solver_validity_power),
        "trained_image_count": int(len(target_items) if int(args.epochs) > 0 else 0),
        "available_target_image_count": int(len(target_items)),
        "final_loss": float(history[-1]["mean_loss"]) if history else None,
        "history": history,
    }
    manifest = {
        "schema_version": "ulfloc_scene_detector_training_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "detector_targets": str(args.detector_targets),
        "source_path": str(args.source_path),
        "images": str(args.images),
        "seed": int(args.seed),
        "solver_validity_power": float(args.solver_validity_power),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "paper_safe_role": "train_selfmap_sparse_detector_only",
        "metrics": metrics,
        "split_audit": split_audit,
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
