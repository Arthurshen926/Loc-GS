#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from loc_gs.training.ulfloc_superpoint_teacher import (
    SCHEMA_VERSION,
    build_superpoint_teacher_targets_with_metrics,
    validate_superpoint_teacher_split_name,
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


def _detections_from_json(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    detections = payload.get("detections", payload.get("targets", payload.get("keypoints")))
    if not isinstance(detections, Mapping):
        raise KeyError("input detections JSON must contain a detections mapping")
    return detections


def _metadata_int(args: argparse.Namespace, payload: Mapping[str, Any], key: str) -> int:
    cli_value = getattr(args, key)
    if cli_value is not None:
        value = int(cli_value)
    elif payload.get(key) is not None:
        value = int(payload[key])
    else:
        metadata = payload.get("metadata")
        value = int(metadata[key]) if isinstance(metadata, Mapping) and metadata.get(key) is not None else 0
    if value <= 0:
        raise ValueError(f"{key} is required and must be positive")
    return value


def _image_root(args: argparse.Namespace) -> Path:
    if args.image_root is not None:
        return Path(args.image_root)
    if args.source_path is None:
        raise ValueError("real SuperPoint extraction requires --image_root or --source_path")
    images = str(args.images or "images")
    return Path(args.source_path) / images


def _image_paths(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.image_list is not None:
        rows = []
        for raw_line in Path(args.image_list).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split()[0]
            if first.lower() == "visual":
                continue
            rows.append(first)
        paths = [Path(row) if Path(row).is_absolute() else root / row for row in rows]
    else:
        paths = sorted(path for path in root.rglob(str(args.image_glob)) if path.is_file())
    if args.max_images is not None:
        paths = paths[: int(args.max_images)]
    if not paths:
        raise FileNotFoundError(f"no images found under {root}")
    return paths


def _extract_real_superpoint_detections(args: argparse.Namespace) -> tuple[dict[str, Any], int, int, list[str]]:
    try:
        from PIL import Image
        import torchvision.transforms.functional as tvf
    except Exception as exc:  # pragma: no cover - exercised only in real extraction mode.
        raise RuntimeError(f"real SuperPoint extraction requires PIL and torchvision: {exc}") from exc

    ulf_root = Path(args.ulf_root)
    if not ulf_root.exists():
        raise FileNotFoundError(f"ULF-Loc root does not exist: {ulf_root}")
    sys.path.insert(0, str(ulf_root))
    try:
        from encoders.sp_encoder.export_image_embeddings import SuperPoint
    except Exception as exc:  # pragma: no cover - depends on local ULF install.
        raise RuntimeError(f"failed to import ULF-Loc SuperPoint from {ulf_root}: {exc}") from exc

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = SuperPoint().to(device).eval()
    root = _image_root(args)
    paths = _image_paths(args, root)
    detections: dict[str, Any] = {}
    height = 0
    width = 0
    with torch.inference_mode():
        for path in paths:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            tensor = tvf.to_tensor(image).unsqueeze(0).to(device)
            result = model.detectAndCompute(
                tensor,
                top_k=int(args.top_k) if args.top_k is not None else None,
                detection_threshold=float(args.detection_threshold),
            )[0]
            keypoints = result["keypoints"].detach().cpu()
            scores = result["keypoint_scores"].detach().cpu()
            rows: list[dict[str, Any]] = []
            for point, score in zip(keypoints, scores):
                rows.append(
                    {
                        "keypoint_xy": [float(point[0].item()), float(point[1].item())],
                        "score": float(score.item()),
                    }
                )
            try:
                image_name = str(path.relative_to(root))
            except ValueError:
                image_name = path.name
            detections[image_name] = rows
    return detections, int(height), int(width), [str(path) for path in paths]


def _split_audit(*, split_name: str, source: str) -> dict[str, Any]:
    return {
        "schema_version": "ulfloc_superpoint_teacher_split_audit_v1",
        "audit_status": "passed",
        "split_name": split_name,
        "test_split_used": False,
        "official_test_used": False,
        "teacher_source": source,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a ULF-Loc SuperPoint teacher target artifact.")
    parser.add_argument("--input_detections_json", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--height", default=None, type=int)
    parser.add_argument("--width", default=None, type=int)
    parser.add_argument("--data_root", action="append", default=[])
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--image_root", default=None, type=Path)
    parser.add_argument("--source_path", default=None, type=Path)
    parser.add_argument("--images", default="images")
    parser.add_argument("--image_list", default=None, type=Path)
    parser.add_argument("--image_glob", default="*.png")
    parser.add_argument("--top_k", default=2048, type=int)
    parser.add_argument("--detection_threshold", default=0.0, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", default=None, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    split_name = validate_superpoint_teacher_split_name(str(args.split_name))
    source = "real_superpoint"
    input_paths: list[str] = []
    if args.input_detections_json is not None:
        payload = _load_json(Path(args.input_detections_json))
        payload_split = str(payload.get("split_name", split_name))
        validate_superpoint_teacher_split_name(payload_split)
        if payload_split != split_name and payload_split.lower() != "unknown":
            raise ValueError(f"split_name mismatch: CLI {split_name!r} != detections {payload_split!r}")
        height = _metadata_int(args, payload, "height")
        width = _metadata_int(args, payload, "width")
        detections = _detections_from_json(payload)
        source = "input_detections_json"
        input_paths = [str(args.input_detections_json)]
    else:
        detections, height, width, input_paths = _extract_real_superpoint_detections(args)

    targets, metrics = build_superpoint_teacher_targets_with_metrics(
        detections,
        height=int(height),
        width=int(width),
        split_name=split_name,
    )
    metrics = dict(metrics)
    metrics.update({"scene": str(args.scene), "teacher_source": source})
    split_audit = _split_audit(split_name=split_name, source=source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(args.scene),
        "split_name": split_name,
        "targets": targets,
        "metadata": metrics,
        "split_audit": split_audit,
        "data_roots": [str(path) for path in (args.data_root or [])],
    }
    manifest = {
        "schema_version": "ulfloc_superpoint_teacher_manifest_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": _command(),
        "scene": str(args.scene),
        "split_name": split_name,
        "data_roots": [str(path) for path in (args.data_root or [])],
        "ulf_root": str(args.ulf_root),
        "input_paths": input_paths,
        "hyperparameters": {
            "height": int(height),
            "width": int(width),
            "top_k": int(args.top_k) if args.top_k is not None else None,
            "detection_threshold": float(args.detection_threshold),
            "teacher_source": source,
        },
        "feedback_enabled": {"residual": False, "selector": False, "rho": False},
        "outputs": {
            "superpoint_teacher": str(output_dir / "superpoint_teacher.pt"),
            "metrics_summary": str(output_dir / "metrics_summary.json"),
            "split_audit": str(output_dir / "split_audit.json"),
        },
    }
    torch.save(artifact, output_dir / "superpoint_teacher.pt")
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
