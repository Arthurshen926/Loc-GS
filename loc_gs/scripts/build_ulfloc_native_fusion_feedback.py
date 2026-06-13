#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.feedback.ulfloc_native_fusion_feedback import build_ulfloc_native_fusion_feedback_from_v4


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"git status unavailable: {exc}\n"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = _torch_load(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"feedback_v4 payload must be a dict: {path}")
    return payload


def _iteration_number(path: Path) -> int:
    name = path.name
    if name.startswith("iteration_"):
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    return -1


def _infer_num_gaussians_from_model_path(model_path: Path) -> int:
    point_cloud_root = Path(model_path) / "point_cloud"
    candidates = sorted(
        point_cloud_root.glob("iteration_*/point_cloud.ply"),
        key=lambda path: _iteration_number(path.parent),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no point_cloud.ply found under {point_cloud_root}")
    ply_path = candidates[0]
    with ply_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                try:
                    count = int(stripped.split()[-1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"invalid vertex count in {ply_path}: {stripped}") from exc
                if count <= 0:
                    raise ValueError(f"non-positive vertex count in {ply_path}: {count}")
                return count
            if stripped == "end_header":
                break
    raise ValueError(f"could not infer vertex count from {ply_path}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ULF-Loc native feature-fusion solver feedback from sparse Feedback v4."
    )
    parser.add_argument("--feedback_v4", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--num_gaussians", type=int, default=None)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--risk_penalty", type=float, default=1.0)
    parser.add_argument("--min_weight", type=float, default=0.25)
    parser.add_argument("--max_weight", type=float, default=1.75)
    parser.add_argument("--fusion_policy", choices=("soft_weight", "view_pruning"), default="soft_weight")
    parser.add_argument("--pruning_negative_threshold", type=float, default=0.0)
    parser.add_argument(
        "--negative_role",
        dest="negative_roles",
        action="append",
        default=None,
        help="Label role to consume as native feature-fusion negative. Repeatable. Defaults to harmful_negative only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    command = list(sys.argv if argv is None else [sys.argv[0], *argv])
    repo = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feedback_path = Path(args.feedback_v4)
    payload = _load_payload(feedback_path)
    if args.num_gaussians is None:
        if args.model_path is None:
            raise ValueError("--num_gaussians or --model_path is required")
        num_gaussians = _infer_num_gaussians_from_model_path(Path(args.model_path))
    else:
        num_gaussians = int(args.num_gaussians)
    artifact = build_ulfloc_native_fusion_feedback_from_v4(
        payload,
        scene=str(args.scene),
        num_gaussians=num_gaussians,
        alpha=float(args.alpha),
        risk_penalty=float(args.risk_penalty),
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
        fusion_policy=str(args.fusion_policy),
        pruning_negative_threshold=float(args.pruning_negative_threshold),
        negative_roles=args.negative_roles,
    )
    with (output / "solver_feedback.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)

    metadata = dict(artifact.get("metadata", {}))
    view_summary = metadata.get("view_landmark_weight_summary", {})
    if not isinstance(view_summary, Mapping):
        view_summary = {}
    weights = torch.as_tensor(artifact["landmark_weights"], dtype=torch.float32)
    metrics = {
        "schema_version": "ulfloc_native_fusion_feedback_metrics_v1",
        "scene": str(args.scene),
        "split_name": str(artifact["split_name"]),
        "num_landmarks": int(artifact["num_landmarks"]),
        "positive_count": int(metadata.get("positive_count", 0)),
        "negative_count": int(metadata.get("negative_count", 0)),
        "ignored_negative_count": int(metadata.get("ignored_negative_count", 0)),
        "neutral_outlier_count": int(metadata.get("neutral_outlier_count", 0)),
        "positive_landmark_count": int(metadata.get("positive_landmark_count", 0)),
        "negative_landmark_count": int(metadata.get("negative_landmark_count", 0)),
        "view_landmark_weights_view_count": int(view_summary.get("view_count", 0)),
        "view_landmark_weights_entry_count": int(view_summary.get("entry_count", 0)),
        "view_landmark_weights_negative_entry_count": int(view_summary.get("negative_entry_count", 0)),
        "weight_min": float(weights.min().item()) if weights.numel() else 1.0,
        "weight_max": float(weights.max().item()) if weights.numel() else 1.0,
        "weight_mean": float(weights.mean().item()) if weights.numel() else 1.0,
        "fusion_policy": str(metadata.get("fusion_policy", args.fusion_policy)),
        "weight_application": str(metadata.get("weight_application", "")),
        "hyperparameters": metadata.get("hyperparameters", {}),
    }
    split_audit = {
        "schema_version": "ulfloc_native_fusion_feedback_split_audit_v1",
        "split_name": str(artifact["split_name"]),
        "test_split_used": False,
        "official_test_used": False,
        "role": "selfmap_sparse_feedback_for_native_train_view_feature_fusion",
        "source_split_audit": payload.get("split_audit", {}),
    }
    manifest = {
        "schema_version": "ulfloc_native_fusion_feedback_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "command": command,
        "scene": str(args.scene),
        "feedback_v4": str(feedback_path.resolve()),
        "model_path": None if args.model_path is None else str(Path(args.model_path).resolve()),
        "output_dir": str(output.resolve()),
        "solver_feedback": str((output / "solver_feedback.pkl").resolve()),
        "role": "native_train_view_feature_fusion_feedback_export",
        "metrics": metrics,
        "split_audit": split_audit,
    }
    (output / "metrics_summary.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "split_audit.json").write_text(
        json.dumps(_json_safe(split_audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n",
        encoding="utf-8",
    )
    (output / "git_status.txt").write_text(_git_status(repo), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
