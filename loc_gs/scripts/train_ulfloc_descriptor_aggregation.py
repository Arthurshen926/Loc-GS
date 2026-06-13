#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from loc_gs.training.ulfloc_aggregation_samples import AggregationSample, build_aggregation_samples
from loc_gs.training.ulfloc_descriptor_aggregation import optimize_landmark_descriptor


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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".pt":
        payload = _torch_load(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Feedback v4 artifact must be a dict: {path}")
    return payload


def _feature_table(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.dim() != 2:
        raise ValueError("keypoints_features.pkl must have shape [landmarks, descriptor_dim]")
    return tensor


def _sampled_idx(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long).reshape(-1).detach().cpu()
    if tensor.numel() == 0:
        raise ValueError("keypoints_sampled_idx.pkl must be non-empty")
    return tensor


def _reject_test_feedback(feedback: Mapping[str, Any]) -> None:
    split = str(feedback.get("split_name", feedback.get("split", ""))).strip()
    audit = feedback.get("split_audit", {})
    audit_test = bool(audit.get("test_split_used", False) or audit.get("official_test_used", False)) if isinstance(audit, Mapping) else False
    if split.lower() == "test" or bool(feedback.get("test_split_used", False)) or bool(feedback.get("official_test_used", False)) or audit_test:
        raise ValueError("test split feedback is not allowed for descriptor aggregation")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _samples_by_sampled_row(samples: list[AggregationSample]) -> dict[int, list[AggregationSample]]:
    grouped: dict[int, list[AggregationSample]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample.sampled_row)].append(sample)
    return dict(grouped)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export ULF aggregation-stage descriptors from Feedback v4.")
    parser.add_argument("--source_log_dir", "--input_log_dir", dest="source_log_dir", required=True, type=Path)
    parser.add_argument("--feedback_v4", required=True, type=Path)
    parser.add_argument("--output_log_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min_native_cosine", type=float, default=0.95)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--negative_margin", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=0.2)
    parser.add_argument("--include_render_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    command = list(sys.argv if argv is None else [sys.argv[0], *argv])
    repo = Path(__file__).resolve().parents[2]
    source = Path(args.source_log_dir)
    output = Path(args.output_log_dir)
    feature_path = source / "keypoints_features.pkl"
    sampled_path = source / "keypoints_sampled_idx.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"source log has no keypoints_features.pkl: {source}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"source log has no keypoints_sampled_idx.pkl: {source}")

    native_features = _feature_table(_load_pickle(feature_path))
    sampled_idx = _sampled_idx(_load_pickle(sampled_path))
    if int(sampled_idx.numel()) != int(native_features.shape[0]):
        raise ValueError("keypoints_sampled_idx.pkl length must match keypoints_features.pkl rows")

    feedback = _load_payload(Path(args.feedback_v4))
    _reject_test_feedback(feedback)
    samples = build_aggregation_samples(feedback, include_render_aug=bool(args.include_render_aug))
    grouped = _samples_by_sampled_row(samples)
    rebuilt = F.normalize(native_features.float(), p=2, dim=-1)

    landmark_metadata: dict[str, Any] = {}
    invalid_sampled_row_count = 0
    rebuilt_landmark_count = 0
    for sampled_row, rows in sorted(grouped.items()):
        if sampled_row < 0 or sampled_row >= int(rebuilt.shape[0]):
            invalid_sampled_row_count += 1
            continue
        result = optimize_landmark_descriptor(
            native_descriptor=rebuilt[sampled_row],
            samples=rows,
            steps=int(args.steps),
            lr=float(args.lr),
            min_native_cosine=float(args.min_native_cosine),
            hidden_dim=int(args.hidden_dim),
            negative_margin=float(args.negative_margin),
            anchor_weight=float(args.anchor_weight),
        )
        rebuilt[sampled_row] = result.descriptor.to(dtype=rebuilt.dtype)
        if not result.metadata.get("fallback_reason"):
            rebuilt_landmark_count += 1
        landmark_metadata[str(sampled_row)] = {
            **result.metadata,
            "gaussian_id": int(sampled_idx[sampled_row].item()),
            "sampled_row": int(sampled_row),
        }

    if output.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"output_log_dir already exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)
    restored = rebuilt
    if native_features.dtype.is_floating_point:
        restored = restored.to(dtype=native_features.dtype)
    _dump_pickle(output / "keypoints_features.pkl", restored)

    split_name = str(feedback.get("split_name", "unknown"))
    split_audit = {
        "schema_version": "ulfloc_descriptor_aggregation_split_audit_v1",
        "split_name": split_name,
        "test_split_used": False,
        "official_test_used": False,
        "role": "offline_feedback_v4_descriptor_aggregation",
    }
    metrics = {
        "descriptor_mode": "ulfloc_solver_feedback_aggregation_v1",
        "scene": str(args.scene),
        "source_log_dir": str(source),
        "output_log_dir": str(output),
        "feedback_v4": str(Path(args.feedback_v4)),
        "sample_count": int(len(samples)),
        "sampled_count": int(sampled_idx.numel()),
        "rebuilt_landmark_count": int(rebuilt_landmark_count),
        "candidate_landmark_count": int(len(grouped)),
        "invalid_sampled_row_count": int(invalid_sampled_row_count),
        "include_render_aug": bool(args.include_render_aug),
        "post_hoc_mean_shift_used": False,
        "aggregation_stage_view_selection": True,
        "same_sampled_idx": True,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "min_native_cosine": float(args.min_native_cosine),
        "landmarks": landmark_metadata,
        "split_audit": split_audit,
    }
    manifest = {
        "schema_version": "ulfloc_descriptor_aggregation_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "command": command,
        "scene": str(args.scene),
        "source_log_dir": str(source.resolve()),
        "output_log_dir": str(output.resolve()),
        "feedback_v4": str(Path(args.feedback_v4).resolve()),
        "split_name": split_name,
        "same_sampled_idx": True,
        "post_hoc_mean_shift_used": False,
        "outputs": {
            "keypoints_features": str((output / "keypoints_features.pkl").resolve()),
            "keypoints_sampled_idx": str((output / "keypoints_sampled_idx.pkl").resolve()),
            "metrics_summary": str((output / "metrics_summary.json").resolve()),
            "split_audit": str((output / "split_audit.json").resolve()),
        },
        "metrics": metrics,
        "split_audit": split_audit,
    }

    _write_json(output / "metrics_summary.json", metrics)
    _write_json(output / "split_audit.json", split_audit)
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "descriptor_aggregation_manifest.json", manifest)
    (output / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    (output / "git_status.txt").write_text(_git_status(repo), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
