#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import torch

from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name
from loc_gs.feedback.rendered_feedback_augmentation import (
    SUPPORTED_MODES,
    accumulate_rendered_feedback,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_repo_root()), text=True).strip()
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(_repo_root()), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _command(argv: list[str] | None) -> str:
    values = sys.argv if argv is None else [sys.argv[0], *argv]
    return " ".join(shlex.quote(str(item)) for item in values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_jsonl_observations(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"observation line {line_no} must be a JSON object")
            if item.get("type") == "manifest":
                continue
            if item.get("type") in {"observation", "record"}:
                yield dict(item.get("observation", item.get("record", {})))
            else:
                yield dict(item)


def _read_jsonl_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"observation line {line_no} must be a JSON object")
            if item.get("type") == "manifest":
                return dict(item.get("manifest", {}))
    return {}


def load_observations(path: Path) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"observations file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_manifest(path), _iter_jsonl_observations(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"observations JSON must contain an object: {path}")
    manifest = dict(payload.get("manifest", {}))
    observations = (dict(item) for item in payload.get("observations", []))
    return manifest, observations


def _manifest_string(manifest: dict[str, Any], keys: tuple[str, ...], default: str = "unknown") -> str:
    for key in keys:
        value = str(manifest.get(key, "")).strip()
        if value:
            return value
    return default


def _data_roots(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    if args.data_root:
        return [str(path) for path in args.data_root]
    roots = manifest.get("data_roots")
    if isinstance(roots, list) and roots:
        return [str(path) for path in roots]
    root = str(manifest.get("data_root", "")).strip()
    return [root] if root else ["unknown"]


def _metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    support = torch.as_tensor(artifact["support_score"], dtype=torch.float32)
    observed = torch.as_tensor(artifact["observed_count"], dtype=torch.long)
    positive = torch.as_tensor(artifact["positive_observed_count"], dtype=torch.long)
    hard_negative = torch.as_tensor(artifact["hard_negative_risk"], dtype=torch.float32)
    artifact_risk = torch.as_tensor(artifact["artifact_risk"], dtype=torch.float32)
    contribution_mass = torch.as_tensor(artifact["contribution_mass"], dtype=torch.float32)
    observed_mask = observed > 0
    metadata = dict(artifact.get("metadata", {}))
    return {
        "schema_version": artifact["schema_version"],
        "mode": metadata.get("mode", "unknown"),
        "split_name": metadata.get("split_name", "unknown"),
        "num_gaussians": int(support.numel()),
        "observed_gaussians": int(observed_mask.sum().item()),
        "positive_observed_gaussians": int((positive > 0).sum().item()),
        "observation_count": int(metadata.get("observation_count", 0)),
        "rendered_observation_count": int(metadata.get("rendered_observation_count", 0)),
        "support_mean_observed": float(support[observed_mask].mean().item()) if observed_mask.any() else 0.0,
        "support_max": float(support.max().item()) if support.numel() else 0.0,
        "hard_negative_mean_observed": (
            float(hard_negative[observed_mask].mean().item()) if observed_mask.any() else 0.0
        ),
        "artifact_risk_mean_observed": (
            float(artifact_risk[observed_mask].mean().item()) if observed_mask.any() else 0.0
        ),
        "contribution_mass_sum": float(contribution_mass.sum().item()) if contribution_mass.numel() else 0.0,
        "metadata": metadata,
    }


def _split_audit(*, split_name: str, source_split_name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    split_names_seen = list(artifact.get("metadata", {}).get("split_names_seen", []))
    source_unknown = source_split_name == "unknown"
    mixed_non_test = bool(split_names_seen) and set(split_names_seen) != {split_name}
    audit_status = "passed"
    if source_unknown:
        audit_status = "unknown"
    elif mixed_non_test:
        audit_status = "passed_with_mixed_non_test_splits"
    return {
        "audit_status": audit_status,
        "paper_safe": audit_status == "passed",
        "split_name": split_name,
        "source_split_name": source_split_name,
        "split_names_seen": split_names_seen,
        "test_split_used": False,
        "notes": (
            "Missing source split information is marked unknown and should stay out of paper-safe tables."
            if source_unknown
            else "Rendered feedback builder rejects test split inputs."
        ),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build rendered-depth augmented Gaussian feedback from observation JSONL records."
    )
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--num_gaussians", required=True, type=int)
    parser.add_argument("--mode", default="observed_only", choices=sorted(SUPPORTED_MODES))
    parser.add_argument("--reprojection_threshold_px", default=8.0, type=float)
    parser.add_argument("--weak_outlier_weight", default=0.02, type=float)
    parser.add_argument("--artifact_weight", default=1.0, type=float)
    parser.add_argument("--depth_margin", default=1.0, type=float)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--map_path", default="")
    parser.add_argument("--data_root", action="append", default=[])
    parser.add_argument("--residual_feedback_enabled", action="store_true")
    parser.add_argument("--selector_feedback_enabled", action="store_true")
    parser.add_argument("--rho_feedback_enabled", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = validate_split_name(str(args.split_name))
    manifest_in, observations = load_observations(Path(args.observations))
    source_split = str(manifest_in.get("split_name", manifest_in.get("split", "unknown"))).strip() or "unknown"
    if source_split != "unknown":
        validate_split_name(source_split)

    artifact = accumulate_rendered_feedback(
        observations,
        num_gaussians=int(args.num_gaussians),
        split_name=split_name,
        mode=str(args.mode),
        reprojection_threshold_px=float(args.reprojection_threshold_px),
        weak_outlier_weight=float(args.weak_outlier_weight),
        artifact_weight=float(args.artifact_weight),
        depth_margin=float(args.depth_margin),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "rendered_feedback_support.pt"
    torch.save(artifact, artifact_path)

    checkpoint_path = str(args.checkpoint_path).strip() or _manifest_string(
        manifest_in, ("checkpoint_path", "checkpoint"), default="unknown"
    )
    map_path = str(args.map_path).strip() or _manifest_string(manifest_in, ("map_path", "map"), default="unknown")
    feedback_flags = {
        "rendered_depth_feedback_enabled": str(args.mode) == "rendered_depth_augmented",
        "residual_feedback_enabled": bool(args.residual_feedback_enabled),
        "selector_feedback_enabled": bool(args.selector_feedback_enabled),
        "rho_feedback_enabled": bool(args.rho_feedback_enabled),
    }
    manifest = {
        "schema_version": artifact["schema_version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": _command(argv),
        "scene": str(args.scene),
        "split_name": split_name,
        "mode": str(args.mode),
        "checkpoint_path": checkpoint_path,
        "map_path": map_path,
        "data_roots": _data_roots(args, manifest_in),
        "observations": str(Path(args.observations)),
        "artifact_path": str(artifact_path),
        "hyperparameters": dict(artifact["metadata"]["hyperparameters"]),
        "feedback_flags": feedback_flags,
        "source_manifest": manifest_in,
    }
    metrics = _metrics(artifact)
    split_audit = _split_audit(split_name=split_name, source_split_name=source_split, artifact=artifact)

    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
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
