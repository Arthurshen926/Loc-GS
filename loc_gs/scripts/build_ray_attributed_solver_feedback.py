#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from loc_gs.feedback.ray_attributed_solver_feedback import (
    accumulate_ray_feedback,
    validate_split_name,
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


def load_observations(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"observations file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        manifest: dict[str, Any] = {}
        observations: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if item.get("type") == "manifest":
                manifest = dict(item.get("manifest", {}))
            elif item.get("type") in {"observation", "record"}:
                observations.append(dict(item.get("observation", item.get("record", {}))))
            else:
                if not isinstance(item, dict):
                    raise ValueError(f"observation line {line_no} must be a JSON object")
                observations.append(dict(item))
        return manifest, observations

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"observations JSON must contain an object: {path}")
    return dict(payload.get("manifest", {})), [dict(item) for item in payload.get("observations", [])]


def load_observation_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"observations file not found: {path}")
    if path.suffix.lower() != ".jsonl":
        manifest, _observations = load_observations(path)
        return manifest
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            item = json.loads(raw)
            if item.get("type") == "manifest":
                return dict(item.get("manifest", {}))
            break
    return {}


def iter_observations(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"observations file not found: {path}")
    if path.suffix.lower() != ".jsonl":
        _manifest, observations = load_observations(path)
        yield from observations
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if item.get("type") == "manifest":
                continue
            if item.get("type") in {"observation", "record"}:
                yield dict(item.get("observation", item.get("record", {})))
            else:
                if not isinstance(item, dict):
                    raise ValueError(f"observation line {line_no} must be a JSON object")
                yield dict(item)


def _metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    support = torch.as_tensor(artifact["support_score"], dtype=torch.float32)
    observed = torch.as_tensor(artifact["observed_count"], dtype=torch.long)
    positive = torch.as_tensor(artifact["positive_observed_count"], dtype=torch.long)
    risk = torch.as_tensor(artifact["hard_negative_risk"], dtype=torch.float32)
    artifact_risk = torch.as_tensor(artifact["artifact_risk"], dtype=torch.float32)
    observed_mask = observed > 0
    return {
        "schema_version": artifact["schema_version"],
        "num_gaussians": int(support.numel()),
        "observed_landmarks": int(observed_mask.sum().item()),
        "positive_observed_landmarks": int((positive > 0).sum().item()),
        "support_mean_observed": float(support[observed_mask].mean().item()) if observed_mask.any() else 0.0,
        "support_max": float(support.max().item()) if support.numel() else 0.0,
        "hard_negative_mean_observed": float(risk[observed_mask].mean().item()) if observed_mask.any() else 0.0,
        "artifact_risk_mean_observed": (
            float(artifact_risk[observed_mask].mean().item()) if observed_mask.any() else 0.0
        ),
        **dict(artifact["metadata"]),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ray-attributed Gaussian solver feedback from cross-view observation JSONL."
    )
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--num_gaussians", required=True, type=int)
    parser.add_argument("--reprojection_threshold_px", default=8.0, type=float)
    parser.add_argument("--weak_outlier_weight", default=0.02, type=float)
    parser.add_argument("--artifact_weight", default=1.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = validate_split_name(str(args.split_name))
    observations_path = Path(args.observations)
    manifest_in = load_observation_manifest(observations_path)
    source_split = str(manifest_in.get("split_name", manifest_in.get("split", split_name))).strip() or split_name
    validate_split_name(source_split)

    artifact = accumulate_ray_feedback(
        iter_observations(observations_path),
        num_gaussians=int(args.num_gaussians),
        split_name=split_name,
        reprojection_threshold_px=float(args.reprojection_threshold_px),
        weak_outlier_weight=float(args.weak_outlier_weight),
        artifact_weight=float(args.artifact_weight),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "ray_solver_feedback.pt"
    torch.save(artifact, artifact_path)

    metrics = _metrics(artifact)
    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "source_split_name": source_split,
        "test_split_used": False,
        "notes": "Ray-attributed solver feedback builder rejects test split inputs.",
    }
    manifest = {
        "schema_version": artifact["schema_version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": _command(argv),
        "scene": str(args.scene),
        "split_name": split_name,
        "observations": str(Path(args.observations)),
        "artifact_path": str(artifact_path),
        "hyperparameters": dict(artifact["metadata"]["hyperparameters"]),
        "source_manifest": manifest_in,
    }
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
