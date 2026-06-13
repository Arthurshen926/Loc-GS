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

from loc_gs.training.ulfloc_solver_feedback_detector_targets import (
    SCHEMA_VERSION,
    compose_detector_targets_with_metrics,
    validate_detector_target_inputs,
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
        raise TypeError(f"expected dict payload: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _targets(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    targets = payload.get("targets")
    if not isinstance(targets, Mapping):
        raise KeyError("visibility_teacher.pt must contain a targets mapping")
    return targets


def _metadata_int(payload: Mapping[str, Any], name: str) -> int | None:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get(name) is not None:
        return int(metadata[name])
    targets = payload.get("targets")
    if isinstance(targets, Mapping):
        for entry in targets.values():
            if isinstance(entry, Mapping) and entry.get(name) is not None:
                return int(entry[name])
    return None


def _resolve_dimension(args: argparse.Namespace, teacher: Mapping[str, Any], name: str) -> int:
    cli_value = getattr(args, name)
    value = int(cli_value) if cli_value is not None else _metadata_int(teacher, name)
    if value is None:
        raise ValueError(f"{name} is required, either as --{name} or in visibility_teacher.pt metadata")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _data_roots(args: argparse.Namespace, teacher: Mapping[str, Any]) -> list[str]:
    roots = [str(path) for path in (args.data_root or [])]
    if roots:
        return roots
    raw_roots = teacher.get("data_roots", teacher.get("data_root", []))
    if isinstance(raw_roots, (str, Path)):
        return [str(raw_roots)]
    if isinstance(raw_roots, list):
        return [str(item) for item in raw_roots]
    return []


def _path_arg_or_payload(args: argparse.Namespace, attr: str, payload: Mapping[str, Any]) -> str:
    value = getattr(args, attr)
    if value is not None:
        return str(value)
    payload_value = payload.get(attr)
    return "unknown" if payload_value is None else str(payload_value)


def _split_audit(
    *,
    split_name: str,
    teacher_split_name: str,
    impact_split_name: str,
    teacher_audit: Mapping[str, Any],
) -> dict[str, Any]:
    has_unknown = any(split.lower() == "unknown" for split in (teacher_split_name, impact_split_name))
    inherited_status = str(teacher_audit.get("audit_status", "")) if teacher_audit else ""
    status = "unknown" if has_unknown or inherited_status == "unknown" else "passed"
    return {
        "schema_version": "ulfloc_solver_feedback_detector_targets_split_audit_v1",
        "audit_status": status,
        "split_name": split_name,
        "teacher_split_name": teacher_split_name,
        "impact_split_name": impact_split_name,
        "test_split_used": False,
        "official_test_used": False,
        "visibility_teacher_audit_status": inherited_status or "unknown",
        "notes": "Builder rejects test split inputs and keeps visibility-teacher positives separate from solver-feedback negative suppression.",
    }


def _manifest(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    teacher: Mapping[str, Any],
    impact: Mapping[str, Any],
    scene: str,
    split_name: str,
    height: int,
    width: int,
    metrics: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    return {
        "schema_version": "ulfloc_solver_feedback_detector_targets_manifest_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": _command(),
        "scene": scene,
        "split": split_name,
        "split_name": split_name,
        "checkpoint_path": _path_arg_or_payload(args, "checkpoint_path", teacher),
        "map_path": _path_arg_or_payload(args, "map_path", teacher),
        "data_roots": _data_roots(args, teacher),
        "hyperparameters": {
            "height": int(height),
            "width": int(width),
            "positive_source": "superpoint_teacher_plus_stdloc_visibility_teacher",
            "negative_source": "solver_feedback_impact.detector_negative",
            "target_storage": "points",
        },
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": False,
        "rho_feedback_enabled": False,
        "feedback_enabled": {
            "residual": False,
            "selector": False,
            "rho": False,
        },
        "solver_feedback_negative_suppression_enabled": True,
        "visibility_teacher": str(args.visibility_teacher),
        "superpoint_teacher": str(args.superpoint_teacher) if args.superpoint_teacher is not None else None,
        "impact_attribution": str(args.impact_attribution),
        "output_dir": str(output_dir),
        "metrics": dict(metrics),
        "split_audit": dict(split_audit),
        "impact_schema_version": str(impact.get("schema_version", "unknown")),
        "outputs": {
            "detector_targets": str(output_dir / "detector_targets.pt"),
            "metrics_summary": str(output_dir / "metrics_summary.json"),
            "split_audit": str(output_dir / "split_audit.json"),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose ULF-Loc scene detector targets from visibility-teacher positives and solver-feedback negative suppression."
    )
    parser.add_argument("--visibility_teacher", required=True, type=Path)
    parser.add_argument("--superpoint_teacher", default=None, type=Path)
    parser.add_argument("--impact_attribution", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split_name", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--height", default=None, type=int)
    parser.add_argument("--width", default=None, type=int)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--map_path", default=None, type=Path)
    parser.add_argument("--data_root", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    teacher = _load_payload(Path(args.visibility_teacher))
    sp_teacher = _load_payload(Path(args.superpoint_teacher)) if args.superpoint_teacher is not None else None
    impact = _load_payload(Path(args.impact_attribution))
    split_name, teacher_split_name, impact_split_name = validate_detector_target_inputs(teacher, impact, sp_teacher)
    if args.split_name is not None and str(args.split_name).strip():
        cli_split = str(args.split_name).strip()
        if cli_split.lower() == "test":
            raise ValueError("test split is not allowed for ULF-Loc solver-feedback detector targets")
        if split_name != "unknown" and cli_split != split_name:
            raise ValueError(f"split_name mismatch: CLI {cli_split!r} != inputs {split_name!r}")
        split_name = cli_split
    height = _resolve_dimension(args, teacher, "height")
    width = _resolve_dimension(args, teacher, "width")
    scene = str(args.scene or teacher.get("scene", impact.get("scene", "unknown")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, metrics = compose_detector_targets_with_metrics(
        _targets(teacher),
        impact,
        height=height,
        width=width,
        sp_teacher_targets=sp_teacher,
    )
    metrics = dict(metrics)
    metrics.update(
        {
            "scene": scene,
            "split_name": split_name,
            "teacher_split_name": teacher_split_name,
            "impact_split_name": impact_split_name,
        }
    )
    teacher_audit = teacher.get("split_audit", {})
    if not isinstance(teacher_audit, Mapping):
        teacher_audit = {}
    split_audit = _split_audit(
        split_name=split_name,
        teacher_split_name=teacher_split_name,
        impact_split_name=impact_split_name,
        teacher_audit=teacher_audit,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "split_name": split_name,
        "targets": targets,
        "metadata": metrics,
        "split_audit": split_audit,
    }

    torch.save(artifact, output_dir / "detector_targets.pt")
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(
        output_dir / "manifest.json",
        _manifest(
            args=args,
            repo_root=repo_root,
            teacher=teacher,
            impact=impact,
            scene=scene,
            split_name=split_name,
            height=height,
            width=width,
            metrics=metrics,
            split_audit=split_audit,
        ),
    )
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(
        json.dumps(
            {
                "detector_targets": str(output_dir / "detector_targets.pt"),
                "positive_detector_point_count": int(metrics["positive_detector_point_count"]),
                "negative_suppression_point_count": int(metrics["negative_suppression_point_count"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
