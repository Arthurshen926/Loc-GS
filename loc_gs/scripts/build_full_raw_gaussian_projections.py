#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.feedback.full_raw_gaussian_projection import (
    PROJECTION_SOURCE,
    SCHEMA_VERSION,
    project_full_raw_gaussians_to_view,
    validate_full_raw_projection_bundle,
)
from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name


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


def _manifest_split(manifest: dict[str, Any], fallback: str) -> str:
    return str(manifest.get("split_name", manifest.get("split", fallback))).strip() or fallback


def load_gaussian_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"gaussian payload not found: {path}")
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("gaussian payload must be a dictionary")
    if "sampled_idx" in payload or "sampled_indices" in payload:
        raise ValueError("full raw projection input must not contain sampled_idx")
    if "xyz" not in payload:
        raise KeyError("gaussian payload must contain xyz")
    return payload


def load_views(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"views file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        manifest: dict[str, Any] = {}
        views: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"view line {line_no} must be a JSON object")
            row_type = str(item.get("type", "")).strip()
            if row_type == "manifest":
                manifest = dict(item.get("manifest", {}))
            elif row_type in {"view", "record"}:
                views.append(dict(item.get("view", item.get("record", {}))))
            else:
                views.append(dict(item))
        return manifest, views

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"views JSON must contain an object: {path}")
    return dict(payload.get("manifest", {})), [dict(item) for item in payload.get("views", [])]


def write_projected_gaussians_jsonl(
    path: Path,
    manifest: dict[str, Any],
    projections: list[dict[str, Any]],
) -> None:
    rows = [{"type": "manifest", "manifest": manifest}]
    rows.extend({"type": "projection", "projection": record} for record in projections)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project all raw Gaussians into train/self-map source views.")
    parser.add_argument("--gaussians", required=True, type=Path)
    parser.add_argument("--views", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = validate_split_name(str(args.split_name))
    gaussian_payload = load_gaussian_payload(Path(args.gaussians))
    views_manifest, views = load_views(Path(args.views))
    views_split = validate_split_name(_manifest_split(views_manifest, split_name))

    all_projections: list[dict[str, Any]] = []
    projected_count = 0
    dropped_behind = 0
    dropped_out_of_frame = 0
    source_gaussian_count: int | None = None
    for view in views:
        view_split = validate_split_name(str(view.get("split_name", view.get("split", split_name))).strip() or split_name)
        del view_split
        bundle = project_full_raw_gaussians_to_view(
            gaussian_xyz=gaussian_payload["xyz"],
            world_to_camera=view["world_to_camera"],
            intrinsic=view["intrinsic"],
            width=int(view["width"]),
            height=int(view["height"]),
            source_view_id=str(view["source_view_id"]),
            split_name=split_name,
            opacity=gaussian_payload.get("opacity"),
            radius_px=gaussian_payload.get("radius_px", gaussian_payload.get("radius")),
        )
        summary = validate_full_raw_projection_bundle(bundle, expected_source_gaussian_count=source_gaussian_count)
        if source_gaussian_count is None:
            source_gaussian_count = int(summary["source_gaussian_count"])
        projected_count += int(summary["projected_gaussian_count"])
        dropped_behind += int(summary["dropped_behind_count"])
        dropped_out_of_frame += int(summary["dropped_out_of_frame_count"])
        all_projections.extend(bundle["projections"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projections_path = output_dir / "projected_gaussians.jsonl"
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "projection_source": PROJECTION_SOURCE,
        "split_name": split_name,
        "views_split_name": views_split,
        "scene": str(args.scene),
        "view_count": int(len(views)),
        "source_gaussian_count": int(source_gaussian_count or 0),
        "projected_gaussian_count": int(projected_count),
        "dropped_behind_count": int(dropped_behind),
        "dropped_out_of_frame_count": int(dropped_out_of_frame),
        "sampled_idx_used": False,
    }
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": _command(argv),
        "scene": str(args.scene),
        "split_name": split_name,
        "projection_source": PROJECTION_SOURCE,
        "sampled_idx_used": False,
        "gaussians": str(Path(args.gaussians)),
        "views": str(Path(args.views)),
        "projections_path": str(projections_path),
        "metrics_summary": metrics,
        "source_manifest": {
            "views": views_manifest,
        },
    }
    write_projected_gaussians_jsonl(projections_path, output_manifest, all_projections)

    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "views_split_name": views_split,
        "projection_source": PROJECTION_SOURCE,
        "sampled_idx_used": False,
        "test_split_used": False,
        "notes": "Full raw Gaussian projection builder rejects test split inputs and sampled_idx payloads.",
    }
    _write_json(output_dir / "manifest.json", output_manifest)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(output_manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")

    print(
        json.dumps(
            {
                "projections_path": str(projections_path),
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
