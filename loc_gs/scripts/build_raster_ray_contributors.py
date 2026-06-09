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

from loc_gs.feedback.raster_ray_contributors import (
    SCHEMA_VERSION,
    raster_intersections_to_ray_records,
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


def load_intersections(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"intersections file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        manifest: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"intersection line {line_no} must be a JSON object")
            row_type = str(item.get("type", "")).strip()
            if row_type == "manifest":
                manifest = dict(item.get("manifest", {}))
            elif row_type in {"intersection", "record"}:
                records.append(dict(item.get("intersection", item.get("record", {}))))
            else:
                records.append(dict(item))
        return manifest, records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"intersections JSON must contain an object: {path}")
    return dict(payload.get("manifest", {})), [dict(item) for item in payload.get("intersections", [])]


def _manifest_split(manifest: dict[str, Any], fallback: str) -> str:
    return str(manifest.get("split_name", manifest.get("split", fallback))).strip() or fallback


def write_source_rays_jsonl(path: Path, manifest: dict[str, Any], rays: list[dict[str, Any]]) -> None:
    rows = [{"type": "manifest", "manifest": manifest}]
    rows.extend({"type": "ray", "ray": ray} for ray in rays)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build source ray Gaussian contributor records from packed raster intersections."
    )
    parser.add_argument("--intersections", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_view_id", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--top_k", default=8, type=int)
    parser.add_argument("--min_contribution", default=0.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = validate_split_name(str(args.split_name))
    source_manifest, intersections = load_intersections(Path(args.intersections))
    source_split = validate_split_name(_manifest_split(source_manifest, split_name))

    bundle = raster_intersections_to_ray_records(
        intersections,
        source_view_id=str(args.source_view_id),
        split_name=split_name,
        width=int(args.width),
        top_k=int(args.top_k),
        min_contribution=float(args.min_contribution),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rays_path = output_dir / "source_rays.jsonl"
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": _command(argv),
        "scene": str(args.scene),
        "split_name": split_name,
        "source_view_id": str(args.source_view_id),
        "intersections": str(Path(args.intersections)),
        "source_rays_path": str(source_rays_path),
        "hyperparameters": {
            "width": int(args.width),
            "top_k": int(args.top_k),
            "min_contribution": float(args.min_contribution),
        },
        "source_manifest": source_manifest,
    }
    write_source_rays_jsonl(source_rays_path, output_manifest, bundle["rays"])

    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "source_split_name": source_split,
        "test_split_used": False,
        "notes": "Raster ray contributor builder rejects test split inputs.",
    }
    _write_json(output_dir / "manifest.json", output_manifest)
    _write_json(output_dir / "metrics_summary.json", dict(bundle["summary"]))
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(output_manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")

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
