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

from loc_gs.feedback.cross_view_ray_observations import (
    SCHEMA_VERSION,
    build_cross_view_ray_observations,
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


def _load_records(path: Path, *, record_key: str, type_names: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        manifest: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"line {line_no} must be a JSON object: {path}")
            row_type = str(item.get("type", "")).strip()
            if row_type == "manifest":
                manifest = dict(item.get("manifest", {}))
            elif row_type in type_names:
                records.append(dict(item.get(record_key, item.get("record", {}))))
            else:
                records.append(dict(item))
        return manifest, records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return dict(payload.get("manifest", {})), [dict(item) for item in payload.get(record_key + "s", [])]


def load_matches(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _load_records(path, record_key="match", type_names={"match", "record"})


def load_source_rays(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _load_records(path, record_key="ray", type_names={"ray", "record"})


def _manifest_split(manifest: dict[str, Any], fallback: str) -> str:
    return str(manifest.get("split_name", manifest.get("split", fallback))).strip() or fallback


def write_observations_jsonl(path: Path, manifest: dict[str, Any], observations: list[dict[str, Any]]) -> None:
    rows = [{"type": "manifest", "manifest": manifest}]
    rows.extend({"type": "observation", "observation": record} for record in observations)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join precomputed cross-view matches with source ray Gaussian contributors."
    )
    parser.add_argument("--matches", required=True, type=Path)
    parser.add_argument("--source_rays", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--radius_px", default=0.0, type=float)
    parser.add_argument("--quantization_px", default=1.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = validate_split_name(str(args.split_name))
    matches_manifest, matches = load_matches(Path(args.matches))
    rays_manifest, source_rays = load_source_rays(Path(args.source_rays))
    matches_split = validate_split_name(_manifest_split(matches_manifest, split_name))
    rays_split = validate_split_name(_manifest_split(rays_manifest, split_name))

    bundle = build_cross_view_ray_observations(
        matches,
        source_rays,
        split_name=split_name,
        radius_px=float(args.radius_px),
        quantization_px=float(args.quantization_px),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "observations.jsonl"
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": _command(argv),
        "scene": str(args.scene),
        "split_name": split_name,
        "matches": str(Path(args.matches)),
        "source_rays": str(Path(args.source_rays)),
        "observations_path": str(observations_path),
        "hyperparameters": {
            "radius_px": float(args.radius_px),
            "quantization_px": float(args.quantization_px),
        },
        "source_manifests": {
            "matches": matches_manifest,
            "source_rays": rays_manifest,
        },
    }
    write_observations_jsonl(observations_path, output_manifest, bundle["observations"])

    metrics = dict(bundle["summary"])
    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "matches_split_name": matches_split,
        "source_rays_split_name": rays_split,
        "test_split_used": False,
        "notes": "Cross-view observation builder rejects test split inputs.",
    }
    _write_json(output_dir / "manifest.json", output_manifest)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(output_manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")

    print(
        json.dumps(
            {
                "observations_path": str(observations_path),
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
