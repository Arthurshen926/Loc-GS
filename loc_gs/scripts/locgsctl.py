#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loc_gs.stdloc_native.commands import CAMBRIDGE_SCENES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_ROOT = Path("output/stdloc/map_cambridge_spgs")
DEFAULT_CHECKPOINT_ROOT = Path("output/stdloc_hybrid")
DEFAULT_DATA_ROOT = Path("/mnt/pool/sqy/Cambridge_stdloc")
DEFAULT_MAP_NAME_OVERRIDES = {
    "GreatCourt": "GreatCourt_stream_stable2",
    "StMarysChurch": "StMarysChurch_stream_fastsave",
}


def _json_dump(payload: dict[str, Any] | list[Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _print_json(payload: dict[str, Any] | list[Any]) -> None:
    print(_json_dump(payload))


def _resolve_repo_path(repo_root: Path, path: str | Path) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else repo_root / raw


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _path_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def command_status(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    paths = {
        "repo_root": repo_root,
        "loc_gs": repo_root / "loc_gs",
        "docs": repo_root / "docs",
        "tests": repo_root / "tests",
        "third_party_stdloc": repo_root / "third_party" / "stdloc",
        "output": repo_root / "output",
        "stdloc_superpoint_weights": repo_root
        / "third_party"
        / "stdloc"
        / "encoders"
        / "sp_encoder"
        / "weights"
        / "superpoint_v1.pth",
    }
    _print_json(
        {
            "git_commit": _git_commit(repo_root),
            "python_executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "paths": {key: _path_status(path) for key, path in paths.items()},
        }
    )
    return 0


def _scene_defaults(
    repo_root: Path,
    *,
    map_root: str | Path = DEFAULT_MAP_ROOT,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> list[dict[str, Any]]:
    map_base = _resolve_repo_path(repo_root, map_root)
    checkpoint_base = _resolve_repo_path(repo_root, checkpoint_root)
    data_base = _resolve_repo_path(repo_root, data_root)
    rows: list[dict[str, Any]] = []
    for scene in CAMBRIDGE_SCENES:
        map_scene = DEFAULT_MAP_NAME_OVERRIDES.get(scene, scene)
        rows.append(
            {
                "scene": scene,
                "data_root": str(data_base / scene),
                "map_path": str(map_base / map_scene),
                "checkpoint_path": str(checkpoint_base / scene / "latest.pth"),
            }
        )
    return rows


def command_list_scenes(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    _print_json(
        {
            "scenes": _scene_defaults(
                repo_root,
                map_root=args.map_root,
                checkpoint_root=args.checkpoint_root,
                data_root=args.data_root,
            )
        }
    )
    return 0


def _summary_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_dir():
        for filename in ("metrics_summary.json", "summary.json", "metrics.json"):
            candidate = raw / filename
            if candidate.exists():
                return candidate
        return raw / "summary.json"
    return raw


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"summary or metrics file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(data: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _as_float(data.get(name))
        if value is not None:
            return value
    return None


def _compact_stage(data: dict[str, Any]) -> dict[str, float]:
    aliases = {
        "median_te_cm": ("median_te_cm", "median_te", "median_translation_cm"),
        "median_re_deg": ("median_re_deg", "median_ae", "median_re", "median_rotation_deg"),
        "recall_10cm_5deg": ("recall_10cm_5deg", "recall_10cm_5d", "r10"),
        "recall_5cm_5deg": ("recall_5cm_5deg", "recall_5cm_5d", "r5"),
        "recall_2cm_2deg": ("recall_2cm_2deg", "recall_2cm_2d", "r2"),
        "avg_inliers": ("avg_inliers", "mean_inliers"),
    }
    out: dict[str, float] = {}
    for key, names in aliases.items():
        value = _first_float(data, names)
        if value is not None:
            out[key] = value
    return out


def _parse_hyperparameters(raw: str | None) -> dict[str, Any]:
    if raw is None or not str(raw).strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--hyperparameters must be a JSON object")
    return data


def summarize_path(path: str | Path) -> dict[str, Any]:
    source = _summary_path(path)
    data = _load_json(source)
    payload: dict[str, Any] = {"source": str(source)}
    for key in ("model_path", "scene", "run_name", "tag"):
        if key in data:
            payload[key] = data[key]
    for stage in ("sparse", "dense"):
        stage_data = data.get(stage, {})
        if isinstance(stage_data, dict):
            compact = _compact_stage(stage_data)
            if compact:
                payload[stage] = compact
    if "dense" not in payload:
        compact = _compact_stage(data)
        if compact:
            payload["dense"] = compact
    return payload


def command_summarize(args: argparse.Namespace) -> int:
    _print_json(summarize_path(args.output_root))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    baseline = summarize_path(args.baseline)
    candidate = summarize_path(args.candidate)
    stage = str(args.stage)
    baseline_stage = baseline.get(stage, {})
    candidate_stage = candidate.get(stage, {})
    if not isinstance(baseline_stage, dict) or not isinstance(candidate_stage, dict):
        raise KeyError(f"stage '{stage}' missing from baseline or candidate summary")
    keys = (
        "median_te_cm",
        "median_re_deg",
        "recall_10cm_5deg",
        "recall_5cm_5deg",
        "recall_2cm_2deg",
    )
    delta: dict[str, float] = {}
    for key in keys:
        left = _as_float(baseline_stage.get(key))
        right = _as_float(candidate_stage.get(key))
        if left is not None and right is not None:
            delta[key] = right - left
    _print_json(
        {
            "stage": stage,
            "baseline_source": baseline["source"],
            "candidate_source": candidate["source"],
            "baseline": {key: baseline_stage[key] for key in keys if key in baseline_stage},
            "candidate": {key: candidate_stage[key] for key in keys if key in candidate_stage},
            "delta": delta,
        }
    )
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    scene = str(args.scene)
    if scene not in CAMBRIDGE_SCENES:
        raise ValueError(f"unknown Cambridge scene '{scene}'; expected one of {', '.join(CAMBRIDGE_SCENES)}")
    repo_root = Path(args.repo_root).resolve()
    default = _scene_defaults(
        repo_root,
        map_root=args.map_root,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
    )
    row = next(item for item in default if item["scene"] == scene)
    checks = {
        "data_root": _path_status(Path(row["data_root"])),
        "map_path": _path_status(Path(row["map_path"])),
        "checkpoint_path": _path_status(Path(row["checkpoint_path"])),
        "loc_gs_package": _path_status(repo_root / "loc_gs"),
        "stdloc_root": _path_status(repo_root / "third_party" / "stdloc"),
    }
    missing = [key for key, value in checks.items() if not value["exists"]]
    payload = {
        "scene": scene,
        "dry_run": bool(args.dry_run),
        "checks": checks,
        "ok": not missing,
        "missing": missing,
        "would_run_long_experiment": False,
    }
    _print_json(payload)
    return 0 if args.dry_run or not missing else 1


def command_manifest(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    command = list(args.command) if args.command else sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    hyperparameters = _parse_hyperparameters(args.hyperparameters)
    manifest = {
        "git_commit": _git_commit(repo_root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "split": args.split,
        "command": command,
        "checkpoint": args.checkpoint,
        "checkpoint_path": args.checkpoint,
        "map": args.map,
        "map_path": args.map,
        "data_root": args.data_root,
        "data_roots": [args.data_root] if args.data_root else [],
        "hyperparameters": hyperparameters,
        "feedback_enabled": bool(args.feedback_enabled),
        "rho": args.rho,
        "residual_enabled": bool(args.residual_enabled),
        "selector_enabled": bool(args.selector_enabled),
        "notes": args.notes,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _print_json(manifest)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="locgsctl",
        description="Agent-friendly Loc-GS experiment status, summary, comparison, and manifest helper.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root. Defaults to this checkout.")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    status = subparsers.add_parser("status", help="Print environment and key path status as compact JSON.")
    status.set_defaults(func=command_status)

    list_scenes = subparsers.add_parser("list-scenes", help="List Cambridge scenes and default paths.")
    list_scenes.add_argument("--map-root", default=str(DEFAULT_MAP_ROOT))
    list_scenes.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    list_scenes.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    list_scenes.set_defaults(func=command_list_scenes)

    smoke = subparsers.add_parser("smoke", help="Run minimal path checks for one Cambridge scene.")
    smoke.add_argument("--scene", required=True, choices=CAMBRIDGE_SCENES)
    smoke.add_argument("--map-root", default=str(DEFAULT_MAP_ROOT))
    smoke.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    smoke.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    smoke.add_argument("--dry-run", action="store_true", help="Report checks without failing on missing data.")
    smoke.set_defaults(func=command_smoke)

    summarize = subparsers.add_parser("summarize", help="Compact a summary.json or metrics file.")
    summarize.add_argument("output_root", help="Run directory or summary/metrics JSON file.")
    summarize.set_defaults(func=command_summarize)

    compare = subparsers.add_parser("compare", help="Compare baseline and candidate summary metrics.")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--stage", default="dense", choices=("dense", "sparse"))
    compare.set_defaults(func=command_compare)

    manifest = subparsers.add_parser("manifest", help="Generate a manifest.json template.")
    manifest.add_argument("--scene", default="")
    manifest.add_argument("--split", default="")
    manifest.add_argument("--checkpoint", default="")
    manifest.add_argument("--map", default="")
    manifest.add_argument("--data-root", default="")
    manifest.add_argument("--hyperparameters", default="{}", help="JSON object with run hyperparameters.")
    manifest.add_argument("--output", default="", help="Optional path to write manifest JSON.")
    manifest.add_argument("--feedback-enabled", action="store_true")
    manifest.add_argument("--rho", type=float, default=None)
    manifest.add_argument("--residual-enabled", action="store_true")
    manifest.add_argument("--selector-enabled", action="store_true")
    manifest.add_argument("--notes", default="")
    manifest.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    manifest.set_defaults(func=command_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--command" in argv:
        command_idx = argv.index("--command")
        if command_idx + 1 < len(argv) and argv[command_idx + 1] == "--":
            del argv[command_idx + 1]
    parser = build_argparser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"locgsctl: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
