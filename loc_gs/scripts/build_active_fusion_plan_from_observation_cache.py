#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "active_fusion_plan_from_observation_cache_v1"


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


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"observation cache must contain a dict: {path}")
    return dict(payload)


def _is_test_split(value: str) -> bool:
    split = str(value).strip().lower()
    return split == "test" or split == "official_test" or split.endswith("_test")


def _split_name(payload: Mapping[str, Any]) -> str:
    for source in (payload, payload.get("split_audit", {})):
        if not isinstance(source, Mapping):
            continue
        for key in ("split_name", "split"):
            value = str(source.get(key, "")).strip()
            if value:
                return value
    return "unknown"


def _reject_test_payload(payload: Mapping[str, Any]) -> None:
    if _is_test_split(_split_name(payload)):
        raise ValueError("test split is not allowed for active fusion plan construction")
    audit = payload.get("split_audit", {})
    if isinstance(audit, Mapping) and (
        bool(audit.get("test_split_used", False)) or bool(audit.get("official_test_used", False))
    ):
        raise ValueError("test split is not allowed for active fusion plan construction")


def _finite_nonnegative(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not torch.isfinite(torch.tensor(number)):
        return 0.0
    return max(0.0, float(number))


def _row_view_id(row: Mapping[str, Any]) -> str:
    for key in ("source_view_id", "view_id", "image_id", "query_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _row_gaussian_id(raw_key: Any, row: Mapping[str, Any]) -> int | None:
    for key in ("gaussian_id", "matched_gaussian_id", "landmark_id"):
        try:
            gid = int(row.get(key))
        except (TypeError, ValueError):
            continue
        if gid >= 0:
            return gid
    try:
        gid = int(raw_key)
    except (TypeError, ValueError):
        return None
    return gid if gid >= 0 else None


def build_active_fusion_plan_from_observation_cache(
    cache: Mapping[str, Any],
    *,
    scene: str,
    split_name: str,
    min_abs_weight: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _reject_test_payload(cache)
    observations = cache.get("observations")
    if not isinstance(observations, Mapping):
        raise ValueError("observation cache must contain explicit observations mapping")
    threshold = max(0.0, float(min_abs_weight))
    lookup: dict[int, set[str]] = {}
    input_pair_count = 0
    kept_pair_count = 0
    for raw_key, rows in observations.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            input_pair_count += 1
            view_id = _row_view_id(row)
            gid = _row_gaussian_id(raw_key, row)
            if gid is None or not view_id:
                continue
            weight = max(_finite_nonnegative(row.get("positive_weight")), _finite_nonnegative(row.get("negative_weight")))
            if weight < threshold:
                continue
            lookup.setdefault(int(gid), set()).add(str(view_id))
            kept_pair_count += 1
    plan = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(scene),
        "split_name": str(split_name),
        "landmark_fusion_plan": {
            str(gid): {"selected_view_ids": sorted(views)}
            for gid, views in sorted(lookup.items())
            if views
        },
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(scene),
        "split_name": str(split_name),
        "observation_cache_split_name": _split_name(cache),
        "input_pair_count": int(input_pair_count),
        "kept_pair_count": int(kept_pair_count),
        "landmark_count": int(len(lookup)),
        "min_abs_weight": float(threshold),
    }
    return plan, metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an active descriptor-fusion view plan from a multiview observation cache.")
    parser.add_argument("--observation_cache", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--min_abs_weight", default=0.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = str(args.split_name).strip()
    if _is_test_split(split):
        raise ValueError("test split is not allowed for active fusion plan construction")
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_payload(Path(args.observation_cache))
    plan, metrics = build_active_fusion_plan_from_observation_cache(
        cache,
        scene=str(args.scene),
        split_name=split,
        min_abs_weight=float(args.min_abs_weight),
    )
    split_audit = {
        "schema_version": "active_fusion_plan_from_observation_cache_split_audit_v1",
        "split_name": split,
        "observation_cache_split_name": metrics["observation_cache_split_name"],
        "test_split_used": False,
        "official_test_used": False,
        "role": "offline_descriptor_fusion_plan_construction",
    }
    manifest = {
        "schema_version": "active_fusion_plan_from_observation_cache_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv if argv is None else [sys.executable, "-m", "loc_gs.scripts.build_active_fusion_plan_from_observation_cache", *argv],
        "scene": str(args.scene),
        "split_name": split,
        "observation_cache": str(Path(args.observation_cache)),
        "metrics": metrics,
        "split_audit": split_audit,
        "official_test_used": False,
        "branch_selection": False,
    }
    _write_json(output_dir / "landmark_fusion_plan.json", plan)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
