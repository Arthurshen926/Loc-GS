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
import numpy as np

from loc_gs.feedback.solver_feedback_impact import build_solver_feedback_impact


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


def _load_jsonl(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if not isinstance(item, dict):
            continue
        if item.get("type") == "manifest":
            manifest = dict(item.get("manifest", {}))
        elif item.get("type") == "record":
            record = item.get("record", {})
            if isinstance(record, dict):
                records.append(record)
        else:
            records.append(item)
    return {"manifest": manifest, "records": records}


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".pt":
        payload = _torch_load(path)
    elif path.suffix.lower() == ".jsonl":
        payload = _load_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"feedback payload must be a dict: {path}")
    return payload


def _json_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "::".join(str(part) for part in key)
    return str(key)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_json_key(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build split-safe positive/negative impact attribution from sparse PnP feedback records."
    )
    parser.add_argument("--feedback", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    command = list(sys.argv if argv is None else [sys.argv[0], *argv])
    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feedback_path = Path(args.feedback)
    payload = _load_payload(feedback_path)
    impact = build_solver_feedback_impact(payload)

    torch.save(impact, output_dir / "impact_attribution.pt")
    (output_dir / "impact_attribution.json").write_text(
        json.dumps(_json_safe(impact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = dict(impact.get("metadata", {}))
    metrics.update(
        {
            "schema_version": "solver_feedback_impact_metrics_v1",
            "split_name": impact["split_name"],
            "record_count": int(impact.get("record_count", 0)),
        }
    )
    split_audit = {
        "schema_version": "solver_feedback_impact_split_audit_v1",
        "split_name": impact["split_name"],
        "test_split_used": False,
        "official_test_used": False,
        "role": "solver_feedback_impact_from_non_test_sparse_feedback",
    }
    manifest = {
        "schema_version": "solver_feedback_impact_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "command": command,
        "feedback": str(feedback_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "split_name": impact["split_name"],
        "test_split_used": False,
        "outputs": {
            "impact_attribution_pt": str((output_dir / "impact_attribution.pt").resolve()),
            "impact_attribution_json": str((output_dir / "impact_attribution.json").resolve()),
            "metrics_summary": str((output_dir / "metrics_summary.json").resolve()),
            "split_audit": str((output_dir / "split_audit.json").resolve()),
        },
        "metrics": metrics,
        "split_audit": split_audit,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_audit.json").write_text(
        json.dumps(_json_safe(split_audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n",
        encoding="utf-8",
    )
    (output_dir / "git_status.txt").write_text(_git_status(repo), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
