#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.feedback.correspondence_supervision import (
    build_correspondence_supervision,
    export_supervision_targets,
)
from loc_gs.feedback.io import load_feedback_bank
from loc_gs.feedback.schema import FeedbackMatchRecord


def _git_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(cwd), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build correspondence-level supervision from a feedback bank.")
    parser.add_argument("--feedback_bank", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split_name", default="")
    parser.add_argument(
        "--sparse_validation_profile",
        type=Path,
        default=None,
        help="Optional non-test sparse PnP validation profile; query_regression_delta_cm becomes hard-negative supervision.",
    )
    return parser


def _query_regression_delta_from_profile(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sparse_validation_profile must contain a JSON object")
    if str(payload.get("split_name", "")).strip().lower() == "test":
        raise ValueError("test split sparse validation profile is not allowed for correspondence supervision")
    raw = payload.get("query_regression_delta_cm", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for query_id, value in raw.items():
        try:
            delta = float(value)
        except (TypeError, ValueError):
            continue
        if delta > 0.0:
            out[str(query_id)] = float(delta)
    return out


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    bank = load_feedback_bank(args.feedback_bank)
    manifest = dict(bank.get("manifest", {}))
    split_name = str(args.split_name or manifest.get("split_name", manifest.get("split", "")))
    records = [FeedbackMatchRecord.from_mapping(record) for record in bank.get("records", [])]
    query_regression_delta = _query_regression_delta_from_profile(args.sparse_validation_profile)
    artifact = build_correspondence_supervision(
        records,
        split_name=split_name,
        query_regression_delta_cm=query_regression_delta,
    )
    targets = export_supervision_targets(artifact)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": "correspondence_supervision_metrics_v1",
        "record_count": int(artifact["record_count"]),
        "positive_count": int(artifact["positive_count"]),
        "negative_count": int(artifact["record_count"] - artifact["positive_count"]),
        "hard_negative_count": int(artifact.get("hard_negative_count", 0)),
        "descriptor_landmark_count": int(len(targets["descriptor_fusion"]["landmark_weights"])),
        "ranking_preference_count": int(len(targets["pnp_ranking"]["pairwise_preferences"])),
        "conflict_edge_count": int(len(targets["conflict_graph"]["edges"])),
    }
    split_audit = {
        "schema_version": "correspondence_supervision_split_audit_v1",
        "split_name": split_name,
        "test_split_used": split_name.strip().lower() == "test",
        "source_feedback_bank": str(args.feedback_bank),
    }
    run_manifest = {
        "schema_version": "correspondence_supervision_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "feedback_bank": str(args.feedback_bank),
        "output_dir": str(output_dir),
        "sparse_validation_profile": str(args.sparse_validation_profile) if args.sparse_validation_profile else None,
        "source_manifest": manifest,
        "split_name": split_name,
    }
    _write_json(output_dir / "correspondence_supervision.json", artifact)
    _write_json(output_dir / "targets.json", targets)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", run_manifest)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path.cwd()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
