#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.feedback.correspondence_supervision import build_correspondence_supervision
from loc_gs.feedback.io import load_feedback_bank
from loc_gs.feedback.schema import FeedbackMatchRecord
from loc_gs.feedback.ulfloc_correspondence_feedback import build_ulfloc_correspondence_feedback


def _git_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(cwd), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ULF-compatible solver feedback from correspondence-level sparse feedback."
    )
    parser.add_argument("--feedback_bank", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_landmarks", required=True, type=int)
    parser.add_argument("--boost_strength", default=0.5, type=float)
    parser.add_argument("--deboost_strength", default=0.5, type=float)
    parser.add_argument("--min_weight", default=0.25, type=float)
    parser.add_argument("--max_weight", default=1.75, type=float)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    bank = load_feedback_bank(args.feedback_bank)
    manifest = dict(bank.get("manifest", {}))
    split_name = str(manifest.get("split_name", manifest.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to build ULF correspondence feedback from test split")
    records = [FeedbackMatchRecord.from_mapping(record) for record in bank.get("records", [])]
    supervision = build_correspondence_supervision(records, split_name=split_name)
    payload = build_ulfloc_correspondence_feedback(
        supervision,
        num_landmarks=int(args.num_landmarks),
        boost_strength=float(args.boost_strength),
        deboost_strength=float(args.deboost_strength),
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "solver_feedback.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    _write_json(output_dir / "correspondence_targets.json", payload["correspondence_targets"])
    _write_json(output_dir / "correspondence_supervision.json", supervision)
    metrics = {
        "schema_version": "ulfloc_correspondence_feedback_metrics_v1",
        **payload["metadata"],
    }
    split_audit = {
        "schema_version": "ulfloc_correspondence_feedback_split_audit_v1",
        "split_name": split_name,
        "test_split_used": False,
        "source_feedback_bank": str(args.feedback_bank),
    }
    run_manifest = {
        "schema_version": "ulfloc_correspondence_feedback_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "feedback_bank": str(args.feedback_bank),
        "output_dir": str(output_dir),
        "num_landmarks": int(args.num_landmarks),
        "source_manifest": manifest,
        "paper_safe_role": "train_selfmap_correspondence_supervision",
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", run_manifest)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path.cwd()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

