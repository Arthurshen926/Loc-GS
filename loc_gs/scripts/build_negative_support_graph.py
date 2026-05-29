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

from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import load_feedback_bank
from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.stdloc_native.negative_support_memory import build_negative_support_graph


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _command_from_argv() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv) if sys.argv else ""


def _json_edge_weights(edge_weights: dict[tuple[int, int], float]) -> dict[str, float]:
    return {f"{int(a)}:{int(b)}": float(weight) for (a, b), weight in edge_weights.items()}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an audited pairwise hard-negative support graph.")
    parser.add_argument("--feedback_bank", required=True)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--output_json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    feedback_bank = Path(args.feedback_bank)
    audit = audit_feedback_bank_v2(feedback_bank)
    split_name = str(audit.get("split_name", "")).strip()
    if split_name.lower() == "test":
        raise ValueError("test split feedback banks are not allowed for negative support graph construction")
    if str(audit.get("audit_status", "unknown")) != "passed":
        raise ValueError(f"feedback bank audit failed: {audit.get('reasons', [])}")
    bank = load_feedback_bank(feedback_bank)
    manifest = dict(bank.get("manifest", {}))
    records = [dict(record) for record in bank.get("records", [])]
    graph = build_negative_support_graph(records, min_score=float(args.min_score))
    metadata = {
        "format": "loc_gs_negative_support_memory_artifact_v1",
        "scene": str(manifest.get("scene", "")),
        "split_name": str(manifest.get("split_name", split_name)),
        "source_feedback_bank": str(feedback_bank),
        "feedback_bank_schema": str(manifest.get("schema_version", "")),
        "feedback_bank_audit_status": str(audit.get("audit_status", "unknown")),
        "split_audit": {**audit, "split_name": str(audit.get("split_name", split_name) or split_name)},
        "min_score": float(args.min_score),
        "record_count": int(len(records)),
        "group_count": int(graph.get("group_count", 0)),
        "edge_count": int(graph.get("edge_count", 0)),
        "unary_count": int(len(graph.get("unary_risk", {}))),
    }
    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graph": graph, "metadata": metadata}, output_pt)
    summary = {
        **metadata,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "output_pt": str(output_pt),
    }
    _write_json(args.output_json, summary)
    audit_payload = artifact_split_audit(metadata, branch_selection=False)
    write_artifact_audit_bundle(
        output_pt.parent,
        manifest={
            "method": "loc_gs_negative_support_memory",
            "git_commit": _git_commit(),
            "timestamp_utc": summary["timestamp_utc"],
            "command": _command_from_argv(),
            "scene": metadata["scene"],
            "split_name": metadata["split_name"],
            "feedback_bank": str(feedback_bank),
            "output_pt": str(output_pt),
            "single_path_deployment": True,
            "branch_selection": False,
            "graph": {
                "edge_count": metadata["edge_count"],
                "unary_count": metadata["unary_count"],
            },
        },
        command=_command_from_argv(),
        metrics_summary={
            "record_count": metadata["record_count"],
            "group_count": metadata["group_count"],
            "edge_count": metadata["edge_count"],
            "unary_count": metadata["unary_count"],
        },
        split_audit=audit_payload,
    )
    print(json.dumps({"output_pt": str(output_pt), "edge_count": metadata["edge_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
