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

from loc_gs.feedback.query_view_validation_pruning import (
    attach_trace_attributions_to_eval_rows,
    load_eval_rows,
    load_sparse_trace_payload,
    prune_active_fusion_plan_from_sparse_validation,
)


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune active descriptor-fusion views using sparse validation traces.")
    parser.add_argument("--active_fusion_plan", required=True, type=Path)
    parser.add_argument("--baseline_run_dir", required=True, type=Path)
    parser.add_argument("--candidate_run_dir", required=True, type=Path)
    parser.add_argument("--candidate_trace_payload", default=None, type=Path)
    parser.add_argument("--max_trace_records_per_query", default=0, type=int)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--protected_te_cm", default=15.0, type=float)
    parser.add_argument("--hard_te_cm", default=20.0, type=float)
    parser.add_argument("--regression_margin_cm", default=20.0, type=float)
    parser.add_argument("--improvement_margin_cm", default=20.0, type=float)
    parser.add_argument("--min_pair_score", default=0.0, type=float)
    parser.add_argument("--min_views_per_landmark", default=0, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = str(args.split_name).strip()
    if split.lower() == "test":
        raise ValueError("test split is not allowed for query/view validation pruning")
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_plan = json.loads(Path(args.active_fusion_plan).read_text(encoding="utf-8"))
    if not isinstance(active_plan, Mapping):
        raise ValueError("active_fusion_plan must contain a JSON object")
    candidate_rows = load_eval_rows(args.candidate_run_dir)
    if args.candidate_trace_payload is not None:
        candidate_rows = attach_trace_attributions_to_eval_rows(
            candidate_rows,
            load_sparse_trace_payload(args.candidate_trace_payload),
            max_records_per_query=int(args.max_trace_records_per_query),
        )
    result = prune_active_fusion_plan_from_sparse_validation(
        active_plan=active_plan,
        baseline_rows=load_eval_rows(args.baseline_run_dir),
        candidate_rows=candidate_rows,
        protected_te_cm=float(args.protected_te_cm),
        hard_te_cm=float(args.hard_te_cm),
        regression_margin_cm=float(args.regression_margin_cm),
        improvement_margin_cm=float(args.improvement_margin_cm),
        min_pair_score=float(args.min_pair_score),
        min_views_per_landmark=int(args.min_views_per_landmark),
    )
    pruned_plan = result["active_fusion_plan"]
    metrics = {
        **dict(result["metrics"]),
        "scene": str(args.scene),
        "split_name": split,
        "active_fusion_plan": str(Path(args.active_fusion_plan)),
        "baseline_run_dir": str(Path(args.baseline_run_dir)),
        "candidate_run_dir": str(Path(args.candidate_run_dir)),
        "candidate_trace_payload": None if args.candidate_trace_payload is None else str(Path(args.candidate_trace_payload)),
        "max_trace_records_per_query": int(args.max_trace_records_per_query),
    }
    split_audit = {
        "schema_version": "query_view_validation_pruning_split_audit_v1",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
        "role": "offline_active_fusion_plan_pruning",
    }
    manifest = {
        "schema_version": "query_view_validation_pruning_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv if argv is None else [sys.executable, "-m", "loc_gs.scripts.build_query_view_validation_pruning", *argv],
        "scene": str(args.scene),
        "split_name": split,
        "active_fusion_plan": str(Path(args.active_fusion_plan)),
        "baseline_run_dir": str(Path(args.baseline_run_dir)),
        "candidate_run_dir": str(Path(args.candidate_run_dir)),
        "candidate_trace_payload": None if args.candidate_trace_payload is None else str(Path(args.candidate_trace_payload)),
        "metrics": metrics,
        "split_audit": split_audit,
        "branch_selection": False,
        "official_test_used": False,
    }
    _write_json(output_dir / "pruned_landmark_fusion_plan.json", pruned_plan)
    _write_json(output_dir / "removed_pairs.json", {"removed_pairs": result["removed_pairs"]})
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
