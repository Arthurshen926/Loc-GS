#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from loc_gs.reporting.artifact_audit import write_artifact_audit_bundle
from loc_gs.scripts.export_ulfloc_sparse_feedback import _git_commit, _git_status
from loc_gs.stdloc_native.sparse_set_composition_audit import audit_sparse_set_composition


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _load_pickle_tensor(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, torch.Tensor):
        return payload.reshape(-1).cpu().to(torch.long)
    return torch.as_tensor(payload, dtype=torch.long).reshape(-1).cpu()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_report(report: Mapping[str, Any]) -> str:
    protected = report.get("protected_support", {})
    validated = report.get("validated_support", {})
    risk = report.get("regression_risk", {})
    lines = [
        "# SparseSet Composition Audit",
        "",
        f"- selected_count: {report.get('selected_count', 0)}",
        f"- baseline_count: {report.get('baseline_count', 0)}",
        f"- baseline_overlap_count: {report.get('baseline_overlap_count', 0)}",
        f"- baseline_overlap_fraction: {float(report.get('baseline_overlap_fraction', 0.0)):.4f}",
        "",
        "## Validation Support",
        "",
        f"- protected landmark selected: {protected.get('landmark_selected_count', 0)} / {protected.get('landmark_count', 0)}",
        f"- protected support mass fraction: {float(protected.get('support_mass_selected_fraction', 0.0)):.4f}",
        f"- validated landmark selected: {validated.get('landmark_selected_count', 0)} / {validated.get('landmark_count', 0)}",
        f"- validated support mass fraction: {float(validated.get('support_mass_selected_fraction', 0.0)):.4f}",
        "",
        "## Regression Risk",
        "",
        f"- risky landmarks selected: {risk.get('selected_count', 0)} / {risk.get('count', 0)}",
        f"- risky mass selected fraction: {float(risk.get('selected_mass_fraction', 0.0)):.4f}",
    ]
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit selected SparseSet composition against baseline/profile evidence.")
    parser.add_argument("--selected_idx", required=True, type=Path)
    parser.add_argument("--baseline_idx", default=None, type=Path)
    parser.add_argument("--sparse_validation_profile", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    selected_idx = _load_pickle_tensor(args.selected_idx)
    baseline_idx = None if args.baseline_idx is None else _load_pickle_tensor(args.baseline_idx)
    profile: dict[str, Any] | None = None
    split_name = "unknown"
    if args.sparse_validation_profile is not None:
        profile = _load_json(args.sparse_validation_profile)
        split_name = str(profile.get("split_name", profile.get("split", "unknown")))
        if split_name.strip().lower() == "test":
            raise ValueError("test split sparse validation profile is not allowed for composition audit")

    report = audit_sparse_set_composition(
        selected_idx=selected_idx,
        baseline_idx=baseline_idx,
        sparse_validation_profile=profile,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "report.json", report)
    (output / "report.md").write_text(_markdown_report(report), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    command = " ".join(
        shlex.quote(part)
        for part in [sys.executable, "-m", "loc_gs.scripts.build_sparse_set_composition_audit", *(argv or sys.argv[1:])]
    )
    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "official_test_used": False,
        "test_split_used": False,
        "checks": {"profile_split": {"status": "passed", "split_name": split_name}},
    }
    manifest = {
        "method": "loc_gs_sparse_set_composition_audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": command,
        "scene": str(args.scene),
        "split": split_name,
        "selected_idx": str(args.selected_idx),
        "baseline_idx": None if args.baseline_idx is None else str(args.baseline_idx),
        "sparse_validation_profile": None
        if args.sparse_validation_profile is None
        else str(args.sparse_validation_profile),
        "metrics_summary": report,
        "split_audit": split_audit,
    }
    write_artifact_audit_bundle(
        output,
        manifest=manifest,
        command=command,
        metrics_summary=report,
        split_audit=split_audit,
    )
    (output / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

