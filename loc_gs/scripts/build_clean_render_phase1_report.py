#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loc_gs.diagnostics.clean_render_phase1_validation import (
    summarize_phase0_action_safety,
    summarize_selected_render_health,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Clean Render Phase 1 Validation",
        "",
        "Diagnostic-only validation of clean-render action safety and selected render health.",
        "",
        "## Action Safety From Phase0 Cases",
        "",
        "| Type | Count | Non-Base Actions | Reg20 | Imp20 | Missing Delta | Median Delta cm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_type, row in metrics["action_safety"]["by_type"].items():
        lines.append(
            f"| {case_type} | {row['count']} | {row['non_base_action_count']} | "
            f"{row['regression_20cm_count']} | {row['improvement_20cm_count']} | "
            f"{row['missing_delta_count']} | "
            f"{row['median_sparse_conditioned_delta_cm']} |"
        )
    lines += [
        "",
        "## Selected Render Health From Case Analysis",
        "",
        "| Category | Count | Non-Base Selected | Median Visible | Median Near-Occluder | Median Feature Cos | Reg20 | Imp20 | Missing Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, row in metrics["selected_render_health"]["by_category"].items():
        lines.append(
            f"| {category} | {row['count']} | {row['non_base_selected_count']} | "
            f"{row['median_visible_ratio']} | {row['median_near_occluder_ratio']} | "
            f"{row['median_feature_cosine']} | {row['regression_20cm_count']} | {row['improvement_20cm_count']} | "
            f"{row['missing_delta_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 1 clean-render validation report.")
    parser.add_argument("--phase0_cases", required=True)
    parser.add_argument("--case_analysis", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    phase0_path = Path(args.phase0_cases)
    case_analysis_path = Path(args.case_analysis)
    phase0_rows = _read_csv(phase0_path)
    case_analysis_rows = _read_csv(case_analysis_path)
    action_safety = summarize_phase0_action_safety(phase0_rows)
    selected_render_health = summarize_selected_render_health(case_analysis_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    metrics = {
        "recipe": "clean_render_phase1_validation",
        "diagnostic_only": True,
        "phase0_cases": str(phase0_path),
        "case_analysis": str(case_analysis_path),
        "action_safety": action_safety,
        "selected_render_health": selected_render_health,
    }
    manifest = {
        "recipe": "clean_render_phase1_validation",
        "diagnostic_only": True,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": _command(),
        "output_dir": str(output_dir),
    }
    source_splits = sorted({str(row.get("source_split", "unknown")) for row in phase0_rows})
    split_audit = {
        "official_test_used": any(split == "test" for split in source_splits),
        "paper_safe_for_tuning": bool(phase0_rows) and all(split == "train" for split in source_splits),
        "source_splits": source_splits,
        "phase0_source": str(phase0_path),
        "case_analysis_source": str(case_analysis_path),
        "split_name": "train/self-map diagnostic" if source_splits == ["train"] else "mixed/unknown diagnostic",
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(status, encoding="utf-8")
    _write_report(output_dir / "report.md", metrics)
    print(json.dumps({"output_dir": str(output_dir), "phase0_cases": len(phase0_rows), "case_analysis_cases": len(case_analysis_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
