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

from loc_gs.diagnostics.clean_render_phase0_benchmark import (
    Phase0Thresholds,
    make_phase0_splits,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _load_rows(paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for raw in paths:
        path = Path(raw)
        source = path if path.is_absolute() else REPO_ROOT / path
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"results must be a list: {source}")
        rel = str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source)
        for item in payload:
            if isinstance(item, dict):
                rows.append({**item, "_source_path": rel})
        sources.append(rel)
    return rows, sources


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "case_type",
        "phase0_split",
        "scene",
        "query_index",
        "image_name",
        "source_split",
        "paper_safe_for_tuning",
        "sparse_te_cm",
        "base_dense_te_cm",
        "sparse_conditioned_dense_te_cm",
        "delta_dense_minus_sparse_cm",
        "sparse_conditioned_delta_cm",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    type_counts = summary["type_counts"]
    split_counts = summary["split_counts"]
    lines = [
        "# Clean Render Phase 0 Benchmark",
        "",
        "Diagnostic train/self-map sample stratification for clean-render validation.",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in type_counts.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "| Phase0 Split | Count |",
        "| --- | ---: |",
        f"| train | {split_counts.get('train', 0)} |",
        f"| val | {split_counts.get('val', 0)} |",
        "",
        "Official test cases are not required for this benchmark and should not be used for tuning.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 0 sparse-to-dense clean-render benchmark splits.")
    parser.add_argument("--results", action="append", required=True, help="Path to train/self-map results.json. Repeatable.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_per_type", type=int, default=50)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--sparse_good_te_cm", type=float, default=30.0)
    parser.add_argument("--dense_bad_margin_cm", type=float, default=20.0)
    parser.add_argument("--sparse_marginal_te_cm", type=float, default=200.0)
    parser.add_argument("--dense_recovery_margin_cm", type=float, default=20.0)
    parser.add_argument("--sparse_catastrophic_te_cm", type=float, default=300.0)
    parser.add_argument("--normal_dense_te_cm", type=float, default=15.0)
    parser.add_argument("--normal_sparse_te_cm", type=float, default=50.0)
    args = parser.parse_args(argv)

    rows, sources = _load_rows(list(args.results))
    observed_splits = sorted({str(row.get("split", "unknown")) for row in rows})
    non_train_splits = [split for split in observed_splits if split != "train"]
    if non_train_splits:
        raise ValueError(
            "Phase0 benchmark requires train split inputs only; "
            f"found source splits {observed_splits}. "
            "Official test/unknown cases must not be used for tuning or hard-case mining."
        )
    thresholds = Phase0Thresholds(
        sparse_good_te_cm=float(args.sparse_good_te_cm),
        dense_bad_margin_cm=float(args.dense_bad_margin_cm),
        sparse_marginal_te_cm=float(args.sparse_marginal_te_cm),
        dense_recovery_margin_cm=float(args.dense_recovery_margin_cm),
        sparse_catastrophic_te_cm=float(args.sparse_catastrophic_te_cm),
        normal_dense_te_cm=float(args.normal_dense_te_cm),
        normal_sparse_te_cm=float(args.normal_sparse_te_cm),
    )
    result = make_phase0_splits(
        rows,
        thresholds=thresholds,
        max_per_type=int(args.max_per_type),
        val_fraction=float(args.val_fraction),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = result["cases"]
    _write_csv(output_dir / "phase0_cases.csv", cases)
    _write_csv(output_dir / "phase0_train.csv", [row for row in cases if row["phase0_split"] == "train"])
    _write_csv(output_dir / "phase0_val.csv", [row for row in cases if row["phase0_split"] == "val"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    metrics = {
        "recipe": "clean_render_phase0_benchmark",
        "diagnostic_only": True,
        "source_results": sources,
        **result["summary"],
    }
    source_splits = sorted({str(row.get("source_split", "unknown")) for row in cases})
    official_test_used = any(split == "test" for split in source_splits)
    paper_safe_for_tuning = bool(cases) and all(split == "train" for split in source_splits)
    split_name = "train/self-map" if source_splits == ["train"] else "mixed/unknown"
    split_audit = {
        "split_name": split_name,
        "source_splits": source_splits,
        "paper_safe_for_tuning": paper_safe_for_tuning,
        "official_test_used": official_test_used,
        "case_count": len(cases),
        "sources": sources,
    }
    manifest = {
        "recipe": "clean_render_phase0_benchmark",
        "diagnostic_only": True,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": _command(),
        "output_dir": str(output_dir),
        "source_results": sources,
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(status, encoding="utf-8")
    _write_report(output_dir / "report.md", result["summary"])
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(cases)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
