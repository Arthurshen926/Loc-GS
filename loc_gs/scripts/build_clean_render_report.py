#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.dense_support.clean_render_generator import select_clean_render_candidate


REPO_ROOT = Path(__file__).resolve().parents[2]


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_candidates(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    paths: tuple[tuple[str, ...], ...] = (
        ("dense", "slcdp_repair_search", "candidates"),
        ("dense", "base_slcdp", "candidates"),
        ("dense", "pgsh_slcdp", "candidates"),
        ("slcdp_repair_search", "candidates"),
        ("base_slcdp", "candidates"),
        ("pgsh_slcdp", "candidates"),
        ("candidates",),
    )
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _parse_case(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def _infer_split(label: str, source: Path, payload: Mapping[str, Any]) -> str:
    split = payload.get("split")
    if isinstance(split, str) and split.strip():
        return split.strip()
    text = f"{label} {source}".lower()
    parts = {part for part in text.replace("-", "_").replace("/", "_").split("_") if part}
    if "test" in parts:
        return "test"
    if "train" in parts:
        return "train"
    return "unknown"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Clean Render Generator Report",
        "",
        "Diagnostic-only. This report evaluates clean dense render candidates and does not select final poses.",
        "",
        "| Case | Candidates | Decision | Selected | Base Score | Selected Score | Score Gain | Anchor Safe | Gating Removed |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        selected = row.get("selected", {}) if isinstance(row.get("selected"), Mapping) else {}
        lines.append(
            "| {case} | {count} | {decision} | {label} | {base:.4f} | {score:.4f} | {gain:.4f} | {anchor} | {gating:.4f} |".format(
                case=row.get("label"),
                count=int(row.get("candidate_count", 0) or 0),
                decision=row.get("decision"),
                label=row.get("selected_label"),
                base=float(row.get("base_score", 0.0) or 0.0),
                score=float(row.get("selected_score", 0.0) or 0.0),
                gain=float(row.get("score_gain", 0.0) or 0.0),
                anchor=selected.get("anchor_safe"),
                gating=float(selected.get("gating_removed_fraction", 0.0) or 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a diagnostic clean-render candidate report.")
    parser.add_argument("--case", action="append", default=[], help="Case as label=path or path. Repeatable.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_splits: list[str] = []
    for raw in args.case:
        label, path = _parse_case(str(raw))
        source = path if path.is_absolute() else REPO_ROOT / path
        payload = _load_json(source)
        source_split = _infer_split(label, source, payload)
        source_splits.append(source_split)
        candidates = _extract_candidates(payload)
        selection = select_clean_render_candidate(candidates)
        rows.append(
            {
                "label": label,
                "source": str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source),
                "source_split": source_split,
                **selection,
            }
        )

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    metrics = {
        "recipe": "clean_render_generator_report",
        "diagnostic_only": True,
        "case_count": int(len(rows)),
        "use_clean_render_candidate_count": int(sum(row.get("decision") == "use_clean_render_candidate" for row in rows)),
        "use_native_clean_render_count": int(sum(row.get("decision") == "use_native_clean_render" for row in rows)),
        "source_splits": sorted(set(source_splits)),
        "cases": rows,
    }
    manifest = {
        "recipe": "clean_render_generator_report",
        "diagnostic_only": True,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": _command(),
        "output_dir": str(output_dir),
    }
    unique_splits = sorted(set(source_splits))
    split_audit = {
        "official_test_used": any(split == "test" for split in unique_splits),
        "paper_safe_for_tuning": bool(rows) and all(split == "train" for split in unique_splits),
        "source_splits": unique_splits,
        "split_name": "train/self-map diagnostic" if unique_splits == ["train"] else "mixed/unknown diagnostic",
        "diagnostic_only": True,
        "not_for_tuning": not (bool(rows) and all(split == "train" for split in unique_splits)),
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(status, encoding="utf-8")
    _write_report(output_dir / "report.md", rows)
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
