#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loc_gs.scripts.locgsctl import summarize_path


ROLE_VALUES = {"main_candidate", "ablation", "diagnostic", "rejected"}
SUMMARY_FILENAMES = ("metrics_summary.json", "summary.json")
MANIFEST_FIELD_GROUPS = (
    ("git_commit", ("git_commit",)),
    ("timestamp_utc", ("timestamp_utc", "timestamp")),
    ("command", ("command",)),
    ("scene", ("scene",)),
    ("split", ("split", "split_name")),
    ("checkpoint_path", ("checkpoint_path", "checkpoint")),
    ("map_path", ("map_path", "map", "baseline_map")),
    ("data_roots", ("data_roots", "data_root")),
    ("hyperparameters", ("hyperparameters",)),
    ("rho", ("rho",)),
    ("feedback_enabled", ("feedback_enabled",)),
    ("residual_enabled", ("residual_enabled",)),
    ("selector_enabled", ("selector_enabled",)),
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _canonical_summary_path(path: Path) -> Path:
    if path.name == "summary.json":
        metrics_summary = path.parent / "metrics_summary.json"
        if metrics_summary.exists():
            return metrics_summary
    return path


def _discover_summary_paths(result_roots: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen_run_dirs: set[Path] = set()
    for raw in result_roots:
        root = Path(raw)
        if root.is_file() and root.name in SUMMARY_FILENAMES:
            candidates = [_canonical_summary_path(root)]
        elif root.exists():
            candidates = []
            for filename in SUMMARY_FILENAMES:
                candidates.extend(sorted(root.rglob(filename)))
        else:
            candidates = []
        for candidate in candidates:
            run_dir = candidate.parent.resolve()
            if run_dir in seen_run_dirs:
                continue
            seen_run_dirs.add(run_dir)
            paths.append(candidate)
    return paths


def _role_from_manifest(manifest: dict[str, Any] | None) -> str | None:
    if not manifest:
        return None
    for key in ("run_role", "role", "board_role"):
        value = str(manifest.get(key, "")).strip()
        if value in ROLE_VALUES:
            return value
    if bool(manifest.get("is_ablation", False)):
        return "ablation"
    if bool(manifest.get("diagnostic", False)):
        return "diagnostic"
    return None


def _has_manifest_value(manifest: dict[str, Any], names: tuple[str, ...]) -> bool:
    for name in names:
        if name not in manifest:
            continue
        value = manifest[name]
        if isinstance(value, bool):
            return True
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if isinstance(value, (list, tuple, set)):
            if any(str(item).strip() for item in value):
                return True
            continue
        return True
    return False


def _missing_manifest_fields(manifest: dict[str, Any]) -> list[str]:
    return [
        field_name
        for field_name, aliases in MANIFEST_FIELD_GROUPS
        if not _has_manifest_value(manifest, aliases)
    ]


def _manifest_command_text(command: Any) -> str:
    if isinstance(command, str):
        return command.strip()
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command).strip()
    return ""


def _audit_bundle_consistency(
    manifest: dict[str, Any] | None,
    compact_metrics: dict[str, Any],
    *,
    run_dir: Path,
) -> list[str]:
    if manifest is None:
        return []
    reasons: list[str] = []
    manifest_scene = str(manifest.get("scene", "")).strip()
    metrics_scene = str(compact_metrics.get("scene", "")).strip()
    if manifest_scene and metrics_scene and manifest_scene != metrics_scene:
        reasons.append("manifest scene mismatch")
    manifest_command = _manifest_command_text(manifest.get("command"))
    command_path = run_dir / "command.txt"
    if command_path.exists():
        command_text = command_path.read_text(encoding="utf-8").strip()
        if not command_text:
            reasons.append("empty command.txt")
        elif manifest_command and manifest_command != command_text:
            reasons.append("manifest command mismatch")
    return reasons


def _paper_safety(
    manifest: dict[str, Any] | None,
    split_audit: dict[str, Any] | None,
    *,
    compact_metrics: dict[str, Any],
    run_dir: Path,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if manifest is None:
        reasons.append("missing manifest")
    else:
        reasons.extend(f"missing manifest field {field}" for field in _missing_manifest_fields(manifest))
        reasons.extend(_audit_bundle_consistency(manifest, compact_metrics, run_dir=run_dir))
    if split_audit is None:
        reasons.append("missing split audit")
    elif str(split_audit.get("audit_status", "unknown")) != "passed":
        reasons.append(f"split audit {split_audit.get('audit_status', 'unknown')}")
    if not (run_dir / "metrics_summary.json").exists():
        reasons.append("missing metrics_summary.json")
    if not (run_dir / "command.txt").exists():
        reasons.append("missing command.txt")
    if not (run_dir / "git_diff.patch").exists() and not (run_dir / "git_status.txt").exists():
        reasons.append("missing git diff/status")
    return (not reasons), "; ".join(reasons) if reasons else "passed"


def _classify_run(
    manifest: dict[str, Any] | None,
    split_audit: dict[str, Any] | None,
    *,
    paper_safe: bool,
) -> str:
    if split_audit is not None and str(split_audit.get("audit_status", "unknown")) == "failed":
        return "rejected"
    role = _role_from_manifest(manifest)
    if role is not None:
        return role
    return "main_candidate" if paper_safe else "diagnostic"


def _row_from_summary(summary_path: Path) -> dict[str, Any]:
    run_dir = summary_path.parent
    manifest = _load_json(run_dir / "manifest.json")
    split_audit = _load_json(run_dir / "split_audit.json")
    compact = summarize_path(summary_path)
    paper_safe, reason = _paper_safety(manifest, split_audit, compact_metrics=compact, run_dir=run_dir)
    role = _classify_run(manifest, split_audit, paper_safe=paper_safe)
    scene = str(
        compact.get(
            "scene",
            (manifest or {}).get("scene", ""),
        )
    )
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "scene": scene,
        "run_role": role,
        "paper_safe": paper_safe,
        "paper_safety_reason": reason,
        "metrics": {
            "dense": compact.get("dense", {}),
            "sparse": compact.get("sparse", {}),
        },
        "manifest_path": str(run_dir / "manifest.json") if manifest is not None else "",
        "split_audit_path": str(run_dir / "split_audit.json") if split_audit is not None else "",
    }


def build_board(result_roots: list[str]) -> dict[str, Any]:
    rows = [_row_from_summary(path) for path in _discover_summary_paths(result_roots)]
    rows.sort(key=lambda row: (row["scene"], row["run_role"], row["run_name"]))
    return {"runs": rows, "run_count": len(rows)}


def _fmt_metric(stage: dict[str, Any], key: str) -> str:
    value = stage.get(key)
    if value is None:
        return ""
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def board_to_markdown(board: dict[str, Any]) -> str:
    lines = [
        "# Experiment Board",
        "",
        "| Run | Scene | Role | Paper-safe | Median cm | Median deg | R@10 | R@5 | R@2 | Reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in board.get("runs", []):
        dense = row.get("metrics", {}).get("dense", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_name", "")),
                    str(row.get("scene", "")),
                    str(row.get("run_role", "")),
                    "yes" if row.get("paper_safe") else "no",
                    _fmt_metric(dense, "median_te_cm"),
                    _fmt_metric(dense, "median_re_deg"),
                    _fmt_metric(dense, "recall_10cm_5deg"),
                    _fmt_metric(dense, "recall_5cm_5deg"),
                    _fmt_metric(dense, "recall_2cm_2deg"),
                    str(row.get("paper_safety_reason", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Loc-GS summary.json or metrics_summary.json files into a research experiment board."
    )
    parser.add_argument("--result_roots", nargs="+", default=["output/stdloc_hybrid"])
    parser.add_argument("--output_markdown", default="")
    parser.add_argument("--output_json", default="")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = build_argparser().parse_args() if args is None else args
    board = build_board([str(root) for root in args.result_roots])
    if args.output_markdown:
        markdown_path = Path(args.output_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(board_to_markdown(board), encoding="utf-8")
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(board, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(board, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
