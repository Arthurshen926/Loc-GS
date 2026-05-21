from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loc_gs.eval.stage_transitions import dense_transition_labels, stage_transition_summary


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"missing results.json under {run_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected {path} to contain a list")
    return [row for row in payload if isinstance(row, dict)]


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Stage Transition Report",
        "",
        f"- run: `{payload['run_dir']}`",
        f"- run_name: `{payload['run_name']}`",
        f"- git_commit: `{payload.get('git_commit')}`",
        "",
        "| run | queries | dense_improved | dense_worsened | r5_rescued | r5_lost | r2_rescued | r2_lost | mean_delta_te_cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {payload['run_name']} | {summary['query_count']} | {summary['improved_count']} | "
            f"{summary['worsened_count']} | {summary['r5_rescued_count']} | {summary['r5_lost_count']} | "
            f"{summary['r2_rescued_count']} | {summary['r2_lost_count']} | "
            f"{summary['mean_dense_minus_sparse_te_cm']:.4f} |"
        ),
        "",
        "## Worst Dense Regressions",
        "",
        "| query | sparse_te_cm | dense_te_cm | delta_te_cm |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["worst_dense_worsened_queries"]:
        lines.append(
            f"| {row['query_id']} | {row['sparse_te_cm']:.4f} | {row['dense_te_cm']:.4f} | "
            f"{row['dense_minus_sparse_te_cm']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sparse-to-dense transition behavior for one STDLoc run.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--te_epsilon_cm", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--output_labels_json", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_name = args.run_name or run_dir.name
    rows = _load_results(run_dir)
    payload = {
        "run_dir": str(run_dir),
        "run_name": str(run_name),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "summary": stage_transition_summary(rows, te_epsilon_cm=args.te_epsilon_cm, top_k=args.top_k),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown(payload), encoding="utf-8")
    if args.output_labels_json:
        output_labels = Path(args.output_labels_json)
        output_labels.parent.mkdir(parents=True, exist_ok=True)
        labels_payload = {
            "label_type": "dense_transition_labels_v1",
            "run_dir": str(run_dir),
            "run_name": str(run_name),
            "timestamp_utc": payload["timestamp_utc"],
            "git_commit": payload.get("git_commit"),
            "labels": dense_transition_labels(rows, te_epsilon_cm=args.te_epsilon_cm),
        }
        output_labels.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": args.output_md,
                "output_labels_json": args.output_labels_json,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
