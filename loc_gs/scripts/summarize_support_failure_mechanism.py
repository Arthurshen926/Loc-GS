#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loc_gs.eval.support_failure_mechanism import (
    build_support_failure_mechanism_report,
    support_failure_mechanism_markdown,
)


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected {path} to contain a JSON object")
    return payload


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize support-count failure with solver, pose, and dense-transition diagnostics."
    )
    parser.add_argument("--solver_report", required=True)
    parser.add_argument("--baseline_metrics", default="")
    parser.add_argument("--candidate_metrics", default="")
    parser.add_argument("--baseline_transition", default="")
    parser.add_argument("--candidate_transition", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    solver_report = _load_json(args.solver_report)
    assert solver_report is not None
    report = build_support_failure_mechanism_report(
        solver_report,
        baseline_metrics=_load_json(args.baseline_metrics),
        candidate_metrics=_load_json(args.candidate_metrics),
        baseline_transition=_load_json(args.baseline_transition),
        candidate_transition=_load_json(args.candidate_transition),
        scene=args.scene or None,
    )
    report.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "inputs": {
                "solver_report": str(args.solver_report),
                "baseline_metrics": str(args.baseline_metrics),
                "candidate_metrics": str(args.candidate_metrics),
                "baseline_transition": str(args.baseline_transition),
                "candidate_transition": str(args.candidate_transition),
            },
        }
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(support_failure_mechanism_markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
