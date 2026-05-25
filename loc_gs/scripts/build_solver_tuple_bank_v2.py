#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.diagnostics.feedback_solver_tuple_bank import compare_feedback_solver_tuple_banks


def _load_index(path: str | Path | None) -> torch.Tensor | None:
    if not path:
        return None
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1).cpu()


def _load_xyz(path: str | Path | None) -> torch.Tensor | None:
    if not path:
        return None
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("xyz", "points", "positions"):
            if key in payload:
                return torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1, 3).cpu()
        raise KeyError(f"{path} does not contain xyz/points/positions")
    return torch.as_tensor(payload, dtype=torch.float32).reshape(-1, 3).cpu()


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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    src = report["source"]["summary"]
    cand = report["candidate"]["summary"]
    delta = report["summary_delta"]
    lines = [
        f"# Feedback Bank v2 Solver Tuple Diagnostic: {report.get('scene', 'unknown')}",
        "",
        "This report uses feedback_bank_v2 real query grouping. It is a mechanism diagnostic, not an official pose metric.",
        "",
        f"- query grouping: `{src.get('query_group_mode', 'unknown')}`",
        f"- split audit: `{src.get('split_audit', {}).get('audit_status', 'unknown')}`",
        "",
        "| Set | Support count | Tuples | Viable tuples | Viable mass | Mean logdet(H) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| source | {src['support_count_sum']} | {src['tuple_count']} | {src['viable_tuple_count']} | "
            f"{src['viable_tuple_mass']:.6f} | {src['mean_logdet_H']:.6f} |"
        ),
        (
            f"| candidate | {cand['support_count_sum']} | {cand['tuple_count']} | {cand['viable_tuple_count']} | "
            f"{cand['viable_tuple_mass']:.6f} | {cand['mean_logdet_H']:.6f} |"
        ),
        "",
        "| Delta | Support count | Viable mass | Mean logdet(H) |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| candidate - source | {delta.get('support_count_sum', 0):+.0f} | "
            f"{delta.get('viable_tuple_mass', 0.0):+.6f} | {delta.get('mean_logdet_H', 0.0):+.6f} |"
        ),
        "",
        "## Query Examples",
        "",
    ]
    for item in report["source"].get("queries", [])[:5]:
        lines.append(
            f"- `{item['query_id']}`: support={item['support_count']}, "
            f"tuples={item['tuple_count']}, logdet={item['mean_logdet_H']:.6f}"
        )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build real-query solver tuple diagnostics from feedback_bank_v2.")
    parser.add_argument("--feedback_bank", required=True)
    parser.add_argument("--source_idx", default="")
    parser.add_argument("--candidate_idx", default="")
    parser.add_argument("--xyz_path", default="")
    parser.add_argument("--group_field", default="image_id")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--reprojection_threshold_px", type=float, default=8.0)
    parser.add_argument("--tuple_size", type=int, default=4)
    parser.add_argument("--max_tuples_per_query", type=int, default=512)
    parser.add_argument("--top_correspondences_per_query", type=int, default=64)
    parser.add_argument("--min_all_inlier_prob", type=float, default=0.05)
    parser.add_argument("--min_logdet_h", type=float, default=-20.0)
    parser.add_argument("--min_spread_2d", type=float, default=0.0)
    parser.add_argument("--min_spread_3d", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    report = compare_feedback_solver_tuple_banks(
        args.feedback_bank,
        source_idx=_load_index(args.source_idx),
        candidate_idx=_load_index(args.candidate_idx),
        xyz=_load_xyz(args.xyz_path),
        group_field=args.group_field,
        tuple_size=args.tuple_size,
        max_tuples_per_query=args.max_tuples_per_query,
        top_correspondences_per_query=args.top_correspondences_per_query,
        reprojection_threshold_px=args.reprojection_threshold_px,
        min_all_inlier_prob=args.min_all_inlier_prob,
        min_logdet_h=args.min_logdet_h,
        min_spread_2d=args.min_spread_2d,
        min_spread_3d=args.min_spread_3d,
        seed=args.seed,
    )
    report.update(
        {
            "scene": args.scene or report["source"]["summary"].get("scene", "unknown"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "feedback_bank": str(args.feedback_bank),
            "source_idx": str(args.source_idx),
            "candidate_idx": str(args.candidate_idx),
        }
    )
    _write_json(args.output_json, report)
    if args.output_report:
        output_report = Path(args.output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()
