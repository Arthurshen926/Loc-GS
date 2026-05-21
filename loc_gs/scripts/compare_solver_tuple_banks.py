#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

from loc_gs.diagnostics.solver_tuple_bank import compare_selection_solver_diagnostics


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _load_index(path: str | Path) -> torch.Tensor:
    value = _load_pickle(path)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1).cpu()


def _load_xyz(path: str | Path | None) -> torch.Tensor | None:
    if not path:
        return None
    xyz_path = Path(path)
    if xyz_path.suffix.lower() == ".ply":
        vertex = PlyData.read(str(xyz_path))["vertex"].data
        xyz = np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float32),
                np.asarray(vertex["y"], dtype=np.float32),
                np.asarray(vertex["z"], dtype=np.float32),
            ],
            axis=1,
        )
        return torch.as_tensor(xyz, dtype=torch.float32)
    payload = torch.load(xyz_path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("xyz", "points", "positions"):
            if key in payload:
                return torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1, 3).cpu()
        raise KeyError(f"{xyz_path} does not contain xyz/points/positions")
    return torch.as_tensor(payload, dtype=torch.float32).reshape(-1, 3).cpu()


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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fmt_delta(value: float | int) -> str:
    if isinstance(value, int):
        return f"{value:+d}"
    return f"{float(value):+.6f}"


def _markdown_report(report: dict[str, Any]) -> str:
    src = report["source"]["summary"]
    cand = report["candidate"]["summary"]
    delta = report["summary_delta"]
    lines = [
        f"# Solver Tuple Diagnostic: {report.get('scene', 'unknown')}",
        "",
        "This report compares native/source and candidate sampled landmark sets with solver-level diagnostics.",
        "It is a mechanism diagnostic, not an official pose metric.",
        "",
        "## Audit",
        "",
        f"- split audit: `{src.get('split_audit', {}).get('audit_status', 'unknown')}`",
        f"- query grouping: `{src.get('query_group_mode', 'unknown')}`",
        f"- support count delta: `{_fmt_delta(delta.get('support_count_sum', 0))}`",
        f"- viable tuple mass delta: `{_fmt_delta(delta.get('viable_tuple_mass', 0.0))}`",
        f"- mean logdet(H) delta: `{_fmt_delta(delta.get('mean_logdet_H', 0.0))}`",
        f"- ambiguity risk delta: `{_fmt_delta(delta.get('mean_ambiguity_risk', 0.0))}`",
        "",
        "## Summary",
        "",
        "| Set | Support count | Tuples | Viable tuples | Viable mass | Mean logdet(H) | Ambiguity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| source | {src['support_count_sum']} | {src['tuple_count']} | "
            f"{src['viable_tuple_count']} | {src['viable_tuple_mass']:.6f} | "
            f"{src['mean_logdet_H']:.6f} | {src['mean_ambiguity_risk']:.6f} |"
        ),
        (
            f"| candidate | {cand['support_count_sum']} | {cand['tuple_count']} | "
            f"{cand['viable_tuple_count']} | {cand['viable_tuple_mass']:.6f} | "
            f"{cand['mean_logdet_H']:.6f} | {cand['mean_ambiguity_risk']:.6f} |"
        ),
        "",
    ]
    if str(src.get("query_group_mode", "")).startswith("synthetic_chunks"):
        lines.extend(
            [
                "## Limitation",
                "",
                "The input cache has no per-row `query_id`, so rows were grouped into synthetic chunks.",
                "Use this as an early diagnostic only; paper-facing tuple banks should store real query/image ids.",
                "",
            ]
        )
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare source/candidate sampled sets with solver tuple diagnostics.")
    parser.add_argument("--episode_cache", required=True)
    parser.add_argument("--source_idx", required=True)
    parser.add_argument("--candidate_idx", required=True)
    parser.add_argument("--num_gaussians", type=int, required=True)
    parser.add_argument("--xyz_path", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--synthetic_group_size", type=int, default=None)
    parser.add_argument("--tuple_size", type=int, default=4)
    parser.add_argument("--max_tuples_per_query", type=int, default=512)
    parser.add_argument("--top_correspondences_per_query", type=int, default=64)
    parser.add_argument("--min_all_inlier_prob", type=float, default=0.05)
    parser.add_argument("--min_logdet_h", type=float, default=-20.0)
    parser.add_argument("--min_spread_2d", type=float, default=0.0)
    parser.add_argument("--min_spread_3d", type=float, default=0.0)
    parser.add_argument("--max_ambiguity_risk", type=float, default=1.0)
    parser.add_argument("--ambiguity_cosine_threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    payload = torch.load(Path(args.episode_cache), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("episode_cache must contain a dict payload")
    base_gaussian_id = payload.get("base_gaussian_id")
    xyz = _load_xyz(args.xyz_path)
    report = compare_selection_solver_diagnostics(
        payload,
        source_idx=_load_index(args.source_idx),
        candidate_idx=_load_index(args.candidate_idx),
        num_gaussians=int(args.num_gaussians),
        base_gaussian_id=base_gaussian_id,
        xyz=xyz,
        tuple_size=int(args.tuple_size),
        max_tuples_per_query=int(args.max_tuples_per_query),
        top_correspondences_per_query=int(args.top_correspondences_per_query),
        score_threshold=float(args.score_threshold),
        reprojection_threshold_px=args.reprojection_threshold_px,
        synthetic_group_size=args.synthetic_group_size,
        min_all_inlier_prob=float(args.min_all_inlier_prob),
        min_logdet_h=float(args.min_logdet_h),
        min_spread_2d=float(args.min_spread_2d),
        min_spread_3d=float(args.min_spread_3d),
        max_ambiguity_risk=float(args.max_ambiguity_risk),
        ambiguity_cosine_threshold=float(args.ambiguity_cosine_threshold),
        seed=int(args.seed),
    )
    metadata = dict(payload.get("metadata", {}))
    report.update(
        {
            "scene": str(args.scene or metadata.get("scene", "unknown")),
            "git_commit": _git_commit(),
            "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "episode_cache": str(args.episode_cache),
            "source_idx": str(args.source_idx),
            "candidate_idx": str(args.candidate_idx),
            "xyz_path": str(args.xyz_path),
        }
    )
    _write_json(args.output_json, report)
    if args.output_report:
        output = Path(args.output_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()

