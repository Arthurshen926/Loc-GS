#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from loc_gs.diagnostics.solver_tuple_bank import build_solver_tuple_bank_for_selection
from loc_gs.scripts.compare_solver_tuple_banks import _load_xyz


def _load_index(path: str | Path) -> torch.Tensor:
    with Path(path).open("rb") as handle:
        value: Any = pickle.load(handle)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1).cpu()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a solver tuple diagnostic bank for one sampled landmark set.")
    parser.add_argument("--episode_cache", required=True)
    parser.add_argument("--selected_idx", required=True)
    parser.add_argument("--num_gaussians", type=int, required=True)
    parser.add_argument("--xyz_path", default="")
    parser.add_argument("--output_json", required=True)
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
    bank = build_solver_tuple_bank_for_selection(
        payload,
        selected_idx=_load_index(args.selected_idx),
        num_gaussians=int(args.num_gaussians),
        base_gaussian_id=payload.get("base_gaussian_id"),
        xyz=_load_xyz(args.xyz_path),
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
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_json": str(output)}, indent=2))


if __name__ == "__main__":
    main()

