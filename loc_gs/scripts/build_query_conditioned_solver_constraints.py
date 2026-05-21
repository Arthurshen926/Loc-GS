#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from loc_gs.stdloc_native.query_conditioned_support import (
    build_query_conditioned_solver_constraints,
)


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _parse_hard_query_ids(value: str) -> list[int] | None:
    if not value:
        return None
    parts = value.replace(",", " ").split()
    return [int(part) for part in parts]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build query-conditioned solver admissibility constraints from a self-map episode cache."
    )
    parser.add_argument("--episode_cache", required=True)
    parser.add_argument("--source_idx", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_gaussians", type=int, required=True)
    parser.add_argument("--hard_query_ids", default="")
    parser.add_argument("--reprojection_threshold_px", type=float, default=4.0)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--min_candidate_positive", type=float, default=0.0)
    parser.add_argument("--min_support_delta", type=float, default=0.0)
    parser.add_argument("--min_viable_tuple_delta", type=float, default=0.0)
    parser.add_argument("--min_logdet_delta", type=float, default=0.0)
    parser.add_argument("--max_ambiguity_delta", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_idx = torch.as_tensor(_load_pickle(args.source_idx), dtype=torch.long).reshape(-1).cpu()
    constraints = build_query_conditioned_solver_constraints(
        args.episode_cache,
        source_idx=source_idx,
        num_gaussians=int(args.num_gaussians),
        hard_query_ids=_parse_hard_query_ids(args.hard_query_ids),
        reprojection_threshold_px=float(args.reprojection_threshold_px),
        score_threshold=args.score_threshold,
        min_candidate_positive=float(args.min_candidate_positive),
        min_support_delta=float(args.min_support_delta),
        min_viable_tuple_delta=float(args.min_viable_tuple_delta),
        min_logdet_delta=float(args.min_logdet_delta),
        max_ambiguity_delta=float(args.max_ambiguity_delta),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(constraints, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "hard_query_count": len(constraints["hard_query_ids"]),
                "candidate_gain_count": constraints["metadata"]["candidate_gain_count"],
                "source_loss_count": constraints["metadata"]["source_loss_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
