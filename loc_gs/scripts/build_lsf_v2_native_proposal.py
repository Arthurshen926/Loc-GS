#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from loc_gs.diagnostics.lsf_v2_native_proposal import build_lsf_v2_native_proposal


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LSF v2 native-score proposal artifacts.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--solver_consensus_support_path", default="")
    parser.add_argument("--localization_support_field_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_pool_size", type=int, default=4096)
    parser.add_argument("--safe_support_threshold", type=float, default=0.5)
    parser.add_argument("--max_hard_negative_risk", type=float, default=0.5)
    parser.add_argument("--max_dense_worsen_risk", type=float, default=0.5)
    parser.add_argument("--min_native_rank", type=float, default=0.75)
    parser.add_argument("--score_source", choices=("auto", "detector", "point_cloud"), default="auto")
    parser.add_argument("--native_weight", type=float, default=0.75)
    parser.add_argument("--support_weight", type=float, default=0.20)
    parser.add_argument("--hard_negative_weight", type=float, default=0.35)
    parser.add_argument("--dense_worsen_weight", type=float, default=0.25)
    parser.add_argument("--support_score_mode", choices=("raw", "rank_observed"), default="raw")
    parser.add_argument("--solver_utility_weight", type=float, default=0.0)
    parser.add_argument("--pose_information_weight", type=float, default=0.0)
    parser.add_argument("--hard_query_support_weight", type=float, default=0.0)
    parser.add_argument("--visibility_weight", type=float, default=0.0)
    parser.add_argument("--ambiguity_weight", type=float, default=0.0)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = build_argparser().parse_args() if args is None else args
    summary = build_lsf_v2_native_proposal(
        source_map=args.source_map,
        output_dir=args.output_dir,
        solver_consensus_support_path=args.solver_consensus_support_path or None,
        localization_support_field_path=args.localization_support_field_path or None,
        candidate_pool_size=args.candidate_pool_size,
        safe_support_threshold=args.safe_support_threshold,
        max_hard_negative_risk=args.max_hard_negative_risk,
        max_dense_worsen_risk=args.max_dense_worsen_risk,
        min_native_rank=args.min_native_rank,
        score_source=args.score_source,
        native_weight=args.native_weight,
        support_weight=args.support_weight,
        hard_negative_weight=args.hard_negative_weight,
        dense_worsen_weight=args.dense_worsen_weight,
        support_score_mode=args.support_score_mode,
        solver_utility_weight=args.solver_utility_weight,
        pose_information_weight=args.pose_information_weight,
        hard_query_support_weight=args.hard_query_support_weight,
        visibility_weight=args.visibility_weight,
        ambiguity_weight=args.ambiguity_weight,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
