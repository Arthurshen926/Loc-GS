#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from loc_gs.stdloc_native.query_conditioned_support import build_feedback_bank_v2_solver_constraints


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _parse_hard_query_ids(value: str) -> list[str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return [part for part in text.replace(",", " ").split() if part]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build v3 solver admissibility constraints from audited feedback_bank_v2."
    )
    parser.add_argument("--feedback_bank", required=True)
    parser.add_argument("--source_idx", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_gaussians", type=int, default=0)
    parser.add_argument("--hard_query_ids", default="")
    parser.add_argument("--hard_query_topk", type=int, default=64)
    parser.add_argument("--hard_query_mode", choices=("dense_worsen", "pose_error", "hard_negative_pressure"), default="dense_worsen")
    parser.add_argument("--group_field", choices=("image_id", "query_id"), default="image_id")
    parser.add_argument("--positive_reprojection_threshold_px", type=float, default=4.0)
    parser.add_argument("--hard_negative_score_threshold", type=float, default=0.65)
    parser.add_argument("--hard_negative_reprojection_threshold_px", type=float, default=8.0)
    parser.add_argument("--dense_delta_bad_cm", type=float, default=5.0)
    parser.add_argument("--min_candidate_positive", type=float, default=0.0)
    parser.add_argument("--min_support_delta", type=float, default=0.0)
    parser.add_argument("--min_viable_tuple_delta", type=float, default=0.0)
    parser.add_argument("--min_logdet_delta", type=float, default=0.0)
    parser.add_argument("--min_min_eigen_delta", type=float, default=0.0)
    parser.add_argument("--max_dense_worsen_delta", type=float, default=0.0)
    parser.add_argument("--max_ambiguity_delta", type=float, default=0.0)
    parser.add_argument("--cvar_alpha", type=float, default=0.2)
    parser.add_argument("--min_cvar_score", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_idx = torch.as_tensor(_load_pickle(args.source_idx), dtype=torch.long).reshape(-1).cpu()
    constraints = build_feedback_bank_v2_solver_constraints(
        args.feedback_bank,
        source_idx=source_idx,
        num_gaussians=int(args.num_gaussians) if int(args.num_gaussians) > 0 else None,
        hard_query_ids=_parse_hard_query_ids(args.hard_query_ids),
        hard_query_topk=int(args.hard_query_topk),
        hard_query_mode=str(args.hard_query_mode),
        group_field=str(args.group_field),
        positive_reprojection_threshold_px=float(args.positive_reprojection_threshold_px),
        hard_negative_score_threshold=float(args.hard_negative_score_threshold),
        hard_negative_reprojection_threshold_px=float(args.hard_negative_reprojection_threshold_px),
        dense_delta_bad_cm=float(args.dense_delta_bad_cm),
        min_candidate_positive=float(args.min_candidate_positive),
        min_support_delta=float(args.min_support_delta),
        min_viable_tuple_delta=float(args.min_viable_tuple_delta),
        min_logdet_delta=float(args.min_logdet_delta),
        min_min_eigen_delta=float(args.min_min_eigen_delta),
        max_dense_worsen_delta=float(args.max_dense_worsen_delta),
        max_ambiguity_delta=float(args.max_ambiguity_delta),
        cvar_alpha=float(args.cvar_alpha),
        min_cvar_score=float(args.min_cvar_score),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(constraints, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "hard_query_count": int(len(constraints["hard_query_ids"])),
                "candidate_gain_count": int(constraints["metadata"]["candidate_gain_count"]),
                "source_loss_count": int(constraints["metadata"]["source_loss_count"]),
                "split_name": constraints["metadata"]["split_name"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
