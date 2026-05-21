#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from loc_gs.dense_support.dense_match_filter import filter_dense_matches


def _load_tensor(path: str | Path) -> torch.Tensor:
    payload: Any = torch.load(Path(path), map_location="cpu")
    return torch.as_tensor(payload).cpu()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a dense support reliability mask to dense matches.")
    parser.add_argument("--matches", required=True)
    parser.add_argument("--reliability", required=True)
    parser.add_argument("--min_reliability", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--reweight_scores", action="store_true")
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    matches = torch.load(Path(args.matches), map_location="cpu")
    if not isinstance(matches, dict):
        raise ValueError("matches must be a dict with query_yx/reference_yx/scores")
    result = filter_dense_matches(
        query_yx=matches["query_yx"],
        reference_yx=matches["reference_yx"],
        scores=matches["scores"],
        reliability=_load_tensor(args.reliability),
        min_reliability=float(args.min_reliability),
        topk=args.topk,
        reweight_scores=bool(args.reweight_scores),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_dir / "filtered_matches.pt")
    summary = dict(result["metadata"])
    summary["kept_fraction"] = (
        float(summary["kept_count"]) / float(summary["input_count"])
        if int(summary["input_count"]) > 0
        else 0.0
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()

