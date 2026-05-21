#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from loc_gs.diagnostics.rendered_reliability import rendered_support_mask


def _load_tensor(path: str | Path) -> torch.Tensor:
    payload: Any = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("tensor", "values", "data", "support", "weights", "ids"):
            if key in payload:
                return torch.as_tensor(payload[key]).cpu()
        raise KeyError(f"{path} does not contain a tensor-like key")
    return torch.as_tensor(payload).cpu()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a rendered alpha/LSF reliability cache from contribution weights.")
    parser.add_argument("--contributor_ids", required=True)
    parser.add_argument("--contribution_weights", required=True)
    parser.add_argument("--gaussian_support", required=True)
    parser.add_argument("--depth_stability", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    return parser


def _tensor_mean(tensor: torch.Tensor) -> float:
    return float(tensor.float().mean().item()) if tensor.numel() else 0.0


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    depth = _load_tensor(args.depth_stability) if args.depth_stability else None
    payload = rendered_support_mask(
        contributor_ids=_load_tensor(args.contributor_ids),
        contribution_weights=_load_tensor(args.contribution_weights),
        gaussian_support=_load_tensor(args.gaussian_support),
        depth_stability=depth,
        threshold=float(args.threshold),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "rendered_support_mask.pt")
    summary = {
        "ray_count": int(payload["reliability"].numel()),
        "keep_count": int(payload["keep"].sum().item()),
        "keep_fraction": _tensor_mean(payload["keep"].float()),
        "support_mean": _tensor_mean(payload["support"]),
        "dominance_mean": _tensor_mean(payload["dominance"]),
        "entropy_mean": _tensor_mean(payload["entropy"]),
        "reliability_mean": _tensor_mean(payload["reliability"]),
        "threshold": float(args.threshold),
        **payload["metadata"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()

