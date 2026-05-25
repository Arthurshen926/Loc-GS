#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import sys
from pathlib import Path
from typing import Any

import torch

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.stdloc_native.selector_resampling import write_resampled_detector_payload
from loc_gs.stdloc_native.soft_prior import _assert_safe_output_map, _mirror_map


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _load_tensor(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("candidate_pool", "sampled_idx", "tensor", "values", "data"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise KeyError(f"{path} does not contain a tensor-like candidate pool")
    return torch.as_tensor(payload, dtype=torch.long).reshape(-1).cpu()


def _source_idx(source_map: Path) -> torch.Tensor:
    path = source_map / "detector" / "sampled_idx.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing source sampled_idx.pkl: {path}")
    return torch.as_tensor(_load_pickle(path), dtype=torch.long).reshape(-1).cpu()


def _source_scores(source_map: Path, source_idx: torch.Tensor, *, min_size: int) -> torch.Tensor:
    scores = torch.zeros((max(0, int(min_size)),), dtype=torch.float32)
    path = source_map / "detector" / "sampled_scores.pkl"
    if not path.exists():
        return scores
    payload = _load_pickle(path)
    if isinstance(payload, dict) and "score_avg" in payload:
        full = torch.as_tensor(payload["score_avg"], dtype=torch.float32).reshape(-1).cpu()
        if full.numel() >= scores.numel():
            return full.clone()
        scores[: full.numel()] = full
        return scores
    raw = (
        torch.as_tensor(payload.get("sampled_scores"), dtype=torch.float32).reshape(-1).cpu()
        if isinstance(payload, dict) and "sampled_scores" in payload
        else torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    )
    if raw.numel() == source_idx.numel():
        valid = (source_idx >= 0) & (source_idx < scores.numel())
        scores[source_idx[valid]] = raw[valid]
    return scores


def _ordered_new_candidates(
    candidate_pool: torch.Tensor,
    *,
    source_idx: torch.Tensor,
    max_candidates: int,
) -> torch.Tensor:
    source = {int(item) for item in source_idx.tolist()}
    selected: list[int] = []
    seen = set(source)
    for raw in candidate_pool.tolist():
        gid = int(raw)
        if gid < 0 or gid in seen:
            continue
        selected.append(gid)
        seen.add(gid)
        if int(max_candidates) >= 0 and len(selected) >= int(max_candidates):
            break
    return torch.tensor(selected, dtype=torch.long)


def _command_from_argv() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv) if sys.argv else ""


def _unlink_audit_files(output_map: Path) -> None:
    for name in (
        "manifest.json",
        "command.txt",
        "metrics_summary.json",
        "split_audit.json",
        "git_status.txt",
        "git_diff.patch",
        "artifact_audit.json",
    ):
        path = output_map / name
        if path.exists() or path.is_symlink():
            path.unlink()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a train-only probe map that appends LSF candidates to native source landmarks."
    )
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--candidate_pool_path", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--max_candidates", type=int, default=4096)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_map = Path(args.source_map)
    output_map = Path(args.output_map)
    _assert_safe_output_map(source_map, output_map)

    source = _source_idx(source_map)
    candidate_pool = _load_tensor(args.candidate_pool_path)
    candidates = _ordered_new_candidates(
        candidate_pool,
        source_idx=source,
        max_candidates=int(args.max_candidates),
    )
    output_idx = torch.cat([source, candidates], dim=0).long().cpu()
    max_gid = int(output_idx.max().item()) + 1 if output_idx.numel() else 0
    full_scores = _source_scores(source_map, source, min_size=max_gid).clamp(0.0, 1.0)
    if full_scores.numel() < max_gid:
        padded = torch.zeros((max_gid,), dtype=torch.float32)
        padded[: full_scores.numel()] = full_scores
        full_scores = padded
    sampled_scores = full_scores[output_idx] if output_idx.numel() else torch.empty((0,), dtype=torch.float32)

    source_manifest = _load_json(source_map / "manifest.json")
    split_name = str(source_manifest.get("split_name", source_manifest.get("split", "unknown")))
    if split_name.strip().lower() == "test":
        raise ValueError("candidate probe maps cannot be derived from a test split source map")

    manifest = {
        "method": "loc_gs_lsf_candidate_probe_map",
        "command": _command_from_argv(),
        "source_map": str(source_map),
        "output_map": str(output_map),
        "candidate_pool_path": str(args.candidate_pool_path),
        "scene": str(source_manifest.get("scene", source_map.name)),
        "split_name": split_name,
        "probe_only": True,
        "paper_facing": False,
        "single_path_deployment": True,
        "branch_selection": False,
        "same_budget": int(source.numel()) == int(output_idx.numel()),
        "source_count": int(source.numel()),
        "candidate_pool_count": int(candidate_pool.numel()),
        "candidate_added_count": int(candidates.numel()),
        "output_count": int(output_idx.numel()),
        "max_candidates": int(args.max_candidates),
        "source_manifest": source_manifest,
    }
    split_audit = artifact_split_audit(source_manifest, branch_selection=False)
    metrics_summary = {
        "source_count": int(source.numel()),
        "candidate_pool_count": int(candidate_pool.numel()),
        "candidate_added_count": int(candidates.numel()),
        "output_count": int(output_idx.numel()),
        "same_budget": bool(manifest["same_budget"]),
        "probe_only": True,
    }
    if args.dry_run:
        print(json.dumps({"manifest": manifest, "metrics_summary": metrics_summary}, indent=2, sort_keys=True))
        return 0

    _mirror_map(source_map, output_map, overwrite=bool(args.overwrite))
    write_resampled_detector_payload(
        output_map / "detector",
        {
            "sampled_idx": output_idx,
            "sampled_scores": sampled_scores,
            "score_avg": full_scores,
            "selector": full_scores,
            "source_count": int(source.numel()),
            "output_count": int(output_idx.numel()),
            "sampled_idx_changed": True,
        },
    )
    _unlink_audit_files(output_map)
    write_artifact_audit_bundle(
        output_map,
        manifest=manifest,
        command=manifest["command"],
        metrics_summary=metrics_summary,
        split_audit=split_audit,
    )
    print(
        "[lsf_candidate_probe_map] wrote "
        f"{output_map} ({source.numel()} source + {candidates.numel()} candidates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
