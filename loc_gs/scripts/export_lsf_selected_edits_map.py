#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from loc_gs.stdloc_native.selector_resampling import write_resampled_detector_payload
from loc_gs.stdloc_native.solver_aware_resampling import (
    apply_selected_local_edits,
    parse_edit_selection,
)


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _load_source_idx(source_map: Path) -> torch.Tensor:
    path = source_map / "detector" / "sampled_idx.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing source sampled_idx.pkl: {path}")
    return torch.as_tensor(_load_pickle(path), dtype=torch.long).reshape(-1).cpu()


def _load_score_payload(candidate_map: Path, source_map: Path, source_idx: torch.Tensor) -> dict[str, torch.Tensor]:
    candidate_scores = candidate_map / "detector" / "sampled_scores.pkl"
    source_scores = source_map / "detector" / "sampled_scores.pkl"
    path = candidate_scores if candidate_scores.exists() else source_scores
    if not path.exists():
        raise FileNotFoundError(f"missing sampled_scores.pkl in candidate/source map: {candidate_scores} / {source_scores}")
    payload = _load_pickle(path)
    if isinstance(payload, dict) and "score_avg" in payload:
        score_avg = torch.as_tensor(payload["score_avg"], dtype=torch.float32).reshape(-1).cpu()
        selector = torch.as_tensor(payload.get("selector", torch.zeros_like(score_avg)), dtype=torch.float32).reshape(-1).cpu()
        if selector.shape[0] != score_avg.shape[0]:
            selector = torch.zeros_like(score_avg)
        return {"score_avg": score_avg, "selector": selector}
    raw = (
        torch.as_tensor(payload.get("sampled_scores"), dtype=torch.float32).reshape(-1).cpu()
        if isinstance(payload, dict) and "sampled_scores" in payload
        else torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    )
    size = int(max(int(source_idx.max().item()) + 1 if source_idx.numel() else 0, int(raw.numel())))
    score_avg = torch.zeros(size, dtype=torch.float32)
    if raw.shape[0] == source_idx.shape[0]:
        valid = (source_idx >= 0) & (source_idx < size)
        score_avg[source_idx[valid]] = raw[valid]
    else:
        score_avg[: raw.numel()] = raw
    return {"score_avg": score_avg, "selector": torch.zeros_like(score_avg)}


def _copy_source_map(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output_map exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)


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


def _infer_candidate_map(edit_manifest: Path) -> Path:
    if edit_manifest.is_dir():
        return edit_manifest
    if edit_manifest.name == "manifest.json":
        return edit_manifest.parent
    return edit_manifest.parent


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an LSF map from a selected non-prefix subset of local edits.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--candidate_map", default="")
    parser.add_argument("--edit_manifest", required=True)
    parser.add_argument("--select_edits", required=True, help="One-based edit selection, e.g. '1-8,10,12' or 'prefix:8'.")
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--strict_conflicts", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_map = Path(args.source_map)
    edit_manifest_path = Path(args.edit_manifest)
    if edit_manifest_path.is_dir():
        edit_manifest_path = edit_manifest_path / "manifest.json"
    candidate_map = Path(args.candidate_map) if args.candidate_map else _infer_candidate_map(edit_manifest_path)
    output_map = Path(args.output_map)
    manifest_payload = json.loads(edit_manifest_path.read_text(encoding="utf-8"))
    edits = [dict(item) for item in manifest_payload.get("edits", [])]
    if not edits:
        raise ValueError(f"edit manifest has no edits: {edit_manifest_path}")
    source_idx = _load_source_idx(source_map)
    score_payload = _load_score_payload(candidate_map, source_map, source_idx)
    utility = score_payload["score_avg"]
    selected_edit_indices = parse_edit_selection(args.select_edits, total_edits=len(edits))
    selected = apply_selected_local_edits(
        source_idx=source_idx,
        edits=edits,
        utility=utility,
        selected_edit_indices=selected_edit_indices,
        strict_conflicts=bool(args.strict_conflicts),
    )
    sampled_idx = selected["sampled_idx"]
    if sampled_idx.numel() and int(sampled_idx.max().item()) >= int(utility.numel()):
        raise IndexError("selected sampled_idx exceeds score_avg length")
    payload = {
        "sampled_idx": sampled_idx,
        "sampled_scores": utility[sampled_idx],
        "score_avg": utility,
        "selector": score_payload["selector"],
        "source_count": int(source_idx.numel()),
        "output_count": int(sampled_idx.numel()),
        "sampled_idx_changed": source_idx.shape != sampled_idx.shape or not torch.equal(source_idx, sampled_idx),
    }
    manifest = {
        "method": "loc_gs_lsf_selected_edits_resampling",
        "git_commit": _git_commit(),
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "loc_gs.scripts.export_lsf_selected_edits_map", *(argv if argv is not None else sys.argv[1:])],
        "source_map": str(source_map),
        "candidate_map": str(candidate_map),
        "edit_manifest": str(edit_manifest_path),
        "output_map": str(output_map),
        "select_edits": str(args.select_edits),
        "selected_edit_indices_one_based": [int(index) + 1 for index in selected_edit_indices],
        "single_path_deployment": True,
        "branch_selection": False,
        "same_budget": int(payload["source_count"]) == int(payload["output_count"]),
        "selected_edits": selected["metadata"],
        "edits": selected["edits"],
        "skipped_edits": selected["skipped_edits"],
        "source_manifest_method": manifest_payload.get("method", ""),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    _copy_source_map(source_map, output_map, bool(args.overwrite))
    write_resampled_detector_payload(output_map / "detector", payload)
    (output_map / "lsf_selected_edits_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_map / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[lsf_selected_edits_export] wrote "
        f"{output_map} ({payload['source_count']} -> {payload['output_count']} sampled landmarks, "
        f"{selected['metadata']['applied_edit_count']} edits)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
