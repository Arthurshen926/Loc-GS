#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"trace payload must be a dict: {path}")
    return payload


def _split_name(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        raise ValueError("trace payload split_name is required")
    return split


def _reject_test_split(payload: Mapping[str, Any]) -> None:
    split = _split_name(payload).lower()
    audit = payload.get("split_audit", {})
    split_audit = audit if isinstance(audit, Mapping) else {}
    if split == "test" or bool(split_audit.get("test_split_used")) or bool(split_audit.get("official_test_used")):
        raise ValueError("refusing to build observation cache from test split")


def _load_sampled_idx(source_log_dir: Path) -> torch.Tensor:
    sampled_path = source_log_dir / "keypoints_sampled_idx.pkl"
    if not sampled_path.exists():
        raise FileNotFoundError(f"source log has no keypoints_sampled_idx.pkl: {source_log_dir}")
    return torch.as_tensor(_load_pickle(sampled_path), dtype=torch.long).reshape(-1).detach().cpu()


def _descriptor_for_record(
    *,
    row: Mapping[str, Any],
    descriptors: Mapping[str, Any],
    offsets: dict[str, int],
) -> torch.Tensor:
    query_id = str(row.get("query_id", ""))
    if not query_id:
        raise ValueError("trace record is missing query_id")
    if query_id not in descriptors:
        raise ValueError(f"query_match_descriptors is missing query {query_id}")
    table = torch.as_tensor(descriptors[query_id], dtype=torch.float32).detach().cpu()
    if table.dim() == 1:
        table = table.reshape(1, -1)
    if table.dim() != 2:
        raise ValueError("per-query match descriptors must have shape [matches, descriptor_dim]")
    index = int(offsets.get(query_id, 0))
    if index >= int(table.shape[0]):
        raise ValueError(f"not enough descriptors for query {query_id}")
    offsets[query_id] = index + 1
    return table[index].reshape(-1).contiguous()


def build_observation_cache(
    trace: Mapping[str, Any],
    *,
    sampled_idx: torch.Tensor,
    scene: str,
    source_log_dir: Path,
) -> dict[str, Any]:
    _reject_test_split(trace)
    records = trace.get("records", [])
    if not isinstance(records, list):
        raise TypeError("trace payload records must be a list")
    descriptors = trace.get("query_match_descriptors", {})
    if not isinstance(descriptors, Mapping):
        raise ValueError("trace payload must contain query_match_descriptors")

    sampled_to_row = {int(gid): int(row) for row, gid in enumerate(sampled_idx.tolist())}
    offsets: dict[str, int] = {}
    query_desc_rows: list[torch.Tensor] = []
    sampled_row_indices: list[int] = []
    gaussian_ids: list[int] = []
    row_source_view_id: list[str] = []
    query_ids: list[str] = []
    for row in records:
        if not isinstance(row, Mapping):
            continue
        gid = row.get("gaussian_id", row.get("landmark_id"))
        if gid is None:
            continue
        raw_gid = int(gid)
        sampled_row = row.get("matched_sampled_index")
        if sampled_row is None:
            sampled_row = sampled_to_row.get(raw_gid)
        if sampled_row is None:
            continue
        sampled_row = int(sampled_row)
        if sampled_row < 0 or sampled_row >= int(sampled_idx.numel()):
            continue
        descriptor = _descriptor_for_record(row=row, descriptors=descriptors, offsets=offsets)
        query_id = str(row.get("query_id", ""))
        image_id = str(row.get("image_id", query_id))
        query_desc_rows.append(descriptor)
        sampled_row_indices.append(sampled_row)
        gaussian_ids.append(raw_gid)
        row_source_view_id.append(image_id)
        query_ids.append(query_id)

    if query_desc_rows:
        query_desc = torch.stack(query_desc_rows, dim=0).contiguous()
        landmark_id = torch.tensor(sampled_row_indices, dtype=torch.long).reshape(-1, 1)
        gaussian_id = torch.tensor(gaussian_ids, dtype=torch.long).reshape(-1, 1)
        candidate_mask = torch.ones_like(landmark_id, dtype=torch.bool)
    else:
        query_desc = torch.empty((0, 0), dtype=torch.float32)
        landmark_id = torch.empty((0, 1), dtype=torch.long)
        gaussian_id = torch.empty((0, 1), dtype=torch.long)
        candidate_mask = torch.empty((0, 1), dtype=torch.bool)
    descriptor_dim = int(query_desc.shape[1]) if query_desc.numel() else 0
    return {
        "schema_version": "ulfloc_trace_observation_cache_v1",
        "scene": str(scene),
        "split_name": _split_name(trace),
        "sampled_idx": sampled_idx,
        "query_desc": query_desc,
        "landmark_id": landmark_id,
        "gaussian_id": gaussian_id,
        "candidate_mask": candidate_mask,
        "row_source_view_id": row_source_view_id,
        "row_query_id": query_ids,
        "observation_format": "packed_pair_cache",
        "observation_id_space": "sampled_row",
        "observation_descriptor_source": "per_match_query_descriptor",
        "source_log_dir": str(source_log_dir),
        "source_trace_schema_version": trace.get("schema_version", "unknown"),
        "record_count": int(len(records)),
        "observation_count": int(query_desc.shape[0]),
        "query_count": int(len(descriptors)),
        "descriptor_dim": int(descriptor_dim),
        "split_audit": dict(trace.get("split_audit", {})) if isinstance(trace.get("split_audit", {}), Mapping) else {},
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ULF feature-fusion observation cache from sparse PnP trace descriptors.")
    parser.add_argument("--trace_payload", required=True, type=Path)
    parser.add_argument("--source_log_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = _load_payload(Path(args.trace_payload))
    sampled_idx = _load_sampled_idx(Path(args.source_log_dir))
    cache = build_observation_cache(
        trace,
        sampled_idx=sampled_idx,
        scene=str(args.scene),
        source_log_dir=Path(args.source_log_dir),
    )
    cache_path = output_dir / "observation_cache.pt"
    torch.save(cache, cache_path)
    split_audit = {
        "schema_version": "ulfloc_trace_observation_cache_split_audit_v1",
        "split_name": cache["split_name"],
        "test_split_used": False,
        "official_test_used": False,
        "role": "selfmap_sparse_trace_descriptor_observation_cache",
        "source_split_audit": cache.get("split_audit", {}),
    }
    metrics = {
        "schema_version": "ulfloc_trace_observation_cache_metrics_v1",
        "scene": str(args.scene),
        "split_name": cache["split_name"],
        "sampled_count": int(sampled_idx.numel()),
        "record_count": int(cache["record_count"]),
        "observation_count": int(cache["observation_count"]),
        "query_count": int(cache["query_count"]),
        "descriptor_dim": int(cache["descriptor_dim"]),
        "observation_format": cache["observation_format"],
        "observation_descriptor_source": cache["observation_descriptor_source"],
    }
    manifest = {
        "schema_version": "ulfloc_trace_observation_cache_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv if argv is None else [sys.executable, "-m", "loc_gs.scripts.build_ulfloc_trace_observation_cache", *argv],
        "scene": str(args.scene),
        "trace_payload": str(args.trace_payload),
        "source_log_dir": str(args.source_log_dir),
        "observation_cache": str(cache_path),
        "metrics": metrics,
        "split_audit": split_audit,
        "test_split_used": False,
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"observation_cache": str(cache_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
