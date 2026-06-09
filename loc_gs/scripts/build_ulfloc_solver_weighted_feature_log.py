#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("descriptor_artifact must contain a dict")
    for key in ("descriptors", "gaussian_ids"):
        if key not in payload:
            raise KeyError(f"descriptor_artifact is missing {key}")
    return payload


def _source_feature_match_stats(
    *,
    artifact: dict[str, Any],
    source_features: torch.Tensor,
    min_cosine: float,
    required: bool,
) -> dict[str, Any]:
    base = artifact.get("base_descriptors")
    if base is None:
        if required:
            raise ValueError("descriptor artifact is missing base_descriptors for source feature audit")
        return {"required": False, "available": False}
    base_features = F.normalize(torch.as_tensor(base, dtype=torch.float32).cpu().squeeze(), p=2, dim=-1)
    source_norm = F.normalize(source_features.detach().cpu().float().squeeze(), p=2, dim=-1)
    if base_features.shape != source_norm.shape:
        raise ValueError("base_descriptors must match source keypoints_features shape")
    base_norm = torch.linalg.norm(torch.as_tensor(base, dtype=torch.float32).cpu().squeeze(), dim=-1)
    source_raw_norm = torch.linalg.norm(source_features.detach().cpu().float().squeeze(), dim=-1)
    matching_zero = (base_norm < 1e-8) & (source_raw_norm < 1e-8)
    cosine = (base_features * source_norm).sum(dim=-1).clamp(-1.0, 1.0)
    threshold = min(max(float(min_cosine), -1.0), 1.0)
    failing_mask = (cosine < threshold) & ~matching_zero
    failing = int(failing_mask.sum().item()) if cosine.numel() else 0
    stats = {
        "required": bool(required),
        "available": True,
        "min_cosine_threshold": float(threshold),
        "min_cosine": float(cosine.min().item()) if cosine.numel() else 1.0,
        "mean_cosine": float(cosine.mean().item()) if cosine.numel() else 1.0,
        "matching_zero_count": int(matching_zero.sum().item()),
        "failing_count": int(failing),
    }
    if failing > 0:
        raise ValueError(
            "descriptor artifact is not compatible with source ULF features: "
            f"{failing} rows below cosine threshold {threshold:.6f}"
        )
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a ULF-Loc sparse log dir with solver-weighted descriptors.")
    parser.add_argument("--source_log_dir", required=True, type=Path)
    parser.add_argument("--descriptor_artifact", required=True, type=Path)
    parser.add_argument("--output_log_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_feature_min_cosine", type=float, default=0.995)
    parser.add_argument("--require_source_feature_match", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    source = Path(args.source_log_dir)
    output = Path(args.output_log_dir)
    artifact_path = Path(args.descriptor_artifact)
    if not source.exists():
        raise FileNotFoundError(f"source_log_dir not found: {source}")
    feature_path = source / "keypoints_features.pkl"
    sampled_path = source / "keypoints_sampled_idx.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"source log has no keypoints_features.pkl: {source}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"source log has no keypoints_sampled_idx.pkl: {source}")

    source_features_raw = _load_pickle(feature_path)
    source_features = torch.as_tensor(source_features_raw)
    sampled_idx = torch.as_tensor(_load_pickle(sampled_path), dtype=torch.long).reshape(-1).cpu()
    artifact = _load_artifact(artifact_path)
    fused = F.normalize(torch.as_tensor(artifact["descriptors"], dtype=torch.float32).cpu().squeeze(), p=2, dim=-1)
    gaussian_ids = torch.as_tensor(artifact["gaussian_ids"], dtype=torch.long).reshape(-1).cpu()
    if fused.shape != source_features.detach().cpu().float().squeeze().shape:
        raise ValueError("artifact descriptors must match source keypoints_features shape")
    if gaussian_ids.shape != sampled_idx.shape or not torch.equal(gaussian_ids, sampled_idx):
        raise ValueError("descriptor artifact gaussian_ids must exactly match keypoints_sampled_idx.pkl")
    match_stats = _source_feature_match_stats(
        artifact=artifact,
        source_features=source_features,
        min_cosine=float(args.source_feature_min_cosine),
        required=bool(args.require_source_feature_match),
    )

    if output.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"output_log_dir already exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)

    source_norm = torch.linalg.norm(source_features.detach().float().cpu().squeeze(), dim=-1, keepdim=True).clamp_min(1e-12)
    restored = fused * source_norm
    restored = restored.to(device=source_features.device, dtype=source_features.dtype)
    _dump_pickle(output / "keypoints_features.pkl", restored)

    artifact_metadata = dict(artifact.get("metadata", {}))
    manifest = {
        "schema_version": "ulfloc_solver_weighted_feature_log_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "scene": str(args.scene),
        "source_log_dir": str(source),
        "output_log_dir": str(output),
        "descriptor_artifact": str(artifact_path),
        "artifact_metadata": artifact_metadata,
        "source_feature_match": match_stats,
        "sampled_count": int(sampled_idx.numel()),
        "descriptor_dim": int(fused.shape[-1]),
        "same_sparse_geometry": True,
        "same_sampled_idx": True,
        "sparse_only_descriptor_override": True,
        "branch_selection": False,
    }
    source_split_name = str(artifact_metadata.get("source_split_name", "unknown"))
    test_split_used = source_split_name.strip().lower() == "test"
    split_audit = {
        "schema_version": "ulfloc_solver_weighted_feature_log_split_audit_v1",
        "source_split_name": source_split_name,
        "test_split_used": bool(test_split_used),
        "official_test_used": bool(test_split_used),
        "role": "sparse_descriptor_override_export",
    }
    (output / "solver_weighted_feature_log_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "split_audit.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8")
    (output / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"output_log_dir": str(output), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
