#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from plyfile import PlyData

from loc_gs.stdloc_native.descriptor_conflict_pruning import (
    prune_descriptor_conflicts,
    prune_pairwise_descriptor_conflicts,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(path), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _iteration_dir(model_path: Path, iteration: int) -> Path:
    if iteration != -1:
        return model_path / "point_cloud" / f"iteration_{iteration}"
    root = model_path / "point_cloud"
    iterations: list[int] = []
    for child in root.glob("iteration_*"):
        try:
            iterations.append(int(child.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not iterations:
        raise FileNotFoundError(f"no point_cloud/iteration_* directory under {model_path}")
    return root / f"iteration_{max(iterations)}"


def _replace_path_with_symlink(source: Path, target: Path, *, force: bool) -> None:
    if target.is_symlink() or target.exists():
        if not force:
            raise FileExistsError(f"target already exists; pass --force to replace: {target}")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _load_pickle_tensor(path: Path, *, dtype: torch.dtype | None = None, cpu: bool = True) -> torch.Tensor:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    tensor = torch.as_tensor(payload)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.cpu() if cpu else tensor


def _load_xyz_from_ply(path: Path) -> torch.Tensor:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    xyz = torch.stack(
        [
            torch.as_tensor(vertex["x"], dtype=torch.float32),
            torch.as_tensor(vertex["y"], dtype=torch.float32),
            torch.as_tensor(vertex["z"], dtype=torch.float32),
        ],
        dim=1,
    )
    return xyz.cpu()


def _safe_input_split_audit(input_log_dir: Path) -> dict[str, Any]:
    path = input_log_dir / "split_audit.json"
    if path.exists():
        audit = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            raise TypeError(f"split_audit.json must contain an object: {path}")
    else:
        audit = {"audit_status": "unknown", "split_name": "unknown"}
    split_name = str(audit.get("split_name", "")).lower()
    if split_name == "test" or bool(audit.get("official_test_used", False)) or bool(audit.get("test_split_used", False)):
        raise ValueError("refusing to prune descriptor conflicts from a test/official-test input log")
    return audit


def _load_protect_scores(path: Path, *, key: str) -> torch.Tensor:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"protect feedback must contain a dict: {path}")
    split_name = str(payload.get("split_name", payload.get("split", ""))).strip().lower()
    if split_name == "test":
        raise ValueError("refusing to use protect feedback exported from test split")
    if key not in payload:
        raise KeyError(f"protect key {key!r} not found in {path}")
    return torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1).cpu()


def _validate_sparse_validation_profile(profile: Mapping[str, Any]) -> None:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip().lower()
    if split_name == "test":
        raise ValueError("refusing to use sparse validation profile from test split")


def _iter_sparse_validation_support(profile: Mapping[str, Any]) -> list[tuple[int, float]]:
    _validate_sparse_validation_profile(profile)
    items: list[tuple[int, float]] = []
    for field_name in ("protected_per_query_support", "validated_per_query_support"):
        raw_field = profile.get(field_name, {})
        if not isinstance(raw_field, Mapping):
            continue
        for raw_support in raw_field.values():
            if not isinstance(raw_support, Mapping):
                continue
            for raw_gid, raw_value in raw_support.items():
                try:
                    gid = int(raw_gid)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if gid < 0 or value <= 0.0:
                    continue
                items.append((gid, value))
    return items


def source_anchor_idx_from_sparse_validation_profile(profile: Mapping[str, Any]) -> torch.Tensor:
    """Return sorted solver-validated landmark ids from a sparse PnP validation profile."""

    gids = sorted({int(gid) for gid, _value in _iter_sparse_validation_support(profile)})
    if not gids:
        raise ValueError("sparse validation profile does not contain protected or validated support")
    return torch.tensor(gids, dtype=torch.long)


def protect_scores_from_sparse_validation_profile(profile: Mapping[str, Any], *, num_gaussians: int) -> torch.Tensor:
    """Aggregate validation-profile support into per-Gaussian protection scores."""

    size = int(num_gaussians)
    if size <= 0:
        raise ValueError("num_gaussians must be positive")
    scores = torch.zeros((size,), dtype=torch.float32)
    for gid, value in _iter_sparse_validation_support(profile):
        if int(gid) >= size:
            raise ValueError(f"sparse validation profile references gid {gid} but map has {size} gaussians")
        scores[int(gid)] += float(value)
    return scores


def _load_sparse_validation_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"sparse validation profile must contain a JSON object: {path}")
    _validate_sparse_validation_profile(payload)
    return payload


def _merge_protect_scores(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None:
        return None if right is None else torch.as_tensor(right, dtype=torch.float32).reshape(-1).cpu()
    if right is None:
        return torch.as_tensor(left, dtype=torch.float32).reshape(-1).cpu()
    left_tensor = torch.as_tensor(left, dtype=torch.float32).reshape(-1).cpu()
    right_tensor = torch.as_tensor(right, dtype=torch.float32).reshape(-1).cpu()
    size = max(int(left_tensor.numel()), int(right_tensor.numel()))
    merged = torch.zeros((size,), dtype=torch.float32)
    merged[: int(left_tensor.numel())] += left_tensor
    merged[: int(right_tensor.numel())] += right_tensor
    return merged


def _write_conflict_csv(path: Path, *, sampled_idx: torch.Tensor, keep_mask: torch.Tensor, result: Any) -> None:
    lines = [
        "position,gid,keep,is_source,nearest_source_gid,nearest_source_cosine,nearest_source_distance_m,conflict_score"
    ]
    scores = result.scores
    for pos, gid in enumerate(sampled_idx.tolist()):
        lines.append(
            ",".join(
                [
                    str(pos),
                    str(int(gid)),
                    "1" if bool(keep_mask[pos].item()) else "0",
                    "1" if bool(scores.source_mask[pos].item()) else "0",
                    str(int(scores.nearest_source_gid[pos].item())),
                    f"{float(scores.nearest_source_cosine[pos].item()):.8g}",
                    f"{float(scores.nearest_source_distance_m[pos].item()):.8g}",
                    f"{float(scores.conflict_score[pos].item()):.8g}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune ULF-Loc selected landmarks whose descriptors conflict with source anchors."
    )
    parser.add_argument("--input_model_path", required=True, type=Path)
    parser.add_argument("--input_log_dir", required=True, type=Path)
    parser.add_argument("--output_model_path", required=True, type=Path)
    parser.add_argument("--conflict_mode", default="source_anchor", choices=("source_anchor", "pairwise"))
    parser.add_argument("--source_anchor_idx", default=None, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output_log_name", required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--min_cosine", default=0.96, type=float)
    parser.add_argument("--min_spatial_distance_m", default=2.0, type=float)
    parser.add_argument("--protect_feedback", default=None, type=Path)
    parser.add_argument("--protect_key", default="landmark_support")
    parser.add_argument("--sparse_validation_profile", default=None, type=Path)
    parser.add_argument("--source_anchor_from_sparse_validation_profile", action="store_true")
    parser.add_argument("--protect_sparse_validation_profile", action="store_true")
    parser.add_argument("--min_protect_score", default=0.0, type=float)
    parser.add_argument("--max_prune_fraction", default=1.0, type=float)
    parser.add_argument("--max_prune_count", default=0, type=int)
    parser.add_argument("--min_keep_count", default=0, type=int)
    parser.add_argument("--pairwise_top_k", default=1, type=int)
    parser.add_argument("--chunk_size", default=2048, type=int)
    parser.add_argument("--source_chunk_size", default=8192, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    input_model_path = Path(args.input_model_path)
    input_log_dir = Path(args.input_log_dir)
    output_model_path = Path(args.output_model_path)
    output_log_dir = output_model_path / str(args.output_log_name)
    output_log_dir.mkdir(parents=True, exist_ok=True)

    audit = _safe_input_split_audit(input_log_dir)
    config = yaml.load(Path(args.cfg).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise TypeError(f"cfg must contain a YAML object: {args.cfg}")
    landmark_file = str(config.get("sample", {}).get("landmark_file_name", "keypoints_sampled_idx.pkl"))
    feature_file = str(config.get("sample", {}).get("kpts_feature_file_name", "keypoints_features.pkl"))

    sampled_idx = _load_pickle_tensor(input_log_dir / landmark_file, dtype=torch.long).reshape(-1)
    features = _load_pickle_tensor(input_log_dir / feature_file, dtype=torch.float32, cpu=False)
    feature_output_device = features.device
    xyz = _load_xyz_from_ply(_iteration_dir(input_model_path, int(args.iteration)) / "point_cloud.ply")
    sparse_validation_profile = (
        None if args.sparse_validation_profile is None else _load_sparse_validation_profile(Path(args.sparse_validation_profile))
    )
    source_anchor_idx = (
        None if args.source_anchor_idx is None else _load_pickle_tensor(Path(args.source_anchor_idx), dtype=torch.long).reshape(-1)
    )
    if bool(args.source_anchor_from_sparse_validation_profile):
        if sparse_validation_profile is None:
            raise ValueError("--sparse_validation_profile is required with --source_anchor_from_sparse_validation_profile")
        validation_anchor_idx = source_anchor_idx_from_sparse_validation_profile(sparse_validation_profile)
        source_anchor_idx = (
            validation_anchor_idx
            if source_anchor_idx is None
            else torch.unique(torch.cat([source_anchor_idx.cpu(), validation_anchor_idx.cpu()]).to(torch.long), sorted=True)
        )
    protect_scores = (
        None
        if args.protect_feedback is None
        else _load_protect_scores(Path(args.protect_feedback), key=str(args.protect_key))
    )
    if bool(args.protect_sparse_validation_profile):
        if sparse_validation_profile is None:
            raise ValueError("--sparse_validation_profile is required with --protect_sparse_validation_profile")
        validation_protect_scores = protect_scores_from_sparse_validation_profile(
            sparse_validation_profile,
            num_gaussians=int(xyz.shape[0]),
        )
        protect_scores = _merge_protect_scores(protect_scores, validation_protect_scores)

    if str(args.conflict_mode) == "source_anchor":
        if source_anchor_idx is None:
            raise ValueError("--source_anchor_idx is required when --conflict_mode=source_anchor")
        result = prune_descriptor_conflicts(
            sampled_idx=sampled_idx,
            features=features,
            xyz=xyz,
            source_anchor_idx=source_anchor_idx,
            protect_scores=protect_scores,
            min_protect_score=float(args.min_protect_score),
            min_cosine=float(args.min_cosine),
            min_spatial_distance_m=float(args.min_spatial_distance_m),
            max_prune_fraction=float(args.max_prune_fraction),
            max_prune_count=int(args.max_prune_count),
            min_keep_count=int(args.min_keep_count),
            chunk_size=int(args.chunk_size),
            source_chunk_size=int(args.source_chunk_size),
            device=args.device,
        )
    else:
        result = prune_pairwise_descriptor_conflicts(
            sampled_idx=sampled_idx,
            features=features,
            xyz=xyz,
            protect_scores=protect_scores,
            min_protect_score=float(args.min_protect_score),
            min_cosine=float(args.min_cosine),
            min_spatial_distance_m=float(args.min_spatial_distance_m),
            max_prune_fraction=float(args.max_prune_fraction),
            max_prune_count=int(args.max_prune_count),
            min_keep_count=int(args.min_keep_count),
            top_k=int(args.pairwise_top_k),
            chunk_size=int(args.chunk_size),
            device=args.device,
        )

    _replace_path_with_symlink(input_model_path / "point_cloud", output_model_path / "point_cloud", force=bool(args.force))
    cfg_args = input_model_path / "cfg_args"
    if cfg_args.exists():
        shutil.copy2(cfg_args, output_model_path / "cfg_args")
    config["model_path"] = str(output_model_path)
    config["log_name"] = str(args.output_log_name)
    output_cfg = output_log_dir / Path(args.cfg).name
    output_cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with (output_log_dir / landmark_file).open("wb") as handle:
        pickle.dump(result.pruned_sampled_idx, handle)
    with (output_log_dir / feature_file).open("wb") as handle:
        pickle.dump(result.pruned_features.to(feature_output_device), handle)
    _write_conflict_csv(output_log_dir / "descriptor_conflicts.csv", sampled_idx=sampled_idx, keep_mask=result.keep_mask, result=result)

    split_name = str(audit.get("split_name", "unknown"))
    metrics = {
        **result.metadata,
        "sampled_path": str(output_log_dir / landmark_file),
        "feature_path": str(output_log_dir / feature_file),
        "input_log_dir": str(input_log_dir),
        "source_anchor_idx": None if args.source_anchor_idx is None else str(Path(args.source_anchor_idx)),
        "sparse_validation_profile": None if args.sparse_validation_profile is None else str(Path(args.sparse_validation_profile)),
        "source_anchor_from_sparse_validation_profile": bool(args.source_anchor_from_sparse_validation_profile),
        "protect_sparse_validation_profile": bool(args.protect_sparse_validation_profile),
        "config_path": str(output_cfg),
        "split_name": split_name,
        "conflict_mode": str(args.conflict_mode),
        "feature_output_device": str(feature_output_device),
        "protect_feedback": None if args.protect_feedback is None else str(Path(args.protect_feedback)),
        "protect_key": str(args.protect_key),
        "protect_sparse_validation_profile": bool(args.protect_sparse_validation_profile),
    }
    split_audit = {
        **audit,
        "test_split_used": False,
        "official_test_used": False,
        "notes": "Descriptor conflict pruning used only the fixed selected ULF log, source anchors, fused descriptors, and 3D positions.",
    }
    manifest = {
        "method": "ulfloc_descriptor_conflict_pruned_sparse_set",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "command": " ".join(sys.argv),
        "input_model_path": str(input_model_path),
        "input_log_dir": str(input_log_dir),
        "output_model_path": str(output_model_path),
        "output_log_name": str(args.output_log_name),
        "cfg": str(Path(args.cfg)),
        "exported_cfg": str(output_cfg),
        "source_anchor_idx": None if args.source_anchor_idx is None else str(Path(args.source_anchor_idx)),
        "sparse_validation_profile": None if args.sparse_validation_profile is None else str(Path(args.sparse_validation_profile)),
        "split_name": split_name,
        "hyperparameters": {
            "conflict_mode": str(args.conflict_mode),
            "min_cosine": float(args.min_cosine),
            "min_spatial_distance_m": float(args.min_spatial_distance_m),
            "protect_feedback": None if args.protect_feedback is None else str(Path(args.protect_feedback)),
            "protect_key": str(args.protect_key),
            "sparse_validation_profile": None if args.sparse_validation_profile is None else str(Path(args.sparse_validation_profile)),
            "source_anchor_from_sparse_validation_profile": bool(args.source_anchor_from_sparse_validation_profile),
            "protect_sparse_validation_profile": bool(args.protect_sparse_validation_profile),
            "min_protect_score": float(args.min_protect_score),
            "max_prune_fraction": float(args.max_prune_fraction),
            "max_prune_count": int(args.max_prune_count),
            "min_keep_count": int(args.min_keep_count),
            "pairwise_top_k": int(args.pairwise_top_k),
            "chunk_size": int(args.chunk_size),
            "source_chunk_size": int(args.source_chunk_size),
            "device": args.device,
        },
        "branch_selection": False,
        "single_path_deployment": True,
        "fixed_native_geometry": True,
        "metrics_summary": metrics,
        "split_audit": split_audit,
    }
    _write_json(output_log_dir / "metrics_summary.json", metrics)
    _write_json(output_log_dir / "manifest.json", manifest)
    _write_json(output_log_dir / "split_audit.json", split_audit)
    (output_log_dir / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_log_dir / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
