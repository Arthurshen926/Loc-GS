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

from loc_gs.scripts.prune_ulfloc_descriptor_conflicts import protect_scores_from_sparse_validation_profile
from loc_gs.stdloc_native.match_competition_pruning import prune_match_competition


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        raise ValueError("refusing to prune match competition from a test/official-test input log")
    return audit


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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_sampled_and_features(input_log_dir: Path, config: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, str, str, str]:
    sample_cfg = config.get("sample", {})
    if not isinstance(sample_cfg, Mapping):
        sample_cfg = {}
    landmark_file = str(sample_cfg.get("landmark_file_name", "keypoints_sampled_idx.pkl"))
    feature_file = str(sample_cfg.get("kpts_feature_file_name", "keypoints_features.pkl"))
    sampled_idx = torch.as_tensor(_load_pickle(input_log_dir / landmark_file), dtype=torch.long).reshape(-1).cpu()
    features = torch.as_tensor(_load_pickle(input_log_dir / feature_file), dtype=torch.float32)
    feature_output_device = str(features.device)
    features = features.cpu()
    if features.ndim != 2:
        features = features.reshape(int(sampled_idx.numel()), -1)
    if int(features.shape[0]) != int(sampled_idx.numel()):
        raise ValueError(f"feature rows {features.shape[0]} do not match sampled_idx length {sampled_idx.numel()}")
    return sampled_idx, features, landmark_file, feature_file, feature_output_device


def _load_scores(path: Path) -> torch.Tensor:
    payload = _load_pickle(path)
    if isinstance(payload, Mapping):
        for key in ("stealer_scores", "risk_scores", "scores"):
            if key in payload:
                payload = payload[key]
                break
    return torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()


def _load_sparse_validation_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise TypeError(f"sparse validation profile must contain an object: {path}")
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip().lower()
    if split_name == "test":
        raise ValueError("refusing to use sparse validation profile from test split")
    return profile


def _write_prune_csv(path: Path, *, sampled_idx: torch.Tensor, keep_mask: torch.Tensor, risk_scores: torch.Tensor) -> None:
    lines = ["position,gid,keep,risk_score"]
    selected_risk = risk_scores[sampled_idx]
    for pos, gid in enumerate(sampled_idx.tolist()):
        lines.append(
            f"{pos},{int(gid)},{1 if bool(keep_mask[pos].item()) else 0},{float(selected_risk[pos].item()):.8g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune ULF-Loc selected landmarks using query match competition scores.")
    parser.add_argument("--input_model_path", required=True, type=Path)
    parser.add_argument("--input_log_dir", required=True, type=Path)
    parser.add_argument("--output_model_path", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output_log_name", required=True)
    parser.add_argument("--risk_scores", required=True, type=Path)
    parser.add_argument("--sparse_validation_profile", default=None, type=Path)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--min_risk_score", default=0.0, type=float)
    parser.add_argument("--min_protect_score", default=1e-6, type=float)
    parser.add_argument("--max_prune_count", default=0, type=int)
    parser.add_argument("--min_keep_count", default=0, type=int)
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
    config = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"cfg must contain a YAML object: {args.cfg}")
    sampled_idx, features, landmark_file, feature_file, feature_output_device = _load_sampled_and_features(input_log_dir, config)
    risk_scores = _load_scores(Path(args.risk_scores))
    protect_scores = None
    if args.sparse_validation_profile is not None:
        profile = _load_sparse_validation_profile(Path(args.sparse_validation_profile))
        protect_scores = protect_scores_from_sparse_validation_profile(
            profile,
            num_gaussians=max(int(risk_scores.numel()), int(sampled_idx.max().item()) + 1),
        )

    result = prune_match_competition(
        sampled_idx=sampled_idx,
        features=features,
        risk_scores=risk_scores,
        protect_scores=protect_scores,
        min_protect_score=float(args.min_protect_score),
        min_risk_score=float(args.min_risk_score),
        max_prune_count=int(args.max_prune_count),
        min_keep_count=int(args.min_keep_count),
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
        pickle.dump(result.pruned_features.to(torch.device(feature_output_device)), handle)
    _write_prune_csv(
        output_log_dir / "match_competition_pruning.csv",
        sampled_idx=sampled_idx,
        keep_mask=result.keep_mask,
        risk_scores=risk_scores,
    )

    split_name = str(audit.get("split_name", "unknown"))
    metrics = {
        **result.metadata,
        "sampled_path": str(output_log_dir / landmark_file),
        "feature_path": str(output_log_dir / feature_file),
        "input_log_dir": str(input_log_dir),
        "risk_scores": str(Path(args.risk_scores)),
        "sparse_validation_profile": None if args.sparse_validation_profile is None else str(Path(args.sparse_validation_profile)),
        "config_path": str(output_cfg),
        "split_name": split_name,
        "feature_output_device": feature_output_device,
    }
    split_audit = {
        **audit,
        "test_split_used": False,
        "official_test_used": False,
        "notes": "Match competition pruning used train/self-map sparse validation support and query replay diagnostics.",
    }
    manifest = {
        "method": "ulfloc_match_competition_pruned_sparse_set",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "command": " ".join(sys.argv),
        "input_model_path": str(input_model_path),
        "input_log_dir": str(input_log_dir),
        "output_model_path": str(output_model_path),
        "output_log_name": str(args.output_log_name),
        "cfg": str(Path(args.cfg)),
        "exported_cfg": str(output_cfg),
        "split_name": split_name,
        "hyperparameters": {
            "risk_scores": str(Path(args.risk_scores)),
            "sparse_validation_profile": None
            if args.sparse_validation_profile is None
            else str(Path(args.sparse_validation_profile)),
            "min_risk_score": float(args.min_risk_score),
            "min_protect_score": float(args.min_protect_score),
            "max_prune_count": int(args.max_prune_count),
            "min_keep_count": int(args.min_keep_count),
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
