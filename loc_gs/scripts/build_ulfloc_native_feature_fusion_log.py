#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"git status unavailable: {exc}\n"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _config_template(input_log_dir: Path, *, log_name: str, model_path: Path, fusion_alpha: float, images: str) -> dict[str, Any]:
    config_path = input_log_dir / "config.yaml"
    if not config_path.exists():
        configs = sorted(input_log_dir.glob("*.yaml"))
        config_path = configs[0] if configs else config_path
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        config = {}
    sample = dict(config.get("sample", {}))
    sample.setdefault("kpts_feature_file_name", "keypoints_features.pkl")
    sample.setdefault("landmark_file_name", "keypoints_sampled_idx.pkl")
    sample.setdefault("fusion_batch_size", 1)
    config["sample"] = sample
    config["feature_type"] = str(config.get("feature_type", "sp"))
    config["longest_edge"] = int(config.get("longest_edge", 640))
    config["log_name"] = str(log_name)
    config["model_path"] = str(model_path)
    config["images"] = str(images)
    config["solver_feedback"] = {
        **dict(config.get("solver_feedback", {})),
        "enabled": True,
        "artifact_path": "solver_feedback.pkl",
        "sampling_alpha": 0.0,
        "fusion_alpha": float(fusion_alpha),
        "alpha": float(fusion_alpha),
    }
    return config


def _prepare_output(
    *,
    input_log_dir: Path,
    solver_feedback: Path,
    output_log_dir: Path,
    overwrite: bool,
) -> torch.Tensor:
    if output_log_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output_log_dir already exists: {output_log_dir}")
        shutil.rmtree(output_log_dir)
    shutil.copytree(input_log_dir, output_log_dir)
    sampled_idx = torch.as_tensor(_load_pickle(input_log_dir / "keypoints_sampled_idx.pkl"), dtype=torch.long).reshape(-1).cpu()
    _dump_pickle(output_log_dir / "keypoints_sampled_idx.pkl", sampled_idx)
    shutil.copy2(solver_feedback, output_log_dir / "solver_feedback.pkl")
    return sampled_idx


def _run_ulfloc_native_feature_fusion(
    *,
    ulf_root: Path,
    model_path: Path,
    source_path: Path,
    images: str,
    output_log_dir: Path,
    sampled_idx: torch.Tensor,
    config: Mapping[str, Any],
) -> None:
    sys.path.insert(0, str(ulf_root))
    cwd = Path.cwd()
    try:
        import os
        from argparse import Namespace

        from encoders.feature_extractor import FeatureExtractor  # type: ignore
        from scene import Scene  # type: ignore
        from scene.gaussian_model import GaussianModel  # type: ignore
        from utils.gsfeature_fusion import feature_fusion  # type: ignore

        os.chdir(str(ulf_root))
        dataset = Namespace(
            sh_degree=3,
            source_path=str(source_path),
            feature_type=str(config.get("feature_type", "sp")),
            gaussian_type="3dgs",
            model_path=str(model_path),
            images=str(images),
            resolution=-1,
            white_background=True,
            longest_edge=int(config.get("longest_edge", 640)),
            data_device="cuda",
            eval=False,
            speedup=False,
            norm_before_render=True,
            render_items=["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
        )
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=-1)
        masks = None
        mask_path = Path(source_path) / images / "masks.pkl"
        if mask_path.exists():
            masks = _load_pickle(mask_path)
        extractor = FeatureExtractor(str(config.get("feature_type", "sp"))).cuda().eval()
        sample_idx_cuda = sampled_idx.to(device="cuda", dtype=torch.long)
        feature_fusion(extractor, scene, gaussians, sample_idx_cuda, masks, str(output_log_dir), dict(config))
    finally:
        Path.chdir(cwd) if hasattr(Path, "chdir") else None
        import os

        os.chdir(str(cwd))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-run ULF native train-view feature fusion with solver feedback.")
    parser.add_argument("--ulf_root", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--input_log_dir", required=True, type=Path)
    parser.add_argument("--solver_feedback", required=True, type=Path)
    parser.add_argument("--output_log_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--fusion_alpha", type=float, default=0.5)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    command = list(sys.argv if argv is None else [sys.argv[0], *argv])
    repo = Path(__file__).resolve().parents[2]
    input_log_dir = Path(args.input_log_dir)
    output_log_dir = Path(args.output_log_dir)
    if not (input_log_dir / "keypoints_sampled_idx.pkl").exists():
        raise FileNotFoundError(f"input_log_dir has no keypoints_sampled_idx.pkl: {input_log_dir}")
    sampled_idx = _prepare_output(
        input_log_dir=input_log_dir,
        solver_feedback=Path(args.solver_feedback),
        output_log_dir=output_log_dir,
        overwrite=bool(args.overwrite),
    )
    config = _config_template(
        input_log_dir,
        log_name=output_log_dir.name,
        model_path=Path(args.model_path),
        fusion_alpha=float(args.fusion_alpha),
        images=str(args.images),
    )
    (output_log_dir / "config.yaml").write_text(yaml.safe_dump(_json_safe(config), sort_keys=True), encoding="utf-8")
    if not bool(args.dry_run):
        _run_ulfloc_native_feature_fusion(
            ulf_root=Path(args.ulf_root),
            model_path=Path(args.model_path),
            source_path=Path(args.source_path),
            images=str(args.images),
            output_log_dir=output_log_dir,
            sampled_idx=sampled_idx,
            config=config,
        )
    metrics = {
        "schema_version": "ulfloc_native_feature_fusion_log_metrics_v1",
        "scene": str(args.scene),
        "sampled_count": int(sampled_idx.numel()),
        "native_ulfloc_feature_fusion_used": not bool(args.dry_run),
        "dry_run": bool(args.dry_run),
        "fusion_alpha": float(args.fusion_alpha),
        "same_sampled_idx": True,
        "post_hoc_descriptor_replacement": False,
        "solver_feedback": str(Path(args.solver_feedback).resolve()),
        "keypoints_features_exists": bool((output_log_dir / "keypoints_features.pkl").exists()),
    }
    split_audit = {
        "schema_version": "ulfloc_native_feature_fusion_log_split_audit_v1",
        "test_split_used": False,
        "official_test_used": False,
        "role": "fixed_sampled_idx_native_train_view_feature_fusion",
    }
    manifest = {
        "schema_version": "ulfloc_native_feature_fusion_log_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "command": command,
        "scene": str(args.scene),
        "ulf_root": str(Path(args.ulf_root).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "source_path": str(Path(args.source_path).resolve()),
        "input_log_dir": str(input_log_dir.resolve()),
        "output_log_dir": str(output_log_dir.resolve()),
        "solver_feedback": str(Path(args.solver_feedback).resolve()),
        "metrics": metrics,
        "split_audit": split_audit,
    }
    (output_log_dir / "metrics_summary.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_log_dir / "split_audit.json").write_text(
        json.dumps(_json_safe(split_audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_log_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_log_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n",
        encoding="utf-8",
    )
    (output_log_dir / "git_status.txt").write_text(_git_status(repo), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

