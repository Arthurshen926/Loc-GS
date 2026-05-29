#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.data.cambridge_semantic_masks import resolve_cambridge_masks_path
from loc_gs.stdloc_native.commands import (
    StdlocTrainConfig,
    build_train_job,
    command_to_shell,
    resolve_scene_images,
    run_job,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iteration_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.replace(",", " ").split() if item)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the source-of-truth STDLoc trainer from Loc-GS.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--map_root", default="output/stdloc/map_cambridge_spgs")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--no_auto_images", action="store_true")
    parser.add_argument("--python_bin", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--detector_iterations", type=int, default=30000)
    parser.add_argument("--detector_folder", default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)
    parser.add_argument("--test_iterations", default="7000 30000")
    parser.add_argument("--save_iterations", default="7000 30000")
    parser.add_argument("--test_detector_iterations", default="30000")
    parser.add_argument("--save_detector_iterations", default="30000")
    parser.add_argument("--stream_cameras", action="store_true")
    parser.add_argument("--train_only_cameras", action="store_true")
    parser.add_argument(
        "--require_semantic_masks",
        action="store_true",
        help="Fail unless the effective STDLoc image root contains masks.pkl.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _git_text(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_root()),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - best-effort audit
        return f"git audit unavailable: {exc}"


def semantic_mask_audit(data_root: Path, scene: str, images: str) -> dict[str, object]:
    scene_root = Path(data_root) / scene
    effective_path = scene_root / images / "masks.pkl" if images != "." else scene_root / "masks.pkl"
    fallback_path = resolve_cambridge_masks_path(scene_root, image_subdir=images)
    return {
        "scene_root": str(scene_root),
        "images": images,
        "stdloc_expected_mask_path": str(effective_path),
        "stdloc_expected_mask_exists": effective_path.exists(),
        "resolved_mask_path": str(fallback_path) if fallback_path is not None else None,
        "resolved_mask_exists": fallback_path is not None,
        "stdloc_train_will_load_masks": effective_path.exists(),
    }


def write_train_audit_files(
    map_dir: Path,
    *,
    args: argparse.Namespace,
    command: str,
    images: str,
    mask_audit: dict[str, object],
) -> None:
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    manifest = {
        "format": "loc_gs_stdloc_native_train_manifest_v2",
        "git_commit": _git_text(["rev-parse", "HEAD"]),
        "git_status_path": str(map_dir / "git_status.txt"),
        "command": command,
        "scene": args.scene,
        "split": "train_only_cameras" if args.train_only_cameras else "all_cameras",
        "data_root": str(Path(args.data_root)),
        "map_root": str(Path(args.map_root)),
        "map_path": str(map_dir),
        "images": images,
        "semantic_mask_audit": mask_audit,
        "hyperparameters": {
            "iterations": int(args.iterations),
            "detector_iterations": int(args.detector_iterations),
            "landmark_num": int(args.landmark_num),
            "landmark_k": int(args.landmark_k),
            "train_only_cameras": bool(args.train_only_cameras),
            "stream_cameras": bool(args.stream_cameras),
        },
        "feedback_enabled": False,
        "selector_feedback_enabled": False,
        "residual_feedback_enabled": False,
        "rho_feedback_enabled": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (map_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (map_dir / "git_status.txt").write_text(_git_text(["status", "--short"]) + "\n", encoding="utf-8")


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    map_dir = Path(args.map_root) / args.scene
    if not args.dry_run:
        map_dir.mkdir(parents=True, exist_ok=True)
    images = args.images if args.no_auto_images else resolve_scene_images(Path(args.data_root), args.scene, args.images)
    mask_audit = semantic_mask_audit(Path(args.data_root), args.scene, images)
    if args.require_semantic_masks and not bool(mask_audit["stdloc_train_will_load_masks"]):
        raise FileNotFoundError(
            "STDLoc semantic masks are required but the effective image root "
            f"does not contain masks.pkl: {mask_audit['stdloc_expected_mask_path']}"
        )
    cfg = StdlocTrainConfig(
        scene=args.scene,
        data_root=Path(args.data_root),
        map_root=Path(args.map_root),
        repo_root=repo_root(),
        python_bin=args.python_bin or "python",
        images=images,
        iterations=args.iterations,
        detector_iterations=args.detector_iterations,
        detector_folder=args.detector_folder,
        landmark_num=args.landmark_num,
        landmark_k=args.landmark_k,
        test_iterations=_iteration_list(args.test_iterations),
        save_iterations=_iteration_list(args.save_iterations),
        test_detector_iterations=_iteration_list(args.test_detector_iterations),
        save_detector_iterations=_iteration_list(args.save_detector_iterations),
        stream_cameras=args.stream_cameras,
        train_only_cameras=args.train_only_cameras,
    )
    job = build_train_job(cfg).with_gpu(args.gpu)
    if args.dry_run:
        print(command_to_shell(job))
        return
    write_train_audit_files(
        map_dir,
        args=args,
        command=command_to_shell(job),
        images=images,
        mask_audit=mask_audit,
    )
    run_job(job)


if __name__ == "__main__":
    main()
