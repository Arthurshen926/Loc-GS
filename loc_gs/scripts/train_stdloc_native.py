#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    map_dir = Path(args.map_root) / args.scene
    if not args.dry_run:
        map_dir.mkdir(parents=True, exist_ok=True)
    cfg = StdlocTrainConfig(
        scene=args.scene,
        data_root=Path(args.data_root),
        map_root=Path(args.map_root),
        repo_root=repo_root(),
        python_bin=args.python_bin or "python",
        images=args.images
        if args.no_auto_images
        else resolve_scene_images(Path(args.data_root), args.scene, args.images),
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
    run_job(job)


if __name__ == "__main__":
    main()
