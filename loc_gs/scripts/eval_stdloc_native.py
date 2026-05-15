#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from loc_gs.stdloc_native.commands import (
    StdlocEvalConfig,
    build_eval_job,
    command_to_shell,
    resolve_scene_images,
    run_job,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the source-of-truth STDLoc evaluator from Loc-GS.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--map_root", default="output/stdloc/map_cambridge_spgs")
    parser.add_argument("--map_scene", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--cfg", default="configs/stdloc_cambridge.yaml")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--no_auto_images", action="store_true")
    parser.add_argument("--python_bin", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval_split", choices=["test", "train"], default="test")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--max_test_cameras", type=int, default=0)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    cfg = StdlocEvalConfig(
        scene=args.scene,
        map_scene=args.map_scene or None,
        data_root=Path(args.data_root),
        map_root=Path(args.map_root),
        output_dir=output_dir,
        repo_root=repo_root(),
        python_bin=args.python_bin or "python",
        cfg=args.cfg,
        images=args.images
        if args.no_auto_images
        else resolve_scene_images(Path(args.data_root), args.scene, args.images),
        eval_split=args.eval_split,
        iteration=args.iteration,
        prefix=args.prefix or None,
        max_test_cameras=args.max_test_cameras if args.max_test_cameras > 0 else None,
        test_stride=args.test_stride,
    )
    job = build_eval_job(cfg).with_gpu(args.gpu)
    if args.dry_run:
        print(command_to_shell(job))
        return
    run_job(job)


if __name__ == "__main__":
    main()
