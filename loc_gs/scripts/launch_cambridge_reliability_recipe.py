#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from loc_gs.scripts.launch_dim_matcher_experiments import query_gpus, select_idle_gpus


DEFAULT_SCENES = ("GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_train_command(
    *,
    gpu_id: int,
    scene: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    max_frames: int = 0,
    max_train_batches: int = 0,
    image_width: int = 640,
    image_height: int = 360,
    amp: bool = True,
    superpoint_weights: str = "third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth",
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        sys.executable,
        "-m",
        "loc_gs.scripts.train_cambridge_hybrid",
        "--scene",
        scene,
        "--output_dir",
        output_dir,
        "--image_width",
        str(int(image_width)),
        "--image_height",
        str(int(image_height)),
        "--epochs",
        str(int(epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--pnp_weight",
        "0.1",
        "--pnp_pose_loss_weight",
        "0.0",
        "--pnp_reprojection_loss_weight",
        "0.0",
        "--pnp_match_loss_weight",
        "0.25",
        "--pnp_quality_loss_weight",
        "0.25",
        "--same_view_match_weight",
        "1.0",
        "--sp_recon_weight",
        "0.05",
        "--detector_recon_weight",
        "0.005",
        "--superpoint_weights",
        superpoint_weights,
        "--device",
        "cuda:0",
    ]
    if amp:
        cmd.append("--amp")
    if max_frames > 0:
        cmd.extend(["--max_frames", str(int(max_frames))])
    if max_train_batches > 0:
        cmd.extend(["--max_train_batches", str(int(max_train_batches))])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    return cmd, env


def _gpu_ids(args: argparse.Namespace) -> list[int]:
    if args.gpus:
        return [int(item) for item in parse_csv(args.gpus)]
    idle = select_idle_gpus(
        query_gpus(),
        max_memory_used_mb=args.max_memory_used_mb,
        max_utilization=args.max_utilization,
    )
    return idle or [0]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch baseline-preserving Cambridge reliability training across GPUs.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--output_root", default="output/stdloc_hybrid")
    parser.add_argument("--tag", default="reliability_recipe")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--superpoint_weights", default="third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth")
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max_memory_used_mb", type=int, default=1000)
    parser.add_argument("--max_utilization", type=int, default=10)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    scenes = parse_csv(args.scenes) or list(DEFAULT_SCENES)
    gpus = _gpu_ids(args)
    output_root = Path(args.output_root)
    if not args.dry_run and not Path(args.superpoint_weights).exists():
        raise FileNotFoundError(f"SuperPoint checkpoint not found: {args.superpoint_weights}")
    pending = list(scenes)
    running: list[tuple[subprocess.Popen, object, str, int]] = []

    def launch(scene: str, gpu_id: int) -> None:
        output_dir = output_root / f"{scene}_{args.tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd, env = build_train_command(
            gpu_id=gpu_id,
            scene=scene,
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_frames=args.max_frames,
            max_train_batches=args.max_train_batches,
            image_width=args.image_width,
            image_height=args.image_height,
            amp=not args.no_amp,
            superpoint_weights=args.superpoint_weights,
        )
        print(" ".join(cmd), f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        if args.dry_run:
            return
        log = (output_dir / "launcher.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((proc, log, scene, gpu_id))

    for gpu_id in gpus:
        if not pending:
            break
        launch(pending.pop(0), gpu_id)
    if args.dry_run:
        return

    while running:
        next_running: list[tuple[subprocess.Popen, object, str, int]] = []
        freed: list[int] = []
        for proc, log, scene, gpu_id in running:
            if proc.poll() is None:
                next_running.append((proc, log, scene, gpu_id))
                continue
            log.close()
            if proc.returncode != 0:
                raise RuntimeError(f"training failed for {scene} on GPU {gpu_id} with code {proc.returncode}")
            freed.append(gpu_id)
        running = next_running
        for gpu_id in freed:
            if pending:
                launch(pending.pop(0), gpu_id)
        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
