#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from loc_gs.scripts.launch_cambridge_reliability_recipe import DEFAULT_SCENES, parse_csv
from loc_gs.scripts.launch_dim_matcher_experiments import query_gpus, select_idle_gpus


def build_calibration_command(
    *,
    gpu_id: int,
    scene: str,
    checkpoint: str,
    output_path: str,
    max_views: int = 256,
    rendered_rehearsal_views: int = 256,
    max_landmarks: int = 16384,
    topk: int = 8,
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        sys.executable,
        "-m",
        "loc_gs.scripts.calibrate_landmark_matchability",
        "--checkpoint",
        checkpoint,
        "--scene",
        scene,
        "--output_path",
        output_path,
        "--max_views",
        str(int(max_views)),
        "--max_landmarks",
        str(int(max_landmarks)),
        "--topk",
        str(int(topk)),
        "--query_detector",
        "stdloc",
        "--query_feature_source",
        "original",
        "--descriptor_source",
        "ply_loc",
        "--rendered_rehearsal_views",
        str(int(rendered_rehearsal_views)),
        "--rendered_query_source",
        "rendered_rgb_teacher",
        "--rendered_rehearsal_pose_mode",
        "mixed",
        "--rendered_rehearsal_interpolation_min",
        "-0.15",
        "--rendered_rehearsal_interpolation_max",
        "1.15",
        "--rendered_pose_noise_trans_m",
        "0.35",
        "--rendered_pose_noise_rot_deg",
        "20.0",
        "--rendered_pair_jitter_trans_m",
        "0.08",
        "--rendered_pair_jitter_rot_deg",
        "5.0",
        "--rendered_view_pair_min_overlap",
        "0.15",
        "--locability_prior_weight",
        "0.05",
        "--device",
        "cuda:0",
    ]
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
    parser = argparse.ArgumentParser(description="Launch query-like Cambridge matchability calibration across GPUs.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--checkpoint_root", default="output/stdloc_hybrid")
    parser.add_argument("--checkpoint_tag", default="reliability_recipe")
    parser.add_argument("--checkpoint_name", default="latest.pth")
    parser.add_argument("--checkpoint_template", default="")
    parser.add_argument("--output_root", default="output/stdloc_hybrid/query_like_matchability_20260512")
    parser.add_argument("--max_views", type=int, default=256)
    parser.add_argument("--rendered_rehearsal_views", type=int, default=256)
    parser.add_argument("--max_landmarks", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max_memory_used_mb", type=int, default=1000)
    parser.add_argument("--max_utilization", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _checkpoint_path(args: argparse.Namespace, scene: str) -> Path:
    if args.checkpoint_template:
        return Path(args.checkpoint_template.format(scene=scene, tag=args.checkpoint_tag))
    return Path(args.checkpoint_root) / f"{scene}_{args.checkpoint_tag}" / args.checkpoint_name


def main() -> None:
    args = build_argparser().parse_args()
    scenes = parse_csv(args.scenes) or list(DEFAULT_SCENES)
    gpus = _gpu_ids(args)
    pending = list(scenes)
    running: list[tuple[subprocess.Popen, object, str, int]] = []
    output_root = Path(args.output_root)

    def launch(scene: str, gpu_id: int) -> None:
        checkpoint = _checkpoint_path(args, scene)
        output_dir = output_root / scene
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "stdloc_bank_query_like.pt"
        if not args.dry_run and not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found for {scene}: {checkpoint}")
        cmd, env = build_calibration_command(
            gpu_id=gpu_id,
            scene=scene,
            checkpoint=str(checkpoint),
            output_path=str(output_path),
            max_views=args.max_views,
            rendered_rehearsal_views=args.rendered_rehearsal_views,
            max_landmarks=args.max_landmarks,
            topk=args.topk,
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
                raise RuntimeError(f"calibration failed for {scene} on GPU {gpu_id} with code {proc.returncode}")
            freed.append(gpu_id)
        running = next_running
        for gpu_id in freed:
            if pending:
                launch(pending.pop(0), gpu_id)
        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
