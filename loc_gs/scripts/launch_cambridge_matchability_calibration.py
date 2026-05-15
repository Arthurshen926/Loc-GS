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
    query_detector: str = "stdloc",
    feedback_detector_full_res: bool = False,
    descriptor_source: str = "ply_loc",
    hybrid_residual_alpha_max: float = 0.05,
    rendered_query_source: str = "rendered_rgb_teacher",
    visibility_check: str = "rendered",
    scene_match_pair_output_path: str = "",
    scene_match_pair_format: str = "pair",
    scene_match_pair_sample_limit: int = 200000,
    scene_match_pair_train_fraction: float = 1.0,
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
        str(query_detector),
        "--query_feature_source",
        "original",
        "--descriptor_source",
        str(descriptor_source),
        "--hybrid_residual_alpha_max",
        str(float(hybrid_residual_alpha_max)),
        "--rendered_rehearsal_views",
        str(int(rendered_rehearsal_views)),
        "--rendered_query_source",
        str(rendered_query_source),
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
        "--visibility_check",
        str(visibility_check),
        "--locability_prior_weight",
        "0.05",
        "--device",
        "cuda:0",
    ]
    if feedback_detector_full_res:
        cmd.append("--feedback_detector_full_res")
    if scene_match_pair_output_path:
        cmd.extend(
            [
                "--scene_match_pair_output_path",
                scene_match_pair_output_path,
                "--scene_match_pair_format",
                str(scene_match_pair_format),
                "--scene_match_pair_sample_limit",
                str(int(scene_match_pair_sample_limit)),
                "--scene_match_pair_train_fraction",
                str(float(scene_match_pair_train_fraction)),
            ]
        )
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
    parser.add_argument("--query_detector", choices=["superpoint", "stdloc", "feedback"], default="stdloc")
    parser.add_argument("--feedback_detector_full_res", action="store_true")
    parser.add_argument(
        "--descriptor_source",
        choices=["hybrid", "ply_loc", "hybrid_ply_blend", "hybrid_ply_gated_residual"],
        default="ply_loc",
    )
    parser.add_argument("--hybrid_residual_alpha_max", type=float, default=0.05)
    parser.add_argument("--rendered_query_source", choices=["rendered_rgb_teacher", "feature_field"], default="rendered_rgb_teacher")
    parser.add_argument("--visibility_check", choices=["none", "rendered"], default="rendered")
    parser.add_argument("--scene_match_pair_output_root", default="")
    parser.add_argument("--scene_match_pair_format", choices=["pair", "listwise"], default="pair")
    parser.add_argument("--scene_match_pair_sample_limit", type=int, default=200000)
    parser.add_argument("--scene_match_pair_train_fraction", type=float, default=1.0)
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
        pair_output_path = (
            Path(args.scene_match_pair_output_root) / scene / "scene_match_pairs.pt"
            if args.scene_match_pair_output_root
            else None
        )
        if pair_output_path is not None:
            pair_output_path.parent.mkdir(parents=True, exist_ok=True)
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
            query_detector=args.query_detector,
            feedback_detector_full_res=args.feedback_detector_full_res,
            descriptor_source=args.descriptor_source,
            hybrid_residual_alpha_max=args.hybrid_residual_alpha_max,
            rendered_query_source=args.rendered_query_source,
            visibility_check=args.visibility_check,
            scene_match_pair_output_path="" if pair_output_path is None else str(pair_output_path),
            scene_match_pair_format=args.scene_match_pair_format,
            scene_match_pair_sample_limit=args.scene_match_pair_sample_limit,
            scene_match_pair_train_fraction=args.scene_match_pair_train_fraction,
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
