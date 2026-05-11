#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loc_gs.scripts.merge_cambridge_eval_shards import merge_eval_shards
from loc_gs.scripts.launch_cambridge_reliability_recipe import DEFAULT_SCENES, parse_csv
from loc_gs.scripts.launch_dim_matcher_experiments import query_gpus, select_idle_gpus


EVAL_RECIPES = (
    "protected",
    "learned_blend",
    "covisibility_select",
    "covisibility_soft_select",
    "covisibility_prosac",
)


@dataclass(frozen=True)
class EvalJob:
    scene: str
    recipe: str
    checkpoint: Path
    output_dir: Path
    label: str
    query_stride: int
    query_offset: int
    group_key: str = ""
    final_output_dir: Path | None = None


def build_eval_command(
    *,
    gpu_id: int,
    scene: str,
    checkpoint: str,
    output_dir: str,
    recipe: str = "protected",
    max_queries: int = 0,
    query_stride: int = 1,
    query_offset: int = 0,
    eval_split: str = "test",
    pnp_reprojection_error: float = 8.0,
) -> tuple[list[str], dict[str, str]]:
    recipe = str(recipe)
    if recipe not in EVAL_RECIPES:
        raise ValueError(f"unsupported reliability eval recipe: {recipe}")
    descriptor_source = "hybrid_ply_blend" if recipe == "learned_blend" else "ply_loc"
    solver = "opencv_prosac" if recipe == "covisibility_prosac" else "opencv"
    cmd = [
        sys.executable,
        "-m",
        "loc_gs.scripts.eval_cambridge_hybrid",
        "--checkpoint",
        checkpoint,
        "--scene",
        scene,
        "--output_dir",
        output_dir,
        "--eval_pose_source",
        "cambridge",
        "--eval_split",
        eval_split,
        "--landmark_source",
        "stdloc_detector",
        "--descriptor_source",
        descriptor_source,
        "--query_detector",
        "stdloc",
        "--query_feature_source",
        "original",
        "--matcher",
        "stdloc_parity",
        "--dense_iters",
        "2",
        "--dense_full_render",
        "--subpixel_refine",
        "--solver",
        solver,
        "--sparse_reprojection_error",
        str(float(pnp_reprojection_error)),
        "--dense_reprojection_error",
        str(float(pnp_reprojection_error)),
        "--pnp_confidence",
        "0.99999",
        "--device",
        "cuda:0",
    ]
    if recipe == "learned_blend":
        cmd.extend(["--ply_loc_feature_weight", "0.9"])
    if recipe == "covisibility_select":
        cmd.extend(
            [
                "--max_landmarks",
                "12000",
                "--landmark_candidate_source",
                "sampled",
                "--landmark_score_mode",
                "matchability",
                "--landmark_score_detector_weight",
                "1.0",
                "--landmark_score_locability_weight",
                "0.0",
                "--landmark_score_visibility_weight",
                "0.75",
                "--landmark_score_geometry_weight",
                "0.5",
                "--landmark_score_observability_weight",
                "0.5",
                "--landmark_score_prior_blend",
                "0.75",
                "--landmark_score_legacy_keep_ratio",
                "0.5",
                "--landmark_score_spatial_grid_size",
                "8",
            ]
        )
    if recipe == "covisibility_soft_select":
        cmd.extend(
            [
                "--max_landmarks",
                "14336",
                "--landmark_candidate_source",
                "sampled",
                "--landmark_score_mode",
                "matchability",
                "--landmark_score_detector_weight",
                "1.0",
                "--landmark_score_locability_weight",
                "0.0",
                "--landmark_score_visibility_weight",
                "0.5",
                "--landmark_score_geometry_weight",
                "0.5",
                "--landmark_score_observability_weight",
                "0.25",
                "--landmark_score_ambiguity_weight",
                "0.35",
                "--landmark_score_ambiguity_radius",
                "0.5",
                "--landmark_score_prior_blend",
                "0.5",
                "--landmark_score_legacy_keep_ratio",
                "0.75",
                "--landmark_score_spatial_grid_size",
                "8",
            ]
        )
    if recipe == "covisibility_prosac":
        cmd.extend(
            [
                "--landmark_candidate_source",
                "sampled",
                "--landmark_score_mode",
                "matchability_prior",
                "--landmark_score_detector_weight",
                "1.0",
                "--landmark_score_locability_weight",
                "0.0",
                "--landmark_score_visibility_weight",
                "0.5",
                "--landmark_score_geometry_weight",
                "0.5",
                "--landmark_score_observability_weight",
                "0.25",
                "--landmark_score_ambiguity_weight",
                "0.35",
                "--landmark_score_ambiguity_radius",
                "0.5",
                "--landmark_score_prior_blend",
                "0.5",
                "--match_filter_calibrated_score_weight",
                "0.25",
                "--match_filter_margin_weight",
                "0.25",
            ]
        )
    if max_queries > 0:
        cmd.extend(["--max_queries", str(int(max_queries))])
    if query_stride != 1:
        cmd.extend(["--query_stride", str(int(query_stride))])
    if query_offset != 0:
        cmd.extend(["--query_offset", str(int(query_offset))])
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


def _checkpoint_path(args: argparse.Namespace, scene: str) -> Path:
    if args.checkpoint_template:
        return Path(args.checkpoint_template.format(scene=scene, tag=args.tag))
    return Path(args.checkpoint_root) / f"{scene}_{args.tag}" / args.checkpoint_name


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch fixed Cambridge reliability eval recipes across GPUs.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--checkpoint_root", default="output/stdloc_hybrid")
    parser.add_argument("--tag", default="reliability_recipe")
    parser.add_argument("--checkpoint_name", default="latest.pth")
    parser.add_argument("--checkpoint_template", default="")
    parser.add_argument("--output_suffix", default="eval_reliability")
    parser.add_argument("--recipes", default="protected,learned_blend")
    parser.add_argument("--eval_split", choices=["train", "test"], default="test")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--query_offset", type=int, default=0)
    parser.add_argument("--query_shards", type=int, default=1)
    parser.add_argument("--pnp_reprojection_error", type=float, default=8.0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max_memory_used_mb", type=int, default=1000)
    parser.add_argument("--max_utilization", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    scenes = parse_csv(args.scenes) or list(DEFAULT_SCENES)
    recipes = parse_csv(args.recipes) or ["protected"]
    for recipe in recipes:
        if recipe not in EVAL_RECIPES:
            raise ValueError(f"unsupported recipe {recipe}; expected one of {', '.join(EVAL_RECIPES)}")
    if args.query_shards < 1:
        raise ValueError("--query_shards must be at least 1")
    if args.query_shards > 1 and (args.query_stride != 1 or args.query_offset != 0):
        raise ValueError("--query_shards cannot be combined with --query_stride/--query_offset")
    gpus = _gpu_ids(args)
    pending: list[EvalJob] = []
    shard_groups: dict[str, dict[str, object]] = {}
    for scene in scenes:
        checkpoint = _checkpoint_path(args, scene)
        for recipe in recipes:
            final_output_dir = checkpoint.parent / f"{args.output_suffix}_{recipe}"
            if args.query_shards == 1:
                pending.append(
                    EvalJob(
                        scene=scene,
                        recipe=recipe,
                        checkpoint=checkpoint,
                        output_dir=final_output_dir,
                        label=f"{scene}/{recipe}",
                        query_stride=int(args.query_stride),
                        query_offset=int(args.query_offset),
                    )
                )
                continue
            group_key = f"{scene}/{recipe}"
            shard_dirs = []
            for offset in range(int(args.query_shards)):
                shard_dir = final_output_dir / "query_shards" / f"offset_{offset}_of_{args.query_shards}"
                shard_dirs.append(shard_dir)
                pending.append(
                    EvalJob(
                        scene=scene,
                        recipe=recipe,
                        checkpoint=checkpoint,
                        output_dir=shard_dir,
                        label=f"{scene}/{recipe}[{offset}/{args.query_shards}]",
                        query_stride=int(args.query_shards),
                        query_offset=offset,
                        group_key=group_key,
                        final_output_dir=final_output_dir,
                    )
                )
            shard_groups[group_key] = {
                "remaining": int(args.query_shards),
                "shard_dirs": shard_dirs,
                "output_dir": final_output_dir,
            }
    running: list[tuple[subprocess.Popen, object, EvalJob, int]] = []

    def launch(job: EvalJob, gpu_id: int) -> None:
        if not args.dry_run and not job.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found for {job.scene}: {job.checkpoint}")
        job.output_dir.mkdir(parents=True, exist_ok=True)
        cmd, env = build_eval_command(
            gpu_id=gpu_id,
            scene=job.scene,
            checkpoint=str(job.checkpoint),
            output_dir=str(job.output_dir),
            recipe=job.recipe,
            max_queries=args.max_queries,
            query_stride=job.query_stride,
            query_offset=job.query_offset,
            eval_split=args.eval_split,
            pnp_reprojection_error=args.pnp_reprojection_error,
        )
        print(" ".join(cmd), f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        if args.dry_run:
            return
        log = (job.output_dir / "launcher.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((proc, log, job, gpu_id))

    for gpu_id in gpus:
        if not pending:
            break
        launch(pending.pop(0), gpu_id)
    if args.dry_run:
        return

    while running:
        next_running: list[tuple[subprocess.Popen, object, EvalJob, int]] = []
        freed: list[int] = []
        for proc, log, job, gpu_id in running:
            if proc.poll() is None:
                next_running.append((proc, log, job, gpu_id))
                continue
            log.close()
            if proc.returncode != 0:
                raise RuntimeError(f"eval failed for {job.label} on GPU {gpu_id} with code {proc.returncode}")
            if job.group_key:
                group = shard_groups[job.group_key]
                group["remaining"] = int(group["remaining"]) - 1
                if int(group["remaining"]) == 0:
                    merge_eval_shards(group["shard_dirs"], group["output_dir"])
                    print(f"merged query shards for {job.group_key} -> {group['output_dir']}")
            freed.append(gpu_id)
        running = next_running
        for gpu_id in freed:
            if pending:
                launch(pending.pop(0), gpu_id)
        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
