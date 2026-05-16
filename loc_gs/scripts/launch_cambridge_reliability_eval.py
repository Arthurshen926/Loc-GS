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
    "covisibility_prosac",
    "scene_matcher_prosac",
    "scene_matcher_residual_prosac",
    "scene_matcher_prosac_magsac",
    "scene_matcher_coverage_prosac",
    "lff_feedback_prosac",
    "lff_residual_prosac",
    "oracle_prosac",
    "candidate_oracle_top1",
    "candidate_oracle_top4",
    "candidate_oracle_top8",
    "candidate_oracle_top16",
    "protected",
    "learned_blend",
    "covisibility_select",
    "covisibility_soft_select",
    "covisibility_prosac_magsac",
    "covisibility_prosac_loftr",
    "lff_residual_prosac_loftr",
)

ARCHIVED_RECIPES = {
    "protected",
    "learned_blend",
    "covisibility_select",
    "covisibility_soft_select",
    "covisibility_prosac_loftr",
    "lff_residual_prosac_loftr",
    "scene_matcher_prosac_magsac",
    "scene_matcher_coverage_prosac",
    "oracle_prosac",
    "candidate_oracle_top1",
    "candidate_oracle_top4",
    "candidate_oracle_top8",
    "candidate_oracle_top16",
}


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


def _candidate_oracle_topk(recipe: str) -> int:
    prefix = "candidate_oracle_top"
    if not str(recipe).startswith(prefix):
        return 0
    return int(str(recipe)[len(prefix) :])


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
    dense_iters: int = 2,
    calibrated_matchability_path: str = "",
    calibrated_matchability_weight: float = 0.25,
    match_calibrated_prior_weight: float = 0.0,
    locability_prior_weight: float = 0.05,
    ply_loc_feature_override: str = "",
    stdloc_detector_dir: str = "",
    scene_matcher_path: str = "",
    scene_matcher_weight: float = 0.35,
    scene_matcher_topk: int = 4,
    scene_matcher_logit_norm: str = "none",
    scene_matcher_logit_clip: float = 0.0,
    scene_matcher_listwise_dustbin: str = "score",
    scene_matcher_candidate_mode: str = "best",
    match_filter_query_score_weight: float = 0.0,
    sparse_match_filter_mode: str = "",
    sparse_match_filter_top_m: int = 0,
    hybrid_residual_alpha_max: float = 0.03,
    selfmap_reliability_path: str = "",
    selfmap_reliability_stage: str = "dense",
    selfmap_reliability_center_cm: float = 10.0,
    selfmap_reliability_temperature_cm: float = 1.0,
) -> tuple[list[str], dict[str, str]]:
    recipe = str(recipe)
    if recipe not in EVAL_RECIPES:
        raise ValueError(f"unsupported reliability eval recipe: {recipe}")
    candidate_oracle_topk = _candidate_oracle_topk(recipe)
    if recipe == "learned_blend":
        descriptor_source = "hybrid_ply_blend"
    elif recipe in {
        "lff_feedback_prosac",
        "lff_residual_prosac",
        "lff_residual_prosac_loftr",
        "scene_matcher_residual_prosac",
    }:
        descriptor_source = "hybrid_ply_gated_residual"
    else:
        descriptor_source = "ply_loc"
    if recipe in {"covisibility_prosac_magsac", "scene_matcher_prosac_magsac"}:
        solver = "opencv_prosac_magsac"
    else:
        solver = (
            "opencv_prosac"
            if recipe in {
                "covisibility_prosac",
                "scene_matcher_prosac",
                "scene_matcher_residual_prosac",
                "scene_matcher_prosac_magsac",
                "scene_matcher_coverage_prosac",
                "covisibility_prosac_loftr",
                "lff_residual_prosac_loftr",
                "lff_feedback_prosac",
                "lff_residual_prosac",
                "oracle_prosac",
                "candidate_oracle_top1",
                "candidate_oracle_top4",
                "candidate_oracle_top8",
                "candidate_oracle_top16",
            }
            else "opencv"
        )
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
        "feedback" if recipe == "lff_feedback_prosac" else "stdloc",
        "--query_feature_source",
        "original",
        "--matcher",
        "stdloc_parity",
        "--dense_iters",
        str(int(dense_iters)),
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
    if ply_loc_feature_override:
        cmd.extend(["--ply_loc_feature_override", str(ply_loc_feature_override)])
    if stdloc_detector_dir:
        cmd.extend(["--stdloc_detector_dir", str(stdloc_detector_dir)])
    if abs(float(locability_prior_weight) - 0.05) > 1e-12:
        cmd.extend(["--locability_prior_weight", str(float(locability_prior_weight))])
    if recipe in {
        "lff_feedback_prosac",
        "lff_residual_prosac",
        "lff_residual_prosac_loftr",
        "scene_matcher_residual_prosac",
    }:
        cmd.extend(["--hybrid_residual_alpha_max", str(float(hybrid_residual_alpha_max))])
    if recipe == "lff_feedback_prosac":
        cmd.append("--feedback_detector_full_res")
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
    if recipe in {
        "covisibility_prosac",
        "scene_matcher_prosac",
        "scene_matcher_residual_prosac",
        "scene_matcher_prosac_magsac",
        "scene_matcher_coverage_prosac",
        "covisibility_prosac_magsac",
        "covisibility_prosac_loftr",
        "lff_residual_prosac_loftr",
        "lff_feedback_prosac",
        "lff_residual_prosac",
        "oracle_prosac",
        "candidate_oracle_top1",
        "candidate_oracle_top4",
        "candidate_oracle_top8",
        "candidate_oracle_top16",
    }:
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
        if float(match_filter_query_score_weight) != 0.0:
            cmd.extend(["--match_filter_query_score_weight", str(float(match_filter_query_score_weight))])
        if str(sparse_match_filter_mode):
            cmd.extend(["--sparse_match_filter_mode", str(sparse_match_filter_mode)])
        if int(sparse_match_filter_top_m) > 0:
            cmd.extend(["--sparse_match_filter_top_m", str(int(sparse_match_filter_top_m))])
    if recipe == "scene_matcher_coverage_prosac":
        cmd.extend(
            [
                "--sparse_match_filter_mode",
                "calibrated_coverage",
                "--sparse_match_filter_top_m",
                "2048",
                "--dense_match_filter_mode",
                "calibrated_coverage",
                "--dense_match_filter_top_m",
                "2048",
                "--match_filter_min_matches",
                "1024",
            ]
        )
    if recipe in {
        "scene_matcher_prosac",
        "scene_matcher_residual_prosac",
        "scene_matcher_prosac_magsac",
        "scene_matcher_coverage_prosac",
    }:
        if not scene_matcher_path:
            raise ValueError("scene_matcher_prosac requires --scene_matcher_template or scene_matcher_path")
        cmd.extend(
            [
                "--scene_matcher_path",
                scene_matcher_path,
                "--scene_matcher_weight",
                str(float(scene_matcher_weight)),
                "--scene_matcher_topk",
                str(int(scene_matcher_topk)),
                "--scene_matcher_listwise_dustbin",
                str(scene_matcher_listwise_dustbin),
                "--scene_matcher_candidate_mode",
                str(scene_matcher_candidate_mode),
            ]
        )
        if str(scene_matcher_logit_norm) != "none":
            cmd.extend(["--scene_matcher_logit_norm", str(scene_matcher_logit_norm)])
        if float(scene_matcher_logit_clip) > 0.0:
            cmd.extend(["--scene_matcher_logit_clip", str(float(scene_matcher_logit_clip))])
    if calibrated_matchability_path:
        cmd.extend(
            [
                "--calibrated_matchability_path",
                calibrated_matchability_path,
                "--landmark_score_calibrated_matchability_weight",
                str(float(calibrated_matchability_weight)),
            ]
        )
        if float(match_calibrated_prior_weight) > 0.0:
            cmd.extend(["--match_calibrated_prior_weight", str(float(match_calibrated_prior_weight))])
    if selfmap_reliability_path:
        cmd.extend(
            [
                "--selfmap_reliability_path",
                selfmap_reliability_path,
                "--selfmap_reliability_stage",
                str(selfmap_reliability_stage),
                "--selfmap_reliability_center_cm",
                str(float(selfmap_reliability_center_cm)),
                "--selfmap_reliability_temperature_cm",
                str(float(selfmap_reliability_temperature_cm)),
            ]
        )
    if recipe in {"covisibility_prosac_loftr", "lff_residual_prosac_loftr"}:
        cmd.extend(
            [
                "--dense_matcher",
                "loftr_rendered",
                "--dim_pipeline",
                "loftr",
                "--dense_iters",
                "1",
                "--loftr_image_scale",
                "1.0",
                "--loftr_min_confidence",
                "0.0",
                "--loftr_max_matches",
                "4096",
            ]
        )
    if recipe == "oracle_prosac":
        cmd.extend(["--oracle_match_order", "sparse_reprojection"])
    if candidate_oracle_topk > 0:
        cmd.extend(
            [
                "--sparse_candidate_topk",
                str(candidate_oracle_topk),
                "--sparse_candidate_oracle_topk",
                str(candidate_oracle_topk),
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
    parser.add_argument("--recipes", default="covisibility_prosac")
    parser.add_argument("--eval_split", choices=["train", "test"], default="test")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--query_offset", type=int, default=0)
    parser.add_argument("--query_shards", type=int, default=1)
    parser.add_argument("--pnp_reprojection_error", type=float, default=8.0)
    parser.add_argument("--dense_iters", type=int, default=2)
    parser.add_argument(
        "--calibrated_matchability_template",
        default="",
        help="Optional per-scene template such as output/calib/{scene}/stdloc_bank_query_like.pt.",
    )
    parser.add_argument("--calibrated_matchability_weight", type=float, default=0.25)
    parser.add_argument(
        "--match_calibrated_prior_weight",
        type=float,
        default=0.0,
        help="Optional calibrated matchability logit-bias weight applied to 2D-3D matches.",
    )
    parser.add_argument(
        "--locability_prior_weight",
        type=float,
        default=0.05,
        help="Global STDLoc landmark prior logit-bias weight used during descriptor matching.",
    )
    parser.add_argument(
        "--ply_loc_feature_override_template",
        default="",
        help="Optional per-scene PLY template whose loc_* fields override checkpoint PLY descriptors.",
    )
    parser.add_argument(
        "--stdloc_detector_dir_template",
        default="",
        help="Optional per-scene detector directory template, e.g. output/maps/{scene}/detector.",
    )
    parser.add_argument(
        "--scene_matcher_template",
        default="",
        help="Per-scene template for SceneMatchNet checkpoints, e.g. output/scenematch/{scene}/best.pt.",
    )
    parser.add_argument("--scene_matcher_weight", type=float, default=0.35)
    parser.add_argument("--scene_matcher_topk", type=int, default=4)
    parser.add_argument("--scene_matcher_logit_norm", choices=["none", "center", "zscore"], default="none")
    parser.add_argument("--scene_matcher_logit_clip", type=float, default=0.0)
    parser.add_argument("--scene_matcher_listwise_dustbin", choices=["score", "drop"], default="score")
    parser.add_argument("--scene_matcher_candidate_mode", choices=["best", "all"], default="best")
    parser.add_argument("--match_filter_query_score_weight", type=float, default=0.0)
    parser.add_argument(
        "--sparse_match_filter_mode",
        choices=["", "none", "score", "image_grid", "xyz_grid", "image_xyz_grid", "local_geometry", "image_pair_geometry", "calibrated_coverage"],
        default="",
    )
    parser.add_argument("--sparse_match_filter_top_m", type=int, default=0)
    parser.add_argument("--hybrid_residual_alpha_max", type=float, default=0.03)
    parser.add_argument(
        "--selfmap_reliability_template",
        default="",
        help="Per-scene self-map eval summary template used for soft LFF reliability.",
    )
    parser.add_argument("--selfmap_reliability_stage", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--selfmap_reliability_center_cm", type=float, default=10.0)
    parser.add_argument("--selfmap_reliability_temperature_cm", type=float, default=1.0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max_memory_used_mb", type=int, default=1000)
    parser.add_argument("--max_utilization", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    scenes = parse_csv(args.scenes) or list(DEFAULT_SCENES)
    recipes = parse_csv(args.recipes) or ["covisibility_prosac"]
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
        calibrated_path = (
            args.calibrated_matchability_template.format(scene=job.scene, tag=args.tag, recipe=job.recipe)
            if args.calibrated_matchability_template
            else ""
        )
        scene_matcher_path = (
            args.scene_matcher_template.format(scene=job.scene, tag=args.tag, recipe=job.recipe)
            if args.scene_matcher_template
            else ""
        )
        selfmap_reliability_path = (
            args.selfmap_reliability_template.format(scene=job.scene, tag=args.tag, recipe=job.recipe)
            if args.selfmap_reliability_template
            else ""
        )
        ply_loc_feature_override = (
            args.ply_loc_feature_override_template.format(scene=job.scene, tag=args.tag, recipe=job.recipe)
            if args.ply_loc_feature_override_template
            else ""
        )
        stdloc_detector_dir = (
            args.stdloc_detector_dir_template.format(scene=job.scene, tag=args.tag, recipe=job.recipe)
            if args.stdloc_detector_dir_template
            else ""
        )
        if calibrated_path and not args.dry_run and not Path(calibrated_path).exists():
            raise FileNotFoundError(f"calibrated matchability not found for {job.scene}: {calibrated_path}")
        if job.recipe in {
            "scene_matcher_prosac",
            "scene_matcher_residual_prosac",
            "scene_matcher_prosac_magsac",
            "scene_matcher_coverage_prosac",
        } and not scene_matcher_path:
            raise ValueError("scene_matcher_prosac requires --scene_matcher_template")
        if scene_matcher_path and not args.dry_run and not Path(scene_matcher_path).exists():
            raise FileNotFoundError(f"scene matcher checkpoint not found for {job.scene}: {scene_matcher_path}")
        if selfmap_reliability_path and not args.dry_run and not Path(selfmap_reliability_path).exists():
            raise FileNotFoundError(
                f"self-map reliability summary not found for {job.scene}: {selfmap_reliability_path}"
            )
        if ply_loc_feature_override and not args.dry_run and not Path(ply_loc_feature_override).exists():
            raise FileNotFoundError(
                f"PLY loc feature override not found for {job.scene}: {ply_loc_feature_override}"
            )
        if stdloc_detector_dir and not args.dry_run and not Path(stdloc_detector_dir).exists():
            raise FileNotFoundError(f"STDLoc detector dir not found for {job.scene}: {stdloc_detector_dir}")
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
            dense_iters=args.dense_iters,
            calibrated_matchability_path=calibrated_path,
            calibrated_matchability_weight=args.calibrated_matchability_weight,
            match_calibrated_prior_weight=args.match_calibrated_prior_weight,
            locability_prior_weight=args.locability_prior_weight,
            ply_loc_feature_override=ply_loc_feature_override,
            stdloc_detector_dir=stdloc_detector_dir,
            scene_matcher_path=scene_matcher_path,
            scene_matcher_weight=args.scene_matcher_weight,
            scene_matcher_topk=args.scene_matcher_topk,
            scene_matcher_logit_norm=args.scene_matcher_logit_norm,
            scene_matcher_logit_clip=args.scene_matcher_logit_clip,
            scene_matcher_listwise_dustbin=args.scene_matcher_listwise_dustbin,
            scene_matcher_candidate_mode=args.scene_matcher_candidate_mode,
            match_filter_query_score_weight=args.match_filter_query_score_weight,
            sparse_match_filter_mode=args.sparse_match_filter_mode,
            sparse_match_filter_top_m=args.sparse_match_filter_top_m,
            hybrid_residual_alpha_max=args.hybrid_residual_alpha_max,
            selfmap_reliability_path=selfmap_reliability_path,
            selfmap_reliability_stage=args.selfmap_reliability_stage,
            selfmap_reliability_center_cm=args.selfmap_reliability_center_cm,
            selfmap_reliability_temperature_cm=args.selfmap_reliability_temperature_cm,
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
