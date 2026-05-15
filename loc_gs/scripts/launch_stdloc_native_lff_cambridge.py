#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from loc_gs.scripts.launch_stdloc_native_cambridge import (
    _launch_jobs,
    assign_scene_gpus,
    parse_map_name_overrides,
    repo_root,
)
from loc_gs.stdloc_native.commands import (
    CAMBRIDGE_SCENES,
    CommandJob,
    StdlocEvalConfig,
    build_eval_job,
    command_to_shell,
    parse_gpu_list,
    resolve_scene_images,
)
from loc_gs.stdloc_native.lff_export import build_lff_feature_map


def _abs_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def _format_template(template: str, *, scene: str, map_scene: str, tag: str) -> str:
    return template.format(scene=scene, map_scene=map_scene, tag=tag)


def _lff_map_scene(scene: str, suffix: str) -> str:
    suffix = suffix.strip("_")
    return f"{scene}_{suffix}" if suffix else scene


def _make_eval_job(
    args: argparse.Namespace,
    *,
    scene: str,
    output_map_scene: str,
    cfg_path: Path,
    gpu: str,
) -> CommandJob:
    images = args.images if args.no_auto_images else resolve_scene_images(args.data_root, scene, args.images)
    cfg = StdlocEvalConfig(
        scene=scene,
        map_scene=output_map_scene,
        data_root=Path(args.data_root),
        map_root=Path(args.output_map_root),
        output_dir=Path(args.output_root) / scene,
        repo_root=repo_root(),
        python_bin=args.python_bin or "python",
        cfg=str(_abs_path(cfg_path)),
        images=images,
        eval_split=args.eval_split,
        max_test_cameras=args.max_test_cameras if args.max_test_cameras > 0 else None,
        test_stride=args.test_stride,
    )
    return build_eval_job(cfg).with_gpu(gpu)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build native STDLoc maps with exported LFF descriptor residuals, then evaluate them."
    )
    parser.add_argument("--scenes", nargs="+", default=list(CAMBRIDGE_SCENES))
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--tag", default="native_lff")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--source_map_root", default="output/stdloc/map_cambridge_spgs")
    parser.add_argument("--map_name_overrides", nargs="*", default=[])
    parser.add_argument("--output_map_root", default="output/stdloc_native/lff_maps")
    parser.add_argument("--output_map_suffix", default="native_lff")
    parser.add_argument("--output_root", default="output/stdloc_native/results/native_lff")
    parser.add_argument("--log_dir", default="logs/stdloc_native_lff")
    parser.add_argument("--base_cfg", default="third_party/stdloc/configs/stdloc_cambridge.yaml")
    parser.add_argument("--checkpoint_template", required=True)
    parser.add_argument("--calibrated_matchability_template", default="")
    parser.add_argument("--selfmap_reliability_template", default="")
    parser.add_argument("--rho", type=float, default=-1.0)
    parser.add_argument("--selfmap_reliability_stage", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--selfmap_reliability_center_cm", type=float, default=10.0)
    parser.add_argument("--selfmap_reliability_temperature_cm", type=float, default=1.0)
    parser.add_argument("--selfmap_reliability_r5_center", type=float, default=0.5)
    parser.add_argument("--selfmap_reliability_r5_temperature", type=float, default=0.1)
    parser.add_argument("--descriptor_alpha_max", type=float, default=0.03)
    parser.add_argument("--decode_chunk_size", type=int, default=16384)
    parser.add_argument(
        "--selector_mode",
        choices=["uniform", "locability", "matchability", "combined", "reliability_boost"],
        default="reliability_boost",
    )
    parser.add_argument("--selector_matchability_weight", type=float, default=1.0)
    parser.add_argument("--selector_source_weight", type=float, default=0.0)
    parser.add_argument("--selector_floor", type=float, default=0.0)
    parser.add_argument("--selector_power", type=float, default=1.0)
    parser.add_argument("--selector_locability_weight", type=float, default=0.0)
    parser.add_argument("--locability_fusion_mode", choices=["blend", "boost"], default="boost")
    parser.add_argument("--prior_blend", type=float, default=0.25)
    parser.add_argument("--score_fusion_mode", choices=["blend", "boost"], default="boost")
    parser.add_argument("--base_sparse_prior_weight", type=float, default=0.0)
    parser.add_argument("--base_dense_prior_weight", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_overwrite_maps", action="store_true")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--no_auto_images", action="store_true")
    parser.add_argument("--python_bin", default="")
    parser.add_argument("--eval_split", choices=["test", "train"], default="test")
    parser.add_argument("--max_test_cameras", type=int, default=0)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    gpus = parse_gpu_list(args.gpus)
    assignments = assign_scene_gpus(list(args.scenes), gpus)
    overrides = parse_map_name_overrides(args.map_name_overrides)
    output_map_root = Path(args.output_map_root)
    jobs: list[tuple[str, str, CommandJob]] = []
    for scene, gpu in assignments:
        map_scene = overrides.get(scene, scene)
        output_map_scene = _lff_map_scene(scene, args.output_map_suffix)
        source_map = Path(args.source_map_root) / map_scene
        output_map = output_map_root / output_map_scene
        output_cfg = output_map / "stdloc_lff.yaml"
        checkpoint_path = _format_template(
            args.checkpoint_template,
            scene=scene,
            map_scene=map_scene,
            tag=args.tag,
        )
        calibration_path = (
            _format_template(args.calibrated_matchability_template, scene=scene, map_scene=map_scene, tag=args.tag)
            if args.calibrated_matchability_template
            else ""
        )
        reliability_path = (
            _format_template(args.selfmap_reliability_template, scene=scene, map_scene=map_scene, tag=args.tag)
            if args.selfmap_reliability_template
            else ""
        )
        if not args.dry_run:
            build_device = f"cuda:{gpu}" if args.device == "auto" and str(gpu) else args.device
            build_lff_feature_map(
                source_map=source_map,
                output_map=output_map,
                checkpoint_path=checkpoint_path,
                base_cfg_path=_abs_path(args.base_cfg),
                output_cfg_path=output_cfg,
                calibration_path=calibration_path or None,
                rho=args.rho if args.rho >= 0.0 else None,
                selfmap_reliability_path=reliability_path or None,
                selfmap_reliability_stage=args.selfmap_reliability_stage,
                selfmap_reliability_center_cm=args.selfmap_reliability_center_cm,
                selfmap_reliability_temperature_cm=args.selfmap_reliability_temperature_cm,
                selfmap_reliability_r5_center=(
                    args.selfmap_reliability_r5_center
                    if args.selfmap_reliability_r5_center >= 0.0
                    else None
                ),
                selfmap_reliability_r5_temperature=args.selfmap_reliability_r5_temperature,
                descriptor_alpha_max=args.descriptor_alpha_max,
                decode_chunk_size=args.decode_chunk_size,
                selector_mode=args.selector_mode,
                selector_matchability_weight=args.selector_matchability_weight,
                selector_source_weight=args.selector_source_weight,
                selector_floor=args.selector_floor,
                selector_power=args.selector_power,
                selector_locability_weight=args.selector_locability_weight,
                locability_fusion_mode=args.locability_fusion_mode,
                prior_blend=args.prior_blend,
                score_fusion_mode=args.score_fusion_mode,
                base_sparse_prior_weight=args.base_sparse_prior_weight,
                base_dense_prior_weight=args.base_dense_prior_weight,
                overwrite=not args.no_overwrite_maps,
                device=build_device,
            )
        job = _make_eval_job(
            args,
            scene=scene,
            output_map_scene=output_map_scene,
            cfg_path=output_cfg,
            gpu=gpu,
        )
        jobs.append(("eval", scene, job))
    if args.dry_run:
        for _phase, _scene, job in jobs:
            print(command_to_shell(job))
        return
    _launch_jobs(jobs, Path(args.log_dir), gpus=gpus)


if __name__ == "__main__":
    main()
