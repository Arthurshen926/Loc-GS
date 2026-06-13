#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_SCENES = ("GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch")


def _reject_test_split(split_name: str) -> str:
    split = str(split_name).strip()
    if not split:
        raise ValueError("split_name is required")
    lowered = split.lower()
    if lowered == "test" or lowered == "official_test" or lowered.endswith("_test"):
        raise ValueError("test split is not allowed for solver-feedback training or model selection")
    return split


def _scene_list(scenes: Sequence[str] | None) -> list[str]:
    if not scenes:
        return list(DEFAULT_SCENES)
    out: list[str] = []
    for scene in scenes:
        for part in str(scene).split(","):
            item = part.strip()
            if item:
                out.append(item)
    if not out:
        raise ValueError("at least one scene is required")
    return out


def build_commands(
    *,
    scenes: Sequence[str] | None = None,
    split_name: str = "train_selfmap",
    output_root: str | Path = "output/ulfloc_solver_feedback_mainline_v2",
    feedback_root: str | Path = "output/feedback_banks",
    ulf_root: str | Path = "/root/ULF-Loc",
    data_root: str | Path = "/mnt/pool/sqy/Cambridge_stdloc",
    best_log_root: str | Path = "output/ulfloc_best",
    python: str | Path = "/root/miniconda3/envs/cybersim_agent/bin/python",
    images: str = "processed",
    eval_split_name: str = "train_dev",
    initial_detector_iterations: int = 1000,
    residual_detector_iterations: int = 1000,
    detector_lr: float = 1e-3,
    solver_feedback_residual_alpha: float = 0.1,
    dry_run: bool = True,
) -> list[list[str]]:
    """Return split-safe dry-run commands for the ULF solver-feedback mainline.

    The launcher deliberately emits commands instead of running them. It is used
    to schedule scene jobs externally while keeping the generated training path
    free of Cambridge official test split references.
    """

    split = _reject_test_split(split_name)
    commands: list[list[str]] = []
    output_root = Path(output_root)
    feedback_root = Path(feedback_root)
    data_root = Path(data_root)
    best_log_root = Path(best_log_root)
    python = str(python)
    ulf_root = str(ulf_root)
    for scene in _scene_list(scenes):
        scene_out = output_root / scene
        source_path = data_root / scene
        input_log_dir = best_log_root / scene / "log"
        model_path = input_log_dir.parent
        cfg = input_log_dir / f"ulfloc_cambridge_solver_feedback_{split}.yaml"
        trace_dir = scene_out / "sparse_trace"
        impact = scene_out / "impact" / "impact_attribution.pt"
        feedback_v4 = scene_out / "feedback_v4" / "feedback_v4.pt"
        trace_payload = trace_dir / "sparse_pnp_trace_payload.pt"
        initial_detector = scene_out / "initial_detector" / "scene_detector.pth"
        residual_detector = scene_out / "residual_detector" / "scene_detector.pth"

        commands.append(
            [
                python,
                "-m",
                "loc_gs.scripts.train_ulfloc_scene_detector_online",
                "--scene",
                scene,
                "--source_path",
                str(source_path),
                "--model_path",
                str(model_path),
                "--input_log_dir",
                str(input_log_dir),
                "--cfg",
                str(cfg),
                "--output_dir",
                str(scene_out / "initial_detector"),
                "--split_name",
                split,
                "--ulf_root",
                ulf_root,
                "--images",
                images,
                "--iterations",
                str(int(initial_detector_iterations)),
                "--lr",
                str(float(detector_lr)),
                "--device",
                "cuda",
                "--data_device",
                "cpu",
                "--mask_mode",
                "stdloc_object_distort",
                "--target_sigma_px",
                "1.5",
                "--positive_weight",
                "8.0",
                "--background_weight",
                "1.0",
            ]
        )
        commands.append(
            [
                python,
                "-m",
                "loc_gs.scripts.eval_ulfloc_sparse_only",
                "--source_path",
                str(source_path),
                "--model_path",
                str(model_path),
                "--input_log_dir",
                str(input_log_dir),
                "--cfg",
                str(cfg),
                "--output_dir",
                str(trace_dir),
                "--ulf_root",
                ulf_root,
                "--scene",
                scene,
                "--split_name",
                split,
                "--images",
                images,
                "--data_device",
                "cpu",
                "--device",
                "cuda",
                "--scene_detector_checkpoint",
                str(initial_detector),
                "--scene_detector_mode",
                "stdloc_fullres",
                "--dump_sparse_match_attributions",
            ]
        )
        commands.append(
            [
                python,
                "-m",
                "loc_gs.scripts.build_sparse_feedback_v4",
                "--trace",
                str(trace_payload),
                "--output_dir",
                str(scene_out / "feedback_v4"),
            ]
        )
        commands.append(
            [
                python,
                "-m",
                "loc_gs.scripts.build_solver_feedback_impact",
                "--feedback",
                str(feedback_v4),
                "--output_dir",
                str(scene_out / "impact"),
            ]
        )
        commands.append(
            [
                python,
                "-m",
                "loc_gs.scripts.train_ulfloc_scene_detector_online",
                "--scene",
                scene,
                "--source_path",
                str(source_path),
                "--model_path",
                str(model_path),
                "--input_log_dir",
                str(input_log_dir),
                "--cfg",
                str(cfg),
                "--output_dir",
                str(scene_out / "residual_detector"),
                "--split_name",
                split,
                "--ulf_root",
                ulf_root,
                "--images",
                images,
                "--iterations",
                str(int(residual_detector_iterations)),
                "--lr",
                str(float(detector_lr)),
                "--device",
                "cuda",
                "--data_device",
                "cpu",
                "--mask_mode",
                "stdloc_object_distort",
                "--target_sigma_px",
                "1.5",
                "--positive_weight",
                "8.0",
                "--background_weight",
                "1.0",
                "--solver_impact",
                str(impact),
                "--solver_feedback_residual_alpha",
                str(float(solver_feedback_residual_alpha)),
                "--suppression_weight",
                "2.0",
                "--init_checkpoint",
                str(initial_detector),
            ]
        )
        commands.append(
            [
                python,
                "-m",
                "loc_gs.scripts.eval_ulfloc_sparse_only",
                "--source_path",
                str(source_path),
                "--model_path",
                str(model_path),
                "--input_log_dir",
                str(input_log_dir),
                "--cfg",
                str(cfg),
                "--output_dir",
                str(scene_out / "eval_train_dev_residual_detector"),
                "--ulf_root",
                ulf_root,
                "--scene",
                scene,
                "--split_name",
                str(eval_split_name),
                "--images",
                images,
                "--data_device",
                "cpu",
                "--device",
                "cuda",
                "--scene_detector_checkpoint",
                str(residual_detector),
                "--scene_detector_mode",
                "stdloc_fullres",
            ]
        )
    if not dry_run:
        raise ValueError("this launcher currently supports dry-run command generation only")
    return commands


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit split-safe ULF solver-feedback mainline commands.")
    parser.add_argument("--scene", action="append", dest="scenes", default=None)
    parser.add_argument("--split_name", default="train_selfmap")
    parser.add_argument("--output_root", default="output/ulfloc_solver_feedback_mainline_v2")
    parser.add_argument("--feedback_root", default="output/feedback_banks")
    parser.add_argument("--ulf_root", default="/root/ULF-Loc")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--best_log_root", default="output/ulfloc_best")
    parser.add_argument("--python", default="/root/miniconda3/envs/cybersim_agent/bin/python")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    commands = build_commands(
        scenes=args.scenes,
        split_name=args.split_name,
        output_root=args.output_root,
        feedback_root=args.feedback_root,
        ulf_root=args.ulf_root,
        data_root=args.data_root,
        best_log_root=args.best_log_root,
        python=args.python,
        images=args.images,
        dry_run=bool(args.dry_run),
    )
    if bool(args.json):
        print(json.dumps(commands, indent=2))
    else:
        for command in commands:
            print(" ".join(shlex.quote(part) for part in command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
