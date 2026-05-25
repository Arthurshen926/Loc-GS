#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import yaml

from loc_gs.stdloc_native.commands import (
    StdlocEvalConfig,
    build_eval_job,
    command_to_shell,
    resolve_scene_images,
    run_job,
)
from loc_gs.scripts.write_native_eval_audit_bundle import (
    _third_party_stdloc_evaluator_modified,
    write_native_eval_audit_bundle,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendored_stdloc_cfg() -> Path:
    return repo_root() / "third_party/stdloc/configs/stdloc_cambridge.yaml"


def _resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else repo_root() / raw


def _check_official_stdloc_parity(resolved_cfg: Path) -> None:
    if resolved_cfg.resolve() != _vendored_stdloc_cfg().resolve():
        raise ValueError(
            "official STDLoc parity requires the vendored STDLoc config: "
            f"{_vendored_stdloc_cfg()}"
        )
    if _third_party_stdloc_evaluator_modified():
        raise ValueError(
            "official STDLoc parity requires an unmodified third_party/stdloc evaluator"
        )


def _check_paper_safe_poselib_evaluator(resolved_cfg: Path) -> None:
    if _third_party_stdloc_evaluator_modified():
        raise ValueError(
            "paper-safe fixed poselib evaluation requires an unmodified third_party/stdloc evaluator"
        )
    solvers = _cfg_solver_names(resolved_cfg)
    if solvers != {"poselib"}:
        raise ValueError(
            "paper-safe fixed poselib evaluation requires sparse and dense solver=poselib"
        )


def _cfg_solver_names(resolved_cfg: Path) -> set[str]:
    try:
        payload = yaml.safe_load(resolved_cfg.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to parse STDLoc cfg: {resolved_cfg}") from exc
    if not isinstance(payload, dict):
        return set()
    solvers: set[str] = set()
    for stage in ("sparse", "dense"):
        raw_stage = payload.get(stage, {})
        if isinstance(raw_stage, dict) and raw_stage.get("solver"):
            solvers.add(str(raw_stage["solver"]))
    return solvers


def _check_cfg_supported_by_current_evaluator(resolved_cfg: Path) -> None:
    solvers = _cfg_solver_names(resolved_cfg)
    opencv_variant_solvers = {"opencv_prosac", "opencv_prosac_magsac"}
    if solvers & opencv_variant_solvers and not _third_party_stdloc_evaluator_modified():
        raise ValueError(
            "cfg uses opencv_prosac/opencv_prosac_magsac and requires the OpenCV "
            "variant evaluator; the current third_party/stdloc evaluator is the "
            "unmodified vendored parity path"
        )


def _sampled_count(map_path: Path) -> int | None:
    sampled_idx = map_path / "detector" / "sampled_idx.pkl"
    if not sampled_idx.exists():
        return None
    try:
        with sampled_idx.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    try:
        return int(len(payload))
    except TypeError:
        try:
            return int(payload.numel())
        except AttributeError:
            return None


def _check_expected_sampled_count(map_path: Path, expected: int) -> None:
    if int(expected) <= 0:
        return
    count = _sampled_count(map_path)
    if count != int(expected):
        raise ValueError(
            f"native map sampled count mismatch for {map_path}: "
            f"expected {int(expected)}, got {count}"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the source-of-truth STDLoc evaluator from Loc-GS.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--map_root", default="output/stdloc/map_cambridge_spgs")
    parser.add_argument("--map_scene", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--cfg", default="third_party/stdloc/configs/stdloc_cambridge.yaml")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--no_auto_images", action="store_true")
    parser.add_argument("--python_bin", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval_split", choices=["test", "train"], default="test")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--max_test_cameras", type=int, default=0)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--expected_sampled_count", type=int, default=0)
    parser.add_argument(
        "--require_official_stdloc_parity",
        action="store_true",
        help="Reject the run unless it uses the vendored STDLoc config and an unmodified vendored evaluator.",
    )
    parser.add_argument(
        "--require_paper_safe_poselib_evaluator",
        action="store_true",
        help="Reject the run unless sparse/dense solvers are poselib and the vendored evaluator is unmodified.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = _resolve_repo_path(args.cfg)
    if bool(args.require_official_stdloc_parity):
        _check_official_stdloc_parity(resolved_cfg)
    if bool(args.require_paper_safe_poselib_evaluator):
        _check_paper_safe_poselib_evaluator(resolved_cfg)
    _check_cfg_supported_by_current_evaluator(resolved_cfg)
    cfg = StdlocEvalConfig(
        scene=args.scene,
        map_scene=args.map_scene or None,
        data_root=Path(args.data_root),
        map_root=Path(args.map_root),
        output_dir=output_dir,
        repo_root=repo_root(),
        python_bin=args.python_bin or sys.executable,
        cfg=resolved_cfg,
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
    map_scene = args.map_scene or args.scene
    resolved_map_path = _resolve_repo_path(Path(args.map_root) / map_scene)
    _check_expected_sampled_count(resolved_map_path, int(args.expected_sampled_count))
    run_job(job)
    if output_dir is not None and (output_dir / "summary.json").exists():
        write_native_eval_audit_bundle(
            output_dir,
            scene=args.scene,
            split=args.eval_split,
            data_root=str(_resolve_repo_path(Path(args.data_root) / args.scene)),
            checkpoint_path="",
            map_path=str(resolved_map_path),
            command=command_to_shell(job),
            feedback_split="native_eval_train" if args.eval_split == "train" else "native_eval",
            quality_gate_mode="disabled",
        )


if __name__ == "__main__":
    main()
