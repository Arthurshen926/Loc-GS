#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from loc_gs.stdloc_native.commands import (
    CAMBRIDGE_SCENES,
    CommandJob,
    StdlocEvalConfig,
    StdlocTrainConfig,
    build_eval_job,
    build_train_job,
    command_to_shell,
    parse_gpu_list,
    resolve_scene_images,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assign_scene_gpus(scenes: list[str], gpus: list[str]) -> list[tuple[str, str]]:
    if not scenes:
        return []
    if not gpus:
        gpus = [""]
    return [(scene, gpus[idx % len(gpus)]) for idx, scene in enumerate(scenes)]


def _make_eval_job(args: argparse.Namespace, scene: str, gpu: str) -> CommandJob:
    output_dir = Path(args.output_root) / scene
    map_scene = parse_map_name_overrides(args.map_name_overrides).get(scene, scene)
    images = args.images if args.no_auto_images else resolve_scene_images(args.data_root, scene, args.images)
    cfg = StdlocEvalConfig(
        scene=scene,
        map_scene=map_scene,
        data_root=Path(args.data_root),
        map_root=Path(args.map_root),
        output_dir=output_dir,
        repo_root=repo_root(),
        python_bin=args.python_bin or "python",
        cfg=args.cfg,
        images=images,
        eval_split=args.eval_split,
        max_test_cameras=args.max_test_cameras if args.max_test_cameras > 0 else None,
        test_stride=args.test_stride,
    )
    return build_eval_job(cfg).with_gpu(gpu)


def _make_train_job(args: argparse.Namespace, scene: str, gpu: str) -> CommandJob:
    images = args.images if args.no_auto_images else resolve_scene_images(args.data_root, scene, args.images)
    cfg = StdlocTrainConfig(
        scene=scene,
        data_root=Path(args.data_root),
        map_root=Path(args.map_root),
        repo_root=repo_root(),
        python_bin=args.python_bin or "python",
        images=images,
        iterations=args.iterations,
        detector_iterations=args.detector_iterations,
        stream_cameras=args.stream_cameras,
        train_only_cameras=args.train_only_cameras,
    )
    return build_train_job(cfg).with_gpu(gpu)


def parse_map_name_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"map override must have SCENE=MAP_SCENE format: {value}")
        scene, map_scene = value.split("=", 1)
        scene = scene.strip()
        map_scene = map_scene.strip()
        if not scene or not map_scene:
            raise ValueError(f"map override must have non-empty scene names: {value}")
        overrides[scene] = map_scene
    return overrides


def _launch_one_job(
    phase: str,
    scene: str,
    job: CommandJob,
    gpu: str,
    log_dir: Path,
) -> tuple[subprocess.Popen, object, str, str, str]:
    if gpu != "":
        job = job.with_gpu(gpu)
    env = os.environ.copy()
    env.update(job.env)
    launch_gpu = job.env.get("CUDA_VISIBLE_DEVICES", "")
    log_path = log_dir / f"{phase}_{scene}_{int(time.time())}.log"
    log_f = log_path.open("w", encoding="utf-8")
    print(f"[launch] {phase}:{scene} gpu={launch_gpu or 'default'} log={log_path}")
    proc = subprocess.Popen(
        job.command,
        cwd=str(job.cwd),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_f, phase, scene, launch_gpu


def _launch_jobs(
    jobs: list[tuple[str, str, CommandJob]],
    log_dir: Path,
    gpus: list[str] | None = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, object, str, str, str]] = []
    dynamic_gpus = gpus is not None
    available_gpus = list(gpus or [])
    if dynamic_gpus and not available_gpus:
        available_gpus = [""]
    active_gpus: set[str] = set()
    while pending or running:
        if dynamic_gpus:
            free_gpus = [gpu for gpu in available_gpus if gpu not in active_gpus]
            while pending and free_gpus:
                phase, scene, job = pending.pop(0)
                gpu = free_gpus.pop(0)
                running.append(_launch_one_job(phase, scene, job, gpu, log_dir))
                active_gpus.add(gpu)
        else:
            launched = []
            for index, (phase, scene, job) in enumerate(pending):
                gpu = job.env.get("CUDA_VISIBLE_DEVICES", "")
                if gpu in active_gpus:
                    continue
                running.append(_launch_one_job(phase, scene, job, "", log_dir))
                active_gpus.add(gpu)
                launched.append(index)
            for index in reversed(launched):
                pending.pop(index)
        if not running:
            raise RuntimeError("No runnable STDLoc jobs were launched")
        next_running = []
        for proc, log_f, phase, scene, gpu in running:
            code = proc.poll()
            if code is None:
                next_running.append((proc, log_f, phase, scene, gpu))
                continue
            log_f.close()
            active_gpus.discard(gpu)
            if code != 0:
                raise RuntimeError(f"STDLoc {phase} failed for {scene} with exit code {code}")
        running = next_running
        if pending or running:
            time.sleep(5)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch native STDLoc Cambridge jobs across GPUs.")
    parser.add_argument("--scenes", nargs="+", default=list(CAMBRIDGE_SCENES))
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--phase", choices=["eval", "train", "both"], default="eval")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--map_root", default="output/stdloc/map_cambridge_spgs")
    parser.add_argument("--map_name_overrides", nargs="*", default=[])
    parser.add_argument("--output_root", default="output/stdloc_native/results")
    parser.add_argument("--log_dir", default="logs/stdloc_native")
    parser.add_argument("--cfg", default="configs/stdloc_cambridge.yaml")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--no_auto_images", action="store_true")
    parser.add_argument("--python_bin", default="")
    parser.add_argument("--eval_split", choices=["test", "train"], default="test")
    parser.add_argument("--max_test_cameras", type=int, default=0)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--detector_iterations", type=int, default=30000)
    parser.add_argument("--stream_cameras", action="store_true")
    parser.add_argument("--train_only_cameras", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    gpus = parse_gpu_list(args.gpus)
    assignments = assign_scene_gpus(list(args.scenes), gpus)
    jobs: list[tuple[str, str, CommandJob]] = []
    for scene, gpu in assignments:
        if args.phase in {"train", "both"}:
            jobs.append(("train", scene, _make_train_job(args, scene, gpu)))
        if args.phase in {"eval", "both"}:
            jobs.append(("eval", scene, _make_eval_job(args, scene, gpu)))
    if args.dry_run:
        for _phase, _scene, job in jobs:
            print(command_to_shell(job))
        return
    _launch_jobs(jobs, Path(args.log_dir), gpus=gpus)


if __name__ == "__main__":
    main()
