from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


CAMBRIDGE_SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)


@dataclass(frozen=True)
class CommandJob:
    command: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)

    def with_gpu(self, gpu: str | int | None) -> "CommandJob":
        if gpu is None or str(gpu) == "":
            return self
        env = dict(self.env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return CommandJob(command=list(self.command), cwd=self.cwd, env=env)


@dataclass(frozen=True)
class StdlocEvalConfig:
    scene: str
    data_root: Path
    map_root: Path
    map_scene: str | None = None
    output_dir: Path | None = None
    repo_root: Path = Path(".")
    stdloc_root: Path = Path("third_party/stdloc")
    python_bin: str = sys.executable
    cfg: str | Path = Path("configs/stdloc_cambridge.yaml")
    images: str = "processed"
    resolution: int = 1
    feature_type: str = "sp"
    gaussian_type: str = "3dgs"
    data_device: str = "cpu"
    eval_split: str = "test"
    iteration: int = -1
    prefix: str | None = None
    max_test_cameras: int | None = None
    test_stride: int = 1
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class StdlocTrainConfig:
    scene: str
    data_root: Path
    map_root: Path
    repo_root: Path = Path(".")
    stdloc_root: Path = Path("third_party/stdloc")
    python_bin: str = sys.executable
    images: str = "processed"
    resolution: int = 1
    feature_type: str = "sp"
    gaussian_type: str = "3dgs"
    data_device: str = "cpu"
    iterations: int = 30000
    detector_iterations: int = 30000
    detector_folder: str = "detector"
    landmark_num: int = 16384
    landmark_k: int = 32
    densify_grad_threshold: float = 0.0004
    position_lr_init: float = 0.000016
    scaling_lr: float = 0.001
    test_iterations: tuple[int, ...] = (7000, 30000)
    save_iterations: tuple[int, ...] = (7000, 30000)
    test_detector_iterations: tuple[int, ...] = (30000,)
    save_detector_iterations: tuple[int, ...] = (30000,)
    stream_cameras: bool = False
    train_only_cameras: bool = False
    extra_args: tuple[str, ...] = ()


def _stdloc_cwd(repo_root: Path, stdloc_root: Path) -> Path:
    return Path(repo_root) / Path(stdloc_root)


def _resolve_path(repo_root: Path, path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(repo_root) / path


def _scene_path(repo_root: Path, root: Path, scene: str) -> str:
    return str(_resolve_path(repo_root, root) / scene)


def resolve_scene_images(data_root: Path | str, scene: str, preferred: str) -> str:
    scene_root = Path(data_root) / scene
    if scene_root.exists() and preferred and not (scene_root / preferred).exists():
        return "."
    return preferred


def _float_arg(value: float) -> str:
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text or "0"


def build_eval_job(config: StdlocEvalConfig) -> CommandJob:
    command = [
        str(config.python_bin),
        "stdloc.py",
        "-s",
        _scene_path(config.repo_root, config.data_root, config.scene),
        "-m",
        _scene_path(config.repo_root, config.map_root, config.map_scene or config.scene),
        "-r",
        str(config.resolution),
        "-f",
        str(config.feature_type),
        "-g",
        str(config.gaussian_type),
        "--images",
        str(config.images),
        "--data_device",
        str(config.data_device),
        "--cfg",
        str(config.cfg),
        "--eval_split",
        str(config.eval_split),
    ]
    if int(config.iteration) != -1:
        command.extend(["--iteration", str(config.iteration)])
    if config.prefix:
        command.extend(["--prefix", str(config.prefix)])
    if config.max_test_cameras is not None and int(config.max_test_cameras) > 0:
        command.extend(["--max_test_cameras", str(config.max_test_cameras)])
    if int(config.test_stride) != 1:
        command.extend(["--test_stride", str(config.test_stride)])
    if config.output_dir is not None:
        command.extend(["--output_path", str(_resolve_path(config.repo_root, config.output_dir))])
    command.extend(str(arg) for arg in config.extra_args)
    return CommandJob(command=command, cwd=_stdloc_cwd(config.repo_root, config.stdloc_root))


def build_train_job(config: StdlocTrainConfig) -> CommandJob:
    command = [
        str(config.python_bin),
        "train.py",
        "-s",
        _scene_path(config.repo_root, config.data_root, config.scene),
        "-m",
        _scene_path(config.repo_root, config.map_root, config.scene),
        "-r",
        str(config.resolution),
        "-f",
        str(config.feature_type),
        "-g",
        str(config.gaussian_type),
        "--iterations",
        str(config.iterations),
        "--data_device",
        str(config.data_device),
        "--train_detector",
        "--train_detector_iterations",
        str(config.detector_iterations),
        "--detector_folder",
        str(config.detector_folder),
        "--landmark_num",
        str(config.landmark_num),
        "--landmark_k",
        str(config.landmark_k),
        "--densify_grad_threshold",
        _float_arg(config.densify_grad_threshold),
        "--images",
        str(config.images),
        "--position_lr_init",
        _float_arg(config.position_lr_init),
        "--scaling_lr",
        _float_arg(config.scaling_lr),
        "--test_iterations",
        *(str(v) for v in config.test_iterations),
        "--save_iterations",
        *(str(v) for v in config.save_iterations),
        "--test_detector_iterations",
        *(str(v) for v in config.test_detector_iterations),
        "--save_detector_iterations",
        *(str(v) for v in config.save_detector_iterations),
    ]
    if config.stream_cameras:
        command.append("--stream_cameras")
    if config.train_only_cameras:
        command.append("--train_only_cameras")
    command.extend(str(arg) for arg in config.extra_args)
    return CommandJob(command=command, cwd=_stdloc_cwd(config.repo_root, config.stdloc_root))


def command_to_shell(job: CommandJob) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(job.env.items())
    )
    command = " ".join(shlex.quote(part) for part in job.command)
    return f"{env_prefix} {command}".strip()


def run_job(job: CommandJob, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(job.env)
    return subprocess.run(
        job.command,
        cwd=str(job.cwd),
        env=env,
        check=check,
        text=True,
    )


def merge_env(job: CommandJob, env: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = dict(env or {})
    merged.update(job.env)
    return merged


def parse_gpu_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = [str(item) for item in value]
    return [item for item in raw if item]
