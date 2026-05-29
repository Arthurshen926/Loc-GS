from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class QvecNormIssue:
    image_name: str
    qnorm: float
    delta: float


def normalize_qvec(qvec: Sequence[float] | np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(qvec, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {values.shape}")
    norm = float(np.linalg.norm(values))
    if norm <= eps:
        raise ValueError("cannot normalize zero-norm qvec")
    return values / norm


def qvec_to_rotmat(qvec: Sequence[float] | np.ndarray, *, normalize: bool = False) -> np.ndarray:
    values = normalize_qvec(qvec) if normalize else np.asarray(qvec, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {values.shape}")
    w, x, y, z = values
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def camera_center_from_colmap(
    qvec: Sequence[float] | np.ndarray,
    tvec: Sequence[float] | np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    rotation = qvec_to_rotmat(qvec, normalize=normalize)
    translation = np.asarray(tvec, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"tvec must have shape (3,), got {translation.shape}")
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return np.linalg.inv(world_to_camera)[:3, 3]


def audit_qvec_norms(
    records: Iterable[tuple[str, Sequence[float] | np.ndarray]],
    *,
    tolerance: float = 0.01,
) -> list[QvecNormIssue]:
    issues: list[QvecNormIssue] = []
    for image_name, qvec in records:
        values = np.asarray(qvec, dtype=np.float64)
        qnorm = float(np.linalg.norm(values))
        delta = abs(qnorm - 1.0)
        if delta > tolerance:
            issues.append(QvecNormIssue(image_name=str(image_name), qnorm=qnorm, delta=delta))
    return issues


def read_cambridge_pose_qvecs(scene_root: Path, *, split: str = "test") -> list[tuple[str, np.ndarray]]:
    split_path = Path(scene_root) / f"dataset_{split}.txt"
    if not split_path.exists():
        return []
    records: list[tuple[str, np.ndarray]] = []
    for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8 or Path(parts[0]).suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        try:
            qvec = np.asarray([float(value) for value in parts[4:8]], dtype=np.float64)
        except ValueError:
            continue
        records.append((parts[0], qvec))
    return records


def read_colmap_image_qvecs(images_bin: Path) -> list[tuple[str, np.ndarray]]:
    images_bin = Path(images_bin)
    if not images_bin.exists():
        return []
    records: list[tuple[str, np.ndarray]] = []
    with images_bin.open("rb") as handle:
        num_images = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(int(num_images)):
            handle.read(4)
            qvec = np.asarray(struct.unpack("<4d", handle.read(32)), dtype=np.float64)
            handle.read(24)
            handle.read(4)
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if char == b"":
                    raise EOFError(f"unexpected EOF while reading image name from {images_bin}")
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            num_points2d = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(24 * int(num_points2d), 1)
            records.append((name_bytes.decode("utf-8"), qvec))
    return records


def _cambridge_split_names(scene_root: Path, split: str) -> set[str]:
    split_path = Path(scene_root) / f"dataset_{split}.txt"
    if not split_path.exists():
        return set()
    names: set[str] = set()
    for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first = stripped.split()[0].strip(",")
        if Path(first).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            names.add(first)
    return names


def cambridge_pose_qnorm_issue_map(
    scene_root: Path,
    *,
    split: str = "test",
    tolerance: float = 0.01,
) -> dict[str, QvecNormIssue]:
    scene_root = Path(scene_root)
    records = read_colmap_image_qvecs(scene_root / "sparse" / "0" / "images.bin")
    if records:
        split_names = _cambridge_split_names(scene_root, split)
        if split_names:
            records = [(name, qvec) for name, qvec in records if name in split_names]
    else:
        records = read_cambridge_pose_qvecs(scene_root, split=split)
    return {
        issue.image_name: issue
        for issue in audit_qvec_norms(
            records,
            tolerance=tolerance,
        )
    }
