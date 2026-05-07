from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CambridgeCameraRecord:
    image_name: str
    pose_w2c: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _c2w_to_w2c(rotation_c2w: np.ndarray, position: np.ndarray) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation_c2w.astype(np.float32)
    c2w[:3, 3] = position.astype(np.float32)
    return np.linalg.inv(c2w).astype(np.float32)


def _w2c_from_rotation_and_center(rotation_w2c: np.ndarray, position: np.ndarray) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation_w2c.astype(np.float32)
    w2c[:3, 3] = -rotation_w2c.astype(np.float32) @ position.astype(np.float32)
    return w2c


class CambridgeHybridDataset(Dataset):
    """Cambridge Landmarks RGB + pose dataset for hybrid Loc-GS training.

    The training path reads STDLoc/3DGS `cameras.json` files generated from
    COLMAP. The test path can read Cambridge `dataset_test.txt` pose files
    directly, which keeps evaluation independent from STDLoc internals.
    """

    def __init__(
        self,
        scene_root: str | Path,
        split: str = "train",
        cameras_json: str | Path | None = None,
        image_subdir: str = "processed",
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        max_frames: int = 0,
    ) -> None:
        self.scene_root = Path(scene_root)
        self.split = split
        self.image_subdir = image_subdir.strip("/")
        self.image_height = image_height
        self.image_width = image_width
        self._fallback_fx = fx
        self._fallback_fy = fy
        self._fallback_cx = cx
        self._fallback_cy = cy

        if cameras_json is not None:
            self.records = self._load_camera_json(Path(cameras_json))
            self.records = self._filter_records_by_split(self.records, split)
        else:
            pose_file = self.scene_root / ("dataset_test.txt" if split == "test" else "dataset_train.txt")
            self.records = self._load_cambridge_pose_file(pose_file)

        if max_frames and max_frames > 0:
            self.records = self.records[: int(max_frames)]
        if not self.records:
            raise ValueError(f"No Cambridge frames found for split={split} under {self.scene_root}")

    def _load_camera_json(self, path: Path) -> list[CambridgeCameraRecord]:
        data = json.loads(path.read_text(encoding="utf-8"))
        records: list[CambridgeCameraRecord] = []
        for item in data:
            width = int(item.get("width", self.image_width or 0))
            height = int(item.get("height", self.image_height or 0))
            if width <= 0 or height <= 0:
                raise ValueError(f"Camera JSON entry lacks valid dimensions: {item}")
            fx = float(item.get("fx", self._fallback_fx or width))
            fy = float(item.get("fy", self._fallback_fy or fx))
            cx = float(item.get("cx", (width - 1) * 0.5))
            cy = float(item.get("cy", (height - 1) * 0.5))
            rotation = np.asarray(item["rotation"], dtype=np.float32)
            position = np.asarray(item["position"], dtype=np.float32)
            records.append(
                CambridgeCameraRecord(
                    image_name=str(item["img_name"]),
                    pose_w2c=_c2w_to_w2c(rotation, position),
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
            )
        return records

    def _split_names(self, split: str) -> Optional[set[str]]:
        pose_file = self.scene_root / ("dataset_test.txt" if split == "test" else "dataset_train.txt")
        if split not in {"train", "test"} or not pose_file.exists():
            return None
        names: set[str] = set()
        for raw in pose_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("Visual") or line.startswith("ImageFile"):
                continue
            parts = line.split()
            if parts:
                names.add(parts[0])
        return names or None

    def _filter_records_by_split(
        self,
        records: list[CambridgeCameraRecord],
        split: str,
    ) -> list[CambridgeCameraRecord]:
        names = self._split_names(split)
        if names is None:
            return records
        filtered = [record for record in records if record.image_name in names]
        return filtered

    def _load_cambridge_pose_file(self, path: Path) -> list[CambridgeCameraRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Cambridge pose split not found: {path}")
        records: list[CambridgeCameraRecord] = []
        width = int(self.image_width or 0)
        height = int(self.image_height or 0)
        fx = float(self._fallback_fx or max(width, 1))
        fy = float(self._fallback_fy or fx)
        cx = float(self._fallback_cx if self._fallback_cx is not None else (max(width, 1) - 1) * 0.5)
        cy = float(self._fallback_cy if self._fallback_cy is not None else (max(height, 1) - 1) * 0.5)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("Visual") or line.startswith("ImageFile"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            image_name = parts[0]
            position = np.asarray([float(v) for v in parts[1:4]], dtype=np.float32)
            qw, qx, qy, qz = [float(v) for v in parts[4:8]]
            rotation_w2c = _quaternion_to_rotation(qw, qx, qy, qz)
            records.append(
                CambridgeCameraRecord(
                    image_name=image_name,
                    pose_w2c=_w2c_from_rotation_and_center(rotation_w2c, position),
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, image_name: str) -> Path:
        candidates = []
        if self.image_subdir:
            candidates.append(self.scene_root / self.image_subdir / image_name)
        candidates.append(self.scene_root / image_name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.image_width is not None and self.image_height is not None:
                image = image.resize((int(self.image_width), int(self.image_height)), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def scaled_intrinsics(self, width: int, height: int, index: int = 0) -> dict[str, float]:
        record = self.records[index]
        sx = float(width) / max(float(record.width), 1.0)
        sy = float(height) / max(float(record.height), 1.0)
        return {
            "fx": record.fx * sx,
            "fy": record.fy * sy,
            "cx": record.cx * sx,
            "cy": record.cy * sy,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        rgb = self._load_rgb(self._image_path(record.image_name))
        _, height, width = rgb.shape
        intr = self.scaled_intrinsics(width, height, idx)
        K = torch.tensor(
            [
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        feature_K = K.clone()
        feature_K[0, :] /= 8.0
        feature_K[1, :] /= 8.0
        return {
            "rgb": rgb,
            "pose_w2c": torch.from_numpy(record.pose_w2c.copy()),
            "K": K,
            "feature_K": feature_K,
            "image_name": record.image_name,
            "frame_idx": torch.tensor(idx, dtype=torch.long),
        }
