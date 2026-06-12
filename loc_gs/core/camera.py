from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

MissingPrincipalPoint = Literal["half_extent", "pixel_center"]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[float(self.fx), 0.0, float(self.cx)], [0.0, float(self.fy), float(self.cy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, *, width: int, height: int) -> "CameraIntrinsics":
        intr = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(intr[0, 0]),
            fy=float(intr[1, 1]),
            cx=float(intr[0, 2]),
            cy=float(intr[1, 2]),
        )


@dataclass(frozen=True)
class CameraRecord:
    image_id: str
    intrinsics: CameraIntrinsics
    pose_c2w: np.ndarray | None = None

    @property
    def pose_w2c(self) -> np.ndarray | None:
        if self.pose_c2w is None:
            return None
        pose = np.asarray(self.pose_c2w, dtype=np.float64).reshape(4, 4)
        inv = np.eye(4, dtype=np.float64)
        rotation = pose[:3, :3]
        center = pose[:3, 3]
        inv[:3, :3] = rotation.T
        inv[:3, 3] = -(rotation.T @ center)
        return inv


def _fallback_principal_point(size: int, mode: MissingPrincipalPoint) -> float:
    if mode == "half_extent":
        return float(size) / 2.0
    if mode == "pixel_center":
        return (float(size) - 1.0) * 0.5
    raise ValueError(f"unsupported missing principal point mode: {mode}")


def load_camera_records(
    cameras_json: str | Path,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    missing_principal_point: MissingPrincipalPoint = "half_extent",
) -> dict[str, CameraRecord]:
    if (target_width is None) != (target_height is None):
        raise ValueError("target_width and target_height must be provided together")
    path = Path(cameras_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"cameras.json must contain a list: {path}")
    records: dict[str, CameraRecord] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        image_id = str(row.get("img_name") or row.get("image_id") or "")
        if not image_id:
            continue
        source_width = int(row["width"])
        source_height = int(row["height"])
        width = int(target_width if target_width is not None else source_width)
        height = int(target_height if target_height is not None else source_height)
        sx = float(width) / max(float(source_width), 1.0)
        sy = float(height) / max(float(source_height), 1.0)
        source_fx = float(row["fx"])
        source_fy = float(row.get("fy", row["fx"]))
        source_cx = float(row["cx"]) if "cx" in row else _fallback_principal_point(source_width, missing_principal_point)
        source_cy = float(row["cy"]) if "cy" in row else _fallback_principal_point(source_height, missing_principal_point)
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=source_fx * sx,
            fy=source_fy * sy,
            cx=source_cx * sx,
            cy=source_cy * sy,
        )
        pose_c2w = None
        if "rotation" in row and "position" in row:
            pose_c2w = np.eye(4, dtype=np.float64)
            pose_c2w[:3, :3] = np.asarray(row["rotation"], dtype=np.float64).reshape(3, 3)
            pose_c2w[:3, 3] = np.asarray(row["position"], dtype=np.float64).reshape(3)
        records[image_id] = CameraRecord(image_id=image_id, intrinsics=intrinsics, pose_c2w=pose_c2w)
    return records
