from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def load_camera_records(cameras_json: str | Path) -> dict[str, CameraRecord]:
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
        width = int(row["width"])
        height = int(row["height"])
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(row["fx"]),
            fy=float(row.get("fy", row["fx"])),
            cx=float(row.get("cx", width / 2.0)),
            cy=float(row.get("cy", height / 2.0)),
        )
        pose_c2w = None
        if "rotation" in row and "position" in row:
            pose_c2w = np.eye(4, dtype=np.float64)
            pose_c2w[:3, :3] = np.asarray(row["rotation"], dtype=np.float64).reshape(3, 3)
            pose_c2w[:3, 3] = np.asarray(row["position"], dtype=np.float64).reshape(3)
        records[image_id] = CameraRecord(image_id=image_id, intrinsics=intrinsics, pose_c2w=pose_c2w)
    return records
