from __future__ import annotations

from dataclasses import dataclass

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
