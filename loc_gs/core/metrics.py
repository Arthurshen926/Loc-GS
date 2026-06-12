from __future__ import annotations

import numpy as np


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return -rotation.T @ translation


def pose_error_cm_deg(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred_w2c, dtype=np.float64).reshape(4, 4)
    gt = np.asarray(gt_w2c, dtype=np.float64).reshape(4, 4)
    te_cm = float(np.linalg.norm(_camera_center_from_w2c(pred) - _camera_center_from_w2c(gt)) * 100.0)
    relative = pred[:3, :3] @ gt[:3, :3].T
    trace = float(np.trace(relative))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    re_deg = float(np.degrees(np.arccos(cos_theta)))
    return te_cm, re_deg
