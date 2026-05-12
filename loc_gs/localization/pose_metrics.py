from __future__ import annotations

from collections.abc import Iterable

import numpy as np


POSE_RECALL_THRESHOLDS: tuple[tuple[str, float, float], ...] = (
    ("recall_5m_10d", 500.0, 10.0),
    ("recall_2m_5d", 200.0, 5.0),
    ("recall_1m_5d", 100.0, 5.0),
    ("recall_50cm_5d", 50.0, 5.0),
    ("recall_25cm_2d", 25.0, 2.0),
    ("recall_10cm_5d", 10.0, 5.0),
    ("recall_5cm_5d", 5.0, 5.0),
    ("recall_2cm_2d", 2.0, 2.0),
)


def pose_recall_metrics(te_cm: Iterable[float], ae_deg: Iterable[float]) -> dict[str, float]:
    te = np.asarray(list(te_cm), dtype=np.float64)
    ae = np.asarray(list(ae_deg), dtype=np.float64)
    if len(te) == 0 or len(ae) != len(te):
        return {key: 0.0 for key, _te, _ae in POSE_RECALL_THRESHOLDS}
    return {
        key: float(((te <= te_thr) & (ae <= ae_thr)).mean())
        for key, te_thr, ae_thr in POSE_RECALL_THRESHOLDS
    }


def pose_error_summary(
    te_cm: Iterable[float],
    ae_deg: Iterable[float],
    inliers: Iterable[int] = (),
) -> dict[str, float]:
    te_values = list(te_cm)
    ae_values = list(ae_deg)
    inlier_values = list(inliers)
    te = np.asarray(te_values, dtype=np.float64)
    ae = np.asarray(ae_values, dtype=np.float64)
    inl = np.asarray(inlier_values, dtype=np.float64)
    valid = len(te) > 0 and len(ae) == len(te)
    out = {
        "median_te": float(np.median(te)) if valid else float("inf"),
        "median_ae": float(np.median(ae)) if valid else float("inf"),
    }
    out.update(pose_recall_metrics(te_values if valid else [], ae_values if valid else []))
    out["avg_inliers"] = float(inl.mean()) if len(inl) else 0.0
    return out


def recall_metric_key(metric: str) -> str:
    aliases = {
        "r5": "recall_5cm_5d",
        "r2": "recall_2cm_2d",
        "r10": "recall_10cm_5d",
        "r25": "recall_25cm_2d",
        "r50": "recall_50cm_5d",
        "r1m": "recall_1m_5d",
        "r2m": "recall_2m_5d",
        "r5m": "recall_5m_10d",
    }
    return aliases.get(metric, metric)
