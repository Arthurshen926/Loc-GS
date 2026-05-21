from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _query_id(row: dict[str, Any], index: int) -> str:
    value = row.get("query_id")
    if value is not None:
        return str(value)
    return f"query_{index:06d}"


def _stage_metric(row: dict[str, Any], stage: str, metric: str) -> float | None:
    aliases = (
        f"{stage}_{metric}",
        f"{stage}_{metric.upper()}",
        f"{stage}_{metric.replace('_cm', '').upper()}",
        f"{stage}_{metric.replace('_deg', '').upper()}",
    )
    for key in aliases:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    nested = row.get(stage)
    if isinstance(nested, dict):
        for key in (metric, metric.upper(), metric.replace("_cm", "").upper(), metric.replace("_deg", "").upper()):
            value = _as_float(nested.get(key))
            if value is not None:
                return value
    return None


def _recall(rows: Iterable[tuple[dict[str, Any], dict[str, Any]]], *, side: int, stage: str, te_cm: float, re_deg: float) -> float:
    total = 0
    ok = 0
    for pair in rows:
        row = pair[side]
        te = _stage_metric(row, stage, "te_cm")
        re = _stage_metric(row, stage, "re_deg")
        if te is None or re is None:
            continue
        total += 1
        if te <= te_cm and re <= re_deg:
            ok += 1
    return float(ok / total) if total else 0.0


def _summarize_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, stage: str) -> dict[str, Any]:
    deltas: list[float] = []
    base_te: list[float] = []
    cand_te: list[float] = []
    for base, cand in pairs:
        b = _stage_metric(base, stage, "te_cm")
        c = _stage_metric(cand, stage, "te_cm")
        if b is None or c is None:
            continue
        base_te.append(b)
        cand_te.append(c)
        deltas.append(c - b)
    if not deltas:
        return {
            "query_count": 0,
            "mean_delta_te_cm": 0.0,
            "median_delta_te_cm": 0.0,
            "baseline_mean_te_cm": 0.0,
            "candidate_mean_te_cm": 0.0,
            "recall_5cm_5deg_delta": 0.0,
            "recall_2cm_2deg_delta": 0.0,
        }
    r5_delta = _recall(pairs, side=1, stage=stage, te_cm=5.0, re_deg=5.0) - _recall(
        pairs, side=0, stage=stage, te_cm=5.0, re_deg=5.0
    )
    r2_delta = _recall(pairs, side=1, stage=stage, te_cm=2.0, re_deg=2.0) - _recall(
        pairs, side=0, stage=stage, te_cm=2.0, re_deg=2.0
    )
    return {
        "query_count": int(len(deltas)),
        "mean_delta_te_cm": float(sum(deltas) / len(deltas)),
        "median_delta_te_cm": float(median(deltas)),
        "baseline_mean_te_cm": float(sum(base_te) / len(base_te)),
        "candidate_mean_te_cm": float(sum(cand_te) / len(cand_te)),
        "recall_5cm_5deg_delta": float(r5_delta),
        "recall_2cm_2deg_delta": float(r2_delta),
    }


def _fraction_count(total: int, fraction: float) -> int:
    if total <= 0:
        return 0
    return max(1, min(total, int(math.ceil(total * float(fraction)))))


def partitioned_stage_deltas(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    stage: str = "dense",
    fractions: tuple[float, ...] = (0.1, 0.2, 0.5),
    easy_fraction: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Compare candidate against baseline on hard/easy partitions defined by baseline error.

    Partitions are formed only from the baseline run, so this is suitable for
    train/self-map diagnostics without peeking at candidate success cases.
    """

    cand_by_id = {_query_id(row, index): row for index, row in enumerate(candidate_rows)}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, base in enumerate(baseline_rows):
        query_id = _query_id(base, index)
        cand = cand_by_id.get(query_id)
        if cand is None:
            continue
        if _stage_metric(base, stage, "te_cm") is None or _stage_metric(cand, stage, "te_cm") is None:
            continue
        pairs.append((base, cand))
    pairs.sort(key=lambda pair: _stage_metric(pair[0], stage, "te_cm") or float("-inf"), reverse=True)
    report: dict[str, dict[str, Any]] = {"all": _summarize_pairs(pairs, stage=stage)}
    for fraction in fractions:
        count = _fraction_count(len(pairs), fraction)
        key = f"hardest_{int(round(float(fraction) * 100))}pct"
        report[key] = _summarize_pairs(pairs[:count], stage=stage)
    easy_count = _fraction_count(len(pairs), easy_fraction)
    report[f"easiest_{int(round(float(easy_fraction) * 100))}pct"] = _summarize_pairs(
        list(reversed(pairs))[:easy_count],
        stage=stage,
    )
    return report
