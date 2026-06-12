from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class SolverFeedbackConfig:
    min_dense_improvement_cm: float = 2.0
    catastrophic_sparse_cm: float = 50.0
    weight_clip_cm: float = 20.0


@dataclass(frozen=True)
class SolverFeedbackLabel:
    scene: str
    split_name: str
    query_id: str
    sparse_te_cm: float
    dense_te_cm: float
    sparse_re_deg: float | None
    dense_re_deg: float | None
    sparse_inliers: int | None
    translation_improvement_cm: float
    rotation_improvement_deg: float | None
    dense_helped: bool
    sparse_catastrophic: bool
    distill_weight: float

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("image_name") or row.get("query_id") or row.get("image_id") or "")


def _float_field(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] is not None:
            return float(row[name])
    return None


def _int_field(row: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        if name in row and row[name] is not None:
            return int(row[name])
    return None


def _index_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        qid = _query_id(row)
        if qid:
            indexed[qid] = row
    return indexed


def build_solver_feedback_labels(
    *,
    scene: str,
    split_name: str,
    sparse_rows: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    cfg: SolverFeedbackConfig | None = None,
) -> list[SolverFeedbackLabel]:
    if cfg is None:
        cfg = SolverFeedbackConfig()
    split = reject_test_split(split_name, purpose="solver feedback teacher labels")
    dense_by_query = _index_rows(dense_rows)
    labels: list[SolverFeedbackLabel] = []
    for sparse in sparse_rows:
        qid = _query_id(sparse)
        if not qid or qid not in dense_by_query:
            continue
        dense = dense_by_query[qid]
        sparse_te = _float_field(sparse, "sparse_te_cm", "sparse_te", "te_cm", "median_te_cm")
        dense_te = _float_field(
            dense,
            "dense_te_cm",
            "sparse_conditioned_dense_te_cm",
            "base_dense_te_cm",
            "dense_te",
            "te_cm",
            "sparse_te_cm",
            "sparse_te",
            "median_te_cm",
        )
        if sparse_te is None or dense_te is None:
            continue
        sparse_re = _float_field(sparse, "sparse_re_deg", "sparse_ae", "re_deg", "median_re_deg")
        dense_re = _float_field(
            dense,
            "dense_re_deg",
            "sparse_conditioned_dense_re_deg",
            "base_dense_re_deg",
            "dense_ae",
            "re_deg",
            "sparse_re_deg",
            "sparse_ae",
            "median_re_deg",
        )
        improvement = float(sparse_te - dense_te)
        rotation_improvement = None if sparse_re is None or dense_re is None else float(sparse_re - dense_re)
        dense_helped = improvement >= float(cfg.min_dense_improvement_cm)
        sparse_catastrophic = float(sparse_te) >= float(cfg.catastrophic_sparse_cm)
        weight = max(0.0, min(float(cfg.weight_clip_cm), improvement)) / max(1.0e-9, float(cfg.weight_clip_cm))
        labels.append(
            SolverFeedbackLabel(
                scene=str(scene),
                split_name=split,
                query_id=qid,
                sparse_te_cm=float(sparse_te),
                dense_te_cm=float(dense_te),
                sparse_re_deg=sparse_re,
                dense_re_deg=dense_re,
                sparse_inliers=_int_field(sparse, "sparse_inliers", "inlier_count"),
                translation_improvement_cm=improvement,
                rotation_improvement_deg=rotation_improvement,
                dense_helped=bool(dense_helped),
                sparse_catastrophic=bool(sparse_catastrophic),
                distill_weight=float(weight if dense_helped else 0.0),
            )
        )
    return labels


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"] if isinstance(row, Mapping)]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    raise ValueError(f"result file must be a list or an object with rows: {path}")


def summarize_solver_feedback_labels(labels: Sequence[SolverFeedbackLabel]) -> dict[str, object]:
    helped = [label for label in labels if label.dense_helped]
    catastrophic = [label for label in labels if label.sparse_catastrophic]
    return {
        "schema_version": "internal_solver_feedback_label_summary_v1",
        "label_count": int(len(labels)),
        "dense_helped_count": int(len(helped)),
        "sparse_catastrophic_count": int(len(catastrophic)),
        "mean_distill_weight": float(sum(label.distill_weight for label in labels) / max(1, len(labels))),
    }
