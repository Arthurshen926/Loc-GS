from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class SparseGateComparison:
    scene: str
    split_name: str
    baseline_metrics_path: str
    candidate_metrics_path: str
    dense_target_cm: float
    baseline_metrics: Mapping[str, Any]
    candidate_metrics: Mapping[str, Any]
    candidate_artifact: CachedCandidateArtifact

    def build_metrics_summary(self) -> dict[str, object]:
        baseline_te = float(self.baseline_metrics["median_te_cm"])
        candidate_te = float(self.candidate_metrics["median_te_cm"])
        dense_target = float(self.dense_target_cm)
        delta = candidate_te - baseline_te
        candidate_metric_source = str(self.candidate_metrics.get("schema_version", "unknown"))
        candidate_pose_metric_status = self.candidate_metrics.get("pose_metric_status")
        baseline_split = self.baseline_metrics.get("split_name")
        candidate_split = self.candidate_metrics.get("split_name")
        baseline_query_count = _maybe_int(self.baseline_metrics.get("query_count"))
        candidate_query_count = _maybe_int(self.candidate_metrics.get("query_count"))
        comparable_split = not baseline_split or not candidate_split or str(baseline_split) == str(candidate_split)
        comparable_count = (
            baseline_query_count is None
            or candidate_query_count is None
            or int(baseline_query_count) == int(candidate_query_count)
        )
        if not comparable_split or not comparable_count:
            status = "diagnostic_split_or_query_mismatch"
        elif candidate_metric_source == "internal_sparse_smoke_metrics_v1" and candidate_pose_metric_status != "verified":
            status = "diagnostic_pose_frame_unverified"
        elif candidate_te <= dense_target and candidate_te <= baseline_te:
            status = "pass"
        elif candidate_te < baseline_te:
            status = "improved_not_target"
        else:
            status = "regression"
        artifact_summary = self.candidate_artifact.summarize_candidate_availability()
        return {
            "schema_version": "internal_sparse_train_dev_gate_v1",
            "scene": self.scene,
            "split_name": self.split_name,
            "sparse_gate_status": status,
            "dense_target_median_te_cm": dense_target,
            "baseline_median_te_cm": baseline_te,
            "candidate_median_te_cm": candidate_te,
            "delta_median_te_cm": delta,
            "relative_median_te_improvement": float((baseline_te - candidate_te) / baseline_te)
            if baseline_te > 0
            else 0.0,
            "target_gap_cm": float(candidate_te - dense_target),
            "baseline_median_re_deg": _maybe_float(self.baseline_metrics.get("median_re_deg")),
            "candidate_median_re_deg": _maybe_float(self.candidate_metrics.get("median_re_deg")),
            "baseline_split_name": None if baseline_split is None else str(baseline_split),
            "candidate_split_name": None if candidate_split is None else str(candidate_split),
            "baseline_query_count": baseline_query_count,
            "candidate_query_count": candidate_query_count,
            "baseline_recall_10cm_5d": _maybe_float(self.baseline_metrics.get("recall_10cm_5d")),
            "candidate_recall_10cm_5d": _maybe_float(self.candidate_metrics.get("recall_10cm_5d")),
            "candidate_metric_source": candidate_metric_source,
            "candidate_pose_metric_status": candidate_pose_metric_status,
            "candidate_artifact": artifact_summary,
            "baseline_metrics_path": self.baseline_metrics_path,
            "candidate_metrics_path": self.candidate_metrics_path,
            "candidate_artifact_path": self.candidate_artifact.source_path,
        }


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def load_metrics_summary(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metrics summary must be a JSON object: {path}")
    return data


def build_sparse_gate_comparison(
    *,
    scene: str,
    split_name: str,
    baseline_metrics_path: str | Path,
    candidate_metrics_path: str | Path,
    dense_target_cm: float,
    candidate_artifact: CachedCandidateArtifact,
) -> SparseGateComparison:
    split = reject_test_split(split_name, purpose="internal sparse gate")
    baseline_metrics = load_metrics_summary(baseline_metrics_path)
    candidate_metrics = load_metrics_summary(candidate_metrics_path)
    for label, metrics in (("baseline", baseline_metrics), ("candidate", candidate_metrics)):
        metric_split = metrics.get("split_name")
        if metric_split:
            reject_test_split(str(metric_split), purpose=f"internal sparse gate {label} metrics")
    return SparseGateComparison(
        scene=str(scene),
        split_name=split,
        baseline_metrics_path=str(baseline_metrics_path),
        candidate_metrics_path=str(candidate_metrics_path),
        dense_target_cm=float(dense_target_cm),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        candidate_artifact=candidate_artifact,
    )
