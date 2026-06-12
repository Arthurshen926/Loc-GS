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
    candidate_coverage_metrics: Mapping[str, Any] | None = None
    candidate_scorer_metrics: Mapping[str, Any] | None = None

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
        coverage_metrics = self.candidate_coverage_metrics
        coverage_split = None if coverage_metrics is None else coverage_metrics.get("split_name")
        coverage_comparable_split = (
            coverage_metrics is None
            or coverage_split is None
            or str(coverage_split) == str(self.split_name)
            or not candidate_split
            or str(coverage_split) == str(candidate_split)
        )
        coverage_complete = True if coverage_metrics is None else bool(coverage_metrics.get("complete_coverage"))
        if not comparable_split or not comparable_count:
            status = "diagnostic_split_or_query_mismatch"
        elif not coverage_comparable_split:
            status = "diagnostic_split_or_query_mismatch"
        elif not coverage_complete:
            status = "blocked_candidate_coverage_incomplete"
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
            "candidate_coverage": _compact_coverage_metrics(coverage_metrics),
            "candidate_rerank_diagnostic": _compact_rerank_diagnostic(self.candidate_metrics),
            "candidate_selected_set_diagnostic": _compact_selected_set_diagnostic(self.candidate_metrics),
            "candidate_inlier_set_diagnostic": _compact_inlier_set_diagnostic(self.candidate_metrics),
            "candidate_post_pnp_rescore_diagnostic": _compact_post_pnp_rescore_diagnostic(self.candidate_metrics),
            "candidate_scorer_training": _compact_candidate_scorer_metrics(self.candidate_scorer_metrics),
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


def _compact_coverage_metrics(metrics: Mapping[str, Any] | None) -> dict[str, object] | None:
    if metrics is None:
        return None
    keys = (
        "schema_version",
        "scene",
        "split_name",
        "requested_query_count",
        "covered_query_count",
        "missing_query_count",
        "coverage_ratio",
        "complete_coverage",
        "coverage_status",
        "missing_query_ids_preview",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _compact_candidate_scorer_metrics(metrics: Mapping[str, Any] | None) -> dict[str, object] | None:
    if metrics is None:
        return None
    keys = (
        "schema_version",
        "feature_materialization",
        "feature_input_policy",
        "paper_safe_sparse_inference",
        "sample_count",
        "label_count",
        "native_top1_correct",
        "trained_top1_correct",
        "dense_teacher_sample_count",
    )
    out = {key: metrics[key] for key in keys if key in metrics}
    native = _maybe_int(metrics.get("native_top1_correct"))
    trained = _maybe_int(metrics.get("trained_top1_correct"))
    if native is not None and trained is not None:
        out["top1_gain"] = int(trained - native)
        out["relative_top1_gain"] = float((trained - native) / native) if native > 0 else 0.0
    return out


def _compact_rerank_diagnostic(metrics: Mapping[str, Any]) -> dict[str, object] | None:
    if not bool(metrics.get("rerank_diagnostic_enabled")):
        return None
    keys = (
        "rerank_diagnostic_enabled",
        "rerank_diagnostic_query_count",
        "native_top1_correct",
        "reranked_top1_correct",
        "reranked_top1_gain",
        "reranked_top1_changed_count",
        "reranked_topk_available",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _compact_selected_set_diagnostic(metrics: Mapping[str, Any]) -> dict[str, object] | None:
    keys = (
        "selected_geometric_correct_count_median",
        "selected_geometric_correct_ratio_median",
        "selected_keypoint_bbox_area_fraction_median",
        "selected_depth_range_m_median",
    )
    out = {key: metrics[key] for key in keys if key in metrics}
    return out or None


def _compact_inlier_set_diagnostic(metrics: Mapping[str, Any]) -> dict[str, object] | None:
    keys = (
        "inlier_geometric_correct_count_median",
        "inlier_geometric_correct_ratio_median",
        "inlier_keypoint_bbox_area_fraction_median",
        "inlier_depth_range_m_median",
    )
    out = {key: metrics[key] for key in keys if key in metrics}
    return out or None


def _compact_post_pnp_rescore_diagnostic(metrics: Mapping[str, Any]) -> dict[str, object] | None:
    if not bool(metrics.get("post_pnp_candidate_rescore_enabled")):
        return None
    keys = (
        "post_pnp_candidate_rescore_enabled",
        "post_pnp_rescore_changed_count_median",
        "post_pnp_rescore_corrected_count_median",
        "post_pnp_rescore_worsened_count_median",
        "post_pnp_rescore_correct_delta_median",
    )
    return {key: metrics[key] for key in keys if key in metrics}


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
    candidate_coverage_metrics_path: str | Path | None = None,
    candidate_scorer_metrics_path: str | Path | None = None,
) -> SparseGateComparison:
    split = reject_test_split(split_name, purpose="internal sparse gate")
    baseline_metrics = load_metrics_summary(baseline_metrics_path)
    candidate_metrics = load_metrics_summary(candidate_metrics_path)
    candidate_coverage_metrics = (
        load_metrics_summary(candidate_coverage_metrics_path) if candidate_coverage_metrics_path is not None else None
    )
    candidate_scorer_metrics = (
        load_metrics_summary(candidate_scorer_metrics_path) if candidate_scorer_metrics_path is not None else None
    )
    for label, metrics in (("baseline", baseline_metrics), ("candidate", candidate_metrics)):
        metric_split = metrics.get("split_name")
        if metric_split:
            reject_test_split(str(metric_split), purpose=f"internal sparse gate {label} metrics")
    if candidate_coverage_metrics is not None and candidate_coverage_metrics.get("split_name"):
        reject_test_split(
            str(candidate_coverage_metrics["split_name"]),
            purpose="internal sparse gate candidate coverage metrics",
        )
    if candidate_scorer_metrics is not None and candidate_scorer_metrics.get("split_name"):
        reject_test_split(
            str(candidate_scorer_metrics["split_name"]),
            purpose="internal sparse gate candidate scorer metrics",
        )
    return SparseGateComparison(
        scene=str(scene),
        split_name=split,
        baseline_metrics_path=str(baseline_metrics_path),
        candidate_metrics_path=str(candidate_metrics_path),
        dense_target_cm=float(dense_target_cm),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        candidate_artifact=candidate_artifact,
        candidate_coverage_metrics=candidate_coverage_metrics,
        candidate_scorer_metrics=candidate_scorer_metrics,
    )
