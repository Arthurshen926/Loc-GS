from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.core.geometry import project_points_w2c
from loc_gs.core.pnp import OpenCvPnPConfig, solve_pnp_ransac
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.rerank import (
    CandidateRerankConfig,
    rerank_candidate_rows,
    summarize_candidate_availability,
)
from loc_gs.sparse.sparse_lgcv import filter_correspondences_by_reprojection


@dataclass(frozen=True)
class SparsePipelineCandidate:
    keypoint_index: int
    landmark_id: int
    point3d: Sequence[float]
    native_score: float
    solver_score: float = 0.0
    geometric_correct: bool | None = None

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "keypoint_index": int(self.keypoint_index),
            "landmark_id": int(self.landmark_id),
            "point3d": [float(v) for v in self.point3d],
            "native_score": float(self.native_score),
            "solver_score": float(self.solver_score),
        }
        if self.geometric_correct is not None:
            row["geometric_correct"] = bool(self.geometric_correct)
        return row


@dataclass(frozen=True)
class SparseLocalizationInput:
    scene: str
    split_name: str
    query_id: str
    intrinsics: CameraIntrinsics
    keypoints_xy: Sequence[Sequence[float]]
    candidates_by_keypoint: Sequence[Sequence[SparsePipelineCandidate]]

    def validate(self) -> None:
        if len(self.keypoints_xy) != len(self.candidates_by_keypoint):
            raise ValueError("keypoints_xy and candidates_by_keypoint must have the same length")
        for row_idx, row in enumerate(self.candidates_by_keypoint):
            if not row:
                raise ValueError(f"candidate row {row_idx} is empty")
            for candidate in row:
                if int(candidate.keypoint_index) != row_idx:
                    raise ValueError(f"candidate row {row_idx} contains mismatched keypoint_index")


@dataclass(frozen=True)
class SparseLocalizationConfig:
    rerank_prefix_fraction: float = 0.75
    solver_weight: float = 1.0
    native_weight: float = 1.0
    reprojection_error_px: float = 8.0
    pnp_iterations: int = 10000
    min_correspondences: int = 4
    pnp_method: str = "epnp"
    refine_with_inliers: bool = False
    second_pnp_enabled: bool = False
    second_pnp_method: str = "iterative"
    lgcv_reprojection_error_px: float = 4.0
    post_pnp_candidate_rescore: bool = False
    post_pnp_reprojection_weight: float = 1.0
    post_pnp_reprojection_score_scale_px: float = 4.0
    post_pnp_rescore_min_improvement_px: float = 2.0
    post_pnp_rescore_max_residual_px: float = 4.0
    post_pnp_rescore_only_initial_outliers: bool = True
    conflict_edges: Mapping[str, float] | None = None
    set_conflict_penalty: float = 0.0


@dataclass(frozen=True)
class SparseLocalizationResult:
    success: bool
    pose_w2c: np.ndarray | None
    inlier_mask: np.ndarray
    selected_landmark_ids: list[int]
    selected_keypoint_indices: list[int]
    availability_summary: dict[str, int]
    pnp_stage_count: int = 0
    initial_inlier_count: int = 0
    lgcv_keep_count: int | None = None
    post_pnp_rescore_changed_count: int = 0
    post_pnp_rescore_corrected_count: int = 0
    post_pnp_rescore_worsened_count: int = 0
    post_pnp_rescore_correct_delta: int = 0
    set_conflict_rerank_changed_count: int = 0
    pnp_method: str = "epnp"
    second_pnp_method: str | None = None
    selected_set_diagnostics: dict[str, float | int] | None = None
    inlier_set_diagnostics: dict[str, float | int] | None = None

    @property
    def inlier_count(self) -> int:
        return int(np.asarray(self.inlier_mask, dtype=bool).sum())


def _rank_candidates(
    candidates: Sequence[SparsePipelineCandidate],
    cfg: SparseLocalizationConfig,
) -> list[dict[str, object]]:
    rows = [candidate.to_row() for candidate in candidates]
    return rerank_candidate_rows(
        rows,
        CandidateRerankConfig(
            prefix_fraction=float(cfg.rerank_prefix_fraction),
            solver_weight=float(cfg.solver_weight),
            native_weight=float(cfg.native_weight),
        ),
    )


def _match_scores(rows: Sequence[dict[str, object]], cfg: SparseLocalizationConfig) -> np.ndarray:
    return np.asarray(
        [
            float(row.get("native_score", 0.0)) * float(cfg.native_weight)
            + float(row.get("solver_score", 0.0)) * float(cfg.solver_weight)
            for row in rows
        ],
        dtype=np.float64,
    )


def _edge_key(first: int | str, second: int | str) -> str:
    left = str(first)
    right = str(second)
    return "::".join(sorted((left, right), key=lambda value: int(value) if value.lstrip("-").isdigit() else value))


def _conflict_weight(cfg: SparseLocalizationConfig, first: int | str, second: int | str) -> float:
    if not cfg.conflict_edges:
        return 0.0
    return float(cfg.conflict_edges.get(_edge_key(first, second), 0.0))


def _select_with_set_conflict(
    ranked: Sequence[dict[str, object]],
    selected_landmark_ids: Sequence[int],
    cfg: SparseLocalizationConfig,
) -> tuple[dict[str, object], bool]:
    if not ranked:
        raise ValueError("ranked candidate rows must not be empty")
    if not selected_landmark_ids or not cfg.conflict_edges or float(cfg.set_conflict_penalty) <= 0.0:
        return dict(ranked[0]), False
    penalty = float(cfg.set_conflict_penalty)
    scored: list[tuple[float, int, dict[str, object]]] = []
    for idx, row in enumerate(ranked):
        landmark_id = int(row["landmark_id"])
        conflict = sum(_conflict_weight(cfg, landmark_id, selected_id) for selected_id in selected_landmark_ids)
        base_score = (
            float(row.get("native_score", 0.0)) * float(cfg.native_weight)
            + float(row.get("solver_score", 0.0)) * float(cfg.solver_weight)
        )
        enriched = dict(row)
        enriched["set_conflict_penalty"] = float(penalty * conflict)
        enriched["set_conflict_runtime_score"] = float(base_score - penalty * conflict)
        scored.append((float(base_score - penalty * conflict), -idx, enriched))
    selected = max(scored, key=lambda item: (item[0], item[1]))[2]
    changed = int(selected["landmark_id"]) != int(ranked[0]["landmark_id"])
    return selected, bool(changed)


def _row_set_diagnostics(
    data: SparseLocalizationInput,
    rows: Sequence[dict[str, object]],
    *,
    prefix: str,
) -> dict[str, float | int]:
    row_count = int(len(rows))
    correct_values = [
        row.get("geometric_correct")
        for row in rows
        if row.get("geometric_correct") is not None
    ]
    correct_count = int(sum(1 for value in correct_values if bool(value)))
    keypoints = np.asarray(data.keypoints_xy, dtype=np.float64).reshape(-1, 2)
    selected_indices = [
        int(row.get("keypoint_index", idx))
        for idx, row in enumerate(rows)
        if 0 <= int(row.get("keypoint_index", idx)) < keypoints.shape[0]
    ]
    bbox_area_fraction = 0.0
    if selected_indices:
        selected_xy = keypoints[np.asarray(selected_indices, dtype=np.int64)]
        min_xy = selected_xy.min(axis=0)
        max_xy = selected_xy.max(axis=0)
        area = max(0.0, float(max_xy[0] - min_xy[0])) * max(0.0, float(max_xy[1] - min_xy[1]))
        denom = max(1.0, float(data.intrinsics.width) * float(data.intrinsics.height))
        bbox_area_fraction = float(area / denom)
    points = np.asarray([row.get("point3d", [np.nan, np.nan, np.nan]) for row in rows], dtype=np.float64)
    depth_range = 0.0
    if points.size and points.ndim == 2 and points.shape[1] >= 3:
        z = points[:, 2]
        finite = z[np.isfinite(z)]
        if finite.size:
            depth_range = float(finite.max() - finite.min())
    return {
        f"{prefix}_count": row_count,
        f"{prefix}_geometric_correct_count": correct_count,
        f"{prefix}_geometric_label_count": int(len(correct_values)),
        f"{prefix}_geometric_correct_ratio": float(correct_count / max(1, len(correct_values))),
        f"{prefix}_keypoint_bbox_area_fraction": float(bbox_area_fraction),
        f"{prefix}_depth_range_m": float(depth_range),
    }


def _selected_set_diagnostics(
    data: SparseLocalizationInput,
    selected_rows: Sequence[dict[str, object]],
) -> dict[str, float | int]:
    return _row_set_diagnostics(data, selected_rows, prefix="selected")


def _inlier_set_diagnostics(
    data: SparseLocalizationInput,
    selected_rows: Sequence[dict[str, object]],
    inlier_mask: np.ndarray,
) -> dict[str, float | int]:
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    inlier_rows = [row for row, keep in zip(selected_rows, mask) if bool(keep)]
    return _row_set_diagnostics(data, inlier_rows, prefix="inlier")


def _post_pnp_rescore_rows(
    data: SparseLocalizationInput,
    pose_w2c: np.ndarray,
    cfg: SparseLocalizationConfig,
    *,
    eligible_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, object]], int]:
    keypoints = np.asarray(data.keypoints_xy, dtype=np.float64).reshape(-1, 2)
    selected_rows: list[dict[str, object]] = []
    changed_count = 0
    scale = max(1.0e-9, float(cfg.post_pnp_reprojection_score_scale_px))
    for row_idx, candidates in enumerate(data.candidates_by_keypoint):
        ranked = _rank_candidates(candidates, cfg)
        if eligible_mask is not None and row_idx < eligible_mask.shape[0] and not bool(eligible_mask[row_idx]):
            selected_rows.append(dict(ranked[0]))
            continue
        points = np.asarray([row["point3d"] for row in ranked], dtype=np.float64).reshape(-1, 3)
        projected_xy, valid = project_points_w2c(points, pose_w2c, data.intrinsics)
        residuals = np.linalg.norm(projected_xy - keypoints[row_idx].reshape(1, 2), axis=1)
        rescored: list[tuple[float, int, dict[str, object]]] = []
        for candidate_idx, row in enumerate(ranked):
            residual = (
                float(residuals[candidate_idx])
                if bool(valid[candidate_idx]) and np.isfinite(residuals[candidate_idx])
                else float("inf")
            )
            base_score = (
                float(row.get("native_score", 0.0)) * float(cfg.native_weight)
                + float(row.get("solver_score", 0.0)) * float(cfg.solver_weight)
            )
            runtime_score = (
                base_score - float(cfg.post_pnp_reprojection_weight) * (residual / scale)
                if np.isfinite(residual)
                else -1.0e12
            )
            enriched = dict(row)
            enriched["post_pnp_reprojection_error_px"] = residual
            enriched["post_pnp_runtime_score"] = runtime_score
            enriched["post_pnp_rescore_candidate_index"] = candidate_idx
            rescored.append((runtime_score, -candidate_idx, enriched))
        best = max(rescored, key=lambda item: (item[0], item[1]))[2]
        initial = dict(ranked[0])
        initial_residual = (
            float(residuals[0]) if bool(valid[0]) and np.isfinite(residuals[0]) else float("inf")
        )
        initial["post_pnp_reprojection_error_px"] = initial_residual
        initial["post_pnp_runtime_score"] = rescored[0][0]
        initial["post_pnp_rescore_candidate_index"] = 0
        best_residual = float(best["post_pnp_reprojection_error_px"])
        improved_enough = initial_residual - best_residual >= float(cfg.post_pnp_rescore_min_improvement_px)
        low_residual = best_residual <= float(cfg.post_pnp_rescore_max_residual_px)
        if not (improved_enough and low_residual):
            best = initial
        if int(best["landmark_id"]) != int(ranked[0]["landmark_id"]):
            changed_count += 1
        selected_rows.append(best)
    return selected_rows, changed_count


def _post_pnp_rescore_label_flow(
    before_rows: Sequence[dict[str, object]],
    after_rows: Sequence[dict[str, object]],
) -> dict[str, int]:
    corrected_count = 0
    worsened_count = 0
    before_correct_count = 0
    after_correct_count = 0
    for before, after in zip(before_rows, after_rows):
        before_value = before.get("geometric_correct")
        after_value = after.get("geometric_correct")
        before_known = before_value is not None
        after_known = after_value is not None
        if before_known and bool(before_value):
            before_correct_count += 1
        if after_known and bool(after_value):
            after_correct_count += 1
        if not (before_known and after_known):
            continue
        before_correct = bool(before_value)
        after_correct = bool(after_value)
        if not before_correct and after_correct:
            corrected_count += 1
        elif before_correct and not after_correct:
            worsened_count += 1
    return {
        "corrected_count": int(corrected_count),
        "worsened_count": int(worsened_count),
        "correct_delta": int(after_correct_count - before_correct_count),
    }


def run_sparse_localization(
    data: SparseLocalizationInput,
    cfg: SparseLocalizationConfig | None = None,
) -> SparseLocalizationResult:
    if cfg is None:
        cfg = SparseLocalizationConfig()
    reject_test_split(data.split_name, purpose="internal sparse localization")
    data.validate()

    rows_by_query = [[candidate.to_row() for candidate in row] for row in data.candidates_by_keypoint]
    availability = summarize_candidate_availability(rows_by_query)

    selected_rows: list[dict[str, object]] = []
    selected_keypoint_indices: list[int] = []
    selected_landmark_ids: list[int] = []
    set_conflict_changed_count = 0
    for row_idx, candidates in enumerate(data.candidates_by_keypoint):
        ranked = _rank_candidates(candidates, cfg)
        selected, conflict_changed = _select_with_set_conflict(ranked, selected_landmark_ids, cfg)
        set_conflict_changed_count += int(conflict_changed)
        selected_rows.append(selected)
        selected_keypoint_indices.append(row_idx)
        selected_landmark_ids.append(int(selected["landmark_id"]))

    points = np.asarray([row["point3d"] for row in selected_rows], dtype=np.float64).reshape(-1, 3)
    keypoints_xy = np.asarray(data.keypoints_xy, dtype=np.float64).reshape(-1, 2)
    scores = _match_scores(selected_rows, cfg)
    selected_diagnostics = _selected_set_diagnostics(data, selected_rows)
    if points.shape[0] < int(cfg.min_correspondences):
        return SparseLocalizationResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros(points.shape[0], dtype=bool),
            selected_landmark_ids=selected_landmark_ids,
            selected_keypoint_indices=selected_keypoint_indices,
            availability_summary=availability,
            pnp_method=str(cfg.pnp_method),
            selected_set_diagnostics=selected_diagnostics,
            inlier_set_diagnostics=_inlier_set_diagnostics(data, selected_rows, np.zeros(points.shape[0], dtype=bool)),
            set_conflict_rerank_changed_count=int(set_conflict_changed_count),
        )

    pnp = solve_pnp_ransac(
        points,
        keypoints_xy,
        data.intrinsics,
        OpenCvPnPConfig(
            reprojection_error_px=float(cfg.reprojection_error_px),
            iterations=int(cfg.pnp_iterations),
            method=str(cfg.pnp_method),
            refine_with_inliers=bool(cfg.refine_with_inliers),
        ),
        match_scores=scores,
    )
    final_pnp = pnp
    final_mask = np.asarray(pnp.inlier_mask, dtype=bool)
    pnp_stage_count = 1 if bool(pnp.success) else 0
    lgcv_keep_count = None
    second_method = None
    post_pnp_changed_count = 0
    post_pnp_corrected_count = 0
    post_pnp_worsened_count = 0
    post_pnp_correct_delta = 0
    if bool(pnp.success) and bool(cfg.second_pnp_enabled) and pnp.pose_w2c is not None:
        if bool(cfg.post_pnp_candidate_rescore):
            eligible = None
            if bool(cfg.post_pnp_rescore_only_initial_outliers):
                eligible = ~np.asarray(pnp.inlier_mask, dtype=bool)
            pre_rescore_rows = [dict(row) for row in selected_rows]
            selected_rows, post_pnp_changed_count = _post_pnp_rescore_rows(
                data,
                pnp.pose_w2c,
                cfg,
                eligible_mask=eligible,
            )
            label_flow = _post_pnp_rescore_label_flow(pre_rescore_rows, selected_rows)
            post_pnp_corrected_count = int(label_flow["corrected_count"])
            post_pnp_worsened_count = int(label_flow["worsened_count"])
            post_pnp_correct_delta = int(label_flow["correct_delta"])
            selected_landmark_ids = [int(row["landmark_id"]) for row in selected_rows]
            points = np.asarray([row["point3d"] for row in selected_rows], dtype=np.float64).reshape(-1, 3)
            scores = _match_scores(selected_rows, cfg)
            selected_diagnostics = _selected_set_diagnostics(data, selected_rows)
        lgcv = filter_correspondences_by_reprojection(
            points,
            keypoints_xy,
            pnp.pose_w2c,
            data.intrinsics,
            threshold_px=float(cfg.lgcv_reprojection_error_px),
        )
        lgcv_keep = np.asarray(lgcv.keep_mask, dtype=bool)
        lgcv_keep_count = int(lgcv.keep_count)
        if lgcv_keep_count >= int(cfg.min_correspondences):
            second_pnp = solve_pnp_ransac(
                points[lgcv_keep],
                keypoints_xy[lgcv_keep],
                data.intrinsics,
                OpenCvPnPConfig(
                    reprojection_error_px=float(cfg.reprojection_error_px),
                    iterations=int(cfg.pnp_iterations),
                    method=str(cfg.second_pnp_method),
                    refine_with_inliers=bool(cfg.refine_with_inliers),
                ),
                match_scores=scores[lgcv_keep],
            )
            pnp_stage_count = 2
            second_method = str(cfg.second_pnp_method)
            if bool(second_pnp.success):
                mapped = np.zeros(points.shape[0], dtype=bool)
                mapped_indices = np.flatnonzero(lgcv_keep)
                mapped[mapped_indices[np.asarray(second_pnp.inlier_mask, dtype=bool)]] = True
                final_pnp = second_pnp
                final_mask = mapped
    return SparseLocalizationResult(
        success=bool(final_pnp.success),
        pose_w2c=final_pnp.pose_w2c,
        inlier_mask=final_mask,
        selected_landmark_ids=selected_landmark_ids,
        selected_keypoint_indices=selected_keypoint_indices,
        availability_summary=availability,
        pnp_stage_count=pnp_stage_count,
        initial_inlier_count=int(pnp.inlier_count),
        lgcv_keep_count=lgcv_keep_count,
        post_pnp_rescore_changed_count=int(post_pnp_changed_count),
        post_pnp_rescore_corrected_count=int(post_pnp_corrected_count),
        post_pnp_rescore_worsened_count=int(post_pnp_worsened_count),
        post_pnp_rescore_correct_delta=int(post_pnp_correct_delta),
        set_conflict_rerank_changed_count=int(set_conflict_changed_count),
        pnp_method=str(cfg.pnp_method),
        second_pnp_method=second_method,
        selected_set_diagnostics=selected_diagnostics,
        inlier_set_diagnostics=_inlier_set_diagnostics(data, selected_rows, final_mask),
    )
