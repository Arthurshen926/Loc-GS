from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

from loc_gs.core.camera import load_camera_records
from loc_gs.core.metrics import pose_error_cm_deg
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.landmarks import CacheLandmarkResolver, load_gaussian_landmark_map
from loc_gs.sparse.pipeline import SparseLocalizationConfig, run_sparse_localization
from loc_gs.sparse.pose_map_frame_audit import audit_pose_map_frame
from loc_gs.sparse.real_inputs import CachedSparseInputConfig, sparse_input_from_cached_batch
from loc_gs.students.descriptor_fusion import descriptor_fusion_score_rows, load_descriptor_fusion
from loc_gs.students.detector_student import detector_score_rows, load_detector_student
from loc_gs.students.landmark_selector import landmark_selector_score_rows, load_landmark_selector
from loc_gs.training.sparse_candidate_scorer import candidate_solver_score_rows, load_candidate_scorer


@dataclass(frozen=True)
class CachedSparseEvalConfig:
    image_width: int | None = None
    image_height: int | None = None
    missing_principal_point: str = "pixel_center"
    query_ids: Sequence[str] | None = None
    max_queries: int | None = None
    max_keypoints: int = 1024
    score_mode: str = "native"
    candidate_scorer: str | Path | None = None
    landmark_selector: str | Path | None = None
    landmark_selector_weight: float = 1.0
    descriptor_fusion: str | Path | None = None
    descriptor_fusion_weight: float = 1.0
    detector_student: str | Path | None = None
    detector_student_weight: float = 1.0
    rerank_prefix_fraction: float = 1.0
    solver_weight: float = 1.0
    native_weight: float = 1.0
    reprojection_error_px: float = 8.0
    pnp_iterations: int = 10000
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
    frame_auto_calibration: bool = True
    frame_calibration_max_rows: int = 2048
    frame_calibration_agreement_threshold_px: float = 0.05


def run_cached_sparse_eval(
    *,
    scene: str,
    split_name: str,
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    cfg: CachedSparseEvalConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = reject_test_split(split_name, purpose="internal sparse cached eval")
    requested_query_ids = [str(query_id) for query_id in cfg.query_ids] if cfg.query_ids is not None else None
    artifact = load_listwise_candidate_artifact(
        candidate_artifact,
        query_ids=requested_query_ids,
        max_batches=cfg.max_queries,
    )
    landmark_map = load_gaussian_landmark_map(point_cloud)
    resolver = CacheLandmarkResolver.from_pair_cache(candidate_artifact, landmark_map)
    if (cfg.image_width is None) != (cfg.image_height is None):
        raise ValueError("image_width and image_height must be provided together")
    resolved_image_width = cfg.image_width
    resolved_image_height = cfg.image_height
    resolved_missing_principal_point = str(cfg.missing_principal_point)
    frame_calibration = _frame_calibration_summary(
        candidate_artifact=candidate_artifact,
        point_cloud=point_cloud,
        cameras_json=cameras_json,
        enabled=bool(cfg.frame_auto_calibration) and cfg.image_width is None,
        max_rows=int(cfg.frame_calibration_max_rows),
        agreement_threshold_px=float(cfg.frame_calibration_agreement_threshold_px),
    )
    if frame_calibration.get("status") == "passed":
        resolved_image_width = int(frame_calibration["best_frame_width"])
        resolved_image_height = int(frame_calibration["best_frame_height"])
        best_frame = str(frame_calibration.get("best_frame", ""))
        if best_frame.endswith("_pixel_center"):
            resolved_missing_principal_point = "pixel_center"
        elif best_frame.endswith("_half_extent"):
            resolved_missing_principal_point = "half_extent"
    if resolved_image_width is None:
        cameras = load_camera_records(cameras_json)
        pose_metric_frame = "camera_json_c2w"
    else:
        cameras = load_camera_records(
            cameras_json,
            target_width=int(resolved_image_width),
            target_height=int(resolved_image_height),
            missing_principal_point=resolved_missing_principal_point,
        )
        if frame_calibration.get("status") == "passed":
            pose_metric_frame = f"camera_json_c2w_{frame_calibration.get('best_frame')}"
        else:
            pose_metric_frame = f"camera_json_c2w_resized_{resolved_missing_principal_point}"
    scorer = load_candidate_scorer(cfg.candidate_scorer) if cfg.candidate_scorer is not None else None
    selector = load_landmark_selector(cfg.landmark_selector) if cfg.landmark_selector is not None else None
    descriptor_fusion = load_descriptor_fusion(cfg.descriptor_fusion) if cfg.descriptor_fusion is not None else None
    detector_student = load_detector_student(cfg.detector_student) if cfg.detector_student is not None else None
    input_cfg = CachedSparseInputConfig(score_mode=cfg.score_mode, max_keypoints=int(cfg.max_keypoints))
    loc_cfg = SparseLocalizationConfig(
        rerank_prefix_fraction=float(cfg.rerank_prefix_fraction),
        solver_weight=float(cfg.solver_weight),
        native_weight=float(cfg.native_weight),
        reprojection_error_px=float(cfg.reprojection_error_px),
        pnp_iterations=int(cfg.pnp_iterations),
        pnp_method=str(cfg.pnp_method),
        refine_with_inliers=bool(cfg.refine_with_inliers),
        second_pnp_enabled=bool(cfg.second_pnp_enabled),
        second_pnp_method=str(cfg.second_pnp_method),
        lgcv_reprojection_error_px=float(cfg.lgcv_reprojection_error_px),
        post_pnp_candidate_rescore=bool(cfg.post_pnp_candidate_rescore),
        post_pnp_reprojection_weight=float(cfg.post_pnp_reprojection_weight),
        post_pnp_reprojection_score_scale_px=float(cfg.post_pnp_reprojection_score_scale_px),
        post_pnp_rescore_min_improvement_px=float(cfg.post_pnp_rescore_min_improvement_px),
        post_pnp_rescore_max_residual_px=float(cfg.post_pnp_rescore_max_residual_px),
        post_pnp_rescore_only_initial_outliers=bool(cfg.post_pnp_rescore_only_initial_outliers),
    )
    requested_query_set = set(requested_query_ids) if requested_query_ids is not None else None
    rows: list[dict[str, object]] = []
    te_values: list[float] = []
    re_values: list[float] = []
    inliers: list[int] = []
    stage_counts: list[float] = []
    lgcv_keep_counts: list[float] = []
    post_pnp_changed_counts: list[float] = []
    post_pnp_corrected_counts: list[float] = []
    post_pnp_worsened_counts: list[float] = []
    post_pnp_correct_deltas: list[float] = []
    selected_correct_counts: list[float] = []
    selected_correct_ratios: list[float] = []
    selected_bbox_area_fractions: list[float] = []
    selected_depth_ranges: list[float] = []
    inlier_correct_counts: list[float] = []
    inlier_correct_ratios: list[float] = []
    inlier_bbox_area_fractions: list[float] = []
    inlier_depth_ranges: list[float] = []
    rerank_native_top1_correct = 0
    rerank_top1_correct = 0
    rerank_top1_changed_count = 0
    rerank_topk_available = 0
    rerank_diagnostic_query_count = 0
    batches = artifact.batches
    available_query_ids = {batch.query_id for batch in batches}
    missing_query_basis = requested_query_ids
    if missing_query_basis is not None and cfg.max_queries is not None:
        missing_query_basis = missing_query_basis[: max(0, int(cfg.max_queries))]
    missing_query_ids = (
        [query_id for query_id in missing_query_basis or [] if query_id not in available_query_ids]
        if requested_query_ids is not None
        else []
    )
    if requested_query_set is not None:
        batches = [batch for batch in batches if batch.query_id in requested_query_set]
    if cfg.max_queries is not None:
        batches = batches[: max(0, int(cfg.max_queries))]
    for batch in batches:
        camera = cameras.get(batch.query_id)
        if camera is None:
            rows.append({"query_id": batch.query_id, "success": False, "reason": "missing_camera"})
            continue
        batch_input_cfg = input_cfg
        solver_score_rows = None
        if scorer is not None:
            solver_score_rows = candidate_solver_score_rows(batch, scorer)
        if selector is not None:
            selector_rows = landmark_selector_score_rows(batch, selector)
            solver_score_rows = _combine_score_rows(
                solver_score_rows,
                selector_rows,
                selector_weight=float(cfg.landmark_selector_weight),
            )
        if descriptor_fusion is not None:
            descriptor_rows = descriptor_fusion_score_rows(batch, descriptor_fusion)
            solver_score_rows = _combine_score_rows(
                solver_score_rows,
                descriptor_rows,
                selector_weight=float(cfg.descriptor_fusion_weight),
            )
        if detector_student is not None:
            detector_rows = detector_score_rows(batch, detector_student)
            solver_score_rows = _combine_score_rows(
                solver_score_rows,
                detector_rows,
                selector_weight=float(cfg.detector_student_weight),
            )
        if solver_score_rows is not None:
            batch_input_cfg = CachedSparseInputConfig(
                score_mode=cfg.score_mode,
                max_keypoints=int(cfg.max_keypoints),
                solver_score_rows=solver_score_rows,
            )
        rerank_diagnostic = None
        if solver_score_rows is not None:
            rerank_diagnostic = _rerank_top1_diagnostic(
                batch,
                solver_score_rows,
                max_keypoints=int(cfg.max_keypoints),
                native_weight=float(cfg.native_weight),
                solver_weight=float(cfg.solver_weight),
            )
            rerank_native_top1_correct += int(rerank_diagnostic["native_top1_correct"])
            rerank_top1_correct += int(rerank_diagnostic["reranked_top1_correct"])
            rerank_top1_changed_count += int(rerank_diagnostic["reranked_top1_changed_count"])
            rerank_topk_available += int(rerank_diagnostic["topk_available"])
            rerank_diagnostic_query_count += int(rerank_diagnostic["query_count"])
        data = sparse_input_from_cached_batch(batch, resolver, intrinsics=camera.intrinsics, cfg=batch_input_cfg)
        result = run_sparse_localization(data, loc_cfg)
        te_cm = None
        re_deg = None
        if result.success and result.pose_w2c is not None and camera.pose_w2c is not None:
            te_cm, re_deg = pose_error_cm_deg(result.pose_w2c, camera.pose_w2c)
            te_values.append(float(te_cm))
            re_values.append(float(re_deg))
        inliers.append(int(result.inlier_count))
        stage_counts.append(float(result.pnp_stage_count))
        if result.lgcv_keep_count is not None:
            lgcv_keep_counts.append(float(result.lgcv_keep_count))
        post_pnp_changed_counts.append(float(result.post_pnp_rescore_changed_count))
        post_pnp_corrected_counts.append(float(result.post_pnp_rescore_corrected_count))
        post_pnp_worsened_counts.append(float(result.post_pnp_rescore_worsened_count))
        post_pnp_correct_deltas.append(float(result.post_pnp_rescore_correct_delta))
        selected_set_diagnostics = dict(result.selected_set_diagnostics or {})
        if selected_set_diagnostics:
            selected_correct_counts.append(float(selected_set_diagnostics.get("selected_geometric_correct_count", 0.0)))
            selected_correct_ratios.append(float(selected_set_diagnostics.get("selected_geometric_correct_ratio", 0.0)))
            selected_bbox_area_fractions.append(
                float(selected_set_diagnostics.get("selected_keypoint_bbox_area_fraction", 0.0))
            )
            selected_depth_ranges.append(float(selected_set_diagnostics.get("selected_depth_range_m", 0.0)))
        inlier_set_diagnostics = dict(result.inlier_set_diagnostics or {})
        if inlier_set_diagnostics:
            inlier_correct_counts.append(float(inlier_set_diagnostics.get("inlier_geometric_correct_count", 0.0)))
            inlier_correct_ratios.append(float(inlier_set_diagnostics.get("inlier_geometric_correct_ratio", 0.0)))
            inlier_bbox_area_fractions.append(
                float(inlier_set_diagnostics.get("inlier_keypoint_bbox_area_fraction", 0.0))
            )
            inlier_depth_ranges.append(float(inlier_set_diagnostics.get("inlier_depth_range_m", 0.0)))
        rows.append(
            {
                "query_id": batch.query_id,
                "success": bool(result.success),
                "inlier_count": int(result.inlier_count),
                "initial_inlier_count": int(result.initial_inlier_count),
                "lgcv_keep_count": result.lgcv_keep_count,
                "post_pnp_rescore_changed_count": int(result.post_pnp_rescore_changed_count),
                "post_pnp_rescore_corrected_count": int(result.post_pnp_rescore_corrected_count),
                "post_pnp_rescore_worsened_count": int(result.post_pnp_rescore_worsened_count),
                "post_pnp_rescore_correct_delta": int(result.post_pnp_rescore_correct_delta),
                "pnp_stage_count": int(result.pnp_stage_count),
                "selected_count": int(len(result.selected_landmark_ids)),
                "te_cm": te_cm,
                "re_deg": re_deg,
                "availability_summary": result.availability_summary,
                "selected_set_diagnostics": selected_set_diagnostics,
                "inlier_set_diagnostics": inlier_set_diagnostics,
                **({} if rerank_diagnostic is None else {"rerank_diagnostic": rerank_diagnostic}),
            }
        )
    success_rows = [row for row in rows if bool(row.get("success"))]
    summary = {
        "schema_version": "internal_sparse_cached_eval_metrics_v1",
        "scene": str(scene),
        "split_name": split,
        "query_filter_enabled": bool(requested_query_ids is not None),
        "requested_query_count": None if requested_query_ids is None else int(len(requested_query_ids)),
        "matched_query_count": None if requested_query_ids is None else int(len(rows)),
        "missing_query_count": None if requested_query_ids is None else int(len(missing_query_ids)),
        "missing_query_ids_preview": missing_query_ids[:10],
        "query_count": int(len(rows)),
        "success_count": int(len(success_rows)),
        "success_rate": float(len(success_rows) / max(1, len(rows))),
        "mean_inliers": float(mean(inliers)) if inliers else 0.0,
        "median_te_cm": _median_or_none(te_values),
        "median_re_deg": _median_or_none(re_values),
        "recall_10cm_5d": _recall(te_values, re_values, te_threshold_cm=10.0, re_threshold_deg=5.0),
        "recall_5cm_5d": _recall(te_values, re_values, te_threshold_cm=5.0, re_threshold_deg=5.0),
        "pose_metric_status": "verified" if te_values else "missing_gt_pose",
        "pose_metric_frame": pose_metric_frame,
        "resolved_image_width": None if resolved_image_width is None else int(resolved_image_width),
        "resolved_image_height": None if resolved_image_height is None else int(resolved_image_height),
        "resolved_missing_principal_point": resolved_missing_principal_point,
        "frame_calibration_status": str(frame_calibration.get("status", "disabled")),
        "frame_calibration_best_frame": frame_calibration.get("best_frame"),
        "frame_calibration_auto_resize_candidates": frame_calibration.get("auto_resize_candidates", []),
        "candidate_scorer_enabled": bool(scorer is not None),
        "landmark_selector_enabled": bool(selector is not None),
        "descriptor_fusion_enabled": bool(descriptor_fusion is not None),
        "detector_student_enabled": bool(detector_student is not None),
        "rerank_diagnostic_enabled": bool(rerank_diagnostic_query_count > 0),
        "rerank_diagnostic_query_count": int(rerank_diagnostic_query_count),
        "native_top1_correct": int(rerank_native_top1_correct),
        "reranked_top1_correct": int(rerank_top1_correct),
        "reranked_top1_gain": int(rerank_top1_correct - rerank_native_top1_correct),
        "reranked_top1_changed_count": int(rerank_top1_changed_count),
        "reranked_topk_available": int(rerank_topk_available),
        "selected_geometric_correct_count_median": _median_or_none(selected_correct_counts),
        "selected_geometric_correct_ratio_median": _median_or_none(selected_correct_ratios),
        "selected_keypoint_bbox_area_fraction_median": _median_or_none(selected_bbox_area_fractions),
        "selected_depth_range_m_median": _median_or_none(selected_depth_ranges),
        "inlier_geometric_correct_count_median": _median_or_none(inlier_correct_counts),
        "inlier_geometric_correct_ratio_median": _median_or_none(inlier_correct_ratios),
        "inlier_keypoint_bbox_area_fraction_median": _median_or_none(inlier_bbox_area_fractions),
        "inlier_depth_range_m_median": _median_or_none(inlier_depth_ranges),
        "pnp_stage_count_median": _median_or_none(stage_counts),
        "lgcv_keep_count_median": _median_or_none(lgcv_keep_counts),
        "post_pnp_rescore_changed_count_median": _median_or_none(post_pnp_changed_counts),
        "post_pnp_rescore_corrected_count_median": _median_or_none(post_pnp_corrected_counts),
        "post_pnp_rescore_worsened_count_median": _median_or_none(post_pnp_worsened_counts),
        "post_pnp_rescore_correct_delta_median": _median_or_none(post_pnp_correct_deltas),
        "post_pnp_candidate_rescore_enabled": bool(cfg.post_pnp_candidate_rescore),
        "candidate_artifact": artifact.summarize_candidate_availability(),
    }
    return summary, rows


def _frame_calibration_summary(
    *,
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    enabled: bool,
    max_rows: int,
    agreement_threshold_px: float,
) -> dict[str, object]:
    if not enabled:
        return {"status": "disabled"}
    try:
        audit = audit_pose_map_frame(
            candidate_artifact=candidate_artifact,
            point_cloud=point_cloud,
            cameras_json=cameras_json,
            max_rows=max_rows,
            agreement_threshold_px=agreement_threshold_px,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    best_width = audit.get("best_frame_width")
    best_height = audit.get("best_frame_height")
    if audit.get("status") == "passed" and best_width is not None and best_height is not None:
        return {
            "status": "passed",
            "best_frame": audit.get("best_frame"),
            "best_frame_width": int(best_width),
            "best_frame_height": int(best_height),
            "auto_resize_candidates": audit.get("auto_resize_candidates", []),
        }
    return {
        "status": str(audit.get("status", "failed")),
        "best_frame": audit.get("best_frame"),
        "auto_resize_candidates": audit.get("auto_resize_candidates", []),
    }


def _combine_score_rows(
    base_rows: Sequence[Sequence[float]] | None,
    selector_rows: Sequence[Sequence[float]],
    *,
    selector_weight: float,
) -> list[list[float]]:
    if base_rows is None:
        return [[float(selector_weight) * float(value) for value in row] for row in selector_rows]
    if len(base_rows) != len(selector_rows):
        raise ValueError("base score rows and selector score rows must have the same row count")
    combined: list[list[float]] = []
    for row_idx, (base, selector) in enumerate(zip(base_rows, selector_rows)):
        if len(base) != len(selector):
            raise ValueError(f"score row {row_idx} length mismatch")
        combined.append(
            [float(base_value) + float(selector_weight) * float(selector_value) for base_value, selector_value in zip(base, selector)]
        )
    return combined


def _rerank_top1_diagnostic(
    batch,
    solver_score_rows: Sequence[Sequence[float]],
    *,
    max_keypoints: int,
    native_weight: float,
    solver_weight: float,
) -> dict[str, int]:
    batch.validate()
    correct_rows = batch.candidate_geometric_correct
    if correct_rows is None:
        return {
            "query_count": 0,
            "native_top1_correct": 0,
            "reranked_top1_correct": 0,
            "reranked_top1_gain": 0,
            "reranked_top1_changed_count": 0,
            "topk_available": 0,
        }
    row_count = min(int(batch.keypoint_count), int(max_keypoints), len(solver_score_rows), len(correct_rows))
    valid_rows = batch.candidate_valid_mask
    native_top1_correct = 0
    reranked_top1_correct = 0
    changed_count = 0
    topk_available = 0
    query_count = 0
    for row_idx in range(row_count):
        scores = batch.candidate_scores[row_idx]
        solver_scores = solver_score_rows[row_idx]
        correct = correct_rows[row_idx]
        topk = min(len(scores), len(solver_scores), len(correct))
        valid_indices = [
            rank
            for rank in range(topk)
            if valid_rows is None or bool(valid_rows[row_idx][rank])
        ]
        if not valid_indices:
            continue
        native_best = valid_indices[0]
        reranked_best = max(
            valid_indices,
            key=lambda rank: (
                float(scores[rank]) * float(native_weight) + float(solver_scores[rank]) * float(solver_weight),
                -int(rank),
            ),
        )
        native_hit = bool(correct[native_best])
        reranked_hit = bool(correct[reranked_best])
        native_top1_correct += int(native_hit)
        reranked_top1_correct += int(reranked_hit)
        changed_count += int(reranked_best != native_best)
        topk_available += int(any(bool(correct[rank]) for rank in valid_indices))
        query_count += 1
    return {
        "query_count": int(query_count),
        "native_top1_correct": int(native_top1_correct),
        "reranked_top1_correct": int(reranked_top1_correct),
        "reranked_top1_gain": int(reranked_top1_correct - native_top1_correct),
        "reranked_top1_changed_count": int(changed_count),
        "topk_available": int(topk_available),
    }


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _recall(
    te_values: Sequence[float],
    re_values: Sequence[float],
    *,
    te_threshold_cm: float,
    re_threshold_deg: float,
) -> float:
    count = min(len(te_values), len(re_values))
    if count == 0:
        return 0.0
    passed = 0
    for te_cm, re_deg in zip(te_values[:count], re_values[:count]):
        if float(te_cm) <= float(te_threshold_cm) and float(re_deg) <= float(re_threshold_deg):
            passed += 1
    return float(passed / count)
