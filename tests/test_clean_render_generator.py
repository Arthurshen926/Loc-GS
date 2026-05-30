import numpy as np

from loc_gs.dense_support.clean_render_generator import (
    CleanRenderPolicy,
    select_clean_render_candidate,
)
from loc_gs.scripts.build_clean_render_report import main as build_clean_render_report_main


def _candidate(label, *, preflight, translation_norm_m=0.0, gating=None):
    return {
        "label": label,
        "pose_w2c": np.eye(4, dtype=np.float32),
        "translation_cam_m": (float(translation_norm_m), 0.0, 0.0),
        "preflight": preflight,
        "render_control": {"gaussian_gating": gating},
    }


def test_clean_render_generator_keeps_native_when_base_render_is_clean():
    base = _candidate(
        "base",
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.88,
            "coverage_grid_cells": 8,
            "feature_cosine_median": 0.72,
            "local_feature_variance_median": 0.03,
            "near_occluder_ratio": 0.02,
            "sparse_inlier_count": 120,
        },
    )
    shifted = _candidate(
        "ray_centroid+1.000",
        translation_norm_m=1.0,
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.90,
            "coverage_grid_cells": 8,
            "feature_cosine_median": 0.73,
            "local_feature_variance_median": 0.03,
            "near_occluder_ratio": 0.02,
            "sparse_inlier_count": 120,
        },
    )

    result = select_clean_render_candidate([base, shifted], policy=CleanRenderPolicy(min_score_gain=0.05))

    assert result["decision"] == "use_native_clean_render"
    assert result["selected_label"] == "base"
    assert result["selected_role"] == "render_pose_only"
    assert result["does_not_select_final_pose"] is True


def test_clean_render_generator_uses_labeled_base_even_when_candidates_are_reordered():
    shifted = _candidate(
        "ray_centroid+1.000",
        translation_norm_m=1.0,
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.90,
            "coverage_grid_cells": 8,
            "feature_cosine_median": 0.73,
            "local_feature_variance_median": 0.03,
            "near_occluder_ratio": 0.02,
            "sparse_inlier_count": 120,
        },
    )
    base = _candidate(
        "base",
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.88,
            "coverage_grid_cells": 8,
            "feature_cosine_median": 0.72,
            "local_feature_variance_median": 0.03,
            "near_occluder_ratio": 0.02,
            "sparse_inlier_count": 120,
        },
    )

    result = select_clean_render_candidate([shifted, base], policy=CleanRenderPolicy(min_score_gain=0.50))

    assert result["base_index"] == 1
    assert result["selected_label"] == "base"
    assert result["decision"] == "use_native_clean_render"


def test_clean_render_generator_selects_guided_gated_candidate_for_cleaner_render():
    base = _candidate(
        "base",
        preflight={
            "decision": "skip_dense_keep_sparse",
            "visible_ratio": 0.05,
            "coverage_grid_cells": 0,
            "feature_cosine_median": 0.0,
            "local_feature_variance_median": 0.000001,
            "near_occluder_ratio": 1.0,
            "sparse_inlier_count": 127,
        },
    )
    clean = _candidate(
        "gated_ray_centroid+1.000",
        translation_norm_m=1.0,
        gating={"conflict_count": 20, "kept_count": 980, "protected_count": 120},
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.72,
            "coverage_grid_cells": 7,
            "feature_cosine_median": 0.48,
            "local_feature_variance_median": 0.025,
            "near_occluder_ratio": 0.08,
            "sparse_inlier_count": 127,
        },
    )

    result = select_clean_render_candidate([base, clean], policy=CleanRenderPolicy(min_score_gain=0.05))

    assert result["decision"] == "use_clean_render_candidate"
    assert result["selected_label"] == "gated_ray_centroid+1.000"
    assert result["selected_score"] > result["base_score"]
    assert result["selected"]["gating_removed_fraction"] < 0.05


def test_clean_render_generator_rejects_candidate_that_drops_sparse_anchors():
    base = _candidate(
        "base",
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.60,
            "coverage_grid_cells": 5,
            "feature_cosine_median": 0.30,
            "local_feature_variance_median": 0.01,
            "near_occluder_ratio": 0.20,
            "sparse_inlier_count": 90,
        },
    )
    pretty_but_unanchored = _candidate(
        "ray_side+2.000",
        translation_norm_m=2.0,
        preflight={
            "decision": "retry_sparse_or_patch_dense",
            "visible_ratio": 0.12,
            "coverage_grid_cells": 1,
            "feature_cosine_median": 0.95,
            "local_feature_variance_median": 0.08,
            "near_occluder_ratio": 0.0,
            "sparse_inlier_count": 90,
        },
    )

    result = select_clean_render_candidate([base, pretty_but_unanchored])

    assert result["selected_label"] == "base"
    assert result["candidate_scores"][1]["anchor_safe"] is False
    assert result["decision"] == "use_native_clean_render"


def test_clean_render_generator_rejects_non_base_when_sparse_anchors_are_not_confident():
    base = _candidate(
        "base",
        preflight={
            "decision": "retry_sparse_or_patch_dense",
            "visible_ratio": 0.30,
            "coverage_grid_cells": 3,
            "feature_cosine_median": 0.40,
            "local_feature_variance_median": 0.01,
            "near_occluder_ratio": 0.40,
            "sparse_inlier_count": 26,
            "sparse_confident": False,
        },
    )
    gated = _candidate(
        "gated_base",
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.93,
            "coverage_grid_cells": 7,
            "feature_cosine_median": 0.72,
            "local_feature_variance_median": 0.02,
            "near_occluder_ratio": 0.08,
            "sparse_inlier_count": 26,
            "sparse_confident": False,
        },
    )

    result = select_clean_render_candidate([base, gated])

    assert result["candidate_scores"][1]["anchor_safe"] is False
    assert result["decision"] == "use_native_clean_render"
    assert result["selected_label"] == "base"


def test_clean_render_generator_penalizes_large_gating_when_quality_is_equal():
    base = _candidate(
        "base",
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.70,
            "coverage_grid_cells": 6,
            "feature_cosine_median": 0.50,
            "local_feature_variance_median": 0.02,
            "near_occluder_ratio": 0.10,
            "sparse_inlier_count": 100,
        },
    )
    over_gated = _candidate(
        "gated_base",
        gating={"conflict_count": 450, "kept_count": 550, "protected_count": 100},
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.70,
            "coverage_grid_cells": 6,
            "feature_cosine_median": 0.50,
            "local_feature_variance_median": 0.02,
            "near_occluder_ratio": 0.10,
            "sparse_inlier_count": 100,
        },
    )

    result = select_clean_render_candidate([base, over_gated])

    assert result["selected_label"] == "base"
    assert result["candidate_scores"][1]["gating_removed_fraction"] > 0.40
    assert result["candidate_scores"][1]["score"] < result["candidate_scores"][0]["score"]


def test_build_clean_render_report_scores_existing_candidate_summaries(tmp_path):
    base = _candidate(
        "base",
        preflight={
            "decision": "skip_dense_keep_sparse",
            "visible_ratio": 0.05,
            "coverage_grid_cells": 0,
            "feature_cosine_median": 0.0,
            "near_occluder_ratio": 1.0,
            "sparse_inlier_count": 80,
        },
    )
    clean = _candidate(
        "gated_ray_centroid+1.000",
        translation_norm_m=1.0,
        gating={"conflict_count": 10, "kept_count": 990},
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.75,
            "coverage_grid_cells": 8,
            "feature_cosine_median": 0.45,
            "near_occluder_ratio": 0.05,
            "sparse_inlier_count": 80,
        },
    )
    base.pop("pose_w2c")
    clean.pop("pose_w2c")
    summary = {
        "dense": {
            "slcdp_repair_search": {
                "candidates": [
                    base,
                    clean,
                ]
            }
        }
    }
    source = tmp_path / "case_summary.json"
    source.write_text(__import__("json").dumps(summary), encoding="utf-8")
    output_dir = tmp_path / "report"

    code = build_clean_render_report_main(
        [
            "--case",
            f"synthetic={source}",
            "--output_dir",
            str(output_dir),
        ]
    )

    assert code == 0
    metrics = __import__("json").loads((output_dir / "metrics_summary.json").read_text())
    assert metrics["case_count"] == 1
    assert metrics["cases"][0]["decision"] == "use_clean_render_candidate"
    assert metrics["cases"][0]["selected_label"] == "gated_ray_centroid+1.000"
    assert (output_dir / "report.md").exists()


def test_build_clean_render_report_writes_split_audit_for_test_diagnostic(tmp_path):
    base = _candidate(
        "base",
        preflight={
            "decision": "accept_dense",
            "visible_ratio": 0.70,
            "coverage_grid_cells": 6,
            "feature_cosine_median": 0.50,
            "near_occluder_ratio": 0.10,
            "sparse_inlier_count": 100,
        },
    )
    base.pop("pose_w2c")
    summary = {"split": "test", "dense": {"slcdp_repair_search": {"candidates": [base]}}}
    source = tmp_path / "kings_test_case_summary.json"
    source.write_text(__import__("json").dumps(summary), encoding="utf-8")
    output_dir = tmp_path / "report"

    code = build_clean_render_report_main(
        [
            "--case",
            f"test_case={source}",
            "--output_dir",
            str(output_dir),
        ]
    )

    assert code == 0
    split_audit = __import__("json").loads((output_dir / "split_audit.json").read_text())
    assert split_audit["official_test_used"] is True
    assert split_audit["paper_safe_for_tuning"] is False
    assert split_audit["source_splits"] == ["test"]
