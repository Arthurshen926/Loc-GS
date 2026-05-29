import argparse

import numpy as np

from loc_gs.dense_support.anchor_conditioned_patch_dense import (
    AnchorConditionedPatchDensePolicy,
    assess_anchor_conditioned_update,
    build_anchor_conditioned_refinement_matches,
    summarize_anchor_conditioned_patch_dense,
)
from loc_gs.scripts import diagnose_patch_guided_sparse as pgsh


def test_acpd_refinement_matches_add_only_reference_consistent_sparse_anchors():
    dense_merged = {
        "xy": np.array([[20.0, 20.0], [24.0, 20.0]], dtype=np.float32),
        "xyz": np.array([[0.0, 0.0, 10.0], [0.5, 0.0, 10.0]], dtype=np.float32),
    }
    sparse = {
        "query_xy": np.array([[1.0, 1.0], [1.6, 1.0], [90.0, 90.0]], dtype=np.float32),
        "p3d": np.array([[1.0, 1.0, 10.0], [1.6, 1.0, 10.0], [1.0, 1.0, 10.0]], dtype=np.float32),
        "inliers": np.array([0, 1, 2], dtype=np.int32),
    }
    intrinsic = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    refinement = build_anchor_conditioned_refinement_matches(
        dense_merged,
        sparse,
        reference_pose_w2c=np.eye(4, dtype=np.float32),
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=AnchorConditionedPatchDensePolicy(anchor_weight=3.0, anchor_max_count=8, anchor_max_reprojection_px=2.0),
    )

    assert refinement["xy"].shape == (4, 2)
    assert refinement["xyz"].shape == (4, 3)
    assert refinement["weights"].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert refinement["diagnostics"]["schema"] == "loc_gs_acpd_sparse_anchor_refinement_v1"
    assert refinement["diagnostics"]["selected_anchor_count"] == 2
    assert refinement["diagnostics"]["candidate_anchor_count"] == 3


def test_acpd_summary_marks_anchor_conditioned_patch_dense_as_not_branch_selector():
    summary = summarize_anchor_conditioned_patch_dense(
        decision="use_dense_pgsh",
        patch_ids=[1, 3],
        merged_match_count=40,
        candidate_match_count=100,
        final_inlier_count=31,
        anchor_diagnostics={
            "candidate_anchor_count": 12,
            "selected_anchor_count": 8,
            "sparse_anchor_weight": 3.0,
        },
        transition_control={"decision": "accept_line_search_dense_update", "selected_fraction": 0.5},
        base_trust={"decision": "accept_dense_pgsh_pose"},
    )

    assert summary["schema"] == "loc_gs_anchor_conditioned_patch_dense_v1"
    assert summary["anchor_conditioned"] is True
    assert summary["query_side_occlusion_module"] == "patch_dense_matching"
    assert summary["map_side_artifact_module"] == "sparse_conditioned_render_control"
    assert summary["uses_gt"] is False
    assert summary["is_oracle_branch_selector"] is False
    assert summary["selected_anchor_count"] == 8
    assert summary["anchor_selected_ratio"] == 8 / 12


def test_acpd_anchor_target_fraction_prevents_dense_match_dilution():
    dense_merged = {
        "xy": np.zeros((100, 2), dtype=np.float32),
        "xyz": np.tile(np.array([[0.0, 0.0, 10.0]], dtype=np.float32), (100, 1)),
    }
    sparse = {
        "query_xy": np.array([[1.0, 1.0], [1.6, 1.0]], dtype=np.float32),
        "p3d": np.array([[1.0, 1.0, 10.0], [1.6, 1.0, 10.0]], dtype=np.float32),
        "inliers": np.array([0, 1], dtype=np.int32),
    }
    intrinsic = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    refinement = build_anchor_conditioned_refinement_matches(
        dense_merged,
        sparse,
        reference_pose_w2c=np.eye(4, dtype=np.float32),
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=AnchorConditionedPatchDensePolicy(
            anchor_weight=1.0,
            anchor_max_count=8,
            anchor_max_reprojection_px=2.0,
            anchor_target_weight_fraction=0.20,
            anchor_max_weight=20.0,
        ),
    )

    assert refinement["diagnostics"]["selected_anchor_count"] == 2
    assert refinement["diagnostics"]["effective_sparse_anchor_weight"] == 12.5
    assert refinement["weights"][-2:].tolist() == [12.5, 12.5]


def test_acpd_refinement_acceptance_rejects_motion_with_tiny_objective_gain():
    decision = assess_anchor_conditioned_update(
        {
            "success": True,
            "before_median_reprojection_error_px": 0.61,
            "after_median_reprojection_error_px": 0.60,
            "before_p90_reprojection_error_px": 1.20,
            "after_p90_reprojection_error_px": 1.19,
            "delta_translation_norm_m": 0.03,
            "delta_rotation_norm_deg": 0.10,
        },
        policy=AnchorConditionedPatchDensePolicy(min_refine_median_gain_px=0.05, min_refine_p90_gain_px=0.10),
    )

    assert decision["decision"] == "reject_anchor_conditioned_update"
    assert decision["failed_checks"]["insufficient_objective_gain_for_motion"] is True


def test_acpd_refinement_acceptance_accepts_clear_no_gt_objective_gain():
    decision = assess_anchor_conditioned_update(
        {
            "success": True,
            "before_median_reprojection_error_px": 1.50,
            "after_median_reprojection_error_px": 1.30,
            "before_p90_reprojection_error_px": 4.0,
            "after_p90_reprojection_error_px": 3.5,
            "delta_translation_norm_m": 0.04,
            "delta_rotation_norm_deg": 0.20,
        },
        policy=AnchorConditionedPatchDensePolicy(min_refine_median_gain_px=0.05, min_refine_p90_gain_px=0.10),
    )

    assert decision["decision"] == "accept_anchor_conditioned_update"


def test_acpd_cli_preset_enables_dense_patch_and_anchor_conditioned_sparse_conditioning():
    args = pgsh.build_argparser().parse_args(["--anchor_conditioned_patch_dense"])

    resolved = pgsh._resolve_anchor_conditioned_patch_dense_args(args)

    assert resolved.run_dense is True
    assert resolved.dense_pgsh is True
    assert resolved.disable_sparse_pgsh is True
    assert resolved.dense_pgsh_reference_local_refine is True
    assert resolved.dense_pgsh_sparse_anchor_weight == 3.0
    assert resolved.dense_pgsh_sparse_anchor_target_weight_fraction == 0.10
    assert resolved.slcdp_render_control == "sparse_conditioned"
