import pytest
import torch

from loc_gs.stdloc_native.query_conditioned_support import (
    build_feedback_bank_v2_solver_constraints,
    build_query_conditioned_solver_constraints,
)
from loc_gs.feedback.io import save_feedback_bank


def _episode_payload():
    return {
        "query_id": torch.tensor([10, 10, 11], dtype=torch.long),
        "candidate_landmark_ids": torch.tensor(
            [
                [0, 2, 3],
                [1, 2, 3],
                [0, 3, 2],
            ],
            dtype=torch.long,
        ),
        "candidate_cosine": torch.tensor(
            [
                [0.90, 0.80, 0.40],
                [0.70, 0.60, 0.95],
                [0.85, 0.90, 0.20],
            ],
            dtype=torch.float32,
        ),
        "candidate_reprojection_error": torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [1.5, 2.5, 9.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "candidate_visible": torch.ones((3, 3), dtype=torch.bool),
        "candidate_pnp_inlier": torch.tensor(
            [
                [True, True, True],
                [True, True, False],
                [True, True, True],
            ]
        ),
        "metadata": {"split_name": "selfmap_train_rendered"},
    }


def test_query_conditioned_constraints_emit_hard_query_candidate_gain_and_source_loss():
    constraints = build_query_conditioned_solver_constraints(
        _episode_payload(),
        source_idx=torch.tensor([0, 1], dtype=torch.long),
        num_gaussians=4,
        hard_query_ids=[10],
        reprojection_threshold_px=3.0,
        score_threshold=0.5,
        min_candidate_positive=0.5,
    )

    assert constraints["hard_query_ids"] == [10]
    assert constraints["candidate_gain"]["2"]["10"]["support"] > 0.0
    assert constraints["candidate_gain"].get("3", {}).get("10", {}).get("support", 0.0) == 0.0
    assert constraints["source_loss"]["0"]["10"]["support"] > 0.0
    assert constraints["source_loss"]["1"]["10"]["support"] > 0.0
    assert constraints["metadata"]["query_group_mode"] == "query_id"
    assert constraints["metadata"]["split_name"] == "selfmap_train_rendered"


def test_query_conditioned_constraints_map_base_landmark_ids_to_gaussian_ids():
    payload = _episode_payload()
    payload["base_gaussian_id"] = torch.tensor([100, 101, 102, 103], dtype=torch.long)

    constraints = build_query_conditioned_solver_constraints(
        payload,
        source_idx=torch.tensor([100, 101], dtype=torch.long),
        num_gaussians=104,
        hard_query_ids=[10],
        reprojection_threshold_px=3.0,
        score_threshold=0.5,
    )

    assert "102" in constraints["candidate_gain"]
    assert "100" in constraints["source_loss"]
    assert "2" not in constraints["candidate_gain"]


def test_query_conditioned_constraints_require_real_query_id():
    payload = _episode_payload()
    payload.pop("query_id")

    with pytest.raises(KeyError, match="query_id"):
        build_query_conditioned_solver_constraints(
            payload,
            source_idx=torch.tensor([0, 1], dtype=torch.long),
            num_gaussians=4,
            hard_query_ids=[10],
        )


def _feedback_record(
    gid,
    *,
    image_id,
    keypoint_id,
    pnp_inlier=True,
    descriptor_score=0.9,
    reprojection_error_px=1.0,
    dense_transition="improved",
    dense_delta_te_cm=-1.0,
    pose_error_t_cm=1.0,
):
    return {
        "scene": "ToyScene",
        "query_id": f"{image_id}::{keypoint_id}",
        "image_id": image_id,
        "source_view_id": image_id,
        "pose_source": "selfmap_rehearsal",
        "keypoint_id": keypoint_id,
        "keypoint_xy": [10.0, 20.0],
        "matched_landmark_id": str(gid),
        "matched_gaussian_id": str(gid),
        "descriptor_score": descriptor_score,
        "detector_score": 0.8,
        "match_rank": 1,
        "pnp_inlier": pnp_inlier,
        "pnp_success": True,
        "reprojection_error_px": reprojection_error_px,
        "depth_consistency": 0.9,
        "visibility_score": 1.0,
        "pose_error_t_cm": pose_error_t_cm,
        "pose_error_r_deg": 0.1,
        "dense_refine_success": dense_transition not in {"worsened", "lost"},
        "dense_transition": dense_transition,
        "dense_delta_te_cm": dense_delta_te_cm,
    }


def test_feedback_bank_v2_solver_constraints_emit_string_hard_queries_and_v3_metrics(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank_path,
        [
            _feedback_record(0, image_id="seq1/frame00001.png", keypoint_id="kp_000000", dense_delta_te_cm=0.0),
            _feedback_record(
                2,
                image_id="seq1/frame00001.png",
                keypoint_id="kp_000001",
                dense_transition="worsened",
                dense_delta_te_cm=5.0,
                pose_error_t_cm=12.0,
            ),
            _feedback_record(3, image_id="seq1/frame00002.png", keypoint_id="kp_000002", dense_delta_te_cm=-1.0),
        ],
        {
            "scene": "ToyScene",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    constraints = build_feedback_bank_v2_solver_constraints(
        bank_path,
        source_idx=torch.tensor([0, 1], dtype=torch.long),
        num_gaussians=4,
        hard_query_topk=1,
        hard_query_mode="dense_worsen",
        cvar_alpha=0.5,
    )

    assert constraints["format"] == "loc_gs_solver_admissibility_v2_feedback_bank"
    assert constraints["hard_query_ids"] == ["seq1/frame00001.png"]
    assert constraints["source_loss"]["0"]["seq1/frame00001.png"]["support"] == 1.0
    assert constraints["candidate_gain"]["2"]["seq1/frame00001.png"]["support"] == 1.0
    assert constraints["candidate_gain"]["2"]["seq1/frame00001.png"]["dense_worsen_risk"] == pytest.approx(1.0)
    assert "min_eigenvalue" in constraints["candidate_gain"]["2"]["seq1/frame00001.png"]
    assert constraints["thresholds"]["max_dense_worsen_delta"] == 0.0
    assert constraints["thresholds"]["cvar_alpha"] == 0.5
    assert constraints["metadata"]["query_group_mode"] == "feedback_bank_v2:image_id"
