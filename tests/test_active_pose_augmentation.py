import json
from pathlib import Path

import pytest

from loc_gs.feedback.active_pose_augmentation import (
    build_active_pose_augmentation,
    candidate_poses_from_observations,
    select_active_pose_candidates,
)
from loc_gs.scripts.build_solver_feedback_active_pose_augmentation import main as build_active_pose_main


def _candidate(
    pose_id,
    *,
    anchor="train_0001",
    query_ids=("q_hard",),
    alpha=0.9,
    selected_visible=16,
    candidate_visible=32,
    feature_variance=0.25,
    artifact=0.05,
    novelty=0.5,
    expected_gain=0.5,
    pose_source="train_perturb",
):
    return {
        "pose_id": pose_id,
        "anchor_view_id": anchor,
        "split_name": "selfmap_train",
        "pose_source": pose_source,
        "query_ids": list(query_ids),
        "alpha_valid_ratio": alpha,
        "stable_mask_ratio": 0.8,
        "selected_landmark_visible_count": selected_visible,
        "candidate_landmark_visible_count": candidate_visible,
        "feature_variance": feature_variance,
        "artifact_score": artifact,
        "novelty_score": novelty,
        "expected_query_gain": expected_gain,
        "covered_cells": [[0, 0], [1, 1]],
        "covered_depth_bins": [3, 7],
    }


def _observation(
    *,
    query_id="q_hard",
    source_view_id="pose_good",
    gaussian_id=7,
    inlier=True,
    reproj=1.0,
    score=0.9,
    margin=0.35,
    local_geometry=0.9,
    artifact=0.02,
):
    return {
        "query_id": query_id,
        "source_view_id": source_view_id,
        "split_name": "selfmap_train",
        "pnp_inlier": inlier,
        "reprojection_error_px": reproj,
        "descriptor_score": score,
        "descriptor_margin": margin,
        "local_geometry_score": local_geometry,
        "ray_artifact_score": artifact,
        "contributors": [{"gaussian_id": gaussian_id, "contribution": 1.0}],
    }


def test_active_pose_selection_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        select_active_pose_candidates([_candidate("p0")], split_name="test")


def test_active_pose_selection_uses_hard_constraints_and_nonweighted_priority():
    candidates = [
        _candidate("bad_artifact", artifact=0.8, expected_gain=10.0),
        _candidate("bad_visibility", selected_visible=1, expected_gain=10.0),
        _candidate("easy_pose", query_ids=("q_easy",), novelty=0.9, expected_gain=0.9),
        _candidate("hard_pose", query_ids=("q_hard",), novelty=0.2, expected_gain=0.4),
    ]

    result = select_active_pose_candidates(
        candidates,
        split_name="selfmap_train",
        hard_query_ids={"q_hard"},
        max_poses=2,
        min_selected_landmark_visible=4,
        max_artifact_score=0.3,
    )

    selected_ids = [row["pose_id"] for row in result["selected_poses"]]
    rejected = {row["pose_id"]: row["reject_reason"] for row in result["rejected_poses"]}
    assert selected_ids[0] == "hard_pose"
    assert "artifact" in rejected["bad_artifact"]
    assert "selected_landmark_visible" in rejected["bad_visibility"]


def test_view_landmark_reliability_selects_clean_solver_inlier_views_and_deboosts_bad_views():
    artifact = build_active_pose_augmentation(
        candidate_poses=[_candidate("pose_good")],
        observations=[
            _observation(source_view_id="pose_good", gaussian_id=7, inlier=True, reproj=1.0),
            _observation(source_view_id="pose_bad", gaussian_id=7, inlier=False, reproj=80.0, score=0.95),
            _observation(source_view_id="pose_artifact", gaussian_id=8, inlier=True, reproj=1.0, artifact=0.9),
        ],
        num_gaussians=10,
        scene="ShopFacade",
        split_name="selfmap_train",
        max_poses=1,
        min_positive_observations=1,
        max_negative_observations=0,
        max_fusion_artifact_score=0.3,
    )

    fusion_plan = artifact["landmark_fusion_plan"]
    rows_by_key = {
        (row["source_view_id"], row["gaussian_id"]): row for row in artifact["view_landmark_reliability"]
    }
    assert fusion_plan["7"]["selected_view_ids"] == ["pose_good"]
    assert rows_by_key[("pose_good", 7)]["status"] == "selected_for_fusion"
    assert rows_by_key[("pose_bad", 7)]["status"] == "deboosted"
    assert rows_by_key[("pose_artifact", 8)]["status"] == "deboosted"
    assert artifact["metrics"]["selected_fusion_view_landmark_pairs"] == 1
    assert artifact["metrics"]["deboosted_view_landmark_pairs"] == 2


def test_active_pose_compact_output_keeps_fusion_plan_without_full_reliability_table():
    artifact = build_active_pose_augmentation(
        candidate_poses=[_candidate("pose_good")],
        observations=[
            _observation(source_view_id="pose_good", gaussian_id=7, inlier=True, reproj=1.0),
            _observation(source_view_id="pose_bad", gaussian_id=8, inlier=False, reproj=80.0, score=0.95),
        ],
        num_gaussians=10,
        scene="ShopFacade",
        split_name="selfmap_train",
        max_poses=1,
        min_positive_observations=1,
        compact=True,
        reliability_sample_limit=1,
    )

    assert artifact["view_landmark_reliability"] == []
    assert artifact["view_landmark_reliability_sample"][0]["source_view_id"] == "pose_good"
    assert artifact["landmark_fusion_plan"]["7"]["selected_view_ids"] == ["pose_good"]
    assert artifact["metrics"]["compact_output"] is True
    assert artifact["metrics"]["selected_fusion_view_landmark_pairs"] == 1


def test_candidate_poses_can_be_inferred_from_observations():
    observations = [
        _observation(query_id="q_hard", source_view_id="aug_a", gaussian_id=1, inlier=True, artifact=0.05),
        _observation(query_id="q_hard", source_view_id="aug_a", gaussian_id=2, inlier=True, artifact=0.10),
        _observation(query_id="q_easy", source_view_id="aug_b", gaussian_id=3, inlier=False, artifact=0.7),
    ]

    candidates = candidate_poses_from_observations(observations, split_name="selfmap_train")

    by_id = {row["pose_id"]: row for row in candidates}
    assert by_id["aug_a"]["selected_landmark_visible_count"] == 2
    assert by_id["aug_a"]["candidate_landmark_visible_count"] == 2
    assert by_id["aug_a"]["query_ids"] == ["q_hard"]
    assert by_id["aug_a"]["artifact_score"] == pytest.approx(0.075)
    assert by_id["aug_b"]["candidate_landmark_visible_count"] == 1
    assert by_id["aug_b"]["selected_landmark_visible_count"] == 0


def test_active_pose_cli_writes_audited_artifacts(tmp_path: Path):
    candidates_path = tmp_path / "candidates.jsonl"
    observations_path = tmp_path / "observations.jsonl"
    output_dir = tmp_path / "out"
    candidates_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "manifest", "manifest": {"split_name": "selfmap_train"}}),
                json.dumps({"type": "candidate_pose", "candidate_pose": _candidate("pose_good")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    observations_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "manifest", "manifest": {"split_name": "selfmap_train"}}),
                json.dumps({"type": "observation", "observation": _observation(source_view_id="pose_good")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = build_active_pose_main(
        [
            "--candidate_poses",
            str(candidates_path),
            "--observations",
            str(observations_path),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ShopFacade",
            "--split_name",
            "selfmap_train",
            "--num_gaussians",
            "10",
            "--max_poses",
            "1",
            "--compact",
        ]
    )

    assert rc == 0
    assert (output_dir / "active_pose_augmentation.json").exists()
    assert (output_dir / "view_landmark_reliability.json").exists()
    assert (output_dir / "selected_poses.csv").exists()
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert split_audit["test_split_used"] is False
    assert split_audit["audit_status"] == "passed"
    assert metrics["selected_pose_count"] == 1
    assert metrics["selected_fusion_view_landmark_pairs"] == 1
    payload = json.loads((output_dir / "active_pose_augmentation.json").read_text(encoding="utf-8"))
    assert payload["view_landmark_reliability"] == []
    assert payload["landmark_fusion_plan"]


def test_active_pose_cli_can_infer_candidate_poses_from_observations(tmp_path: Path):
    observations_path = tmp_path / "observations.jsonl"
    output_dir = tmp_path / "out"
    observations_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "manifest", "manifest": {"split_name": "selfmap_train"}}),
                json.dumps({"type": "observation", "observation": _observation(source_view_id="pose_good")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = build_active_pose_main(
        [
            "--observations",
            str(observations_path),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ShopFacade",
            "--split_name",
            "selfmap_train",
            "--num_gaussians",
            "10",
            "--max_poses",
            "1",
            "--infer_candidate_poses_from_observations",
        ]
    )

    assert rc == 0
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    selected = (output_dir / "selected_poses.csv").read_text(encoding="utf-8")
    assert metrics["selected_pose_count"] == 1
    assert "pose_good" in selected
