import json
import subprocess
import sys

import pytest
import torch

from loc_gs.diagnostics.solver_consensus_support import build_solver_consensus_support
from loc_gs.feedback.io import save_feedback_bank


def _record(
    gaussian_id,
    *,
    image_id,
    keypoint_id,
    descriptor_score=0.9,
    pnp_inlier=True,
    reprojection_error_px=1.0,
    dense_transition="improved",
    dense_delta_te_cm=-1.0,
):
    return {
        "scene": "ToyScene",
        "query_id": f"{image_id}::{keypoint_id}",
        "image_id": image_id,
        "source_view_id": image_id,
        "pose_source": "selfmap_rehearsal",
        "keypoint_id": keypoint_id,
        "keypoint_xy": [10.0, 20.0],
        "matched_landmark_id": str(gaussian_id),
        "matched_gaussian_id": str(gaussian_id),
        "descriptor_score": descriptor_score,
        "detector_score": 0.8,
        "match_rank": 1,
        "pnp_inlier": pnp_inlier,
        "pnp_success": True,
        "reprojection_error_px": reprojection_error_px,
        "depth_consistency": 0.9,
        "visibility_score": 0.85,
        "dense_refine_success": dense_transition not in {"lost", "worsened"},
        "dense_transition": dense_transition,
        "dense_delta_te_cm": dense_delta_te_cm,
    }


def _write_feedback_bank(path, *, split_name="selfmap_train", schema_version="feedback_bank_v2"):
    records = [
        _record(0, image_id="seq1/frame00001.png", keypoint_id="kp_000000", descriptor_score=0.95),
        _record(0, image_id="seq1/frame00002.png", keypoint_id="kp_000001", descriptor_score=0.85),
        _record(
            1,
            image_id="seq1/frame00001.png",
            keypoint_id="kp_000002",
            descriptor_score=0.95,
            pnp_inlier=False,
            reprojection_error_px=12.0,
            dense_transition="worsened",
            dense_delta_te_cm=3.0,
        ),
        _record(
            1,
            image_id="seq1/frame00002.png",
            keypoint_id="kp_000003",
            descriptor_score=0.7,
            pnp_inlier=False,
            reprojection_error_px=9.0,
            dense_transition="lost",
            dense_delta_te_cm=0.5,
        ),
        _record(
            2,
            image_id="seq1/frame00002.png",
            keypoint_id="kp_000004",
            descriptor_score=0.8,
            pnp_inlier=True,
            reprojection_error_px=2.0,
            dense_transition="unchanged",
            dense_delta_te_cm=4.0,
        ),
    ]
    save_feedback_bank(
        path,
        records,
        {
            "scene": "ToyScene",
            "split_name": split_name,
            "schema_version": schema_version,
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )


def test_solver_consensus_support_rejects_audit_failed_bank(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_feedback_bank(bank_path, split_name="test")

    with pytest.raises(ValueError, match="feedback bank v2 audit failed"):
        build_solver_consensus_support(bank_path)


def test_solver_consensus_support_builds_expected_tensors_and_metadata(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_feedback_bank(bank_path)

    artifact = build_solver_consensus_support(bank_path, num_gaussians=4, support_threshold=0.5)

    for name in (
        "support_score",
        "inlier_consensus",
        "weighted_support",
        "hard_negative_risk",
        "dense_worsen_risk",
        "observed_count",
        "positive_observed_count",
    ):
        assert artifact[name].shape == (4,)

    assert artifact["observed_count"].tolist() == [2, 2, 1, 0]
    assert artifact["positive_observed_count"].tolist() == [2, 0, 1, 0]
    assert artifact["inlier_consensus"][0] == pytest.approx(1.0)
    assert artifact["inlier_consensus"][1] == pytest.approx(0.0)
    assert artifact["hard_negative_risk"][1] == pytest.approx(1.0)
    assert artifact["dense_worsen_risk"][1] == pytest.approx(1.0)
    assert artifact["dense_worsen_risk"][2] > 0.0
    assert artifact["support_score"][0] > artifact["support_score"][1]
    assert artifact["weighted_support"][0] > artifact["weighted_support"][1]
    assert artifact["support_score"][3] == pytest.approx(0.0)
    assert artifact["metadata"]["schema"] == "solver_consensus_support_v1"
    assert artifact["metadata"]["split"] == "selfmap_train"
    assert artifact["metadata"]["image_group_count"] == 2
    assert artifact["metadata"]["source_feedback_bank"] == str(bank_path)
    assert artifact["metadata"]["selected_count"] == 1


def test_solver_consensus_support_is_deterministic(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_feedback_bank(bank_path)

    first = build_solver_consensus_support(bank_path, num_gaussians=4, support_threshold=0.5)
    second = build_solver_consensus_support(bank_path, num_gaussians=4, support_threshold=0.5)

    for name in (
        "support_score",
        "inlier_consensus",
        "weighted_support",
        "hard_negative_risk",
        "dense_worsen_risk",
        "observed_count",
        "positive_observed_count",
    ):
        assert torch.equal(first[name], second[name])
    assert first["metadata"] == second["metadata"]


def test_solver_consensus_support_can_ignore_non_inlier_matches_for_ulfloc_feedback(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    records = [
        _record(
            0,
            image_id="seq1/frame00001.png",
            keypoint_id="kp_good",
            descriptor_score=0.9,
            pnp_inlier=True,
            reprojection_error_px=1.0,
        ),
        *[
            _record(
                0,
                image_id="seq1/frame00001.png",
                keypoint_id=f"kp_bad_{idx}",
                descriptor_score=0.95,
                pnp_inlier=False,
                reprojection_error_px=1200.0,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            )
            for idx in range(4)
        ],
        _record(
            1,
            image_id="seq1/frame00001.png",
            keypoint_id="kp_bad_only",
            descriptor_score=0.95,
            pnp_inlier=False,
            reprojection_error_px=1200.0,
            dense_transition="not_run_sparse_feedback",
            dense_delta_te_cm=0.0,
        ),
    ]
    save_feedback_bank(
        bank_path,
        records,
        {
            "scene": "ToyScene",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    artifact = build_solver_consensus_support(
        bank_path,
        num_gaussians=3,
        support_threshold=0.5,
        evidence_mode="inlier_positive_only",
    )

    assert artifact["observed_count"].tolist() == [5, 1, 0]
    assert artifact["positive_observed_count"].tolist() == [1, 0, 0]
    assert artifact["support_score"][0] >= 0.5
    assert artifact["support_score"][1] == pytest.approx(0.0)
    assert artifact["hard_negative_risk"][0] == pytest.approx(0.0)
    assert artifact["hard_negative_risk"][1] == pytest.approx(0.0)
    assert artifact["metadata"]["selected_count"] == 1
    assert artifact["metadata"]["hyperparameters"]["evidence_mode"] == "inlier_positive_only"


def test_build_solver_consensus_support_cli_writes_artifact_and_manifest(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    output_dir = tmp_path / "support"
    _write_feedback_bank(bank_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_consensus_support",
            "--feedback_bank",
            str(bank_path),
            "--output_dir",
            str(output_dir),
            "--num_gaussians",
            "4",
            "--support_threshold",
            "0.5",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(completed.stdout)
    artifact_path = output_dir / "solver_consensus_support.pt"
    manifest_path = output_dir / "manifest.json"
    assert payload["artifact_path"] == str(artifact_path)
    assert payload["manifest_path"] == str(manifest_path)
    assert artifact_path.exists()
    assert manifest_path.exists()

    artifact = torch.load(artifact_path, map_location="cpu")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert artifact["metadata"]["source_feedback_bank"] == str(bank_path)
    assert artifact["metadata"]["selected_count"] == 1
    assert manifest["artifact"] == "solver_consensus_support.pt"
    assert manifest["metadata"]["image_group_count"] == 2
    assert manifest["feedback_bank"] == str(bank_path)


def test_build_solver_consensus_support_cli_records_repo_commit_outside_repo_cwd(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    output_dir = tmp_path / "support"
    _write_feedback_bank(bank_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_consensus_support",
            "--feedback_bank",
            str(bank_path),
            "--output_dir",
            str(output_dir),
            "--num_gaussians",
            "4",
        ],
        check=True,
        cwd="/tmp",
        env={"PYTHONPATH": "/root/Loc-GS:/root/Loc-GS/third_party/stdloc"},
        stdout=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"] != "unknown"
    assert len(manifest["git_commit"]) == 40
