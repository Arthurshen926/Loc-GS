import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from loc_gs.dense_support.dense_transition_guard import (
    DenseTransitionGuardPolicy,
    build_dense_transition_report,
    should_accept_dense,
)
from loc_gs.diagnostics.match_visualization import project_points


def _camera() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return pose, intrinsic, (96, 72)


def _points() -> np.ndarray:
    return np.array(
        [
            [-0.20, -0.10, 4.0],
            [0.20, -0.10, 4.2],
            [-0.15, 0.15, 4.5],
            [0.18, 0.12, 5.0],
            [0.00, 0.00, 5.5],
            [0.28, 0.18, 6.0],
        ],
        dtype=np.float32,
    )


def _sparse_inliers() -> dict:
    pose, intrinsic, image_size = _camera()
    xy, valid = project_points(_points(), pose, intrinsic, width=image_size[0], height=image_size[1])
    return {
        "query_xy": xy[valid].astype(np.float32),
        "p3d": _points()[valid],
        "inliers": np.arange(int(valid.sum()), dtype=np.int32),
        "inlier_ratio": 0.72,
        "intrinsic": intrinsic,
        "image_size": image_size,
    }


def test_guard_accepts_dense_when_sparse_evidence_is_absent() -> None:
    sparse_pose, _intrinsic, _image_size = _camera()
    dense_pose = sparse_pose.copy()
    dense_pose[0, 3] = 2.0

    decision = should_accept_dense(
        sparse_pose,
        dense_pose,
        sparse_inliers={},
        dense_stats={"valid_dense_match_ratio": 0.02, "artifact_depth_risk_score": 1.0},
        policy=DenseTransitionGuardPolicy(),
    )

    assert decision["decision"] == "accept_dense"
    assert decision["sparse_support_strength"] == "none"
    assert decision["uses_gt"] is False
    assert "no_sparse_evidence" in decision["reasons"]
    assert decision["failed_checks"]["pose_translation_trust_region"] is True


def test_guard_rejects_high_confidence_sparse_when_dense_breaks_observable_checks() -> None:
    sparse_pose, _intrinsic, _image_size = _camera()
    dense_pose = sparse_pose.copy()
    dense_pose[0, 3] = 0.45

    decision = should_accept_dense(
        sparse_pose,
        dense_pose,
        sparse_inliers=_sparse_inliers(),
        dense_stats={"valid_dense_match_ratio": 0.15, "artifact_depth_risk_score": 0.90},
        policy=DenseTransitionGuardPolicy(
            high_confidence_min_inliers=4,
            high_confidence_min_inlier_ratio=0.5,
            max_translation_delta_m=0.10,
            max_rotation_delta_deg=3.0,
            max_sparse_reprojection_error_px=2.0,
            max_sparse_reprojection_worsening_px=0.5,
            min_sparse_reprojection_retained_ratio=0.80,
            min_dense_valid_match_ratio=0.40,
            max_artifact_depth_risk_score=0.50,
        ),
    )

    assert decision["decision"] == "reject_dense_keep_sparse"
    assert decision["sparse_support_strength"] == "high"
    assert decision["dense_worsened_proxy"] is True
    assert decision["failed_checks"]["pose_translation_trust_region"] is True
    assert decision["failed_checks"]["sparse_reprojection_preservation"] is True
    assert decision["failed_checks"]["dense_valid_match_ratio"] is True
    assert decision["failed_checks"]["artifact_depth_risk"] is True
    assert decision["selected_pose_source"] == "sparse"


def test_guard_accepts_high_confidence_sparse_when_dense_stays_inside_trust_region() -> None:
    sparse_pose, _intrinsic, _image_size = _camera()
    dense_pose = sparse_pose.copy()
    dense_pose[0, 3] = 0.005

    decision = should_accept_dense(
        sparse_pose,
        dense_pose,
        sparse_inliers=_sparse_inliers(),
        dense_stats={"valid_dense_match_ratio": 0.95, "artifact_depth_risk_score": 0.05},
        policy=DenseTransitionGuardPolicy(
            high_confidence_min_inliers=4,
            high_confidence_min_inlier_ratio=0.5,
            max_translation_delta_m=0.10,
            max_sparse_reprojection_error_px=2.0,
            min_sparse_reprojection_retained_ratio=0.80,
            min_dense_valid_match_ratio=0.40,
            max_artifact_depth_risk_score=0.50,
        ),
    )

    assert decision["decision"] == "accept_dense"
    assert decision["dense_worsened_proxy"] is False
    assert decision["selected_pose_source"] == "dense"
    assert decision["checks"]["sparse_reprojection"]["retained_ratio"] == 1.0


def test_report_counts_proxy_worsening_from_summary_style_records() -> None:
    report = build_dense_transition_report(
        [
            {
                "query_id": "risky",
                "sparse_inlier_count": 80,
                "sparse_inlier_ratio": 0.8,
                "translation_delta_m": 0.25,
                "rotation_delta_deg": 1.0,
                "retained_sparse_inlier_ratio": 0.45,
                "valid_dense_match_ratio": 0.75,
                "artifact_depth_risk_score": 0.10,
            },
            {
                "query_id": "safe",
                "sparse_inlier_count": 90,
                "sparse_inlier_ratio": 0.85,
                "translation_delta_m": 0.02,
                "rotation_delta_deg": 0.5,
                "retained_sparse_inlier_ratio": 0.98,
                "valid_dense_match_ratio": 0.85,
                "artifact_depth_risk_score": 0.15,
            },
            {
                "query_id": "weak_sparse",
                "sparse_inlier_count": 3,
                "sparse_inlier_ratio": 0.1,
                "translation_delta_m": 1.5,
                "rotation_delta_deg": 12.0,
                "retained_sparse_inlier_ratio": 0.0,
                "valid_dense_match_ratio": 0.05,
                "artifact_depth_risk_score": 0.95,
            },
        ],
        policy=DenseTransitionGuardPolicy(
            high_confidence_min_inliers=10,
            high_confidence_min_inlier_ratio=0.5,
            max_translation_delta_m=0.10,
            max_rotation_delta_deg=3.0,
            min_sparse_reprojection_retained_ratio=0.80,
            min_dense_valid_match_ratio=0.40,
            max_artifact_depth_risk_score=0.50,
        ),
        scene="ShopFacade",
        split_name="train",
    )

    assert report["schema"] == "loc_gs_dense_transition_guard_report_v1"
    assert report["summary"]["record_count"] == 3
    assert report["summary"]["accepted_count"] == 2
    assert report["summary"]["rejected_count"] == 1
    assert report["summary"]["dense_worsened_proxy_count"] == 1
    assert report["decisions"][0]["decision"] == "reject_dense_keep_sparse"
    assert report["decisions"][2]["sparse_support_strength"] == "weak"


def test_cli_writes_report_artifacts_from_result_json(tmp_path: Path) -> None:
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "report"
    input_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "query_id": "q0",
                        "sparse_inlier_count": 40,
                        "sparse_inlier_ratio": 0.7,
                        "translation_delta_m": 0.2,
                        "rotation_delta_deg": 0.5,
                        "retained_sparse_inlier_ratio": 0.2,
                        "valid_dense_match_ratio": 0.8,
                        "artifact_depth_risk_score": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_dense_transition_report",
            "--input",
            str(input_path),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train",
            "--high_confidence_min_inliers",
            "10",
        ],
        check=True,
    )

    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["summary"]["dense_worsened_proxy_count"] == 1
    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "command.txt").exists()
    assert manifest["uses_gt"] is False
    assert manifest["required_integration_hook"] == "dense_stats_and_sparse_inliers"
