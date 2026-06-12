import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.sparse.pose_map_frame_audit import audit_pose_map_frame


def _write_ply(path: Path, points: np.ndarray) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {int(points.shape[0])}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in points:
            handle.write(struct.pack("<3f", *[float(v) for v in row]))
    return path


def _write_cameras(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "img_name": "img.png",
                    "width": 240,
                    "height": 180,
                    "fx": 140.0,
                    "fy": 140.0,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_pair_cache(path: Path, keypoints_yx: np.ndarray) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 1,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.arange(int(keypoints_yx.shape[0]), dtype=torch.int64),
        "query_yx": torch.tensor(keypoints_yx, dtype=torch.float32),
        "landmark_id": torch.arange(int(keypoints_yx.shape[0]), dtype=torch.int64).reshape(-1, 1),
        "cosine": torch.ones((int(keypoints_yx.shape[0]), 1), dtype=torch.float32),
        "label": torch.zeros(int(keypoints_yx.shape[0]), dtype=torch.int64),
        "candidate_mask": torch.ones((int(keypoints_yx.shape[0]), 1), dtype=torch.bool),
        "reprojection_error": torch.zeros((int(keypoints_yx.shape[0]), 1), dtype=torch.float32),
        "query_id": [f"img.png::kp{i}" for i in range(int(keypoints_yx.shape[0]))],
        "image_id": ["img.png"] * int(keypoints_yx.shape[0]),
        "keypoint_id": [f"kp{i}" for i in range(int(keypoints_yx.shape[0]))],
        "source_phase": ["train"] * int(keypoints_yx.shape[0]),
    }
    torch.save(payload, path)
    return path


def _synthetic_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    points = np.array(
        [
            [-0.6, -0.3, 3.0],
            [0.5, -0.2, 3.2],
            [-0.4, 0.4, 2.8],
            [0.7, 0.5, 3.5],
            [0.0, 0.0, 2.7],
            [-0.8, 0.2, 3.4],
        ],
        dtype=np.float64,
    )
    cameras = _write_cameras(tmp_path / "cameras.json")
    resized_intr = load_camera_records(
        cameras,
        target_width=120,
        target_height=90,
        missing_principal_point="pixel_center",
    )["img.png"].intrinsics
    keypoints_xy, valid = project_points_w2c(points, np.eye(4, dtype=np.float64), resized_intr)
    assert bool(valid.all())
    keypoints_yx = np.stack([keypoints_xy[:, 1], keypoints_xy[:, 0]], axis=1)
    return (
        _write_pair_cache(tmp_path / "pairs.pt", keypoints_yx),
        _write_ply(tmp_path / "point_cloud.ply", points),
        cameras,
    )


def test_pose_map_frame_audit_identifies_resized_pixel_center_cache_frame(tmp_path: Path):
    pair_cache, point_cloud, cameras = _synthetic_fixture(tmp_path)

    summary = audit_pose_map_frame(
        candidate_artifact=pair_cache,
        point_cloud=point_cloud,
        cameras_json=cameras,
        target_width=120,
        target_height=90,
        max_rows=64,
    )

    assert summary["status"] == "passed"
    assert summary["best_frame"] == "resized_pixel_center"
    assert summary["evaluated_positive_count"] == 6
    resized = summary["frame_hypotheses"]["resized_pixel_center"]
    native = summary["frame_hypotheses"]["native_half_extent"]
    assert resized["median_abs_delta_to_stored_error_px"] < 1.0e-4
    assert resized["median_reprojection_error_px"] < 1.0e-4
    assert native["median_reprojection_error_px"] > 20.0


def test_pose_map_frame_audit_auto_tests_common_resized_camera_frames(tmp_path: Path):
    pair_cache, point_cloud, cameras = _synthetic_fixture(tmp_path)

    summary = audit_pose_map_frame(
        candidate_artifact=pair_cache,
        point_cloud=point_cloud,
        cameras_json=cameras,
        max_rows=64,
    )

    assert summary["status"] == "passed"
    assert summary["best_frame"] == "auto_120x90_pixel_center"
    assert [120, 90] in summary["auto_resize_candidates"]
    resized = summary["frame_hypotheses"]["auto_120x90_pixel_center"]
    assert resized["median_abs_delta_to_stored_error_px"] < 1.0e-4


def test_pose_map_frame_audit_rejects_test_split(tmp_path: Path):
    pair_cache, point_cloud, cameras = _synthetic_fixture(tmp_path)
    payload = torch.load(pair_cache, map_location="cpu")
    payload["metadata"]["source_split_name"] = "test"
    torch.save(payload, pair_cache)

    with pytest.raises(ValueError, match="test split"):
        audit_pose_map_frame(
            candidate_artifact=pair_cache,
            point_cloud=point_cloud,
            cameras_json=cameras,
            target_width=120,
            target_height=90,
        )


def test_pose_map_frame_audit_rejects_test_feedback_bank_split(tmp_path: Path):
    pair_cache, point_cloud, cameras = _synthetic_fixture(tmp_path)
    payload = torch.load(pair_cache, map_location="cpu")
    payload["metadata"]["split_audit"] = {
        "audit_status": "passed",
        "checks": {"feedback_bank_split": {"split_name": "test"}},
    }
    torch.save(payload, pair_cache)

    with pytest.raises(ValueError, match="test split"):
        audit_pose_map_frame(
            candidate_artifact=pair_cache,
            point_cloud=point_cloud,
            cameras_json=cameras,
            target_width=120,
            target_height=90,
        )


def test_pose_map_frame_audit_cli_writes_manifest_and_metrics(tmp_path: Path):
    pair_cache, point_cloud, cameras = _synthetic_fixture(tmp_path)
    out = tmp_path / "audit"

    from loc_gs.scripts.audit_internal_pose_map_frame import main

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(pair_cache),
            "--point_cloud",
            str(point_cloud),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--output_dir",
            str(out),
            "--max_rows",
            "64",
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "passed"
    assert metrics["best_frame"] == "resized_pixel_center"
    assert manifest["schema_version"] == "internal_pose_map_frame_audit_manifest_v1"
    assert manifest["hyperparameters"]["image_width"] == 120
    assert split_audit["split_name"] == "train"
    assert (out / "command.txt").is_file()
