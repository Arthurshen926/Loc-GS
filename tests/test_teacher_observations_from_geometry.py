from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.scripts.build_internal_teacher_observations_from_geometry import main
from loc_gs.teacher.geometric_observations import build_teacher_observations_from_geometry


POINTS = np.array(
    [
        (-0.5, -0.4, 3.0),
        (0.5, -0.4, 3.1),
        (-0.4, 0.5, 2.9),
        (0.4, 0.5, 3.3),
    ],
    dtype=np.float64,
)


def _write_ply(path: Path) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 4\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in POINTS:
            handle.write(struct.pack("<3f", *row))
    return path


def _write_cameras(path: Path, *, split_name: str = "train") -> Path:
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
                    "split_name": split_name,
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_candidate_artifact(
    path: Path,
    cameras: Path,
    *,
    split_name: str = "train",
    target_width: int | None = None,
    target_height: int | None = None,
) -> Path:
    camera = load_camera_records(
        cameras,
        target_width=target_width,
        target_height=target_height,
        missing_principal_point="pixel_center",
    )["img.png"]
    projected_xy, valid = project_points_w2c(POINTS, camera.pose_w2c, camera.intrinsics)
    assert bool(valid.all())
    query_yx = np.stack([projected_xy[:, 1], projected_xy[:, 0]], axis=1)
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": split_name,
            "topk": 2,
            "split_audit": {"audit_status": "passed"},
        },
        "base_gaussian_id": torch.arange(4, dtype=torch.int64),
        "query_yx": torch.tensor(query_yx, dtype=torch.float32),
        "landmark_id": torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.int64),
        "cosine": torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=torch.float32),
        "label": torch.zeros((4,), dtype=torch.int64),
        "reprojection_error": torch.tensor(
            [
                [0.0, 1.0e3],
                [0.0, 1.0e3],
                [0.0, 1.0e3],
                [0.0, 1.0e3],
            ],
            dtype=torch.float32,
        ),
        "candidate_mask": torch.ones((4, 2), dtype=torch.bool),
        "query_id": [f"img.png::kp_{idx:06d}" for idx in range(4)],
        "image_id": ["img.png"] * 4,
        "keypoint_id": [f"kp_{idx:06d}" for idx in range(4)],
        "source_phase": [split_name] * 4,
    }
    torch.save(payload, path)
    return path


def test_build_teacher_observations_from_geometry_projects_candidates_and_assigns_roles(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    observations, summary = build_teacher_observations_from_geometry(
        candidate_artifact=_write_candidate_artifact(tmp_path / "pairs.pt", cameras),
        point_cloud=_write_ply(tmp_path / "point_cloud.ply"),
        cameras_json=cameras,
        scene="GreatCourt",
        split_name="train",
        dense_consistency_reprojection_px=1.0,
        sparse_inlier_reprojection_px=2.0,
        hard_negative_reprojection_px=8.0,
    )

    assert summary["schema_version"] == "internal_geometry_teacher_observation_summary_v1"
    assert summary["candidate_row_count"] == 4
    assert summary["observation_count"] == 8
    assert summary["dense_consistent_count"] == 4
    assert summary["hard_negative_count"] == 4
    first_row = [row for row in observations if row["keypoint_id"] == "kp_000000"]
    assert first_row[0]["candidate_rank"] == 0
    assert first_row[0]["dense_consistent"] is True
    assert first_row[0]["label_role"] == "protected_support"
    assert first_row[1]["dense_consistent"] is False
    assert first_row[1]["label_role"] == "hard_negative"


def test_build_teacher_observations_from_geometry_auto_calibrates_resized_cache_frame(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    observations, summary = build_teacher_observations_from_geometry(
        candidate_artifact=_write_candidate_artifact(
            tmp_path / "pairs.pt",
            cameras,
            target_width=120,
            target_height=90,
        ),
        point_cloud=_write_ply(tmp_path / "point_cloud.ply"),
        cameras_json=cameras,
        scene="GreatCourt",
        split_name="train",
        dense_consistency_reprojection_px=1.0,
        sparse_inlier_reprojection_px=2.0,
    )

    assert summary["frame_calibration_status"] == "passed"
    assert summary["resolved_target_width"] == 120
    assert summary["resolved_target_height"] == 90
    assert summary["resolved_missing_principal_point"] == "pixel_center"
    assert summary["dense_consistent_count"] == 4
    assert len(observations) == 8


def test_build_teacher_observations_from_geometry_rejects_test_split(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    with pytest.raises(ValueError, match="test split"):
        build_teacher_observations_from_geometry(
            candidate_artifact=_write_candidate_artifact(tmp_path / "pairs.pt", cameras, split_name="test"),
            point_cloud=_write_ply(tmp_path / "point_cloud.ply"),
            cameras_json=cameras,
            scene="GreatCourt",
            split_name="test",
        )


def test_teacher_observations_from_geometry_cli_writes_auditable_bundle(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    out = tmp_path / "observations"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(_write_candidate_artifact(tmp_path / "pairs.pt", cameras)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--output_dir",
            str(out),
            "--dense_consistency_reprojection_px",
            "1.0",
        ]
    )

    assert rc == 0
    rows = [json.loads(line) for line in (out / "teacher_observations.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert len(rows) == 8
    assert summary["dense_consistent_count"] == 4
    assert manifest["schema_version"] == "internal_geometry_teacher_observation_manifest_v1"
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["dense_inference_enabled"] is False
    assert split_audit["audit_status"] == "passed"
