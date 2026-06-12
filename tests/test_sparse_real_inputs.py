import json
import struct
from pathlib import Path

import numpy as np
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.landmarks import CacheLandmarkResolver, load_gaussian_landmark_map
from loc_gs.sparse.real_inputs import (
    CachedSparseInputConfig,
    sparse_input_from_cached_batch,
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
    rows = [
        (-0.2, -0.1, 2.0),
        (0.2, -0.1, 2.2),
        (-0.2, 0.2, 2.4),
        (0.2, 0.2, 2.6),
    ]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<3f", *row))
    return path


def _write_pair_cache(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 2,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        "query_yx": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[3, 1], [0, 2]], dtype=torch.int64),
        "cosine": torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32),
        "label": torch.tensor([1, 0], dtype=torch.int64),
        "candidate_mask": torch.tensor([[True, True], [True, True]], dtype=torch.bool),
        "query_id": ["img.png::kp0", "img.png::kp1"],
        "image_id": ["img.png", "img.png"],
        "keypoint_id": ["kp0", "kp1"],
        "source_phase": ["train", "train"],
    }
    torch.save(payload, path)
    return path


def test_camera_records_load_cambridge_json_intrinsics(tmp_path: Path):
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
        json.dumps(
            [
                {
                    "img_name": "img.png",
                    "width": 1920,
                    "height": 1080,
                    "fx": 1665.0,
                    "fy": 1664.0,
                    "position": [1.0, 2.0, 3.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_camera_records(cameras)

    assert records["img.png"].intrinsics.width == 1920
    assert records["img.png"].intrinsics.cx == 960.0
    assert records["img.png"].intrinsics.cy == 540.0
    np.testing.assert_allclose(records["img.png"].pose_w2c[:3, 3], [-1.0, -2.0, -3.0])


def test_cached_batch_converts_to_sparse_localization_input_with_resolved_xyz(tmp_path: Path):
    ply = _write_ply(tmp_path / "point_cloud.ply")
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    artifact = load_listwise_candidate_artifact(pair_cache)
    landmark_map = load_gaussian_landmark_map(ply)
    resolver = CacheLandmarkResolver.from_pair_cache(pair_cache, landmark_map)

    data = sparse_input_from_cached_batch(
        artifact.batches[0],
        resolver,
        intrinsics=load_camera_records_from_inline()["img.png"].intrinsics,
        cfg=CachedSparseInputConfig(score_mode="teacher_oracle"),
    )

    assert data.query_id == "img.png"
    assert data.keypoints_xy == [[20.0, 10.0], [40.0, 30.0]]
    assert len(data.candidates_by_keypoint) == 2
    first = data.candidates_by_keypoint[0]
    assert [candidate.landmark_id for candidate in first] == [3, 1]
    np.testing.assert_allclose(first[0].point3d, [0.2, 0.2, 2.6])
    assert first[0].solver_score == 0.0
    assert first[1].solver_score == 1.0


def load_camera_records_from_inline():
    from loc_gs.core.camera import CameraRecord, CameraIntrinsics

    return {
        "img.png": CameraRecord(
            image_id="img.png",
            intrinsics=CameraIntrinsics(width=80, height=60, fx=50.0, fy=50.0, cx=40.0, cy=30.0),
        )
    }
