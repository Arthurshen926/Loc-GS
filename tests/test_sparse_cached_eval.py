import json
import struct
from pathlib import Path

import numpy as np
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.sparse.cached_eval import CachedSparseEvalConfig, run_cached_sparse_eval


def _write_ply(path: Path) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 6\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    rows = [
        (-0.5, -0.4, 3.0),
        (0.5, -0.4, 3.1),
        (-0.4, 0.5, 2.9),
        (0.4, 0.5, 3.3),
        (0.0, 0.0, 2.6),
        (-0.7, 0.1, 3.4),
    ]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<3f", *row))
    return path


def _write_cameras(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "img_name": "a.png",
                    "width": 240,
                    "height": 180,
                    "fx": 140.0,
                    "fy": 140.0,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
                {
                    "img_name": "b.png",
                    "width": 240,
                    "height": 180,
                    "fx": 140.0,
                    "fy": 140.0,
                    "position": [0.02, -0.01, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_pair_cache(path: Path, cameras: Path) -> Path:
    points = np.array(
        [
            (-0.5, -0.4, 3.0),
            (0.5, -0.4, 3.1),
            (-0.4, 0.5, 2.9),
            (0.4, 0.5, 3.3),
            (0.0, 0.0, 2.6),
            (-0.7, 0.1, 3.4),
        ],
        dtype=np.float64,
    )
    records = load_camera_records(cameras, target_width=120, target_height=90, missing_principal_point="pixel_center")
    all_yx: list[list[float]] = []
    image_ids: list[str] = []
    query_ids: list[str] = []
    for image_id in ("a.png", "b.png"):
        keypoints_xy, valid = project_points_w2c(points, records[image_id].pose_w2c, records[image_id].intrinsics)
        assert bool(valid.all())
        for idx, xy in enumerate(keypoints_xy):
            all_yx.append([float(xy[1]), float(xy[0])])
            image_ids.append(image_id)
            query_ids.append(f"{image_id}::kp{idx}")
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train_dev",
            "topk": 1,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.arange(6, dtype=torch.int64),
        "query_yx": torch.tensor(all_yx, dtype=torch.float32),
        "landmark_id": torch.arange(6, dtype=torch.int64).repeat(2).reshape(12, 1),
        "cosine": torch.ones((12, 1), dtype=torch.float32),
        "label": torch.zeros(12, dtype=torch.int64),
        "candidate_mask": torch.ones((12, 1), dtype=torch.bool),
        "query_id": query_ids,
        "image_id": image_ids,
        "keypoint_id": [qid.rsplit("::", 1)[1] for qid in query_ids],
        "source_phase": ["train_dev"] * 12,
    }
    torch.save(payload, path)
    return path


def test_cached_sparse_eval_reports_verified_pose_metrics(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")

    summary, rows = run_cached_sparse_eval(
        scene="GreatCourt",
        split_name="train_dev",
        candidate_artifact=_write_pair_cache(tmp_path / "pairs.pt", cameras),
        point_cloud=_write_ply(tmp_path / "point_cloud.ply"),
        cameras_json=cameras,
        cfg=CachedSparseEvalConfig(
            image_width=120,
            image_height=90,
            max_queries=2,
            max_keypoints=16,
            pnp_method="epnp",
            second_pnp_enabled=True,
            second_pnp_method="iterative",
            refine_with_inliers=True,
            reprojection_error_px=1.0,
        ),
    )

    assert summary["schema_version"] == "internal_sparse_cached_eval_metrics_v1"
    assert summary["scene"] == "GreatCourt"
    assert summary["split_name"] == "train_dev"
    assert summary["pose_metric_status"] == "verified"
    assert summary["query_count"] == 2
    assert summary["success_count"] == 2
    assert summary["recall_10cm_5d"] == 1.0
    assert summary["recall_5cm_5d"] == 1.0
    assert summary["pnp_stage_count_median"] == 2.0
    assert len(rows) == 2
    assert rows[0]["te_cm"] < 1.0
