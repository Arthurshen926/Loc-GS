import json
import struct
from pathlib import Path

import numpy as np
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.scripts.eval_internal_sparse_cached import main


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
    camera = load_camera_records(cameras, target_width=120, target_height=90, missing_principal_point="pixel_center")[
        "img.png"
    ]
    keypoints_xy, valid = project_points_w2c(points, camera.pose_w2c, camera.intrinsics)
    assert bool(valid.all())
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train_dev",
            "topk": 1,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.arange(6, dtype=torch.int64),
        "query_yx": torch.tensor([[float(xy[1]), float(xy[0])] for xy in keypoints_xy], dtype=torch.float32),
        "landmark_id": torch.arange(6, dtype=torch.int64).reshape(6, 1),
        "cosine": torch.ones((6, 1), dtype=torch.float32),
        "label": torch.zeros(6, dtype=torch.int64),
        "candidate_mask": torch.ones((6, 1), dtype=torch.bool),
        "query_id": [f"img.png::kp{i}" for i in range(6)],
        "image_id": ["img.png"] * 6,
        "keypoint_id": [f"kp{i}" for i in range(6)],
        "source_phase": ["train_dev"] * 6,
    }
    torch.save(payload, path)
    return path


def test_eval_internal_sparse_cached_cli_writes_auditable_eval_bundle(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    out = tmp_path / "eval"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--output_dir",
            str(out),
            "--max_queries",
            "1",
            "--second_pnp_enabled",
            "--refine_with_inliers",
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "internal_sparse_cached_eval_metrics_v1"
    assert metrics["pose_metric_status"] == "verified"
    assert metrics["recall_10cm_5d"] == 1.0
    assert manifest["schema_version"] == "internal_sparse_cached_eval_manifest_v1"
    assert manifest["inference_stage"] == "sparse_only"
    assert manifest["dense_inference_enabled"] is False
    assert manifest["hyperparameters"]["second_pnp_enabled"] is True
    assert rows[0]["query_id"] == "img.png"
    assert split_audit["audit_status"] == "passed"
    assert (out / "command.txt").is_file()
    assert (out / "git_status.txt").is_file()


def test_eval_internal_sparse_cached_cli_accepts_query_id_filter(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("img.png\nmissing.png\n", encoding="utf-8")
    out = tmp_path / "eval_filtered"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--query_ids",
            str(query_ids),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["query_filter_enabled"] is True
    assert metrics["requested_query_count"] == 2
    assert metrics["matched_query_count"] == 1
    assert metrics["missing_query_count"] == 1
    assert metrics["missing_query_ids_preview"] == ["missing.png"]
