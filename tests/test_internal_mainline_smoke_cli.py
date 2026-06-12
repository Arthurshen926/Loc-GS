import json
import struct
from pathlib import Path

import numpy as np
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.scripts.run_internal_mainline_smoke import main


POINTS = np.array(
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
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in POINTS:
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


def _projected_keypoints_yx(cameras: Path) -> np.ndarray:
    camera = load_camera_records(cameras)["img.png"]
    keypoints_xy, valid = project_points_w2c(POINTS, camera.pose_w2c, camera.intrinsics)
    assert bool(valid.all())
    return np.stack([keypoints_xy[:, 1], keypoints_xy[:, 0]], axis=1)


def _write_feature_map_cache(path: Path, cameras: Path) -> Path:
    keypoints_yx = _projected_keypoints_yx(cameras)
    descriptor_map = torch.zeros((6, 180, 240), dtype=torch.float32)
    score_map = torch.zeros((180, 240), dtype=torch.float32)
    for idx, yx in enumerate(keypoints_yx):
        y = int(round(float(yx[0])))
        x = int(round(float(yx[1])))
        descriptor_map[idx, max(0, y - 2) : min(180, y + 3), max(0, x - 2) : min(240, x + 3)] = 1.0
        score_map[y, x] = 1.0 - 0.01 * idx
    payload = {
        "metadata": {
            "schema_version": "internal_query_feature_map_cache_v1",
            "scene": "GreatCourt",
            "split_name": "train",
            "source": "synthetic_internal_mainline_smoke",
            "split_audit": {"audit_status": "passed"},
        },
        "entries": [
            {
                "image_id": "img.png",
                "descriptor_map": descriptor_map,
                "score_map": score_map,
                "keypoints_yx": torch.tensor(keypoints_yx, dtype=torch.float32),
                "keypoint_scores": torch.linspace(1.0, 0.5, steps=6),
            }
        ],
    }
    torch.save(payload, path)
    return path


def _write_base_candidate_artifact(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 1,
            "split_audit": {"audit_status": "passed"},
        },
        "base_gaussian_id": torch.arange(6, dtype=torch.int64),
        "base_landmark_desc": torch.eye(6, dtype=torch.float32),
        "query_desc": torch.eye(6, dtype=torch.float32)[:1],
        "query_yx": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "landmark_desc": torch.eye(6, dtype=torch.float32)[:1].reshape(1, 1, 6),
        "landmark_id": torch.tensor([[0]], dtype=torch.int64),
        "cosine": torch.tensor([[1.0]], dtype=torch.float32),
        "label": torch.tensor([0], dtype=torch.int64),
        "candidate_mask": torch.ones((1, 1), dtype=torch.bool),
        "query_id": ["seed.png::kp0"],
        "image_id": ["seed.png"],
        "keypoint_id": ["kp0"],
        "source_phase": ["train"],
    }
    torch.save(payload, path)
    return path


def _write_completion_shard(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "scene": "GreatCourt",
                "split_name": "train",
                "shard_id": "synthetic_smoke",
                "query_ids": ["img.png"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_teacher_observations(path: Path) -> Path:
    rows = [
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "image_id": "img.png",
            "keypoint_id": f"kp_{idx:06d}",
            "candidate_rank": 0,
            "geometric_correct": True,
            "dense_consistent": True,
            "sparse_inlier": True,
            "reprojection_error_px": 0.5,
            "solver_weight": 2.0,
        }
        for idx in range(6)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_internal_mainline_smoke_cli_runs_completion_distillation_training_and_sparse_eval(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    out = tmp_path / "mainline_smoke"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--completion_shard",
            str(_write_completion_shard(tmp_path / "completion_shard.json")),
            "--feature_map_cache",
            str(_write_feature_map_cache(tmp_path / "feature_maps.pt", cameras)),
            "--base_candidate_artifact",
            str(_write_base_candidate_artifact(tmp_path / "base_pairs.pt")),
            "--teacher_observations",
            str(_write_teacher_observations(tmp_path / "teacher_observations.jsonl")),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--output_dir",
            str(out),
            "--topk",
            "1",
            "--max_keypoints",
            "16",
            "--epochs",
            "20",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (out / "query_features.pt").is_file()
    assert (out / "candidate_shard.pt").is_file()
    assert (out / "distilled_candidates.pt").is_file()
    assert (out / "candidate_scorer.json").is_file()
    assert (out / "sparse_eval_results.json").is_file()
    assert summary["schema_version"] == "internal_mainline_smoke_summary_v1"
    assert summary["candidate_shard"]["keypoint_count"] == 6
    assert summary["distillation"]["matched_observation_count"] == 6
    assert summary["candidate_scorer"]["dense_teacher_sample_count"] == 6
    assert summary["sparse_eval"]["success_count"] == 1
    assert summary["sparse_eval"]["median_te_cm"] < 1.0
    assert manifest["inference_stage"] == "internal_mainline_smoke"
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["dense_inference_enabled"] is False


def test_internal_mainline_smoke_cli_can_generate_teacher_observations_from_geometry(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    out = tmp_path / "mainline_smoke"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--completion_shard",
            str(_write_completion_shard(tmp_path / "completion_shard.json")),
            "--feature_map_cache",
            str(_write_feature_map_cache(tmp_path / "feature_maps.pt", cameras)),
            "--base_candidate_artifact",
            str(_write_base_candidate_artifact(tmp_path / "base_pairs.pt")),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--output_dir",
            str(out),
            "--topk",
            "1",
            "--max_keypoints",
            "16",
            "--epochs",
            "20",
            "--dense_consistency_reprojection_px",
            "1.0",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out / "teacher_observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert summary["teacher_observation_source"] == "geometry"
    assert summary["distillation"]["matched_observation_count"] == 6
    assert manifest["teacher_observations"] == str(out / "teacher_observations.jsonl")
    assert manifest["hyperparameters"]["dense_consistency_reprojection_px"] == 1.0
