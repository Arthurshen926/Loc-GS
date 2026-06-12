import json
from pathlib import Path

import torch

from loc_gs.scripts.build_internal_candidate_shard_from_feature_maps import main
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact


def _write_base_candidate_artifact(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train_dev",
            "topk": 2,
            "split_audit": {"audit_status": "passed"},
        },
        "base_gaussian_id": torch.tensor([101, 202], dtype=torch.int64),
        "base_landmark_desc": torch.eye(2, dtype=torch.float32),
        "query_desc": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "query_yx": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "landmark_desc": torch.eye(2, dtype=torch.float32).reshape(1, 2, 2),
        "landmark_id": torch.tensor([[0, 1]], dtype=torch.int64),
        "cosine": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "label": torch.tensor([0], dtype=torch.int64),
        "candidate_mask": torch.ones((1, 2), dtype=torch.bool),
        "query_id": ["seed.png::kp0"],
        "image_id": ["seed.png"],
        "keypoint_id": ["kp0"],
        "source_phase": ["train_dev"],
    }
    torch.save(payload, path)
    return path


def _write_feature_map_cache(path: Path) -> Path:
    descriptor_map = torch.zeros((2, 2, 2), dtype=torch.float32)
    descriptor_map[:, 0, 0] = torch.tensor([1.0, 0.0])
    descriptor_map[:, 1, 1] = torch.tensor([0.0, 1.0])
    payload = {
        "metadata": {
            "schema_version": "internal_query_feature_map_cache_v1",
            "scene": "GreatCourt",
            "split_name": "train_dev",
            "source": "internal_3dgs_render_feature_map_cache",
            "split_audit": {"audit_status": "passed"},
        },
        "entries": [
            {
                "image_id": "q1.png",
                "descriptor_map": descriptor_map,
                "keypoints_yx": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
                "keypoint_scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
            }
        ],
    }
    torch.save(payload, path)
    return path


def test_candidate_shard_from_feature_maps_cli_builds_query_cache_and_candidate_artifact(tmp_path: Path):
    shard = tmp_path / "completion_shard.json"
    shard.write_text(
        json.dumps(
            {
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "shard_id": "shard_000",
                "query_ids": ["q1.png", "missing.png"],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--completion_shard",
            str(shard),
            "--feature_map_cache",
            str(_write_feature_map_cache(tmp_path / "feature_maps.pt")),
            "--base_candidate_artifact",
            str(_write_base_candidate_artifact(tmp_path / "base_pairs.pt")),
            "--output_dir",
            str(out),
            "--topk",
            "2",
        ]
    )

    assert rc == 0
    query_cache = torch.load(out / "query_features.pt", map_location="cpu")
    artifact = load_listwise_candidate_artifact(out / "candidate_shard.pt")
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert query_cache["image_id"] == ["q1.png", "q1.png"]
    assert artifact.batch_count == 1
    assert artifact.keypoint_count == 2
    assert summary["query_feature_summary"]["missing_query_count"] == 1
    assert summary["candidate_shard_summary"]["missing_feature_query_count"] == 1
    assert manifest["schema_version"] == "internal_candidate_shard_from_feature_maps_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert (out / "command.txt").is_file()
