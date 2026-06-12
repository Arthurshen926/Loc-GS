import json
from pathlib import Path

import torch

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.candidate_completion_runner import build_candidate_shards_from_completion_plan
from loc_gs.scripts.build_internal_candidate_shards_from_query_cache import main


def _write_base_artifact(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 2,
            "split_audit": {"audit_status": "passed"},
        },
        "base_gaussian_id": torch.tensor([10, 20, 30], dtype=torch.int64),
        "base_landmark_desc": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
            dtype=torch.float32,
        ),
        "query_yx": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[0, 1]], dtype=torch.int64),
        "cosine": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "label": torch.tensor([-1], dtype=torch.int64),
        "candidate_mask": torch.ones((1, 2), dtype=torch.bool),
        "query_id": ["seed.png::kp0"],
        "image_id": ["seed.png"],
        "keypoint_id": ["kp0"],
    }
    torch.save(payload, path)
    return path


def _write_query_feature_cache(path: Path) -> Path:
    payload = {
        "metadata": {"scene": "GreatCourt", "split_name": "train_dev_seed13_20p"},
        "image_id": ["missing_a.png", "missing_b.png"],
        "keypoint_id": ["kp0", "kp0"],
        "query_yx": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        "query_desc": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "query_score": torch.tensor([0.9, 0.8], dtype=torch.float32),
    }
    torch.save(payload, path)
    return path


def _write_completion_plan(path: Path) -> Path:
    shards = [
        {
            "schema_version": "internal_candidate_completion_shard_v1",
            "shard_id": "candidate_completion_shard_000000",
            "scene": "GreatCourt",
            "split_name": "train_dev_seed13_20p",
            "query_ids": ["missing_a.png"],
            "query_count": 1,
        },
        {
            "schema_version": "internal_candidate_completion_shard_v1",
            "shard_id": "candidate_completion_shard_000001",
            "scene": "GreatCourt",
            "split_name": "train_dev_seed13_20p",
            "query_ids": ["missing_b.png"],
            "query_count": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(shard, sort_keys=True) for shard in shards) + "\n", encoding="utf-8")
    return path


def test_candidate_completion_runner_builds_all_shards_from_query_cache(tmp_path: Path):
    output_dir = tmp_path / "out"

    summary = build_candidate_shards_from_completion_plan(
        completion_plan=_write_completion_plan(tmp_path / "plan.jsonl"),
        query_feature_cache=_write_query_feature_cache(tmp_path / "query_features.pt"),
        base_candidate_artifact=_write_base_artifact(tmp_path / "base.pt"),
        output_dir=output_dir,
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        topk=2,
    )

    assert summary["schema_version"] == "internal_candidate_completion_runner_summary_v1"
    assert summary["plan_shard_count"] == 2
    assert summary["built_shard_count"] == 2
    assert summary["skipped_shard_count"] == 0
    assert summary["keypoint_count"] == 2
    artifact_paths = [Path(path) for path in summary["output_artifacts"]]
    assert len(artifact_paths) == 2
    assert load_listwise_candidate_artifact(artifact_paths[0]).keypoint_count == 1
    assert load_listwise_candidate_artifact(artifact_paths[1]).keypoint_count == 1


def test_candidate_completion_runner_cli_writes_paths_and_audit_material(tmp_path: Path):
    out = tmp_path / "out"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev_seed13_20p",
            "--completion_plan",
            str(_write_completion_plan(tmp_path / "plan.jsonl")),
            "--query_feature_cache",
            str(_write_query_feature_cache(tmp_path / "query_features.pt")),
            "--base_candidate_artifact",
            str(_write_base_artifact(tmp_path / "base.pt")),
            "--output_dir",
            str(out),
            "--topk",
            "2",
            "--landmark_chunk_size",
            "2",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    shard_paths = (out / "candidate_shard_paths.txt").read_text(encoding="utf-8").strip().splitlines()
    assert summary["built_shard_count"] == 2
    assert summary["landmark_chunk_size"] == 2
    assert len(shard_paths) == 2
    assert manifest["schema_version"] == "internal_candidate_completion_runner_manifest_v1"
    assert manifest["hyperparameters"]["landmark_chunk_size"] == 2
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert split_audit["audit_status"] == "passed"
