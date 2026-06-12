import json
from pathlib import Path

import pytest
import torch

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.candidate_shard_builder import build_candidate_shard_artifact
from loc_gs.scripts.build_internal_candidate_shard_artifact import main


def _write_base_artifact(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 2,
            "split_audit": {"audit_status": "passed", "checks": {}},
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
        "source_phase": ["train"],
    }
    torch.save(payload, path)
    return path


def _write_large_base_artifact(path: Path) -> Path:
    base_desc = torch.arange(64, dtype=torch.float32).reshape(32, 2)
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 2,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.arange(32, dtype=torch.int64),
        "base_landmark_desc": base_desc,
        "query_yx": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[0, 1]], dtype=torch.int64),
        "cosine": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "label": torch.tensor([-1], dtype=torch.int64),
        "candidate_mask": torch.ones((1, 2), dtype=torch.bool),
        "query_id": ["seed.png::kp0"],
        "image_id": ["seed.png"],
        "keypoint_id": ["kp0"],
        "source_phase": ["train"],
    }
    torch.save(payload, path)
    return path


def _write_ranked_base_artifact(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 3,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.arange(6, dtype=torch.int64) + 100,
        "base_landmark_desc": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ],
            dtype=torch.float32,
        ),
        "query_yx": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[0, 1, 2]], dtype=torch.int64),
        "cosine": torch.tensor([[1.0, 0.9, 0.1]], dtype=torch.float32),
        "label": torch.tensor([-1], dtype=torch.int64),
        "candidate_mask": torch.ones((1, 3), dtype=torch.bool),
        "query_id": ["seed.png::kp0"],
        "image_id": ["seed.png"],
        "keypoint_id": ["kp0"],
        "source_phase": ["train"],
    }
    torch.save(payload, path)
    return path


def _write_query_feature_cache(path: Path) -> Path:
    payload = {
        "metadata": {"scene": "GreatCourt", "split_name": "train_dev_seed13_20p"},
        "image_id": ["missing_a.png", "missing_a.png", "not_in_shard.png"],
        "keypoint_id": ["kp0", "kp1", "kp0"],
        "query_yx": torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32),
        "query_desc": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=torch.float32),
        "query_score": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
    }
    torch.save(payload, path)
    return path


def _write_ranked_query_feature_cache(path: Path) -> Path:
    payload = {
        "metadata": {"scene": "GreatCourt", "split_name": "train_dev_seed13_20p"},
        "image_id": ["missing_a.png", "missing_a.png"],
        "keypoint_id": ["kp0", "kp1"],
        "query_yx": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        "query_desc": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        "query_score": torch.tensor([0.9, 0.8], dtype=torch.float32),
    }
    torch.save(payload, path)
    return path


def _storage_nbytes(tensor: torch.Tensor) -> int:
    if hasattr(tensor, "untyped_storage"):
        return int(tensor.untyped_storage().nbytes())
    return int(tensor.storage().size() * tensor.element_size())


def _shard() -> dict[str, object]:
    return {
        "schema_version": "internal_candidate_completion_shard_v1",
        "shard_id": "candidate_completion_shard_000000",
        "scene": "GreatCourt",
        "split_name": "train_dev_seed13_20p",
        "query_ids": ["missing_a.png"],
        "query_count": 1,
    }


def test_candidate_shard_builder_generates_listwise_artifact_from_query_features(tmp_path: Path):
    output = tmp_path / "candidate_shard.pt"

    summary = build_candidate_shard_artifact(
        completion_shard=_shard(),
        query_feature_cache=_write_query_feature_cache(tmp_path / "query_features.pt"),
        base_candidate_artifact=_write_base_artifact(tmp_path / "base.pt"),
        output_artifact=output,
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        topk=2,
    )

    payload = torch.load(output, map_location="cpu")
    artifact = load_listwise_candidate_artifact(output)
    assert summary["schema_version"] == "internal_candidate_shard_artifact_summary_v1"
    assert summary["query_count"] == 1
    assert summary["keypoint_count"] == 2
    assert summary["missing_feature_query_count"] == 0
    assert artifact.batch_count == 1
    assert artifact.batches[0].query_id == "missing_a.png"
    assert payload["query_id"] == ["missing_a.png::kp0", "missing_a.png::kp1"]
    assert payload["image_id"] == ["missing_a.png", "missing_a.png"]
    assert payload["landmark_id"].tolist() == [[0, 2], [1, 2]]
    assert payload["base_gaussian_id"].tolist() == [10, 20, 30]
    assert payload["label"].tolist() == [-1, -1]
    assert payload["metadata"]["source"] == "internal_candidate_shard_builder"


def test_candidate_shard_builder_clones_limited_base_landmark_storage(tmp_path: Path):
    output = tmp_path / "candidate_shard.pt"

    build_candidate_shard_artifact(
        completion_shard=_shard(),
        query_feature_cache=_write_query_feature_cache(tmp_path / "query_features.pt"),
        base_candidate_artifact=_write_large_base_artifact(tmp_path / "base.pt"),
        output_artifact=output,
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        topk=2,
        max_landmarks=3,
    )

    payload = torch.load(output, map_location="cpu")
    base_desc = payload["base_landmark_desc"]
    base_ids = payload["base_gaussian_id"]
    assert tuple(base_desc.shape) == (3, 2)
    assert tuple(base_ids.shape) == (3,)
    assert _storage_nbytes(base_desc) == base_desc.numel() * base_desc.element_size()
    assert _storage_nbytes(base_ids) == base_ids.numel() * base_ids.element_size()


def test_candidate_shard_builder_chunked_topk_matches_full_bank(tmp_path: Path):
    base = _write_ranked_base_artifact(tmp_path / "base.pt")
    query = _write_ranked_query_feature_cache(tmp_path / "query_features.pt")
    full_output = tmp_path / "candidate_shard_full.pt"
    chunked_output = tmp_path / "candidate_shard_chunked.pt"

    build_candidate_shard_artifact(
        completion_shard=_shard(),
        query_feature_cache=query,
        base_candidate_artifact=base,
        output_artifact=full_output,
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        topk=3,
    )
    chunked_summary = build_candidate_shard_artifact(
        completion_shard=_shard(),
        query_feature_cache=query,
        base_candidate_artifact=base,
        output_artifact=chunked_output,
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        topk=3,
        landmark_chunk_size=2,
    )

    full_payload = torch.load(full_output, map_location="cpu")
    chunked_payload = torch.load(chunked_output, map_location="cpu")
    assert chunked_summary["landmark_chunk_size"] == 2
    assert chunked_payload["metadata"]["landmark_chunk_size"] == 2
    assert chunked_payload["landmark_id"].tolist() == full_payload["landmark_id"].tolist()
    assert torch.allclose(chunked_payload["cosine"], full_payload["cosine"])


def test_candidate_shard_builder_can_omit_repeated_base_landmark_descriptors(tmp_path: Path):
    output = tmp_path / "candidate_shard.pt"

    summary = build_candidate_shard_artifact(
        completion_shard=_shard(),
        query_feature_cache=_write_query_feature_cache(tmp_path / "query_features.pt"),
        base_candidate_artifact=_write_base_artifact(tmp_path / "base.pt"),
        output_artifact=output,
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        topk=2,
        include_base_landmark_desc=False,
    )

    payload = torch.load(output, map_location="cpu")
    artifact = load_listwise_candidate_artifact(output)
    assert summary["base_landmark_desc_included"] is False
    assert artifact.keypoint_count == 2
    assert "base_gaussian_id" in payload
    assert "base_landmark_desc" not in payload
    assert tuple(payload["landmark_desc"].shape) == (2, 2, 2)


def test_candidate_shard_builder_rejects_test_split(tmp_path: Path):
    shard = dict(_shard())
    shard["split_name"] = "test"
    with pytest.raises(ValueError, match="test split"):
        build_candidate_shard_artifact(
            completion_shard=shard,
            query_feature_cache=_write_query_feature_cache(tmp_path / "query_features.pt"),
            base_candidate_artifact=_write_base_artifact(tmp_path / "base.pt"),
            output_artifact=tmp_path / "candidate_shard.pt",
            scene="GreatCourt",
            split_name="test",
            topk=2,
        )


def test_candidate_shard_builder_cli_writes_manifest_summary_and_artifact(tmp_path: Path):
    shard_path = tmp_path / "completion_shard.json"
    shard_path.write_text(json.dumps(_shard()), encoding="utf-8")
    out = tmp_path / "out"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev_seed13_20p",
            "--completion_shard",
            str(shard_path),
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
            "--omit_base_landmark_desc",
        ]
    )

    assert rc == 0
    payload = torch.load(out / "candidate_shard.pt", map_location="cpu")
    artifact = load_listwise_candidate_artifact(out / "candidate_shard.pt")
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert artifact.keypoint_count == 2
    assert summary["keypoint_count"] == 2
    assert summary["landmark_chunk_size"] == 2
    assert summary["base_landmark_desc_included"] is False
    assert "base_landmark_desc" not in payload
    assert manifest["schema_version"] == "internal_candidate_shard_artifact_manifest_v1"
    assert manifest["hyperparameters"]["landmark_chunk_size"] == 2
    assert manifest["hyperparameters"]["include_base_landmark_desc"] is False
    assert manifest["dense_inference_enabled"] is False
    assert split_audit["audit_status"] == "passed"
    assert (out / "command.txt").is_file()
