import json
from pathlib import Path

import pytest
import torch

from loc_gs.sparse.query_feature_cache import build_query_feature_cache_from_listwise_artifact
from loc_gs.scripts.build_internal_query_feature_cache import main


def _write_pair_cache(path: Path, *, split_name: str = "train_dev") -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": split_name,
            "topk": 2,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.tensor([10, 20], dtype=torch.int64),
        "base_landmark_desc": torch.eye(2, dtype=torch.float32),
        "query_desc": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32),
        "query_yx": torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32),
        "query_score": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
        "landmark_id": torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64),
        "cosine": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.8, 0.2]], dtype=torch.float32),
        "label": torch.tensor([0, 0, -1], dtype=torch.int64),
        "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
        "query_id": ["q1.png::kp0", "q1.png::kp1", "q2.png::kp0"],
        "image_id": ["q1.png", "q1.png", "q2.png"],
        "keypoint_id": ["kp0", "kp1", "kp0"],
        "source_phase": [split_name, split_name, split_name],
    }
    torch.save(payload, path)
    return path


def test_query_feature_cache_builder_extracts_filtered_query_descriptors(tmp_path: Path):
    output = tmp_path / "query_features.pt"

    summary = build_query_feature_cache_from_listwise_artifact(
        source_artifact=_write_pair_cache(tmp_path / "pairs.pt"),
        output_cache=output,
        scene="GreatCourt",
        split_name="train_dev",
        query_ids=["q1.png", "missing.png"],
    )

    payload = torch.load(output, map_location="cpu")
    assert summary["schema_version"] == "internal_query_feature_cache_summary_v1"
    assert summary["requested_query_count"] == 2
    assert summary["query_count"] == 1
    assert summary["keypoint_count"] == 2
    assert summary["missing_query_count"] == 1
    assert payload["image_id"] == ["q1.png", "q1.png"]
    assert payload["keypoint_id"] == ["kp0", "kp1"]
    assert payload["query_desc"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert payload["query_yx"].tolist() == [[5.0, 6.0], [7.0, 8.0]]
    assert payload["metadata"]["source"] == "internal_query_feature_cache_builder"


def test_query_feature_cache_builder_rejects_test_split(tmp_path: Path):
    with pytest.raises(ValueError, match="test split"):
        build_query_feature_cache_from_listwise_artifact(
            source_artifact=_write_pair_cache(tmp_path / "pairs.pt", split_name="test"),
            output_cache=tmp_path / "query_features.pt",
            scene="GreatCourt",
            split_name="test",
        )


def test_query_feature_cache_cli_writes_manifest_summary_and_cache(tmp_path: Path):
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("q1.png\nmissing.png\n", encoding="utf-8")
    out = tmp_path / "out"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--source_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt")),
            "--query_ids",
            str(query_ids),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    payload = torch.load(out / "query_features.pt", map_location="cpu")
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert payload["image_id"] == ["q1.png", "q1.png"]
    assert summary["missing_query_count"] == 1
    assert manifest["schema_version"] == "internal_query_feature_cache_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert split_audit["audit_status"] == "passed"
