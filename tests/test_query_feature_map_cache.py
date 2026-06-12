import json
from pathlib import Path

import pytest
import torch

from loc_gs.sparse.query_feature_cache import build_query_feature_cache_from_feature_map_cache
from loc_gs.scripts.build_internal_query_feature_cache_from_maps import main


def _descriptor_map() -> torch.Tensor:
    desc = torch.zeros((2, 3, 3), dtype=torch.float32)
    desc[:, 0, 0] = torch.tensor([1.0, 0.0])
    desc[:, 1, 2] = torch.tensor([0.5, 0.5])
    desc[:, 2, 1] = torch.tensor([0.0, 1.0])
    return desc


def _write_feature_map_cache(path: Path, *, split_name: str = "train_dev") -> Path:
    payload = {
        "metadata": {
            "schema_version": "internal_query_feature_map_cache_v1",
            "scene": "GreatCourt",
            "split_name": split_name,
            "split_audit": {"audit_status": "passed"},
        },
        "entries": [
            {
                "image_id": "q1.png",
                "descriptor_map": _descriptor_map(),
                "score_map": torch.tensor(
                    [
                        [0.7, 0.1, 0.0],
                        [0.0, 0.2, 0.6],
                        [0.3, 0.9, 0.4],
                    ],
                    dtype=torch.float32,
                ),
                "keypoints_yx": torch.tensor([[0.0, 0.0], [2.0, 1.0]], dtype=torch.float32),
                "keypoint_scores": torch.tensor([0.7, 0.9], dtype=torch.float32),
            },
            {
                "image_id": "q2.png",
                "descriptor_map": _descriptor_map(),
                "score_map": torch.tensor(
                    [
                        [0.1, 0.8, 0.0],
                        [0.0, 0.2, 0.6],
                        [0.3, 0.4, 0.5],
                    ],
                    dtype=torch.float32,
                ),
            },
        ],
    }
    torch.save(payload, path)
    return path


def test_query_feature_cache_from_maps_samples_keypoints_and_tracks_missing_queries(tmp_path: Path):
    output = tmp_path / "query_features.pt"

    summary = build_query_feature_cache_from_feature_map_cache(
        feature_map_cache=_write_feature_map_cache(tmp_path / "feature_maps.pt"),
        output_cache=output,
        scene="GreatCourt",
        split_name="train_dev",
        query_ids=["q1.png", "missing.png"],
        max_keypoints=8,
    )

    payload = torch.load(output, map_location="cpu")
    assert summary["schema_version"] == "internal_query_feature_cache_summary_v1"
    assert summary["source_feature_map_cache"] == str(tmp_path / "feature_maps.pt")
    assert summary["requested_query_count"] == 2
    assert summary["query_count"] == 1
    assert summary["keypoint_count"] == 2
    assert summary["missing_query_count"] == 1
    assert payload["metadata"]["source"] == "internal_feature_map_cache_sampler"
    assert payload["image_id"] == ["q1.png", "q1.png"]
    assert payload["keypoint_id"] == ["kp_000000", "kp_000001"]
    assert payload["query_yx"].tolist() == [[2.0, 1.0], [0.0, 0.0]]
    assert payload["query_score"].tolist() == pytest.approx([0.9, 0.7])
    assert payload["query_desc"].tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_query_feature_cache_from_maps_can_select_top_score_map_points(tmp_path: Path):
    output = tmp_path / "query_features.pt"

    summary = build_query_feature_cache_from_feature_map_cache(
        feature_map_cache=_write_feature_map_cache(tmp_path / "feature_maps.pt"),
        output_cache=output,
        scene="GreatCourt",
        split_name="train_dev",
        query_ids=["q2.png"],
        max_keypoints=2,
    )

    payload = torch.load(output, map_location="cpu")
    assert summary["query_count"] == 1
    assert summary["keypoint_count"] == 2
    assert payload["image_id"] == ["q2.png", "q2.png"]
    assert payload["query_yx"].tolist() == [[0.0, 1.0], [1.0, 2.0]]
    assert payload["query_score"].tolist() == pytest.approx([0.8, 0.6])
    assert payload["query_desc"].shape == (2, 2)


def test_query_feature_cache_from_maps_rejects_test_split(tmp_path: Path):
    with pytest.raises(ValueError, match="test split"):
        build_query_feature_cache_from_feature_map_cache(
            feature_map_cache=_write_feature_map_cache(tmp_path / "feature_maps.pt", split_name="test"),
            output_cache=tmp_path / "query_features.pt",
            scene="GreatCourt",
            split_name="test",
        )


def test_query_feature_map_cache_cli_writes_manifest_summary_and_cache(tmp_path: Path):
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("q1.png\nmissing.png\n", encoding="utf-8")
    out = tmp_path / "out"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--feature_map_cache",
            str(_write_feature_map_cache(tmp_path / "feature_maps.pt")),
            "--query_ids",
            str(query_ids),
            "--output_dir",
            str(out),
            "--max_keypoints",
            "4",
        ]
    )

    assert rc == 0
    payload = torch.load(out / "query_features.pt", map_location="cpu")
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert payload["image_id"] == ["q1.png", "q1.png"]
    assert summary["missing_query_count"] == 1
    assert manifest["schema_version"] == "internal_query_feature_map_cache_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert split_audit["audit_status"] == "passed"
