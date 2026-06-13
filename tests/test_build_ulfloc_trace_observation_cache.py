import json
import pickle
import subprocess
import sys

import torch


def _write_source_log(path):
    path.mkdir()
    sampled_idx = torch.tensor([10, 20], dtype=torch.long)
    features = torch.eye(2, dtype=torch.float32)
    with (path / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled_idx, handle)
    with (path / "keypoints_features.pkl").open("wb") as handle:
        pickle.dump(features, handle)
    return sampled_idx


def test_build_ulfloc_trace_observation_cache_uses_per_match_query_descriptors(tmp_path):
    source_log = tmp_path / "log"
    sampled_idx = _write_source_log(source_log)
    trace = tmp_path / "trace.pt"
    torch.save(
        {
            "schema_version": "ulfloc_sparse_pnp_trace_payload_v1",
            "scene": "ShopFacade",
            "split_name": "selfmap_train_seed13_20p",
            "records": [
                {"query_id": "q1", "image_id": "im1", "gaussian_id": 10, "label": 1},
                {"query_id": "q1", "image_id": "im1", "gaussian_id": 20, "label": 0},
                {"query_id": "q2", "image_id": "im2", "gaussian_id": 20, "label": 1},
            ],
            "query_match_descriptors": {
                "q1": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                "q2": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            },
            "split_audit": {"test_split_used": False, "official_test_used": False},
        },
        trace,
    )
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_trace_observation_cache",
            "--trace_payload",
            str(trace),
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    cache = torch.load(output / "observation_cache.pt", map_location="cpu")
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))

    assert torch.equal(cache["sampled_idx"], sampled_idx)
    assert cache["observation_descriptor_source"] == "per_match_query_descriptor"
    assert cache["observation_format"] == "packed_pair_cache"
    assert "observations" not in cache
    assert cache["query_desc"].tolist() == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert cache["landmark_id"].tolist() == [[0], [1], [1]]
    assert cache["gaussian_id"].tolist() == [[10], [20], [20]]
    assert cache["row_source_view_id"] == ["im1", "im1", "im2"]
    assert metrics["observation_count"] == 3
    assert metrics["descriptor_dim"] == 2


def test_build_ulfloc_trace_observation_cache_rejects_test_split(tmp_path):
    source_log = tmp_path / "log"
    _write_source_log(source_log)
    trace = tmp_path / "trace.pt"
    torch.save(
        {
            "schema_version": "ulfloc_sparse_pnp_trace_payload_v1",
            "scene": "ShopFacade",
            "split_name": "test",
            "records": [],
            "query_match_descriptors": {},
            "split_audit": {"test_split_used": True},
        },
        trace,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_trace_observation_cache",
            "--trace_payload",
            str(trace),
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(tmp_path / "out"),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr
