import json
from pathlib import Path

import pytest
import torch

from loc_gs.sparse.query_feature_cache import build_query_feature_cache_from_superpoint_dir
from loc_gs.scripts.build_internal_query_feature_cache_from_superpoint_dir import main


def _write_superpoint_dir(root: Path, *, scene: str = "GreatCourt") -> Path:
    (root / "descriptor").mkdir(parents=True)
    (root / "detector").mkdir(parents=True)
    descriptor = torch.zeros((2, 2, 2), dtype=torch.float32)
    descriptor[:, 0, 0] = torch.tensor([1.0, 0.0])
    descriptor[:, 0, 1] = torch.tensor([0.0, 1.0])
    detector = torch.zeros((65, 2, 2), dtype=torch.float32)
    detector[0, 0, 0] = 10.0
    detector[1, 0, 0] = 9.0
    torch.save(descriptor, root / "descriptor" / "rgb_6.pt")
    torch.save(detector, root / "detector" / "rgb_6.pt")
    (root / "frame_manifest.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "feature_type": "superpoint",
                "descriptor_dim": 2,
                "frames": [
                    {
                        "source_file": "frame00006.png",
                        "saved_stem": "rgb_6",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_query_feature_cache_from_superpoint_dir_samples_detector_keypoints_and_tracks_missing_queries(
    tmp_path: Path,
):
    superpoint_dir = _write_superpoint_dir(tmp_path / "sp")
    output = tmp_path / "query_features.pt"

    summary = build_query_feature_cache_from_superpoint_dir(
        superpoint_dir=superpoint_dir,
        output_cache=output,
        scene="GreatCourt",
        split_name="train_dev",
        query_ids=["seq2/frame00006.png", "seq2/missing.png"],
        image_id_prefix="seq2",
        max_keypoints=1,
    )

    payload = torch.load(output, map_location="cpu")
    assert summary["schema_version"] == "internal_query_feature_cache_summary_v1"
    assert summary["source_superpoint_dir"] == str(superpoint_dir)
    assert summary["requested_query_count"] == 2
    assert summary["query_count"] == 1
    assert summary["keypoint_count"] == 1
    assert summary["missing_query_count"] == 1
    assert payload["metadata"]["source"] == "internal_superpoint_feature_dir_sampler"
    assert payload["image_id"] == ["seq2/frame00006.png"]
    assert payload["keypoint_id"] == ["kp_000000"]
    assert torch.allclose(payload["query_yx"], torch.tensor([[0.0, 0.0]], dtype=torch.float32))
    assert payload["query_score"].shape == (1,)
    assert payload["query_score"][0].item() > 0.0
    assert torch.allclose(payload["query_desc"], torch.tensor([[1.0, 0.0]], dtype=torch.float32))


def test_query_feature_cache_from_superpoint_dir_rejects_test_split(tmp_path: Path):
    with pytest.raises(ValueError, match="test split"):
        build_query_feature_cache_from_superpoint_dir(
            superpoint_dir=_write_superpoint_dir(tmp_path / "sp"),
            output_cache=tmp_path / "query_features.pt",
            scene="GreatCourt",
            split_name="test",
        )


def test_query_feature_cache_from_superpoint_dir_cli_writes_manifest_summary_and_cache(tmp_path: Path):
    superpoint_dir = _write_superpoint_dir(tmp_path / "sp")
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("seq2/frame00006.png\nseq2/missing.png\n", encoding="utf-8")
    out = tmp_path / "out"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--superpoint_dir",
            str(superpoint_dir),
            "--query_ids",
            str(query_ids),
            "--image_id_prefix",
            "seq2",
            "--output_dir",
            str(out),
            "--max_keypoints",
            "1",
        ]
    )

    assert rc == 0
    payload = torch.load(out / "query_features.pt", map_location="cpu")
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert payload["image_id"] == ["seq2/frame00006.png"]
    assert summary["missing_query_count"] == 1
    assert manifest["schema_version"] == "internal_query_superpoint_dir_cache_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert split_audit["audit_status"] == "passed"
