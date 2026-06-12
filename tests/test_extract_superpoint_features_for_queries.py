import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loc_gs.data.superpoint_query_extractor import extract_superpoint_feature_dir_for_queries
from loc_gs.scripts import extract_internal_superpoint_features_for_queries as cli


class _FakeSuperPoint(torch.nn.Module):
    def forward(self, images: torch.Tensor):
        batch, _, height, width = images.shape
        desc = torch.zeros((batch, 2, height // 8, width // 8), dtype=torch.float32, device=images.device)
        desc[:, 0, :, :] = 1.0
        det = torch.zeros((batch, 65, height // 8, width // 8), dtype=torch.float32, device=images.device)
        det[:, 0, 0, 0] = 10.0
        return desc, det


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), int(value), dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_extract_superpoint_feature_dir_for_queries_writes_manifest_and_features(tmp_path: Path):
    data_root = tmp_path / "GreatCourt"
    _write_image(data_root / "seq2" / "frame00006.png", 32)
    _write_image(data_root / "seq2" / "frame00007.png", 64)
    output_dir = tmp_path / "sp"

    summary = extract_superpoint_feature_dir_for_queries(
        scene="GreatCourt",
        split_name="train_dev",
        data_root=data_root,
        query_ids=["seq2/frame00006.png", "seq2/missing.png", "seq2/frame00007.png"],
        output_dir=output_dir,
        model=_FakeSuperPoint(),
        device=torch.device("cpu"),
        batch_size=2,
        strict_missing=False,
    )

    manifest = json.loads((output_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "internal_superpoint_query_feature_extraction_summary_v1"
    assert summary["requested_query_count"] == 3
    assert summary["extracted_query_count"] == 2
    assert summary["missing_query_count"] == 1
    assert summary["external_runtime_dependency"] == "forbidden"
    assert manifest["schema_version"] == "internal_superpoint_feature_dir_v1"
    assert manifest["scene"] == "GreatCourt"
    assert manifest["split_name"] == "train_dev"
    assert [frame["source_file"] for frame in manifest["frames"]] == [
        "seq2/frame00006.png",
        "seq2/frame00007.png",
    ]
    assert [frame["saved_stem"] for frame in manifest["frames"]] == [
        "seq2__frame00006",
        "seq2__frame00007",
    ]
    descriptor = torch.load(output_dir / "descriptor" / "seq2__frame00006.pt", map_location="cpu")
    detector = torch.load(output_dir / "detector" / "seq2__frame00006.pt", map_location="cpu")
    assert descriptor.shape == (2, 2, 2)
    assert detector.shape == (65, 2, 2)


def test_extract_superpoint_feature_dir_for_queries_rejects_test_split(tmp_path: Path):
    with pytest.raises(ValueError, match="test split"):
        extract_superpoint_feature_dir_for_queries(
            scene="GreatCourt",
            split_name="test",
            data_root=tmp_path,
            query_ids=[],
            output_dir=tmp_path / "sp",
            model=_FakeSuperPoint(),
            device=torch.device("cpu"),
        )


def test_extract_superpoint_features_for_queries_cli_writes_audit_material(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "GreatCourt"
    _write_image(data_root / "seq2" / "frame00006.png", 32)
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("seq2/frame00006.png\nseq2/missing.png\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(cli, "load_superpoint_model", lambda *, weights, device: _FakeSuperPoint())

    rc = cli.main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--data_root",
            str(data_root),
            "--query_ids",
            str(query_ids),
            "--output_dir",
            str(output_dir),
            "--weights",
            str(tmp_path / "weights.pth"),
            "--device",
            "cpu",
            "--batch_size",
            "1",
            "--allow_missing",
        ]
    )

    assert rc == 0
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    command = (output_dir / "command.txt").read_text(encoding="utf-8")
    assert summary["extracted_query_count"] == 1
    assert summary["missing_query_count"] == 1
    assert manifest["schema_version"] == "internal_superpoint_query_feature_extraction_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert split_audit["audit_status"] == "passed"
    assert "extract_internal_superpoint_features_for_queries" in command
