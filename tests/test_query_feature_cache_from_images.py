import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from loc_gs.data.superpoint_query_extractor import extract_superpoint_query_feature_cache_for_queries
from loc_gs.scripts import build_internal_query_feature_cache_from_images as cli


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


def test_query_feature_cache_from_images_samples_sparse_superpoint_features(tmp_path: Path):
    data_root = tmp_path / "GreatCourt"
    _write_image(data_root / "seq2" / "frame00006.png", 32)
    output = tmp_path / "query_features.pt"

    summary = extract_superpoint_query_feature_cache_for_queries(
        scene="GreatCourt",
        split_name="train_dev",
        data_root=data_root,
        query_ids=["seq2/frame00006.png", "seq2/missing.png"],
        output_cache=output,
        model=_FakeSuperPoint(),
        device=torch.device("cpu"),
        max_keypoints=1,
        strict_missing=False,
    )

    payload = torch.load(output, map_location="cpu")
    assert summary["schema_version"] == "internal_query_feature_cache_summary_v1"
    assert summary["source"] == "internal_superpoint_image_sampler"
    assert summary["requested_query_count"] == 2
    assert summary["query_count"] == 1
    assert summary["keypoint_count"] == 1
    assert summary["missing_query_count"] == 1
    assert payload["metadata"]["source"] == "internal_superpoint_image_sampler"
    assert payload["image_id"] == ["seq2/frame00006.png"]
    assert payload["keypoint_id"] == ["kp_000000"]
    assert torch.allclose(payload["query_yx"], torch.tensor([[0.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(payload["query_desc"], torch.tensor([[1.0, 0.0]], dtype=torch.float32))


def test_query_feature_cache_from_images_cli_writes_manifest_summary_and_cache(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "GreatCourt"
    _write_image(data_root / "seq2" / "frame00006.png", 32)
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("seq2/frame00006.png\nseq2/missing.png\n", encoding="utf-8")
    out = tmp_path / "out"
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
            str(out),
            "--weights",
            str(tmp_path / "weights.pth"),
            "--device",
            "cpu",
            "--max_keypoints",
            "1",
            "--allow_missing",
        ]
    )

    assert rc == 0
    payload = torch.load(out / "query_features.pt", map_location="cpu")
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert payload["image_id"] == ["seq2/frame00006.png"]
    assert summary["keypoint_count"] == 1
    assert manifest["schema_version"] == "internal_query_feature_cache_from_images_manifest_v1"
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert split_audit["audit_status"] == "passed"
