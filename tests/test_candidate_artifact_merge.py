import json
from pathlib import Path

import pytest
import torch

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.candidate_artifact_merge import merge_listwise_candidate_artifacts
from loc_gs.scripts.merge_internal_candidate_artifacts import main


def _write_pair_cache(path: Path, *, image_id: str, split_name: str = "train_dev") -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": split_name,
            "topk": 2,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.tensor([10, 20, 30, 40], dtype=torch.int64),
        "base_landmark_desc": torch.ones((4, 2), dtype=torch.float16),
        "query_desc": torch.ones((2, 2), dtype=torch.float16),
        "landmark_desc": torch.ones((2, 2, 2), dtype=torch.float16),
        "cosine": torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32),
        "margin": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "query_score": torch.tensor([0.5, 0.6], dtype=torch.float32),
        "landmark_prior": torch.tensor([[0.0, 0.1], [0.2, 0.3]], dtype=torch.float32),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "query_yx": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        "label": torch.tensor([0, 1], dtype=torch.int64),
        "query_id": [f"{image_id}::kp_0000", f"{image_id}::kp_0001"],
        "image_id": [image_id, image_id],
        "keypoint_id": ["kp_0000", "kp_0001"],
        "source_phase": [split_name, split_name],
    }
    torch.save(payload, path)
    return path


def test_merge_listwise_candidate_artifacts_concatenates_rows_for_internal_eval(tmp_path: Path):
    first = _write_pair_cache(tmp_path / "first.pt", image_id="img_a.png")
    second = _write_pair_cache(tmp_path / "second.pt", image_id="img_b.png")
    output = tmp_path / "merged.pt"

    summary = merge_listwise_candidate_artifacts(
        input_artifacts=[first, second],
        output_artifact=output,
        scene="GreatCourt",
        split_name="train_dev",
    )

    payload = torch.load(output, map_location="cpu")
    artifact = load_listwise_candidate_artifact(output)
    assert summary["schema_version"] == "internal_candidate_artifact_merge_summary_v1"
    assert summary["input_artifact_count"] == 2
    assert summary["row_count"] == 4
    assert payload["query_yx"].shape == (4, 2)
    assert payload["query_id"] == [
        "img_a.png::kp_0000",
        "img_a.png::kp_0001",
        "img_b.png::kp_0000",
        "img_b.png::kp_0001",
    ]
    assert payload["base_gaussian_id"].tolist() == [10, 20, 30, 40]
    assert artifact.batch_count == 2
    assert artifact.keypoint_count == 4
    assert artifact.metadata["source_artifact_count"] == 2


def test_merge_listwise_candidate_artifacts_rejects_duplicate_query_keypoints(tmp_path: Path):
    first = _write_pair_cache(tmp_path / "first.pt", image_id="img_a.png")
    second = _write_pair_cache(tmp_path / "second.pt", image_id="img_a.png")

    with pytest.raises(ValueError, match="duplicate query keypoint"):
        merge_listwise_candidate_artifacts(
            input_artifacts=[first, second],
            output_artifact=tmp_path / "merged.pt",
            scene="GreatCourt",
            split_name="train_dev",
        )


def test_merge_internal_candidate_artifacts_cli_writes_manifest_and_summary(tmp_path: Path):
    first = _write_pair_cache(tmp_path / "first.pt", image_id="img_a.png")
    second = _write_pair_cache(tmp_path / "second.pt", image_id="img_b.png")
    out = tmp_path / "merge"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--input_artifacts",
            str(first),
            str(second),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    assert (out / "merged_candidates.pt").is_file()
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 4
    assert summary["query_count"] == 2
    assert manifest["schema_version"] == "internal_candidate_artifact_merge_manifest_v1"
    assert manifest["inference_stage"] == "candidate_artifact_merge"
    assert manifest["dense_inference_enabled"] is False
    assert split_audit["audit_status"] == "passed"
    assert "loc_gs.scripts.merge_internal_candidate_artifacts" in (out / "command.txt").read_text(encoding="utf-8")
