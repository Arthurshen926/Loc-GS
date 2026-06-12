import json
from pathlib import Path

import pytest
import torch

from loc_gs.sparse.candidate_coverage import (
    audit_candidate_artifact_coverage,
    load_requested_query_ids_from_results,
)
from loc_gs.scripts.audit_internal_candidate_coverage import main


def _write_pair_cache(path: Path, *, split_name: str = "train_dev") -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": split_name,
            "topk": 2,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "query_yx": torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[101, 102], [201, 202], [301, 302]], dtype=torch.int64),
        "cosine": torch.tensor([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]], dtype=torch.float32),
        "label": torch.tensor([0, 1, 0], dtype=torch.int64),
        "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
        "query_id": ["img_a.png::kp_0001", "img_a.png::kp_0002", "img_b.png::kp_0001"],
        "image_id": ["img_a.png", "img_a.png", "img_b.png"],
        "keypoint_id": ["kp_0001", "kp_0002", "kp_0001"],
        "source_phase": [split_name, split_name, split_name],
    }
    torch.save(payload, path)
    return path


def _write_results(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"query_id": "img_a.png", "te_cm": 5.0, "re_deg": 1.0},
                    {"query_id": "img_b.png", "te_cm": 8.0, "re_deg": 2.0},
                    {"query_id": "img_c.png", "te_cm": 15.0, "re_deg": 3.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_candidate_coverage_audit_reports_missing_requested_queries(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    requested = ["img_a.png", "img_b.png", "img_c.png"]

    summary, details = audit_candidate_artifact_coverage(
        candidate_artifact=pair_cache,
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        requested_query_ids=requested,
    )

    assert summary["schema_version"] == "internal_candidate_coverage_v1"
    assert summary["scene"] == "GreatCourt"
    assert summary["split_name"] == "train_dev_seed13_20p"
    assert summary["artifact_split_name"] == "train_dev"
    assert summary["requested_query_count"] == 3
    assert summary["artifact_query_count"] == 2
    assert summary["covered_query_count"] == 2
    assert summary["missing_query_count"] == 1
    assert summary["coverage_ratio"] == pytest.approx(2.0 / 3.0)
    assert summary["complete_coverage"] is False
    assert details.covered_query_ids == ["img_a.png", "img_b.png"]
    assert details.missing_query_ids == ["img_c.png"]


def test_candidate_coverage_cli_writes_auditable_bundle_from_results(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    results = _write_results(tmp_path / "results.json")
    output_dir = tmp_path / "coverage"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev_seed13_20p",
            "--candidate_artifact",
            str(pair_cache),
            "--query_results",
            str(results),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert summary["covered_query_count"] == 2
    assert summary["missing_query_count"] == 1
    assert (output_dir / "covered_query_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "img_a.png",
        "img_b.png",
    ]
    assert (output_dir / "missing_query_ids.txt").read_text(encoding="utf-8").splitlines() == ["img_c.png"]
    assert manifest["schema_version"] == "internal_candidate_coverage_manifest_v1"
    assert manifest["query_results"] == str(results)
    assert manifest["dense_inference_enabled"] is False
    assert split_audit["audit_status"] == "passed"
    assert (output_dir / "command.txt").is_file()


def test_candidate_coverage_rejects_test_split_request(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")

    with pytest.raises(ValueError, match="test split"):
        audit_candidate_artifact_coverage(
            candidate_artifact=pair_cache,
            scene="GreatCourt",
            split_name="test",
            requested_query_ids=["img_a.png"],
        )


def test_candidate_coverage_extracts_query_ids_from_result_rows(tmp_path: Path):
    results = _write_results(tmp_path / "results.json")

    assert load_requested_query_ids_from_results(results) == ["img_a.png", "img_b.png", "img_c.png"]
