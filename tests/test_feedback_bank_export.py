import json

import pytest
import torch

from loc_gs.feedback.io import load_feedback_bank
from loc_gs.scripts.export_feedback_bank_from_cambridge import build_argparser, main


def _write_listwise_pair_cache(path):
    payload = {
        "cosine": torch.tensor([[0.9, 0.4, 0.2], [0.8, 0.7, 0.1]], dtype=torch.float32),
        "query_score": torch.tensor([0.75, 0.25], dtype=torch.float32),
        "landmark_prior": torch.tensor([[0.8, 0.3, 0.1], [0.6, 0.5, 0.2]], dtype=torch.float32),
        "candidate_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "reprojection_error": torch.tensor([[1.0, 12.0, float("inf")], [9.0, 10.0, 11.0]], dtype=torch.float32),
        "query_yx": torch.tensor([[20.0, 10.0], [24.0, 12.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[101, 102, 103], [201, 202, 203]], dtype=torch.long),
        "base_gaussian_id": torch.arange(3000, dtype=torch.long) + 1000,
        "label": torch.tensor([0, 3], dtype=torch.long),
        "metadata": {
            "scene": "ShopFacade",
            "format": "listwise",
            "topk": 3,
            "source": "calibrate_landmark_matchability",
        },
    }
    torch.save(payload, path)


def test_export_feedback_bank_from_listwise_pair_cache(tmp_path):
    pair_cache = tmp_path / "scene_match_pairs.pt"
    _write_listwise_pair_cache(pair_cache)
    selfmap_summary = tmp_path / "selfmap_summary.json"
    selfmap_summary.write_text(
        json.dumps({"dense": {"median_te": 2.5, "median_ae": 0.11, "recall_5cm_5d": 0.8}}),
        encoding="utf-8",
    )
    baseline_summary = tmp_path / "baseline_summary.json"
    baseline_summary.write_text(
        json.dumps({"dense": {"median_te": 3.0, "median_ae": 0.12, "recall_5cm_5d": 0.7}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "export"

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--pair_cache",
            str(pair_cache),
            "--selfmap_summary",
            str(selfmap_summary),
            "--baseline_summary",
            str(baseline_summary),
            "--output_path",
            str(output_dir),
            "--split_name",
            "selfmap_train",
        ]
    )

    assert main(args) == 0

    bank_path = output_dir / "feedback_bank.jsonl"
    bank = load_feedback_bank(bank_path)
    records = bank["records"]
    summary = json.loads((output_dir / "feedback_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(records) == 5
    assert records[0]["scene"] == "ShopFacade"
    assert records[0]["query_id"] == "query_000000"
    assert records[0]["source_view_id"] == "scene_match_pairs.pt"
    assert records[0]["keypoint_xy"] == [10.0, 20.0]
    assert records[0]["matched_landmark_id"] == "101"
    assert records[0]["matched_gaussian_id"] == "1101"
    assert records[0]["pnp_inlier"] is True
    assert records[1]["pnp_inlier"] is False
    assert records[-1]["query_id"] == "query_000001"
    assert summary["record_count"] == 5
    assert summary["pnp_inlier_rate"] == pytest.approx(1 / 5)
    assert manifest["split_name"] == "selfmap_train"
    assert manifest["pair_cache_metadata"]["format"] == "listwise"
    assert manifest["baseline_relative"]["median_te_delta_cm"] == pytest.approx(-0.5)


def test_export_feedback_bank_treats_negative_listwise_label_as_no_pnp_success(tmp_path):
    pair_cache = tmp_path / "scene_match_pairs.pt"
    _write_listwise_pair_cache(pair_cache)
    payload = torch.load(pair_cache, map_location="cpu")
    payload["label"] = torch.tensor([-1, 0], dtype=torch.long)
    torch.save(payload, pair_cache)
    selfmap_summary = tmp_path / "selfmap_summary.json"
    selfmap_summary.write_text(json.dumps({"dense": {"median_te": 2.5}}), encoding="utf-8")
    output_dir = tmp_path / "export"
    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--pair_cache",
            str(pair_cache),
            "--selfmap_summary",
            str(selfmap_summary),
            "--output_path",
            str(output_dir),
            "--split_name",
            "selfmap_train",
        ]
    )

    assert main(args) == 0

    records = load_feedback_bank(output_dir / "feedback_bank.jsonl")["records"]
    first_query_records = [record for record in records if record["query_id"] == "query_000000"]
    assert first_query_records
    assert all(record["pnp_success"] is False for record in first_query_records)
    assert all(record["pnp_inlier"] is False for record in first_query_records)


def test_export_feedback_bank_dry_run_reports_schema_without_files(tmp_path, capsys):
    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--pair_cache",
            str(tmp_path / "missing_pairs.pt"),
            "--selfmap_summary",
            str(tmp_path / "missing_summary.json"),
            "--output_path",
            str(tmp_path / "export"),
            "--split_name",
            "selfmap_train",
            "--dry_run",
        ]
    )

    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    assert payload["would_write"]["feedback_bank"].endswith("feedback_bank.jsonl")
    assert payload["schema"]["record_fields"]
    assert not (tmp_path / "export").exists()


def test_export_feedback_bank_rejects_test_split(tmp_path):
    pair_cache = tmp_path / "scene_match_pairs.pt"
    _write_listwise_pair_cache(pair_cache)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"dense": {"median_te": 2.5}}), encoding="utf-8")
    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--pair_cache",
            str(pair_cache),
            "--selfmap_summary",
            str(summary),
            "--output_path",
            str(tmp_path / "export"),
            "--split_name",
            "test",
        ]
    )

    with pytest.raises(ValueError, match="test split"):
        main(args)
