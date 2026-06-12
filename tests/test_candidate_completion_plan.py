import json
from pathlib import Path

import pytest

from loc_gs.sparse.candidate_completion_plan import build_candidate_completion_plan
from loc_gs.scripts.build_internal_candidate_completion_plan import main


def test_candidate_completion_plan_shards_missing_queries_for_internal_builder():
    summary, shards = build_candidate_completion_plan(
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        missing_query_ids=["img_c.png", "img_d.png", "img_e.png"],
        shard_size=2,
        base_candidate_artifact="existing.pt",
    )

    assert summary["schema_version"] == "internal_candidate_completion_plan_v1"
    assert summary["completion_required"] is True
    assert summary["missing_query_count"] == 3
    assert summary["shard_count"] == 2
    assert summary["shard_size"] == 2
    assert shards[0]["shard_id"] == "candidate_completion_shard_000000"
    assert shards[0]["query_ids"] == ["img_c.png", "img_d.png"]
    assert shards[0]["candidate_backend"] == "internal_sparse_pipeline"
    assert shards[0]["dense_inference_enabled"] is False
    assert shards[1]["query_ids"] == ["img_e.png"]


def test_candidate_completion_plan_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_candidate_completion_plan(
            scene="GreatCourt",
            split_name="test",
            missing_query_ids=["img_c.png"],
            shard_size=2,
            base_candidate_artifact=None,
        )


def test_candidate_completion_plan_cli_reads_coverage_dir_and_writes_auditable_bundle(tmp_path: Path):
    coverage_dir = tmp_path / "coverage"
    coverage_dir.mkdir()
    (coverage_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_candidate_coverage_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev_seed13_20p",
                "requested_query_count": 5,
                "covered_query_count": 2,
                "missing_query_count": 3,
                "complete_coverage": False,
            }
        ),
        encoding="utf-8",
    )
    (coverage_dir / "missing_query_ids.txt").write_text("img_c.png\nimg_d.png\nimg_e.png\n", encoding="utf-8")
    out = tmp_path / "completion"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev_seed13_20p",
            "--coverage_dir",
            str(coverage_dir),
            "--base_candidate_artifact",
            "existing.pt",
            "--shard_size",
            "2",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lines = [json.loads(line) for line in (out / "candidate_completion_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["missing_query_count"] == 3
    assert summary["shard_count"] == 2
    assert summary["coverage_requested_query_count"] == 5
    assert summary["coverage_covered_query_count"] == 2
    assert manifest["schema_version"] == "internal_candidate_completion_plan_manifest_v1"
    assert manifest["base_candidate_artifact"] == "existing.pt"
    assert manifest["dense_inference_enabled"] is False
    assert lines[0]["query_ids"] == ["img_c.png", "img_d.png"]
    assert lines[1]["query_ids"] == ["img_e.png"]
    assert (out / "missing_query_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "img_c.png",
        "img_d.png",
        "img_e.png",
    ]
    assert "loc_gs.scripts.build_internal_candidate_completion_plan" in (out / "command.txt").read_text(
        encoding="utf-8"
    )
