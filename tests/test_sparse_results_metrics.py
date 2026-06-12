import json
from pathlib import Path

from loc_gs.sparse.results_metrics import build_sparse_metrics_from_results
from loc_gs.scripts.build_internal_sparse_metrics_from_results import main


def _write_results(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"image_name": "a.png", "sparse_te_cm": 4.0, "sparse_re_deg": 0.2, "sparse_inliers": 12},
                    {"image_name": "b.png", "sparse_te_cm": 12.0, "sparse_re_deg": 0.3, "sparse_inliers": 10},
                    {"image_name": "c.png", "sparse_te_cm": 6.0, "sparse_re_deg": 6.0, "sparse_inliers": 8},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_sparse_metrics_from_results_filters_to_query_ids(tmp_path: Path):
    summary, rows = build_sparse_metrics_from_results(
        results_path=_write_results(tmp_path / "results.json"),
        scene="GreatCourt",
        split_name="train_dev_subset",
        query_ids=("a.png", "c.png"),
    )

    assert [row["query_id"] for row in rows] == ["a.png", "c.png"]
    assert summary["schema_version"] == "internal_sparse_results_metrics_v1"
    assert summary["query_count"] == 2
    assert summary["median_te_cm"] == 5.0
    assert summary["median_re_deg"] == 3.1
    assert summary["recall_10cm_5d"] == 0.5
    assert summary["recall_5cm_5d"] == 0.5
    assert summary["query_filter_enabled"] is True
    assert summary["requested_query_count"] == 2
    assert summary["missing_query_count"] == 0


def test_build_internal_sparse_metrics_from_results_cli_writes_bundle(tmp_path: Path):
    results = _write_results(tmp_path / "results.json")
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("a.png\nc.png\n", encoding="utf-8")
    out = tmp_path / "metrics"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev_subset",
            "--results",
            str(results),
            "--query_ids",
            str(query_ids),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "filtered_results.json").read_text(encoding="utf-8"))
    assert metrics["query_count"] == 2
    assert metrics["recall_10cm_5d"] == 0.5
    assert manifest["schema_version"] == "internal_sparse_results_metrics_manifest_v1"
    assert manifest["inference_stage"] == "metrics_recompute_only"
    assert [row["query_id"] for row in rows] == ["a.png", "c.png"]
    assert (out / "command.txt").is_file()
