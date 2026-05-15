import json
from pathlib import Path

from loc_gs.stdloc_native.results import (
    load_stdloc_query_results,
    load_stdloc_summary,
)


def test_load_stdloc_summary_normalizes_stage_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": "map_cambridge/ShopFacade",
                "dense": {
                    "median_te": 2.5,
                    "median_ae": 0.12,
                    "recall_5cm_5d": 0.8,
                    "recall_2cm_2d": 0.3,
                    "avg_inliers": 42.0,
                },
                "sparse": {
                    "median_te": 4.0,
                    "median_ae": 0.2,
                    "recall_5cm_5d": 0.5,
                    "recall_2cm_2d": 0.2,
                    "avg_inliers": 20.0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = load_stdloc_summary(run_dir)

    assert summary["dense"]["median_te_cm"] == 2.5
    assert summary["dense"]["median_re_deg"] == 0.12
    assert summary["dense"]["recall_5cm_5deg"] == 0.8
    assert summary["sparse"]["avg_inliers"] == 20.0


def test_load_stdloc_query_results_accepts_official_results_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "image_name": "seq/frame0001.png",
                    "sparse_AE": 0.3,
                    "sparse_TE": 5.0,
                    "dense_AE": 0.2,
                    "dense_TE": 3.0,
                    "sparse": {"inliers": 12},
                    "dense": [{"inliers": 25}],
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = load_stdloc_query_results(run_dir)

    assert rows == [
        {
            "query_id": "seq/frame0001.png",
            "sparse_re_deg": 0.3,
            "sparse_te_cm": 5.0,
            "sparse_inliers": 12,
            "dense_re_deg": 0.2,
            "dense_te_cm": 3.0,
            "dense_inliers": 25,
        }
    ]


def test_load_stdloc_query_results_falls_back_to_stable_row_ids(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "sparse_AE": 0.3,
                    "sparse_TE": 5.0,
                    "dense_AE": 0.2,
                    "dense_TE": 3.0,
                    "sparse": {"inliers": 12},
                    "dense": [{"inliers": 25}],
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = load_stdloc_query_results(run_dir)

    assert rows[0]["query_id"] == "query_000000"
