import json

import pytest

from loc_gs.scripts.merge_cambridge_eval_shards import merge_eval_shards
from loc_gs.scripts.launch_cambridge_reliability_eval import build_argparser


def _write_shard(path, *, offset, rows):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "scene": "ShopFacade",
                "query_offset": offset,
                "query_stride": 2,
                "queries": len(rows),
                "localized": sum(1 for row in rows if row.get("dense_te") is not None),
                "dense": {},
                "sparse": {},
            }
        ),
        encoding="utf-8",
    )
    (path / "results.json").write_text(json.dumps(rows), encoding="utf-8")


def test_merge_eval_shards_recomputes_summary_and_results(tmp_path):
    shard0 = tmp_path / "offset_0_of_2"
    shard1 = tmp_path / "offset_1_of_2"
    _write_shard(
        shard0,
        offset=0,
        rows=[
            {
                "image_name": "b.png",
                "dense_te": 2.0,
                "dense_ae": 1.0,
                "dense_inliers": 5,
                "sparse_te": 3.0,
                "sparse_ae": 1.0,
                "sparse_inliers": 4,
                "matchability": {"precision": 0.5},
            },
            {
                "image_name": "d.png",
                "dense_te": None,
                "dense_ae": None,
                "dense_inliers": 0,
                "sparse_te": None,
                "sparse_ae": None,
                "sparse_inliers": 0,
            },
        ],
    )
    _write_shard(
        shard1,
        offset=1,
        rows=[
            {
                "image_name": "a.png",
                "dense_te": 10.0,
                "dense_ae": 1.0,
                "dense_inliers": 7,
                "sparse_te": 6.0,
                "sparse_ae": 1.0,
                "sparse_inliers": 8,
                "matchability": {"precision": 1.0},
            },
            {
                "image_name": "c.png",
                "dense_te": 4.0,
                "dense_ae": 6.0,
                "dense_inliers": 3,
                "sparse_te": 4.0,
                "sparse_ae": 1.0,
                "sparse_inliers": 2,
            },
        ],
    )

    merged = merge_eval_shards([shard0, shard1], tmp_path / "merged")

    assert merged["queries"] == 4
    assert merged["localized"] == 3
    assert merged["dense"]["median_te"] == 4.0
    assert merged["dense"]["recall_5cm_5d"] == pytest.approx(1.0 / 3.0)
    assert merged["sparse"]["avg_inliers"] == pytest.approx((4 + 8 + 2) / 3)
    assert merged["matchability"]["precision"] == pytest.approx(0.75)
    rows = json.loads((tmp_path / "merged" / "results.json").read_text(encoding="utf-8"))
    assert [row["image_name"] for row in rows] == ["a.png", "b.png", "c.png", "d.png"]


def test_reliability_eval_launcher_exposes_query_shards_argument():
    parser = build_argparser()
    args = parser.parse_args(["--query_shards", "2", "--query_stride", "2"])

    assert args.query_shards == 2
    assert args.query_stride == 2
