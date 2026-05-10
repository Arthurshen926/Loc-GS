import json

from loc_gs.scripts.merge_cambridge_eval_results import merge_eval_dirs


def _write_eval_dir(path, image_name, dense_te, dense_ae, sparse_te=None, sparse_ae=None):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "scene": "Scene",
                "query_offset": 0,
                "query_stride": 2,
                "dense": {},
                "sparse": {},
                "localized": 1,
                "queries": 1,
            }
        )
    )
    (path / "results.json").write_text(
        json.dumps(
            [
                {
                    "image_name": image_name,
                    "dense_te": dense_te,
                    "dense_ae": dense_ae,
                    "dense_inliers": 10,
                    "sparse_te": sparse_te,
                    "sparse_ae": sparse_ae,
                    "sparse_inliers": 5,
                    "localized": dense_te is not None,
                }
            ]
        )
    )


def test_merge_eval_dirs_recomputes_metrics(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    out = tmp_path / "merged"
    _write_eval_dir(a, "b.png", 4.0, 1.0, 6.0, 1.0)
    _write_eval_dir(b, "a.png", 8.0, 1.0, None, None)

    summary = merge_eval_dirs([a, b], out)

    assert summary["queries"] == 2
    assert summary["localized"] == 2
    assert summary["dense"]["median_te"] == 6.0
    assert summary["dense"]["recall_5cm_5d"] == 0.5
    assert summary["sparse"]["median_te"] == 6.0
    assert summary["query_stride"] == 1
    details = json.loads((out / "results.json").read_text())
    assert [row["image_name"] for row in details] == ["a.png", "b.png"]
