import json

from loc_gs.scripts.audit_cambridge_parity import audit_cambridge_parity


def test_audit_cambridge_parity_writes_query_csv_and_summary(tmp_path):
    native_dir = tmp_path / "native"
    stdloc_dir = tmp_path / "stdloc"
    out_dir = tmp_path / "audit"
    native_dir.mkdir()
    stdloc_dir.mkdir()
    (native_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "image_name": "q1.png",
                    "sparse_te": 12.0,
                    "sparse_ae": 1.2,
                    "sparse_inliers": 20,
                    "dense_te": 10.0,
                    "dense_ae": 0.8,
                    "dense_inliers": 28,
                },
                {
                    "image_name": "q2.png",
                    "sparse_te": 16.0,
                    "sparse_ae": 1.6,
                    "sparse_inliers": 12,
                    "dense_te": 14.0,
                    "dense_ae": 1.1,
                    "dense_inliers": 18,
                },
            ]
        ),
        encoding="utf-8",
    )
    (stdloc_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "image_name": "q1.png",
                    "sparse_te": 9.0,
                    "sparse_ae": 0.9,
                    "sparse_inliers": 30,
                    "dense_te": 8.0,
                    "dense_ae": 0.7,
                    "dense_inliers": 40,
                },
                {
                    "image_name": "q2.png",
                    "sparse_te": 11.0,
                    "sparse_ae": 1.1,
                    "sparse_inliers": 24,
                    "dense_te": 10.0,
                    "dense_ae": 0.9,
                    "dense_inliers": 32,
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = audit_cambridge_parity(
        native_dir=native_dir,
        stdloc_dir=stdloc_dir,
        output_dir=out_dir,
        scene="KingsCollege",
    )

    csv_text = (out_dir / "parity_audit.csv").read_text(encoding="utf-8")
    assert "query_id,native_sparse_te_cm,stdloc_sparse_te_cm,sparse_delta_te_cm" in csv_text
    assert "q1.png,12.0,9.0,3.0" in csv_text
    assert summary["scene"] == "KingsCollege"
    assert summary["common_queries"] == 2
    assert summary["native"]["dense"]["median_te_cm"] == 12.0
    assert summary["stdloc"]["dense"]["median_te_cm"] == 9.0
    assert summary["delta"]["dense_median_te_cm"] == 3.0
    assert (out_dir / "summary.json").exists()


def test_audit_cambridge_parity_uses_row_ids_when_results_have_no_query_name(tmp_path):
    native_dir = tmp_path / "native"
    stdloc_dir = tmp_path / "stdloc"
    out_dir = tmp_path / "audit"
    native_dir.mkdir()
    stdloc_dir.mkdir()
    (native_dir / "results.json").write_text(
        json.dumps([{"dense_TE": 3.0, "dense_AE": 0.3, "dense": [{"inliers": 10}]}]),
        encoding="utf-8",
    )
    (stdloc_dir / "results.json").write_text(
        json.dumps([{"dense_TE": 2.0, "dense_AE": 0.2, "dense": [{"inliers": 12}]}]),
        encoding="utf-8",
    )

    summary = audit_cambridge_parity(
        native_dir=native_dir,
        stdloc_dir=stdloc_dir,
        output_dir=out_dir,
        scene="ShopFacade",
    )

    assert summary["common_queries"] == 1
    assert summary["native"]["dense"]["median_te_cm"] == 3.0
    assert summary["stdloc"]["dense"]["median_re_deg"] == 0.2
    assert "query_000000" in (out_dir / "parity_audit.csv").read_text(encoding="utf-8")
