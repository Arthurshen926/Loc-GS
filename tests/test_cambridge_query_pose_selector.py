import json
from pathlib import Path

from loc_gs.scripts.select_cambridge_query_pose import (
    align_query_rows,
    evaluate_query_selector,
    select_query_rows,
)


def _write_eval(path: Path, values: list[tuple[float, float, int]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image_name": f"frame{i:04d}.png",
            "dense_te": te,
            "dense_ae": ae,
            "dense_inliers": inliers,
        }
        for i, (te, ae, inliers) in enumerate(values)
    ]
    (path / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    (path / "summary.json").write_text("{}", encoding="utf-8")
    return path


def test_align_query_rows_falls_back_to_index_when_names_are_missing():
    aligned = align_query_rows(
        {
            "a": [{"image_name": "idx:000000"}, {"image_name": "idx:000001"}],
            "b": [{"image_name": "real0.png"}, {"image_name": "real1.png"}],
        }
    )

    assert len(aligned) == 2
    assert aligned[0]["a"]["image_name"] == "idx:000000"
    assert aligned[0]["b"]["image_name"] == "real0.png"


def test_query_selector_uses_calibration_delta_to_guard_baseline():
    selected = select_query_rows(
        {
            "stdloc_baseline": [
                {"image_name": "a", "te": 4.0, "ae": 1.0, "inliers": 100},
                {"image_name": "b", "te": 4.0, "ae": 1.0, "inliers": 100},
            ],
            "learned": [
                {"image_name": "a", "te": 2.0, "ae": 1.0, "inliers": 102},
                {"image_name": "b", "te": 30.0, "ae": 1.0, "inliers": 103},
            ],
        },
        baseline_branch="stdloc_baseline",
        branch_deltas={"stdloc_baseline": 0.0, "learned": -0.2},
        mode="calibrated_confidence",
        branch_prior_weight=5.0,
        min_inliers=4,
        align_by="name",
    )

    assert {row["selected_branch"] for row in selected} == {"stdloc_baseline"}


def test_evaluate_query_selector_reports_oracle_upper_bound(tmp_path):
    baseline = _write_eval(tmp_path / "baseline", [(10.0, 1.0, 100), (2.0, 1.0, 100)])
    learned = _write_eval(tmp_path / "learned", [(1.0, 1.0, 50), (20.0, 1.0, 200)])
    manifest = {
        "Scene": {
            "branches": {
                "stdloc_baseline": {"eval_dir": str(baseline)},
                "learned": {"eval_dir": str(learned)},
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = evaluate_query_selector(manifest_path, mode="oracle")

    assert summary["macro"]["mean_median_te"] == 1.5
    assert summary["scenes"][0]["branch_counts"] == {"learned": 1, "stdloc_baseline": 1}
