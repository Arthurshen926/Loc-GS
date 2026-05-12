import json
from pathlib import Path

from loc_gs.scripts.analyze_cambridge_recall_headroom import collect_branch_metrics, oracle_by_metric
from loc_gs.scripts.eval_cambridge_branch_selected import evaluate_selected_branches
from loc_gs.scripts.select_cambridge_branch import (
    load_result_rows,
    select_branches,
    select_scene_branch,
)


def _write_eval_dir(path: Path, values: list[tuple[float, float]], *, upper: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (te, ae) in enumerate(values):
        if upper:
            rows.append({"dense_TE": te, "dense_AE": ae, "dense": [{"inliers": 100 + idx}]})
        else:
            rows.append(
                {
                    "image_name": f"seq/frame{idx:05d}.png",
                    "dense_te": te,
                    "dense_ae": ae,
                    "dense_inliers": 100 + idx,
                }
            )
    (path / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps({"dense": {}}), encoding="utf-8")
    return path


def test_select_scene_branch_penalizes_tiny_recall_gain_with_large_te_regression():
    branch, metrics = select_scene_branch(
        {
            "stdloc_baseline": {
                "median_te": 17.69,
                "median_ae": 0.5,
                "recall_5cm_5d": 0.017,
                "recall_2cm_2d": 0.0,
                "localized": 343,
                "queries": 343,
                "avg_inliers": 100.0,
            },
            "historical_selected": {
                "median_te": 19.88,
                "median_ae": 0.5,
                "recall_5cm_5d": 0.0204,
                "recall_2cm_2d": 0.0,
                "localized": 343,
                "queries": 343,
                "avg_inliers": 100.0,
            },
        },
        baseline_branch="stdloc_baseline",
        metric="combined",
        te_penalty_per_cm=0.002,
    )

    assert branch == "stdloc_baseline"
    assert metrics["median_te"] == 17.69


def test_select_scene_branch_filters_low_recall_candidate_before_median_te():
    branch, metrics = select_scene_branch(
        {
            "stdloc_baseline": {
                "median_te": 20.33,
                "median_ae": 0.2,
                "recall_5cm_5d": 0.041,
                "recall_2cm_2d": 0.0,
                "localized": 120,
                "queries": 120,
                "avg_inliers": 100.0,
            },
            "historical_selected": {
                "median_te": 15.81,
                "median_ae": 0.2,
                "recall_5cm_5d": 0.067,
                "recall_2cm_2d": 0.0,
                "localized": 120,
                "queries": 120,
                "avg_inliers": 100.0,
            },
        },
        baseline_branch="stdloc_baseline",
        metric="median_te",
        candidate_min_r5=0.2,
    )

    assert branch == "stdloc_baseline"
    assert metrics["median_te"] == 20.33


def test_select_scene_branch_recall_metric_respects_median_te_guard():
    branch, metrics = select_scene_branch(
        {
            "stdloc_baseline": {
                "median_te": 10.0,
                "median_ae": 0.2,
                "recall_10cm_5d": 0.60,
                "recall_5cm_5d": 0.40,
                "recall_2cm_2d": 0.10,
                "localized": 120,
                "queries": 120,
                "avg_inliers": 100.0,
            },
            "historical_selected": {
                "median_te": 18.0,
                "median_ae": 0.2,
                "recall_10cm_5d": 0.75,
                "recall_5cm_5d": 0.55,
                "recall_2cm_2d": 0.10,
                "localized": 120,
                "queries": 120,
                "avg_inliers": 100.0,
            },
        },
        baseline_branch="stdloc_baseline",
        metric="r10",
        max_median_te_increase_cm=2.0,
    )

    assert branch == "stdloc_baseline"
    assert metrics["median_te"] == 10.0


def test_select_scene_branch_can_optimize_broader_recall():
    branch, metrics = select_scene_branch(
        {
            "stdloc_baseline": {
                "median_te": 10.0,
                "median_ae": 0.2,
                "recall_10cm_5d": 0.60,
                "recall_5cm_5d": 0.40,
                "recall_2cm_2d": 0.10,
                "localized": 120,
                "queries": 120,
                "avg_inliers": 100.0,
            },
            "historical_selected": {
                "median_te": 11.0,
                "median_ae": 0.2,
                "recall_10cm_5d": 0.75,
                "recall_5cm_5d": 0.55,
                "recall_2cm_2d": 0.10,
                "localized": 120,
                "queries": 120,
                "avg_inliers": 100.0,
            },
        },
        baseline_branch="stdloc_baseline",
        metric="r10",
        max_median_te_increase_cm=2.0,
    )

    assert branch == "historical_selected"
    assert metrics["recall_10cm_5d"] == 0.75


def test_select_scene_branch_prefers_learned_on_calibration_tie():
    branch, metrics = select_scene_branch(
        {
            "stdloc_baseline": {
                "median_te": 1.80,
                "median_ae": 0.09,
                "recall_5cm_5d": 0.828,
                "recall_2cm_2d": 0.58,
                "localized": 29,
                "queries": 29,
                "avg_inliers": 100.0,
            },
            "historical_selected": {
                "median_te": 2.20,
                "median_ae": 0.10,
                "recall_5cm_5d": 0.793,
                "recall_2cm_2d": 0.44,
                "localized": 29,
                "queries": 29,
                "avg_inliers": 100.0,
            },
        },
        baseline_branch="stdloc_baseline",
        metric="median_te",
        candidate_min_r5=0.2,
        tie_prefer_branch="historical_selected",
        tie_max_te_increase_cm=0.5,
        tie_max_r5_drop=0.05,
        tie_min_r5=0.2,
    )

    assert branch == "historical_selected"
    assert metrics["median_te"] == 2.20


def test_select_branches_uses_calibration_stride_and_preserves_baseline(tmp_path):
    baseline_a = _write_eval_dir(tmp_path / "A_baseline", [(4.0, 1.0), (20.0, 1.0), (4.5, 1.0)])
    selected_a = _write_eval_dir(tmp_path / "A_selected", [(3.0, 1.0), (30.0, 1.0), (3.5, 1.0)])
    baseline_b = _write_eval_dir(tmp_path / "B_baseline", [(4.0, 1.0), (20.0, 1.0), (4.5, 1.0)])
    selected_b = _write_eval_dir(tmp_path / "B_selected", [(12.0, 1.0), (1.0, 1.0), (12.5, 1.0)])
    manifest = {
        "SceneA": {
            "stdloc_baseline": {"eval_dir": str(baseline_a)},
            "historical_selected": {"eval_dir": str(selected_a)},
        },
        "SceneB": {
            "stdloc_baseline": {"eval_dir": str(baseline_b)},
            "historical_selected": {"eval_dir": str(selected_b)},
        },
    }

    selection = select_branches(
        manifest,
        stage="dense",
        calibration_ids=None,
        calibration_stride=2,
        calibration_offset=0,
        baseline_branch="stdloc_baseline",
        metric="combined",
        te_penalty_per_cm=0.002,
        allow_r5_drop=0.0,
        r5_tie=0.01,
    )

    assert selection["selected_branch"] == {
        "SceneA": "historical_selected",
        "SceneB": "stdloc_baseline",
    }
    assert selection["scenes"]["SceneA"]["calibration_rows"]["stdloc_baseline"] == 2


def test_eval_selected_branches_aggregates_test_dirs(tmp_path):
    baseline = _write_eval_dir(tmp_path / "baseline", [(4.0, 1.0), (10.0, 1.0)])
    selected = _write_eval_dir(tmp_path / "selected", [(2.0, 1.0), (3.0, 1.0)], upper=True)
    manifest = {
        "scenes": {
            "ShopFacade": {
                "branches": {
                    "stdloc_baseline": {"test_dir": str(baseline)},
                    "historical_selected": {"test_dir": str(selected)},
                }
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selected_path = tmp_path / "selected_branch.json"
    selected_path.write_text(json.dumps({"selected_branch": {"ShopFacade": "historical_selected"}}), encoding="utf-8")

    summary = evaluate_selected_branches(selected_path, manifest_path)

    assert summary["rows"][0]["selected_branch"] == "historical_selected"
    assert summary["macro"]["macro_recall_5cm_5d"] == 1.0
    rows = load_result_rows(selected, stage="dense")
    assert rows[0]["image_name"] == "idx:000000"
    assert rows[0]["te"] == 2.0


def test_recall_headroom_oracle_reports_metric_specific_selection(tmp_path):
    baseline = _write_eval_dir(tmp_path / "baseline", [(4.0, 1.0), (12.0, 1.0)])
    selected = _write_eval_dir(tmp_path / "selected", [(2.0, 1.0), (3.0, 1.0)])
    manifest = {
        "ShopFacade": {
            "stdloc_baseline": {"eval_dir": str(baseline)},
            "historical_selected": {"eval_dir": str(selected)},
        }
    }

    branch_metrics = collect_branch_metrics(manifest)
    oracle = oracle_by_metric(branch_metrics, split="test", target_metric="r5")

    assert oracle["selected_branch"] == {"ShopFacade": "historical_selected"}
    assert oracle["macro"]["macro_recall_5cm_5d"] == 1.0
