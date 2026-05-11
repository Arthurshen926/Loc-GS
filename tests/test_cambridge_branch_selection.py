import json
from pathlib import Path

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
