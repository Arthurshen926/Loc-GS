import json
from pathlib import Path

from loc_gs.scripts.build_cambridge_eval_report import RunRoots, build_report, summarize_run


def _write_run(root: Path, scene: str, rows: list[dict]) -> None:
    run_dir = root / scene
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "sampled_count": 16384,
                "landmark_count": 16384,
                "map_size_mb": 1.25,
            }
        ),
        encoding="utf-8",
    )


def _row(sparse_te: float, sparse_re: float, dense_te: float, dense_re: float) -> dict:
    return {
        "sparse": {"inliers": 10},
        "dense": [{"inliers": 100}],
        "sparse_TE": sparse_te,
        "sparse_AE": sparse_re,
        "dense_TE": dense_te,
        "dense_AE": dense_re,
    }


def test_summarize_run_computes_full_recall_thresholds(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "ShopFacade",
        [
            _row(4.0, 1.0, 4.0, 1.0),
            _row(12.0, 1.0, 12.0, 1.0),
            _row(20.0, 1.0, 20.0, 1.0),
        ],
    )

    summary = summarize_run(tmp_path / "ShopFacade")

    assert summary["dense"]["median_te_cm"] == 12.0
    assert summary["dense"]["R50cm5deg"] == 1.0
    assert summary["dense"]["R15cm5deg"] == 2 / 3
    assert summary["dense"]["R10cm5deg"] == 1 / 3
    assert summary["dense"]["R5cm5deg"] == 1 / 3
    assert summary["dense"]["R2cm2deg"] == 0.0
    assert summary["dense"]["p90_te_cm"] == 18.4
    assert summary["dense"]["p95_te_cm"] == 19.2
    assert summary["dense"]["cvar10_te_cm"] == 20.0
    assert summary["dense"]["severe_rate_1m"] == 0.0
    assert summary["dense"]["catastrophic_rate_5m"] == 0.0


def test_build_report_writes_metrics_deltas_and_qualitative_cases(tmp_path: Path) -> None:
    scene = "ShopFacade"
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    data_root = tmp_path / "data"
    output_dir = tmp_path / "report"
    scene_root = data_root / scene
    scene_root.mkdir(parents=True)
    (scene_root / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "seq1/b.png 0 0 0 1 0 0 0\n"
        "seq1/a.png 0 0 0 1 0 0 0\n",
        encoding="utf-8",
    )
    _write_run(
        baseline_root,
        scene,
        [
            _row(8.0, 1.0, 4.0, 1.0),
            _row(20.0, 1.0, 20.0, 1.0),
        ],
    )
    _write_run(
        candidate_root,
        scene,
        [
            _row(3.0, 1.0, 12.0, 1.0),
            _row(18.0, 1.0, 8.0, 1.0),
        ],
    )

    report = build_report(
        RunRoots(
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            data_root=data_root,
            output_dir=output_dir,
            baseline_label="baseline",
            candidate_label="candidate",
        ),
        scenes=(scene,),
        top_k=2,
    )

    assert (output_dir / "cambridge_metrics.csv").exists()
    assert (output_dir / "cambridge_deltas.csv").exists()
    assert (output_dir / "qualitative_cases.csv").exists()
    assert (output_dir / "report.md").exists()
    assert report["macro"]["delta"]["dense"]["median_te_cm"] == -2.0
    paired = report["paired_rows"][0]
    assert paired["dense_regression_20cm_count"] == 0
    assert paired["dense_improvement_20cm_count"] == 0
    assert paired["sparse_correct_dense_wrong_count"] == 1
    assert paired["candidate_dense_worsened_count"] == 1
    assert (output_dir / "hard_cases.csv").exists()
    assert len(report["hard_cases"]) == 2
    categories = {category for case in report["qualitative_cases"] for category in case["categories"]}
    assert "candidate_worse_dense" in categories
    assert "candidate_better_dense" in categories
    assert "sparse_help_dense_hurt" in categories


def test_build_report_marks_cambridge_pose_qnorm_outliers(tmp_path: Path) -> None:
    scene = "StMarysChurch"
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    data_root = tmp_path / "data"
    output_dir = tmp_path / "report"
    scene_root = data_root / scene
    scene_root.mkdir(parents=True)
    (scene_root / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "seq13/frame00173.png 0 0 0 1 0 0 0\n"
        "seq13/frame00174.png 0 0 0 0.003359 0.017963 -1.018482 0.234648\n",
        encoding="utf-8",
    )
    _write_run(baseline_root, scene, [_row(1.0, 1.0, 1.0, 1.0), _row(900.0, 1.0, 900.0, 1.0)])
    _write_run(candidate_root, scene, [_row(1.0, 1.0, 1.0, 1.0), _row(900.0, 1.0, 900.0, 1.0)])

    report = build_report(
        RunRoots(
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            data_root=data_root,
            output_dir=output_dir,
            baseline_label="baseline",
            candidate_label="candidate",
        ),
        scenes=(scene,),
        top_k=2,
    )

    outlier = [case for case in report["qualitative_cases"] if case["image_name"] == "seq13/frame00174.png"][0]
    assert "pose_qnorm_outlier" in outlier["categories"]
    assert outlier["pose_qnorm"] == 1.045323
