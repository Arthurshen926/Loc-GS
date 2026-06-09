import json
from pathlib import Path

from loc_gs.scripts.build_sparse_gate_report import (
    build_feature_stump_gate_cv,
    build_feature_rule_gate_cv,
    build_inlier_bin_gate_cv,
    build_sparse_gate_report,
    main,
)


def _write_run(root: Path, scene: str, run_name: str, rows: list[dict]) -> Path:
    run_dir = root / scene / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"split": "train", "scene": scene, "run_name": run_name}),
        encoding="utf-8",
    )
    (run_dir / "split_audit.json").write_text(
        json.dumps({"audit_status": "train_only", "paper_safe": True}),
        encoding="utf-8",
    )
    return run_dir


def test_oracle_gate_keeps_base_when_candidate_hurts_and_reports_bins(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_run(
        root,
        "ShopFacade",
        "base",
        [
            {"image_name": "a.png", "sparse_te": 4.0, "sparse_ae": 1.0, "sparse_inliers": 700},
            {"image_name": "b.png", "sparse_te": 30.0, "sparse_ae": 1.0, "sparse_inliers": 40},
            {"image_name": "c.png", "sparse_te": 8.0, "sparse_ae": 1.0, "sparse_inliers": 180},
        ],
    )
    _write_run(
        root,
        "ShopFacade",
        "correction",
        [
            {"image_name": "a.png", "sparse_te": 7.0, "sparse_ae": 1.0, "sparse_inliers": 900},
            {"image_name": "b.png", "sparse_te": 12.0, "sparse_ae": 1.0, "sparse_inliers": 160},
            {"image_name": "c.png", "sparse_te": 8.4, "sparse_ae": 1.0, "sparse_inliers": 200},
        ],
    )

    report = build_sparse_gate_report(
        root=root,
        scenes=("ShopFacade",),
        base_name="base",
        candidates={"step4": "correction"},
        stage="sparse",
        improvement_margin_cm=1.0,
    )

    candidate = report["scene_reports"]["ShopFacade"]["candidates"]["step4"]
    assert candidate["paired"]["helped_count"] == 1
    assert candidate["paired"]["harmed_count"] == 1
    assert candidate["paired"]["unchanged_count"] == 1
    assert candidate["oracle"]["median_te_cm"] == 8.0
    assert candidate["candidate"]["median_te_cm"] == 8.4
    assert candidate["confidence_bins"][0]["bin"] == "inliers_low"
    low_bin = {row["bin"]: row for row in candidate["confidence_bins"]}["inliers_low"]
    assert low_bin["helped_count"] == 1
    high_bin = {row["bin"]: row for row in candidate["confidence_bins"]}["inliers_high"]
    assert high_bin["harmed_count"] == 1
    assert report["macro"]["step4"]["scene_count"] == 1
    assert report["macro"]["step4"]["oracle"]["median_te_cm"] == 8.0


def test_cli_writes_report_artifacts_and_skips_missing_candidate_scene(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_run(
        root,
        "ShopFacade",
        "base",
        [{"image_name": "a.png", "sparse_te": 5.0, "sparse_ae": 1.0, "sparse_inliers": 100}],
    )
    _write_run(
        root,
        "ShopFacade",
        "correction",
        [{"image_name": "a.png", "sparse_te": 4.0, "sparse_ae": 1.0, "sparse_inliers": 110}],
    )
    _write_run(
        root,
        "OldHospital",
        "base",
        [{"image_name": "b.png", "sparse_te": 10.0, "sparse_ae": 1.0, "sparse_inliers": 50}],
    )
    output = tmp_path / "report"

    assert (
        main(
            [
                "--root",
                str(root),
                "--scene",
                "ShopFacade",
                "--scene",
                "OldHospital",
                "--base-name",
                "base",
                "--candidate",
                "step4=correction",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert payload["scene_reports"]["OldHospital"]["candidates"]["step4"]["status"] == "missing"
    assert (output / "report.md").exists()
    assert (output / "oracle_cases.csv").exists()
    assert (output / "manifest.json").exists()
    assert (output / "command.txt").exists()


def test_inlier_bin_gate_cv_activates_only_train_positive_bins() -> None:
    base_rows = [
        {"query_id": "low0", "query_index": 0, "sparse_te_cm": 20.0, "sparse_re_deg": 1.0, "sparse_inliers": 40},
        {"query_id": "low1", "query_index": 1, "sparse_te_cm": 24.0, "sparse_re_deg": 1.0, "sparse_inliers": 45},
        {"query_id": "high0", "query_index": 0, "sparse_te_cm": 4.0, "sparse_re_deg": 1.0, "sparse_inliers": 800},
        {"query_id": "high1", "query_index": 1, "sparse_te_cm": 5.0, "sparse_re_deg": 1.0, "sparse_inliers": 820},
    ]
    candidate_rows = [
        {"query_id": "low0", "query_index": 0, "sparse_te_cm": 10.0, "sparse_re_deg": 1.0, "sparse_inliers": 80},
        {"query_id": "low1", "query_index": 1, "sparse_te_cm": 12.0, "sparse_re_deg": 1.0, "sparse_inliers": 90},
        {"query_id": "high0", "query_index": 0, "sparse_te_cm": 8.0, "sparse_re_deg": 1.0, "sparse_inliers": 900},
        {"query_id": "high1", "query_index": 1, "sparse_te_cm": 9.0, "sparse_re_deg": 1.0, "sparse_inliers": 910},
    ]

    report = build_inlier_bin_gate_cv(
        base_rows,
        candidate_rows,
        stage="sparse",
        folds=2,
        min_train_queries=1,
        improvement_margin_cm=1.0,
    )

    assert report["summary"]["median_te_cm"] == 7.5
    assert report["paired"]["activated_count"] == 2
    assert report["paired"]["harmed_count"] == 0
    assert {decision["confidence_bin"] for decision in report["decisions"] if decision["activated"]} == {"inliers_low"}


def test_feature_stump_gate_cv_learns_observable_confidence_threshold() -> None:
    base_rows = [
        {
            "query_id": "low0",
            "query_index": 0,
            "sparse_te_cm": 20.0,
            "sparse_re_deg": 1.0,
            "sparse_inliers": 40,
            "sparse_confidence": {"sparse_pose_inlier_8px": 0.10},
        },
        {
            "query_id": "low1",
            "query_index": 1,
            "sparse_te_cm": 24.0,
            "sparse_re_deg": 1.0,
            "sparse_inliers": 45,
            "sparse_confidence": {"sparse_pose_inlier_8px": 0.15},
        },
        {
            "query_id": "high0",
            "query_index": 0,
            "sparse_te_cm": 4.0,
            "sparse_re_deg": 1.0,
            "sparse_inliers": 800,
            "sparse_confidence": {"sparse_pose_inlier_8px": 0.90},
        },
        {
            "query_id": "high1",
            "query_index": 1,
            "sparse_te_cm": 5.0,
            "sparse_re_deg": 1.0,
            "sparse_inliers": 820,
            "sparse_confidence": {"sparse_pose_inlier_8px": 0.85},
        },
    ]
    candidate_rows = [
        {"query_id": "low0", "query_index": 0, "sparse_te_cm": 10.0, "sparse_re_deg": 1.0, "sparse_inliers": 80},
        {"query_id": "low1", "query_index": 1, "sparse_te_cm": 12.0, "sparse_re_deg": 1.0, "sparse_inliers": 90},
        {"query_id": "high0", "query_index": 0, "sparse_te_cm": 8.0, "sparse_re_deg": 1.0, "sparse_inliers": 900},
        {"query_id": "high1", "query_index": 1, "sparse_te_cm": 9.0, "sparse_re_deg": 1.0, "sparse_inliers": 910},
    ]

    report = build_feature_stump_gate_cv(
        base_rows,
        candidate_rows,
        features=("sparse_pose_inlier_8px",),
        stage="sparse",
        folds=2,
        min_train_queries=1,
        min_train_activated=1,
        improvement_margin_cm=1.0,
    )

    assert report["summary"]["median_te_cm"] == 7.5
    assert report["paired"]["activated_count"] == 2
    assert report["paired"]["harmed_count"] == 0
    assert {decision["source"] for decision in report["decisions"] if decision["query_id"].startswith("low")} == {
        "candidate"
    }


def test_feature_stump_gate_cv_rejects_train_catastrophic_regression() -> None:
    base_rows = []
    candidate_rows = []
    for index in range(6):
        base_rows.append(
            {
                "query_id": f"low{index}",
                "query_index": index,
                "sparse_te_cm": 100.0 if index >= 2 else 10.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 80,
                "sparse_confidence": {"sparse_pose_inlier_8px": 0.10},
            }
        )
        candidate_rows.append(
            {
                "query_id": f"low{index}",
                "query_index": index,
                "sparse_te_cm": 90.0 if index >= 2 else 70.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 100,
            }
        )
    for index in range(6, 8):
        base_rows.append(
            {
                "query_id": f"high{index}",
                "query_index": index,
                "sparse_te_cm": 5.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 800,
                "sparse_confidence": {"sparse_pose_inlier_8px": 0.90},
            }
        )
        candidate_rows.append(
            {
                "query_id": f"high{index}",
                "query_index": index,
                "sparse_te_cm": 7.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 900,
            }
        )

    report = build_feature_stump_gate_cv(
        base_rows,
        candidate_rows,
        features=("sparse_pose_inlier_8px",),
        stage="sparse",
        folds=2,
        min_train_queries=1,
        min_train_activated=1,
        improvement_margin_cm=1.0,
        stump_max_train_regression_50=0,
    )

    assert report["paired"]["activated_count"] == 0
    assert report["summary"]["median_te_cm"] == 55.0


def test_feature_stump_gate_cv_can_reject_train_20cm_regression() -> None:
    base_rows = []
    candidate_rows = []
    for index, delta in enumerate((30.0, 30.0, -30.0, -30.0, -30.0, -30.0)):
        base_rows.append(
            {
                "query_id": f"low{index}",
                "query_index": index,
                "sparse_te_cm": 40.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 80,
                "sparse_confidence": {"sparse_pose_inlier_8px": 0.10},
            }
        )
        candidate_rows.append(
            {
                "query_id": f"low{index}",
                "query_index": index,
                "sparse_te_cm": 40.0 + delta,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 100,
            }
        )

    report = build_feature_stump_gate_cv(
        base_rows,
        candidate_rows,
        features=("sparse_pose_inlier_8px",),
        stage="sparse",
        folds=2,
        min_train_queries=1,
        min_train_activated=1,
        improvement_margin_cm=1.0,
        stump_max_train_regression_20=0,
        stump_max_train_regression_50=0,
    )

    assert report["paired"]["activated_count"] == 0


def test_feature_rule_gate_cv_uses_second_condition_to_filter_harmful_subset() -> None:
    base_rows = []
    candidate_rows = []
    for index in range(8):
        helped = index in {0, 1, 4, 5}
        base_rows.append(
            {
                "query_id": f"q{index}",
                "query_index": index,
                "sparse_te_cm": 40.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 80,
                "sparse_confidence": {
                    "sparse_pose_inlier_8px": 0.10,
                    "sparse_pose_reproj_median_px": 1.0 if helped else 9.0,
                },
            }
        )
        candidate_rows.append(
            {
                "query_id": f"q{index}",
                "query_index": index,
                "sparse_te_cm": 10.0 if helped else 55.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 100,
            }
        )
    for index in range(8, 10):
        base_rows.append(
            {
                "query_id": f"decoy{index}",
                "query_index": index,
                "sparse_te_cm": 40.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 800,
                "sparse_confidence": {
                    "sparse_pose_inlier_8px": 0.90,
                    "sparse_pose_reproj_median_px": 1.0,
                },
            }
        )
        candidate_rows.append(
            {
                "query_id": f"decoy{index}",
                "query_index": index,
                "sparse_te_cm": 55.0,
                "sparse_re_deg": 1.0,
                "sparse_inliers": 900,
            }
        )

    report = build_feature_rule_gate_cv(
        base_rows,
        candidate_rows,
        features=("sparse_pose_inlier_8px", "sparse_pose_reproj_median_px"),
        stage="sparse",
        folds=2,
        max_depth=2,
        min_train_queries=1,
        min_train_activated=1,
        improvement_margin_cm=1.0,
    )

    assert report["paired"]["activated_count"] == 4
    assert report["paired"]["helped_count"] == 4
    assert report["paired"]["harmed_count"] == 0
    assert {decision["source"] for decision in report["decisions"] if decision["query_id"] in {"q0", "q1", "q4", "q5"}} == {
        "candidate"
    }
    assert {decision["source"] for decision in report["decisions"] if decision["query_id"].startswith("decoy")} == {"base"}
