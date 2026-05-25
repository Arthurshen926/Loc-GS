import json

from loc_gs.reporting.precision_primary_gate import evaluate_precision_primary_gate
from loc_gs.scripts.write_precision_primary_gate_report import main


def _run(scene, dense_te, dense_re, dense_r5, dense_r2):
    return {
        "scene": scene,
        "run_dir": f"/tmp/{scene}",
        "dense": {
            "median_te_cm": dense_te,
            "median_re_deg": dense_re,
            "recall_5cm_5deg": dense_r5,
            "recall_2cm_2deg": dense_r2,
        },
    }


def test_precision_primary_gate_accepts_pose_precision_gain_with_recall_tolerance():
    report = evaluate_precision_primary_gate(
        {"ShopFacade": _run("ShopFacade", 6.65, 0.36, 0.445, 0.138)},
        {"ShopFacade": _run("ShopFacade", 5.29, 0.31, 0.493, 0.164)},
        required_positive_scenes=("ShopFacade",),
        max_recall_drop=0.002,
        max_abs_median_te_cm=100.0,
        max_abs_median_re_deg=5.0,
    )

    scene = report["scene_results"]["ShopFacade"]
    assert report["passed"] is True
    assert scene["passed"] is True
    assert scene["precision_improved"] is True
    assert scene["recall_safe"] is True
    assert scene["absolute_pose_usable"] is True


def test_precision_primary_gate_rejects_pose_regression_even_when_recall_improves():
    report = evaluate_precision_primary_gate(
        {"OldHospital": _run("OldHospital", 620.2, 10.8, 0.035, 0.002)},
        {"OldHospital": _run("OldHospital", 638.7, 12.0, 0.036, 0.005)},
        neutral_scenes=("OldHospital",),
        max_recall_drop=0.002,
        max_abs_median_te_cm=100.0,
        max_abs_median_re_deg=5.0,
    )

    scene = report["scene_results"]["OldHospital"]
    assert report["passed"] is False
    assert scene["precision_improved"] is False
    assert "missing_precision_gain" in scene["reasons"]


def test_precision_primary_gate_rejects_unusable_absolute_pose():
    report = evaluate_precision_primary_gate(
        {"StMarysChurch": _run("StMarysChurch", 4369.9, 106.1, 0.1015, 0.0316)},
        {"StMarysChurch": _run("StMarysChurch", 4364.6, 103.8, 0.1002, 0.0323)},
        required_positive_scenes=("StMarysChurch",),
        max_recall_drop=0.002,
        max_abs_median_te_cm=100.0,
        max_abs_median_re_deg=5.0,
    )

    scene = report["scene_results"]["StMarysChurch"]
    assert scene["precision_improved"] is True
    assert scene["recall_safe"] is True
    assert scene["absolute_pose_usable"] is False
    assert scene["passed"] is False
    assert "absolute_pose_not_usable" in scene["reasons"]


def _write_run(root, scene, dense_te, dense_re, dense_r5, dense_r2):
    run = root / scene
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te_cm": dense_te,
                    "median_re_deg": dense_re,
                    "recall_5cm_5deg": dense_r5,
                    "recall_2cm_2deg": dense_r2,
                }
            }
        ),
        encoding="utf-8",
    )
    return run


def test_write_precision_primary_gate_report_cli(tmp_path):
    native = tmp_path / "native"
    cand = tmp_path / "candidate"
    shop_native = _write_run(native, "ShopFacade", 6.65, 0.36, 0.445, 0.138)
    shop_candidate = _write_run(cand, "ShopFacade", 5.29, 0.31, 0.493, 0.164)
    output = tmp_path / "precision.json"

    assert (
        main(
            [
                "--baseline",
                f"ShopFacade={shop_native}",
                "--candidate",
                f"ShopFacade={shop_candidate}",
                "--required-positive-scene",
                "ShopFacade",
                "--max-recall-drop",
                "0.002",
                "--max-abs-median-te-cm",
                "100",
                "--max-abs-median-re-deg",
                "5",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["scene_results"]["ShopFacade"]["passed"] is True
