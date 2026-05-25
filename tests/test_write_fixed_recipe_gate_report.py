import json

from loc_gs.scripts.write_fixed_recipe_gate_report import main


def _write_run(root, scene, dense_te, dense_r5, dense_r2):
    run = root / scene
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te_cm": dense_te,
                    "median_re_deg": 0.1,
                    "recall_5cm_5deg": dense_r5,
                    "recall_2cm_2deg": dense_r2,
                },
                "sparse": {},
                "sampled_count": 8192,
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "split": "train",
                "map_path": "/tmp/map",
                "data_roots": ["/tmp/cambridge/ShopFacade"],
                "hyperparameters": {"max_test_cameras": 80, "eval_split": "train"},
                "single_path_evaluator": True,
                "feedback_enabled": False,
                "rho_feedback_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "split_audit.json").write_text(
        json.dumps({"audit_status": "unknown", "paper_safe": False}),
        encoding="utf-8",
    )
    return run


def test_write_fixed_recipe_gate_report_cli_selects_scene_level_candidate(tmp_path):
    native = tmp_path / "native"
    cand = tmp_path / "candidate"
    shop_native = _write_run(native, "ShopFacade", 5.0, 0.5, 0.2)
    old_native = _write_run(native, "OldHospital", 10.0, 0.2, 0.1)
    shop_candidate = _write_run(cand, "ShopFacade", 4.0, 0.55, 0.21)
    old_candidate = _write_run(cand, "OldHospital", 12.0, 0.25, 0.11)
    output = tmp_path / "report.json"

    assert (
        main(
            [
                "--baseline",
                f"ShopFacade={shop_native}",
                "--baseline",
                f"OldHospital={old_native}",
                "--candidate",
                f"v3:ShopFacade={shop_candidate}",
                "--candidate",
                f"v3:OldHospital={old_candidate}",
                "--candidate-order",
                "v3",
                "--required-positive-scene",
                "ShopFacade",
                "--neutral-scene",
                "OldHospital",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    selected = report["recipe"]["selected_runs"]
    assert selected["ShopFacade"]["selected"] == "v3"
    assert selected["OldHospital"]["selected"] == "native"
    assert report["recipe"]["global_gate"]["passed"] is True
    assert report["recipe"]["selected_runs"]["ShopFacade"]["run_dir"] == str(shop_candidate)
    assert report["recipe"]["selected_runs"]["ShopFacade"]["data_roots"] == ["/tmp/cambridge/ShopFacade"]
    assert report["recipe"]["selected_runs"]["ShopFacade"]["hyperparameters"]["max_test_cameras"] == 80


def test_write_fixed_recipe_gate_report_cli_supports_precision_primary_recall_warnings(tmp_path):
    native = tmp_path / "native"
    cand = tmp_path / "candidate"
    old_native = _write_run(native, "OldHospital", 10.0, 0.2, 0.1)
    old_candidate = _write_run(cand, "OldHospital", 9.0, 0.18, 0.08)
    output = tmp_path / "report.json"

    assert (
        main(
            [
                "--baseline",
                f"OldHospital={old_native}",
                "--candidate",
                f"v6:OldHospital={old_candidate}",
                "--candidate-order",
                "v6",
                "--neutral-scene",
                "OldHospital",
                "--recall-policy",
                "warn",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    selected = report["recipe"]["selected_runs"]["OldHospital"]
    assert selected["selected"] == "v6"
    assert report["recipe"]["global_gate"]["passed"] is True
    assert report["recipe"]["global_gate"]["scene_results"]["OldHospital"]["warnings"] == [
        "recall_2cm_2deg_regression",
        "recall_5cm_5deg_regression",
    ]
