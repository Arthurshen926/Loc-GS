import json

from loc_gs.reporting.frozen_recipe_validation import build_frozen_recipe_validation_report
from loc_gs.scripts.write_frozen_recipe_validation_report import main


def _run(
    scene,
    root,
    dense_te,
    dense_r5,
    dense_r2,
    *,
    split="train",
    audit_status="passed",
    split_audit_payload=None,
    manifest_updates=None,
):
    run = root / scene
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te_cm": dense_te,
                    "median_re_deg": 0.2,
                    "recall_5cm_5deg": dense_r5,
                    "recall_2cm_2deg": dense_r2,
                },
                "sparse": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "scene": scene,
        "split": split,
        "map_path": f"/tmp/maps/{scene}",
        "single_path_evaluator": True,
        "feedback_enabled": False,
        "residual_enabled": False,
        "selector_enabled": False,
        "rho_feedback_enabled": False,
        "quality_gate": {"mode": "disabled", "per_query_branch_selection": False},
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    split_audit_payload = split_audit_payload or {
        "audit_status": audit_status,
        "paper_safe": audit_status == "passed",
        "checks": {
            "feedback_bank_split": {
                "status": audit_status,
                "split_name": "native_eval_train",
            }
        },
    }
    (run / "split_audit.json").write_text(json.dumps(split_audit_payload), encoding="utf-8")
    (run / "command.txt").write_text("eval command", encoding="utf-8")
    (run / "git_status.txt").write_text("M some_file.py\n", encoding="utf-8")
    return run


def _run_summary(scene, root, dense_te, dense_r5, dense_r2):
    run_dir = _run(scene, root, dense_te, dense_r5, dense_r2)
    return {
        "scene": scene,
        "run_dir": str(run_dir),
        "dense": {
            "median_te_cm": dense_te,
            "recall_5cm_5deg": dense_r5,
            "recall_2cm_2deg": dense_r2,
        },
        "manifest": {
            "split": "train",
            "single_path_evaluator": True,
            "feedback_enabled": False,
            "rho_feedback_enabled": False,
        },
        "split_audit": {"audit_status": "passed", "paper_safe": True},
    }


def test_frozen_recipe_validation_does_not_reselect_native_when_selected_run_regresses(tmp_path):
    baseline = {
        "ShopFacade": _run_summary("ShopFacade", tmp_path / "native", 5.0, 0.5, 0.2),
    }
    selected = {
        "ShopFacade": _run_summary("ShopFacade", tmp_path / "selected", 6.0, 0.49, 0.2),
    }

    report = build_frozen_recipe_validation_report(
        baseline,
        selected,
        required_positive_scenes=("ShopFacade",),
        split="train",
    )

    assert report["policy"] == "frozen_recipe_validation_no_reselection"
    assert report["global_gate"]["passed"] is False
    assert set(report["global_gate"]["scene_results"]["ShopFacade"]["reasons"]) == {
        "median_te_regression",
        "missing_required_positive_gain",
        "recall_5cm_5deg_regression",
    }
    assert report["runs"]["ShopFacade"]["selected"]["run_dir"].endswith("/selected/ShopFacade")
    assert report["checks"]["no_reselection"] is True


def test_write_frozen_recipe_validation_report_cli(tmp_path):
    shop_native = _run("ShopFacade", tmp_path / "native", 5.0, 0.5, 0.2)
    old_native = _run("OldHospital", tmp_path / "native", 10.0, 0.2, 0.1)
    shop_selected = _run("ShopFacade", tmp_path / "selected", 4.0, 0.55, 0.21)
    old_selected = _run("OldHospital", tmp_path / "selected", 10.0, 0.2, 0.1)
    output = tmp_path / "validation.json"

    assert (
        main(
            [
                "--baseline",
                f"ShopFacade={shop_native}",
                "--baseline",
                f"OldHospital={old_native}",
                "--selected",
                f"ShopFacade={shop_selected}",
                "--selected",
                f"OldHospital={old_selected}",
                "--required-positive-scene",
                "ShopFacade",
                "--neutral-scene",
                "OldHospital",
                "--split",
                "train",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["global_gate"]["passed"] is True
    assert payload["checks"]["paper_safe_validation_ready"] is True
    assert payload["runs"]["ShopFacade"]["selected"]["run_dir"] == str(shop_selected)


def test_frozen_recipe_validation_checks_feedback_bank_split_from_split_audit(tmp_path):
    split_audit = {
        "audit_status": "failed",
        "paper_safe": False,
        "checks": {
            "feedback_bank_split": {
                "status": "failed",
                "split_name": "test",
            }
        },
    }
    shop_native = _run("ShopFacade", tmp_path / "native", 5.0, 0.5, 0.2)
    shop_selected = _run(
        "ShopFacade",
        tmp_path / "selected",
        4.0,
        0.55,
        0.21,
        split_audit_payload=split_audit,
    )
    output = tmp_path / "validation.json"

    assert (
        main(
            [
                "--baseline",
                f"ShopFacade={shop_native}",
                "--selected",
                f"ShopFacade={shop_selected}",
                "--required-positive-scene",
                "ShopFacade",
                "--split",
                "train",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["checks"]["feedback_bank_split_not_test"] is False
    assert payload["checks"]["feedback_bank_split_statuses"] == ["failed"]
    assert payload["checks"]["feedback_bank_split_names"] == ["test"]
    assert payload["checks"]["paper_safe_validation_ready"] is False


def test_frozen_recipe_validation_rejects_enabled_selector_or_residual_eval(tmp_path):
    shop_native = _run("ShopFacade", tmp_path / "native", 5.0, 0.5, 0.2)
    shop_selected = _run(
        "ShopFacade",
        tmp_path / "selected",
        4.0,
        0.55,
        0.21,
        manifest_updates={"selector_enabled": True},
    )
    output = tmp_path / "validation.json"

    assert (
        main(
            [
                "--baseline",
                f"ShopFacade={shop_native}",
                "--selected",
                f"ShopFacade={shop_selected}",
                "--required-positive-scene",
                "ShopFacade",
                "--split",
                "train",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["global_gate"]["passed"] is True
    assert payload["checks"]["feedback_disabled_at_eval"] is False
    assert payload["checks"]["paper_safe_validation_ready"] is False
