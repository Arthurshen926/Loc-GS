import json

from loc_gs.reporting.fixed_recipe_gate import (
    build_scene_level_acceptance_recipe,
    evaluate_fixed_recipe_gate,
)


def _run(scene, dense_te, dense_r5, dense_r2):
    return {
        "scene": scene,
        "run_dir": f"/tmp/{scene}",
        "dense": {
            "median_te_cm": dense_te,
            "median_re_deg": 0.1,
            "recall_5cm_5deg": dense_r5,
            "recall_2cm_2deg": dense_r2,
        },
    }


def test_fixed_recipe_gate_passes_only_when_hard_scenes_are_neutral_and_macro_improves():
    baseline = {
        "ShopFacade": _run("ShopFacade", 5.0, 0.5, 0.2),
        "OldHospital": _run("OldHospital", 10.0, 0.2, 0.1),
        "StMarysChurch": _run("StMarysChurch", 4.0, 0.6, 0.3),
    }
    candidate = {
        "ShopFacade": _run("ShopFacade", 4.0, 0.55, 0.22),
        "OldHospital": _run("OldHospital", 10.0, 0.2, 0.1),
        "StMarysChurch": _run("StMarysChurch", 4.0, 0.6, 0.3),
    }

    report = evaluate_fixed_recipe_gate(
        baseline,
        candidate,
        required_positive_scenes=("ShopFacade",),
        neutral_scenes=("OldHospital", "StMarysChurch"),
    )

    assert report["passed"] is True
    assert report["macro_delta"]["median_te_cm"] < 0.0
    assert report["scene_results"]["ShopFacade"]["passed"] is True


def test_fixed_recipe_gate_rejects_oldhospital_dense_regression_even_if_macro_improves():
    baseline = {
        "ShopFacade": _run("ShopFacade", 10.0, 0.4, 0.1),
        "OldHospital": _run("OldHospital", 1.0, 0.4, 0.1),
        "StMarysChurch": _run("StMarysChurch", 10.0, 0.4, 0.1),
    }
    candidate = {
        "ShopFacade": _run("ShopFacade", 1.0, 0.8, 0.2),
        "OldHospital": _run("OldHospital", 2.0, 0.4, 0.1),
        "StMarysChurch": _run("StMarysChurch", 10.0, 0.4, 0.1),
    }

    report = evaluate_fixed_recipe_gate(
        baseline,
        candidate,
        required_positive_scenes=("ShopFacade",),
        neutral_scenes=("OldHospital", "StMarysChurch"),
    )

    assert report["passed"] is False
    assert "median_te_regression" in report["scene_results"]["OldHospital"]["reasons"]


def test_fixed_recipe_gate_payload_is_json_serializable():
    report = evaluate_fixed_recipe_gate(
        {"ShopFacade": _run("ShopFacade", 5.0, 0.5, 0.2)},
        {"ShopFacade": _run("ShopFacade", 4.0, 0.5, 0.2)},
        required_positive_scenes=("ShopFacade",),
        neutral_scenes=(),
    )

    json.dumps(report)


def test_scene_level_acceptance_selects_candidate_only_when_predeclared_gate_passes():
    baseline = {
        "ShopFacade": _run("ShopFacade", 5.0, 0.5, 0.2),
        "OldHospital": _run("OldHospital", 10.0, 0.2, 0.1),
    }
    candidates = {
        "v3": {
            "ShopFacade": _run("ShopFacade", 4.0, 0.55, 0.21),
            "OldHospital": _run("OldHospital", 12.0, 0.25, 0.11),
        }
    }

    report = build_scene_level_acceptance_recipe(
        baseline,
        candidates,
        candidate_order=("v3",),
        required_positive_scenes=("ShopFacade",),
        neutral_scenes=("OldHospital",),
    )

    assert report["selected_runs"]["ShopFacade"]["selected"] == "v3"
    assert report["selected_runs"]["OldHospital"]["selected"] == "native"
    assert report["candidate_evaluations"]["OldHospital"]["v3"]["reasons"] == ["median_te_regression"]
    assert report["global_gate"]["passed"] is True
    json.dumps(report)


def test_scene_level_acceptance_honors_candidate_order():
    baseline = {"ShopFacade": _run("ShopFacade", 5.0, 0.5, 0.2)}
    candidates = {
        "v2": {"ShopFacade": _run("ShopFacade", 4.8, 0.51, 0.2)},
        "v3": {"ShopFacade": _run("ShopFacade", 4.0, 0.55, 0.22)},
    }

    report = build_scene_level_acceptance_recipe(
        baseline,
        candidates,
        candidate_order=("v2", "v3"),
        required_positive_scenes=("ShopFacade",),
    )

    assert report["selected_runs"]["ShopFacade"]["selected"] == "v2"


def test_fixed_recipe_gate_can_warn_on_recall_drop_for_precision_primary_policy():
    baseline = {
        "OldHospital": _run("OldHospital", 10.0, 0.2, 0.1),
    }
    candidate = {
        "OldHospital": _run("OldHospital", 9.0, 0.18, 0.08),
    }

    report = evaluate_fixed_recipe_gate(
        baseline,
        candidate,
        neutral_scenes=("OldHospital",),
        recall_policy="warn",
    )

    scene = report["scene_results"]["OldHospital"]
    assert report["passed"] is True
    assert scene["passed"] is True
    assert scene["reasons"] == []
    assert scene["warnings"] == [
        "recall_2cm_2deg_regression",
        "recall_5cm_5deg_regression",
    ]
    assert report["macro_warnings"] == [
        "macro_recall_2cm_2deg_regression",
        "macro_recall_5cm_5deg_regression",
    ]
