import json

from loc_gs.reporting import frozen_recipe as frozen_recipe_module
from loc_gs.reporting.frozen_recipe import build_frozen_recipe_manifest
from loc_gs.scripts.write_frozen_recipe_manifest import main


def _write_run(
    root,
    scene,
    *,
    audit_status="passed",
    map_path="/tmp/map",
    feedback=False,
    official_parity=True,
    paper_safe_poselib=None,
):
    if paper_safe_poselib is None:
        paper_safe_poselib = official_parity
    run = root / scene
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te_cm": 4.0,
                    "median_re_deg": 0.2,
                    "recall_5cm_5deg": 0.5,
                    "recall_2cm_2deg": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "map_path": map_path,
                "single_path_evaluator": True,
                "single_path_deployment": True,
                "feedback_enabled": feedback,
                "rho_feedback_enabled": False,
                "same_budget": True,
                "quality_gate": {"mode": "disabled", "per_query_branch_selection": False},
                "evaluator_safety": {
                    "official_stdloc_parity_candidate": official_parity,
                    "paper_safe_fixed_poselib_candidate": paper_safe_poselib,
                    "evaluator_variant": "poselib" if paper_safe_poselib else "opencv_prosac_magsac",
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "split_audit.json").write_text(
        json.dumps({"audit_status": audit_status}),
        encoding="utf-8",
    )
    return run


def _acceptance_report(run_dir):
    return {
        "split": "train-dev-q80",
        "recipe": {
            "global_gate": {
                "passed": True,
                "macro_delta": {
                    "median_te_cm": -1.0,
                    "recall_5cm_5deg": 0.1,
                    "recall_2cm_2deg": 0.0,
                },
            },
            "thresholds": {"max_median_te_regression_cm": 0.0, "max_recall_drop": 0.0},
            "candidate_order": ["v3", "v2"],
            "required_positive_scenes": ["ShopFacade"],
            "neutral_scenes": ["OldHospital"],
            "selected_runs": {
                "ShopFacade": {
                    "scene": "ShopFacade",
                    "selected": "v3",
                    "reason": "candidate_passed_predeclared_gate",
                    "run_dir": str(run_dir),
                }
            },
            "candidate_evaluations": {
                "ShopFacade": {
                    "v2": {
                        "passed": False,
                        "reasons": ["recall_5cm_5deg_regression"],
                        "delta": {"recall_5cm_5deg": -0.1},
                    },
                    "v3": {"passed": True, "reasons": [], "delta": {"median_te_cm": -1.0}},
                }
            },
        },
    }


def test_build_frozen_recipe_manifest_records_selected_maps_and_rejections(tmp_path, monkeypatch):
    run = _write_run(tmp_path / "runs", "ShopFacade", map_path="/tmp/selected_map")
    monkeypatch.setattr(
        frozen_recipe_module,
        "_third_party_stdloc_evaluator_modified",
        lambda: False,
    )

    manifest = build_frozen_recipe_manifest(
        {"q80": _acceptance_report(run)},
        freeze_id="lsf-freeze-test",
        notes="unit test",
    )

    selected = manifest["splits"]["q80"]["selected_runs"]["ShopFacade"]
    assert selected["map_path"] == "/tmp/selected_map"
    assert selected["audit_status"] == "passed"
    assert selected["single_path_evaluator"] is True
    assert manifest["splits"]["q80"]["rejected_candidates"]["ShopFacade"][0]["candidate"] == "v2"
    assert manifest["checks"]["acceptance_gates_passed"] is True
    assert manifest["paper_facing_ready"] is True
    json.dumps(manifest)


def test_build_frozen_recipe_manifest_not_paper_ready_with_unknown_selected_audit(tmp_path):
    run = _write_run(tmp_path / "runs", "ShopFacade", audit_status="unknown")

    manifest = build_frozen_recipe_manifest(
        {"q80": _acceptance_report(run)},
        freeze_id="lsf-freeze-test",
    )

    assert manifest["checks"]["all_selected_runs_split_audit_passed"] is False
    assert manifest["paper_facing_ready"] is False


def test_build_frozen_recipe_manifest_not_paper_ready_with_modified_stdloc_evaluator(
    tmp_path, monkeypatch
):
    run = _write_run(tmp_path / "runs", "ShopFacade")
    monkeypatch.setattr(
        frozen_recipe_module,
        "_third_party_stdloc_evaluator_modified",
        lambda: True,
    )

    manifest = build_frozen_recipe_manifest(
        {"q80": _acceptance_report(run)},
        freeze_id="lsf-freeze-test",
    )

    assert manifest["checks"]["third_party_stdloc_evaluator_modified"] is True
    assert manifest["paper_facing_ready"] is False


def test_build_frozen_recipe_manifest_not_paper_ready_with_non_official_selected_run(
    tmp_path, monkeypatch
):
    run = _write_run(tmp_path / "runs", "ShopFacade", official_parity=False)
    monkeypatch.setattr(
        frozen_recipe_module,
        "_third_party_stdloc_evaluator_modified",
        lambda: False,
    )

    manifest = build_frozen_recipe_manifest(
        {"q80": _acceptance_report(run)},
        freeze_id="lsf-freeze-test",
    )

    assert manifest["checks"]["all_selected_runs_official_stdloc_parity"] is False
    assert manifest["checks"]["all_selected_runs_paper_safe_fixed_poselib"] is False
    assert manifest["paper_facing_ready"] is False


def test_build_frozen_recipe_manifest_allows_poselib_method_cfg_not_baseline_parity(
    tmp_path, monkeypatch
):
    run = _write_run(
        tmp_path / "runs",
        "ShopFacade",
        official_parity=False,
        paper_safe_poselib=True,
    )
    monkeypatch.setattr(
        frozen_recipe_module,
        "_third_party_stdloc_evaluator_modified",
        lambda: False,
    )

    manifest = build_frozen_recipe_manifest(
        {"q80": _acceptance_report(run)},
        freeze_id="lsf-freeze-test",
    )

    assert manifest["checks"]["all_selected_runs_official_stdloc_parity"] is False
    assert manifest["checks"]["all_selected_runs_paper_safe_fixed_poselib"] is True
    assert manifest["paper_facing_ready"] is True


def test_write_frozen_recipe_manifest_cli(tmp_path):
    run = _write_run(tmp_path / "runs", "ShopFacade")
    report = tmp_path / "acceptance.json"
    report.write_text(json.dumps(_acceptance_report(run)), encoding="utf-8")
    output = tmp_path / "frozen.json"

    assert (
        main(
            [
                "--acceptance-report",
                f"q80={report}",
                "--freeze-id",
                "lsf-freeze-test",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["freeze_id"] == "lsf-freeze-test"
    assert payload["splits"]["q80"]["selected_runs"]["ShopFacade"]["selected"] == "v3"
