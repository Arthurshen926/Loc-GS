import json

from loc_gs.export.manifest import audit_split_usage, write_export_eval_audit_bundle


def test_split_audit_passes_for_disjoint_non_test_feedback_and_validation_gate():
    audit = audit_split_usage(
        selfmap_image_ids=["train_1", "train_2"],
        calibration_image_ids=["calib_1"],
        test_image_ids=["test_1", "test_2"],
        feedback_bank_manifest={"split_name": "selfmap_train"},
        quality_gate={"mode": "selfmap_validation", "per_query_branch_selection": False},
    )

    assert audit["audit_status"] == "passed"
    assert audit["checks"]["image_id_disjointness"]["status"] == "passed"
    assert audit["checks"]["feedback_bank_split"]["status"] == "passed"
    assert audit["checks"]["quality_gate"]["status"] == "passed"


def test_split_audit_marks_missing_split_info_unknown_not_passed():
    audit = audit_split_usage(
        selfmap_image_ids=None,
        calibration_image_ids=None,
        test_image_ids=None,
        feedback_bank_manifest={},
        quality_gate={},
    )

    assert audit["audit_status"] == "unknown"
    assert audit["checks"]["image_id_disjointness"]["status"] == "unknown"
    assert audit["checks"]["feedback_bank_split"]["status"] == "unknown"
    assert audit["checks"]["quality_gate"]["status"] == "unknown"


def test_split_audit_fails_on_overlap_test_feedback_or_per_query_gate():
    audit = audit_split_usage(
        selfmap_image_ids=["train_1", "test_1"],
        calibration_image_ids=["calib_1"],
        test_image_ids=["test_1"],
        feedback_bank_manifest={"split_name": "test"},
        quality_gate={"mode": "per_query_branch_selection", "per_query_branch_selection": True},
    )

    assert audit["audit_status"] == "failed"
    assert audit["checks"]["image_id_disjointness"]["status"] == "failed"
    assert audit["checks"]["feedback_bank_split"]["status"] == "failed"
    assert audit["checks"]["quality_gate"]["status"] == "failed"
    assert audit["checks"]["image_id_disjointness"]["overlap"] == ["test_1"]


def test_write_export_eval_audit_bundle_writes_required_files(tmp_path):
    split_audit = audit_split_usage(
        selfmap_image_ids=["train_1"],
        calibration_image_ids=["calib_1"],
        test_image_ids=["test_1"],
        feedback_bank_manifest={"split_name": "calibration"},
        quality_gate={"mode": "disabled", "per_query_branch_selection": False},
    )

    paths = write_export_eval_audit_bundle(
        tmp_path,
        manifest={
            "scene": "ShopFacade",
            "baseline_map": "output/stdloc/map_cambridge_spgs/ShopFacade",
            "checkpoint": "output/stdloc_hybrid/ShopFacade/latest.pth",
            "feedback_bank": "output/feedback_banks/ShopFacade/feedback_bank.jsonl",
            "rho": 0.25,
            "residual_enabled": True,
            "selector_enabled": False,
            "single_path_evaluator": True,
        },
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
        metrics_summary={"dense": {"median_te_cm": 2.5, "recall_5cm_5deg": 0.8}},
        split_audit=split_audit,
        git_diff_text="diff --git a/file b/file\n",
    )

    assert set(paths) >= {
        "manifest",
        "command",
        "metrics_summary",
        "split_audit",
        "git_diff",
    }
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "command.txt").read_text(encoding="utf-8") == "python -m loc_gs.scripts.eval_stdloc_native\n"
    assert json.loads((tmp_path / "metrics_summary.json").read_text(encoding="utf-8"))["dense"]["median_te_cm"] == 2.5
    assert json.loads((tmp_path / "split_audit.json").read_text(encoding="utf-8"))["audit_status"] == "passed"
    assert (tmp_path / "git_diff.patch").exists()
