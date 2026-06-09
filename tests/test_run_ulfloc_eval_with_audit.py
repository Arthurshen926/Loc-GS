import json
from argparse import Namespace
from pathlib import Path

import pytest

from loc_gs.scripts.run_ulfloc_eval_with_audit import (
    build_ulfloc_eval_command,
    create_ulfloc_eval_source_overlay,
    metrics_from_ulfloc_results,
    split_audit_for_ulfloc_eval,
    write_ulfloc_eval_audit,
)


def _result(sparse_te, sparse_re, dense_te, dense_re):
    return {
        "sparse": {"inliers": 8},
        "dense": [{"inliers": 24}],
        "sparse_TE": sparse_te,
        "sparse_AE": sparse_re,
        "dense_TE": dense_te,
        "dense_AE": dense_re,
    }


def test_metrics_from_ulfloc_results_adds_hard_and_recall_metrics():
    metrics = metrics_from_ulfloc_results(
        [
            _result(4.0, 1.0, 4.0, 1.0),
            _result(12.0, 1.0, 12.0, 1.0),
            _result(200.0, 1.0, 600.0, 1.0),
        ]
    )

    assert metrics["dense"]["median_te_cm"] == pytest.approx(12.0)
    assert metrics["dense"]["R50cm5deg"] == pytest.approx(2 / 3)
    assert metrics["dense"]["R15cm5deg"] == pytest.approx(2 / 3)
    assert metrics["dense"]["R10cm5deg"] == pytest.approx(1 / 3)
    assert metrics["dense"]["R5cm5deg"] == pytest.approx(1 / 3)
    assert metrics["dense"]["R2cm2deg"] == pytest.approx(0.0)
    assert metrics["dense"]["p95_te_cm"] == pytest.approx(541.2)
    assert metrics["dense"]["cvar10_te_cm"] == pytest.approx(600.0)
    assert metrics["dense"]["severe_rate_1m"] == pytest.approx(1 / 3)
    assert metrics["dense"]["catastrophic_rate_5m"] == pytest.approx(1 / 3)


def test_metrics_from_ulfloc_results_adds_sparse_to_dense_transition_metrics():
    metrics = metrics_from_ulfloc_results(
        [
            _result(4.0, 1.0, 60.0, 1.0),
            _result(20.0, 1.0, 45.0, 1.0),
            _result(80.0, 1.0, 30.0, 1.0),
        ]
    )

    paired = metrics["paired_sparse_dense"]
    assert paired["dense_worsened_count"] == 2
    assert paired["dense_worsened_5cm_count"] == 2
    assert paired["dense_regression_20cm_count"] == 2
    assert paired["sparse_correct_dense_wrong_count"] == 1
    assert paired["dense_improved_20cm_count"] == 1


def test_metrics_from_ulfloc_results_summarizes_dense_guard_decisions():
    guarded_reject = _result(4.0, 1.0, 4.0, 1.0)
    guarded_reject["dense"] = [
        {
            "inliers": 10,
            "selected_pose_source": "native_dense_rejected_before_retry",
            "dense_transition_guard": {"decision": "reject_dense_keep_sparse"},
        },
        {
            "inliers": 20,
            "selected_pose_source": "dense_feedback_retry",
            "dense_render_feedback_retry": True,
            "dense_transition_guard": {"decision": "accept_dense"},
        },
    ]
    guarded_sparse = _result(5.0, 1.0, 5.0, 1.0)
    guarded_sparse["dense"] = [
        {
            "inliers": 10,
            "selected_pose_source": "sparse",
            "dense_transition_guard": {"decision": "reject_dense_keep_sparse"},
        }
    ]
    guarded_risky = _result(6.0, 1.0, 3.0, 1.0)
    guarded_risky["dense"] = [
        {
            "inliers": 30,
            "selected_pose_source": "native_dense_risky_before_retry",
            "native_dense_risky_before_retry": True,
            "dense_transition_guard": {"decision": "accept_dense"},
        }
    ]

    metrics = metrics_from_ulfloc_results([guarded_reject, guarded_sparse, guarded_risky])

    guard = metrics["dense_guard"]
    assert guard["guard_entry_count"] == 4
    assert guard["accept_dense_count"] == 2
    assert guard["reject_dense_keep_sparse_count"] == 2
    assert guard["selected_sparse_count"] == 1
    assert guard["selected_dense_feedback_retry_count"] == 1
    assert guard["retry_entry_count"] == 1
    assert guard["native_dense_rejected_before_retry_count"] == 1
    assert guard["native_dense_risky_before_retry_count"] == 1


def test_split_audit_marks_official_test_as_not_tuning_safe():
    audit = split_audit_for_ulfloc_eval(split_name="test", feedback_manifest=None)

    assert audit["audit_status"] == "passed"
    assert audit["official_test_used"] is True
    assert audit["paper_safe_for_tuning"] is False
    assert audit["checks"]["feedback_bank_split"]["status"] == "passed"


def test_split_audit_rejects_test_feedback_manifest():
    audit = split_audit_for_ulfloc_eval(
        split_name="test",
        feedback_manifest={"split_name": "test"},
    )

    assert audit["audit_status"] == "failed"
    assert audit["checks"]["feedback_bank_split"]["status"] == "failed"


def test_split_audit_checks_train_dev_disjoint_from_official_test():
    audit = split_audit_for_ulfloc_eval(
        split_name="train-dev",
        feedback_manifest={"split_name": "selfmap_train_a"},
        query_image_ids={"seq1/frame00001.png"},
        official_test_image_ids={"seq2/frame00002.png"},
        feedback_image_ids={"seq3/frame00003.png"},
    )

    assert audit["audit_status"] == "passed"
    assert audit["paper_safe_for_tuning"] is True
    assert audit["checks"]["query_official_test_disjointness"]["status"] == "passed"
    assert audit["checks"]["feedback_eval_disjointness"]["status"] == "passed"


def test_split_audit_fails_feedback_that_overlaps_eval_queries():
    audit = split_audit_for_ulfloc_eval(
        split_name="train-dev",
        feedback_manifest={"split_name": "selfmap_train_a"},
        query_image_ids={"seq1/frame00001.png"},
        official_test_image_ids={"seq2/frame00002.png"},
        feedback_image_ids={"seq1/frame00001.png"},
    )

    assert audit["audit_status"] == "failed"
    assert audit["checks"]["feedback_eval_disjointness"]["overlap"] == ["seq1/frame00001.png"]


def test_split_audit_marks_missing_feedback_eval_ids_unknown():
    audit = split_audit_for_ulfloc_eval(
        split_name="train-dev",
        feedback_manifest={"split_name": "selfmap_train_a"},
        query_image_ids={"seq1/frame00001.png"},
        official_test_image_ids={"seq2/frame00002.png"},
    )

    assert audit["audit_status"] == "unknown"
    assert audit["checks"]["feedback_eval_disjointness"]["status"] == "unknown"


def test_split_audit_fails_train_dev_that_overlaps_official_test():
    audit = split_audit_for_ulfloc_eval(
        split_name="train-dev",
        feedback_manifest={"split_name": "selfmap_train_a"},
        query_image_ids={"seq1/frame00001.png"},
        official_test_image_ids={"seq1/frame00001.png"},
    )

    assert audit["audit_status"] == "failed"
    assert audit["checks"]["query_official_test_disjointness"]["overlap"] == ["seq1/frame00001.png"]


def test_run_ulfloc_eval_command_resolves_cambridge_source_for_loader(tmp_path):
    source = Path("loc_gs/scripts/run_ulfloc_eval_with_audit.py").read_text(encoding="utf-8")

    assert "resolve_ulfloc_source_path_for_loader" in source
    assert "Path(args.model_path) / \"_ulf_loader_links\"" in source


def test_create_ulfloc_eval_source_overlay_writes_heldout_test_list(tmp_path):
    source = tmp_path / "ShopFacade"
    source.mkdir()
    (source / "processed").mkdir()
    (source / "sparse").mkdir()
    (source / "dataset_train.txt").write_text("seq1/a.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    (source / "dataset_test.txt").write_text("seq2/test.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    query_list = tmp_path / "train_dev.txt"
    query_list.write_text("seq1/a.png 0 0 0 1 0 0 0\nseq1/b.png 0 0 0 1 0 0 0\n", encoding="utf-8")

    overlay = create_ulfloc_eval_source_overlay(
        source_path=source,
        query_list=query_list,
        overlay_root=tmp_path / "overlays",
        split_name="train-dev",
    )

    assert "cambridge" in str(overlay)
    assert (overlay / "processed").is_symlink()
    assert (overlay / "sparse").is_symlink()
    assert (overlay / "dataset_test.txt").read_text(encoding="utf-8").splitlines() == [
        "seq1/a.png 0 0 0 1 0 0 0",
        "seq1/b.png 0 0 0 1 0 0 0",
    ]


def test_build_ulfloc_eval_command_uses_query_list_overlay(tmp_path):
    source = tmp_path / "ShopFacade"
    source.mkdir()
    (source / "processed").mkdir()
    (source / "sparse").mkdir()
    (source / "dataset_test.txt").write_text("seq2/test.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    query_list = tmp_path / "train_dev.txt"
    query_list.write_text("seq1/a.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    args = Namespace(
        python="/python",
        ulf_root=Path("/ulf"),
        source_path=source,
        model_path=model,
        cfg=Path("/ulf/config.yaml"),
        images="processed",
        feature_type="sp",
        iteration=-1,
        data_device="cpu",
        longest_edge=640,
        prefix=None,
        cuda_visible_devices="",
        split_name="train-dev",
        query_list=query_list,
    )

    command, _ = build_ulfloc_eval_command(args)

    source_arg = Path(command[command.index("-s") + 1])
    assert source_arg.parent == model / "_locgs_eval_source_overlays"
    assert (source_arg / "dataset_test.txt").read_text(encoding="utf-8").strip().startswith("seq1/a.png")


def test_build_ulfloc_eval_command_respects_cuda_argument():
    args = Namespace(
        python="/python",
        ulf_root=Path("/ulf"),
        source_path=Path("/data/ShopFacade"),
        model_path=Path("/model/ShopFacade"),
        cfg=Path("/ulf/config.yaml"),
        images="processed",
        feature_type="sp",
        iteration=-1,
        data_device="cpu",
        longest_edge=640,
        prefix=None,
        cuda_visible_devices="2",
        query_list=None,
    )

    command, env = build_ulfloc_eval_command(args)

    assert command[:2] == ["/python", "/ulf/ulfloc.py"]
    assert "--cfg" in command
    assert Path(command[command.index("--cfg") + 1]).is_absolute()
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_write_ulfloc_eval_audit_writes_required_bundle(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    summary = {
        "model_path": "/model/ShopFacade",
        "sparse": {"median_te": 3.0, "median_ae": 0.1},
        "dense": {"median_te": 2.5, "median_ae": 0.09},
    }
    results = [_result(3.0, 0.1, 2.5, 0.09)]
    (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (eval_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    write_ulfloc_eval_audit(
        output_dir=eval_dir,
        command=["python", "ulfloc.py"],
        scene="ShopFacade",
        split_name="test",
        source_path=Path("/data/ShopFacade"),
        model_path=Path("/model/ShopFacade"),
        cfg=Path("/ulf/config.yaml"),
        feedback_manifest=None,
        ulf_root=Path("/ulf"),
        hyperparameters={"longest_edge": 640},
    )

    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((eval_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((eval_dir / "split_audit.json").read_text(encoding="utf-8"))

    assert manifest["scene"] == "ShopFacade"
    assert manifest["split"] == "test"
    assert manifest["real_query_path"] == "ULF-Loc native: sparse PnP -> LGCV dense refinement"
    assert metrics["dense"]["median_te_cm"] == pytest.approx(2.5)
    assert "paired_sparse_dense" in metrics
    assert split_audit["paper_safe_for_tuning"] is False
    assert (eval_dir / "command.txt").exists()
    assert (eval_dir / "git_status.txt").exists()
    assert (eval_dir / "ulf_git_status.txt").exists()


def test_write_ulfloc_eval_audit_attaches_image_names_from_ulfloc_log_order(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    summary = {
        "model_path": "/model/OldHospital",
        "sparse": {"median_te": 10.0, "median_ae": 0.1},
        "dense": {"median_te": 8.0, "median_ae": 0.09},
    }
    results = [
        _result(10.0, 0.1, 8.0, 0.09),
        _result(20.0, 0.2, 9.0, 0.10),
    ]
    (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (eval_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (eval_dir / "output.log").write_text(
        "\n".join(
            [
                "Localize image:seq9/frame00076.png",
                "sparse: AE: 0.1deg, TE: 10.0cm",
                "dense: AE: 0.09deg, TE: 8.0cm",
                "Localize image:seq1/frame00003.png",
                "sparse: AE: 0.2deg, TE: 20.0cm",
                "dense: AE: 0.10deg, TE: 9.0cm",
            ]
        ),
        encoding="utf-8",
    )

    write_ulfloc_eval_audit(
        output_dir=eval_dir,
        command=["python", "ulfloc.py"],
        scene="OldHospital",
        split_name="train-dev",
        source_path=Path("/data/OldHospital"),
        model_path=Path("/model/OldHospital"),
        cfg=Path("/ulf/config.yaml"),
        feedback_manifest=None,
        ulf_root=Path("/ulf"),
        hyperparameters={"longest_edge": 640},
        query_image_ids={"seq9/frame00076.png", "seq1/frame00003.png"},
    )

    repaired_results = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))

    assert [row["image_name"] for row in repaired_results] == [
        "seq9/frame00076.png",
        "seq1/frame00003.png",
    ]
    assert manifest["query_result_order_source"] == "output.log:Localize image"
