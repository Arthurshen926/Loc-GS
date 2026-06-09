from __future__ import annotations

import json

import pytest

from loc_gs.feedback.io import save_feedback_bank
from loc_gs.feedback.audit import audit_feedback_bank_v3
from loc_gs.feedback.schema import FeedbackMatchRecord
from loc_gs.stdloc_native.sparse_pnp_validation import (
    build_sparse_pnp_validation_profile,
    load_ulfloc_sparse_rows,
    sparse_validation_landmark_risk_tensor,
)
from loc_gs.scripts.build_sparse_pnp_validation_profile import build_argparser, main as build_profile_main
from loc_gs.scripts.export_ulfloc_sparse_feedback import build_argparser as build_sparse_feedback_argparser


def _write_run(path, rows, image_names):
    path.mkdir(parents=True)
    (path / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    log_lines = []
    for image_name, row in zip(image_names, rows):
        log_lines.append(f"Localize image:{image_name}")
        log_lines.append(f"sparse: AE: {row['sparse_AE']}deg, TE: {row['sparse_TE']}cm")
    (path / "output.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _v3_record(**kwargs):
    defaults = dict(
        scene="ShopFacade",
        query_id="q_good.png",
        image_id="q_good.png",
        source_role="candidate_trace",
        keypoint_id="kp:1",
        keypoint_xy=(12.0, 34.0),
        matched_landmark_id="sampled:7",
        matched_gaussian_id="7",
        descriptor_score=0.8,
        descriptor_margin=0.2,
        detector_score=0.7,
        pnp_inlier=True,
        reprojection_error_px=1.0,
        local_geometry_score=0.9,
        depth_m=6.0,
        camera_xyz=(1.0, 2.0, 6.0),
        bearing=(0.0, 0.1, 1.0),
        query_xy_norm=(0.25, 0.5),
        image_cell=(1, 2),
        depth_bin=3,
        pose_success=True,
        pnp_success=True,
        query_sparse_te_cm=4.0,
    )
    defaults.update(kwargs)
    return FeedbackMatchRecord(**defaults)


def test_load_ulfloc_sparse_rows_uses_output_log_query_ids(tmp_path):
    run = tmp_path / "run"
    _write_run(
        run,
        [{"sparse_AE": 0.1, "sparse_TE": 3.0, "sparse": {"inliers": 12}}],
        ["seq/frame00001.png"],
    )

    rows = load_ulfloc_sparse_rows(run)

    assert rows[0]["query_id"] == "seq/frame00001.png"
    assert rows[0]["sparse_te_cm"] == 3.0
    assert rows[0]["sparse_inliers"] == 12


def test_sparse_pnp_validation_profile_marks_protected_regressions(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        [
            {"sparse_AE": 0.1, "sparse_TE": 4.0, "sparse": {"inliers": 100}},
            {"sparse_AE": 0.2, "sparse_TE": 60.0, "sparse": {"inliers": 20}},
        ],
        ["q_good.png", "q_hard.png"],
    )
    _write_run(
        candidate,
        [
            {"sparse_AE": 4.0, "sparse_TE": 90.0, "sparse": {"inliers": 40}},
            {"sparse_AE": 0.1, "sparse_TE": 20.0, "sparse": {"inliers": 60}},
        ],
        ["q_good.png", "q_hard.png"],
    )

    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        protected_te_cm=15.0,
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
    )

    assert profile["split_name"] == "train_dev"
    assert profile["schema"] == "loc_gs_sparse_pnp_validation_profile_v2"
    assert profile["attribution_status"] == "metric_only"
    assert profile["metrics"]["paired_query_count"] == 2
    assert profile["metrics"]["regression_20cm_count"] == 1
    assert profile["metrics"]["improvement_20cm_count"] == 1
    assert profile["protected_regression_query_ids"] == ["q_good.png"]
    assert profile["hard_improved_query_ids"] == ["q_hard.png"]


def test_feedback_bank_v3_audit_rejects_test_split(tmp_path):
    bank = tmp_path / "bank.jsonl"
    save_feedback_bank(
        bank,
        [_v3_record()],
        {
            "schema_version": "feedback_bank_v3",
            "scene": "ShopFacade",
            "split_name": "test",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    audit = audit_feedback_bank_v3(bank)

    assert audit["audit_status"] == "failed"
    assert any("test split" in reason for reason in audit["reasons"])


def test_sparse_pnp_validation_profile_v2_uses_v3_attribution_and_outlier_risk(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        [
            {"sparse_AE": 0.1, "sparse_TE": 4.0},
            {"sparse_AE": 1.0, "sparse_TE": 80.0},
        ],
        ["q_good.png", "q_hard.png"],
    )
    _write_run(
        candidate,
        [
            {"sparse_AE": 5.0, "sparse_TE": 90.0},
            {"sparse_AE": 0.1, "sparse_TE": 10.0},
        ],
        ["q_good.png", "q_hard.png"],
    )
    baseline_bank = tmp_path / "baseline_v3.jsonl"
    candidate_bank = tmp_path / "candidate_v3.jsonl"
    save_feedback_bank(
        baseline_bank,
        [
            _v3_record(
                source_role="baseline_trace",
                query_id="q_good.png",
                image_id="q_good.png",
                matched_gaussian_id="7",
                matched_landmark_id="sampled:7",
                pnp_inlier=True,
            )
        ],
        {
            "schema_version": "feedback_bank_v3",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )
    save_feedback_bank(
        candidate_bank,
        [
            _v3_record(
                source_role="candidate_trace",
                query_id="q_good.png",
                image_id="q_good.png",
                matched_gaussian_id="11",
                matched_landmark_id="sampled:11",
                pnp_inlier=False,
                descriptor_score=0.95,
                detector_score=0.9,
                descriptor_margin=0.1,
                reprojection_error_px=12.0,
            ),
            _v3_record(
                source_role="candidate_trace",
                query_id="q_hard.png",
                image_id="q_hard.png",
                matched_gaussian_id="17",
                matched_landmark_id="sampled:17",
                pnp_inlier=True,
                reprojection_error_px=1.0,
            ),
        ],
        {
            "schema_version": "feedback_bank_v3",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        baseline_feedback_bank=baseline_bank,
        candidate_feedback_bank=candidate_bank,
        protected_te_cm=15.0,
        hard_te_cm=20.0,
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
    )

    assert profile["schema"] == "loc_gs_sparse_pnp_validation_profile_v2"
    assert profile["attribution_status"] == "attributed"
    assert profile["landmark_regression_risk"]["11"] > 0.0
    assert profile["protected_per_query_support"]["q_good.png"]["7"] > 0.0
    assert profile["validated_per_query_support"]["q_hard.png"]["17"] > 0.0
    observation = profile["validated_per_query_observations"]["q_hard.png"]["17"]
    assert observation["image_cell"] == [1, 2]
    assert observation["depth_bin"] == 3


def test_sparse_pnp_validation_profile_extracts_candidate_validated_hard_improvement_support(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, [{"sparse_AE": 1.0, "sparse_TE": 80.0}], ["q_hard.png"])
    _write_run(candidate, [{"sparse_AE": 0.1, "sparse_TE": 10.0}], ["q_hard.png"])
    bank = tmp_path / "candidate_feedback.jsonl"
    save_feedback_bank(
        bank,
        [
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_hard.png",
                image_id="q_hard.png",
                keypoint_id="kp:1",
                matched_landmark_id="sampled:17",
                matched_gaussian_id="17",
                descriptor_score=0.8,
                pnp_inlier=True,
                reprojection_error_px=1.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            ),
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_hard.png",
                image_id="q_hard.png",
                keypoint_id="kp:2",
                matched_landmark_id="sampled:18",
                matched_gaussian_id="18",
                descriptor_score=0.8,
                pnp_inlier=False,
                reprojection_error_px=1.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            ),
        ],
        {
            "schema_version": "feedback_bank_v2",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        candidate_feedback_bank=bank,
        hard_te_cm=20.0,
        improvement_margin_cm=20.0,
    )

    assert profile["hard_improved_query_ids"] == ["q_hard.png"]
    assert profile["validated_per_query_support"] == {"q_hard.png": {"17": pytest.approx(52.5)}}
    assert profile["metrics"]["validated_query_count"] == 1
    assert profile["metrics"]["validated_landmark_support_count"] == 1


def test_sparse_pnp_validation_profile_extracts_baseline_protected_support(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, [{"sparse_AE": 0.1, "sparse_TE": 3.0}], ["q_good.png"])
    _write_run(candidate, [{"sparse_AE": 5.0, "sparse_TE": 80.0}], ["q_good.png"])
    bank = tmp_path / "baseline_feedback.jsonl"
    save_feedback_bank(
        bank,
        [
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_good.png",
                image_id="q_good.png",
                keypoint_id="kp:1",
                matched_landmark_id="sampled:7",
                matched_gaussian_id="7",
                descriptor_score=0.8,
                pnp_inlier=True,
                reprojection_error_px=1.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            ),
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_good.png",
                image_id="q_good.png",
                keypoint_id="kp:2",
                matched_landmark_id="sampled:8",
                matched_gaussian_id="8",
                descriptor_score=0.8,
                pnp_inlier=False,
                reprojection_error_px=20.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            ),
        ],
        {
            "schema_version": "feedback_bank_v2",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        baseline_feedback_bank=bank,
        protected_te_cm=15.0,
        regression_margin_cm=20.0,
    )

    assert profile["protected_per_query_support"] == {"q_good.png": {"7": pytest.approx(57.75)}}
    assert profile["metrics"]["protected_landmark_support_count"] == 1


def test_sparse_pnp_validation_profile_preserves_baseline_protected_observations(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, [{"sparse_AE": 0.1, "sparse_TE": 3.0}], ["q_good.png"])
    _write_run(candidate, [{"sparse_AE": 5.0, "sparse_TE": 80.0}], ["q_good.png"])
    bank = tmp_path / "baseline_feedback.jsonl"
    save_feedback_bank(
        bank,
        [
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_good.png",
                image_id="q_good.png",
                keypoint_id="kp:1",
                matched_landmark_id="sampled:7",
                matched_gaussian_id="7",
                descriptor_score=0.8,
                detector_score=0.7,
                pnp_inlier=True,
                reprojection_error_px=1.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
                query_xy_norm=(0.25, 0.5),
                bearing=(0.0, 0.1, 1.0),
                camera_xyz=(1.0, 2.0, 6.0),
                depth_m=6.0,
                descriptor_margin=0.33,
                local_geometry_score=0.9,
            )
        ],
        {
            "schema_version": "feedback_bank_v2",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        baseline_feedback_bank=bank,
        protected_te_cm=15.0,
        regression_margin_cm=20.0,
    )

    observation = profile["protected_per_query_observations"]["q_good.png"]["7"]
    assert observation["xy_norm"] == pytest.approx([0.25, 0.5])
    assert observation["bearing"] == pytest.approx([0.0, 0.1, 1.0])
    assert observation["camera_xyz"] == pytest.approx([1.0, 2.0, 6.0])
    assert observation["depth_m"] == pytest.approx(6.0)
    assert observation["descriptor_margin"] == pytest.approx(0.33)
    assert observation["local_geometry_score"] == pytest.approx(0.9)
    assert profile["metrics"]["protected_observation_entry_count"] == 1


def test_sparse_pnp_validation_profile_can_protect_all_baseline_good_support(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, [{"sparse_AE": 0.1, "sparse_TE": 3.0}], ["q_good.png"])
    _write_run(candidate, [{"sparse_AE": 0.1, "sparse_TE": 4.0}], ["q_good.png"])
    bank = tmp_path / "baseline_feedback.jsonl"
    save_feedback_bank(
        bank,
        [
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_good.png",
                image_id="q_good.png",
                keypoint_id="kp:1",
                matched_landmark_id="sampled:7",
                matched_gaussian_id="7",
                descriptor_score=0.8,
                pnp_inlier=True,
                reprojection_error_px=1.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            )
        ],
        {
            "schema_version": "feedback_bank_v2",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        baseline_feedback_bank=bank,
        protected_te_cm=15.0,
        regression_margin_cm=20.0,
        protect_all_baseline_good_queries=True,
    )

    assert profile["protected_regression_query_ids"] == []
    assert profile["protected_per_query_support"] == {"q_good.png": {"7": pytest.approx(9.0)}}
    assert profile["metrics"]["protected_query_count"] == 1


def test_sparse_pnp_validation_profile_extracts_candidate_regression_basin_support(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, [{"sparse_AE": 0.1, "sparse_TE": 3.0}], ["q_good.png"])
    _write_run(candidate, [{"sparse_AE": 5.0, "sparse_TE": 80.0}], ["q_good.png"])
    bank = tmp_path / "candidate_feedback.jsonl"
    save_feedback_bank(
        bank,
        [
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_good.png",
                image_id="q_good.png",
                keypoint_id="kp:1",
                matched_landmark_id="sampled:7",
                matched_gaussian_id="7",
                descriptor_score=0.8,
                pnp_inlier=True,
                reprojection_error_px=1.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            ),
            FeedbackMatchRecord(
                scene="ShopFacade",
                query_id="q_good.png",
                image_id="q_good.png",
                keypoint_id="kp:2",
                matched_landmark_id="sampled:8",
                matched_gaussian_id="8",
                descriptor_score=0.7,
                pnp_inlier=True,
                reprojection_error_px=2.0,
                pnp_success=True,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
            ),
        ],
        {
            "schema_version": "feedback_bank_v2",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        candidate_feedback_bank=bank,
        protected_te_cm=15.0,
        regression_margin_cm=20.0,
    )

    assert profile["regression_per_query_risk_support"]["q_good.png"]["7"] == pytest.approx(57.75)
    assert profile["regression_per_query_risk_support"]["q_good.png"]["8"] == pytest.approx(47.1625)
    assert profile["metrics"]["regression_basin_landmark_support_count"] == 2


def test_sparse_pnp_validation_rejects_test_split(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, [{"sparse_AE": 0.1, "sparse_TE": 1.0}], ["q.png"])
    _write_run(candidate, [{"sparse_AE": 0.1, "sparse_TE": 1.0}], ["q.png"])

    with pytest.raises(ValueError, match="test split"):
        build_sparse_pnp_validation_profile(
            baseline_run_dir=baseline,
            candidate_run_dir=candidate,
            scene="ShopFacade",
            split_name="test",
        )


def test_sparse_validation_landmark_risk_tensor_uses_profile_entries():
    profile = {
        "landmark_regression_risk": {"2": 3.0, "5": 1.5, "99": 4.0, "bad": 1.0},
    }

    risk = sparse_validation_landmark_risk_tensor(profile, num_gaussians=8)

    assert risk.tolist() == [0.0, 0.0, 3.0, 0.0, 0.0, 1.5, 0.0, 0.0]


def test_build_sparse_pnp_validation_profile_argparser_accepts_sparse_only_inputs():
    args = build_argparser().parse_args(
        [
            "--baseline_run_dir",
            "/runs/base",
            "--candidate_run_dir",
            "/runs/cand",
            "--output_dir",
            "/runs/profile",
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_dev",
            "--candidate_feedback_bank",
            "/runs/cand/feedback_bank.jsonl",
            "--baseline_feedback_bank",
            "/runs/base/feedback_bank.jsonl",
            "--regression_margin_cm",
            "10",
            "--protect_all_baseline_good_queries",
        ]
    )

    assert args.scene == "ShopFacade"
    assert args.split_name == "train_dev"
    assert str(args.baseline_feedback_bank) == "/runs/base/feedback_bank.jsonl"
    assert args.regression_margin_cm == 10.0
    assert args.protect_all_baseline_good_queries is True


def test_build_sparse_pnp_validation_profile_cli_preserves_upstream_feedback_split_audit(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, [{"sparse_AE": 0.1, "sparse_TE": 3.0}], ["q_good.png"])
    _write_run(candidate, [{"sparse_AE": 0.1, "sparse_TE": 4.0}], ["q_good.png"])
    bank = tmp_path / "feedback_v3.jsonl"
    save_feedback_bank(
        bank,
        [_v3_record(source_role="baseline_trace", query_id="q_good.png", image_id="q_good.png")],
        {
            "schema_version": "feedback_bank_v3",
            "scene": "ShopFacade",
            "split_name": "train_dev",
            "query_id_source": "real_image_id",
            "split_audit": {"audit_status": "passed", "split_name": "train_dev"},
        },
    )
    output = tmp_path / "profile"

    rc = build_profile_main(
        [
            "--baseline_run_dir",
            str(baseline),
            "--candidate_run_dir",
            str(candidate),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_dev",
            "--baseline_feedback_bank",
            str(bank),
            "--protect_all_baseline_good_queries",
        ]
    )

    assert rc == 0
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "passed"
    assert split_audit["checks"]["upstream_split_audit"]["status"] == "passed"


def test_export_ulfloc_sparse_feedback_argparser_accepts_camera_split():
    args = build_sparse_feedback_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--model_path",
            "/models/map",
            "--config",
            "/cfg.yml",
            "--output_dir",
            "/out",
            "--split_name",
            "train_dev_validation",
            "--camera_split",
            "test",
        ]
    )

    assert args.camera_split == "test"
