import pytest
from pathlib import Path

import numpy as np

from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import save_feedback_bank
from loc_gs.scripts import export_native_stdloc_feedback_bank
from loc_gs.stdloc_native.native_feedback import records_from_native_sparse_capture


def test_native_sparse_capture_records_are_feedback_bank_v2_auditable(tmp_path):
    records = records_from_native_sparse_capture(
        scene="ShopFacade",
        image_name="seq2/frame00001.png",
        keypoint_indices=[17, 42],
        keypoint_xy=[(10.5, 20.5), (30.5, 40.5)],
        landmark_ids=[3, 9],
        gaussian_ids=[101, 205],
        descriptor_scores=[0.83, 0.91],
        detector_scores=[0.4, 0.7],
        match_ranks=[2, 1],
        inlier_indices=[1],
        reprojection_errors_px=[12.0, 0.7],
        visibility_scores=[0.1, 0.9],
        sparse_te_cm=3.5,
        sparse_re_deg=0.2,
        dense_te_cm=2.0,
        dense_re_deg=0.1,
        dense_transition="improved",
        dense_delta_te_cm=-1.5,
    )

    bank_path = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank_path,
        records,
        {
            "scene": "ShopFacade",
            "schema_version": "feedback_bank_v2",
            "split_name": "selfmap_train_native_stdloc",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    audit = audit_feedback_bank_v2(bank_path)
    assert audit["audit_status"] == "passed"
    assert audit["record_count"] == 2
    assert audit["query_count"] == 1
    assert records[0].matched_gaussian_id == "101"
    assert records[1].pnp_inlier is True


def test_native_feedback_exporter_rejects_test_split(tmp_path):
    args = export_native_stdloc_feedback_bank.build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--data_root",
            "/data/cambridge",
            "--map_root",
            "/maps/stdloc",
            "--output_dir",
            str(tmp_path / "out"),
            "--split_audit_json",
            str(tmp_path / "split_audit.json"),
            "--eval_split",
            "test",
        ]
    )

    with pytest.raises(ValueError, match="test split"):
        export_native_stdloc_feedback_bank.validate_args(args)


def test_native_feedback_dataset_defaults_keep_stdloc_sh_degree(tmp_path):
    stdloc_module = export_native_stdloc_feedback_bank._prepare_stdloc_imports()
    args = export_native_stdloc_feedback_bank.build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--data_root",
            "/data/cambridge",
            "--map_root",
            "/maps/stdloc",
            "--output_dir",
            str(tmp_path / "out"),
            "--split_audit_json",
            str(tmp_path / "split_audit.json"),
        ]
    )

    dataset = export_native_stdloc_feedback_bank._build_dataset(
        stdloc_module,
        args,
        Path("/data/cambridge/ShopFacade"),
        Path("/maps/stdloc/ShopFacade"),
    )

    assert dataset.sh_degree == 3


def test_native_feedback_resolves_missing_processed_images_to_scene_root(tmp_path):
    scene_root = tmp_path / "Cambridge" / "KingsCollege"
    scene_root.mkdir(parents=True)

    images = export_native_stdloc_feedback_bank._resolve_scene_images_for_export(
        scene_root,
        "processed",
    )

    assert images == "."


def test_feedback_solve_pose_omits_scores_for_vendored_poselib_solver():
    class VendoredModule:
        @staticmethod
        def solve_pose(*args, **kwargs):
            assert "match_scores" not in kwargs
            return np.eye(4, dtype=np.float32), np.array([0, 2], dtype=np.int32)

    pose, inliers = export_native_stdloc_feedback_bank._solve_pose_for_feedback(
        VendoredModule,
        np.zeros((4, 2), dtype=np.float64),
        np.zeros((4, 3), dtype=np.float64),
        np.eye(3, dtype=np.float64),
        "poselib",
        12.0,
        0.999,
        100,
        10,
        match_scores=np.ones(4, dtype=np.float64),
    )

    assert pose.shape == (4, 4)
    assert inliers.tolist() == [0, 2]


def test_feedback_solve_pose_routes_opencv_prosac_to_local_solver(monkeypatch):
    calls = []

    class VendoredModule:
        @staticmethod
        def solve_pose(*args, **kwargs):
            raise AssertionError("vendored solve_pose should not receive opencv_prosac_magsac")

    def fake_opencv_solver(*args, **kwargs):
        calls.append((args, kwargs))
        return np.eye(4, dtype=np.float32), np.array([1, 3], dtype=np.int32)

    monkeypatch.setattr(export_native_stdloc_feedback_bank, "_solve_opencv_prosac_pose", fake_opencv_solver)

    pose, inliers = export_native_stdloc_feedback_bank._solve_pose_for_feedback(
        VendoredModule,
        np.zeros((4, 2), dtype=np.float64),
        np.zeros((4, 3), dtype=np.float64),
        np.eye(3, dtype=np.float64),
        "opencv_prosac_magsac",
        12.0,
        0.999,
        100,
        10,
        match_scores=np.ones(4, dtype=np.float64),
    )

    assert pose.shape == (4, 4)
    assert inliers.tolist() == [1, 3]
    assert calls


def test_native_feedback_cfg_rejects_opencv_prosac_dense_solver(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
sparse:
  solver: opencv_prosac_magsac
dense:
  solver: opencv_prosac_magsac
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dense solver"):
        export_native_stdloc_feedback_bank._validate_cfg_supported_by_feedback_exporter(cfg)
