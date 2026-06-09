import pytest
import re

from loc_gs.scripts.export_ulfloc_sparse_feedback import (
    configure_sparse_feedback_eval_overrides,
    cross_view_matches_from_ulfloc_sparse_details,
    compact_ulfloc_sparse_details,
    load_dataset_image_list,
    records_from_ulfloc_sparse_details,
    resolve_ulfloc_source_path_for_loader,
    resolve_train_images_to_read,
    save_cross_view_matches_jsonl,
    validate_split_name,
)


def test_records_from_ulfloc_sparse_details_preserves_ulfloc_global_gaussian_ids():
    details = {
        "matched_sampled_indices": [0, 2, 1],
        "matched_gaussian_ids": [101, 303, 202],
        "query_keypoint_indices": [7, 8, 9],
        "keypoint_xy": [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        "detector_score": [0.9, 0.8, 0.7],
        "descriptor_score": [0.91, 0.42, 0.11],
        "pnp_inlier_mask": [True, False, True],
        "reprojection_error_px": [0.5, 12.0, 2.0],
        "pnp_success": True,
        "pose_error_t_cm": 4.2,
        "pose_error_r_deg": 0.1,
    }

    records = records_from_ulfloc_sparse_details(
        scene="ShopFacade",
        image_id="shop/query_0001.png",
        details=details,
        split_name="selfmap_train",
    )

    assert len(records) == 3
    assert records[0].query_id == "shop/query_0001.png"
    assert records[0].matched_landmark_id == "sampled:0"
    assert records[0].matched_gaussian_id == "101"
    assert records[0].descriptor_score == pytest.approx(0.91)
    assert records[0].descriptor_margin == pytest.approx(0.0)
    assert records[1].pnp_inlier is False
    assert records[1].dense_transition == "not_run_sparse_feedback"
    assert records[2].keypoint_id == "kp:9"
    assert records[2].pose_source == "ulfloc_sparse_pnp_train_selfmap"


def test_configure_sparse_feedback_eval_overrides_matches_sparse_eval_path(tmp_path):
    config = {"sparse": {"kpts_num": 2048, "nms": 4}, "log_name": "native_log"}
    output_path = configure_sparse_feedback_eval_overrides(
        config,
        model_path=tmp_path / "model",
        input_log_dir=tmp_path / "solver_fused_log",
        scene_detector_checkpoint=tmp_path / "detector.pth",
        scene_detector_blend_alpha=0.25,
        scene_detector_candidate_top_k=4096,
        scene_detector_native_keep_fraction=0.5,
    )

    assert output_path == tmp_path / "solver_fused_log"
    assert config["scene_specific_detector"]["enabled"] is True
    assert config["scene_specific_detector"]["checkpoint"] == str((tmp_path / "detector.pth").resolve())
    assert config["scene_specific_detector"]["candidate_top_k"] == 4096
    assert config["scene_specific_detector"]["blend_alpha"] == pytest.approx(0.25)
    assert config["scene_specific_detector"]["native_keep_fraction"] == pytest.approx(0.5)
    assert config["scene_specific_detector"]["mode"] == "rerank_superpoint"


def test_configure_sparse_feedback_eval_overrides_keeps_native_log_without_detector(tmp_path):
    config = {"log_name": "native_log"}

    output_path = configure_sparse_feedback_eval_overrides(
        config,
        model_path=tmp_path / "model",
        input_log_dir=None,
        scene_detector_checkpoint=None,
        scene_detector_blend_alpha=0.5,
        scene_detector_candidate_top_k=0,
        scene_detector_native_keep_fraction=0.0,
    )

    assert output_path == tmp_path / "model" / "native_log"
    assert config["scene_specific_detector"] == {"enabled": False}


def test_cross_view_matches_from_ulfloc_sparse_details_use_train_selfmap_keypoint_as_source_xy():
    details = {
        "matched_sampled_indices": [0, 2],
        "matched_gaussian_ids": [101, 303],
        "query_keypoint_indices": [7, 8],
        "keypoint_xy": [[10.0, 20.0], [30.0, 40.0]],
        "detector_score": [0.9, 0.8],
        "descriptor_score": [0.91, 0.42],
        "pnp_inlier_mask": [True, False],
        "reprojection_error_px": [0.5, 12.0],
    }

    matches = cross_view_matches_from_ulfloc_sparse_details(
        scene="ShopFacade",
        image_id="seq1/frame001.png",
        details=details,
        split_name="selfmap_train",
    )

    assert len(matches) == 2
    assert matches[0]["query_id"] == "seq1/frame001.png"
    assert matches[0]["source_view_id"] == "seq1/frame001.png"
    assert matches[0]["query_xy"] == [10.0, 20.0]
    assert matches[0]["source_xy"] == [10.0, 20.0]
    assert matches[0]["matched_gaussian_id"] == 101
    assert matches[0]["pnp_inlier"] is True
    assert matches[0]["teacher_source"] == "ulfloc_sampled_landmarks"
    assert matches[0]["feedback_role"] == "bootstrap_sparse_teacher"
    assert matches[0]["paper_safe_role"] == "diagnostic_not_main_full_raw_feedback"
    assert matches[1]["descriptor_score"] == pytest.approx(0.42)


def test_cross_view_matches_from_ulfloc_sparse_details_add_sparse_pnp_geometry_observations():
    details = {
        "matched_sampled_indices": [0],
        "matched_gaussian_ids": [101],
        "query_keypoint_indices": [7],
        "keypoint_xy": [[3.5, 7.5]],
        "query_xy": [[4.0, 8.0]],
        "p3d": [[1.0, 2.0, 3.0]],
        "intrinsic": [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        "image_size": [16, 32],
        "pose_w2c": [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "descriptor_score": [0.91],
        "descriptor_margin": [0.37],
        "pnp_inlier_mask": [True],
        "reprojection_error_px": [0.5],
    }

    matches = cross_view_matches_from_ulfloc_sparse_details(
        scene="ShopFacade",
        image_id="seq1/frame001.png",
        details=details,
        split_name="selfmap_train",
    )

    row = matches[0]
    assert row["query_xy"] == [4.0, 8.0]
    assert row["source_xy"] == [3.5, 7.5]
    assert row["query_xy_norm"] == pytest.approx([0.25, 0.25])
    assert row["bearing"] == pytest.approx([2.0, 2.0, 1.0])
    assert row["camera_xyz"] == pytest.approx([2.0, 2.0, 3.0])
    assert row["descriptor_margin"] == pytest.approx(0.37)


def test_feedback_bank_records_from_ulfloc_sparse_details_add_sparse_pnp_geometry_observations():
    details = {
        "matched_sampled_indices": [0],
        "matched_gaussian_ids": [101],
        "query_keypoint_indices": [7],
        "keypoint_xy": [[3.5, 7.5]],
        "query_xy": [[4.0, 8.0]],
        "p3d": [[1.0, 2.0, 3.0]],
        "intrinsic": [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        "image_size": [16, 32],
        "pose_w2c": [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "descriptor_score": [0.91],
        "descriptor_margin": [0.37],
        "pnp_inlier_mask": [True],
        "reprojection_error_px": [0.5],
    }

    records = records_from_ulfloc_sparse_details(
        scene="ShopFacade",
        image_id="seq1/frame001.png",
        details=details,
        split_name="selfmap_train",
    )

    record = records[0].to_dict()
    assert record["split_name"] == "selfmap_train"
    assert record["source_role"] == "baseline_trace"
    assert record["pose_success"] is False
    assert record["query_sparse_te_cm"] is None
    assert record["query_xy_norm"] == pytest.approx([0.25, 0.25])
    assert record["bearing"] == pytest.approx([2.0, 2.0, 1.0])
    assert record["camera_xyz"] == pytest.approx([2.0, 2.0, 3.0])
    assert record["depth_m"] == pytest.approx(3.0)
    assert record["image_cell"] == [2, 2]
    assert record["depth_bin"] == 1
    assert record["descriptor_margin"] == pytest.approx(0.37)


def test_feedback_bank_records_can_mark_candidate_trace_and_sparse_pose_success():
    details = {
        "matched_sampled_indices": [0],
        "matched_gaussian_ids": [101],
        "query_keypoint_indices": [7],
        "keypoint_xy": [[4.0, 8.0]],
        "query_xy": [[4.0, 8.0]],
        "p3d": [[1.0, 2.0, 8.0]],
        "intrinsic": [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        "image_size": [16, 32],
        "pose_w2c": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "descriptor_score": [0.91],
        "descriptor_margin": [0.37],
        "detector_score": [0.8],
        "pnp_inlier_mask": [True],
        "reprojection_error_px": [0.5],
        "pnp_success": True,
        "pose_error_t_cm": 7.5,
    }

    records = records_from_ulfloc_sparse_details(
        scene="ShopFacade",
        image_id="seq1/frame001.png",
        details=details,
        split_name="selfmap_train",
        source_role="candidate_trace",
    )

    record = records[0].to_dict()
    assert record["split_name"] == "selfmap_train"
    assert record["source_role"] == "candidate_trace"
    assert record["pose_success"] is True
    assert record["query_sparse_te_cm"] == pytest.approx(7.5)
    assert record["image_cell"] == [2, 2]
    assert record["depth_bin"] == 2


def test_feedback_bank_records_mark_negative_camera_depth_as_invalid_depth_bin():
    details = {
        "matched_sampled_indices": [0],
        "matched_gaussian_ids": [101],
        "query_keypoint_indices": [7],
        "keypoint_xy": [[4.0, 8.0]],
        "query_xy": [[4.0, 8.0]],
        "p3d": [[1.0, 2.0, -0.5]],
        "intrinsic": [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        "image_size": [16, 32],
        "pose_w2c": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "descriptor_score": [0.91],
        "pnp_inlier_mask": [False],
        "reprojection_error_px": [100.0],
    }

    records = records_from_ulfloc_sparse_details(
        scene="ShopFacade",
        image_id="seq1/frame001.png",
        details=details,
        split_name="selfmap_train",
    )

    record = records[0].to_dict()
    assert record["depth_m"] == pytest.approx(-0.5)
    assert record["depth_bin"] == -1


def test_compact_ulfloc_sparse_details_keeps_inliers_and_top_non_inlier_matches():
    details = {
        "matched_sampled_indices": [0, 1, 2, 3],
        "matched_gaussian_ids": [100, 101, 102, 103],
        "query_keypoint_indices": [7, 8, 9, 10],
        "keypoint_xy": [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]],
        "detector_score": [0.1, 0.2, 0.3, 0.4],
        "descriptor_score": [0.1, 0.9, 0.2, 0.8],
        "pnp_inlier_mask": [True, False, True, False],
        "reprojection_error_px": [1.0, 20.0, 2.0, 30.0],
        "pose_error_t_cm": 8.0,
    }

    compact = compact_ulfloc_sparse_details(
        details,
        inliers_only=True,
        max_non_inlier_records=1,
    )

    assert compact["matched_gaussian_ids"] == [100, 101, 102]
    assert compact["pnp_inlier_mask"] == [True, False, True]
    assert compact["sparse_total_match_count"] == 4
    assert compact["sparse_retained_match_count"] == 3
    assert compact["pose_error_t_cm"] == 8.0


def test_save_cross_view_matches_jsonl_writes_phase2a_compatible_rows(tmp_path):
    path = tmp_path / "matches.jsonl"
    matches = [
        {
            "query_id": "q1",
            "source_view_id": "q1",
            "query_xy": [1.0, 2.0],
            "source_xy": [1.0, 2.0],
            "descriptor_score": 0.9,
            "pnp_inlier": True,
        }
    ]

    save_cross_view_matches_jsonl(
        path,
        matches,
        metadata={"scene": "ShopFacade", "split_name": "selfmap_train"},
    )

    rows = [__import__("json").loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "manifest"
    assert rows[0]["manifest"]["split_name"] == "selfmap_train"
    assert rows[1]["type"] == "match"
    assert rows[1]["match"]["source_xy"] == [1.0, 2.0]


def test_validate_split_name_rejects_test_feedback():
    with pytest.raises(ValueError, match="test split"):
        validate_split_name("test")


def test_resolve_ulfloc_sparse_source_path_for_loader_adds_lowercase_cambridge_symlink(tmp_path):
    source = tmp_path / "Cambridge_stdloc" / "ShopFacade"
    source.mkdir(parents=True)

    resolved = resolve_ulfloc_source_path_for_loader(source, tmp_path / "links")

    assert "cambridge" in str(resolved)
    assert resolved.exists()
    assert resolved.resolve() == source.resolve()


def test_load_dataset_image_list_reads_cambridge_train_file(tmp_path):
    list_path = tmp_path / "dataset_train.txt"
    list_path.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "\n"
        "# comment\n"
        "seq2/frame00001.png -1 0 0 1 0 0 0\n"
        "seq2/frame00002.png -2 0 0 1 0 0 0\n",
        encoding="utf-8",
    )

    assert load_dataset_image_list(list_path) == ["seq2/frame00001.png", "seq2/frame00002.png"]


def test_resolve_train_images_to_read_ignores_directory_candidate_and_uses_dataset_train(tmp_path):
    source = tmp_path / "scene"
    source.mkdir()
    (source / "dataset_train.txt").write_text("seq2/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")

    images, path = resolve_train_images_to_read(source, tmp_path)

    assert images == ["seq2/frame00001.png"]
    assert path == str(source / "dataset_train.txt")


def test_ulfloc_loc_coarse_exposes_optional_sparse_match_details():
    source = open("/root/ULF-Loc/ulfloc.py", encoding="utf-8").read()

    assert "def loc_coarse(self, camera, query_img, fovx, fovy, return_matches=False)" in source
    assert '"matched_gaussian_ids"' in source
    assert '"pnp_inlier_mask"' in source
    assert '"reprojection_error_px"' in source
    assert "torch.as_tensor(self.sampled_idx).detach().cpu()" in source


def test_ulfloc_feature_fusion_samples_sparse_landmark_features_without_full_image_upsample():
    source = open("/root/ULF-Loc/utils/gsfeature_fusion.py", encoding="utf-8").read()

    assert "F.interpolate(feature_map, size=(image_height, image_width)" not in source
    assert "grid_sample(feature_map" in source


def test_ulfloc_train_respects_external_cuda_visible_devices():
    source = open("/root/ULF-Loc/train.py", encoding="utf-8").read()

    hardcoded_cuda_assignment = re.compile(
        r"os\.environ\s*\[\s*['\"]CUDA_VISIBLE_DEVICES['\"]\s*\]\s*=\s*['\"]0['\"]"
    )
    assert hardcoded_cuda_assignment.search(source) is None


def test_ulfloc_eval_respects_external_cuda_visible_devices():
    source = open("/root/ULF-Loc/ulfloc.py", encoding="utf-8").read()

    hardcoded_cuda_assignment = re.compile(
        r"os\.environ\s*\[\s*['\"]CUDA_VISIBLE_DEVICES['\"]\s*\]\s*=\s*['\"]0['\"]"
    )
    assert hardcoded_cuda_assignment.search(source) is None


def test_export_ulfloc_sparse_feedback_separates_scene_data_and_compute_device():
    from loc_gs.scripts.export_ulfloc_sparse_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--model_path",
            "/models/OldHospital",
            "--config",
            "/cfg.yaml",
            "--output_dir",
            "/out",
            "--data_device",
            "cpu",
            "--compute_device",
            "cuda",
            "--inliers_only_for_bank",
            "--max_non_inlier_records_per_image",
            "8",
        ]
    )

    assert args.data_device == "cpu"
    assert args.compute_device == "cuda"
    assert args.inliers_only_for_bank is True
    assert args.max_non_inlier_records_per_image == 8
