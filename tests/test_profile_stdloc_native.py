import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import loc_gs.scripts.profile_stdloc_native as profile_stdloc_native
from loc_gs.scripts.profile_stdloc_native import (
    apply_config_overrides,
    build_profile_manifest,
)


def test_build_profile_manifest_includes_required_audit_fields(tmp_path):
    selector_manifest = {
        "checkpoint_path": "output/train/ShopFacade/unified_lff_v2.pt",
        "selector_weight": 1.0,
        "source_score_weight": 1.0,
        "coverage_grid": 8,
    }
    dataset = SimpleNamespace(
        model_path="output/maps/ShopFacade",
        source_path="/mnt/pool/sqy/Cambridge_stdloc/ShopFacade",
        images="processed",
        feature_type="sp",
        gaussian_type="3dgs",
        longest_edge=1600,
        norm_before_render=False,
    )
    args = SimpleNamespace(
        eval_split="test",
        test_stride=1,
        max_test_cameras=20,
        warmup_cameras=2,
        iteration=-1,
        data_device="cpu",
    )

    manifest = build_profile_manifest(
        args=args,
        dataset=dataset,
        config={
            "sparse": {"landmark_path": "detector/sampled_idx.pkl"},
            "dense": {"iters": 1},
        },
        scene_name="ShopFacade",
        method_name="selector8192_solo",
        cfg_path=Path("third_party/stdloc/configs/stdloc_cambridge.yaml"),
        output_path=tmp_path,
        command=["python", "-m", "loc_gs.scripts.profile_stdloc_native"],
        selector_manifest=selector_manifest,
        timestamp="2026-05-17T00:00:00Z",
        git_commit="abc123",
    )

    assert manifest["git_commit"] == "abc123"
    assert manifest["command"] == ["python", "-m", "loc_gs.scripts.profile_stdloc_native"]
    assert manifest["scene"] == "ShopFacade"
    assert manifest["split"] == "test"
    assert manifest["checkpoint_path"] == "output/train/ShopFacade/unified_lff_v2.pt"
    assert manifest["map_path"] == "output/maps/ShopFacade"
    assert manifest["data_roots"] == ["/mnt/pool/sqy/Cambridge_stdloc/ShopFacade"]
    assert manifest["feedback"]["residual"] is False
    assert manifest["feedback"]["selector"] is False
    assert manifest["feedback"]["rho"] is False
    assert manifest["hyperparameters"]["max_test_cameras"] == 20
    assert manifest["hyperparameters"]["warmup_cameras"] == 2
    assert manifest["selector_resampling"]["coverage_grid"] == 8

    json.dumps(manifest)


def test_apply_config_overrides_records_fast_mode_caps():
    config = {
        "sparse": {"max_iterations": 100000, "min_iterations": 1000, "confidence": 0.99999},
        "dense": {"iters": 1, "max_iterations": 1000, "min_iterations": 100},
    }
    args = SimpleNamespace(
        sparse_max_iterations=20000,
        sparse_min_iterations=100,
        sparse_confidence=0.999,
        sparse_reprojection_error=None,
        sparse_landmark_prior_weight=0.1,
        dense_iters=1,
        dense_max_iterations=500,
        dense_min_iterations=50,
        dense_confidence=None,
        dense_reprojection_error=10.0,
        dense_locability_prior_weight=0.2,
    )

    overrides = apply_config_overrides(config, args)

    assert config["sparse"]["max_iterations"] == 20000
    assert config["sparse"]["min_iterations"] == 100
    assert config["sparse"]["confidence"] == 0.999
    assert config["sparse"]["landmark_prior_weight"] == 0.1
    assert config["dense"]["max_iterations"] == 500
    assert config["dense"]["min_iterations"] == 50
    assert config["dense"]["reprojection_error"] == 10.0
    assert config["dense"]["locability_prior_weight"] == 0.2
    assert overrides["sparse"]["max_iterations"] == 20000
    assert overrides["sparse"]["landmark_prior_weight"] == 0.1
    assert overrides["dense"]["reprojection_error"] == 10.0
    assert overrides["dense"]["locability_prior_weight"] == 0.2


def test_compute_pose_reliability_stats_reports_residuals_and_inlier_ratio():
    p3d = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    p2d = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    intrinsic = np.eye(3, dtype=np.float32)
    pose_w2c = np.eye(4, dtype=np.float32)

    stats = profile_stdloc_native.compute_pose_reliability_stats(
        p2d,
        p3d,
        intrinsic,
        pose_w2c,
        np.array([0, 1], dtype=np.int32),
        solver="opencv",
        reprojection_error=8.0,
        confidence=0.999,
        max_iterations=1000,
        min_iterations=50,
    )

    assert stats["solver"] == "opencv"
    assert stats["match_count"] == 3
    assert stats["inlier_count"] == 2
    assert stats["inlier_ratio"] == 2 / 3
    assert stats["success"] is True
    assert stats["all_reprojection_mean_px"] == 1.0
    assert stats["all_reprojection_median_px"] == 1.0
    assert stats["all_reprojection_p90_px"] == 1.8
    assert stats["inlier_reprojection_mean_px"] == 0.5
    assert stats["inlier_reprojection_median_px"] == 0.5
