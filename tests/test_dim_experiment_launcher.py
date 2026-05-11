import sys

from loc_gs.scripts.launch_dim_matcher_experiments import build_eval_command, select_idle_gpus
from loc_gs.scripts.launch_cambridge_reliability_recipe import build_train_command
from loc_gs.scripts.launch_cambridge_reliability_eval import build_eval_command as build_reliability_eval_command


def test_select_idle_gpus_filters_memory_and_utilization():
    rows = [
        {"index": 0, "memory_used": 100, "utilization": 0},
        {"index": 1, "memory_used": 12000, "utilization": 0},
        {"index": 2, "memory_used": 100, "utilization": 80},
    ]

    assert select_idle_gpus(rows, max_memory_used_mb=1000, max_utilization=10) == [0]


def test_build_eval_command_sets_gpu_and_lightglue_options():
    cmd, env = build_eval_command(
        gpu_id=3,
        checkpoint="output/stdloc_hybrid/ShopFacade_full_sota/origteacher_e2_nocache/latest.pth",
        output_dir="output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q5",
        scene="ShopFacade",
        sparse_matcher="lightglue",
        dense_matcher="lightglue_rendered",
        dim_pipeline="superpoint+lightglue",
        max_queries=5,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert cmd[:3] == ["python", "-m", "loc_gs.scripts.eval_cambridge_hybrid"]
    assert "--sparse_matcher" in cmd
    assert "lightglue" in cmd
    assert "--dense_matcher" in cmd
    assert "lightglue_rendered" in cmd
    assert "--dim_pipeline" in cmd
    assert "superpoint+lightglue" in cmd


def test_build_train_command_uses_large_batch_and_conservative_pnp_recipe():
    cmd, env = build_train_command(
        gpu_id=2,
        scene="ShopFacade",
        output_dir="output/stdloc_hybrid/ShopFacade_reliability_recipe",
        epochs=1,
        batch_size=6,
        num_workers=4,
        max_frames=8,
        max_train_batches=1,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cmd[:3] == [sys.executable, "-m", "loc_gs.scripts.train_cambridge_hybrid"]
    assert cmd[cmd.index("--batch_size") + 1] == "6"
    assert cmd[cmd.index("--pnp_weight") + 1] == "0.05"
    assert cmd[cmd.index("--pnp_pose_loss_weight") + 1] == "0.0"
    assert cmd[cmd.index("--pnp_locability_loss_weight") + 1] == "0.1"
    assert cmd[cmd.index("--pnp_locability_target_prior_weight") + 1] == "0.5"
    assert cmd[cmd.index("--same_view_match_weight") + 1] == "1.0"
    assert cmd[cmd.index("--locability_prior_target_weight") + 1] == "0.02"
    assert cmd[cmd.index("--rehearsal_pose_mode") + 1] == "mixed"
    assert cmd[cmd.index("--rehearsal_interpolation_min") + 1] == "-0.15"
    assert cmd[cmd.index("--rehearsal_interpolation_max") + 1] == "1.15"
    assert cmd[cmd.index("--pose_noise_trans_m") + 1] == "0.35"
    assert cmd[cmd.index("--pose_noise_rot_deg") + 1] == "20.0"
    assert cmd[cmd.index("--ply_residual_reg_weight") + 1] == "0.02"
    assert "--amp" in cmd


def test_build_reliability_eval_command_uses_fixed_protected_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=1,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_reliability_protected",
        recipe="protected",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cmd[:3] == [sys.executable, "-m", "loc_gs.scripts.eval_cambridge_hybrid"]
    assert cmd[cmd.index("--eval_pose_source") + 1] == "cambridge"
    assert cmd[cmd.index("--descriptor_source") + 1] == "ply_loc"
    assert cmd[cmd.index("--matcher") + 1] == "stdloc_parity"
    assert cmd[cmd.index("--solver") + 1] == "opencv"
    assert "--dense_full_render" in cmd
    assert "--subpixel_refine" in cmd
    assert "--mnn" not in cmd
    assert "--match_filter_mode" not in cmd


def test_build_reliability_eval_command_can_use_global_learned_blend_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=2,
        scene="ShopFacade",
        checkpoint="output/stdloc_hybrid/ShopFacade_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/ShopFacade_reliability_recipe/eval_reliability_learned_blend",
        recipe="learned_blend",
        max_queries=5,
        query_stride=2,
        query_offset=1,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cmd[cmd.index("--descriptor_source") + 1] == "hybrid_ply_blend"
    assert cmd[cmd.index("--ply_loc_feature_weight") + 1] == "0.9"
    assert cmd[cmd.index("--max_queries") + 1] == "5"
    assert cmd[cmd.index("--query_stride") + 1] == "2"
    assert cmd[cmd.index("--query_offset") + 1] == "1"


def test_build_reliability_eval_command_can_use_covisibility_select_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=0,
        scene="KingsCollege",
        checkpoint="output/stdloc_hybrid/KingsCollege_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/KingsCollege_reliability_recipe/eval_reliability_covisibility_select",
        recipe="covisibility_select",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert cmd[cmd.index("--descriptor_source") + 1] == "ply_loc"
    assert cmd[cmd.index("--max_landmarks") + 1] == "12000"
    assert cmd[cmd.index("--landmark_candidate_source") + 1] == "sampled"
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability"
    assert cmd[cmd.index("--landmark_score_detector_weight") + 1] == "1.0"
    assert cmd[cmd.index("--landmark_score_locability_weight") + 1] == "0.0"
    assert cmd[cmd.index("--landmark_score_visibility_weight") + 1] == "0.75"
    assert cmd[cmd.index("--landmark_score_observability_weight") + 1] == "0.5"
    assert cmd[cmd.index("--landmark_score_prior_blend") + 1] == "0.75"
    assert cmd[cmd.index("--landmark_score_legacy_keep_ratio") + 1] == "0.5"
    assert cmd[cmd.index("--landmark_score_spatial_grid_size") + 1] == "8"
    assert "--ply_loc_feature_weight" not in cmd


def test_build_reliability_eval_command_can_use_covisibility_soft_select_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=0,
        scene="OldHospital",
        checkpoint="output/stdloc_hybrid/OldHospital_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/OldHospital_reliability_recipe/eval_reliability_covisibility_soft_select",
        recipe="covisibility_soft_select",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert cmd[cmd.index("--descriptor_source") + 1] == "ply_loc"
    assert cmd[cmd.index("--max_landmarks") + 1] == "14336"
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability"
    assert cmd[cmd.index("--landmark_score_visibility_weight") + 1] == "0.5"
    assert cmd[cmd.index("--landmark_score_observability_weight") + 1] == "0.25"
    assert cmd[cmd.index("--landmark_score_ambiguity_weight") + 1] == "0.35"
    assert cmd[cmd.index("--landmark_score_ambiguity_radius") + 1] == "0.5"
    assert cmd[cmd.index("--landmark_score_prior_blend") + 1] == "0.5"
    assert cmd[cmd.index("--landmark_score_legacy_keep_ratio") + 1] == "0.75"
    assert "--ply_loc_feature_weight" not in cmd


def test_build_reliability_eval_command_can_use_covisibility_prosac_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=1,
        scene="StMarysChurch",
        checkpoint="output/stdloc_hybrid/StMarysChurch_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/StMarysChurch_reliability_recipe/eval_reliability_covisibility_prosac",
        recipe="covisibility_prosac",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cmd[cmd.index("--descriptor_source") + 1] == "ply_loc"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert "--max_landmarks" not in cmd
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability_prior"
    assert cmd[cmd.index("--landmark_score_visibility_weight") + 1] == "0.5"
    assert cmd[cmd.index("--landmark_score_ambiguity_weight") + 1] == "0.35"
    assert cmd[cmd.index("--landmark_score_prior_blend") + 1] == "0.5"
    assert cmd[cmd.index("--match_filter_calibrated_score_weight") + 1] == "0.25"
    assert cmd[cmd.index("--match_filter_margin_weight") + 1] == "0.25"
    assert "--ply_loc_feature_weight" not in cmd
