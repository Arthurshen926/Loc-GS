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
        localization_batch_size=4,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cmd[:3] == [sys.executable, "-m", "loc_gs.scripts.train_cambridge_hybrid"]
    assert cmd[cmd.index("--batch_size") + 1] == "6"
    assert cmd[cmd.index("--pnp_weight") + 1] == "0.05"
    assert cmd[cmd.index("--pnp_pose_loss_weight") + 1] == "0.0"
    assert cmd[cmd.index("--localization_batch_size") + 1] == "4"
    assert cmd[cmd.index("--pnp_locability_loss_weight") + 1] == "0.1"
    assert cmd[cmd.index("--pnp_locability_target_prior_weight") + 1] == "0.5"
    assert cmd[cmd.index("--localization_descriptor_source") + 1] == "hybrid_ply_gated_residual"
    assert cmd[cmd.index("--hybrid_residual_alpha_max") + 1] == "0.03"
    assert cmd[cmd.index("--pnp_feedback_detector_weight") + 1] == "0.05"
    assert "--pnp_feedback_detector_init_from_stdloc" in cmd
    assert cmd[cmd.index("--pnp_feedback_detector_anchor_weight") + 1] == "0.1"
    assert cmd[cmd.index("--pnp_feedback_detector_prior_weight") + 1] == "0.25"
    assert "--pnp_feedback_detector_full_res" in cmd
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


def test_build_reliability_eval_command_can_use_oracle_prosac_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=2,
        scene="KingsCollege",
        checkpoint="output/stdloc_hybrid/KingsCollege_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/KingsCollege_reliability_recipe/eval_oracle_prosac",
        recipe="oracle_prosac",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cmd[cmd.index("--descriptor_source") + 1] == "ply_loc"
    assert cmd[cmd.index("--query_detector") + 1] == "stdloc"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert cmd[cmd.index("--oracle_match_order") + 1] == "sparse_reprojection"
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability_prior"


def test_build_reliability_eval_command_can_use_topk_candidate_oracle_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=0,
        scene="KingsCollege",
        checkpoint="output/stdloc_hybrid/KingsCollege_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/KingsCollege_reliability_recipe/eval_candidate_oracle_top8",
        recipe="candidate_oracle_top8",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert cmd[cmd.index("--descriptor_source") + 1] == "ply_loc"
    assert cmd[cmd.index("--query_detector") + 1] == "stdloc"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert cmd[cmd.index("--sparse_candidate_topk") + 1] == "8"
    assert cmd[cmd.index("--sparse_candidate_oracle_topk") + 1] == "8"
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability_prior"


def test_build_reliability_eval_command_can_use_scene_matcher_prosac_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=0,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_scene_matcher",
        recipe="scene_matcher_prosac",
        scene_matcher_path="output/scenematch/GreatCourt/best.pt",
        scene_matcher_weight=0.45,
        scene_matcher_topk=6,
        scene_matcher_logit_norm="zscore",
        scene_matcher_logit_clip=1.25,
        scene_matcher_listwise_dustbin="drop",
        scene_matcher_candidate_mode="all",
        match_filter_query_score_weight=0.2,
        dense_iters=3,
        sparse_match_filter_mode="score",
        sparse_match_filter_top_m=1536,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert cmd[cmd.index("--descriptor_source") + 1] == "ply_loc"
    assert cmd[cmd.index("--scene_matcher_path") + 1] == "output/scenematch/GreatCourt/best.pt"
    assert cmd[cmd.index("--scene_matcher_weight") + 1] == "0.45"
    assert cmd[cmd.index("--scene_matcher_topk") + 1] == "6"
    assert cmd[cmd.index("--scene_matcher_logit_norm") + 1] == "zscore"
    assert cmd[cmd.index("--scene_matcher_logit_clip") + 1] == "1.25"
    assert cmd[cmd.index("--scene_matcher_listwise_dustbin") + 1] == "drop"
    assert cmd[cmd.index("--scene_matcher_candidate_mode") + 1] == "all"
    assert cmd[cmd.index("--match_filter_query_score_weight") + 1] == "0.2"
    assert cmd[cmd.index("--dense_iters") + 1] == "3"
    assert cmd[cmd.index("--sparse_match_filter_mode") + 1] == "score"
    assert cmd[cmd.index("--sparse_match_filter_top_m") + 1] == "1536"
    assert "--max_landmarks" not in cmd


def test_build_reliability_eval_command_can_use_scene_matcher_residual_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=2,
        scene="OldHospital",
        checkpoint="output/stdloc_hybrid/OldHospital_lff/latest.pth",
        output_dir="output/stdloc_hybrid/OldHospital_lff/eval_scene_matcher_residual",
        recipe="scene_matcher_residual_prosac",
        scene_matcher_path="output/scenematch/OldHospital/best.pt",
        scene_matcher_weight=0.1,
        scene_matcher_topk=4,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cmd[cmd.index("--query_detector") + 1] == "stdloc"
    assert cmd[cmd.index("--descriptor_source") + 1] == "hybrid_ply_gated_residual"
    assert cmd[cmd.index("--hybrid_residual_alpha_max") + 1] == "0.03"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert cmd[cmd.index("--scene_matcher_path") + 1] == "output/scenematch/OldHospital/best.pt"
    assert cmd[cmd.index("--scene_matcher_weight") + 1] == "0.1"
    assert cmd[cmd.index("--scene_matcher_topk") + 1] == "4"


def test_build_reliability_eval_command_can_attach_selfmap_reliability_summary():
    cmd, env = build_reliability_eval_command(
        gpu_id=1,
        scene="ShopFacade",
        checkpoint="output/stdloc_hybrid/ShopFacade_lff/latest.pth",
        output_dir="output/stdloc_hybrid/ShopFacade_lff/eval_unified_soft",
        recipe="scene_matcher_residual_prosac",
        scene_matcher_path="output/scenematch/ShopFacade/best.pt",
        selfmap_reliability_path="output/selfmap/ShopFacade/summary.json",
        selfmap_reliability_center_cm=9.5,
        selfmap_reliability_temperature_cm=1.5,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cmd[cmd.index("--selfmap_reliability_path") + 1] == "output/selfmap/ShopFacade/summary.json"
    assert cmd[cmd.index("--selfmap_reliability_center_cm") + 1] == "9.5"
    assert cmd[cmd.index("--selfmap_reliability_temperature_cm") + 1] == "1.5"


def test_build_reliability_eval_command_can_use_lff_feedback_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=0,
        scene="ShopFacade",
        checkpoint="output/stdloc_hybrid/ShopFacade_lff/latest.pth",
        output_dir="output/stdloc_hybrid/ShopFacade_lff/eval_lff_feedback",
        recipe="lff_feedback_prosac",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert cmd[cmd.index("--query_detector") + 1] == "feedback"
    assert "--feedback_detector_full_res" in cmd
    assert cmd[cmd.index("--descriptor_source") + 1] == "hybrid_ply_gated_residual"
    assert cmd[cmd.index("--hybrid_residual_alpha_max") + 1] == "0.03"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"


def test_build_reliability_eval_command_can_use_lff_residual_with_stdloc_detector():
    cmd, env = build_reliability_eval_command(
        gpu_id=1,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_lff/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_lff/eval_lff_residual",
        recipe="lff_residual_prosac",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cmd[cmd.index("--query_detector") + 1] == "stdloc"
    assert cmd[cmd.index("--descriptor_source") + 1] == "hybrid_ply_gated_residual"
    assert cmd[cmd.index("--hybrid_residual_alpha_max") + 1] == "0.03"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"


def test_build_reliability_eval_command_can_override_lff_residual_alpha():
    cmd, _env = build_reliability_eval_command(
        gpu_id=1,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_lff/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_lff/eval_lff_residual_alpha005",
        recipe="lff_residual_prosac",
        hybrid_residual_alpha_max=0.05,
    )

    assert cmd[cmd.index("--descriptor_source") + 1] == "hybrid_ply_gated_residual"
    assert cmd[cmd.index("--hybrid_residual_alpha_max") + 1] == "0.05"


def test_build_reliability_eval_command_can_use_scene_matcher_coverage_recipe():
    cmd, _env = build_reliability_eval_command(
        gpu_id=0,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_scene_matcher_coverage",
        recipe="scene_matcher_coverage_prosac",
        scene_matcher_path="output/scenematch/GreatCourt/best.pt",
    )

    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert cmd[cmd.index("--sparse_match_filter_mode") + 1] == "calibrated_coverage"
    assert cmd[cmd.index("--sparse_match_filter_top_m") + 1] == "2048"
    assert cmd[cmd.index("--dense_match_filter_mode") + 1] == "calibrated_coverage"
    assert cmd[cmd.index("--dense_match_filter_top_m") + 1] == "2048"
    assert cmd[cmd.index("--match_filter_min_matches") + 1] == "1024"
    assert cmd[cmd.index("--scene_matcher_path") + 1] == "output/scenematch/GreatCourt/best.pt"


def test_build_reliability_eval_command_can_use_scene_matcher_magsac_recipe():
    cmd, _env = build_reliability_eval_command(
        gpu_id=0,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_scene_matcher_magsac",
        recipe="scene_matcher_prosac_magsac",
        scene_matcher_path="output/scenematch/GreatCourt/best.pt",
    )

    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac_magsac"
    assert cmd[cmd.index("--scene_matcher_path") + 1] == "output/scenematch/GreatCourt/best.pt"


def test_build_reliability_eval_command_can_attach_calibrated_matchability():
    cmd, env = build_reliability_eval_command(
        gpu_id=1,
        scene="OldHospital",
        checkpoint="output/stdloc_hybrid/OldHospital_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/OldHospital_reliability_recipe/eval_reliability_covisibility_prosac",
        recipe="covisibility_prosac",
        calibrated_matchability_path="output/calib/OldHospital/stdloc_bank_query_like.pt",
        calibrated_matchability_weight=0.4,
        match_calibrated_prior_weight=0.15,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cmd[cmd.index("--calibrated_matchability_path") + 1] == "output/calib/OldHospital/stdloc_bank_query_like.pt"
    assert cmd[cmd.index("--landmark_score_calibrated_matchability_weight") + 1] == "0.4"
    assert cmd[cmd.index("--match_calibrated_prior_weight") + 1] == "0.15"


def test_build_reliability_eval_command_can_attach_unified_ply_loc_override():
    cmd, _env = build_reliability_eval_command(
        gpu_id=0,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_unified",
        recipe="scene_matcher_prosac",
        scene_matcher_path="output/scenematch/GreatCourt/best.pt",
        ply_loc_feature_override="output/unified_lff_v2/maps_desc_only_20260515/GreatCourt/point_cloud/iteration_30000/point_cloud.ply",
    )

    assert cmd[cmd.index("--ply_loc_feature_override") + 1].endswith(
        "maps_desc_only_20260515/GreatCourt/point_cloud/iteration_30000/point_cloud.ply"
    )


def test_build_reliability_eval_command_can_attach_unified_detector_dir():
    cmd, _env = build_reliability_eval_command(
        gpu_id=0,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_unified_gate",
        recipe="covisibility_prosac",
        ply_loc_feature_override=(
            "output/unified_lff_v2/maps_gate010/GreatCourt/point_cloud/iteration_30000/point_cloud.ply"
        ),
        stdloc_detector_dir="output/unified_lff_v2/maps_gate010/GreatCourt/detector",
    )

    assert cmd[cmd.index("--ply_loc_feature_override") + 1].endswith(
        "maps_gate010/GreatCourt/point_cloud/iteration_30000/point_cloud.ply"
    )
    assert cmd[cmd.index("--stdloc_detector_dir") + 1].endswith("maps_gate010/GreatCourt/detector")


def test_build_reliability_eval_command_can_override_match_prior_weight():
    cmd, _env = build_reliability_eval_command(
        gpu_id=0,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_unified_gate_prior010",
        recipe="covisibility_prosac",
        locability_prior_weight=0.10,
    )

    assert cmd[cmd.index("--locability_prior_weight") + 1] == "0.1"


def test_build_reliability_eval_command_can_use_covisibility_prosac_magsac_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=0,
        scene="GreatCourt",
        checkpoint="output/stdloc_hybrid/GreatCourt_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/GreatCourt_reliability_recipe/eval_reliability_covisibility_prosac_magsac",
        recipe="covisibility_prosac_magsac",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac_magsac"
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability_prior"
    assert cmd[cmd.index("--match_filter_calibrated_score_weight") + 1] == "0.25"
    assert cmd[cmd.index("--match_filter_margin_weight") + 1] == "0.25"
    assert "--dense_matcher" not in cmd


def test_build_reliability_eval_command_can_use_covisibility_prosac_loftr_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=2,
        scene="ShopFacade",
        checkpoint="output/stdloc_hybrid/ShopFacade_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/ShopFacade_reliability_recipe/eval_reliability_covisibility_prosac_loftr",
        recipe="covisibility_prosac_loftr",
    )

    dense_iters = [idx for idx, item in enumerate(cmd) if item == "--dense_iters"]

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability_prior"
    assert cmd[cmd.index("--dense_matcher") + 1] == "loftr_rendered"
    assert cmd[cmd.index("--dim_pipeline") + 1] == "loftr"
    assert cmd[dense_iters[-1] + 1] == "1"
    assert cmd[cmd.index("--loftr_image_scale") + 1] == "1.0"
    assert cmd[cmd.index("--loftr_min_confidence") + 1] == "0.0"
    assert cmd[cmd.index("--loftr_max_matches") + 1] == "4096"
    assert "--dense_match_filter_mode" not in cmd
    assert "--dense_match_filter_top_m" not in cmd


def test_build_reliability_eval_command_can_use_lff_residual_prosac_loftr_recipe():
    cmd, env = build_reliability_eval_command(
        gpu_id=1,
        scene="ShopFacade",
        checkpoint="output/stdloc_hybrid/ShopFacade_reliability_recipe/latest.pth",
        output_dir="output/stdloc_hybrid/ShopFacade_reliability_recipe/eval_lff_residual_prosac_loftr",
        recipe="lff_residual_prosac_loftr",
    )

    dense_iters = [idx for idx, item in enumerate(cmd) if item == "--dense_iters"]

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cmd[cmd.index("--descriptor_source") + 1] == "hybrid_ply_gated_residual"
    assert cmd[cmd.index("--hybrid_residual_alpha_max") + 1] == "0.03"
    assert cmd[cmd.index("--solver") + 1] == "opencv_prosac"
    assert cmd[cmd.index("--landmark_score_mode") + 1] == "matchability_prior"
    assert cmd[cmd.index("--dense_matcher") + 1] == "loftr_rendered"
    assert cmd[cmd.index("--dim_pipeline") + 1] == "loftr"
    assert cmd[dense_iters[-1] + 1] == "1"
    assert cmd[cmd.index("--loftr_max_matches") + 1] == "4096"
