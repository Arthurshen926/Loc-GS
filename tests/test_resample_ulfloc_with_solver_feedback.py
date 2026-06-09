import pickle
from pathlib import Path

import pytest
import torch

from loc_gs.scripts.resample_ulfloc_with_solver_feedback import (
    _aggregate_per_query_support_vector,
    _optional_path_string,
    apply_solver_feedback_alpha_overrides,
    build_spatial_cluster_ids,
    build_kc_solver_selection_scores,
    choose_native_safe_core_and_tail,
    descriptor_conflict_edges_from_sparse_validation_profile,
    merge_per_query_support,
    merge_native_safe_core_sampled_idx,
    per_query_observations_from_sparse_validation_profile,
    per_query_observations_from_ray_feedback,
    per_query_support_from_ray_feedback,
    per_query_support_from_sparse_validation_profile,
    prepare_reused_sampled_idx,
    sparse_validation_hard_reject_exempt_tensor_from_profile,
    select_sparse_solver_aware_kc_sampled_idx,
    merge_solver_coverage_coreset_sampled_idx,
    merge_solver_query_coverage_sampled_idx,
    prepare_ulfloc_solver_feedback_output,
    read_ply_vertex_count,
    resolve_solver_feedback_path_for_resample,
    summarize_solver_feedback_score_components,
    validate_solver_feedback_matches_source_map,
    write_map_resample_audit_bundle,
    write_ulfloc_used_config,
)


def _write_ply(path: Path, vertex_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        + b"\x00" * 12
    )


def _write_feedback(path: Path, count: int, split_name: str = "selfmap_train") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": "ulfloc_solver_feedback_v1",
                "split_name": split_name,
                "landmark_weights": torch.ones(count),
            },
            handle,
        )


def test_read_ply_vertex_count_reads_binary_ply_header(tmp_path: Path):
    ply = tmp_path / "point_cloud.ply"
    _write_ply(ply, 123)

    assert read_ply_vertex_count(ply) == 123


def test_validate_solver_feedback_matches_source_map_rejects_count_mismatch(tmp_path: Path):
    model = tmp_path / "model"
    feedback = tmp_path / "solver_feedback.pkl"
    _write_ply(model / "point_cloud" / "iteration_30000" / "point_cloud.ply", 3)
    _write_feedback(feedback, 4)

    with pytest.raises(ValueError, match="does not match source map vertex count"):
        validate_solver_feedback_matches_source_map(
            feedback_path=feedback,
            source_model_path=model,
            iteration=30000,
        )


def test_prepare_ulfloc_solver_feedback_output_links_native_map_and_copies_feedback(tmp_path: Path):
    source = tmp_path / "native"
    output = tmp_path / "candidate"
    feedback = tmp_path / "support" / "solver_feedback.pkl"
    _write_ply(source / "point_cloud" / "iteration_30000" / "point_cloud.ply", 5)
    (source / "cfg_args").write_text("Namespace(model_path='native')", encoding="utf-8")
    _write_feedback(feedback, 5)

    paths = prepare_ulfloc_solver_feedback_output(
        source_model_path=source,
        output_model_path=output,
        feedback_path=feedback,
        log_name="test",
        iteration=30000,
    )

    assert (output / "point_cloud").exists()
    assert (output / "cfg_args").read_text(encoding="utf-8") == "Namespace(model_path='native')"
    assert (output / "test" / "solver_feedback.pkl").exists()
    assert paths["point_cloud"].exists()
    assert paths["solver_feedback"].exists()


def test_write_ulfloc_used_config_persists_eval_yaml(tmp_path: Path):
    output_dir = tmp_path / "candidate" / "train_dev"
    cfg_path = tmp_path / "source" / "ulfloc_train_dev.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("log_name: train_dev\n", encoding="utf-8")

    written = write_ulfloc_used_config(
        output_eval_dir=output_dir,
        cfg_path=cfg_path,
        config={"log_name": "train_dev", "model_path": "/candidate"},
    )

    assert written == output_dir / "ulfloc_train_dev.yaml"
    assert "model_path: /candidate" in written.read_text(encoding="utf-8")


def test_prepare_ulfloc_solver_feedback_output_allows_native_no_feedback(tmp_path: Path):
    source = tmp_path / "native"
    output = tmp_path / "selfmap_native"
    _write_ply(source / "point_cloud" / "iteration_30000" / "point_cloud.ply", 5)
    (source / "cfg_args").write_text("Namespace(model_path='native')", encoding="utf-8")

    paths = prepare_ulfloc_solver_feedback_output(
        source_model_path=source,
        output_model_path=output,
        feedback_path=None,
        log_name="train_dev_seed13_20p_selfmap_native",
        iteration=30000,
    )

    assert (output / "point_cloud").exists()
    assert (output / "cfg_args").read_text(encoding="utf-8") == "Namespace(model_path='native')"
    assert not (output / "train_dev_seed13_20p_selfmap_native" / "solver_feedback.pkl").exists()
    assert paths["point_cloud"].exists()
    assert "solver_feedback" not in paths


def test_resolve_solver_feedback_path_inherits_enabled_source_eval_artifact(tmp_path: Path):
    source = tmp_path / "source_model"
    feedback = source / "train_dev" / "solver_feedback.pkl"
    _write_ply(source / "point_cloud" / "iteration_30000" / "point_cloud.ply", 5)
    _write_feedback(feedback, 5)

    resolved, role = resolve_solver_feedback_path_for_resample(
        explicit_feedback_path=None,
        source_model_path=source,
        log_name="train_dev",
        config={"solver_feedback": {"enabled": True, "artifact_path": "solver_feedback.pkl"}},
    )

    assert resolved == feedback
    assert role == "inherited_source_eval"


def test_resolve_solver_feedback_path_keeps_disabled_config_as_none(tmp_path: Path):
    resolved, role = resolve_solver_feedback_path_for_resample(
        explicit_feedback_path=None,
        source_model_path=tmp_path / "source_model",
        log_name="train_dev",
        config={"solver_feedback": {"enabled": False, "artifact_path": "solver_feedback.pkl"}},
    )

    assert resolved is None
    assert role == "disabled"


def test_resample_argparser_accepts_native_no_feedback_mode():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/selfmap_native",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.solver_feedback is None


def test_resample_argparser_accepts_solver_query_coverage_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/query_coverage",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--native_sampled_idx",
            "/native/keypoints_sampled_idx.pkl",
            "--solver_admissibility_path",
            "/feedback/solver_constraints.json",
            "--selection_policy",
            "solver_query_coverage",
            "--coverage_saturation_mode",
            "native_percentile",
            "--tail_cvar_alpha",
            "0.1",
            "--require_candidate_gain",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.selection_policy == "solver_query_coverage"
    assert str(args.solver_admissibility_path) == "/feedback/solver_constraints.json"
    assert args.coverage_saturation_mode == "native_percentile"
    assert args.tail_cvar_alpha == 0.1
    assert args.require_candidate_gain is True


def test_resample_argparser_accepts_sparse_solver_aware_kc_policy():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/sparse_solver_aware",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--native_sampled_idx",
            "/models/native/sampled_idx.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "sparse_solver_aware_kc",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.selection_policy == "sparse_solver_aware_kc"
    assert str(args.solver_admissibility_path) == "/feedback/sparse_solver_constraints.json"


def test_resample_argparser_accepts_full_gaussian_sparse_set_without_native_sampled_idx():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_max_landmarks",
            "20000",
            "--full_set_min_marginal_gain",
            "0.1",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.selection_policy == "full_gaussian_sparse_set"
    assert args.native_sampled_idx is None
    assert args.full_set_max_landmarks == 20000
    assert args.full_set_min_marginal_gain == 0.1
    assert args.full_set_max_per_spatial_cluster == 0
    assert args.full_set_kc_anchor_max_per_spatial_cluster == 0


def test_resample_argparser_accepts_profile_backed_full_gaussian_sparse_set_without_legacy_feedback():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set_profile",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_sparse_validation_profile",
            "/feedback/sparse_pnp_validation_profile.json",
            "--full_set_max_landmarks",
            "20000",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.selection_policy == "full_gaussian_sparse_set"
    assert args.solver_feedback is None
    assert args.solver_admissibility_path is None
    assert str(args.full_set_sparse_validation_profile) == "/feedback/sparse_pnp_validation_profile.json"


def test_optional_path_string_keeps_missing_legacy_path_as_none():
    assert _optional_path_string(None) is None
    assert _optional_path_string("/feedback/sparse_pnp_validation_profile.json") == "/feedback/sparse_pnp_validation_profile.json"


def test_sparse_validation_query_support_aggregates_to_solver_support_vector():
    support = {
        "protected:q1": {1: 2.0, 3: 0.5},
        "validated:q2": {1: 4.0, 4: 1.5},
    }

    vector = _aggregate_per_query_support_vector(support, length=6)

    assert vector.tolist() == pytest.approx([0.0, 6.0, 0.0, 0.5, 1.5, 0.0])


def test_resample_argparser_accepts_solver_feedback_alpha_overrides():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/fused",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--cfg",
            "/configs/ulfloc.yaml",
            "--solver_sampling_alpha",
            "0.05",
            "--solver_fusion_alpha",
            "0.1",
        ]
    )

    assert args.solver_sampling_alpha == pytest.approx(0.05)
    assert args.solver_fusion_alpha == pytest.approx(0.1)


def test_resample_argparser_accepts_reuse_sampled_idx():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/fused",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--cfg",
            "/configs/ulfloc.yaml",
            "--reuse_sampled_idx",
            "/models/native/train_dev/keypoints_sampled_idx.pkl",
        ]
    )

    assert str(args.reuse_sampled_idx) == "/models/native/train_dev/keypoints_sampled_idx.pkl"


def test_prepare_reused_sampled_idx_copies_source_to_eval_dir(tmp_path: Path):
    source = tmp_path / "source" / "keypoints_sampled_idx.pkl"
    target_dir = tmp_path / "candidate" / "train_dev"
    source.parent.mkdir(parents=True)
    with source.open("wb") as handle:
        pickle.dump(torch.tensor([7, 3, 11], dtype=torch.long), handle)

    summary = prepare_reused_sampled_idx(
        source_sampled_idx=source,
        output_eval_dir=target_dir,
        landmark_file_name="keypoints_sampled_idx.pkl",
    )

    assert summary["reuse_sampled_idx"] == str(source)
    assert summary["reused_sampled_count"] == 3
    assert (target_dir / "keypoints_sampled_idx.pkl").is_file()
    with (target_dir / "keypoints_sampled_idx.pkl").open("rb") as handle:
        copied = torch.as_tensor(pickle.load(handle), dtype=torch.long)
    assert copied.tolist() == [7, 3, 11]


def test_apply_solver_feedback_alpha_overrides_updates_config_without_losing_existing_fields():
    config = {
        "solver_feedback": {
            "enabled": True,
            "artifact_path": "solver_feedback.pkl",
            "sampling_alpha": 0.05,
            "fusion_alpha": 0.0,
        }
    }

    updated = apply_solver_feedback_alpha_overrides(
        config,
        sampling_alpha=0.07,
        fusion_alpha=0.15,
    )

    assert updated["solver_feedback"]["enabled"] is True
    assert updated["solver_feedback"]["artifact_path"] == "solver_feedback.pkl"
    assert updated["solver_feedback"]["sampling_alpha"] == pytest.approx(0.07)
    assert updated["solver_feedback"]["fusion_alpha"] == pytest.approx(0.15)


def test_full_gaussian_sparse_set_cluster_cap_is_independent_from_insertion_cap():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--max_insertions_per_cluster",
            "2",
            "--full_set_kc_anchor_max_per_spatial_cluster",
            "3",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.max_insertions_per_cluster == 2
    assert args.full_set_max_per_spatial_cluster == 0
    assert args.full_set_kc_anchor_max_per_spatial_cluster == 3


def test_resample_argparser_accepts_full_gaussian_query_prefill_spatial_cap():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_query_prefill_max_per_spatial_cluster",
            "1",
            "--full_set_query_geometry_weight",
            "3.5",
            "--full_set_query_depth_weight",
            "1.25",
            "--full_set_query_bearing_weight",
            "2.5",
            "--full_set_query_pnp_geometry_weight",
            "4.5",
            "--full_set_query_pnp_distance_scale_m",
            "3.0",
            "--full_set_main_min_query_gain",
            "0.75",
            "--full_set_main_dynamic_lookahead",
            "256",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.full_set_query_prefill_max_per_spatial_cluster == 1
    assert args.full_set_query_geometry_weight == 3.5
    assert args.full_set_query_depth_weight == 1.25
    assert args.full_set_query_bearing_weight == 2.5
    assert args.full_set_query_pnp_geometry_weight == 4.5
    assert args.full_set_query_pnp_distance_scale_m == 3.0
    assert args.full_set_main_min_query_gain == 0.75
    assert args.full_set_main_dynamic_lookahead == 256


def test_resample_argparser_accepts_full_gaussian_source_anchor_idx():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_source_anchor_idx",
            "/models/native/keypoints_sampled_idx.pkl",
            "--full_set_source_anchor_mode",
            "soft",
            "--full_set_source_anchor_weight",
            "0.25",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert str(args.full_set_source_anchor_idx) == "/models/native/keypoints_sampled_idx.pkl"
    assert args.full_set_source_anchor_mode == "soft"
    assert args.full_set_source_anchor_weight == 0.25


def test_resample_argparser_accepts_full_gaussian_query_candidate_pool_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_candidate_pool_kc_top_k",
            "11",
            "--full_set_candidate_pool_visibility_top_k",
            "13",
            "--full_set_candidate_pool_mask_top_k",
            "17",
            "--full_set_candidate_pool_solver_top_k",
            "19",
            "--full_set_candidate_pool_match_strength_top_k",
            "23",
            "--full_set_candidate_pool_query_top_k",
            "7",
            "--full_set_candidate_pool_query_cell_top_k",
            "3",
            "--full_set_candidate_pool_query_depth_top_k",
            "2",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.full_set_candidate_pool_kc_top_k == 11
    assert args.full_set_candidate_pool_visibility_top_k == 13
    assert args.full_set_candidate_pool_mask_top_k == 17
    assert args.full_set_candidate_pool_solver_top_k == 19
    assert args.full_set_candidate_pool_match_strength_top_k == 23
    assert args.full_set_candidate_pool_query_top_k == 7
    assert args.full_set_candidate_pool_query_cell_top_k == 3
    assert args.full_set_candidate_pool_query_depth_top_k == 2


def test_resample_argparser_accepts_match_feedback_objective_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_support_match_strength",
            "/feedback/support_match_strength.pkl",
            "--full_set_match_competition_risk",
            "/feedback/stealer_scores.pkl",
            "--full_set_support_match_strength_weight",
            "2.5",
            "--full_set_match_competition_risk_weight",
            "1.75",
            "--full_set_match_competition_risk_reject_threshold",
            "0.2",
            "--full_set_query_match_dominance",
            "/feedback/query_match_dominance.json",
            "--full_set_query_match_strength_weight",
            "3.5",
            "--full_set_query_match_competition_weight",
            "2.25",
            "--full_set_query_match_competition_ratio_weight",
            "4.75",
            "--full_set_candidate_pool_query_match_strength_top_k",
            "29",
            "--full_set_query_match_reference_sampled_idx",
            "/maps/kc_sampled_idx.pkl",
            "--full_set_query_match_risk_reference_margin",
            "0.75",
            "--full_set_query_match_risk_reference_weight",
            "6.5",
            "--full_set_query_match_nonreference_competition_weight",
            "8.25",
            "--full_set_final_prune_query_match_risk_reference",
            "--full_set_final_prune_nonreference_query_match_competition",
            "--full_set_preserve_source_descriptors",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert str(args.full_set_support_match_strength) == "/feedback/support_match_strength.pkl"
    assert str(args.full_set_match_competition_risk) == "/feedback/stealer_scores.pkl"
    assert args.full_set_support_match_strength_weight == 2.5
    assert args.full_set_match_competition_risk_weight == 1.75
    assert args.full_set_match_competition_risk_reject_threshold == 0.2
    assert str(args.full_set_query_match_dominance) == "/feedback/query_match_dominance.json"
    assert args.full_set_query_match_strength_weight == 3.5
    assert args.full_set_query_match_competition_weight == 2.25
    assert args.full_set_query_match_competition_ratio_weight == 4.75
    assert args.full_set_candidate_pool_query_match_strength_top_k == 29
    assert str(args.full_set_query_match_reference_sampled_idx) == "/maps/kc_sampled_idx.pkl"
    assert args.full_set_query_match_risk_reference_margin == 0.75
    assert args.full_set_query_match_risk_reference_weight == 6.5
    assert args.full_set_query_match_nonreference_competition_weight == 8.25
    assert args.full_set_final_prune_query_match_risk_reference is True
    assert args.full_set_final_prune_nonreference_query_match_competition is True
    assert args.full_set_preserve_source_descriptors is True


def test_resample_argparser_accepts_precision_fill_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_precision_fill_target_count",
            "12000",
            "--full_set_precision_fill_min_score",
            "0.25",
            "--full_set_precision_fill_kc_weight",
            "1.5",
            "--full_set_precision_fill_visibility_weight",
            "0.5",
            "--full_set_precision_fill_mask_weight",
            "0.25",
            "--full_set_precision_fill_solver_weight",
            "0.75",
            "--full_set_precision_fill_query_support_weight",
            "0.1",
            "--full_set_precision_fill_support_match_strength_weight",
            "2.0",
            "--full_set_precision_fill_ambiguity_risk_weight",
            "0.3",
            "--full_set_precision_fill_sparse_validation_risk_weight",
            "0.4",
            "--full_set_precision_fill_match_competition_risk_weight",
            "1.25",
            "--full_set_precision_fill_descriptor_conflict_weight",
            "3.0",
            "--full_set_precision_fill_require_query_support",
            "--full_set_precision_fill_max_per_spatial_cluster",
            "2",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.full_set_precision_fill_target_count == 12000
    assert args.full_set_precision_fill_min_score == 0.25
    assert args.full_set_precision_fill_kc_weight == 1.5
    assert args.full_set_precision_fill_visibility_weight == 0.5
    assert args.full_set_precision_fill_mask_weight == 0.25
    assert args.full_set_precision_fill_solver_weight == 0.75
    assert args.full_set_precision_fill_query_support_weight == 0.1
    assert args.full_set_precision_fill_support_match_strength_weight == 2.0
    assert args.full_set_precision_fill_ambiguity_risk_weight == 0.3
    assert args.full_set_precision_fill_sparse_validation_risk_weight == 0.4
    assert args.full_set_precision_fill_match_competition_risk_weight == 1.25
    assert args.full_set_precision_fill_descriptor_conflict_weight == 3.0
    assert args.full_set_precision_fill_require_query_support is True
    assert args.full_set_precision_fill_max_per_spatial_cluster == 2


def test_preserve_source_descriptors_for_shared_landmarks_keeps_existing_features():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import (
        _preserve_source_descriptors_for_shared_landmarks,
    )

    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    source_idx = torch.tensor([20, 40, 10], dtype=torch.long)
    source_features = torch.tensor(
        [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )

    updated, count = _preserve_source_descriptors_for_shared_landmarks(
        sampled_idx,
        features,
        source_idx,
        source_features,
    )

    assert count == 2
    assert torch.allclose(updated[0], torch.tensor([0.9, 0.1]))
    assert torch.allclose(updated[1], torch.tensor([0.2, 0.8]))
    assert torch.allclose(updated[2], torch.tensor([0.5, 0.5]))


def test_resample_argparser_accepts_source_query_prefill_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_source_query_prefill_target_fraction",
            "0.25",
            "--full_set_source_query_prefill_max_candidates_per_query",
            "31",
            "--full_set_source_query_prefill_max_per_spatial_cluster",
            "2",
            "--full_set_source_query_prefill_min_cells_per_query",
            "4",
            "--full_set_source_query_prefill_min_depth_bins_per_query",
            "3",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.full_set_source_query_prefill_target_fraction == 0.25
    assert args.full_set_source_query_prefill_max_candidates_per_query == 31
    assert args.full_set_source_query_prefill_max_per_spatial_cluster == 2
    assert args.full_set_source_query_prefill_min_cells_per_query == 4
    assert args.full_set_source_query_prefill_min_depth_bins_per_query == 3


def test_resample_argparser_accepts_full_gaussian_anchor_and_augmentation_gates():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/sparse_solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_force_source_anchor",
            "--full_set_augmentation_min_solver_support",
            "0.2",
            "--full_set_augmentation_min_query_support",
            "3.0",
            "--full_set_augmentation_min_kc_score",
            "0.4",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.full_set_force_source_anchor is True
    assert args.full_set_augmentation_min_solver_support == 0.2
    assert args.full_set_augmentation_min_query_support == 3.0
    assert args.full_set_augmentation_min_kc_score == 0.4


def test_resample_argparser_accepts_full_gaussian_conflict_and_final_pruning_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_descriptor_conflict_graph",
            "/feedback/conflicts.json",
            "--full_set_descriptor_conflict_weight",
            "2.5",
            "--full_set_final_prune_no_query_utility",
            "--full_set_final_prune_conflict_threshold",
            "0.75",
            "--full_set_final_prune_min_keep_count",
            "1024",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert str(args.full_set_descriptor_conflict_graph) == "/feedback/conflicts.json"
    assert args.full_set_descriptor_conflict_weight == 2.5
    assert args.full_set_final_prune_no_query_utility is True
    assert args.full_set_final_prune_conflict_threshold == 0.75
    assert args.full_set_final_prune_min_keep_count == 1024


def test_resample_argparser_accepts_full_gaussian_v3_validation_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "OldHospital",
            "--source_path",
            "/data/OldHospital",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/full_sparse_set",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_auto_descriptor_conflict_top_k",
            "4",
            "--full_set_auto_descriptor_conflict_candidate_limit",
            "4096",
            "--full_set_auto_descriptor_conflict_min_cosine",
            "0.97",
            "--full_set_auto_descriptor_conflict_min_spatial_distance_m",
            "1.5",
            "--full_set_validation_min_query_landmarks",
            "8",
            "--full_set_validation_min_query_cells",
            "4",
            "--full_set_validation_min_query_depth_bins",
            "3",
            "--full_set_validation_min_query_bearing_spread",
            "0.2",
            "--full_set_validation_min_query_camera_spread_m",
            "2.5",
            "--full_set_local_search_rounds",
            "3",
            "--full_set_local_search_candidates_per_query",
            "64",
            "--full_set_local_search_max_additions",
            "512",
            "--full_set_local_search_reserve_count",
            "256",
            "--full_set_local_prune_conflict_threshold",
            "1.25",
            "--full_set_local_prune_validate_queries",
            "--cfg",
            "/configs/ulfloc.yaml",
        ]
    )

    assert args.full_set_auto_descriptor_conflict_top_k == 4
    assert args.full_set_auto_descriptor_conflict_candidate_limit == 4096
    assert args.full_set_auto_descriptor_conflict_min_cosine == 0.97
    assert args.full_set_auto_descriptor_conflict_min_spatial_distance_m == 1.5
    assert args.full_set_validation_min_query_landmarks == 8
    assert args.full_set_validation_min_query_cells == 4
    assert args.full_set_validation_min_query_depth_bins == 3
    assert args.full_set_validation_min_query_bearing_spread == 0.2
    assert args.full_set_validation_min_query_camera_spread_m == 2.5
    assert args.full_set_local_search_rounds == 3
    assert args.full_set_local_search_candidates_per_query == 64
    assert args.full_set_local_search_max_additions == 512
    assert args.full_set_local_search_reserve_count == 256
    assert args.full_set_local_prune_conflict_threshold == 1.25
    assert args.full_set_local_prune_validate_queries is True


def test_resample_argparser_accepts_full_gaussian_sparse_validation_profile_options():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/src",
            "--output_model_path",
            "/models/out",
            "--cfg",
            "/cfg.yml",
            "--solver_feedback",
            "/feedback/solver_feedback.pkl",
            "--solver_admissibility_path",
            "/feedback/solver_constraints.json",
            "--selection_policy",
            "full_gaussian_sparse_set",
            "--full_set_sparse_validation_profile",
            "/feedback/sparse_validation.json",
            "--full_set_sparse_validation_risk_weight",
            "3.5",
            "--full_set_sparse_validation_risk_reject_threshold",
            "0.1",
            "--full_set_sparse_validation_source_anchor_risk_exempt",
            "--full_set_sparse_validation_risk_expand_top_k",
            "128",
            "--full_set_sparse_validation_risk_expand_neighbors_per_seed",
            "12",
            "--full_set_sparse_validation_risk_expand_candidate_limit",
            "4096",
            "--full_set_sparse_validation_risk_expand_min_cosine",
            "0.96",
            "--full_set_sparse_validation_risk_expand_min_spatial_distance_m",
            "2.0",
            "--full_set_sparse_validation_risk_expand_weight",
            "0.75",
            "--full_set_sparse_validation_conflict_max_landmarks_per_query",
            "32",
            "--full_set_validation_query_prefill_target_fraction",
            "1.0",
            "--full_set_validation_query_prefill_max_candidates_per_query",
            "64",
            "--full_set_validation_query_prefill_max_per_spatial_cluster",
            "2",
            "--full_set_validation_query_prefill_min_cells_per_query",
            "4",
            "--full_set_validation_query_prefill_min_depth_bins_per_query",
            "3",
        ]
    )

    assert str(args.full_set_sparse_validation_profile) == "/feedback/sparse_validation.json"
    assert args.full_set_sparse_validation_risk_weight == 3.5
    assert args.full_set_sparse_validation_risk_reject_threshold == 0.1
    assert args.full_set_sparse_validation_source_anchor_risk_exempt is True
    assert args.full_set_sparse_validation_risk_expand_top_k == 128
    assert args.full_set_sparse_validation_risk_expand_neighbors_per_seed == 12
    assert args.full_set_sparse_validation_risk_expand_candidate_limit == 4096
    assert args.full_set_sparse_validation_risk_expand_min_cosine == 0.96
    assert args.full_set_sparse_validation_risk_expand_min_spatial_distance_m == 2.0
    assert args.full_set_sparse_validation_risk_expand_weight == 0.75
    assert args.full_set_sparse_validation_conflict_max_landmarks_per_query == 32
    assert args.full_set_validation_query_prefill_target_fraction == 1.0
    assert args.full_set_validation_query_prefill_max_candidates_per_query == 64
    assert args.full_set_validation_query_prefill_max_per_spatial_cluster == 2
    assert args.full_set_validation_query_prefill_min_cells_per_query == 4
    assert args.full_set_validation_query_prefill_min_depth_bins_per_query == 3


def test_full_set_descriptor_features_are_loaded_for_validation_risk_expansion():
    from argparse import Namespace

    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import full_set_needs_descriptor_features

    assert full_set_needs_descriptor_features(
        Namespace(
            full_set_auto_descriptor_conflict_top_k=0,
            full_set_sparse_validation_risk_expand_top_k=32,
        )
    )
    assert not full_set_needs_descriptor_features(
        Namespace(
            full_set_auto_descriptor_conflict_top_k=0,
            full_set_sparse_validation_risk_expand_top_k=0,
        )
    )


def test_sparse_validation_conflict_graph_connects_risk_to_protected_support():
    profile = {
        "split_name": "train_dev_seed13_20p_validation",
        "regression_per_query_risk_support": {
            "q_good.png": {"10": 4.0, "11": 2.0},
            "q_other.png": {"20": 1.0},
        },
        "protected_per_query_support": {
            "q_good.png": {"30": 8.0, "31": 4.0},
            "q_other.png": {"40": 1.0},
        },
    }

    edges = descriptor_conflict_edges_from_sparse_validation_profile(
        profile,
        max_landmarks_per_query=2,
    )

    assert edges[10][11] == pytest.approx(0.5)
    assert edges[10][30] == pytest.approx(1.0)
    assert edges[10][31] == pytest.approx(0.5)
    assert edges[11][30] == pytest.approx(0.5)
    assert edges[20][40] == pytest.approx(1.0)


def test_sparse_validation_profile_support_merges_protected_and_validated_support():
    profile = {
        "split_name": "train_dev_seed13_20p_validation",
        "protected_per_query_support": {
            "q_good.png": {"10": 2.0},
        },
        "validated_per_query_support": {
            "q_hard.png": {"20": 5.0},
            "q_good.png": {"11": 3.0},
        },
    }

    support = per_query_support_from_sparse_validation_profile(profile)

    assert support == {
        "q_good.png": {10: 2.0, 11: 3.0},
        "q_hard.png": {20: 5.0},
    }


def test_sparse_validation_profile_observations_merge_protected_and_validated_geometry():
    profile = {
        "split_name": "train_dev_seed13_20p_validation",
        "protected_per_query_observations": {
            "q_good.png": {
                "10": {
                    "xy_norm": [0.25, 0.5],
                    "bearing": [0.0, 0.1, 1.0],
                    "camera_xyz": [1.0, 2.0, 6.0],
                    "depth_m": 6.0,
                    "descriptor_margin": 0.33,
                    "local_geometry_score": 0.9,
                },
            },
        },
        "validated_per_query_observations": {
            "q_hard.png": {
                "20": {
                    "xy_norm": [0.75, 0.2],
                    "bearing": [0.2, 0.0, 1.0],
                    "camera_xyz": [3.0, 1.0, 8.0],
                    "depth_m": 8.0,
                },
            },
        },
    }

    observations = per_query_observations_from_sparse_validation_profile(profile)

    assert observations == {
        "q_good.png": {
            10: {
                "xy_norm": [0.25, 0.5],
                "bearing": [0.0, 0.1, 1.0],
                "camera_xyz": [1.0, 2.0, 6.0],
                "depth_m": 6.0,
                "descriptor_margin": 0.33,
                "local_geometry_score": 0.9,
            }
        },
        "q_hard.png": {
            20: {
                "xy_norm": [0.75, 0.2],
                "bearing": [0.2, 0.0, 1.0],
                "camera_xyz": [3.0, 1.0, 8.0],
                "depth_m": 8.0,
            }
        },
    }


def test_sparse_validation_profile_support_can_limit_to_protected_regressions():
    profile = {
        "split_name": "train_dev_seed13_20p_sparse_validation",
        "protected_regression_query_ids": ["q_regressed.png"],
        "protected_per_query_support": {
            "q_regressed.png": {"10": 2.0},
            "q_baseline_good_not_regressed.png": {"20": 5.0},
        },
        "validated_per_query_support": {
            "q_hard_fixed.png": {"30": 7.0},
        },
    }

    support = per_query_support_from_sparse_validation_profile(
        profile,
        support_scope="protected_regressions",
    )

    assert support == {
        "q_regressed.png": {10: 2.0},
        "q_hard_fixed.png": {30: 7.0},
    }


def test_sparse_validation_hard_reject_exempt_tensor_uses_only_protected_regressions():
    profile = {
        "split_name": "train_dev_seed13_20p_sparse_validation",
        "protected_regression_query_ids": ["q_regressed.png"],
        "protected_per_query_support": {
            "q_regressed.png": {"1": 1.0},
            "q_baseline_good_not_regressed.png": {"2": 2.0},
        },
        "validated_per_query_support": {
            "q_hard_fixed.png": {"3": 3.0},
        },
    }

    mask = sparse_validation_hard_reject_exempt_tensor_from_profile(profile, num_gaussians=5)

    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, True, False, False, False]


def test_accumulate_descriptor_proxy_averages_near_keypoint_descriptors():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import accumulate_descriptor_proxy

    descriptor_sum = torch.zeros((3, 2), dtype=torch.float32)
    descriptor_count = torch.zeros(3, dtype=torch.float32)
    visible_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    nearest_indices = torch.tensor([1, 0, 1], dtype=torch.long)
    near_keypoint = torch.tensor([True, True, False])
    keypoint_descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    accumulate_descriptor_proxy(
        descriptor_sum,
        descriptor_count,
        visible_indices=visible_indices,
        nearest_keypoint_indices=nearest_indices,
        near_keypoint=near_keypoint,
        keypoint_descriptors=keypoint_descriptors,
    )

    assert descriptor_sum.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    assert descriptor_count.tolist() == [1.0, 1.0, 0.0]


def test_resample_argparser_accepts_sample_random_seed():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--source_path",
            "/data/ShopFacade",
            "--source_model_path",
            "/models/native",
            "--output_model_path",
            "/models/candidate",
            "--cfg",
            "/configs/ulfloc.yaml",
            "--sample_random_seed",
            "13",
        ]
    )

    assert args.sample_random_seed == 13


def test_resample_script_limits_ulfloc_scene_to_train_images():
    source = Path("loc_gs/scripts/resample_ulfloc_with_solver_feedback.py").read_text(encoding="utf-8")

    assert "resolve_train_images_to_read" in source
    assert "resolve_ulfloc_source_path_for_loader" in source
    assert "images_to_read=train_images_to_read" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mask helper follows ULF CUDA mask path")
def test_resize_mask_channel_cuda_returns_two_dimensional_mask():
    from loc_gs.scripts.resample_ulfloc_with_solver_feedback import _resize_mask_channel_cuda

    mask = _resize_mask_channel_cuda(torch.ones(2, 3), height=2, width=3)

    assert tuple(mask.shape) == (2, 3)
    assert mask.dtype == torch.bool


def test_native_safe_core_uses_scores_not_gaussian_id_order_for_protection():
    native = torch.tensor([90, 91, 1, 2, 3, 4], dtype=torch.long)
    scores = torch.zeros(100, dtype=torch.float32)
    scores[90] = 0.1
    scores[91] = 0.0
    scores[1] = 0.2
    scores[2] = 0.3
    scores[3] = 9.0
    scores[4] = 8.0

    safe_core, tail = choose_native_safe_core_and_tail(native, safe_count=3, protection_scores=scores)

    assert safe_core.tolist() == [2, 3, 4]
    assert tail.tolist() == [90, 91, 1]


def test_merge_native_safe_core_keeps_native_prefix_and_inserts_weighted_candidates():
    native = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    candidate = torch.tensor([99, 98, 10, 97, 96, 95], dtype=torch.long)
    weights = torch.ones(100, dtype=torch.float32)
    weights[97] = 3.0
    weights[99] = 2.0
    weights[98] = 1.5

    merged = merge_native_safe_core_sampled_idx(
        native,
        candidate,
        solver_weights=weights,
        safe_core_fraction=0.5,
        max_solver_insertions=2,
    )

    assert merged.tolist() == [10, 11, 12, 97, 99, 13]
    assert len(set(merged.tolist())) == 6


def test_solver_coverage_merge_drops_low_score_native_not_high_id_tail():
    native = torch.tensor([90, 91, 1, 2, 3, 4], dtype=torch.long)
    candidate = torch.tensor([92, 93, 94, 95, 96, 97], dtype=torch.long)
    scores = torch.ones(100, dtype=torch.float32)
    scores[90] = 0.1
    scores[91] = 0.0
    scores[1] = 0.2
    scores[2] = 0.3
    scores[3] = 9.0
    scores[4] = 8.0
    scores[92] = 5.0
    scores[93] = 4.0
    cluster_ids = torch.arange(100, dtype=torch.long)

    merged = merge_solver_coverage_coreset_sampled_idx(
        native,
        candidate,
        solver_scores=scores,
        cluster_ids=cluster_ids,
        safe_core_fraction=0.5,
        max_solver_insertions=2,
    )

    sampled = set(merged.tolist())
    assert {2, 3, 4}.issubset(sampled)
    assert {92, 93}.issubset(sampled)
    assert 91 not in sampled


def test_merge_native_safe_core_guard_rejects_low_margin_insertions():
    native = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    candidate = torch.tensor([99, 98, 97, 96, 95, 94], dtype=torch.long)
    weights = torch.ones(100, dtype=torch.float32)
    weights[13] = 1.0
    weights[14] = 1.2
    weights[15] = 1.0
    weights[99] = 1.05
    weights[98] = 1.45
    weights[97] = 0.8

    merged = merge_native_safe_core_sampled_idx(
        native,
        candidate,
        solver_weights=weights,
        safe_core_fraction=0.5,
        max_solver_insertions=3,
        min_solver_insertion_score=1.0,
        min_solver_insertion_margin=0.2,
    )

    assert merged.tolist() == [10, 11, 14, 98, 12, 13]


def test_kc_solver_selection_score_penalizes_feedback_risks():
    payload = {
        "landmark_weights": torch.tensor([1.0, 1.8, 1.4, 1.2], dtype=torch.float32),
        "dense_worsen_risk": torch.tensor([0.0, 0.9, 0.0, 0.0], dtype=torch.float32),
        "ambiguity_risk": torch.tensor([0.0, 0.0, 0.8, 0.0], dtype=torch.float32),
        "artifact_risk": torch.tensor([0.0, 0.0, 0.0, 0.7], dtype=torch.float32),
    }

    score = build_kc_solver_selection_scores(
        payload,
        candidate_sampled_idx=torch.tensor([1, 2, 3], dtype=torch.long),
        kc_rank_weight=0.0,
        dense_worsen_risk_weight=1.0,
        ambiguity_risk_weight=1.0,
        artifact_risk_weight=1.0,
    )

    assert score[1] < score[0]
    assert score[2] < score[0]
    assert score[3] < score[0]


def test_solver_feedback_score_component_summary_reports_missing_weighted_risks():
    payload = {
        "schema_version": "ulfloc_solver_feedback_v1",
        "split_name": "selfmap_train",
        "landmark_weights": torch.ones(4),
        "landmark_support": torch.ones(4),
        "landmark_risk": torch.zeros(4),
    }

    summary = summarize_solver_feedback_score_components(
        payload,
        support_weight=0.25,
        regression_risk_weight=1.0,
        dense_worsen_risk_weight=1.0,
        ambiguity_risk_weight=1.0,
        artifact_risk_weight=1.0,
        hard_negative_risk_weight=1.0,
    )

    assert summary["components"]["support"]["present_keys"] == ["landmark_support"]
    assert summary["components"]["hard_negative_risk"]["present_keys"] == ["landmark_risk"]
    assert summary["has_query_level_constraints"] is False
    assert summary["missing_weighted_components"] == [
        "regression_risk",
        "dense_worsen_risk",
        "ambiguity_risk",
        "artifact_risk",
    ]


def test_native_safe_merge_can_use_kc_solver_selection_scores():
    native = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    candidate = torch.tensor([99, 98, 10, 97, 96, 95], dtype=torch.long)
    payload = {
        "landmark_weights": torch.ones(100, dtype=torch.float32),
        "dense_worsen_risk": torch.zeros(100, dtype=torch.float32),
        "ambiguity_risk": torch.zeros(100, dtype=torch.float32),
    }
    payload["landmark_weights"][97] = 2.0
    payload["landmark_weights"][99] = 1.1
    payload["dense_worsen_risk"][97] = 1.5

    scores = build_kc_solver_selection_scores(
        payload,
        candidate_sampled_idx=candidate,
        kc_rank_weight=0.25,
        dense_worsen_risk_weight=1.0,
    )
    merged = merge_native_safe_core_sampled_idx(
        native,
        candidate,
        solver_weights=scores,
        safe_core_fraction=0.5,
        max_solver_insertions=2,
    )

    assert merged.tolist() == [10, 11, 12, 99, 98, 13]


def test_solver_coverage_coreset_saturates_repeated_candidate_cluster():
    native = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    candidate = torch.tensor([90, 91, 92, 93, 94, 95], dtype=torch.long)
    scores = torch.ones(100, dtype=torch.float32)
    scores[90] = 5.0
    scores[91] = 4.9
    scores[92] = 4.8
    scores[93] = 3.0
    scores[94] = 2.9
    scores[95] = 2.8
    cluster_ids = torch.zeros(100, dtype=torch.long)
    cluster_ids[90] = 7
    cluster_ids[91] = 7
    cluster_ids[92] = 7
    cluster_ids[93] = 8
    cluster_ids[94] = 9
    cluster_ids[95] = 10

    merged = merge_solver_coverage_coreset_sampled_idx(
        native,
        candidate,
        solver_scores=scores,
        cluster_ids=cluster_ids,
        safe_core_fraction=0.5,
        max_solver_insertions=3,
        max_insertions_per_cluster=1,
    )

    assert merged.tolist() == [10, 11, 12, 90, 93, 94]


def test_build_spatial_cluster_ids_uses_stable_grid_hash_without_compaction():
    cluster_ids = build_spatial_cluster_ids(
        torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.2, 0.3],
                [10.0, 0.2, 0.3],
            ],
            dtype=torch.float32,
        ),
        grid_size_m=1.0,
    )

    assert int(cluster_ids[0].item()) == int(cluster_ids[1].item())
    assert int(cluster_ids[2].item()) != int(cluster_ids[0].item())
    assert int(cluster_ids[2].item()) != 1


def test_solver_query_coverage_prefers_undercovered_hard_query_over_unary_score():
    native = torch.tensor([0, 1, 2], dtype=torch.long)
    candidate = torch.tensor([3, 4, 5], dtype=torch.long)
    scores = torch.tensor([0.0, 0.0, 0.1, 0.99, 0.5, 0.4], dtype=torch.float32)
    constraints = {
        "hard_query_ids": ["easy", "hard"],
        "candidate_gain": {
            "3": {"easy": {"support": 1000.0}},
            "4": {"hard": {"support": 2.0}},
        },
        "source_loss": {
            "0": {"easy": {"support": 100.0}},
            "1": {"hard": {"support": 1.0}},
            "2": {},
        },
        "thresholds": {},
    }

    result = merge_solver_query_coverage_sampled_idx(
        native,
        candidate,
        solver_scores=scores,
        solver_constraints=constraints,
        safe_core_fraction=0.0,
        max_solver_insertions=1,
        coverage_saturation_mode="native_percentile",
        coverage_saturation_percentile=0.5,
        hard_query_min_gain=0.1,
    )

    sampled = set(result.sampled_idx.tolist())
    assert 4 in sampled
    assert 3 not in sampled
    assert result.metadata["coverage_policy"] == "solver_coverage_coreset"
    assert result.metadata["num_admissible_replacements"] == 1


def test_solver_query_coverage_preserves_native_safe_core():
    native = torch.tensor([0, 1, 2, 6], dtype=torch.long)
    candidate = torch.tensor([3, 4, 5, 7], dtype=torch.long)
    scores = torch.tensor([0.0, 0.0, 0.1, 0.99, 0.5, 0.4, 0.2, 0.1], dtype=torch.float32)
    constraints = {
        "hard_query_ids": ["hard"],
        "candidate_gain": {"4": {"hard": {"support": 2.0}}},
        "source_loss": {"0": {"hard": {"support": 100.0}}, "1": {}, "2": {}, "6": {}},
    }

    result = merge_solver_query_coverage_sampled_idx(
        native,
        candidate,
        solver_scores=scores,
        solver_constraints=constraints,
        safe_core_fraction=0.5,
        max_solver_insertions=1,
        hard_query_min_gain=0.1,
    )

    sampled = set(result.sampled_idx.tolist())
    assert {2, 6}.issubset(sampled)
    assert 4 in sampled
    assert result.metadata["safe_core_dropped_count"] == 0


def test_sparse_solver_aware_kc_selects_full_set_by_hard_query_coverage_not_insertion_budget():
    native = torch.tensor([0, 1, 2, 6], dtype=torch.long)
    candidate = torch.tensor([3, 4, 5, 7], dtype=torch.long)
    scores = torch.tensor([0.2, 0.1, 0.1, 9.0, 0.6, 0.5, 0.2, 0.4], dtype=torch.float32)
    constraints = {
        "hard_query_ids": ["hard"],
        "candidate_gain": {
            "3": {"easy": {"support": 100.0, "viable_tuple_mass": 100.0}},
            "4": {"hard": {"support": 2.0, "viable_tuple_mass": 2.0}},
            "5": {"hard": {"support": 1.0, "viable_tuple_mass": 1.0}},
        },
        "source_loss": {
            "0": {"hard": {"support": 2.0, "viable_tuple_mass": 2.0}},
            "1": {"easy": {"support": 1.0, "viable_tuple_mass": 1.0}},
        },
        "thresholds": {},
    }

    result = select_sparse_solver_aware_kc_sampled_idx(
        native,
        candidate,
        solver_scores=scores,
        solver_constraints=constraints,
        target_count=3,
        coverage_saturation_mode="native_percentile",
        coverage_saturation_percentile=0.5,
        hard_query_weight=4.0,
    )

    sampled = set(result.sampled_idx.tolist())
    assert 0 in sampled
    assert 4 in sampled
    assert 3 not in sampled
    assert result.metadata["selection_policy"] == "sparse_solver_aware_kc"
    assert result.metadata["budget_mode"] == "full_set"


def test_sparse_solver_aware_kc_uses_cluster_cap_before_unary_fill():
    native = torch.tensor([0, 1], dtype=torch.long)
    candidate = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    scores = torch.tensor([0.1, 0.1, 5.0, 4.0, 1.0, 0.5], dtype=torch.float32)
    clusters = torch.tensor([0, 1, 9, 9, 10, 11], dtype=torch.long)
    constraints = {
        "hard_query_ids": ["hard"],
        "candidate_gain": {
            "2": {"hard": {"support": 1.0, "viable_tuple_mass": 1.0}},
            "3": {"hard": {"support": 1.0, "viable_tuple_mass": 1.0}},
            "4": {"hard": {"support": 1.0, "viable_tuple_mass": 1.0}},
        },
        "source_loss": {},
        "thresholds": {},
    }

    result = select_sparse_solver_aware_kc_sampled_idx(
        native,
        candidate,
        solver_scores=scores,
        solver_constraints=constraints,
        target_count=3,
        cluster_ids=clusters,
        max_per_cluster=1,
    )

    sampled = set(result.sampled_idx.tolist())
    assert 2 in sampled
    assert 3 not in sampled
    assert 4 in sampled
    assert result.metadata["cluster_cap"]["enabled"] is True


def test_ray_feedback_per_query_support_is_merged_into_full_set_coverage():
    ray_payload = {
        "schema_version": "ray_attributed_solver_feedback_v1",
        "per_query_support": {
            "q0": {2: 1.5, 3: 0.25},
            "q1": {"4": 2.0},
        },
        "metadata": {"split_name": "selfmap_train"},
    }
    base = {"q0": {0: 1.0, 2: 0.5}}

    merged = merge_per_query_support(base, per_query_support_from_ray_feedback(ray_payload))

    assert merged["q0"][0] == 1.0
    assert merged["q0"][2] == 2.0
    assert merged["q0"][3] == 0.25
    assert merged["q1"][4] == 2.0


def test_sparse_validation_profile_protected_support_is_parsed_for_full_set_coverage():
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v1",
        "split_name": "train_dev",
        "protected_per_query_support": {
            "q_good": {"7": 3.0, "bad": 9.0},
            "q_empty": {},
        },
    }

    parsed = per_query_support_from_sparse_validation_profile(profile)

    assert parsed == {"q_good": {7: 3.0}}


def test_sparse_validation_profile_regression_basin_builds_conflict_edges():
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v1",
        "split_name": "train_dev",
        "regression_per_query_risk_support": {
            "q_bad": {"7": 10.0, "8": 5.0, "9": 1.0},
            "q_ignored": {"bad": 9.0, "10": -1.0},
        },
    }

    edges = descriptor_conflict_edges_from_sparse_validation_profile(
        profile,
        max_landmarks_per_query=2,
    )

    assert set(edges) == {7, 8}
    assert edges[7][8] == pytest.approx(0.5)
    assert edges[8][7] == pytest.approx(0.5)


def test_sparse_validation_profile_conflict_edges_reject_test_split():
    profile = {
        "split_name": "test",
        "regression_per_query_risk_support": {"q": {"1": 1.0, "2": 1.0}},
    }

    with pytest.raises(ValueError, match="test split"):
        descriptor_conflict_edges_from_sparse_validation_profile(profile)


def test_ray_feedback_per_query_observations_are_parsed_for_query_geometry():
    ray_payload = {
        "schema_version": "ray_attributed_solver_feedback_v1",
        "per_query_observations": {
            "q0": {
                2: {"cell": [1, 3], "depth_bin": 4},
                "3": {"cell_x": 2, "cell_y": 5, "depth_bin": 7},
            },
            "q1": {"bad": {"cell": [0, 0], "depth_bin": 0}},
        },
        "metadata": {"split_name": "selfmap_train"},
    }

    parsed = per_query_observations_from_ray_feedback(ray_payload)

    assert parsed == {
        "q0": {
            2: {"cell": [1, 3], "depth_bin": 4},
            3: {"cell_x": 2, "cell_y": 5, "depth_bin": 7},
        }
    }


def test_native_safe_merge_zero_insertions_preserves_native_order_exactly():
    native = torch.tensor([10, 30, 20, 40], dtype=torch.long)
    candidate = torch.tensor([99, 98, 97, 96], dtype=torch.long)
    weights = torch.zeros(100, dtype=torch.float32)
    weights[20] = 10.0
    weights[40] = 8.0

    merged = merge_native_safe_core_sampled_idx(
        native,
        candidate,
        solver_weights=weights,
        safe_core_fraction=0.5,
        max_solver_insertions=0,
    )

    assert merged.tolist() == native.tolist()


def test_solver_coverage_zero_insertions_preserves_native_order_exactly():
    native = torch.tensor([10, 30, 20, 40], dtype=torch.long)
    candidate = torch.tensor([99, 98, 97, 96], dtype=torch.long)
    scores = torch.zeros(100, dtype=torch.float32)
    scores[20] = 10.0
    scores[40] = 8.0

    merged = merge_solver_coverage_coreset_sampled_idx(
        native,
        candidate,
        solver_scores=scores,
        safe_core_fraction=0.5,
        max_solver_insertions=0,
    )

    assert merged.tolist() == native.tolist()


def test_merge_native_safe_core_rejects_invalid_duplicate_native_set():
    native = torch.tensor([1, 1, 2], dtype=torch.long)
    candidate = torch.tensor([3, 4, 5], dtype=torch.long)

    with pytest.raises(ValueError, match="native sampled_idx must be unique"):
        merge_native_safe_core_sampled_idx(native, candidate)


def test_write_map_resample_audit_bundle_keeps_map_prefixed_copy(tmp_path: Path):
    manifest = {"method": "ulfloc_solver_feedback_fixed_map_resampling"}
    metrics = {"sampled_count": 3}
    split_audit = {"audit_status": "passed"}

    write_map_resample_audit_bundle(tmp_path, manifest, metrics, split_audit)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "split_audit.json").exists()
    assert (tmp_path / "map_manifest.json").exists()
    assert (tmp_path / "map_metrics_summary.json").exists()
    assert (tmp_path / "map_split_audit.json").exists()
