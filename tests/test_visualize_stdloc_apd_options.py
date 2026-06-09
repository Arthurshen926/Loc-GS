from loc_gs.scripts.visualize_stdloc_hard_matches import (
    _resolve_slcdp_effective_options,
    _select_render_label_for_dense,
    build_argparser,
)


def test_apd_dense_cli_disables_legacy_transition_controllers():
    args = build_argparser().parse_args(
        [
            "--report_dir",
            "output/reports/example",
            "--candidate_root",
            "output/candidate",
            "--output_dir",
            "output/diagnostics/example",
            "--slcdp_transition_control",
            "--slcdp_soft_transition_control",
            "--apd_dense",
            "--apd_include_patch_candidates",
            "--apd_dense_group_weight",
            "0.75",
            "--apd_anchor_group_weight",
            "1.25",
            "--apd_anchor_monotonic",
            "--apd_anchor_monotonic_epsilon_px",
            "0.5",
            "--apd_risk_weighted",
            "--apd_risk_max_beta",
            "0.8",
            "--sadc_dense",
            "--sadc_include_patch_candidates",
            "--sadc_mode",
            "conflict_filter_patch_append",
            "--sadc_max_candidates",
            "1024",
            "--sadc_native_drop_percentile",
            "97",
            "--sadc_min_native_keep_ratio",
            "0.95",
            "--sadc_patch_add_percentile",
            "92",
            "--sadc_max_patch_fraction",
            "0.1",
            "--sadc_anchor_monotonic",
            "--sadc_anchor_monotonic_epsilon_px",
            "0.6",
            "--sadc_activation_mode",
            "dense_damage_risk",
            "--sadc_activation_min_risk",
            "0.45",
            "--sadc_activation_min_sparse_confidence",
            "0.25",
        ]
    )

    options = _resolve_slcdp_effective_options(args)

    assert options["apd_dense"] is True
    assert options["apd_include_patch_candidates"] is True
    assert options["apd_dense_group_weight"] == 0.75
    assert options["apd_anchor_group_weight"] == 1.25
    assert options["apd_anchor_monotonic"] is True
    assert options["apd_anchor_monotonic_epsilon_px"] == 0.5
    assert options["apd_risk_weighted"] is True
    assert options["apd_risk_max_beta"] == 0.8
    assert options["sadc_dense"] is True
    assert options["sadc_include_patch_candidates"] is True
    assert options["sadc_mode"] == "conflict_filter_patch_append"
    assert options["sadc_max_candidates"] == 1024
    assert options["sadc_native_drop_percentile"] == 97.0
    assert options["sadc_min_native_keep_ratio"] == 0.95
    assert options["sadc_patch_add_percentile"] == 92.0
    assert options["sadc_max_patch_fraction"] == 0.1
    assert options["sadc_anchor_monotonic"] is True
    assert options["sadc_anchor_monotonic_epsilon_px"] == 0.6
    assert options["sadc_activation_mode"] == "dense_damage_risk"
    assert options["sadc_activation_min_risk"] == 0.45
    assert options["sadc_activation_min_sparse_confidence"] == 0.25
    assert options["slcdp_transition_control"] is False
    assert options["slcdp_soft_transition_control"] is False


def test_apd_dense_uses_clean_render_label_instead_of_legacy_repair_skip():
    label = _select_render_label_for_dense(
        clean_render_selection={"selected_label": "gated_ray_centroid+1.000"},
        repair_selection={"decision": "skip_dense_keep_sparse", "selected_label": "base"},
        apd_dense=True,
    )

    assert label == "gated_ray_centroid+1.000"


def test_legacy_dense_keeps_repair_skip_as_base_label():
    label = _select_render_label_for_dense(
        clean_render_selection={"selected_label": "gated_ray_centroid+1.000"},
        repair_selection={"decision": "skip_dense_keep_sparse", "selected_label": "gated_ray_centroid+1.000"},
        apd_dense=False,
    )

    assert label == "base"


def test_apd_dense_defaults_keep_native_pose_with_no_regression_gate():
    args = build_argparser().parse_args(
        [
            "--report_dir",
            "output/reports/example",
            "--candidate_root",
            "output/candidate",
            "--output_dir",
            "output/diagnostics/example",
            "--apd_dense",
        ]
    )

    options = _resolve_slcdp_effective_options(args)

    assert options["apd_dense"] is True
    assert options["apd_use_refined_pose"] is False
    assert options["apd_no_regression_gate"] is True
