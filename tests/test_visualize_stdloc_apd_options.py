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
        ]
    )

    options = _resolve_slcdp_effective_options(args)

    assert options["apd_dense"] is True
    assert options["apd_include_patch_candidates"] is True
    assert options["apd_dense_group_weight"] == 0.75
    assert options["apd_anchor_group_weight"] == 1.25
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
