import json
import shlex

import pytest

from loc_gs.scripts.run_sparse_pnp_validation_loop import (
    build_next_resample_command,
    run_sparse_pnp_validation_loop_once,
)
from loc_gs.stdloc_native.sparse_pnp_validation import load_ulfloc_sparse_rows


def _write_run(path, rows, image_names):
    path.mkdir(parents=True)
    (path / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    log_lines = []
    for image_name, row in zip(image_names, rows):
        log_lines.append(f"Localize image:{image_name}")
        log_lines.append(f"sparse: AE: {row['sparse_AE']}deg, TE: {row['sparse_TE']}cm")
    (path / "output.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def test_build_next_resample_command_replaces_profile_and_output_model():
    command = (
        "python -m loc_gs.scripts.resample_ulfloc_with_solver_feedback "
        "--scene ShopFacade --output_model_path /old/model "
        "--full_set_sparse_validation_profile /old/profile.json --force"
    )

    updated = build_next_resample_command(
        command,
        next_output_model_path="/new/model",
        sparse_validation_profile="/new/profile.json",
    )

    parts = shlex.split(updated)
    assert parts[parts.index("--output_model_path") + 1] == "/new/model"
    assert parts[parts.index("--full_set_sparse_validation_profile") + 1] == "/new/profile.json"
    assert parts.count("--full_set_sparse_validation_profile") == 1


def test_build_next_resample_command_appends_missing_profile():
    command = "python -m loc_gs.scripts.resample_ulfloc_with_solver_feedback --output_model_path /old/model"

    updated = build_next_resample_command(
        command,
        next_output_model_path="/new/model",
        sparse_validation_profile="/new/profile.json",
    )

    parts = shlex.split(updated)
    assert parts[parts.index("--output_model_path") + 1] == "/new/model"
    assert parts[parts.index("--full_set_sparse_validation_profile") + 1] == "/new/profile.json"


def test_build_next_resample_command_normalizes_direct_script_path_to_module_invocation():
    command = (
        "/root/Loc-GS/loc_gs/scripts/resample_ulfloc_with_solver_feedback.py "
        "--scene ShopFacade --output_model_path /old/model"
    )

    updated = build_next_resample_command(
        command,
        next_output_model_path="/new/model",
        sparse_validation_profile="/new/profile.json",
    )

    parts = shlex.split(updated)
    assert parts[:3] == [
        "/root/miniconda3/envs/cybersim_agent/bin/python",
        "-m",
        "loc_gs.scripts.resample_ulfloc_with_solver_feedback",
    ]
    assert parts[parts.index("--output_model_path") + 1] == "/new/model"


def test_build_next_resample_command_can_force_regression_guard_mode():
    command = (
        "python -m loc_gs.scripts.resample_ulfloc_with_solver_feedback "
        "--scene ShopFacade --output_model_path /old/model "
        "--full_set_sparse_validation_support_scope all "
        "--full_set_validation_query_prefill_target_fraction 0.5 "
        "--full_set_validation_query_prefill_max_candidates_per_query 96 "
        "--full_set_main_min_query_gain 0.5"
    )

    updated = build_next_resample_command(
        command,
        next_output_model_path="/new/model",
        sparse_validation_profile="/new/profile.json",
        repair_regressions=True,
    )

    parts = shlex.split(updated)
    assert parts[parts.index("--full_set_sparse_validation_support_scope") + 1] == "all"
    assert parts[parts.index("--full_set_validation_query_prefill_target_fraction") + 1] == "1.0"
    assert parts[parts.index("--full_set_validation_query_prefill_max_candidates_per_query") + 1] == "256"
    assert "--full_set_validation_query_prefill_force_all" in parts
    assert parts[parts.index("--full_set_sparse_validation_risk_reject_threshold") + 1] == "0.0"
    assert parts[parts.index("--full_set_sparse_validation_risk_weight") + 1] == "0.25"
    assert "--full_set_sparse_validation_risk_expand_top_k" not in parts
    assert parts[parts.index("--full_set_main_min_query_gain") + 1] == "0.5"


def test_build_next_resample_command_preserves_full_budget_when_guarding_regressions():
    command = (
        "python -m loc_gs.scripts.resample_ulfloc_with_solver_feedback "
        "--scene ShopFacade --output_model_path /old/model "
        "--full_set_max_landmarks 20000 "
        "--full_set_final_prune_no_query_utility "
        "--full_set_final_prune_min_keep_count 16000"
    )

    updated = build_next_resample_command(
        command,
        next_output_model_path="/new/model",
        sparse_validation_profile="/new/profile.json",
        repair_regressions=True,
    )

    parts = shlex.split(updated)
    assert "--full_set_final_prune_no_query_utility" in parts
    assert parts[parts.index("--full_set_final_prune_min_keep_count") + 1] == "20000"


def test_build_next_resample_command_forces_full_gaussian_policy_and_removes_metric_only_flag():
    command = (
        "python -m loc_gs.scripts.resample_ulfloc_with_solver_feedback "
        "--scene ShopFacade --output_model_path /old/model "
        "--selection_policy solver_feedback_topk "
        "--full_set_allow_metric_only_sparse_validation_profile "
        "--full_set_source_anchor_mode hard "
        "--full_set_source_anchor_weight 3.0 "
        "--full_set_kc_anchor_count 2048"
    )

    updated = build_next_resample_command(
        command,
        next_output_model_path="/new/model",
        sparse_validation_profile="/new/profile.json",
    )

    parts = shlex.split(updated)
    assert parts[parts.index("--selection_policy") + 1] == "full_gaussian_sparse_set"
    assert "--full_set_allow_metric_only_sparse_validation_profile" not in parts
    assert parts[parts.index("--full_set_sparse_validation_profile") + 1] == "/new/profile.json"
    assert parts[parts.index("--full_set_source_anchor_mode") + 1] == "off"
    assert parts[parts.index("--full_set_source_anchor_weight") + 1] == "0.0"
    assert parts[parts.index("--full_set_kc_anchor_count") + 1] == "0"


def test_sparse_pnp_validation_loop_once_builds_profile_and_next_command(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        [
            {"sparse_AE": 0.1, "sparse_TE": 3.0, "sparse": {"inliers": 100}},
            {"sparse_AE": 0.2, "sparse_TE": 80.0, "sparse": {"inliers": 20}},
        ],
        ["q_good.png", "q_hard.png"],
    )
    _write_run(
        candidate,
        [
            {"sparse_AE": 0.8, "sparse_TE": 45.0, "sparse": {"inliers": 40}},
            {"sparse_AE": 0.1, "sparse_TE": 8.0, "sparse": {"inliers": 80}},
        ],
        ["q_good.png", "q_hard.png"],
    )
    candidate_manifest = candidate / "map_manifest.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "command": (
                    "python -m loc_gs.scripts.resample_ulfloc_with_solver_feedback "
                    "--scene ShopFacade --output_model_path /old/model"
                )
            }
        ),
        encoding="utf-8",
    )

    result = run_sparse_pnp_validation_loop_once(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        scene="ShopFacade",
        split_name="train_dev",
        output_dir=tmp_path / "loop",
        candidate_map_manifest=candidate_manifest,
        next_output_model_path=tmp_path / "next_model",
        protected_te_cm=15.0,
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
    )

    profile = json.loads((tmp_path / "loop" / "sparse_pnp_validation_profile.json").read_text())
    next_command = (tmp_path / "loop" / "next_resample_command.txt").read_text()
    assert profile["protected_regression_query_ids"] == ["q_good.png"]
    assert profile["hard_improved_query_ids"] == ["q_hard.png"]
    assert result["metrics"]["regression_20cm_count"] == 1
    assert str(tmp_path / "next_model") in next_command
    assert str(tmp_path / "loop" / "sparse_pnp_validation_profile.json") in next_command
    assert (tmp_path / "loop" / "manifest.json").exists()
    assert (tmp_path / "loop" / "split_audit.json").exists()


def test_sparse_pnp_validation_loop_refuses_test_split(tmp_path):
    with pytest.raises(ValueError, match="test split"):
        run_sparse_pnp_validation_loop_once(
            baseline_run_dir=tmp_path / "baseline",
            candidate_run_dir=tmp_path / "candidate",
            scene="ShopFacade",
            split_name="test",
            output_dir=tmp_path / "loop",
        )


def test_load_ulfloc_sparse_rows_accepts_locgs_rows_schema(tmp_path):
    run = tmp_path / "locgs_eval"
    run.mkdir()
    (run / "results.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "image_name": "q0.png",
                        "sparse_te_cm": 2.5,
                        "sparse_re_deg": 0.1,
                        "sparse_inliers": 100,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = load_ulfloc_sparse_rows(run)

    assert rows[0]["query_id"] == "q0.png"
    assert rows[0]["sparse_te_cm"] == 2.5
    assert rows[0]["sparse_inliers"] == 100
