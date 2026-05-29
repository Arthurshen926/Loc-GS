import json
import subprocess
import sys

import torch


def _stdloc_result_row(sparse_te: float, sparse_re: float, dense_te: float, dense_re: float) -> dict:
    return {
        "sparse": {"inliers": 20},
        "dense": [{"inliers": 100}],
        "sparse_TE": sparse_te,
        "sparse_AE": sparse_re,
        "dense_TE": dense_te,
        "dense_AE": dense_re,
    }


def test_build_dense_transition_audit_cli_writes_gate_artifact(tmp_path):
    baseline = tmp_path / "baseline_results.json"
    candidate = tmp_path / "candidate_results.json"
    output = tmp_path / "transition_audit"
    baseline.write_text(
        json.dumps(
            [
                _stdloc_result_row(3.0, 1.0, 15.0, 1.0),
                _stdloc_result_row(30.0, 1.0, 25.0, 1.0),
                _stdloc_result_row(6.0, 1.0, 7.0, 1.0),
            ]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            [
                _stdloc_result_row(3.0, 1.0, 4.0, 1.0),
                _stdloc_result_row(12.0, 1.0, 55.0, 1.0),
                _stdloc_result_row(4.0, 1.0, 12.0, 1.0),
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_dense_transition_audit",
            "--baseline_results",
            str(baseline),
            "--candidate_results",
            str(candidate),
            "--scene",
            "ToyScene",
            "--split_name",
            "selfmap_train",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    audit = json.loads((output / "dense_transition_audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    hard_cases = (output / "hard_dense_cases.csv").read_text(encoding="utf-8")
    assert audit["candidate_dense_worsened_count"] == 2
    assert audit["baseline_dense_worsened_count"] == 1
    assert audit["dense_worsened_reduction_rate"] == -1.0
    assert audit["candidate_sparse_correct_dense_wrong_count"] == 1
    assert audit["candidate_regression_20cm_count"] == 1
    assert metrics["candidate_regression_50cm_count"] == 0
    assert manifest["recipe"] == "lsf_v10_dense_transition_audit"
    assert "candidate_dense_worsened" in hard_cases


def test_build_dense_transition_audit_cli_rejects_test_split_without_override(tmp_path):
    baseline = tmp_path / "baseline_results.json"
    candidate = tmp_path / "candidate_results.json"
    output = tmp_path / "transition_audit"
    rows = [_stdloc_result_row(1.0, 1.0, 1.0, 1.0)]
    baseline.write_text(json.dumps(rows), encoding="utf-8")
    candidate.write_text(json.dumps(rows), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_dense_transition_audit",
            "--baseline_results",
            str(baseline),
            "--candidate_results",
            str(candidate),
            "--scene",
            "ToyScene",
            "--split_name",
            "test",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split dense transition audits are not allowed" in result.stderr


def test_build_failure_profile_from_solver_constraints_cli_writes_train_only_profile(tmp_path):
    baseline = tmp_path / "baseline_results.json"
    candidate = tmp_path / "candidate_results.json"
    constraints = tmp_path / "solver_constraints.json"
    output = tmp_path / "failure_profile"
    baseline.write_text(
        json.dumps(
            [
                {"image_name": "q_easy", "sparse_TE": 5.0, "dense_TE": 8.0},
                {"image_name": "q_hard", "sparse_TE": 40.0, "dense_TE": 45.0},
            ]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            [
                {"image_name": "q_easy", "sparse_TE": 5.0, "dense_TE": 7.0},
                {"image_name": "q_hard", "sparse_TE": 20.0, "dense_TE": 31.0},
            ]
        ),
        encoding="utf-8",
    )
    constraints.write_text(
        json.dumps(
            {
                "candidate_gain": {
                    "10": {"q_hard": {"support": 1.0, "dense_worsen_risk": 0.0}},
                    "11": {"q_hard": {"support": 1.0, "ambiguity": 2.0}},
                },
                "source_loss": {
                    "1": {"q_hard": {"support": 10.0}},
                    "2": {"q_hard": {"support": 1.0}},
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_failure_profile_from_solver_constraints",
            "--baseline_results",
            str(baseline),
            "--candidate_results",
            str(candidate),
            "--solver_constraints",
            str(constraints),
            "--scene",
            "ToyScene",
            "--split_name",
            "train",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    profile = json.loads((output / "failure_profile.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert profile["query_baseline_dense_te_cm"]["q_hard"] == 45.0
    assert profile["dense_worsened_query_ids"] == ["q_hard"]
    assert profile["ambiguity_risk"]["11"] == 2.0
    assert profile["source_loss"]["2"] > profile["source_loss"]["1"]
    assert manifest["recipe"] == "lsf_v7_failure_profile"


def test_build_query_support_banks_cli_writes_audited_artifact(tmp_path):
    observations = tmp_path / "observations.json"
    support = tmp_path / "support.json"
    output = tmp_path / "banks"
    observations.write_text(
        json.dumps(
            [
                {"query_id": "easy", "keypoint_count": 200, "detector_centroid_yx": [0.5, 0.5], "topk_match_entropy": 0.1, "sparse_inliers": 80, "pnp_confidence": 0.9},
                {"query_id": "hard", "keypoint_count": 40, "detector_centroid_yx": [0.2, 0.7], "topk_match_entropy": 0.8, "sparse_inliers": 6, "pnp_confidence": 0.2},
            ]
        ),
        encoding="utf-8",
    )
    support.write_text(json.dumps({"1": {"easy": 1.0}, "2": {"hard": 3.0}}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_query_support_banks",
            "--query_observations",
            str(observations),
            "--landmark_support",
            str(support),
            "--native_safe_core_ids",
            "1",
            "--hard_query_ids",
            "hard",
            "--split_name",
            "selfmap_train",
            "--output_dir",
            str(output),
            "--router_topk",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output / "support_banks.pt", map_location="cpu")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [bank["name"] for bank in artifact["banks"]] == [
        "native_safe_core",
        "ambiguity_safe",
        "occlusion_robust",
        "hard_tail_recovery",
    ]
    assert manifest["recipe"] == "lsf_v8_support_banks_top2"
    assert (output / "metrics_summary.json").exists()
    assert (output / "split_audit.json").exists()


def test_build_query_support_banks_cli_accepts_failure_profile_support(tmp_path):
    observations = tmp_path / "observations.json"
    profile = tmp_path / "failure_profile.json"
    output = tmp_path / "banks"
    observations.write_text(
        json.dumps(
            [
                {"query_id": "q0", "keypoint_count": 100, "detector_centroid_yx": [0.5, 0.5], "topk_match_entropy": 0.8, "sparse_inliers": 20, "pnp_confidence": 0.4},
                {"query_id": "q1", "keypoint_count": 30, "detector_centroid_yx": [0.2, 0.8], "topk_match_entropy": 0.2, "sparse_inliers": 4, "pnp_confidence": 0.1},
            ]
        ),
        encoding="utf-8",
    )
    profile.write_text(
        json.dumps(
            {
                "candidate_query_gain": {
                    "10": {"q0": {"support": 1.0, "viable_tuple_mass": 2.0}},
                    "20": {"q1": {"support": 2.0, "dense_worsen_risk": 0.5}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_query_support_banks",
            "--query_observations",
            str(observations),
            "--failure_profile",
            str(profile),
            "--native_safe_core_ids",
            "10",
            "--hard_query_ids",
            "q1",
            "--split_name",
            "selfmap_train",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output / "support_banks.pt", map_location="cpu")
    hard_tail = next(bank for bank in artifact["banks"] if bank["name"] == "hard_tail_recovery")
    assert hard_tail["landmark_weights"][20] > 0.0


def test_build_query_support_banks_cli_rejects_mismatched_failure_profile_queries(tmp_path):
    observations = tmp_path / "observations.json"
    profile = tmp_path / "failure_profile.json"
    output = tmp_path / "banks"
    observations.write_text(
        json.dumps(
            [
                {"query_id": "observed", "keypoint_count": 100, "detector_centroid_yx": [0.5, 0.5], "topk_match_entropy": 0.2, "sparse_inliers": 20, "pnp_confidence": 0.4},
            ]
        ),
        encoding="utf-8",
    )
    profile.write_text(
        json.dumps({"candidate_query_gain": {"10": {"unmatched": {"support": 1.0}}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_query_support_banks",
            "--query_observations",
            str(observations),
            "--failure_profile",
            str(profile),
            "--split_name",
            "selfmap_train",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "support query ids do not overlap query observations" in result.stderr


def test_build_query_observations_from_stdloc_results_cli_writes_observable_only(tmp_path):
    results = tmp_path / "results.json"
    split = tmp_path / "dataset_train.txt"
    output = tmp_path / "observations"
    results.write_text(
        json.dumps(
            [
                _stdloc_result_row(3.0, 1.0, 4.0, 1.0) | {"sparse": {"inliers": 40}},
                _stdloc_result_row(7.0, 1.0, 8.0, 1.0) | {"sparse": {"inliers": 5}},
            ]
        ),
        encoding="utf-8",
    )
    split.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "seq/a.png 0 0 0 1 0 0 0\n"
        "seq/b.png 0 0 0 1 0 0 0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_query_observations_from_stdloc_results",
            "--results",
            str(results),
            "--split_file",
            str(split),
            "--split_name",
            "selfmap_train",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    observations = json.loads((output / "query_observations.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert observations[0]["query_id"] == "seq/a.png"
    assert observations[0]["sparse_inliers"] == 40
    assert 0.0 < observations[0]["pnp_confidence"] <= 1.0
    assert "dense_te_cm" not in observations[0]
    assert manifest["uses_pose_error_or_gt"] is False


def test_build_stable_region_mask_cli_writes_proxy_artifact(tmp_path):
    image = tmp_path / "image.pt"
    output = tmp_path / "mask"
    tensor = torch.zeros(3, 4, 4)
    tensor[:, :2, :] = 1.0
    tensor[1, 2:, :] = 0.9
    torch.save(tensor, image)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_stable_region_mask",
            "--image_tensor",
            str(image),
            "--split_name",
            "selfmap_train",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output / "stable_region_mask.pt", map_location="cpu")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert artifact["metadata"]["external_model"] == "none"
    assert manifest["recipe"] == "lsf_v11_stable_proxy_mask"
    assert manifest["single_path_deployment"] is True


def test_build_dense_match_verifier_artifact_cli_writes_mask_and_audit(tmp_path):
    payload = tmp_path / "dense_input.pt"
    output = tmp_path / "dense_verifier"
    torch.save(
        {
            "query_yx": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            "reference_yx": torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32),
            "descriptor_scores": torch.tensor([0.9, 0.9], dtype=torch.float32),
            "local_geometry": torch.tensor([0.9, 0.1], dtype=torch.float32),
            "lsf_support": torch.tensor([0.9, 0.1], dtype=torch.float32),
            "alpha_dominance": torch.tensor([0.9, 0.1], dtype=torch.float32),
            "ambiguity": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "depth_uncertainty": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "pose_leverage": torch.tensor([0.8, 0.1], dtype=torch.float32),
        },
        payload,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_dense_match_verifier_artifact",
            "--input",
            str(payload),
            "--split_name",
            "selfmap_train",
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output / "dense_match_verifier.pt", map_location="cpu")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert artifact["metadata"]["kept_count"] == 1
    assert manifest["recipe"] == "lsf_v10_dense_verifier_mask"
    assert manifest["branch_selection"] is False


def test_build_multihypothesis_pnp_diagnostic_cli_requires_diagnostic_and_marks_not_main(tmp_path):
    matches = tmp_path / "matches.json"
    output = tmp_path / "multihyp"
    matches.write_text(
        json.dumps(
            [
                {"group": "a", "score": 1.0, "inlier_prior": 5, "logdet_H": 1.0},
                {"group": "b", "score": 0.5, "inlier_prior": 4, "dense_verifier_score": 1.0},
            ]
        ),
        encoding="utf-8",
    )

    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_multihypothesis_pnp_diagnostic",
            "--matches_json",
            str(matches),
            "--output_dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0

    good = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_multihypothesis_pnp_diagnostic",
            "--matches_json",
            str(matches),
            "--output_dir",
            str(output),
            "--diagnostic",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert good.returncode == 0, good.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))
    assert manifest["diagnostic_only"] is True
    assert manifest["main_method_allowed"] is False
    assert split_audit["audit_status"] == "failed"
