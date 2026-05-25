import json
import subprocess
import sys

import pytest
import torch

from loc_gs.diagnostics.localization_support_field import build_localization_support_field


def _write_support(path):
    torch.save(
        {
            "support_score": torch.tensor([0.0, 0.03, 0.02, 0.04], dtype=torch.float32),
            "observed_count": torch.tensor([0, 3, 2, 4], dtype=torch.long),
            "inlier_consensus": torch.tensor([0.0, 0.6, 0.4, 0.8], dtype=torch.float32),
            "hard_negative_risk": torch.tensor([0.0, 0.1, 0.2, 0.0], dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([0.0, 0.0, 0.3, 0.0], dtype=torch.float32),
            "metadata": {
                "schema": "solver_consensus_support_v1",
                "scene": "ToyScene",
                "split": "selfmap_train_rendered",
                "source_feedback_bank": "feedback_bank.jsonl",
            },
        },
        path,
    )


def _write_constraints(path):
    path.write_text(
        json.dumps(
            {
                "format": "loc_gs_solver_admissibility_v2_feedback_bank",
                "hard_query_ids": ["hard"],
                "candidate_gain": {
                    "3": {
                        "hard": {
                            "support": 2.0,
                            "viable_tuple_mass": 1.5,
                            "logdet_H": 1.2,
                            "min_eigenvalue": 0.8,
                            "dense_worsen_risk": 0.0,
                            "ambiguity": 0.1,
                        }
                    }
                },
                "source_loss": {
                    "1": {
                        "hard": {
                            "support": 1.0,
                            "viable_tuple_mass": 1.0,
                            "logdet_H": 0.9,
                            "min_eigenvalue": 0.7,
                            "dense_worsen_risk": 0.0,
                            "ambiguity": 0.0,
                        }
                    }
                },
                "thresholds": {
                    "cvar_weights": {
                        "support": 1.0,
                        "viable_tuple_mass": 1.0,
                        "logdet_H": 1.0,
                        "min_eigenvalue": 1.0,
                        "dense_worsen_risk": -1.0,
                        "ambiguity": -1.0,
                    }
                },
                "metadata": {"split_name": "selfmap_train_rendered"},
            }
        ),
        encoding="utf-8",
    )


def test_build_localization_support_field_merges_support_and_solver_metrics(tmp_path):
    support_path = tmp_path / "solver_consensus_support.pt"
    constraints_path = tmp_path / "solver_constraints.json"
    _write_support(support_path)
    _write_constraints(constraints_path)

    payload = build_localization_support_field(
        solver_consensus_support_path=support_path,
        solver_admissibility_path=constraints_path,
        support_score_mode="rank_observed",
    )

    assert payload["metadata"]["schema"] == "localization_support_field_v1"
    assert payload["metadata"]["split_name"] == "selfmap_train_rendered"
    assert payload["support_for_selection"].tolist() == pytest.approx([0.0, 0.5, 0.0, 1.0])
    assert payload["solver_set_utility"][3] > payload["solver_set_utility"][2]
    assert payload["solver_set_utility"][1] > 0.0
    assert payload["pose_information"][3] > 0.0
    assert payload["pose_information_gain"][3] > 0.0
    assert payload["hard_query_support"][3] > 0.0
    assert payload["ambiguity_risk"][3] > 0.0
    assert payload["visibility_stability"].tolist() == pytest.approx([0.0, 0.75, 0.5, 1.0])


def test_build_localization_support_field_cli_writes_artifact_and_manifest(tmp_path):
    support_path = tmp_path / "solver_consensus_support.pt"
    output_dir = tmp_path / "lsf"
    _write_support(support_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_localization_support_field",
            "--solver_consensus_support_path",
            str(support_path),
            "--output_dir",
            str(output_dir),
            "--support_score_mode",
            "rank_observed",
        ],
        check=True,
    )

    payload = torch.load(output_dir / "localization_support_field.pt", map_location="cpu")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["schema"] == "localization_support_field_v1"
    assert manifest["artifact"] == "localization_support_field.pt"
    assert manifest["support_score_mode"] == "rank_observed"
    assert manifest["command"]
    assert manifest["split_audit"]["audit_status"] == "unknown"
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "split_audit.json").exists()
    assert (output_dir / "git_status.txt").exists()
