import json
import pickle
import subprocess
import sys

import torch
from plyfile import PlyData


def _dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def test_export_lsf_solver_aware_map_writes_same_budget_detector_payload(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2, 3], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(4, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    risk_path = tmp_path / "risk.pt"
    safe_core_path = tmp_path / "safe.pt"
    torch.save(torch.tensor([0.9, 0.8, 0.2, 0.3, 0.95, 0.99], dtype=torch.float32), selector_path)
    torch.save(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32), positive_path)
    torch.save(torch.zeros(6, dtype=torch.float32), risk_path)
    torch.save(torch.tensor([0, 1], dtype=torch.long), safe_core_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--hard_negative_risk_path",
            str(risk_path),
            "--safe_core_path",
            str(safe_core_path),
            "--output_map",
            str(output),
            "--min_positive_support",
            "0.1",
            "--max_edits",
            "2",
        ],
        check=True,
    )

    sampled = pickle.load((output / "detector" / "sampled_idx.pkl").open("rb"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(sampled) == 4
    assert 4 in torch.as_tensor(sampled).tolist()
    assert 5 not in torch.as_tensor(sampled).tolist()
    assert manifest["method"] == "loc_gs_lsf_solver_aware_resampling"
    assert manifest["solver_aware"]["safe_core_dropped_count"] == 0


def test_export_lsf_solver_aware_map_rejects_source_as_output_without_deleting_it(tmp_path):
    source = tmp_path / "source_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    torch.save(torch.ones(4, dtype=torch.float32), selector_path)
    torch.save(torch.ones(4, dtype=torch.float32), positive_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--output_map",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must differ" in result.stderr
    assert (source / "detector" / "sampled_idx.pkl").exists()


def test_export_lsf_solver_aware_map_uses_solver_consensus_support_artifact(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    support_path = tmp_path / "solver_consensus_support.pt"
    torch.save(
        {
            "support_score": torch.tensor([0.1, 0.2, 0.1, 0.95, 0.99], dtype=torch.float32),
            "hard_negative_risk": torch.zeros(5, dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "metadata": {
                "schema": "solver_consensus_support_v1",
                "scene": "ToyScene",
                "split": "selfmap_train",
                "selected_count": 2,
            },
        },
        support_path,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--solver_consensus_support_path",
            str(support_path),
            "--output_map",
            str(output),
            "--min_positive_support",
            "0.5",
            "--max_hard_negative_risk",
            "0.5",
            "--max_edits",
            "2",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert 3 in sampled.tolist()
    assert 4 not in sampled.tolist()
    assert manifest["solver_consensus_support_path"] == str(support_path)
    assert manifest["solver_consensus_support"]["schema"] == "solver_consensus_support_v1"
    assert manifest["evidence_gate"]["rejected_by_hard_negative"] == 1


def test_export_lsf_solver_aware_map_can_write_fixed_dense_locability(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(
        source / "detector" / "sampled_scores.pkl",
        {
            "sampled_scores": torch.ones(2, dtype=torch.float32),
            "score_avg": torch.ones(3, dtype=torch.float32),
        },
    )
    point_cloud = source / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    point_cloud.parent.mkdir(parents=True)
    point_cloud.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    lsf_path = tmp_path / "localization_support_field.pt"
    torch.save(
        {
            "support_for_selection": torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32),
            "support_score": torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32),
            "hard_negative_risk": torch.zeros(3, dtype=torch.float32),
            "dense_worsen_risk": torch.zeros(3, dtype=torch.float32),
            "metadata": {"schema": "localization_support_field_v1", "split_name": "selfmap_train"},
        },
        lsf_path,
    )
    safe_core_path = tmp_path / "safe_core_protect.pt"
    torch.save(torch.tensor([1], dtype=torch.long), safe_core_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--localization_support_field_path",
            str(lsf_path),
            "--output_map",
            str(output),
            "--write_dense_locability",
            "--max_edits",
            "0",
        ],
        check=True,
    )

    data = PlyData.read(str(output / "point_cloud" / "iteration_30000" / "point_cloud.ply"))["vertex"].data
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "locability_logit" in data.dtype.names
    assert data["locability_logit"][1] > data["locability_logit"][0]
    assert manifest["dense_support"]["enabled"] is True
    assert manifest["localization_support_field_path"] == str(lsf_path)


def test_export_lsf_solver_aware_map_can_disable_lsf_risk_for_sparse_edits(tmp_path):
    source = tmp_path / "source_map"
    default_output = tmp_path / "default_output_map"
    disabled_output = tmp_path / "disabled_output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(
        source / "detector" / "sampled_scores.pkl",
        {
            "sampled_scores": torch.ones(2, dtype=torch.float32),
            "score_avg": torch.full((3,), 0.5, dtype=torch.float32),
        },
    )
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    lsf_path = tmp_path / "localization_support_field.pt"
    torch.save(
        {
            "support_for_selection": torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32),
            "support_score": torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32),
            "hard_negative_risk": torch.zeros(3, dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            "metadata": {"schema": "localization_support_field_v1", "split_name": "selfmap_train"},
        },
        lsf_path,
    )
    safe_core_path = tmp_path / "safe_core.pt"
    torch.save(torch.tensor([1], dtype=torch.long), safe_core_path)
    common = [
        sys.executable,
        "-m",
        "loc_gs.scripts.export_lsf_solver_aware_map",
        "--source_map",
        str(source),
        "--localization_support_field_path",
        str(lsf_path),
        "--safe_core_path",
        str(safe_core_path),
        "--max_edits",
        "1",
    ]

    subprocess.run([*common, "--output_map", str(default_output)], check=True)
    subprocess.run(
        [*common, "--output_map", str(disabled_output), "--disable_lsf_sparse_risk"],
        check=True,
    )

    default_sampled = torch.as_tensor(pickle.load((default_output / "detector" / "sampled_idx.pkl").open("rb")))
    disabled_sampled = torch.as_tensor(pickle.load((disabled_output / "detector" / "sampled_idx.pkl").open("rb")))
    disabled_manifest = json.loads((disabled_output / "manifest.json").read_text(encoding="utf-8"))
    assert default_sampled.tolist() == [1, 2]
    assert disabled_sampled.tolist() == [0, 1]
    assert disabled_manifest["hyperparameters"]["disable_lsf_sparse_risk"] is True


def test_export_lsf_solver_aware_map_can_penalize_dense_worsen_in_candidate_order(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(2, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    lsf_path = tmp_path / "localization_support_field.pt"
    torch.save(
        {
            "support_for_selection": torch.tensor([0.1, 0.1, 1.0, 0.5], dtype=torch.float32),
            "support_score": torch.ones(4, dtype=torch.float32),
            "hard_negative_risk": torch.zeros(4, dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([0.0, 0.0, 0.2, 0.0], dtype=torch.float32),
            "metadata": {"schema": "localization_support_field_v1", "split_name": "selfmap_train"},
        },
        lsf_path,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--localization_support_field_path",
            str(lsf_path),
            "--output_map",
            str(output),
            "--max_edits",
            "1",
            "--extra_dense_worsen_penalty",
            "0.5",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert 3 in sampled.tolist()
    assert 2 not in sampled.tolist()
    assert manifest["hyperparameters"]["extra_dense_worsen_penalty"] == 0.5
    assert manifest["utility_shaping"]["extra_dense_worsen_penalty"] == 0.5


def test_export_lsf_solver_aware_map_can_protect_high_native_score_sources(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(
        source / "detector" / "sampled_scores.pkl",
        {
            "sampled_scores": torch.tensor([0.9, 0.1], dtype=torch.float32),
            "score_avg": torch.tensor([0.9, 0.1, 0.5], dtype=torch.float32),
        },
    )
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    lsf_path = tmp_path / "localization_support_field.pt"
    torch.save(
        {
            "support_for_selection": torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32),
            "support_score": torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32),
            "hard_negative_risk": torch.zeros(3, dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            "metadata": {"schema": "localization_support_field_v1", "split_name": "selfmap_train"},
        },
        lsf_path,
    )
    safe_core_path = tmp_path / "safe_core_protect.pt"
    torch.save(torch.tensor([1], dtype=torch.long), safe_core_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--localization_support_field_path",
            str(lsf_path),
            "--output_map",
            str(output),
            "--safe_core_path",
            str(safe_core_path),
            "--max_edits",
            "1",
            "--protect_source_score_min",
            "0.8",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sampled.tolist() == [0, 1]
    assert manifest["hyperparameters"]["protect_source_score_min"] == 0.8
    assert manifest["solver_aware"]["protected_source_score_count"] == 1


def test_export_lsf_solver_aware_map_honors_solver_admissibility_constraints(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    constraints_path = tmp_path / "solver_constraints.json"
    torch.save(torch.tensor([0.9, 0.8, 0.1, 0.99], dtype=torch.float32), selector_path)
    torch.save(torch.ones(4, dtype=torch.float32), positive_path)
    constraints_path.write_text(
        json.dumps(
            {
                "hard_query_ids": [0],
                "candidate_gain": {"3": {"0": {"support": 1.0, "viable_tuple_mass": 0.0, "logdet_H": -1.0}}},
                "source_loss": {"2": {"0": {"support": 0.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0}}},
                "thresholds": {"min_logdet_delta": 0.0},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--solver_admissibility_path",
            str(constraints_path),
            "--output_map",
            str(output),
            "--max_edits",
            "1",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sampled.tolist() == [0, 1, 2]
    assert manifest["solver_aware"]["num_rejected_by_admissibility"] == 1
    assert manifest["solver_admissibility"]["enabled"] is True


def test_export_lsf_solver_aware_map_accepts_string_query_solver_constraints(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    constraints_path = tmp_path / "solver_constraints.json"
    torch.save(torch.tensor([0.9, 0.8, 0.1, 0.99], dtype=torch.float32), selector_path)
    torch.save(torch.ones(4, dtype=torch.float32), positive_path)
    constraints_path.write_text(
        json.dumps(
            {
                "hard_query_ids": ["seq1/frame00001.png"],
                "candidate_gain": {
                    "3": {"seq1/frame00001.png": {"support": 1.0, "viable_tuple_mass": 0.0, "logdet_H": -1.0}}
                },
                "source_loss": {
                    "2": {"seq1/frame00001.png": {"support": 0.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0}}
                },
                "thresholds": {"min_logdet_delta": 0.0},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--solver_admissibility_path",
            str(constraints_path),
            "--output_map",
            str(output),
            "--max_edits",
            "1",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sampled.tolist() == [0, 1, 2]
    assert manifest["solver_admissibility"]["hard_query_count"] == 1
    assert manifest["solver_aware"]["num_rejected_by_admissibility"] == 1


def test_export_lsf_solver_aware_map_can_require_candidate_gain(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    constraints_path = tmp_path / "solver_constraints.json"
    torch.save(torch.tensor([0.9, 0.8, 0.1, 0.99, 0.98], dtype=torch.float32), selector_path)
    torch.save(torch.ones(5, dtype=torch.float32), positive_path)
    constraints_path.write_text(
        json.dumps(
            {
                "hard_query_ids": [0],
                "candidate_gain": {"4": {"0": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}}},
                "source_loss": {},
                "thresholds": {},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--solver_admissibility_path",
            str(constraints_path),
            "--require_candidate_gain",
            "--output_map",
            str(output),
            "--max_edits",
            "1",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sampled.tolist() == [0, 1, 4]
    assert manifest["solver_admissibility"]["require_candidate_gain"] is True


def test_export_lsf_solver_aware_map_can_cap_edits_from_candidate_gain_count(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    constraints_path = tmp_path / "solver_constraints.json"
    torch.save(torch.tensor([0.1, 0.1, 0.1, 0.99, 0.98], dtype=torch.float32), selector_path)
    torch.save(torch.ones(5, dtype=torch.float32), positive_path)
    constraints_path.write_text(
        json.dumps(
            {
                "hard_query_ids": [0],
                "candidate_gain": {
                    "3": {"0": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
                    "4": {"0": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
                },
                "source_loss": {},
                "thresholds": {},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--solver_admissibility_path",
            str(constraints_path),
            "--require_candidate_gain",
            "--output_map",
            str(output),
            "--max_edits",
            "10",
            "--max_edits_candidate_gain_fraction",
            "0.5",
            "--min_adaptive_edits",
            "1",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(set(sampled.tolist()) & {3, 4}) == 1
    assert manifest["hyperparameters"]["max_edits"] == 10
    assert manifest["hyperparameters"]["effective_max_edits"] == 1
    assert manifest["adaptive_edit_budget"]["candidate_gain_count"] == 2
    assert manifest["adaptive_edit_budget"]["candidate_gain_fraction"] == 0.5
    assert manifest["solver_aware"]["added_non_native_count"] == 1


def test_export_lsf_solver_aware_map_honors_v3_dense_and_cvar_thresholds(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    constraints_path = tmp_path / "solver_constraints_v3.json"
    torch.save(torch.tensor([0.9, 0.8, 0.1, 0.99], dtype=torch.float32), selector_path)
    torch.save(torch.ones(4, dtype=torch.float32), positive_path)
    constraints_path.write_text(
        json.dumps(
            {
                "hard_query_ids": ["easy", "hard"],
                "candidate_gain": {
                    "3": {
                        "easy": {
                            "support": 3.0,
                            "viable_tuple_mass": 0.0,
                            "logdet_H": 0.0,
                            "min_eigenvalue": 0.0,
                            "dense_worsen_risk": 0.0,
                        },
                        "hard": {
                            "support": -2.0,
                            "viable_tuple_mass": 0.0,
                            "logdet_H": 0.0,
                            "min_eigenvalue": 0.0,
                            "dense_worsen_risk": 0.5,
                        },
                    }
                },
                "source_loss": {
                    "2": {
                        "easy": {
                            "support": 0.0,
                            "viable_tuple_mass": 0.0,
                            "logdet_H": 0.0,
                            "min_eigenvalue": 0.0,
                            "dense_worsen_risk": 0.0,
                        },
                        "hard": {
                            "support": 0.0,
                            "viable_tuple_mass": 0.0,
                            "logdet_H": 0.0,
                            "min_eigenvalue": 0.0,
                            "dense_worsen_risk": 0.0,
                        },
                    }
                },
                "thresholds": {
                    "min_support_delta": -10.0,
                    "min_logdet_delta": -10.0,
                    "max_dense_worsen_delta": 0.0,
                    "cvar_alpha": 0.5,
                    "min_cvar_score": 0.0,
                    "cvar_weights": {
                        "support": 1.0,
                        "viable_tuple_mass": 0.0,
                        "logdet_H": 0.0,
                        "min_eigenvalue": 0.0,
                        "ambiguity": 0.0,
                        "dense_worsen_risk": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--solver_admissibility_path",
            str(constraints_path),
            "--output_map",
            str(output),
            "--max_edits",
            "1",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sampled.tolist() == [0, 1, 2]
    assert manifest["solver_aware"]["num_rejected_by_admissibility"] == 1
    assert manifest["solver_admissibility"]["rejected_examples"][0]["reasons"] == [
        "dense_worsen_rise",
        "hard_query_cvar",
    ]
    assert manifest["solver_admissibility"]["thresholds"]["cvar_alpha"] == 0.5


def test_export_lsf_solver_aware_map_records_candidate_pool_path(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(2, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    candidate_pool_path = tmp_path / "candidate_pool.pt"
    torch.save(torch.tensor([0.1, 0.1, 0.9, 0.8], dtype=torch.float32), selector_path)
    torch.save(torch.ones(4, dtype=torch.float32), positive_path)
    torch.save(torch.tensor([2, 3], dtype=torch.long), candidate_pool_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--candidate_pool_path",
            str(candidate_pool_path),
            "--output_map",
            str(output),
            "--max_edits",
            "1",
        ],
        check=True,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_pool_path"] == str(candidate_pool_path)
    assert manifest["candidate_pool_required"] is False
    assert manifest["solver_aware"]["candidate_pool_count"] == 2


def test_export_lsf_solver_aware_map_require_candidate_pool_rejects_missing_pool(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(2, dtype=torch.float32))
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    torch.save(torch.ones(3, dtype=torch.float32), selector_path)
    torch.save(torch.ones(3, dtype=torch.float32), positive_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--output_map",
            str(output),
            "--require_candidate_pool",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "candidate_pool_path is required" in result.stderr
    assert not output.exists()


def test_export_lsf_solver_aware_map_uses_coverage_policy(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(3, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    selector_path = tmp_path / "selector.pt"
    positive_path = tmp_path / "positive.pt"
    candidate_pool_path = tmp_path / "candidate_pool.pt"
    constraints_path = tmp_path / "solver_constraints.json"
    torch.save(torch.tensor([0.2, 0.2, 0.0, 0.9, 0.8, 0.95], dtype=torch.float32), selector_path)
    torch.save(torch.ones(6, dtype=torch.float32), positive_path)
    torch.save(torch.tensor([3, 4, 5], dtype=torch.long), candidate_pool_path)
    constraints_path.write_text(
        json.dumps(
            {
                "hard_query_ids": ["q0", "q1"],
                "candidate_gain": {
                    "3": {"q0": {"support": 2.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
                    "4": {"q1": {"support": 2.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
                    "5": {"q0": {"support": 1.0, "viable_tuple_mass": 0.1, "dense_worsen_risk": 3.0}},
                },
                "source_loss": {
                    "0": {"q0": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
                    "1": {"q1": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
                    "2": {},
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
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_solver_aware_map",
            "--source_map",
            str(source),
            "--selector_path",
            str(selector_path),
            "--positive_support_path",
            str(positive_path),
            "--candidate_pool_path",
            str(candidate_pool_path),
            "--solver_admissibility_path",
            str(constraints_path),
            "--selection_policy",
            "coverage",
            "--output_map",
            str(output),
            "--max_edits",
            "2",
            "--min_coverage_gain",
            "0.1",
        ],
        check=True,
    )

    sampled = set(torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb"))).tolist())
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {3, 4}.issubset(sampled)
    assert 5 not in sampled
    assert manifest["selection_policy"] == "coverage"
    assert manifest["solver_aware"]["coverage_policy"] == "solver_coverage_coreset"
