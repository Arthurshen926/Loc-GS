import json
import pickle
import subprocess
import sys

import torch


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
