import json
import pickle
import subprocess
import sys

import torch


def _dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def test_export_lsf_selected_edits_map_writes_non_prefix_subset(tmp_path):
    source = tmp_path / "source_map"
    candidate = tmp_path / "candidate_map"
    output = tmp_path / "output_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1, 2, 3], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(4, dtype=torch.float32))
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    _dump_pickle(
        candidate / "detector" / "sampled_scores.pkl",
        {
            "sampled_scores": torch.tensor([0.4, 0.3, 0.9, 0.7], dtype=torch.float32),
            "score_avg": torch.tensor([0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7], dtype=torch.float32),
            "selector": torch.zeros(7, dtype=torch.float32),
        },
    )
    edit_manifest = candidate / "manifest.json"
    edit_manifest.write_text(
        json.dumps(
            {
                "method": "source_solver_aware",
                "edits": [
                    {"add_id": 4, "drop_id": 2, "utility_gain": 0.8},
                    {"add_id": 5, "drop_id": 1, "utility_gain": 0.7},
                    {"add_id": 6, "drop_id": 3, "utility_gain": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_selected_edits_map",
            "--source_map",
            str(source),
            "--candidate_map",
            str(candidate),
            "--edit_manifest",
            str(edit_manifest),
            "--select_edits",
            "1,3",
            "--output_map",
            str(output),
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    scores = pickle.load((output / "detector" / "sampled_scores.pkl").open("rb"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sampled.tolist() == [0, 1, 4, 6]
    assert torch.allclose(
        torch.as_tensor(scores["sampled_scores"]),
        torch.tensor([0.4, 0.3, 0.9, 0.7], dtype=torch.float32),
    )
    assert manifest["method"] == "loc_gs_lsf_selected_edits_resampling"
    assert manifest["selected_edit_indices_one_based"] == [1, 3]
    assert manifest["selected_edits"]["same_budget"] is True


def test_export_lsf_selected_edits_map_rejects_source_as_output_without_deleting_it(tmp_path):
    source = tmp_path / "source_map"
    candidate = tmp_path / "candidate_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(source / "detector" / "sampled_scores.pkl", torch.ones(2, dtype=torch.float32))
    _dump_pickle(
        candidate / "detector" / "sampled_scores.pkl",
        {
            "sampled_scores": torch.ones(2, dtype=torch.float32),
            "score_avg": torch.ones(3, dtype=torch.float32),
        },
    )
    edit_manifest = candidate / "manifest.json"
    edit_manifest.write_text(
        json.dumps({"edits": [{"add_id": 2, "drop_id": 1, "utility_gain": 1.0}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_selected_edits_map",
            "--source_map",
            str(source),
            "--candidate_map",
            str(candidate),
            "--edit_manifest",
            str(edit_manifest),
            "--select_edits",
            "1",
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
