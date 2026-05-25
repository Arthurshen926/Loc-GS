import json
import pickle
import subprocess
import sys

import torch


def _dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def test_export_lsf_candidate_probe_map_appends_candidates_with_audit(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "probe_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(
        source / "detector" / "sampled_scores.pkl",
        {
            "sampled_scores": torch.tensor([0.2, 0.3], dtype=torch.float32),
            "score_avg": torch.tensor([0.2, 0.3, 0.8, 0.7], dtype=torch.float32),
        },
    )
    (source / "manifest.json").write_text(
        json.dumps({"scene": "ToyScene", "split_name": "selfmap_train"}),
        encoding="utf-8",
    )
    candidate_pool = tmp_path / "candidate_pool.pt"
    torch.save(torch.tensor([1, 2, 3, 2], dtype=torch.long), candidate_pool)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_candidate_probe_map",
            "--source_map",
            str(source),
            "--candidate_pool_path",
            str(candidate_pool),
            "--output_map",
            str(output),
            "--max_candidates",
            "3",
        ],
        check=True,
    )

    sampled = torch.as_tensor(pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")))
    scores = pickle.load((output / "detector" / "sampled_scores.pkl").open("rb"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))

    assert sampled.tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(
        torch.as_tensor(scores["sampled_scores"]),
        torch.tensor([0.2, 0.3, 0.8, 0.7], dtype=torch.float32),
    )
    assert manifest["method"] == "loc_gs_lsf_candidate_probe_map"
    assert manifest["probe_only"] is True
    assert manifest["same_budget"] is False
    assert manifest["source_count"] == 2
    assert manifest["candidate_added_count"] == 2
    assert source_manifest == {"scene": "ToyScene", "split_name": "selfmap_train"}
    assert metrics["output_count"] == 4


def test_export_lsf_candidate_probe_map_rejects_source_as_output(tmp_path):
    source = tmp_path / "source_map"
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    candidate_pool = tmp_path / "candidate_pool.pt"
    torch.save(torch.tensor([2], dtype=torch.long), candidate_pool)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_lsf_candidate_probe_map",
            "--source_map",
            str(source),
            "--candidate_pool_path",
            str(candidate_pool),
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
