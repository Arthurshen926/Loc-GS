import json
import pickle
import subprocess
import sys

import torch


def test_build_query_conditioned_solver_constraints_cli_writes_exporter_payload(tmp_path):
    cache = tmp_path / "episode.pt"
    source_idx = tmp_path / "sampled_idx.pkl"
    output = tmp_path / "constraints.json"
    torch.save(
        {
            "query_id": torch.tensor([7, 7], dtype=torch.long),
            "candidate_landmark_ids": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            "candidate_cosine": torch.tensor([[0.9, 0.8], [0.7, 0.4]], dtype=torch.float32),
            "candidate_reprojection_error": torch.tensor([[1.0, 2.0], [1.0, 1.0]], dtype=torch.float32),
            "candidate_visible": torch.ones((2, 2), dtype=torch.bool),
            "candidate_pnp_inlier": torch.ones((2, 2), dtype=torch.bool),
            "metadata": {"split_name": "selfmap_train_rendered"},
        },
        cache,
    )
    with source_idx.open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_query_conditioned_solver_constraints",
            "--episode_cache",
            str(cache),
            "--source_idx",
            str(source_idx),
            "--output_json",
            str(output),
            "--num_gaussians",
            "4",
            "--hard_query_ids",
            "7",
            "--reprojection_threshold_px",
            "3.0",
            "--score_threshold",
            "0.5",
            "--min_candidate_positive",
            "0.5",
            "--min_logdet_delta",
            "0.0",
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "loc_gs_solver_admissibility_v1"
    assert payload["hard_query_ids"] == [7]
    assert payload["candidate_gain"]["2"]["7"]["support"] == 1.0
    assert "3" not in payload["candidate_gain"]
    assert payload["source_loss"]["0"]["7"]["support"] == 1.0
    assert payload["thresholds"]["min_logdet_delta"] == 0.0
