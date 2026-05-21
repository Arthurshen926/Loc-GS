import json
import pickle
import subprocess
import sys

import torch


def test_compare_solver_tuple_banks_cli_writes_json_and_markdown(tmp_path):
    payload = {
        "query_id": torch.zeros(4, dtype=torch.long),
        "query_yx": torch.tensor(
            [[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]],
            dtype=torch.float32,
        ),
        "landmark_id": torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.long),
        "cosine": torch.full((4, 2), 0.9, dtype=torch.float32),
        "candidate_mask": torch.ones((4, 2), dtype=torch.bool),
        "reprojection_error": torch.full((4, 2), 1.0, dtype=torch.float32),
        "metadata": {
            "scene": "ToyScene",
            "reprojection_threshold_px": 3.0,
            "split_audit": {"audit_status": "passed"},
        },
    }
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 1.0],
            [0.0, 0.01, 1.0],
            [0.01, 0.01, 1.0],
        ],
        dtype=torch.float32,
    )
    cache_path = tmp_path / "cache.pt"
    source_path = tmp_path / "source.pkl"
    candidate_path = tmp_path / "candidate.pkl"
    xyz_path = tmp_path / "xyz.pt"
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    torch.save(payload, cache_path)
    torch.save(xyz, xyz_path)
    with source_path.open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3], dtype=torch.long), handle)
    with candidate_path.open("wb") as handle:
        pickle.dump(torch.tensor([4, 5, 6, 7], dtype=torch.long), handle)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.compare_solver_tuple_banks",
            "--episode_cache",
            str(cache_path),
            "--source_idx",
            str(source_path),
            "--candidate_idx",
            str(candidate_path),
            "--num_gaussians",
            "8",
            "--xyz_path",
            str(xyz_path),
            "--output_json",
            str(json_path),
            "--output_report",
            str(md_path),
            "--min_logdet_h",
            "-5",
            "--min_spread_3d",
            "0.05",
        ],
        check=True,
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary_delta"]["support_count_sum"] == 0
    assert report["summary_delta"]["viable_tuple_mass"] < 0
    assert "support count" in md_path.read_text(encoding="utf-8")

