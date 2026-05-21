import json
import subprocess
import sys

import torch


def test_eval_dense_support_mask_cli_filters_match_tensor(tmp_path):
    matches_path = tmp_path / "matches.pt"
    reliability_path = tmp_path / "reliability.pt"
    out_dir = tmp_path / "out"
    torch.save(
        {
            "query_yx": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "reference_yx": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "scores": torch.tensor([0.1, 0.9, 0.8, 0.7], dtype=torch.float32),
        },
        matches_path,
    )
    torch.save(torch.tensor([0.0, 0.8, 0.9, 0.2], dtype=torch.float32), reliability_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.eval_dense_support_mask",
            "--matches",
            str(matches_path),
            "--reliability",
            str(reliability_path),
            "--min_reliability",
            "0.5",
            "--output_dir",
            str(out_dir),
        ],
        check=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    filtered = torch.load(out_dir / "filtered_matches.pt", map_location="cpu")
    assert summary["kept_count"] == 2
    assert filtered["indices"].tolist() == [1, 2]

