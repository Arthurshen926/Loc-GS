import json
import subprocess
import sys

import torch


def test_build_alpha_reliability_cache_cli_outputs_summary_and_tensors(tmp_path):
    contributor_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    weights = torch.tensor([[0.9, 0.1], [0.5, 0.5]], dtype=torch.float32)
    support = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    ids_path = tmp_path / "ids.pt"
    weights_path = tmp_path / "weights.pt"
    support_path = tmp_path / "support.pt"
    out_dir = tmp_path / "alpha_cache"
    torch.save(contributor_ids, ids_path)
    torch.save(weights, weights_path)
    torch.save(support, support_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_alpha_reliability_cache",
            "--contributor_ids",
            str(ids_path),
            "--contribution_weights",
            str(weights_path),
            "--gaussian_support",
            str(support_path),
            "--threshold",
            "0.3",
            "--output_dir",
            str(out_dir),
        ],
        check=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    payload = torch.load(out_dir / "rendered_support_mask.pt", map_location="cpu")
    assert summary["composition_proxy"] is True
    assert summary["exact_raster_weights"] is False
    assert summary["ray_count"] == 2
    assert payload["keep"].tolist() == [True, False]

