from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import torch


def test_build_ulfloc_native_feature_fusion_log_dry_run_prepares_fixed_sample_and_feedback(tmp_path: Path) -> None:
    source_log = tmp_path / "source_log"
    source_log.mkdir()
    sampled_idx = torch.tensor([3, 7], dtype=torch.long)
    with (source_log / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled_idx, handle)
    with (source_log / "keypoints_features.pkl").open("wb") as handle:
        pickle.dump(torch.eye(2), handle)
    feedback = tmp_path / "solver_feedback.pkl"
    with feedback.open("wb") as handle:
        pickle.dump({"schema_version": "ulfloc_solver_feedback_v1", "split_name": "selfmap_train"}, handle)
    output = tmp_path / "native_fused_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_native_feature_fusion_log",
            "--ulf_root",
            "/root/ULF-Loc",
            "--model_path",
            str(tmp_path / "model"),
            "--source_path",
            str(tmp_path / "scene"),
            "--input_log_dir",
            str(source_log),
            "--solver_feedback",
            str(feedback),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--dry_run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with (output / "keypoints_sampled_idx.pkl").open("rb") as handle:
        saved_idx = pickle.load(handle)
    with (output / "solver_feedback.pkl").open("rb") as handle:
        saved_feedback = pickle.load(handle)

    assert torch.equal(torch.as_tensor(saved_idx), sampled_idx)
    assert saved_feedback["split_name"] == "selfmap_train"
    assert (output / "config.yaml").exists()
    assert (output / "manifest.json").exists()
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()

