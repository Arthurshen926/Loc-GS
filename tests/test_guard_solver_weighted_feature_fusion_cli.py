import json

import torch
import torch.nn.functional as F

from loc_gs.scripts.guard_solver_weighted_feature_fusion import main


def test_guard_solver_weighted_feature_fusion_cli_writes_audited_artifact(tmp_path):
    native = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    artifact_path = tmp_path / "solver_fused.pt"
    torch.save(
        {
            "descriptors": F.normalize(native + torch.tensor([[0.0, 0.5], [0.5, 0.0]]), p=2, dim=-1),
            "base_descriptors": native,
            "gaussian_ids": torch.tensor([10, 11], dtype=torch.long),
            "metadata": {"source_split_name": "train_dev"},
        },
        artifact_path,
    )
    pair_cache_path = tmp_path / "pair_cache.pt"
    torch.save(
        {
            "landmark_id": torch.tensor([[0], [1]], dtype=torch.long),
            "candidate_mask": torch.ones(2, 1, dtype=torch.bool),
            "cosine": torch.full((2, 1), 0.9),
            "reprojection_error": torch.ones(2, 1),
            "metadata": {"split_name": "train_dev", "processed_images": 2, "query_keypoint_count": 2, "topk": 1},
        },
        pair_cache_path,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "split_name": "train_dev",
                "paired_queries": [
                    {"query_id": "q0", "delta_sparse_te_cm": 30.0},
                    {"query_id": "q1", "delta_sparse_te_cm": -30.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "guarded" / "selected_landmark_descriptors.pt"

    assert (
        main(
            [
                "--descriptor_artifact",
                str(artifact_path),
                "--pair_cache",
                str(pair_cache_path),
                "--validation_profile",
                str(profile_path),
                "--output",
                str(output_path),
                "--scene",
                "ShopFacade",
                "--split_name",
                "train_dev",
            ]
        )
        == 0
    )

    guarded = torch.load(output_path, map_location="cpu")
    assert guarded["metadata"]["guarded_landmark_count"] == 1
    assert (output_path.parent / "manifest.json").exists()
    assert (output_path.parent / "metrics_summary.json").exists()
    assert (output_path.parent / "split_audit.json").exists()
    assert (output_path.parent / "command.txt").exists()
    assert (output_path.parent / "git_status.txt").exists()
