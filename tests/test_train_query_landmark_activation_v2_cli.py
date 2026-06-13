from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def _write_cache(path: Path, *, split_name: str = "selfmap_train") -> None:
    torch.save(
        {
            "schema_version": "query_landmark_activation_v2_cache_test",
            "scene": "ShopFacade",
            "split_name": split_name,
            "query_token_dim": 6,
            "landmark_token_dim": 8,
            "landmark_ids": torch.tensor([10, 20, 30], dtype=torch.long),
            "safe_core_indices": torch.tensor([0], dtype=torch.long),
            "examples": [
                {
                    "query_id": "q1",
                    "query_tokens": torch.randn(5, 6),
                    "landmark_tokens": torch.randn(3, 8),
                    "geometry_visible": torch.tensor([1.0, 1.0, 0.0]),
                    "solver_positive": torch.tensor([1.0, 0.0, 0.0]),
                    "harmful_negative": torch.tensor([0.0, 0.0, 1.0]),
                    "protected_support": torch.tensor([1.0, 0.0, 0.0]),
                }
            ],
            "split_audit": {
                "audit_status": "passed",
                "split_name": split_name,
                "test_split_used": split_name == "test",
                "official_test_used": split_name == "test",
            },
        },
        path,
    )


def test_train_query_landmark_activation_v2_cli_writes_token_checkpoint_and_audits(tmp_path: Path) -> None:
    cache = tmp_path / "activation_v2_cache.pt"
    _write_cache(cache)
    output = tmp_path / "activation_v2"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_query_landmark_activation_v2",
            "--cache",
            str(cache),
            "--output_dir",
            str(output),
            "--epochs",
            "2",
            "--hidden_dim",
            "12",
            "--attention_top_k",
            "3",
            "--top_n",
            "2",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    checkpoint = torch.load(output / "landmark_activation_v2.pth", map_location="cpu", weights_only=False)
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))

    assert checkpoint["activation_model_type"] == "v2_tokens"
    assert checkpoint["query_feature_mode"] == "token_cross_attention"
    assert checkpoint["landmark_token_components"] == ["descriptor_l2", "xyz_scene_normalized", "prior_score"]
    assert checkpoint["query_token_dim"] == 6
    assert checkpoint["landmark_token_dim"] == 8
    assert "state_dict" in checkpoint
    assert metrics["activation_model_type"] == "v2_tokens"
    assert metrics["trained_example_count"] == 1
    assert metrics["uses_mean_query_descriptor"] is False
    assert manifest["source_cache_path"] == str(cache)
    assert split_audit["test_split_used"] is False
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()


def test_train_query_landmark_activation_v2_cli_rejects_test_cache(tmp_path: Path) -> None:
    cache = tmp_path / "activation_v2_cache.pt"
    _write_cache(cache, split_name="test")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_query_landmark_activation_v2",
            "--cache",
            str(cache),
            "--output_dir",
            str(tmp_path / "activation_v2"),
            "--epochs",
            "0",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_train_query_landmark_activation_v2_cli_accepts_global_landmark_tokens(tmp_path: Path) -> None:
    cache = tmp_path / "activation_v2_cache.pt"
    torch.save(
        {
            "schema_version": "query_landmark_activation_v2_cache_test",
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "query_token_dim": 6,
            "landmark_token_dim": 8,
            "landmark_ids": torch.tensor([10, 20, 30], dtype=torch.long),
            "landmark_tokens": torch.randn(3, 8),
            "examples": [
                {
                    "query_id": "q1",
                    "query_tokens": torch.randn(5, 6),
                    "geometry_visible": torch.tensor([1.0, 1.0, 0.0]),
                    "solver_positive": torch.tensor([1.0, 0.0, 0.0]),
                    "harmful_negative": torch.tensor([0.0, 0.0, 1.0]),
                    "protected_support": torch.tensor([1.0, 0.0, 0.0]),
                }
            ],
            "split_audit": {"split_name": "selfmap_train", "test_split_used": False, "official_test_used": False},
        },
        cache,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_query_landmark_activation_v2",
            "--cache",
            str(cache),
            "--output_dir",
            str(tmp_path / "activation_v2"),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
