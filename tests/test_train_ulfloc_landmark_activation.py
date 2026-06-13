import json
import subprocess
import sys

import torch

from loc_gs.scripts.train_ulfloc_landmark_activation import activation_loss


def _write_cache(path, *, split_name="train_selfmap"):
    torch.save(
        {
            "schema_version": "ulfloc_landmark_activation_cache_v1",
            "scene": "ShopFacade",
            "split_name": split_name,
            "query_dim": 2,
            "landmark_dim": 3,
            "top_n": 2,
            "landmark_ids": torch.tensor([10, 20, 30], dtype=torch.long),
            "landmark_features": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            "safe_core_ids": torch.tensor([10], dtype=torch.long),
            "safe_core_indices": torch.tensor([0], dtype=torch.long),
            "examples": [
                {
                    "query_id": "q1",
                    "query_global_feature": torch.tensor([1.0, 0.0], dtype=torch.float32),
                    "positive_ids": torch.tensor([0], dtype=torch.long),
                    "negative_ids": torch.tensor([2], dtype=torch.long),
                    "positive_landmark_ids": torch.tensor([10], dtype=torch.long),
                    "negative_landmark_ids": torch.tensor([30], dtype=torch.long),
                },
                {
                    "query_id": "q2",
                    "query_global_feature": torch.tensor([0.0, 1.0], dtype=torch.float32),
                    "positive_ids": torch.tensor([1], dtype=torch.long),
                    "negative_ids": torch.tensor([2], dtype=torch.long),
                    "positive_landmark_ids": torch.tensor([20], dtype=torch.long),
                    "negative_landmark_ids": torch.tensor([30], dtype=torch.long),
                },
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


def test_activation_loss_adds_pairwise_margin_when_negative_beats_positive():
    good_scores = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    bad_scores = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float32)
    positive = torch.tensor([0], dtype=torch.long)
    negative = torch.tensor([1], dtype=torch.long)

    good_loss = activation_loss(good_scores, positive, negative, margin=0.5)
    bad_loss = activation_loss(bad_scores, positive, negative, margin=0.5)

    assert bad_loss > good_loss


def test_train_ulfloc_landmark_activation_cli_writes_checkpoint_and_audit_bundle(tmp_path):
    cache = tmp_path / "cache.pt"
    _write_cache(cache)
    output = tmp_path / "activation"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_landmark_activation",
            "--cache",
            str(cache),
            "--output_dir",
            str(output),
            "--epochs",
            "2",
            "--lr",
            "0.01",
            "--hidden_dim",
            "8",
            "--top_n",
            "2",
            "--seed",
            "7",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    checkpoint = torch.load(output / "landmark_activation.pth", map_location="cpu")
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))

    assert checkpoint["query_dim"] == 2
    assert checkpoint["landmark_dim"] == 3
    assert checkpoint["top_n"] == 2
    assert checkpoint["safe_core_ids"].tolist() == [10]
    assert checkpoint["source_cache_path"] == str(cache)
    assert checkpoint["split_name"] == "train_selfmap"
    assert "state_dict" in checkpoint
    assert metrics["epochs"] == 2
    assert metrics["trained_example_count"] == 2
    assert manifest["source_cache_path"] == str(cache)
    assert split_audit["test_split_used"] is False
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()


def test_train_ulfloc_landmark_activation_cli_rejects_test_cache(tmp_path):
    cache = tmp_path / "cache.pt"
    _write_cache(cache, split_name="test")
    output = tmp_path / "activation"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_landmark_activation",
            "--cache",
            str(cache),
            "--output_dir",
            str(output),
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
