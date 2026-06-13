import json
import subprocess
import sys

import torch

from loc_gs.scripts.build_active_fusion_plan_from_observation_cache import (
    build_active_fusion_plan_from_observation_cache,
)


def test_build_active_fusion_plan_from_observation_cache_keeps_weighted_pairs():
    cache = {
        "split_name": "train_selfmap",
        "observations": {
            "10": [
                {"gaussian_id": 10, "source_view_id": "view_a", "positive_weight": 1.0},
                {"gaussian_id": 10, "source_view_id": "view_b", "negative_weight": 0.5},
                {"gaussian_id": 10, "source_view_id": "view_c", "positive_weight": 0.0},
            ],
            "11": [{"source_view_id": "view_d", "positive_weight": 0.25}],
        },
    }

    plan, metrics = build_active_fusion_plan_from_observation_cache(
        cache,
        scene="ShopFacade",
        split_name="train_selfmap",
        min_abs_weight=0.1,
    )

    assert plan["landmark_fusion_plan"]["10"]["selected_view_ids"] == ["view_a", "view_b"]
    assert plan["landmark_fusion_plan"]["11"]["selected_view_ids"] == ["view_d"]
    assert metrics["input_pair_count"] == 4
    assert metrics["kept_pair_count"] == 3
    assert metrics["landmark_count"] == 2


def test_build_active_fusion_plan_from_observation_cache_rejects_test():
    try:
        build_active_fusion_plan_from_observation_cache(
            {"split_name": "test", "observations": {}},
            scene="ShopFacade",
            split_name="test",
        )
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("expected test split rejection")


def test_build_active_fusion_plan_from_observation_cache_cli_writes_audit(tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "split_name": "train_selfmap",
            "observations": {
                "10": [{"source_view_id": "view_a", "positive_weight": 1.0}],
            },
        },
        cache_path,
    )
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_active_fusion_plan_from_observation_cache",
            "--observation_cache",
            str(cache_path),
            "--output_dir",
            str(out),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_selfmap",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads((out / "landmark_fusion_plan.json").read_text(encoding="utf-8"))
    assert plan["landmark_fusion_plan"]["10"]["selected_view_ids"] == ["view_a"]
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["kept_pair_count"] == 1
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["test_split_used"] is False
    assert (out / "manifest.json").exists()
    assert (out / "command.txt").exists()
    assert (out / "git_status.txt").exists()
