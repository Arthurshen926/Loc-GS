from pathlib import Path

import pytest
import torch

from loc_gs.scripts.train_internal_sparse_students import _write_jsonl
from loc_gs.simulation.query_sampler import SimulatedQuerySpec
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.teacher.distillation_artifact import SolverFeedbackRow
from loc_gs.teacher.online_episode import (
    OnlineEpisodeConfig,
    build_online_sparse_dense_episodes,
    summarize_online_sparse_dense_episodes,
)


def _write_pair_cache(path: Path) -> Path:
    payload = {
        "metadata": {"format": "listwise", "scene": "GreatCourt", "source_split_name": "train", "topk": 2},
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64),
        "cosine": torch.tensor([[0.90, 0.10], [0.20, 0.80], [0.70, 0.30]], dtype=torch.float32),
        "label": torch.tensor([0, 1, 1], dtype=torch.int64),
        "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
        "query_id": ["a.png::kp0", "a.png::kp1", "b.png::kp0"],
        "image_id": ["a.png", "a.png", "b.png"],
        "keypoint_id": ["kp0", "kp1", "kp0"],
        "source_phase": ["train", "train", "train"],
    }
    torch.save(payload, path)
    return path


def _spec(idx: int, source: str, split_name: str = "train") -> SimulatedQuerySpec:
    return SimulatedQuerySpec(
        scene="GreatCourt",
        split_name=split_name,
        synthetic_query_id=f"sim/GreatCourt/{split_name}/{idx:06d}",
        source_image_id=source,
        translation_delta_m=(0.0, 0.0, 0.0),
        rotation_delta_deg=(0.0, 0.0, 0.0),
    )


def test_online_sparse_dense_episodes_bind_simulated_queries_to_candidates_and_feedback(tmp_path: Path):
    artifact = load_listwise_candidate_artifact(_write_pair_cache(tmp_path / "pairs.pt"))
    episodes = build_online_sparse_dense_episodes(
        [_spec(0, "a.png"), _spec(1, "missing.png"), _spec(2, "b.png")],
        artifact,
        [
            SolverFeedbackRow("GreatCourt", "train", "a.png", dense_helped=True, distill_weight=0.5),
            SolverFeedbackRow("GreatCourt", "train", "b.png", dense_helped=False, distill_weight=0.0),
        ],
        cfg=OnlineEpisodeConfig(max_keypoints_per_episode=1),
    )

    assert [episode.synthetic_query_id for episode in episodes] == [
        "sim/GreatCourt/train/000000",
        "sim/GreatCourt/train/000002",
    ]
    assert [episode.source_image_id for episode in episodes] == ["a.png", "b.png"]
    assert episodes[0].candidate_keypoint_count == 1
    assert episodes[0].dense_helped is True
    assert episodes[0].distill_weight == 0.5
    summary = summarize_online_sparse_dense_episodes(episodes)
    assert summary["online_episode_count"] == 2
    assert summary["missing_candidate_count"] == 1
    assert summary["dense_helped_episode_count"] == 1


def test_online_sparse_dense_episodes_reject_test_split(tmp_path: Path):
    artifact = load_listwise_candidate_artifact(_write_pair_cache(tmp_path / "pairs.pt"))

    with pytest.raises(ValueError, match="test split"):
        build_online_sparse_dense_episodes([_spec(0, "a.png", split_name="test")], artifact, [])


def test_online_episode_jsonl_writer_uses_internal_schema(tmp_path: Path):
    artifact = load_listwise_candidate_artifact(_write_pair_cache(tmp_path / "pairs.pt"))
    episodes = build_online_sparse_dense_episodes([_spec(0, "a.png")], artifact, [])
    path = tmp_path / "episodes.jsonl"

    _write_jsonl(path, [episode.to_json_dict() for episode in episodes])

    assert '"schema_version": "internal_online_sparse_dense_episode_v1"' in path.read_text(encoding="utf-8")
