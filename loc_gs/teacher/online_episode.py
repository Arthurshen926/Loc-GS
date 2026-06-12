from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from loc_gs.simulation.query_sampler import SimulatedQuerySpec
from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.rerank import summarize_candidate_availability
from loc_gs.teacher.distillation_artifact import SolverFeedbackRow


@dataclass(frozen=True)
class OnlineEpisodeConfig:
    max_keypoints_per_episode: int | None = None


@dataclass(frozen=True)
class OnlineSparseDenseEpisode:
    scene: str
    split_name: str
    episode_id: str
    synthetic_query_id: str
    source_image_id: str
    render_engine: str
    candidate_keypoint_count: int
    source_candidate_keypoint_count: int
    top1_correct: int
    topk_available: int
    oracle_gap: int
    feedback_matched: bool
    dense_helped: bool
    distill_weight: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_online_sparse_dense_episode_v1",
            **asdict(self),
        }


class OnlineEpisodeList(list[OnlineSparseDenseEpisode]):
    def __init__(self, values: Sequence[OnlineSparseDenseEpisode], *, missing_candidate_count: int) -> None:
        super().__init__(values)
        self.missing_candidate_count = int(missing_candidate_count)


def build_online_sparse_dense_episodes(
    specs: Sequence[SimulatedQuerySpec],
    artifact: CachedCandidateArtifact,
    feedback_rows: Sequence[SolverFeedbackRow],
    *,
    cfg: OnlineEpisodeConfig | None = None,
) -> OnlineEpisodeList:
    if cfg is None:
        cfg = OnlineEpisodeConfig()
    artifact_split = reject_test_split(artifact.split_name, purpose="online sparse-dense episode generation")
    batches_by_query = {batch.query_id: batch for batch in artifact.batches}
    feedback_by_query = {row.query_id: row for row in feedback_rows}
    for feedback in feedback_rows:
        reject_test_split(feedback.split_name, purpose="online sparse-dense episode generation")
        if feedback.scene not in {"unknown", artifact.scene}:
            raise ValueError(f"feedback scene mismatch: expected {artifact.scene}, got {feedback.scene}")

    episodes: list[OnlineSparseDenseEpisode] = []
    missing_candidate_count = 0
    for idx, spec in enumerate(specs):
        split = reject_test_split(spec.split_name, purpose="online sparse-dense episode generation")
        if spec.scene != artifact.scene:
            raise ValueError(f"simulation scene mismatch: expected {artifact.scene}, got {spec.scene}")
        batch = batches_by_query.get(spec.source_image_id)
        if batch is None:
            missing_candidate_count += 1
            continue
        limit = batch.keypoint_count
        if cfg.max_keypoints_per_episode is not None:
            limit = min(limit, max(0, int(cfg.max_keypoints_per_episode)))
        correct_rows = (batch.candidate_geometric_correct or [])[:limit]
        availability = summarize_candidate_availability(
            [[{"geometric_correct": bool(value)} for value in row] for row in correct_rows]
        )
        feedback = feedback_by_query.get(spec.source_image_id)
        episodes.append(
            OnlineSparseDenseEpisode(
                scene=artifact.scene,
                split_name=artifact_split if artifact_split != "unknown" else split,
                episode_id=f"episode/{idx:06d}",
                synthetic_query_id=spec.synthetic_query_id,
                source_image_id=spec.source_image_id,
                render_engine=spec.render_engine,
                candidate_keypoint_count=int(limit),
                source_candidate_keypoint_count=int(batch.keypoint_count),
                top1_correct=int(availability["top1_correct"]),
                topk_available=int(availability["topk_available"]),
                oracle_gap=int(availability["oracle_gap"]),
                feedback_matched=bool(feedback is not None),
                dense_helped=bool(feedback.dense_helped) if feedback is not None else False,
                distill_weight=float(feedback.distill_weight) if feedback is not None else 0.0,
            )
        )
    return OnlineEpisodeList(episodes, missing_candidate_count=missing_candidate_count)


def summarize_online_sparse_dense_episodes(episodes: Sequence[OnlineSparseDenseEpisode]) -> dict[str, object]:
    missing = int(getattr(episodes, "missing_candidate_count", 0))
    return {
        "schema_version": "internal_online_sparse_dense_episode_summary_v1",
        "online_episode_count": int(len(episodes)),
        "missing_candidate_count": missing,
        "candidate_keypoint_count": int(sum(episode.candidate_keypoint_count for episode in episodes)),
        "source_image_count": int(len({episode.source_image_id for episode in episodes})),
        "dense_helped_episode_count": int(sum(1 for episode in episodes if episode.dense_helped)),
        "feedback_matched_episode_count": int(sum(1 for episode in episodes if episode.feedback_matched)),
        "top1_correct": int(sum(episode.top1_correct for episode in episodes)),
        "topk_available": int(sum(episode.topk_available for episode in episodes)),
        "oracle_gap": int(sum(episode.oracle_gap for episode in episodes)),
        "render_engine": "3dgs",
    }


def build_online_candidate_payload(
    payload: Mapping[str, Any],
    episodes: Sequence[OnlineSparseDenseEpisode],
) -> dict[str, Any]:
    image_ids = _as_list(_required(payload, "image_id"))
    row_count = len(image_ids)
    rows_by_image: dict[str, list[int]] = {}
    for row_idx, image_id in enumerate(image_ids):
        rows_by_image.setdefault(str(image_id), []).append(row_idx)

    selected_indices: list[int] = []
    episode_ids: list[str] = []
    synthetic_query_ids: list[str] = []
    for episode in episodes:
        source_rows = rows_by_image.get(episode.source_image_id, [])
        for row_idx in source_rows[: int(episode.candidate_keypoint_count)]:
            selected_indices.append(row_idx)
            episode_ids.append(episode.episode_id)
            synthetic_query_ids.append(episode.synthetic_query_id)
    if not selected_indices:
        raise ValueError("online episodes did not select any candidate rows")

    selected = torch.as_tensor(selected_indices, dtype=torch.long)
    out: dict[str, Any] = {}
    for key, value in payload.items():
        out[key] = _select_rows(value, selected, selected_indices, row_count)
    out["online_episode_id"] = episode_ids
    out["synthetic_query_id"] = synthetic_query_ids
    metadata = dict(out.get("metadata", {}) if isinstance(out.get("metadata"), Mapping) else {})
    metadata.update(
        {
            "online_episode_count": int(len(episodes)),
            "online_candidate_row_count": int(len(selected_indices)),
            "training_source": "online_3dgs_sparse_dense_episode",
        }
    )
    out["metadata"] = metadata
    return out


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"candidate artifact is missing required field: {key}")
    return payload[key]


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _select_rows(value: Any, selected: torch.Tensor, selected_indices: Sequence[int], row_count: int) -> Any:
    if hasattr(value, "detach") and getattr(value, "shape", (None,))[0] == row_count:
        return value.detach().cpu().index_select(0, selected)
    if isinstance(value, list) and len(value) == row_count:
        return [value[idx] for idx in selected_indices]
    return value
