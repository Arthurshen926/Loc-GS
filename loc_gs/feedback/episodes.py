from __future__ import annotations

from typing import Any, Iterable

from loc_gs.feedback.schema import FeedbackEpisode, FeedbackMatchRecord, FeedbackPoseRecord


def flatten_episode_matches(
    episodes: Iterable[FeedbackEpisode | dict[str, Any]],
) -> list[FeedbackMatchRecord]:
    records: list[FeedbackMatchRecord] = []
    for item in episodes:
        episode = FeedbackEpisode.from_mapping(item)
        for match in episode.matches:
            if not match.scene:
                match.scene = episode.scene
            if not match.query_id:
                match.query_id = episode.query_id
            if not match.source_view_id:
                match.source_view_id = episode.source_view_id
            if not match.pose_source:
                match.pose_source = episode.pose_source
            records.append(match)
    return records


def make_episode(
    *,
    scene: str,
    query_id: str,
    source_view_id: str = "",
    pose_source: str = "selfmap",
    split_name: str = "",
    matches: Iterable[FeedbackMatchRecord | dict[str, Any]] = (),
    pose: FeedbackPoseRecord | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FeedbackEpisode:
    return FeedbackEpisode(
        scene=scene,
        query_id=query_id,
        source_view_id=source_view_id,
        pose_source=pose_source,
        split_name=split_name,
        matches=[FeedbackMatchRecord.from_mapping(record) for record in matches],
        pose=FeedbackPoseRecord.from_mapping(pose) if pose is not None else None,
        metadata=dict(metadata or {}),
    )
