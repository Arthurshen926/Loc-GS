from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.candidate_shard_builder import build_candidate_shard_artifact


def build_candidate_shards_from_completion_plan(
    *,
    completion_plan: str | Path,
    query_feature_cache: str | Path,
    base_candidate_artifact: str | Path,
    output_dir: str | Path,
    scene: str,
    split_name: str,
    topk: int,
    max_landmarks: int | None = None,
    landmark_chunk_size: int | None = None,
    include_base_landmark_desc: bool = True,
    max_shards: int | None = None,
    skip_empty_shards: bool = False,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate completion runner")
    shards = load_completion_plan_shards(completion_plan)
    if max_shards is not None:
        shards = shards[: max(0, int(max_shards))]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_artifacts: list[str] = []
    shard_summaries: list[dict[str, object]] = []
    skipped_shards: list[dict[str, object]] = []
    keypoint_count = 0
    query_count = 0
    missing_feature_query_count = 0
    for index, shard in enumerate(shards):
        shard_id = str(shard.get("shard_id") or f"completion_shard_{index:06d}")
        shard_dir = out / "shards" / shard_id
        shard_artifact = shard_dir / "candidate_shard.pt"
        try:
            summary = build_candidate_shard_artifact(
                completion_shard=shard,
                query_feature_cache=query_feature_cache,
                base_candidate_artifact=base_candidate_artifact,
                output_artifact=shard_artifact,
                scene=str(scene),
                split_name=split,
                topk=int(topk),
                max_landmarks=max_landmarks,
                landmark_chunk_size=landmark_chunk_size,
                include_base_landmark_desc=bool(include_base_landmark_desc),
            )
        except ValueError as exc:
            if not skip_empty_shards or "does not contain any requested completion shard queries" not in str(exc):
                raise
            skipped_shards.append({"shard_id": shard_id, "reason": str(exc)})
            continue
        output_artifacts.append(str(shard_artifact))
        shard_summaries.append(summary)
        keypoint_count += int(summary.get("keypoint_count", 0))
        query_count += int(summary.get("query_count", 0))
        missing_feature_query_count += int(summary.get("missing_feature_query_count", 0))
    return {
        "schema_version": "internal_candidate_completion_runner_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "completion_plan": str(completion_plan),
        "query_feature_cache": str(query_feature_cache),
        "base_candidate_artifact": str(base_candidate_artifact),
        "output_dir": str(out),
        "plan_shard_count": int(len(shards)),
        "built_shard_count": int(len(output_artifacts)),
        "skipped_shard_count": int(len(skipped_shards)),
        "query_count": int(query_count),
        "keypoint_count": int(keypoint_count),
        "missing_feature_query_count": int(missing_feature_query_count),
        "topk": int(topk),
        "max_landmarks": None if max_landmarks is None else int(max_landmarks),
        "landmark_chunk_size": None if landmark_chunk_size is None else int(landmark_chunk_size),
        "base_landmark_desc_included": bool(include_base_landmark_desc),
        "max_shards": None if max_shards is None else int(max_shards),
        "skip_empty_shards": bool(skip_empty_shards),
        "output_artifacts": output_artifacts,
        "skipped_shards": skipped_shards,
        "shard_summaries": shard_summaries,
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }


def load_completion_plan_shards(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"completion plan row must be an object: {source}:{line_number}")
        rows.append(dict(payload))
    if not rows:
        raise ValueError(f"completion plan is empty: {source}")
    return rows
