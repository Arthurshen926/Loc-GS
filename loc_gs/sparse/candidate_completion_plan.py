from __future__ import annotations

from pathlib import Path
from typing import Sequence

from loc_gs.sparse.audit import reject_test_split


def build_candidate_completion_plan(
    *,
    scene: str,
    split_name: str,
    missing_query_ids: Sequence[str],
    shard_size: int,
    base_candidate_artifact: str | Path | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = reject_test_split(split_name, purpose="internal candidate completion plan")
    if int(shard_size) <= 0:
        raise ValueError("shard_size must be positive")
    missing = _dedupe_preserving_order(missing_query_ids)
    shards: list[dict[str, object]] = []
    for shard_idx, start in enumerate(range(0, len(missing), int(shard_size))):
        query_ids = missing[start : start + int(shard_size)]
        shards.append(
            {
                "schema_version": "internal_candidate_completion_shard_v1",
                "shard_id": f"candidate_completion_shard_{shard_idx:06d}",
                "scene": str(scene),
                "split_name": split,
                "query_ids": query_ids,
                "query_count": int(len(query_ids)),
                "candidate_backend": "internal_sparse_pipeline",
                "role": "candidate_generation_shard",
                "base_candidate_artifact": None if base_candidate_artifact is None else str(base_candidate_artifact),
                "dense_teacher_enabled": False,
                "dense_inference_enabled": False,
                "external_runtime_dependency": "forbidden",
            }
        )
    summary = {
        "schema_version": "internal_candidate_completion_plan_v1",
        "scene": str(scene),
        "split_name": split,
        "missing_query_count": int(len(missing)),
        "shard_count": int(len(shards)),
        "shard_size": int(shard_size),
        "completion_required": bool(len(missing) > 0),
        "candidate_backend": "internal_sparse_pipeline",
        "base_candidate_artifact": None if base_candidate_artifact is None else str(base_candidate_artifact),
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }
    return summary, shards


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = str(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
