from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class CandidateCoverageDetails:
    requested_query_ids: list[str]
    artifact_query_ids: list[str]
    covered_query_ids: list[str]
    missing_query_ids: list[str]


def audit_candidate_artifact_coverage(
    *,
    candidate_artifact: str | Path,
    scene: str,
    split_name: str,
    requested_query_ids: Sequence[str],
) -> tuple[dict[str, object], CandidateCoverageDetails]:
    split = reject_test_split(split_name, purpose="internal candidate coverage audit")
    requested = _dedupe_preserving_order(str(query_id) for query_id in requested_query_ids)
    artifact = load_listwise_candidate_artifact(candidate_artifact)
    artifact_query_ids = [str(batch.query_id) for batch in artifact.batches]
    artifact_set = set(artifact_query_ids)
    covered = [query_id for query_id in requested if query_id in artifact_set]
    missing = [query_id for query_id in requested if query_id not in artifact_set]
    requested_count = len(requested)
    coverage_ratio = float(len(covered) / requested_count) if requested_count else 0.0
    split_audit = artifact.metadata.get("split_audit")
    split_audit_status = (
        str(split_audit.get("audit_status", "unknown")) if isinstance(split_audit, Mapping) else "unknown"
    )
    summary: dict[str, object] = {
        "schema_version": "internal_candidate_coverage_v1",
        "scene": str(scene),
        "split_name": split,
        "candidate_artifact": str(candidate_artifact),
        "artifact_scene": artifact.scene,
        "artifact_split_name": artifact.split_name,
        "artifact_format": artifact.artifact_format,
        "artifact_topk": int(artifact.topk),
        "artifact_query_count": int(len(artifact_query_ids)),
        "artifact_keypoint_count": int(artifact.keypoint_count),
        "requested_query_count": int(requested_count),
        "covered_query_count": int(len(covered)),
        "missing_query_count": int(len(missing)),
        "coverage_ratio": coverage_ratio,
        "complete_coverage": bool(requested_count > 0 and len(missing) == 0),
        "coverage_status": _coverage_status(requested_count=requested_count, missing_count=len(missing)),
        "covered_query_ids_preview": covered[:10],
        "missing_query_ids_preview": missing[:10],
        "split_audit_status": split_audit_status,
    }
    details = CandidateCoverageDetails(
        requested_query_ids=requested,
        artifact_query_ids=artifact_query_ids,
        covered_query_ids=covered,
        missing_query_ids=missing,
    )
    return summary, details


def load_requested_query_ids_from_results(path: str | Path) -> list[str]:
    rows = _load_result_rows(path)
    return _dedupe_preserving_order(_query_id(row) for row in rows)


def _dedupe_preserving_order(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = str(raw_value)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _coverage_status(*, requested_count: int, missing_count: int) -> str:
    if requested_count <= 0:
        return "empty_request"
    if missing_count == 0:
        return "complete"
    return "partial"


def _load_result_rows(path: str | Path) -> list[Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError(f"results file must contain a list or rows list: {path}")
    return [row for row in rows if isinstance(row, Mapping)]


def _query_id(row: Mapping[str, Any]) -> str:
    for key in ("query_id", "image_id", "image_name"):
        if key in row:
            return str(row[key])
    raise ValueError(f"result row is missing query id field: {row}")
