from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split


def merge_listwise_candidate_artifacts(
    *,
    input_artifacts: Sequence[str | Path],
    output_artifact: str | Path,
    scene: str,
    split_name: str,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate artifact merge")
    paths = [Path(path) for path in input_artifacts]
    if not paths:
        raise ValueError("at least one input artifact is required")
    payloads = [_load_torch_payload(path) for path in paths]
    artifacts = [load_listwise_candidate_artifact(path) for path in paths]
    for path, payload in zip(paths, payloads):
        fmt = _artifact_format(payload)
        if fmt != "listwise":
            raise ValueError(f"unsupported candidate artifact format for merge: {fmt} ({path})")
    row_counts = [_row_count(payload, path) for payload, path in zip(payloads, paths)]
    _reject_duplicate_query_keypoints(payloads)
    _validate_base_fields(payloads, paths)
    row_fields = _common_row_fields(payloads, row_counts)
    required = {"query_yx", "landmark_id", "cosine", "label", "query_id"}
    missing_required = sorted(required - set(row_fields))
    if missing_required:
        raise ValueError(f"input artifacts are missing required row fields for merge: {missing_required}")
    merged: dict[str, Any] = {}
    for key in sorted(row_fields):
        merged[key] = _concat_values(key, [payload[key] for payload in payloads])
    for key in ("base_gaussian_id", "base_landmark_desc"):
        if key in payloads[0]:
            merged[key] = payloads[0][key]
    split_audit = _build_split_audit(artifacts, split_name=split)
    metadata = _merged_metadata(payloads, artifacts, paths, scene=str(scene), split_name=split, split_audit=split_audit)
    merged["metadata"] = metadata
    output_path = Path(output_artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_torch_payload(merged, output_path)
    merged_artifact = load_listwise_candidate_artifact(output_path)
    return {
        "schema_version": "internal_candidate_artifact_merge_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "output_artifact": str(output_path),
        "input_artifact_count": int(len(paths)),
        "input_artifacts": [str(path) for path in paths],
        "row_count": int(sum(row_counts)),
        "query_count": int(merged_artifact.batch_count),
        "keypoint_count": int(merged_artifact.keypoint_count),
        "topk": int(merged_artifact.topk),
        "artifact_format": "listwise",
        "split_audit_status": str(split_audit["audit_status"]),
    }


def _load_torch_payload(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is present in the project environment.
        raise RuntimeError("torch is required to read cached candidate .pt artifacts") from exc
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate artifact must contain a mapping payload: {path}")
    return payload


def _save_torch_payload(payload: Mapping[str, Any], path: Path) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is present in the project environment.
        raise RuntimeError("torch is required to write cached candidate .pt artifacts") from exc
    torch.save(dict(payload), path)


def _artifact_format(payload: Mapping[str, Any]) -> str:
    meta = payload.get("metadata", {})
    if isinstance(meta, Mapping):
        return str(meta.get("format") or payload.get("format") or "listwise")
    return str(payload.get("format") or "listwise")


def _row_count(payload: Mapping[str, Any], path: Path) -> int:
    if "query_yx" not in payload:
        raise ValueError(f"candidate artifact is missing query_yx: {path}")
    return _length(payload["query_yx"])


def _length(value: Any) -> int:
    if hasattr(value, "shape"):
        return int(value.shape[0])
    return len(value)


def _is_row_field(value: Any, row_count: int) -> bool:
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return False
    try:
        return _length(value) == int(row_count)
    except TypeError:
        return False


def _common_row_fields(payloads: Sequence[Mapping[str, Any]], row_counts: Sequence[int]) -> set[str]:
    common = set(payloads[0].keys())
    for payload in payloads[1:]:
        common &= set(payload.keys())
    common.discard("metadata")
    common.discard("base_gaussian_id")
    common.discard("base_landmark_desc")
    return {
        key
        for key in common
        if all(_is_row_field(payload[key], row_count) for payload, row_count in zip(payloads, row_counts))
    }


def _concat_values(key: str, values: Sequence[Any]) -> Any:
    first = values[0]
    if hasattr(first, "detach"):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is present in the project environment.
            raise RuntimeError("torch is required to merge cached candidate tensors") from exc
        return torch.cat([value.detach().cpu() if hasattr(value, "detach") else value for value in values], dim=0)
    if isinstance(first, list):
        merged: list[Any] = []
        for value in values:
            if not isinstance(value, list):
                raise ValueError(f"row field type mismatch for {key}")
            merged.extend(value)
        return merged
    if isinstance(first, tuple):
        merged_tuple: list[Any] = []
        for value in values:
            merged_tuple.extend(list(value))
        return merged_tuple
    raise ValueError(f"unsupported row field type for {key}: {type(first).__name__}")


def _query_ids(payload: Mapping[str, Any]) -> list[str]:
    if "query_id" not in payload:
        raise ValueError("candidate artifact is missing query_id")
    values = payload["query_id"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [str(value) for value in values]


def _reject_duplicate_query_keypoints(payloads: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(query_id for payload in payloads for query_id in _query_ids(payload))
    duplicates = sorted(query_id for query_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"duplicate query keypoint ids in candidate artifact merge: {preview}")


def _validate_base_fields(payloads: Sequence[Mapping[str, Any]], paths: Sequence[Path]) -> None:
    if "base_gaussian_id" not in payloads[0]:
        raise ValueError(f"candidate artifact is missing base_gaussian_id: {paths[0]}")
    base = payloads[0]["base_gaussian_id"]
    for payload, path in zip(payloads[1:], paths[1:]):
        if "base_gaussian_id" not in payload:
            raise ValueError(f"candidate artifact is missing base_gaussian_id: {path}")
        other = payload["base_gaussian_id"]
        if hasattr(base, "shape") and hasattr(other, "shape") and tuple(base.shape) != tuple(other.shape):
            raise ValueError(f"base_gaussian_id shape mismatch for candidate artifact: {path}")
        if hasattr(base, "detach") and hasattr(other, "detach"):
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("torch is required to compare base_gaussian_id tensors") from exc
            if not bool(torch.equal(base.detach().cpu(), other.detach().cpu())):
                raise ValueError(f"base_gaussian_id values mismatch for candidate artifact: {path}")
    if "base_landmark_desc" not in payloads[0]:
        return
    base_desc = payloads[0]["base_landmark_desc"]
    for payload, path in zip(payloads[1:], paths[1:]):
        if "base_landmark_desc" not in payload:
            continue
        other = payload["base_landmark_desc"]
        if hasattr(base_desc, "shape") and hasattr(other, "shape") and tuple(base_desc.shape) != tuple(other.shape):
            raise ValueError(f"base_landmark_desc shape mismatch for candidate artifact: {path}")


def _build_split_audit(artifacts: Sequence[Any], *, split_name: str) -> dict[str, object]:
    input_statuses: list[dict[str, object]] = []
    all_passed = True
    for artifact in artifacts:
        audit = artifact.metadata.get("split_audit")
        if isinstance(audit, Mapping):
            status = str(audit.get("audit_status", "unknown"))
        else:
            status = "unknown"
        if status != "passed":
            all_passed = False
        input_statuses.append(
            {
                "source_path": artifact.source_path,
                "split_name": artifact.split_name,
                "audit_status": status,
            }
        )
    return {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed" if all_passed else "unknown",
        "split_name": split_name,
        "official_test_used": False,
        "test_split_used": False,
        "input_artifact_audits": input_statuses,
    }


def _merged_metadata(
    payloads: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Any],
    paths: Sequence[Path],
    *,
    scene: str,
    split_name: str,
    split_audit: Mapping[str, object],
) -> dict[str, object]:
    first_meta = payloads[0].get("metadata", {})
    topk = artifacts[0].topk if artifacts else 0
    metadata: dict[str, object] = dict(first_meta) if isinstance(first_meta, Mapping) else {}
    metadata.update(
        {
            "format": "listwise",
            "scene": scene,
            "source_split_name": split_name,
            "topk": int(topk),
            "source": "internal_candidate_artifact_merge",
            "source_artifact_count": int(len(paths)),
            "source_artifacts": [str(path) for path in paths],
            "source_artifact_splits": [artifact.split_name for artifact in artifacts],
            "split_audit": dict(split_audit),
        }
    )
    return metadata
