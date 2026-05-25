from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loc_gs.feedback.io import load_feedback_bank


_SYNTHETIC_QUERY_ID = re.compile(r"^(query|pair)_\d+$")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _record_missing(record: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if not _present(record.get(field))]


def _identity_base(value: str) -> str:
    text = str(value).strip()
    if text.startswith("rendered_from:"):
        text = text.split(":", 1)[1]
    return text.split("::", 1)[0]


def _synthetic_identity(value: str) -> bool:
    base = _identity_base(value)
    return bool(_SYNTHETIC_QUERY_ID.match(base))


def audit_feedback_bank_v2(path: str | Path) -> dict[str, Any]:
    """Audit whether a feedback bank is usable as query-level LSF supervision.

    This is intentionally stricter than the legacy feedback-bank loader. Older
    banks may remain useful diagnostics, but they should not pass this v2 audit
    unless they carry real query identity, split audit, and dense-stage outcome.
    """

    bank = load_feedback_bank(path)
    manifest = dict(bank.get("manifest", {}))
    records = [dict(record) for record in bank.get("records", [])]
    reasons: list[str] = []

    split_name = str(manifest.get("split_name", manifest.get("split", ""))).strip()
    if not split_name:
        reasons.append("missing split_name")
    elif split_name.lower() == "test":
        reasons.append("test split is not allowed")

    schema_version = str(manifest.get("schema_version", "")).strip()
    if schema_version != "feedback_bank_v2":
        reasons.append("schema_version must be feedback_bank_v2")

    query_id_source = str(manifest.get("query_id_source", "")).strip().lower()
    if query_id_source in {"", "synthetic", "synthetic_index"}:
        reasons.append("synthetic query_id source is not allowed")

    split_audit = manifest.get("split_audit")
    audit_status = split_audit.get("audit_status") if isinstance(split_audit, dict) else None
    if audit_status != "passed":
        reasons.append("split_audit.audit_status must be passed")

    required = (
        "scene",
        "query_id",
        "image_id",
        "keypoint_id",
        "matched_landmark_id",
        "matched_gaussian_id",
        "descriptor_score",
        "pnp_inlier",
        "reprojection_error_px",
    )
    query_ids: set[str] = set()
    image_ids: set[str] = set()
    for index, record in enumerate(records):
        query_id = str(record.get("query_id", ""))
        if query_id:
            query_ids.add(query_id)
        image_id = str(record.get("image_id", "")).strip()
        if image_id:
            image_ids.add(image_id)
        missing = _record_missing(record, required)
        if missing:
            reasons.append(f"record {index} missing required fields: {', '.join(missing)}")
        if _synthetic_identity(query_id):
            reasons.append(f"record {index} uses synthetic query_id: {query_id}")
        if _synthetic_identity(image_id):
            reasons.append(f"record {index} uses synthetic image_id: {image_id}")
        has_dense_transition = _present(record.get("dense_transition"))
        has_dense_delta = _present(record.get("dense_delta_te_cm"))
        if not has_dense_transition and not has_dense_delta:
            reasons.append(f"record {index} missing dense outcome")

    return {
        "format": "loc_gs_feedback_bank_v2_audit",
        "path": str(path),
        "audit_status": "passed" if not reasons else "failed",
        "paper_safe_candidate": not reasons,
        "record_count": int(len(records)),
        "query_count": int(len(query_ids)),
        "image_group_count": int(len(image_ids)),
        "split_name": split_name,
        "schema_version": schema_version,
        "query_id_source": query_id_source,
        "reasons": reasons,
    }
