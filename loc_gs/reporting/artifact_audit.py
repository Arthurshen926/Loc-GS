from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from loc_gs.export.manifest import write_export_eval_audit_bundle


def _walk_mappings(value: Any, *, depth: int = 0) -> list[Mapping[str, Any]]:
    if depth > 4 or not isinstance(value, Mapping):
        return []
    items: list[Mapping[str, Any]] = [value]
    for nested in value.values():
        if isinstance(nested, Mapping):
            items.extend(_walk_mappings(nested, depth=depth + 1))
    return items


def _find_upstream_split_audit(*sources: Mapping[str, Any] | None) -> dict[str, Any] | None:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for mapping in _walk_mappings(source):
            for key in ("split_audit", "audit"):
                audit = mapping.get(key)
                if isinstance(audit, Mapping) and "audit_status" in audit:
                    return dict(audit)
            if "audit_status" in mapping and (
                "checks" in mapping or "schema_version" in mapping or "paper_safe_candidate" in mapping
            ):
                return dict(mapping)
    return None


def _find_split_name(*sources: Mapping[str, Any] | None) -> str:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for mapping in _walk_mappings(source):
            split = str(mapping.get("split_name", mapping.get("split", ""))).strip()
            if split:
                return split
    return ""


def artifact_split_audit(
    *sources: Mapping[str, Any] | None,
    branch_selection: bool | None = None,
) -> dict[str, Any]:
    """Build a compact split/provenance audit for non-eval LSF artifacts."""

    upstream = _find_upstream_split_audit(*sources)
    split_name = _find_split_name(*sources)
    checks: dict[str, dict[str, Any]] = {}
    if upstream is None:
        if not split_name:
            checks["feedback_bank_split"] = {
                "status": "unknown",
                "reason": "split_name is missing",
            }
        elif split_name.lower() == "test":
            checks["feedback_bank_split"] = {
                "status": "failed",
                "split_name": split_name,
                "reason": "feedback-derived artifacts must not use test split",
            }
        else:
            checks["feedback_bank_split"] = {
                "status": "unknown",
                "split_name": split_name,
                "reason": "split is non-test but no upstream split audit was found",
            }
    else:
        checks["upstream_split_audit"] = {
            "status": str(upstream.get("audit_status", "unknown")),
            "split_name": str(upstream.get("split_name", split_name)),
            "source": "nested_manifest",
        }
    if branch_selection is not None:
        checks["branch_selection"] = {
            "status": "failed" if bool(branch_selection) else "passed",
            "branch_selection": bool(branch_selection),
        }

    statuses = {str(check.get("status", "unknown")) for check in checks.values()}
    if "failed" in statuses:
        status = "failed"
    elif "unknown" in statuses:
        status = "unknown"
    else:
        status = "passed"
    return {
        "audit_status": status,
        "checks": checks,
        "upstream_audit": upstream or {},
    }


def write_artifact_audit_bundle(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    command: str,
    metrics_summary: dict[str, Any],
    split_audit: dict[str, Any],
) -> dict[str, str]:
    paths = write_export_eval_audit_bundle(
        output_dir,
        manifest=manifest,
        command=command,
        metrics_summary=metrics_summary,
        split_audit=split_audit,
    )
    audit_index = {
        "artifact_audit_bundle": "loc_gs_artifact_audit_bundle_v1",
        "paths": paths,
    }
    Path(output_dir, "artifact_audit.json").write_text(
        json.dumps(audit_index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths
