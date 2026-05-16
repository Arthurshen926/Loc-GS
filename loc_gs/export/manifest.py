from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_status_text() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(_repo_root()),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"git status unavailable: {exc}\n"
    return result.stdout


def _as_id_set(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value) for value in values if str(value)}


def _check_status(checks: dict[str, dict[str, Any]]) -> str:
    statuses = {str(check.get("status", "unknown")) for check in checks.values()}
    if "failed" in statuses:
        return "failed"
    if "unknown" in statuses:
        return "unknown"
    return "passed"


def audit_split_usage(
    *,
    selfmap_image_ids: Iterable[str] | None,
    calibration_image_ids: Iterable[str] | None,
    test_image_ids: Iterable[str] | None,
    feedback_bank_manifest: dict[str, Any] | None,
    quality_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Audit split usage for native STDLoc-compatible export/eval artifacts."""

    checks: dict[str, dict[str, Any]] = {}
    selfmap_ids = _as_id_set(selfmap_image_ids)
    calibration_ids = _as_id_set(calibration_image_ids)
    test_ids = _as_id_set(test_image_ids)
    if selfmap_ids is None or calibration_ids is None or test_ids is None:
        checks["image_id_disjointness"] = {
            "status": "unknown",
            "reason": "selfmap, calibration, or test image ids are missing",
            "overlap": [],
        }
    else:
        overlap = sorted((selfmap_ids | calibration_ids) & test_ids)
        checks["image_id_disjointness"] = {
            "status": "failed" if overlap else "passed",
            "overlap": overlap,
        }

    feedback = dict(feedback_bank_manifest or {})
    split_name = str(feedback.get("split_name", feedback.get("split", ""))).strip()
    if not split_name:
        checks["feedback_bank_split"] = {
            "status": "unknown",
            "reason": "feedback bank split_name is missing",
        }
    elif split_name.lower() == "test":
        checks["feedback_bank_split"] = {
            "status": "failed",
            "split_name": split_name,
            "reason": "feedback bank split_name must not be test",
        }
    else:
        checks["feedback_bank_split"] = {"status": "passed", "split_name": split_name}

    gate = dict(quality_gate or {})
    mode = str(gate.get("mode", "")).strip()
    per_query = gate.get("per_query_branch_selection")
    if mode in {"per_query_branch_selection", "per-query-branch-selection", "query_branch_selection"} or per_query is True:
        checks["quality_gate"] = {
            "status": "failed",
            "mode": mode,
            "per_query_branch_selection": bool(per_query),
            "reason": "quality gate must not select branches per query",
        }
    elif not mode and per_query is None:
        checks["quality_gate"] = {
            "status": "unknown",
            "reason": "quality gate mode is missing",
        }
    else:
        checks["quality_gate"] = {
            "status": "passed",
            "mode": mode,
            "per_query_branch_selection": bool(per_query),
        }

    return {
        "audit_status": _check_status(checks),
        "checks": checks,
    }


def _command_text(command: list[str] | tuple[str, ...] | str) -> str:
    if isinstance(command, str):
        return command.rstrip() + "\n"
    return " ".join(str(part) for part in command).rstrip() + "\n"


def write_export_eval_audit_bundle(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    command: list[str] | tuple[str, ...] | str,
    metrics_summary: dict[str, Any],
    split_audit: dict[str, Any],
    git_diff_text: str | None = None,
) -> dict[str, str]:
    """Write the required audit files for a paper-facing export/eval artifact."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_payload = dict(manifest)
    manifest_payload.setdefault("git_commit", _git_commit())
    manifest_payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    if isinstance(command, str):
        manifest_payload["command"] = command.rstrip()
    else:
        manifest_payload["command"] = [str(part) for part in command]
    paths = {
        "manifest": root / "manifest.json",
        "command": root / "command.txt",
        "metrics_summary": root / "metrics_summary.json",
        "split_audit": root / "split_audit.json",
    }
    paths["manifest"].write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    paths["command"].write_text(_command_text(command), encoding="utf-8")
    paths["metrics_summary"].write_text(json.dumps(metrics_summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["split_audit"].write_text(json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8")
    if git_diff_text is not None:
        diff_path = root / "git_diff.patch"
        diff_path.write_text(str(git_diff_text), encoding="utf-8")
        paths["git_diff"] = diff_path
    else:
        status_path = root / "git_status.txt"
        status_path.write_text(_git_status_text(), encoding="utf-8")
        paths["git_status"] = status_path
    return {key: str(value) for key, value in paths.items()}
