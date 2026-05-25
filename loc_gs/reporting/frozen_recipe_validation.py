from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.reporting.fixed_recipe_gate import evaluate_fixed_recipe_gate


def _dense(row: Mapping[str, Any]) -> Mapping[str, Any]:
    dense = row.get("dense", {})
    return dense if isinstance(dense, Mapping) else {}


def _manifest(row: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = row.get("manifest", {})
    return manifest if isinstance(manifest, Mapping) else {}


def _split_audit(row: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = row.get("split_audit", {})
    return audit if isinstance(audit, Mapping) else {}


def _quality_gate_is_per_query(manifest: Mapping[str, Any]) -> bool:
    gate = manifest.get("quality_gate", {})
    if not isinstance(gate, Mapping):
        return False
    mode = str(gate.get("mode", "")).strip()
    return bool(gate.get("per_query_branch_selection")) or mode in {
        "per_query_branch_selection",
        "per-query-branch-selection",
        "query_branch_selection",
    }


def _manifest_flag(manifest: Mapping[str, Any], *names: str) -> bool:
    for name in names:
        if bool(manifest.get(name)):
            return True
    feedback = manifest.get("feedback", {})
    if isinstance(feedback, Mapping):
        for name in names:
            feedback_name = name.removesuffix("_enabled")
            if bool(feedback.get(feedback_name)):
                return True
    return False


def _feedback_bank_split_check(split_audit: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = split_audit.get("checks", {})
    if not isinstance(checks, Mapping):
        return {}
    feedback_check = checks.get("feedback_bank_split", {})
    return feedback_check if isinstance(feedback_check, Mapping) else {}


def _selected_run_checks(selected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(selected.values())
    audit_statuses = [str(_split_audit(row).get("audit_status", "unknown")) for row in rows]
    splits = [str(_manifest(row).get("split", "")).strip() for row in rows]
    feedback_split_checks = [_feedback_bank_split_check(_split_audit(row)) for row in rows]
    feedback_split_statuses = [
        str(check.get("status", "unknown")).strip() or "unknown"
        for check in feedback_split_checks
    ]
    feedback_split_names = [
        str(check.get("split_name", "")).strip()
        for check in feedback_split_checks
    ]
    all_audited = bool(rows) and all(status == "passed" for status in audit_statuses)
    all_single_path = bool(rows) and all(bool(_manifest(row).get("single_path_evaluator")) for row in rows)
    no_per_query_branch = all(not _quality_gate_is_per_query(_manifest(row)) for row in rows)
    feedback_disabled = all(
        not _manifest_flag(
            _manifest(row),
            "feedback_enabled",
            "residual_enabled",
            "selector_enabled",
            "rho_feedback_enabled",
        )
        for row in rows
    )
    no_test_split = all(split != "test" for split in splits if split)
    feedback_bank_split_not_test = bool(rows) and all(
        status == "passed" and name.lower() != "test"
        for status, name in zip(feedback_split_statuses, feedback_split_names)
    )
    sidecar_status: dict[str, dict[str, bool]] = {}
    required_sidecars = ("manifest.json", "command.txt", "metrics_summary.json", "split_audit.json")
    for scene, row in selected.items():
        run_dir = Path(str(row.get("run_dir", "")))
        sidecar_status[str(scene)] = {
            name: (run_dir / name).exists()
            for name in required_sidecars
        }
        sidecar_status[str(scene)]["git_status_or_diff"] = (
            (run_dir / "git_status.txt").exists() or (run_dir / "git_diff.patch").exists()
        )
    all_sidecars = bool(sidecar_status) and all(
        all(present for present in scene_status.values())
        for scene_status in sidecar_status.values()
    )
    return {
        "no_reselection": True,
        "selected_run_audit_statuses": audit_statuses,
        "all_selected_runs_split_audit_passed": all_audited,
        "all_selected_runs_single_path": all_single_path,
        "no_per_query_branch_selection": no_per_query_branch,
        "feedback_disabled_at_eval": feedback_disabled,
        "selected_run_splits": splits,
        "selected_run_split_not_test": no_test_split,
        "feedback_bank_split_statuses": feedback_split_statuses,
        "feedback_bank_split_names": feedback_split_names,
        "feedback_bank_split_not_test": feedback_bank_split_not_test,
        "selected_run_sidecars": sidecar_status,
        "all_selected_run_sidecars_present": all_sidecars,
    }


def build_frozen_recipe_validation_report(
    baseline: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, Mapping[str, Any]],
    *,
    required_positive_scenes: Sequence[str] = (),
    neutral_scenes: Sequence[str] = (),
    max_median_te_regression_cm: float = 0.0,
    max_recall_drop: float = 0.0,
    split: str = "train",
    notes: str = "",
) -> dict[str, Any]:
    """Validate a previously frozen scene-level recipe without reselecting maps."""

    gate = evaluate_fixed_recipe_gate(
        baseline,
        selected,
        required_positive_scenes=required_positive_scenes,
        neutral_scenes=neutral_scenes,
        max_median_te_regression_cm=max_median_te_regression_cm,
        max_recall_drop=max_recall_drop,
    )
    checks = _selected_run_checks(selected)
    checks["paper_safe_validation_ready"] = bool(
        gate["passed"]
        and checks["all_selected_runs_split_audit_passed"]
        and checks["all_selected_runs_single_path"]
        and checks["no_per_query_branch_selection"]
        and checks["feedback_disabled_at_eval"]
        and checks["feedback_bank_split_not_test"]
        and checks["all_selected_run_sidecars_present"]
    )
    scenes = sorted(set(baseline) | set(selected) | set(required_positive_scenes) | set(neutral_scenes))
    return {
        "format": "loc_gs_frozen_recipe_validation_v1",
        "policy": "frozen_recipe_validation_no_reselection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "notes": notes,
        "baseline_scenes": sorted(baseline),
        "selected_scenes": sorted(selected),
        "runs": {
            scene: {
                "baseline": baseline.get(scene, {}),
                "selected": selected.get(scene, {}),
                "dense_delta": gate["scene_results"].get(scene, {}).get("delta", {}),
            }
            for scene in scenes
        },
        "global_gate": gate,
        "checks": checks,
        "paper_safety_note": (
            "This validates a pre-frozen scene-level recipe only; it does not "
            "perform per-query or post-hoc scene reselection."
        ),
    }
