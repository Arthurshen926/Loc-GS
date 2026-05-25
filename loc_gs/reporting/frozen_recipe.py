from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


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


def _third_party_stdloc_evaluator_modified() -> bool:
    evaluator_paths = [
        "third_party/stdloc/stdloc.py",
        "third_party/stdloc/utils",
    ]
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--", *evaluator_paths],
            cwd=str(_repo_root()),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _quality_gate_is_per_query(manifest: Mapping[str, Any]) -> bool:
    gate = manifest.get("quality_gate")
    if not isinstance(gate, Mapping):
        return False
    mode = str(gate.get("mode", "")).strip()
    return bool(gate.get("per_query_branch_selection")) or mode in {
        "per_query_branch_selection",
        "per-query-branch-selection",
        "query_branch_selection",
    }


def _selected_run_summary(scene: str, row: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(row.get("run_dir", "")))
    manifest = _load_json_if_exists(run_dir / "manifest.json")
    metrics = _load_json_if_exists(run_dir / "metrics_summary.json")
    split_audit = _load_json_if_exists(run_dir / "split_audit.json")
    dense = metrics.get("dense", {}) if isinstance(metrics.get("dense"), Mapping) else {}
    evaluator_safety = (
        manifest.get("evaluator_safety", {})
        if isinstance(manifest.get("evaluator_safety", {}), Mapping)
        else {}
    )
    return {
        "scene": scene,
        "selected": row.get("selected", "unknown"),
        "selection_reason": row.get("reason", ""),
        "run_dir": str(run_dir),
        "map_path": manifest.get("map_path", ""),
        "metrics_summary_path": str(run_dir / "metrics_summary.json"),
        "manifest_path": str(run_dir / "manifest.json"),
        "split_audit_path": str(run_dir / "split_audit.json"),
        "dense": dict(dense),
        "audit_status": split_audit.get("audit_status", "unknown"),
        "single_path_evaluator": bool(manifest.get("single_path_evaluator", False)),
        "single_path_deployment": bool(
            manifest.get("single_path_deployment", manifest.get("map_single_path_deployment", False))
        ),
        "per_query_branch_selection": _quality_gate_is_per_query(manifest),
        "feedback_enabled": bool(manifest.get("feedback_enabled", False)),
        "rho_feedback_enabled": bool(manifest.get("rho_feedback_enabled", False)),
        "same_budget": bool(manifest.get("same_budget", manifest.get("map_same_budget", False))),
        "evaluator_safety": dict(evaluator_safety),
        "official_stdloc_parity_candidate": bool(
            evaluator_safety.get("official_stdloc_parity_candidate", False)
        ),
        "paper_safe_fixed_poselib_candidate": bool(
            evaluator_safety.get("paper_safe_fixed_poselib_candidate", False)
        ),
        "run_role": manifest.get("run_role", ""),
    }


def _rejections(candidate_evaluations: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rejected: dict[str, list[dict[str, Any]]] = {}
    for scene, raw_candidates in candidate_evaluations.items():
        if not isinstance(raw_candidates, Mapping):
            continue
        scene_rejections: list[dict[str, Any]] = []
        for name, raw_eval in raw_candidates.items():
            if not isinstance(raw_eval, Mapping) or bool(raw_eval.get("passed", False)):
                continue
            scene_rejections.append(
                {
                    "candidate": str(name),
                    "reasons": list(raw_eval.get("reasons", [])),
                    "delta": dict(raw_eval.get("delta", {})),
                }
            )
        rejected[str(scene)] = scene_rejections
    return rejected


def build_frozen_recipe_manifest(
    acceptance_reports: Mapping[str, Mapping[str, Any]],
    *,
    freeze_id: str,
    notes: str = "",
    require_passed: bool = True,
) -> dict[str, Any]:
    """Build a fixed train-dev recipe manifest from acceptance reports.

    The manifest freezes scene-level map choices before any full/test evaluation.
    It does not authorize per-query branch selection.
    """

    splits: dict[str, Any] = {}
    gate_failures: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for label, payload in acceptance_reports.items():
        recipe = payload.get("recipe", {}) if isinstance(payload, Mapping) else {}
        global_gate = recipe.get("global_gate", {}) if isinstance(recipe, Mapping) else {}
        if require_passed and not bool(global_gate.get("passed", False)):
            gate_failures.append(str(label))
        selected = {
            scene: _selected_run_summary(scene, row)
            for scene, row in (recipe.get("selected_runs", {}) if isinstance(recipe, Mapping) else {}).items()
            if isinstance(row, Mapping)
        }
        selected_rows.extend(selected.values())
        splits[str(label)] = {
            "split": payload.get("split", label),
            "notes": payload.get("notes", ""),
            "passed": bool(global_gate.get("passed", False)),
            "macro_delta": dict(global_gate.get("macro_delta", {})),
            "thresholds": dict(recipe.get("thresholds", {})) if isinstance(recipe, Mapping) else {},
            "required_positive_scenes": list(recipe.get("required_positive_scenes", []))
            if isinstance(recipe, Mapping)
            else [],
            "neutral_scenes": list(recipe.get("neutral_scenes", [])) if isinstance(recipe, Mapping) else [],
            "candidate_order": list(recipe.get("candidate_order", [])) if isinstance(recipe, Mapping) else [],
            "selected_runs": selected,
            "rejected_candidates": _rejections(
                recipe.get("candidate_evaluations", {}) if isinstance(recipe, Mapping) else {}
            ),
        }

    selected_audit_statuses = [str(row.get("audit_status", "unknown")) for row in selected_rows]
    all_selected_audited = bool(selected_rows) and all(status == "passed" for status in selected_audit_statuses)
    all_single_path = bool(selected_rows) and all(bool(row.get("single_path_evaluator")) for row in selected_rows)
    selected_official_parity = [
        bool(row.get("official_stdloc_parity_candidate", False)) for row in selected_rows
    ]
    all_selected_official_parity = bool(selected_rows) and all(selected_official_parity)
    selected_paper_safe_poselib = [
        bool(row.get("paper_safe_fixed_poselib_candidate", False)) for row in selected_rows
    ]
    all_selected_paper_safe_poselib = bool(selected_rows) and all(selected_paper_safe_poselib)
    no_per_query_branch = all(not bool(row.get("per_query_branch_selection")) for row in selected_rows)
    no_feedback = all(
        not bool(row.get("feedback_enabled")) and not bool(row.get("rho_feedback_enabled"))
        for row in selected_rows
    )
    gates_passed = not gate_failures
    stdloc_evaluator_modified = _third_party_stdloc_evaluator_modified()

    return {
        "format": "loc_gs_frozen_scene_level_recipe_v1",
        "freeze_id": freeze_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "status": "frozen_train_dev_recipe",
        "selection_scope": "scene_level_map_acceptance_not_per_query",
        "real_query_path": "matching -> fixed cfg PnP solver -> STDLoc dense refinement",
        "splits": splits,
        "checks": {
            "acceptance_gates_passed": gates_passed,
            "gate_failures": gate_failures,
            "all_selected_runs_split_audit_passed": all_selected_audited,
            "selected_run_audit_statuses": selected_audit_statuses,
            "all_selected_runs_single_path": all_single_path,
            "all_selected_runs_official_stdloc_parity": all_selected_official_parity,
            "selected_run_official_stdloc_parity": selected_official_parity,
            "all_selected_runs_paper_safe_fixed_poselib": all_selected_paper_safe_poselib,
            "selected_run_paper_safe_fixed_poselib": selected_paper_safe_poselib,
            "no_per_query_branch_selection": no_per_query_branch,
            "feedback_disabled_at_eval": no_feedback,
            "third_party_stdloc_evaluator_modified": stdloc_evaluator_modified,
        },
        "paper_facing_ready": bool(
            gates_passed
            and all_selected_audited
            and all_single_path
            and all_selected_paper_safe_poselib
            and no_per_query_branch
            and no_feedback
            and not stdloc_evaluator_modified
        ),
        "notes": notes,
    }
