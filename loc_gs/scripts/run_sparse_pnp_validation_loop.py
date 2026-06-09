#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.scripts.export_ulfloc_sparse_feedback import _git_commit, _git_status
from loc_gs.stdloc_native.sparse_pnp_validation import build_sparse_pnp_validation_profile

_DEFAULT_PYTHON = "/root/miniconda3/envs/cybersim_agent/bin/python"


def _normalize_resample_invocation(parts: list[str]) -> list[str]:
    if not parts:
        return parts
    script = "loc_gs/scripts/resample_ulfloc_with_solver_feedback.py"
    module = "loc_gs.scripts.resample_ulfloc_with_solver_feedback"
    first = parts[0]
    if first.endswith(script):
        return [_DEFAULT_PYTHON, "-m", module, *parts[1:]]
    if len(parts) >= 2 and parts[1].endswith(script):
        return [parts[0], "-m", module, *parts[2:]]
    return parts


def _replace_or_append_option(parts: list[str], option: str, value: str) -> list[str]:
    out = list(parts)
    if option in out:
        index = out.index(option)
        if index + 1 >= len(out):
            raise ValueError(f"option {option} is missing a value")
        out[index + 1] = str(value)
        return out
    out.extend([option, str(value)])
    return out


def _append_flag(parts: list[str], option: str) -> list[str]:
    out = list(parts)
    if option not in out:
        out.append(option)
    return out


def _remove_flag(parts: list[str], option: str) -> list[str]:
    return [part for part in parts if part != option]


def _option_value(parts: Sequence[str], option: str) -> str | None:
    if option not in parts:
        return None
    index = list(parts).index(option)
    if index + 1 >= len(parts):
        raise ValueError(f"option {option} is missing a value")
    return str(parts[index + 1])


def _apply_regression_repair_resample_options(parts: list[str]) -> list[str]:
    """Make the next sparse-set export guard real PnP regressions.

    The validation loop should not become a replacement sampler that throws away
    the K.C./visibility distribution that made the sparse baseline strong.  It
    therefore keeps all baseline-good protected support, applies regression risk
    as a soft guard, and preserves the full landmark budget when a previous
    command had enabled no-query final pruning.
    """

    out = list(parts)
    forced_options = {
        "--full_set_sparse_validation_support_scope": "all",
        "--full_set_validation_query_prefill_target_fraction": "1.0",
        "--full_set_validation_query_prefill_max_candidates_per_query": "256",
        "--full_set_sparse_validation_risk_reject_threshold": "0.0",
        "--full_set_sparse_validation_risk_weight": "0.25",
    }
    for option, value in forced_options.items():
        out = _replace_or_append_option(out, option, value)
    out = _append_flag(out, "--full_set_validation_query_prefill_force_all")
    max_landmarks = _option_value(out, "--full_set_max_landmarks")
    if max_landmarks is not None:
        try:
            max_count = int(max_landmarks)
        except ValueError:
            max_count = 0
        if max_count > 0:
            out = _replace_or_append_option(out, "--full_set_final_prune_min_keep_count", str(max_count))
    return out


def build_next_resample_command(
    command: str | Sequence[str],
    *,
    next_output_model_path: str | Path,
    sparse_validation_profile: str | Path,
    repair_regressions: bool = False,
) -> str:
    if isinstance(command, str):
        parts = shlex.split(command)
    else:
        parts = [str(part) for part in command]
    if not parts:
        raise ValueError("resample command must not be empty")
    parts = _normalize_resample_invocation(parts)
    parts = _replace_or_append_option(parts, "--selection_policy", "full_gaussian_sparse_set")
    parts = _remove_flag(parts, "--full_set_allow_metric_only_sparse_validation_profile")
    parts = _replace_or_append_option(parts, "--full_set_source_anchor_mode", "off")
    parts = _replace_or_append_option(parts, "--full_set_source_anchor_weight", "0.0")
    parts = _replace_or_append_option(parts, "--full_set_kc_anchor_count", "0")
    parts = _replace_or_append_option(parts, "--output_model_path", str(next_output_model_path))
    parts = _replace_or_append_option(
        parts,
        "--full_set_sparse_validation_profile",
        str(sparse_validation_profile),
    )
    if bool(repair_regressions):
        parts = _apply_regression_repair_resample_options(parts)
    return " ".join(shlex.quote(part) for part in parts)


def _load_manifest_command(path: str | Path | None) -> str | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"manifest must be a JSON object: {path}")
    command = payload.get("command")
    if isinstance(command, list):
        return " ".join(shlex.quote(str(part)) for part in command)
    if isinstance(command, str) and command.strip():
        return command
    raise ValueError(f"manifest does not contain a usable command: {path}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sparse_pnp_validation_loop_once(
    *,
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    scene: str,
    split_name: str,
    output_dir: str | Path,
    baseline_feedback_bank: str | Path | None = None,
    candidate_feedback_bank: str | Path | None = None,
    candidate_map_manifest: str | Path | None = None,
    next_output_model_path: str | Path | None = None,
    protected_te_cm: float = 15.0,
    hard_te_cm: float = 20.0,
    regression_margin_cm: float = 20.0,
    improvement_margin_cm: float = 20.0,
    protect_all_baseline_good_queries: bool = True,
    repair_regressions: bool = False,
) -> dict[str, Any]:
    split = str(split_name).strip()
    if split.lower() == "test":
        raise ValueError("test split is not allowed for sparse PnP validation loop")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=baseline_run_dir,
        candidate_run_dir=candidate_run_dir,
        scene=str(scene),
        split_name=split,
        baseline_feedback_bank=baseline_feedback_bank,
        candidate_feedback_bank=candidate_feedback_bank,
        protected_te_cm=float(protected_te_cm),
        hard_te_cm=float(hard_te_cm),
        regression_margin_cm=float(regression_margin_cm),
        improvement_margin_cm=float(improvement_margin_cm),
        protect_all_baseline_good_queries=bool(protect_all_baseline_good_queries),
    )
    profile_path = output / "sparse_pnp_validation_profile.json"
    _write_json(profile_path, profile)

    next_command: str | None = None
    manifest_command = _load_manifest_command(candidate_map_manifest)
    if manifest_command is not None and next_output_model_path is not None:
        next_command = build_next_resample_command(
            manifest_command,
            next_output_model_path=next_output_model_path,
            sparse_validation_profile=profile_path,
            repair_regressions=bool(repair_regressions),
        )
        (output / "next_resample_command.txt").write_text(next_command + "\n", encoding="utf-8")

    metrics = dict(profile.get("metrics", {}))
    metrics.update(
        {
            "profile_path": str(profile_path),
            "next_resample_command_path": None if next_command is None else str(output / "next_resample_command.txt"),
            "loop_closed": bool(next_command is not None),
        }
    )
    split_audit = {
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
        "checks": {
            "baseline_run_dir": {"status": "present", "path": str(Path(baseline_run_dir))},
            "candidate_run_dir": {"status": "present", "path": str(Path(candidate_run_dir))},
            "profile_split": {"status": "passed", "split_name": split},
        },
    }
    manifest = {
        "method": "loc_gs_sparse_pnp_validation_loop_once",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "command": " ".join(
            shlex.quote(part)
            for part in [sys.executable, "-m", "loc_gs.scripts.run_sparse_pnp_validation_loop", *sys.argv[1:]]
        ),
        "scene": str(scene),
        "split": split,
        "baseline_run_dir": str(Path(baseline_run_dir)),
        "candidate_run_dir": str(Path(candidate_run_dir)),
        "baseline_feedback_bank": None if baseline_feedback_bank is None else str(Path(baseline_feedback_bank)),
        "candidate_feedback_bank": None if candidate_feedback_bank is None else str(Path(candidate_feedback_bank)),
        "candidate_map_manifest": None if candidate_map_manifest is None else str(Path(candidate_map_manifest)),
        "next_output_model_path": None if next_output_model_path is None else str(Path(next_output_model_path)),
        "profile_path": str(profile_path),
        "next_resample_command": next_command,
        "metrics_summary": metrics,
        "split_audit": split_audit,
        "branch_selection": False,
        "single_path_deployment": True,
        "official_test_used": False,
        "diagnostic_only": True,
        "repair_regressions": bool(repair_regressions),
    }
    write_artifact_audit_bundle(
        output,
        manifest=manifest,
        command=str(manifest["command"]),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    (output / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")
    return {"profile": profile, "metrics": metrics, "next_resample_command": next_command}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one sparse-PnP validation loop step from real eval outputs.")
    parser.add_argument("--baseline_run_dir", required=True, type=Path)
    parser.add_argument("--candidate_run_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--baseline_feedback_bank", default=None, type=Path)
    parser.add_argument("--candidate_feedback_bank", default=None, type=Path)
    parser.add_argument("--candidate_map_manifest", default=None, type=Path)
    parser.add_argument("--next_output_model_path", default=None, type=Path)
    parser.add_argument("--protected_te_cm", default=15.0, type=float)
    parser.add_argument("--hard_te_cm", default=20.0, type=float)
    parser.add_argument("--regression_margin_cm", default=20.0, type=float)
    parser.add_argument("--improvement_margin_cm", default=20.0, type=float)
    parser.add_argument("--protect_all_baseline_good_queries", action="store_true")
    parser.add_argument(
        "--repair_regressions",
        action="store_true",
        help="Rewrite the next resample command to focus sparse validation support on real PnP regressions.",
    )
    parser.add_argument("--execute_next_resample", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    result = run_sparse_pnp_validation_loop_once(
        baseline_run_dir=args.baseline_run_dir,
        candidate_run_dir=args.candidate_run_dir,
        scene=str(args.scene),
        split_name=str(args.split_name),
        output_dir=args.output_dir,
        baseline_feedback_bank=args.baseline_feedback_bank,
        candidate_feedback_bank=args.candidate_feedback_bank,
        candidate_map_manifest=args.candidate_map_manifest,
        next_output_model_path=args.next_output_model_path,
        protected_te_cm=float(args.protected_te_cm),
        hard_te_cm=float(args.hard_te_cm),
        regression_margin_cm=float(args.regression_margin_cm),
        improvement_margin_cm=float(args.improvement_margin_cm),
        protect_all_baseline_good_queries=bool(args.protect_all_baseline_good_queries),
        repair_regressions=bool(args.repair_regressions),
    )
    command = result.get("next_resample_command")
    if bool(args.execute_next_resample):
        if not command:
            raise ValueError("--execute_next_resample requires candidate_map_manifest and next_output_model_path")
        subprocess.run(shlex.split(str(command)), check=True)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
