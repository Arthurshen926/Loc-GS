#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.stdloc_native.evidence_gate import build_evidence_gate
from loc_gs.stdloc_native.selector_resampling import write_resampled_detector_payload
from loc_gs.stdloc_native.soft_prior import _assert_safe_output_map, _latest_point_cloud_path, _write_point_cloud_locability
from loc_gs.stdloc_native.solver_admissibility import make_replacement_admissibility_checker
from loc_gs.stdloc_native.solver_aware_resampling import solver_aware_local_edit
from loc_gs.stdloc_native.solver_coverage_coreset import build_solver_coverage_tables, solver_coverage_local_edit


def _load_tensor(path: str | Path | None) -> torch.Tensor | None:
    if not path:
        return None
    payload: Any = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("tensor", "values", "data", "selector", "score", "support", "risk"):
            if key in payload:
                return torch.as_tensor(payload[key]).cpu()
        raise KeyError(f"{path} does not contain a tensor-like key")
    return torch.as_tensor(payload).cpu()


def _load_solver_consensus_support(path: str | Path | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path:
        return {}, {}
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("solver consensus support artifact must be a dict")
    tensors: dict[str, torch.Tensor] = {}
    for key in ("support_score", "hard_negative_risk", "dense_worsen_risk"):
        if key in payload:
            tensors[key] = torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1).cpu()
    if "support_score" not in tensors:
        raise KeyError("solver consensus support artifact must contain support_score")
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    return tensors, metadata


def _load_localization_support_field(path: str | Path | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path:
        return {}, {}
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("localization support field artifact must be a dict")
    tensors: dict[str, torch.Tensor] = {}
    for key in (
        "support_for_selection",
        "support_score",
        "hard_negative_risk",
        "dense_worsen_risk",
        "solver_set_utility",
        "pose_information",
        "ambiguity_risk",
    ):
        if key in payload:
            tensors[key] = torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1).cpu()
    if "support_for_selection" not in tensors and "support_score" not in tensors:
        raise KeyError("localization support field must contain support_for_selection or support_score")
    size = int((tensors["support_for_selection"] if "support_for_selection" in tensors else tensors["support_score"]).numel())
    for key, tensor in tensors.items():
        if tensor.numel() != size:
            raise ValueError(f"{key} length {tensor.numel()} does not match localization support field length {size}")
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    split = str(metadata.get("split_name", metadata.get("split", ""))).strip().lower()
    if split == "test":
        raise ValueError("test split localization support fields are not allowed")
    return tensors, metadata


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _landmark_query_keyed_nested(payload: Any) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    if not payload:
        return result
    for outer_key, query_map in dict(payload).items():
        outer = int(outer_key)
        result[outer] = {}
        for query_key, metrics in dict(query_map).items():
            result[outer][str(query_key)] = dict(metrics)
    return result


def _load_solver_admissibility_payload(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_solver_admissibility(path: str | Path | None, *, require_candidate_gain: bool = False):
    if not path:
        return None, {"enabled": False}
    payload = _load_solver_admissibility_payload(path)
    if payload is None:
        return None, {"enabled": False}
    thresholds = dict(payload.get("thresholds", {}))
    hard_query_ids = [str(item) for item in payload.get("hard_query_ids", [])]
    candidate_gain = _landmark_query_keyed_nested(payload.get("candidate_gain", {}))
    source_loss = _landmark_query_keyed_nested(payload.get("source_loss", {}))
    rejected: list[dict[str, Any]] = []

    def _threshold(name: str, default: Any, *aliases: str) -> Any:
        for key in (name, *aliases):
            if key in thresholds:
                return thresholds[key]
        return default

    cvar_alpha = _threshold("cvar_alpha", None, "hard_query_cvar_alpha")
    checker = make_replacement_admissibility_checker(
        hard_query_ids=hard_query_ids,
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_support_delta=float(thresholds.get("min_support_delta", 0.0)),
        min_viable_tuple_delta=float(thresholds.get("min_viable_tuple_delta", 0.0)),
        min_logdet_delta=float(thresholds.get("min_logdet_delta", 0.0)),
        min_min_eigen_delta=float(_threshold("min_min_eigen_delta", 0.0, "min_eigen_delta")),
        max_dense_worsen_delta=float(
            _threshold("max_dense_worsen_delta", 0.0, "max_dense_worsen_risk_delta")
        ),
        max_ambiguity_delta=float(thresholds.get("max_ambiguity_delta", 0.0)),
        cvar_alpha=float(cvar_alpha) if cvar_alpha is not None else None,
        min_cvar_score=float(_threshold("min_cvar_score", 0.0, "min_hard_query_cvar_score")),
        cvar_weights=_threshold("cvar_weights", None, "hard_query_cvar_weights"),
        require_candidate_gain=bool(require_candidate_gain),
    )

    def _callback(add_id: int, drop_id: int) -> bool:
        decision = checker(add_id=int(add_id), drop_id=int(drop_id))
        if not decision.admissible and len(rejected) < 50:
            rejected.append(
                {
                    "add_id": int(add_id),
                    "drop_id": int(drop_id),
                    "reasons": list(decision.reasons),
                    "delta": decision.delta,
                    "cvar": decision.cvar,
                }
            )
        return bool(decision.admissible)

    metadata = {
        "enabled": True,
        "path": str(path),
        "hard_query_count": int(len(hard_query_ids)),
        "candidate_gain_count": int(len(candidate_gain)),
        "source_loss_count": int(len(source_loss)),
        "thresholds": thresholds,
        "v3_hard_query_gate": bool(
            any(
                key in thresholds
                for key in (
                    "min_min_eigen_delta",
                    "min_eigen_delta",
                    "max_dense_worsen_delta",
                    "max_dense_worsen_risk_delta",
                    "cvar_alpha",
                    "hard_query_cvar_alpha",
                    "min_cvar_score",
                    "min_hard_query_cvar_score",
                )
            )
        ),
        "require_candidate_gain": bool(require_candidate_gain),
        "rejected_examples": rejected,
    }
    return _callback, metadata


def _resolve_effective_max_edits(
    requested_max_edits: int,
    admissibility_metadata: dict[str, Any],
    *,
    candidate_gain_fraction: float,
    min_adaptive_edits: int,
) -> tuple[int, dict[str, Any]]:
    requested = max(0, int(requested_max_edits))
    fraction = float(candidate_gain_fraction)
    candidate_gain_count = int(admissibility_metadata.get("candidate_gain_count", 0) or 0)
    enabled = fraction > 0.0
    adaptive_limit = requested
    if enabled:
        adaptive_limit = int(math.floor(float(candidate_gain_count) * fraction))
        if candidate_gain_count > 0:
            adaptive_limit = max(int(min_adaptive_edits), adaptive_limit)
        adaptive_limit = max(0, adaptive_limit)
    effective = min(requested, adaptive_limit)
    return effective, {
        "enabled": bool(enabled),
        "requested_max_edits": requested,
        "effective_max_edits": int(effective),
        "candidate_gain_count": int(candidate_gain_count),
        "candidate_gain_fraction": fraction,
        "min_adaptive_edits": int(min_adaptive_edits),
        "adaptive_limit": int(adaptive_limit),
    }


def _load_source_idx(source_map: Path) -> torch.Tensor:
    path = source_map / "detector" / "sampled_idx.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing source sampled_idx.pkl: {path}")
    return torch.as_tensor(_load_pickle(path), dtype=torch.long).reshape(-1).cpu()


def _load_source_scores(source_map: Path, sampled_idx: torch.Tensor, size: int) -> torch.Tensor:
    scores = torch.zeros((int(size),), dtype=torch.float32)
    path = source_map / "detector" / "sampled_scores.pkl"
    if not path.exists():
        return scores
    payload = _load_pickle(path)
    if isinstance(payload, dict) and "score_avg" in payload:
        raw = torch.as_tensor(payload["score_avg"], dtype=torch.float32).reshape(-1).cpu()
        if raw.shape[0] == size:
            return raw
    raw = (
        torch.as_tensor(payload.get("sampled_scores"), dtype=torch.float32).reshape(-1).cpu()
        if isinstance(payload, dict) and "sampled_scores" in payload
        else torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    )
    if raw.shape[0] == sampled_idx.shape[0]:
        valid = (sampled_idx >= 0) & (sampled_idx < int(size))
        scores[sampled_idx[valid]] = raw[valid]
    return scores


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _command_from_argv() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv) if sys.argv else ""


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an evidence-gated solver-aware LSF sampled map.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--selector_path", default="")
    parser.add_argument("--solver_consensus_support_path", default="")
    parser.add_argument("--localization_support_field_path", default="")
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--positive_support_path", default="")
    parser.add_argument("--hard_negative_risk_path", default="")
    parser.add_argument("--alpha_reliability_path", default="")
    parser.add_argument("--multiview_stability_path", default="")
    parser.add_argument("--pose_utility_path", default="")
    parser.add_argument("--safe_core_path", default="")
    parser.add_argument("--candidate_pool_path", default="")
    parser.add_argument("--solver_admissibility_path", default="")
    parser.add_argument("--require_candidate_gain", action="store_true")
    parser.add_argument("--selection_policy", choices=("local", "coverage"), default="local")
    parser.add_argument("--require_candidate_pool", action="store_true")
    parser.add_argument("--min_positive_support", type=float, default=0.0)
    parser.add_argument("--max_hard_negative_risk", type=float, default=1.0)
    parser.add_argument("--min_alpha_reliability", type=float, default=0.0)
    parser.add_argument("--min_multiview_stability", type=float, default=0.0)
    parser.add_argument("--min_pose_utility", type=float, default=0.0)
    parser.add_argument("--keep_source", action="store_true")
    parser.add_argument("--max_edits", type=int, default=256)
    parser.add_argument(
        "--max_edits_candidate_gain_fraction",
        type=float,
        default=0.0,
        help="If positive, cap edits by floor(candidate_gain_count * fraction) from solver admissibility metadata.",
    )
    parser.add_argument(
        "--min_adaptive_edits",
        type=int,
        default=0,
        help="Minimum edit budget when candidate-gain adaptive budget is enabled and candidate_gain_count > 0.",
    )
    parser.add_argument(
        "--admissibility_drop_scan",
        type=int,
        default=1,
        help="Number of low-utility removable source landmarks to try for each candidate under solver admissibility.",
    )
    parser.add_argument("--write_dense_locability", action="store_true")
    parser.add_argument(
        "--protect_source_score_min",
        type=float,
        default=-1.0,
        help="Protect native source landmarks whose detector score_avg is at least this value.",
    )
    parser.add_argument(
        "--disable_lsf_sparse_risk",
        action="store_true",
        help="Do not auto-use LSF hard-negative/dense-worsen risk as a sparse edit penalty.",
    )
    parser.add_argument(
        "--extra_dense_worsen_penalty",
        type=float,
        default=0.0,
        help="Additional utility penalty applied to dense_worsen_risk during sparse candidate ordering.",
    )
    parser.add_argument(
        "--min_coverage_gain",
        type=float,
        default=0.0,
        help="Minimum hard-query coverage gain for --selection_policy coverage replacements.",
    )
    parser.add_argument(
        "--min_query_coverage",
        type=float,
        default=1.0,
        help="Per-query source support floor protected by --selection_policy coverage.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _copy_source_map(source: Path, output: Path, overwrite: bool) -> None:
    _assert_safe_output_map(source, output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output_map exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if bool(args.require_candidate_pool) and not str(args.candidate_pool_path).strip():
        raise ValueError("candidate_pool_path is required when require_candidate_pool is enabled")
    if str(args.selection_policy) == "coverage" and not str(args.solver_admissibility_path).strip():
        raise ValueError("solver_admissibility_path is required when selection_policy=coverage")
    source_map = Path(args.source_map)
    output_map = Path(args.output_map)
    source_idx = _load_source_idx(source_map)
    support_artifact, support_metadata = _load_solver_consensus_support(args.solver_consensus_support_path)
    lsf_artifact, lsf_metadata = _load_localization_support_field(args.localization_support_field_path)
    selector = _load_tensor(args.selector_path)
    if selector is None and lsf_artifact:
        selector = lsf_artifact.get("support_for_selection", lsf_artifact.get("support_score"))
    if selector is None and support_artifact:
        selector = support_artifact["support_score"]
    if selector is None:
        raise ValueError("selector_path or solver_consensus_support_path is required")
    selector = selector.float().reshape(-1).cpu().clamp(0.0, 1.0)
    size = int(selector.shape[0])
    positive = _load_tensor(args.positive_support_path)
    if positive is None and lsf_artifact:
        positive = lsf_artifact.get("support_score", lsf_artifact.get("support_for_selection"))
    if positive is None and support_artifact:
        positive = support_artifact["support_score"]
    risk = _load_tensor(args.hard_negative_risk_path)
    use_auto_lsf_risk = not bool(args.disable_lsf_sparse_risk)
    if risk is None and lsf_artifact and use_auto_lsf_risk:
        risk = lsf_artifact.get("hard_negative_risk")
    if risk is None and support_artifact and use_auto_lsf_risk:
        risk = support_artifact.get("hard_negative_risk")
    if use_auto_lsf_risk and lsf_artifact and "dense_worsen_risk" in lsf_artifact:
        dense_risk = lsf_artifact["dense_worsen_risk"]
        risk = dense_risk if risk is None else torch.maximum(torch.as_tensor(risk, dtype=torch.float32).reshape(-1).cpu(), dense_risk)
    if use_auto_lsf_risk and support_artifact and "dense_worsen_risk" in support_artifact:
        dense_risk = support_artifact["dense_worsen_risk"]
        risk = dense_risk if risk is None else torch.maximum(torch.as_tensor(risk, dtype=torch.float32).reshape(-1).cpu(), dense_risk)
    alpha = _load_tensor(args.alpha_reliability_path)
    stability = _load_tensor(args.multiview_stability_path)
    pose = _load_tensor(args.pose_utility_path)
    gate = build_evidence_gate(
        selector=selector,
        positive_support=positive,
        hard_negative_risk=risk,
        alpha_reliability=alpha,
        multiview_stability=stability,
        pose_utility=pose,
        source_idx=source_idx,
        keep_source=bool(args.keep_source),
        min_positive_support=float(args.min_positive_support),
        max_hard_negative_risk=float(args.max_hard_negative_risk),
        min_alpha_reliability=float(args.min_alpha_reliability),
        min_multiview_stability=float(args.min_multiview_stability),
        min_pose_utility=float(args.min_pose_utility),
    )
    dense_worsen_for_penalty = torch.zeros_like(selector)
    if use_auto_lsf_risk and lsf_artifact and "dense_worsen_risk" in lsf_artifact:
        dense_worsen_for_penalty = torch.maximum(
            dense_worsen_for_penalty,
            lsf_artifact["dense_worsen_risk"].float().reshape(-1).cpu().clamp(0.0, 1.0),
        )
    if use_auto_lsf_risk and support_artifact and "dense_worsen_risk" in support_artifact:
        dense_worsen_for_penalty = torch.maximum(
            dense_worsen_for_penalty,
            support_artifact["dense_worsen_risk"].float().reshape(-1).cpu().clamp(0.0, 1.0),
        )
    safe_core = _load_tensor(args.safe_core_path)
    candidate_pool = _load_tensor(args.candidate_pool_path)
    if candidate_pool is None:
        candidate_pool = torch.arange(size, dtype=torch.long)
    source_scores = _load_source_scores(source_map, source_idx, size)
    extra_dense_worsen_penalty = float(args.extra_dense_worsen_penalty)
    utility = (
        gate["score"]
        + 0.05 * source_scores.clamp(0.0, 1.0)
        - extra_dense_worsen_penalty * dense_worsen_for_penalty
    ).clamp_min(0.0).float()
    admissibility_callback, admissibility_metadata = _load_solver_admissibility(
        args.solver_admissibility_path,
        require_candidate_gain=bool(args.require_candidate_gain),
    )
    effective_max_edits, adaptive_edit_budget = _resolve_effective_max_edits(
        int(args.max_edits),
        admissibility_metadata,
        candidate_gain_fraction=float(args.max_edits_candidate_gain_fraction),
        min_adaptive_edits=int(args.min_adaptive_edits),
    )
    safe_core_tensor = safe_core.long().reshape(-1).cpu() if safe_core is not None else None
    protected_source_score_count = 0
    protect_score_min = float(args.protect_source_score_min)
    if protect_score_min >= 0.0:
        valid_source = source_idx[(source_idx >= 0) & (source_idx < int(source_scores.numel()))]
        if valid_source.numel():
            protected = valid_source[source_scores[valid_source] >= protect_score_min].long().reshape(-1).cpu()
        else:
            protected = torch.empty((0,), dtype=torch.long)
        protected_source_score_count = int(protected.numel())
        if protected.numel():
            safe_core_tensor = (
                protected
                if safe_core_tensor is None
                else torch.unique(torch.cat([safe_core_tensor, protected], dim=0)).long().cpu()
            )
    if str(args.selection_policy) == "local":
        edit = solver_aware_local_edit(
            source_idx=source_idx,
            candidate_pool=candidate_pool.long().reshape(-1).cpu(),
            utility=utility,
            evidence_mask=gate["mask"],
            safe_core=safe_core_tensor,
            is_admissible=admissibility_callback,
            max_edits=int(effective_max_edits),
            max_drop_scan=int(args.admissibility_drop_scan),
        )
    else:
        solver_payload = _load_solver_admissibility_payload(args.solver_admissibility_path) or {}
        edit = solver_coverage_local_edit(
            source_idx=source_idx,
            candidate_pool=candidate_pool.long().reshape(-1).cpu(),
            utility=utility,
            evidence_mask=gate["mask"],
            coverage_tables=build_solver_coverage_tables(solver_payload),
            safe_core=safe_core_tensor,
            is_admissible=admissibility_callback,
            max_edits=int(effective_max_edits),
            max_drop_scan=int(args.admissibility_drop_scan),
            min_coverage_gain=float(args.min_coverage_gain),
            min_query_coverage=float(args.min_query_coverage),
        )
    edit["metadata"]["protected_source_score_count"] = int(protected_source_score_count)
    edit["metadata"]["effective_max_edits"] = int(effective_max_edits)
    payload = {
        "sampled_idx": edit["sampled_idx"],
        "sampled_scores": utility[edit["sampled_idx"]],
        "score_avg": utility,
        "selector": selector,
        "source_count": int(source_idx.numel()),
        "output_count": int(edit["sampled_idx"].numel()),
        "sampled_idx_changed": source_idx.shape != edit["sampled_idx"].shape or not torch.equal(source_idx, edit["sampled_idx"]),
    }
    scene_name = str(
        lsf_metadata.get("scene", support_metadata.get("scene", source_map.name))
    )
    split_name = str(
        lsf_metadata.get(
            "split_name",
            lsf_metadata.get("split", support_metadata.get("split_name", support_metadata.get("split", "unknown"))),
        )
    )
    manifest = {
        "method": (
            "loc_gs_lsf_solver_coverage_coreset"
            if str(args.selection_policy) == "coverage"
            else "loc_gs_lsf_solver_aware_resampling"
        ),
        "git_commit": _git_commit(),
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "command": _command_from_argv(),
        "scene": scene_name,
        "split_name": split_name,
        "data_root": "unknown",
        "checkpoint_path": "unknown",
        "map_path": str(source_map),
        "source_map": str(source_map),
        "output_map": str(output_map),
        "selector_path": str(args.selector_path),
        "solver_consensus_support_path": str(args.solver_consensus_support_path),
        "localization_support_field_path": str(args.localization_support_field_path),
        "positive_support_path": str(args.positive_support_path),
        "hard_negative_risk_path": str(args.hard_negative_risk_path),
        "alpha_reliability_path": str(args.alpha_reliability_path),
        "pose_utility_path": str(args.pose_utility_path),
        "safe_core_path": str(args.safe_core_path),
        "candidate_pool_path": str(args.candidate_pool_path),
        "candidate_pool_required": bool(args.require_candidate_pool),
        "solver_admissibility_path": str(args.solver_admissibility_path),
        "selection_policy": str(args.selection_policy),
        "single_path_deployment": True,
        "branch_selection": False,
        "same_budget": int(payload["source_count"]) == int(payload["output_count"]),
        "hyperparameters": {
            "min_positive_support": float(args.min_positive_support),
            "max_hard_negative_risk": float(args.max_hard_negative_risk),
            "min_alpha_reliability": float(args.min_alpha_reliability),
            "min_multiview_stability": float(args.min_multiview_stability),
            "min_pose_utility": float(args.min_pose_utility),
            "keep_source": bool(args.keep_source),
            "max_edits": int(args.max_edits),
            "effective_max_edits": int(effective_max_edits),
            "max_edits_candidate_gain_fraction": float(args.max_edits_candidate_gain_fraction),
            "min_adaptive_edits": int(args.min_adaptive_edits),
            "admissibility_drop_scan": int(args.admissibility_drop_scan),
            "write_dense_locability": bool(args.write_dense_locability),
            "require_candidate_gain": bool(args.require_candidate_gain),
            "require_candidate_pool": bool(args.require_candidate_pool),
            "selection_policy": str(args.selection_policy),
            "min_coverage_gain": float(args.min_coverage_gain),
            "min_query_coverage": float(args.min_query_coverage),
            "disable_lsf_sparse_risk": bool(args.disable_lsf_sparse_risk),
            "protect_source_score_min": float(args.protect_source_score_min),
            "extra_dense_worsen_penalty": extra_dense_worsen_penalty,
        },
        "evidence_gate": gate["metadata"],
        "adaptive_edit_budget": adaptive_edit_budget,
        "utility_shaping": {
            "source_score_bonus": 0.05,
            "extra_dense_worsen_penalty": extra_dense_worsen_penalty,
        },
        "solver_aware": edit["metadata"],
        "solver_admissibility": admissibility_metadata,
        "solver_consensus_support": support_metadata,
        "localization_support_field": lsf_metadata,
        "dense_support": {
            "enabled": bool(args.write_dense_locability),
            "source": "localization_support_field" if lsf_artifact else ("selector" if selector is not None else "none"),
        },
        "edits": edit["edits"][:50],
    }
    manifest["split_audit"] = artifact_split_audit(
        support_metadata,
        lsf_metadata,
        admissibility_metadata,
        branch_selection=False,
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    _copy_source_map(source_map, output_map, bool(args.overwrite))
    write_resampled_detector_payload(output_map / "detector", payload)
    if bool(args.write_dense_locability):
        point_cloud_path = _latest_point_cloud_path(source_map)
        if point_cloud_path is None:
            raise FileNotFoundError(f"source map has no point_cloud/iteration_*/point_cloud.ply: {source_map}")
        dense_values = (
            lsf_artifact.get("support_for_selection", lsf_artifact.get("support_score"))
            if lsf_artifact
            else selector
        )
        if dense_values is None:
            raise ValueError("dense locability export requires selector or localization support field")
        output_point_cloud = output_map / point_cloud_path.relative_to(source_map)
        if output_point_cloud.exists() or output_point_cloud.is_symlink():
            output_point_cloud.unlink()
        _write_point_cloud_locability(point_cloud_path, output_point_cloud, torch.as_tensor(dense_values).float())
        manifest["dense_support"]["updated_point_cloud"] = str(point_cloud_path.relative_to(source_map))
    (output_map / "lsf_solver_aware_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_map / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metrics_summary = {
        "sampled_count": int(payload["output_count"]),
        "source_count": int(payload["source_count"]),
        "same_budget": bool(manifest["same_budget"]),
        "num_edits": int(len(edit["edits"])),
        "num_rejected_by_admissibility": int(edit["metadata"].get("num_rejected_by_admissibility", 0)),
        "dense_locability_written": bool(args.write_dense_locability),
    }
    write_artifact_audit_bundle(
        output_map,
        manifest=manifest,
        command=manifest["command"],
        metrics_summary=metrics_summary,
        split_audit=manifest["split_audit"],
    )
    print(
        "[lsf_solver_aware_export] wrote "
        f"{output_map} ({payload['source_count']} -> {payload['output_count']} sampled landmarks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
