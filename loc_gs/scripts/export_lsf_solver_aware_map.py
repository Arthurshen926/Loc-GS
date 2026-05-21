#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from loc_gs.stdloc_native.evidence_gate import build_evidence_gate
from loc_gs.stdloc_native.selector_resampling import write_resampled_detector_payload
from loc_gs.stdloc_native.solver_admissibility import is_replacement_admissible
from loc_gs.stdloc_native.solver_aware_resampling import solver_aware_local_edit


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


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _int_keyed_nested(payload: Any) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = {}
    if not payload:
        return result
    for outer_key, query_map in dict(payload).items():
        outer = int(outer_key)
        result[outer] = {}
        for query_key, metrics in dict(query_map).items():
            result[outer][int(query_key)] = dict(metrics)
    return result


def _load_solver_admissibility(path: str | Path | None, *, require_candidate_gain: bool = False):
    if not path:
        return None, {"enabled": False}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = dict(payload.get("thresholds", {}))
    hard_query_ids = [int(item) for item in payload.get("hard_query_ids", [])]
    candidate_gain = _int_keyed_nested(payload.get("candidate_gain", {}))
    source_loss = _int_keyed_nested(payload.get("source_loss", {}))
    rejected: list[dict[str, Any]] = []

    def _callback(add_id: int, drop_id: int) -> bool:
        decision = is_replacement_admissible(
            add_id=int(add_id),
            drop_id=int(drop_id),
            hard_query_ids=hard_query_ids,
            candidate_gain=candidate_gain,
            source_loss=source_loss,
            min_support_delta=float(thresholds.get("min_support_delta", 0.0)),
            min_viable_tuple_delta=float(thresholds.get("min_viable_tuple_delta", 0.0)),
            min_logdet_delta=float(thresholds.get("min_logdet_delta", 0.0)),
            max_ambiguity_delta=float(thresholds.get("max_ambiguity_delta", 0.0)),
            require_candidate_gain=bool(require_candidate_gain),
        )
        if not decision.admissible and len(rejected) < 50:
            rejected.append(
                {
                    "add_id": int(add_id),
                    "drop_id": int(drop_id),
                    "reasons": list(decision.reasons),
                    "delta": decision.delta,
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
        "require_candidate_gain": bool(require_candidate_gain),
        "rejected_examples": rejected,
    }
    return _callback, metadata


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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an evidence-gated solver-aware LSF sampled map.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--selector_path", required=True)
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
    parser.add_argument("--min_positive_support", type=float, default=0.0)
    parser.add_argument("--max_hard_negative_risk", type=float, default=1.0)
    parser.add_argument("--min_alpha_reliability", type=float, default=0.0)
    parser.add_argument("--min_multiview_stability", type=float, default=0.0)
    parser.add_argument("--min_pose_utility", type=float, default=0.0)
    parser.add_argument("--keep_source", action="store_true")
    parser.add_argument("--max_edits", type=int, default=256)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _copy_source_map(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output_map exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_map = Path(args.source_map)
    output_map = Path(args.output_map)
    source_idx = _load_source_idx(source_map)
    selector = _load_tensor(args.selector_path)
    if selector is None:
        raise ValueError("selector_path did not load a tensor")
    selector = selector.float().reshape(-1).cpu().clamp(0.0, 1.0)
    size = int(selector.shape[0])
    positive = _load_tensor(args.positive_support_path)
    risk = _load_tensor(args.hard_negative_risk_path)
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
    safe_core = _load_tensor(args.safe_core_path)
    candidate_pool = _load_tensor(args.candidate_pool_path)
    if candidate_pool is None:
        candidate_pool = torch.arange(size, dtype=torch.long)
    source_scores = _load_source_scores(source_map, source_idx, size)
    utility = (gate["score"] + 0.05 * source_scores.clamp(0.0, 1.0)).float()
    admissibility_callback, admissibility_metadata = _load_solver_admissibility(
        args.solver_admissibility_path,
        require_candidate_gain=bool(args.require_candidate_gain),
    )
    edit = solver_aware_local_edit(
        source_idx=source_idx,
        candidate_pool=candidate_pool.long().reshape(-1).cpu(),
        utility=utility,
        evidence_mask=gate["mask"],
        safe_core=safe_core.long().reshape(-1).cpu() if safe_core is not None else None,
        is_admissible=admissibility_callback,
        max_edits=int(args.max_edits),
    )
    payload = {
        "sampled_idx": edit["sampled_idx"],
        "sampled_scores": utility[edit["sampled_idx"]],
        "score_avg": utility,
        "selector": selector,
        "source_count": int(source_idx.numel()),
        "output_count": int(edit["sampled_idx"].numel()),
        "sampled_idx_changed": source_idx.shape != edit["sampled_idx"].shape or not torch.equal(source_idx, edit["sampled_idx"]),
    }
    manifest = {
        "method": "loc_gs_lsf_solver_aware_resampling",
        "git_commit": _git_commit(),
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "source_map": str(source_map),
        "output_map": str(output_map),
        "selector_path": str(args.selector_path),
        "positive_support_path": str(args.positive_support_path),
        "hard_negative_risk_path": str(args.hard_negative_risk_path),
        "alpha_reliability_path": str(args.alpha_reliability_path),
        "pose_utility_path": str(args.pose_utility_path),
        "safe_core_path": str(args.safe_core_path),
        "solver_admissibility_path": str(args.solver_admissibility_path),
        "single_path_deployment": True,
        "branch_selection": False,
        "same_budget": int(payload["source_count"]) == int(payload["output_count"]),
        "evidence_gate": gate["metadata"],
        "solver_aware": edit["metadata"],
        "solver_admissibility": admissibility_metadata,
        "edits": edit["edits"][:50],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    _copy_source_map(source_map, output_map, bool(args.overwrite))
    write_resampled_detector_payload(output_map / "detector", payload)
    (output_map / "lsf_solver_aware_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_map / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[lsf_solver_aware_export] wrote "
        f"{output_map} ({payload['source_count']} -> {payload['output_count']} sampled landmarks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
