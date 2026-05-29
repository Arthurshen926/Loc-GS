#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


_POSITIVE_FIELDS = ("support", "viable_tuple_mass", "logdet_H", "min_eigenvalue", "min_eigen", "lambda_min")
_NEGATIVE_FIELDS = ("dense_worsen_risk", "dense_worsen", "ambiguity", "ambiguity_risk")


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _metric_utility(metrics: Mapping[str, Any]) -> float:
    positive = sum(max(0.0, float(metrics.get(field, 0.0) or 0.0)) for field in _POSITIVE_FIELDS)
    negative = sum(max(0.0, float(metrics.get(field, 0.0) or 0.0)) for field in _NEGATIVE_FIELDS)
    return float(positive - negative)


def _risk(metrics: Mapping[str, Any], names: tuple[str, ...]) -> float:
    return float(sum(max(0.0, float(metrics.get(name, 0.0) or 0.0)) for name in names))


def _load_results(path: str | Path) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("results.json must contain a list")
    return [dict(row) for row in rows]


def _dense_te_by_query(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {str(row["image_name"]): float(row["dense_TE"]) for row in rows if "image_name" in row and "dense_TE" in row}


def _dense_worsened_queries(rows: list[dict[str, Any]], *, margin_cm: float) -> list[str]:
    out: list[str] = []
    for row in rows:
        if "image_name" not in row or "sparse_TE" not in row or "dense_TE" not in row:
            continue
        if float(row["dense_TE"]) - float(row["sparse_TE"]) >= float(margin_cm):
            out.append(str(row["image_name"]))
    return sorted(set(out))


def _candidate_risk_tables(candidate_gain: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    dense: dict[str, float] = {}
    ambiguity: dict[str, float] = {}
    for landmark_id, query_map in candidate_gain.items():
        dense[str(landmark_id)] = float(
            sum(_risk(dict(metrics), ("dense_worsen_risk", "dense_worsen")) for metrics in dict(query_map).values())
        )
        ambiguity[str(landmark_id)] = float(
            sum(_risk(dict(metrics), ("ambiguity", "ambiguity_risk")) for metrics in dict(query_map).values())
        )
    return dense, ambiguity


def _drop_priority_from_source_loss(source_loss: Mapping[str, Any]) -> dict[str, float]:
    utility: dict[str, float] = {}
    for landmark_id, query_map in source_loss.items():
        utility[str(landmark_id)] = float(sum(max(0.0, _metric_utility(dict(metrics))) for metrics in dict(query_map).values()))
    if not utility:
        return {}
    max_utility = max(utility.values())
    return {landmark_id: float(max_utility - value) for landmark_id, value in utility.items()}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v7 failure-aware profile from train/self-map evaluation and solver constraints.")
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--candidate_results", required=True)
    parser.add_argument("--solver_constraints", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--dense_worsen_margin_cm", type=float, default=5.0)
    parser.add_argument("--protected_te_cm", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip()
    if split_name.lower() == "test":
        raise ValueError("test split results are not allowed for failure-profile construction")
    baseline_rows = _load_results(args.baseline_results)
    candidate_rows = _load_results(args.candidate_results)
    constraints = json.loads(Path(args.solver_constraints).read_text(encoding="utf-8"))
    if not isinstance(constraints, dict):
        raise TypeError("solver constraints must be a JSON object")
    candidate_gain = dict(constraints.get("candidate_gain", {}))
    dense_risk, ambiguity_risk = _candidate_risk_tables(candidate_gain)
    profile = {
        "schema": "loc_gs_failure_profile_v1",
        "scene": str(args.scene),
        "split_name": split_name or "unknown",
        "source": "train_or_selfmap_eval_plus_solver_constraints",
        "query_baseline_dense_te_cm": _dense_te_by_query(baseline_rows),
        "dense_worsened_query_ids": _dense_worsened_queries(
            candidate_rows,
            margin_cm=float(args.dense_worsen_margin_cm),
        ),
        "candidate_query_gain": candidate_gain,
        "candidate_regression_risk": {},
        "dense_worsen_risk": dense_risk,
        "ambiguity_risk": ambiguity_risk,
        "source_loss": _drop_priority_from_source_loss(dict(constraints.get("source_loss", {}))),
        "metadata": {
            "protected_te_cm": float(args.protected_te_cm),
            "dense_worsen_margin_cm": float(args.dense_worsen_margin_cm),
            "solver_constraints": str(args.solver_constraints),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "failure_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": profile["split_name"],
        "recipe": "lsf_v7_failure_profile",
        "artifact": str(profile_path),
        "single_path_deployment": True,
        "branch_selection": False,
    }
    metrics = {
        "query_count": int(len(profile["query_baseline_dense_te_cm"])),
        "candidate_gain_count": int(len(profile["candidate_query_gain"])),
        "dense_worsened_query_count": int(len(profile["dense_worsened_query_ids"])),
        "source_loss_count": int(len(profile["source_loss"])),
    }
    manifest = {
        "method": "loc_gs_lsf_v7_failure_profile",
        **metadata,
        "baseline_results": str(args.baseline_results),
        "candidate_results": str(args.candidate_results),
        "solver_constraints": str(args.solver_constraints),
        "profile_path": str(profile_path),
    }
    split_audit = artifact_split_audit(metadata, branch_selection=False)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    print(json.dumps({"profile": str(profile_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
