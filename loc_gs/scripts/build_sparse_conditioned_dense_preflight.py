#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from loc_gs.dense_support.sparse_conditioned_dense_preflight import (
    SLCDPThresholds,
    evaluate_slcdp_from_summaries,
)
from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_match_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("summaries"), list):
        return [dict(item) for item in payload["summaries"] if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    raise TypeError(f"match summary must be a mapping, list, or {{'summaries': [...]}}: {path}")


def _load_feature_summary(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return dict(_load_json(path))


def _write_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "scene",
        "query_index",
        "image_name",
        "decision",
        "reasons",
        "sparse_confident",
        "sparse_inlier_count",
        "same_pixel_cosine_p50",
        "query_to_render_max_cosine_mean",
        "coarse_mnn_count",
        "sparse_te_cm",
        "dense_te_cm",
        "accepted_final_te_cm",
        "dense_te_reduction_cm",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["reasons"] = ";".join(str(item) for item in row.get("reasons", []))
            writer.writerow(flat)


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            out.append(parsed)
    return out


def _metrics(cases: list[dict[str, Any]], *, scene: str, split_name: str) -> dict[str, Any]:
    reductions = _finite([case.get("dense_te_reduction_cm") for case in cases])
    accepted = _finite([case.get("accepted_final_te_cm") for case in cases])
    dense = _finite([case.get("dense_te_cm") for case in cases])
    decision_counts = {
        "accept_dense_count": sum(1 for case in cases if case.get("decision") == "accept_dense"),
        "skip_dense_keep_sparse_count": sum(1 for case in cases if case.get("decision") == "skip_dense_keep_sparse"),
        "retry_sparse_or_patch_dense_count": sum(1 for case in cases if case.get("decision") == "retry_sparse_or_patch_dense"),
    }
    return {
        "schema": "loc_gs_slcdp_metrics_v1",
        "scene": scene,
        "split_name": split_name,
        "query_count": int(len(cases)),
        **decision_counts,
        "mean_dense_te_cm": float(np.mean(dense)) if dense else None,
        "mean_accepted_final_te_cm": float(np.mean(accepted)) if accepted else None,
        "mean_dense_te_reduction_cm": float(np.mean(reductions)) if reductions else None,
        "median_dense_te_reduction_cm": float(np.median(reductions)) if reductions else None,
        "diagnostic_only": True,
        "paper_safe_for_tuning": split_name.lower() != "test",
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Sparse-Landmark-Conditioned Dense Preflight diagnostics.")
    parser.add_argument("--match_summary", required=True)
    parser.add_argument("--feature_summary", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--min_sparse_inliers", type=int, default=SLCDPThresholds.min_sparse_inliers)
    parser.add_argument("--min_feature_cosine", type=float, default=SLCDPThresholds.min_feature_cosine)
    parser.add_argument(
        "--min_query_to_render_max_cosine",
        type=float,
        default=SLCDPThresholds.min_query_to_render_max_cosine,
    )
    parser.add_argument("--min_coarse_mnn_count", type=int, default=SLCDPThresholds.min_coarse_mnn_count)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip() or "unknown"
    thresholds = SLCDPThresholds(
        min_sparse_inliers=int(args.min_sparse_inliers),
        min_feature_cosine=float(args.min_feature_cosine),
        min_query_to_render_max_cosine=float(args.min_query_to_render_max_cosine),
        min_coarse_mnn_count=int(args.min_coarse_mnn_count),
    )
    feature_summary = _load_feature_summary(args.feature_summary)
    cases = [
        evaluate_slcdp_from_summaries(case, feature_summary, thresholds=thresholds)
        for case in _load_match_cases(args.match_summary)
    ]
    metrics = _metrics(cases, scene=str(args.scene), split_name=split_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "loc_gs_slcdp_summary_v1",
        "scene": str(args.scene),
        "split_name": split_name,
        "diagnostic_only": True,
        "paper_safe_for_tuning": split_name.lower() != "test",
        "cases": cases,
        "thresholds": thresholds.__dict__,
    }
    summary_path = output_dir / "slcdp_summary.json"
    cases_path = output_dir / "slcdp_cases.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_cases_csv(cases_path, cases)
    manifest = {
        "method": "loc_gs_sparse_landmark_conditioned_dense_preflight",
        "recipe": "lsf_v10_slcdp_diagnostic",
        "scene": str(args.scene),
        "split_name": split_name,
        "diagnostic_only": True,
        "paper_safe_for_tuning": split_name.lower() != "test",
        "branch_selection": any(case.get("branch_selection") for case in cases),
        "match_summary": str(args.match_summary),
        "feature_summary": str(args.feature_summary),
        "artifact": str(summary_path),
    }
    split_audit = artifact_split_audit(manifest, branch_selection=bool(manifest["branch_selection"]))
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    print(json.dumps({"summary": str(summary_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
