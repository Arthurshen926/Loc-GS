#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _load_results(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if not isinstance(payload, list):
        raise TypeError(f"STDLoc results must be a list or {{'results': [...]}}: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_metric(row: Mapping[str, Any], stage: str, metric: str) -> float | None:
    candidates = (
        row.get(f"{stage}_{metric}"),
        row.get(f"{stage}_{metric.upper()}"),
        row.get(f"{stage}_{'AE' if metric.lower() == 're' else metric}"),
        row.get(f"{stage}_{'TE' if metric.lower() == 'te' else metric}"),
    )
    for value in candidates:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


def _query_id(row: Mapping[str, Any], index: int) -> str:
    for name in ("image_name", "query_id", "image_id", "name"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return f"query_{index:06d}"


def _row_metrics(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "query_index": int(index),
        "query_id": _query_id(row, index),
        "sparse_te_cm": _stage_metric(row, "sparse", "TE"),
        "sparse_re_deg": _stage_metric(row, "sparse", "AE"),
        "dense_te_cm": _stage_metric(row, "dense", "TE"),
        "dense_re_deg": _stage_metric(row, "dense", "AE"),
    }


def _is_sparse_correct(row: Mapping[str, Any], *, max_te_cm: float, max_re_deg: float) -> bool:
    sparse_te = _float(row.get("sparse_te_cm"))
    sparse_re = _float(row.get("sparse_re_deg"))
    return sparse_te is not None and sparse_re is not None and sparse_te <= max_te_cm and sparse_re <= max_re_deg


def _is_dense_wrong(row: Mapping[str, Any], *, max_te_cm: float, max_re_deg: float) -> bool:
    dense_te = _float(row.get("dense_te_cm"))
    dense_re = _float(row.get("dense_re_deg"))
    return dense_te is not None and dense_re is not None and (dense_te > max_te_cm or dense_re > max_re_deg)


def _dense_worsened(row: Mapping[str, Any], *, margin_cm: float) -> bool:
    sparse_te = _float(row.get("sparse_te_cm"))
    dense_te = _float(row.get("dense_te_cm"))
    return sparse_te is not None and dense_te is not None and dense_te - sparse_te >= margin_cm


def _dense_improved(row: Mapping[str, Any], *, margin_cm: float) -> bool:
    sparse_te = _float(row.get("sparse_te_cm"))
    dense_te = _float(row.get("dense_te_cm"))
    return sparse_te is not None and dense_te is not None and dense_te - sparse_te <= -margin_cm


def _reduction_rate(baseline_count: int, candidate_count: int) -> float | None:
    if baseline_count <= 0:
        return None
    return float((baseline_count - candidate_count) / baseline_count)


def build_dense_transition_audit(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    scene: str,
    split_name: str,
    sparse_correct_te_cm: float = 5.0,
    sparse_correct_re_deg: float = 5.0,
    dense_wrong_te_cm: float = 10.0,
    dense_wrong_re_deg: float = 5.0,
    dense_worsen_margin_cm: float = 5.0,
    regression_20cm: float = 20.0,
    regression_50cm: float = 50.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = min(len(baseline_rows), len(candidate_rows))
    pairs: list[dict[str, Any]] = []
    baseline_dense_worsened = 0
    candidate_dense_worsened = 0
    baseline_sparse_ok_dense_wrong = 0
    candidate_sparse_ok_dense_wrong = 0
    baseline_dense_improved = 0
    candidate_dense_improved = 0
    candidate_regression_20 = 0
    candidate_regression_50 = 0
    candidate_improvement_20 = 0

    for index in range(count):
        baseline = _row_metrics(baseline_rows[index], index)
        candidate = _row_metrics(candidate_rows[index], index)
        categories: list[str] = []
        if _dense_worsened(baseline, margin_cm=dense_worsen_margin_cm):
            baseline_dense_worsened += 1
            categories.append("baseline_dense_worsened")
        if _dense_worsened(candidate, margin_cm=dense_worsen_margin_cm):
            candidate_dense_worsened += 1
            categories.append("candidate_dense_worsened")
        if _dense_improved(baseline, margin_cm=dense_worsen_margin_cm):
            baseline_dense_improved += 1
        if _dense_improved(candidate, margin_cm=dense_worsen_margin_cm):
            candidate_dense_improved += 1
            categories.append("candidate_dense_improved")
        if _is_sparse_correct(
            baseline,
            max_te_cm=sparse_correct_te_cm,
            max_re_deg=sparse_correct_re_deg,
        ) and _is_dense_wrong(baseline, max_te_cm=dense_wrong_te_cm, max_re_deg=dense_wrong_re_deg):
            baseline_sparse_ok_dense_wrong += 1
            categories.append("baseline_sparse_correct_dense_wrong")
        if _is_sparse_correct(
            candidate,
            max_te_cm=sparse_correct_te_cm,
            max_re_deg=sparse_correct_re_deg,
        ) and _is_dense_wrong(candidate, max_te_cm=dense_wrong_te_cm, max_re_deg=dense_wrong_re_deg):
            candidate_sparse_ok_dense_wrong += 1
            categories.append("candidate_sparse_correct_dense_wrong")

        b_dense = _float(baseline.get("dense_te_cm"))
        c_dense = _float(candidate.get("dense_te_cm"))
        delta_dense = None
        if b_dense is not None and c_dense is not None:
            delta_dense = c_dense - b_dense
            if delta_dense >= regression_20cm:
                candidate_regression_20 += 1
                categories.append("candidate_regression_20cm")
            if delta_dense >= regression_50cm:
                candidate_regression_50 += 1
                categories.append("candidate_regression_50cm")
            if delta_dense <= -regression_20cm:
                candidate_improvement_20 += 1
                categories.append("candidate_improvement_20cm")

        pairs.append(
            {
                "scene": scene,
                "split_name": split_name,
                "query_index": index,
                "query_id": candidate["query_id"],
                "baseline_sparse_te_cm": baseline.get("sparse_te_cm"),
                "baseline_sparse_re_deg": baseline.get("sparse_re_deg"),
                "baseline_dense_te_cm": baseline.get("dense_te_cm"),
                "baseline_dense_re_deg": baseline.get("dense_re_deg"),
                "candidate_sparse_te_cm": candidate.get("sparse_te_cm"),
                "candidate_sparse_re_deg": candidate.get("sparse_re_deg"),
                "candidate_dense_te_cm": candidate.get("dense_te_cm"),
                "candidate_dense_re_deg": candidate.get("dense_re_deg"),
                "delta_dense_te_cm": delta_dense,
                "candidate_dense_delta_from_sparse_cm": (
                    candidate["dense_te_cm"] - candidate["sparse_te_cm"]
                    if candidate.get("dense_te_cm") is not None and candidate.get("sparse_te_cm") is not None
                    else None
                ),
                "categories": categories,
            }
        )

    audit = {
        "schema": "loc_gs_dense_transition_audit_v1",
        "scene": scene,
        "split_name": split_name,
        "query_count": int(count),
        "baseline_dense_worsened_count": int(baseline_dense_worsened),
        "candidate_dense_worsened_count": int(candidate_dense_worsened),
        "dense_worsened_reduction_rate": _reduction_rate(baseline_dense_worsened, candidate_dense_worsened),
        "baseline_sparse_correct_dense_wrong_count": int(baseline_sparse_ok_dense_wrong),
        "candidate_sparse_correct_dense_wrong_count": int(candidate_sparse_ok_dense_wrong),
        "sparse_correct_dense_wrong_reduction_rate": _reduction_rate(
            baseline_sparse_ok_dense_wrong,
            candidate_sparse_ok_dense_wrong,
        ),
        "baseline_dense_improved_count": int(baseline_dense_improved),
        "candidate_dense_improved_count": int(candidate_dense_improved),
        "candidate_regression_20cm_count": int(candidate_regression_20),
        "candidate_regression_50cm_count": int(candidate_regression_50),
        "candidate_improvement_20cm_count": int(candidate_improvement_20),
        "thresholds": {
            "sparse_correct_te_cm": float(sparse_correct_te_cm),
            "sparse_correct_re_deg": float(sparse_correct_re_deg),
            "dense_wrong_te_cm": float(dense_wrong_te_cm),
            "dense_wrong_re_deg": float(dense_wrong_re_deg),
            "dense_worsen_margin_cm": float(dense_worsen_margin_cm),
            "regression_20cm": float(regression_20cm),
            "regression_50cm": float(regression_50cm),
        },
    }
    hard_cases = [row for row in pairs if row["categories"]]
    return audit, hard_cases


def _write_hard_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "scene",
        "split_name",
        "query_index",
        "query_id",
        "categories",
        "baseline_sparse_te_cm",
        "baseline_dense_te_cm",
        "candidate_sparse_te_cm",
        "candidate_dense_te_cm",
        "delta_dense_te_cm",
        "candidate_dense_delta_from_sparse_cm",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["categories"] = ";".join(str(item) for item in flat.get("categories", []))
            writer.writerow(flat)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v10 sparse-to-dense transition audit for dense verifier gates.")
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--candidate_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--allow_test_split", action="store_true")
    parser.add_argument("--sparse_correct_te_cm", type=float, default=5.0)
    parser.add_argument("--sparse_correct_re_deg", type=float, default=5.0)
    parser.add_argument("--dense_wrong_te_cm", type=float, default=10.0)
    parser.add_argument("--dense_wrong_re_deg", type=float, default=5.0)
    parser.add_argument("--dense_worsen_margin_cm", type=float, default=5.0)
    parser.add_argument("--regression_20cm", type=float, default=20.0)
    parser.add_argument("--regression_50cm", type=float, default=50.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip() or "unknown"
    if split_name.lower() == "test" and not bool(args.allow_test_split):
        raise ValueError("test split dense transition audits are not allowed without --allow_test_split")

    audit, hard_cases = build_dense_transition_audit(
        _load_results(args.baseline_results),
        _load_results(args.candidate_results),
        scene=str(args.scene),
        split_name=split_name,
        sparse_correct_te_cm=float(args.sparse_correct_te_cm),
        sparse_correct_re_deg=float(args.sparse_correct_re_deg),
        dense_wrong_te_cm=float(args.dense_wrong_te_cm),
        dense_wrong_re_deg=float(args.dense_wrong_re_deg),
        dense_worsen_margin_cm=float(args.dense_worsen_margin_cm),
        regression_20cm=float(args.regression_20cm),
        regression_50cm=float(args.regression_50cm),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "dense_transition_audit.json"
    hard_cases_path = output_dir / "hard_dense_cases.csv"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    _write_hard_cases(hard_cases_path, hard_cases)

    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": split_name,
        "recipe": "lsf_v10_dense_transition_audit",
        "artifact": str(audit_path),
        "single_path_deployment": True,
        "branch_selection": False,
        "paper_safe_for_tuning": split_name.lower() != "test",
    }
    manifest = {
        "method": "loc_gs_lsf_v10_dense_transition_audit",
        **metadata,
        "baseline_results": str(args.baseline_results),
        "candidate_results": str(args.candidate_results),
        "hard_cases": str(hard_cases_path),
    }
    split_audit = artifact_split_audit(metadata, branch_selection=False)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=audit,
        split_audit=split_audit,
    )
    print(json.dumps({"audit": str(audit_path), "hard_cases": str(hard_cases_path), **audit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
