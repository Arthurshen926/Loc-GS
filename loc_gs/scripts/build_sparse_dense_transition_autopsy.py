#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _first_float(source: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _finite_float(source.get(key), default=None)
        if value is not None:
            return value
    return None


def _nested_float(source: Mapping[str, Any], keys: tuple[str, ...], nested_keys: tuple[str, ...] = ("dense",)) -> float | None:
    value = _first_float(source, keys)
    if value is not None:
        return value
    for nested_key in nested_keys:
        nested = source.get(nested_key)
        if isinstance(nested, Mapping):
            value = _first_float(nested, keys)
            if value is not None:
                return value
    return None


def _nested_decision(source: Mapping[str, Any]) -> str:
    nested_mappings: list[Mapping[str, Any]] = []

    def visit(mapping: Mapping[str, Any]) -> None:
        nested_mappings.append(mapping)
        for value in mapping.values():
            if isinstance(value, Mapping):
                visit(value)

    visit(source)
    for key in ("transition_control", "slcdp_transition_control", "soft_sdcg"):
        for mapping in nested_mappings:
            nested = mapping.get(key)
            if isinstance(nested, Mapping) and nested.get("decision") is not None:
                return str(nested.get("decision"))
    for key in ("dense_pgsh", "anchor_conditioned_patch_dense"):
        for mapping in nested_mappings:
            nested = mapping.get(key)
            if isinstance(nested, Mapping) and nested.get("decision") is not None:
                return str(nested.get("decision"))
    for key in ("dense_pgsh_decision", "decision"):
        for mapping in nested_mappings:
            if mapping.get(key) is not None:
                return str(mapping.get(key))
    return "unknown"


def _case_key(source: Mapping[str, Any], fallback: str) -> str:
    for key in ("image_name", "query_id", "case_id"):
        if source.get(key):
            return str(source.get(key))
    return fallback


def _summary_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    paths = sorted(path.rglob("pgsh_summary.json"))
    if paths:
        return paths
    return sorted(path.rglob("*.json"))


def _extract_case(*, variant: str, path: Path, payload: Mapping[str, Any], sparse_good_cm: float) -> dict[str, Any]:
    sparse_te = _first_float(payload, ("captured_sparse_te_cm", "sparse_te_cm", "base_sparse_te_cm"))
    dense_te = _nested_float(payload, ("captured_dense_te_cm", "dense_te_cm", "base_dense_te_cm"))
    effective_te = _nested_float(
        payload,
        (
            "dense_pgsh_effective_te_cm",
            "soft_sdcg_effective_te_cm",
            "slcdp_effective_te_cm",
            "captured_dense_te_cm",
            "dense_te_cm",
        ),
    )
    decision = _nested_decision(payload)
    dense_worsened = bool(sparse_te is not None and dense_te is not None and dense_te > sparse_te)
    effective_worsened = bool(sparse_te is not None and effective_te is not None and effective_te > sparse_te)
    sparse_good_dense_bad = bool(
        sparse_te is not None
        and dense_te is not None
        and sparse_te <= float(sparse_good_cm)
        and dense_te > float(sparse_good_cm)
    )
    return {
        "variant": str(variant),
        "scene": str(payload.get("scene", "")),
        "query_id": _case_key(payload, path.stem),
        "path": str(path),
        "decision": decision,
        "sparse_te_cm": sparse_te,
        "dense_te_cm": dense_te,
        "effective_te_cm": effective_te,
        "dense_worsened": dense_worsened,
        "effective_worsened": effective_worsened,
        "sparse_good_dense_bad": sparse_good_dense_bad,
        "diagnostic_only": True,
    }


def _summarize_variant(cases: list[dict[str, Any]]) -> dict[str, Any]:
    effective = np.asarray(
        [case["effective_te_cm"] for case in cases if case.get("effective_te_cm") is not None],
        dtype=np.float64,
    )
    dense = np.asarray([case["dense_te_cm"] for case in cases if case.get("dense_te_cm") is not None], dtype=np.float64)
    sparse = np.asarray([case["sparse_te_cm"] for case in cases if case.get("sparse_te_cm") is not None], dtype=np.float64)
    return {
        "case_count": int(len(cases)),
        "sparse_count": int(sparse.size),
        "dense_count": int(dense.size),
        "effective_count": int(effective.size),
        "median_sparse_te_cm": float(np.median(sparse)) if sparse.size else None,
        "median_dense_te_cm": float(np.median(dense)) if dense.size else None,
        "median_effective_te_cm": float(np.median(effective)) if effective.size else None,
        "dense_worsened_count": int(sum(bool(case.get("dense_worsened")) for case in cases)),
        "effective_worsened_count": int(sum(bool(case.get("effective_worsened")) for case in cases)),
        "sparse_good_dense_bad_count": int(sum(bool(case.get("sparse_good_dense_bad")) for case in cases)),
        "diagnostic_only": True,
    }


def _parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("--variant must have form label=path")
    label, raw_path = str(value).split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("variant label must be non-empty")
    return label, Path(raw_path).expanduser()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "scene",
        "query_id",
        "decision",
        "sparse_te_cm",
        "dense_te_cm",
        "effective_te_cm",
        "dense_worsened",
        "effective_worsened",
        "sparse_good_dense_bad",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Sparse-Dense Transition Autopsy",
        "",
        "Diagnostic-only. Do not use official test hard cases for route selection or tuning.",
        "",
        "| Variant | Cases | Median Sparse TE | Median Dense TE | Median Effective TE | Dense Worsened | Effective Worsened | Sparse-Good Dense-Bad |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in summary["variants"].items():
        lines.append(
            "| {label} | {case_count} | {median_sparse_te_cm} | {median_dense_te_cm} | {median_effective_te_cm} | {dense_worsened_count} | {effective_worsened_count} | {sparse_good_dense_bad_count} |".format(
                label=label,
                **metrics,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", default=[], help="Variant input as label=path")
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sparse_good_cm", type=float, default=20.0)
    parser.add_argument("--allow_test_diagnostic", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.variant:
        raise SystemExit("at least one --variant label=path is required")
    split_name = str(args.split_name)
    if split_name.lower() == "test" and not bool(args.allow_test_diagnostic):
        raise SystemExit("test split autopsy requires --allow_test_diagnostic and must remain diagnostic-only")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    inputs: dict[str, str] = {}
    for item in args.variant:
        label, raw_path = _parse_variant(item)
        inputs[label] = str(raw_path)
        variant_cases: list[dict[str, Any]] = []
        for path in _summary_paths(raw_path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                continue
            variant_cases.append(
                _extract_case(
                    variant=label,
                    path=path,
                    payload=payload,
                    sparse_good_cm=float(args.sparse_good_cm),
                )
            )
        by_variant[label] = variant_cases
        all_cases.extend(variant_cases)

    summary = {
        "schema": "loc_gs_sparse_dense_transition_autopsy_v1",
        "split_name": split_name,
        "variant_count": int(len(by_variant)),
        "case_count": int(len(all_cases)),
        "variants": {label: _summarize_variant(cases) for label, cases in by_variant.items()},
        "diagnostic_only": True,
    }
    metrics_summary = {
        "recipe": "sparse_dense_transition_autopsy",
        "split_name": split_name,
        "variant_count": int(len(by_variant)),
        "case_count": int(len(all_cases)),
        "diagnostic_only": True,
        "variants": summary["variants"],
    }
    manifest = {
        "recipe": "sparse_dense_transition_autopsy",
        "git_commit": _git_commit(),
        "command": _command(),
        "split_name": split_name,
        "inputs": inputs,
        "diagnostic_only": True,
    }
    split_audit = artifact_split_audit({"split_name": split_name}, branch_selection=False)

    (output_dir / "autopsy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "variant_cases.csv", all_cases)
    _write_report(output_dir / "report.md", summary)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics_summary,
        split_audit=split_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
