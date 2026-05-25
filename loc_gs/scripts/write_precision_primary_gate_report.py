#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loc_gs.reporting.precision_primary_gate import evaluate_precision_primary_gate


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _run_dir(path: str | Path) -> Path:
    raw = Path(path)
    return raw.parent if raw.is_file() else raw


def _metrics_summary_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_file() else _run_dir(raw) / "metrics_summary.json"


def _load_run(path: str | Path, *, scene: str) -> dict[str, Any]:
    root = _run_dir(path)
    metrics_path = _metrics_summary_path(path)
    metrics = _load_json(metrics_path)
    manifest = _load_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    split_audit = _load_json(root / "split_audit.json") if (root / "split_audit.json").exists() else {}
    return {
        "scene": scene,
        "run_dir": str(root),
        "metrics_summary_path": str(metrics_path),
        "dense": metrics.get("dense", {}),
        "sparse": metrics.get("sparse", {}),
        "manifest": {
            "split": manifest.get("split"),
            "map_path": manifest.get("map_path"),
            "single_path_evaluator": manifest.get("single_path_evaluator"),
            "run_role": manifest.get("run_role"),
        },
        "split_audit": {
            "audit_status": split_audit.get("audit_status", "unknown"),
            "paper_safe": split_audit.get("paper_safe", False),
        },
    }


def _parse_scene_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"run spec must be Scene=PATH, got {spec!r}")
    scene, path = spec.split("=", 1)
    scene = scene.strip()
    if not scene:
        raise ValueError(f"missing scene in run spec {spec!r}")
    return scene, path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a precision-primary pose gate report.")
    parser.add_argument("--baseline", action="append", default=[], help="Scene=RUN_DIR or Scene=metrics_summary.json")
    parser.add_argument("--candidate", action="append", default=[], help="Scene=RUN_DIR or Scene=metrics_summary.json")
    parser.add_argument("--required-positive-scene", action="append", default=[])
    parser.add_argument("--neutral-scene", action="append", default=[])
    parser.add_argument("--max-recall-drop", type=float, default=0.002)
    parser.add_argument("--min-median-te-gain-cm", type=float, default=0.0)
    parser.add_argument("--min-median-re-gain-deg", type=float, default=0.0)
    parser.add_argument("--max-abs-median-te-cm", type=float, default=100.0)
    parser.add_argument("--max-abs-median-re-deg", type=float, default=5.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    baseline: dict[str, dict[str, Any]] = {}
    candidate: dict[str, dict[str, Any]] = {}
    for spec in args.baseline:
        scene, path = _parse_scene_spec(spec)
        baseline[scene] = _load_run(path, scene=scene)
    for spec in args.candidate:
        scene, path = _parse_scene_spec(spec)
        candidate[scene] = _load_run(path, scene=scene)

    gate = evaluate_precision_primary_gate(
        baseline,
        candidate,
        required_positive_scenes=args.required_positive_scene,
        neutral_scenes=args.neutral_scene,
        max_recall_drop=args.max_recall_drop,
        min_median_te_gain_cm=args.min_median_te_gain_cm,
        min_median_re_gain_deg=args.min_median_re_gain_deg,
        max_abs_median_te_cm=args.max_abs_median_te_cm,
        max_abs_median_re_deg=args.max_abs_median_re_deg,
    )
    payload = {
        "format": "loc_gs_precision_primary_gate_v1",
        "split": args.split,
        "notes": args.notes,
        "baseline_scenes": sorted(baseline),
        "candidate_scenes": sorted(candidate),
        "gate": gate,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "passed": gate["passed"],
                "macro_delta": gate["macro_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
