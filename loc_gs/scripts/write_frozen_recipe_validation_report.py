#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loc_gs.reporting.frozen_recipe_validation import build_frozen_recipe_validation_report


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
    return raw if raw.is_file() else raw / "metrics_summary.json"


def _load_run(path: str | Path, *, scene: str) -> dict[str, Any]:
    root = _run_dir(path)
    metrics = _load_json(_metrics_summary_path(path))
    manifest = _load_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    split_audit = _load_json(root / "split_audit.json") if (root / "split_audit.json").exists() else {}
    return {
        "scene": scene,
        "run_dir": str(root),
        "metrics_summary_path": str(_metrics_summary_path(path)),
        "dense": metrics.get("dense", {}),
        "sparse": metrics.get("sparse", {}),
        "manifest": {
            "split": manifest.get("split"),
            "map_path": manifest.get("map_path"),
            "single_path_evaluator": manifest.get("single_path_evaluator"),
            "feedback": manifest.get("feedback", {}),
            "feedback_enabled": manifest.get("feedback_enabled"),
            "residual_enabled": manifest.get("residual_enabled"),
            "selector_enabled": manifest.get("selector_enabled"),
            "rho_feedback_enabled": manifest.get("rho_feedback_enabled"),
            "quality_gate": manifest.get("quality_gate", {}),
        },
        "split_audit": {
            "audit_status": split_audit.get("audit_status", "unknown"),
            "paper_safe": split_audit.get("paper_safe", False),
            "checks": split_audit.get("checks", {}),
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
    parser = argparse.ArgumentParser(
        description="Validate a frozen scene-level recipe against native baselines without reselection."
    )
    parser.add_argument("--baseline", action="append", default=[], help="Scene=RUN_DIR or Scene=metrics_summary.json")
    parser.add_argument("--selected", action="append", default=[], help="Scene=RUN_DIR or Scene=metrics_summary.json")
    parser.add_argument("--required-positive-scene", action="append", default=[])
    parser.add_argument("--neutral-scene", action="append", default=[])
    parser.add_argument("--max-median-te-regression-cm", type=float, default=0.0)
    parser.add_argument("--max-recall-drop", type=float, default=0.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    baseline: dict[str, dict[str, Any]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for spec in args.baseline:
        scene, path = _parse_scene_spec(spec)
        baseline[scene] = _load_run(path, scene=scene)
    for spec in args.selected:
        scene, path = _parse_scene_spec(spec)
        selected[scene] = _load_run(path, scene=scene)

    payload = build_frozen_recipe_validation_report(
        baseline,
        selected,
        required_positive_scenes=args.required_positive_scene,
        neutral_scenes=args.neutral_scene,
        max_median_te_regression_cm=args.max_median_te_regression_cm,
        max_recall_drop=args.max_recall_drop,
        split=args.split,
        notes=args.notes,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "passed": payload["global_gate"]["passed"],
                "paper_safe_validation_ready": payload["checks"]["paper_safe_validation_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
