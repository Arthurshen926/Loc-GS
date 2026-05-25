#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loc_gs.reporting.fixed_recipe_gate import build_scene_level_acceptance_recipe


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _run_dir(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_file():
        return raw.parent
    return raw


def _metrics_summary_path(path: str | Path) -> Path:
    root = _run_dir(path)
    if Path(path).is_file():
        return Path(path)
    return root / "metrics_summary.json"


def _load_run(path: str | Path, *, scene: str) -> dict[str, Any]:
    root = _run_dir(path)
    metrics_path = _metrics_summary_path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics_summary.json not found: {metrics_path}")
    metrics = _load_json(metrics_path)
    manifest = _load_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    split_audit = _load_json(root / "split_audit.json") if (root / "split_audit.json").exists() else {}
    return {
        "scene": scene,
        "run_dir": str(root),
        "metrics_summary_path": str(metrics_path),
        "dense": metrics.get("dense", {}),
        "sparse": metrics.get("sparse", {}),
        "landmark_count": metrics.get("landmark_count", metrics.get("sampled_count")),
        "map_size_mb": metrics.get("map_size_mb"),
        "manifest": {
            "split": manifest.get("split"),
            "map_path": manifest.get("map_path"),
            "data_roots": manifest.get("data_roots", []),
            "hyperparameters": manifest.get("hyperparameters", {}),
            "single_path_evaluator": manifest.get("single_path_evaluator"),
            "feedback_enabled": manifest.get("feedback_enabled"),
            "rho_feedback_enabled": manifest.get("rho_feedback_enabled"),
        },
        "data_roots": manifest.get("data_roots", []),
        "hyperparameters": manifest.get("hyperparameters", {}),
        "split_audit": {
            "audit_status": split_audit.get("audit_status", "unknown"),
            "paper_safe": split_audit.get("paper_safe", False),
        },
    }


def _parse_baseline_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"baseline spec must be Scene=PATH, got {spec!r}")
    scene, path = spec.split("=", 1)
    scene = scene.strip()
    if not scene:
        raise ValueError(f"missing scene in baseline spec {spec!r}")
    return scene, path


def _parse_candidate_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec or ":" not in spec.split("=", 1)[0]:
        raise ValueError(f"candidate spec must be Candidate:Scene=PATH, got {spec!r}")
    left, path = spec.split("=", 1)
    name, scene = left.split(":", 1)
    name = name.strip()
    scene = scene.strip()
    if not name or not scene:
        raise ValueError(f"missing candidate or scene in candidate spec {spec!r}")
    return name, scene, path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a train-dev fixed recipe scene-level acceptance gate report."
    )
    parser.add_argument("--baseline", action="append", default=[], help="Scene=RUN_DIR or Scene=metrics_summary.json")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="CandidateName:Scene=RUN_DIR or CandidateName:Scene=metrics_summary.json",
    )
    parser.add_argument("--candidate-order", nargs="*", default=[])
    parser.add_argument("--required-positive-scene", action="append", default=[])
    parser.add_argument("--neutral-scene", action="append", default=[])
    parser.add_argument("--max-median-te-regression-cm", type=float, default=0.0)
    parser.add_argument("--max-recall-drop", type=float, default=0.0)
    parser.add_argument("--recall-policy", choices=("hard", "warn"), default="hard")
    parser.add_argument("--split", default="train-dev")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    baseline: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.baseline:
        scene, path = _parse_baseline_spec(spec)
        baseline[scene] = _load_run(path, scene=scene)
    for spec in args.candidate:
        name, scene, path = _parse_candidate_spec(spec)
        candidates.setdefault(name, {})[scene] = _load_run(path, scene=scene)

    report = build_scene_level_acceptance_recipe(
        baseline,
        candidates,
        candidate_order=args.candidate_order,
        required_positive_scenes=args.required_positive_scene,
        neutral_scenes=args.neutral_scene,
        max_median_te_regression_cm=args.max_median_te_regression_cm,
        max_recall_drop=args.max_recall_drop,
        recall_policy=args.recall_policy,
    )
    payload = {
        "split": args.split,
        "notes": args.notes,
        "baseline_scenes": sorted(baseline),
        "candidate_names": sorted(candidates),
        "recipe": report,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "passed": payload["recipe"]["global_gate"]["passed"],
                "selected": {
                    scene: row["selected"]
                    for scene, row in payload["recipe"]["selected_runs"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
