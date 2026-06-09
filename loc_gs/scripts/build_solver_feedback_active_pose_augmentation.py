#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loc_gs.feedback.active_pose_augmentation import (
    SCHEMA_VERSION,
    build_active_pose_augmentation,
    candidate_poses_from_observations,
)
from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_repo_root()), text=True).strip()
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(_repo_root()), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _command(argv: list[str] | None) -> str:
    values = sys.argv if argv is None else [sys.argv[0], *argv]
    return " ".join(shlex.quote(str(item)) for item in values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path, *, record_types: set[str], payload_keys: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {line_no} must be a JSON object")
            if item.get("type") == "manifest":
                continue
            if str(item.get("type", "")) in record_types:
                for key in payload_keys:
                    if key in item:
                        yield dict(item[key])
                        break
                else:
                    yield dict(item)
            else:
                yield dict(item)


def _read_jsonl_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {line_no} must be a JSON object")
            if item.get("type") == "manifest":
                return dict(item.get("manifest", {}))
            break
    return {}


def load_records(
    path: Path,
    *,
    list_key: str,
    record_types: set[str],
    payload_keys: tuple[str, ...],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_manifest(path), _iter_jsonl(path, record_types=record_types, payload_keys=payload_keys)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    manifest = dict(payload.get("manifest", {}))
    return manifest, (dict(item) for item in payload.get(list_key, []))


def _manifest_split(manifest: dict[str, Any], default: str = "unknown") -> str:
    split = str(manifest.get("split_name", manifest.get("split", default))).strip() or default
    if split != "unknown":
        validate_split_name(split)
    return split


def _write_selected_csv(path: Path, selected_poses: list[dict[str, Any]]) -> None:
    fieldnames = [
        "selection_rank",
        "pose_id",
        "anchor_view_id",
        "pose_source",
        "query_ids",
        "alpha_valid_ratio",
        "selected_landmark_visible_count",
        "candidate_landmark_visible_count",
        "feature_variance",
        "artifact_score",
        "novelty_score",
        "expected_query_gain",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_poses:
            out = {key: row.get(key, "") for key in fieldnames}
            out["query_ids"] = "|".join(str(item) for item in row.get("query_ids", []))
            writer.writerow(out)


def _split_audit(split_name: str, candidate_split: str, observation_split: str) -> dict[str, Any]:
    unknown = candidate_split == "unknown" or observation_split == "unknown"
    return {
        "audit_status": "unknown" if unknown else "passed",
        "paper_safe": not unknown,
        "split_name": split_name,
        "candidate_split_name": candidate_split,
        "observation_split_name": observation_split,
        "test_split_used": False,
        "notes": (
            "Missing source split information is marked unknown and should stay out of paper-safe tables."
            if unknown
            else "Active pose augmentation builder rejects test split inputs."
        ),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build solver-feedback active pose augmentation and view-landmark fusion plan."
    )
    parser.add_argument("--candidate_poses", type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--infer_candidate_poses_from_observations", action="store_true")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--num_gaussians", required=True, type=int)
    parser.add_argument("--hard_query_id", action="append", default=[])
    parser.add_argument("--max_poses", default=32, type=int)
    parser.add_argument("--max_per_anchor", default=4, type=int)
    parser.add_argument("--min_alpha_valid", default=0.5, type=float)
    parser.add_argument("--min_stable_mask_ratio", default=0.0, type=float)
    parser.add_argument("--min_selected_landmark_visible", default=1, type=int)
    parser.add_argument("--min_candidate_landmark_visible", default=1, type=int)
    parser.add_argument("--min_feature_variance", default=0.0, type=float)
    parser.add_argument("--max_artifact_score", default=0.5, type=float)
    parser.add_argument("--reprojection_threshold_px", default=4.0, type=float)
    parser.add_argument("--min_positive_observations", default=1, type=int)
    parser.add_argument("--max_negative_observations", default=0, type=int)
    parser.add_argument("--max_fusion_artifact_score", default=0.3, type=float)
    parser.add_argument("--max_observations", default=0, type=int)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--reliability_sample_limit", default=64, type=int)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--map_path", default="")
    parser.add_argument("--data_root", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = validate_split_name(str(args.split_name))
    observation_manifest, observation_records = load_records(
        Path(args.observations),
        list_key="observations",
        record_types={"observation", "record"},
        payload_keys=("observation", "record"),
    )
    observation_records_list = list(observation_records)
    if int(args.max_observations) > 0:
        observation_records_list = observation_records_list[: int(args.max_observations)]
    if args.candidate_poses is not None:
        candidate_manifest, candidate_records_iter = load_records(
            Path(args.candidate_poses),
            list_key="candidate_poses",
            record_types={"candidate_pose", "pose", "record"},
            payload_keys=("candidate_pose", "pose", "record"),
        )
        candidate_records = list(candidate_records_iter)
    elif bool(args.infer_candidate_poses_from_observations):
        candidate_manifest = dict(observation_manifest)
        candidate_records = candidate_poses_from_observations(
            observation_records_list,
            split_name=split_name,
        )
    else:
        raise ValueError("--candidate_poses is required unless --infer_candidate_poses_from_observations is set")
    candidate_split = _manifest_split(candidate_manifest)
    observation_split = _manifest_split(observation_manifest)

    artifact = build_active_pose_augmentation(
        candidate_poses=candidate_records,
        observations=observation_records_list,
        num_gaussians=int(args.num_gaussians),
        scene=str(args.scene),
        split_name=split_name,
        hard_query_ids=args.hard_query_id,
        max_poses=int(args.max_poses),
        max_per_anchor=int(args.max_per_anchor),
        min_alpha_valid=float(args.min_alpha_valid),
        min_stable_mask_ratio=float(args.min_stable_mask_ratio),
        min_selected_landmark_visible=int(args.min_selected_landmark_visible),
        min_candidate_landmark_visible=int(args.min_candidate_landmark_visible),
        min_feature_variance=float(args.min_feature_variance),
        max_artifact_score=float(args.max_artifact_score),
        reprojection_threshold_px=float(args.reprojection_threshold_px),
        min_positive_observations=int(args.min_positive_observations),
        max_negative_observations=int(args.max_negative_observations),
        max_fusion_artifact_score=float(args.max_fusion_artifact_score),
        compact=bool(args.compact),
        reliability_sample_limit=int(args.reliability_sample_limit),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_path = output_dir / "active_pose_augmentation.json"
    reliability_path = output_dir / "view_landmark_reliability.json"
    fusion_path = output_dir / "landmark_fusion_plan.json"
    selected_csv_path = output_dir / "selected_poses.csv"
    _write_json(active_path, artifact)
    _write_json(
        reliability_path,
        {
            "schema_version": SCHEMA_VERSION,
            "scene": str(args.scene),
            "split_name": split_name,
            "view_landmark_reliability": artifact["view_landmark_reliability"],
            "view_landmark_reliability_sample": artifact["view_landmark_reliability_sample"],
            "compact_output": bool(args.compact),
        },
    )
    _write_json(
        fusion_path,
        {
            "schema_version": SCHEMA_VERSION,
            "scene": str(args.scene),
            "split_name": split_name,
            "landmark_fusion_plan": artifact["landmark_fusion_plan"],
        },
    )
    _write_selected_csv(selected_csv_path, artifact["selected_poses"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": _command(argv),
        "scene": str(args.scene),
        "split_name": split_name,
        "candidate_poses": str(Path(args.candidate_poses)) if args.candidate_poses is not None else "inferred",
        "observations": str(Path(args.observations)),
        "checkpoint_path": str(args.checkpoint_path).strip() or candidate_manifest.get("checkpoint_path", "unknown"),
        "map_path": str(args.map_path).strip() or candidate_manifest.get("map_path", "unknown"),
        "data_roots": [str(item) for item in args.data_root] or candidate_manifest.get("data_roots", ["unknown"]),
        "artifact_path": str(active_path),
        "source_manifests": {
            "candidate_poses": candidate_manifest,
            "observations": observation_manifest,
        },
        "candidate_pose_source": (
            "observation_aggregate"
            if args.candidate_poses is None and bool(args.infer_candidate_poses_from_observations)
            else "candidate_pose_file"
        ),
        "hyperparameters": artifact["hyperparameters"],
        "max_observations": int(args.max_observations),
        "compact_output": bool(args.compact),
        "reliability_sample_limit": int(args.reliability_sample_limit),
    }
    split_audit = _split_audit(split_name, candidate_split, observation_split)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics_summary.json", artifact["metrics"])
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_path": str(active_path),
                "metrics_path": str(output_dir / "metrics_summary.json"),
                "selected_poses_path": str(selected_csv_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
