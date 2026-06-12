#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.cached_eval import CachedSparseEvalConfig, run_cached_sparse_eval
from loc_gs.sparse.candidate_shard_builder import build_candidate_shard_artifact, load_completion_shard
from loc_gs.sparse.query_feature_cache import build_query_feature_cache_from_feature_map_cache
from loc_gs.teacher.distillation_artifact import (
    DistillationArtifactConfig,
    build_distillation_payload_from_observations,
    load_teacher_observation_rows,
)
from loc_gs.teacher.geometric_observations import build_teacher_observations_from_geometry
from loc_gs.training.sparse_candidate_scorer import (
    CandidateScorerConfig,
    classify_feature_input_policy,
    train_candidate_scorer,
)
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact


SPARSE_ONLY_SAFE_FEATURES = (
    "native_score",
    "negative_rank",
    "valid",
    "margin",
    "query_score",
    "landmark_prior",
)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except Exception as exc:
        return f"unknown: {exc}\n"


def _feature_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("feature_names must contain at least one feature")
    return names


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an internal sparse-dense-distilled mainline smoke.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--completion_shard", type=Path, required=True)
    parser.add_argument("--shard_id", default=None)
    parser.add_argument("--feature_map_cache", type=Path, required=True)
    parser.add_argument("--base_candidate_artifact", type=Path, required=True)
    parser.add_argument("--teacher_observations", type=Path, default=None)
    parser.add_argument("--point_cloud", type=Path, required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--max_keypoints", type=int, default=2048)
    parser.add_argument("--max_landmarks", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--feature_names", default=",".join(SPARSE_ONLY_SAFE_FEATURES))
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--missing_principal_point", choices=["half_extent", "pixel_center"], default="pixel_center")
    parser.add_argument("--pnp_method", choices=["epnp", "iterative"], default="epnp")
    parser.add_argument("--pnp_iterations", type=int, default=10000)
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--dense_consistency_reprojection_px", type=float, default=4.0)
    parser.add_argument("--sparse_inlier_reprojection_px", type=float, default=8.0)
    parser.add_argument("--hard_negative_reprojection_px", type=float, default=8.0)
    parser.add_argument("--second_pnp_enabled", action="store_true")
    parser.add_argument("--second_pnp_method", choices=["epnp", "iterative"], default="iterative")
    parser.add_argument("--refine_with_inliers", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal mainline smoke")
    if (args.image_width is None) != (args.image_height is None):
        raise ValueError("--image_width and --image_height must be provided together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "loc_gs.scripts.run_internal_mainline_smoke", *(argv or sys.argv[1:])]

    shard = load_completion_shard(args.completion_shard, shard_id=args.shard_id)
    query_ids = [str(value) for value in shard.get("query_ids", [])]
    query_cache = args.output_dir / "query_features.pt"
    candidate_shard = args.output_dir / "candidate_shard.pt"
    distilled_artifact = args.output_dir / "distilled_candidates.pt"
    candidate_scorer_path = args.output_dir / "candidate_scorer.json"

    query_summary = build_query_feature_cache_from_feature_map_cache(
        feature_map_cache=args.feature_map_cache,
        output_cache=query_cache,
        scene=str(args.scene),
        split_name=split,
        query_ids=query_ids,
        max_keypoints=int(args.max_keypoints),
        score_threshold=args.score_threshold,
    )
    candidate_summary = build_candidate_shard_artifact(
        completion_shard=shard,
        query_feature_cache=query_cache,
        base_candidate_artifact=args.base_candidate_artifact,
        output_artifact=candidate_shard,
        scene=str(args.scene),
        split_name=split,
        topk=int(args.topk),
        max_landmarks=args.max_landmarks,
    )
    candidate_payload = torch.load(candidate_shard, map_location="cpu")
    if not isinstance(candidate_payload, dict):
        raise ValueError(f"candidate shard must contain a dict payload: {candidate_shard}")
    teacher_observation_source = "provided"
    teacher_observations = args.teacher_observations
    geometry_observation_summary = None
    if teacher_observations is None:
        teacher_observation_source = "geometry"
        observations, geometry_observation_summary = build_teacher_observations_from_geometry(
            candidate_artifact=candidate_shard,
            point_cloud=args.point_cloud,
            cameras_json=args.cameras_json,
            scene=str(args.scene),
            split_name=split,
            target_width=args.image_width,
            target_height=args.image_height,
            missing_principal_point=str(args.missing_principal_point),
            dense_consistency_reprojection_px=float(args.dense_consistency_reprojection_px),
            sparse_inlier_reprojection_px=float(args.sparse_inlier_reprojection_px),
            hard_negative_reprojection_px=float(args.hard_negative_reprojection_px),
        )
        teacher_observations = args.output_dir / "teacher_observations.jsonl"
        with teacher_observations.open("w", encoding="utf-8") as handle:
            for row in observations:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    distilled_payload, distillation_summary = build_distillation_payload_from_observations(
        candidate_payload,
        load_teacher_observation_rows(str(teacher_observations)),
        scene=str(args.scene),
        split_name=split,
        cfg=DistillationArtifactConfig(),
    )
    torch.save(distilled_payload, distilled_artifact)

    distilled = load_listwise_candidate_artifact(distilled_artifact)
    feature_names = _feature_names(str(args.feature_names))
    feature_policy = classify_feature_input_policy(feature_names)
    model, scorer_summary = train_candidate_scorer(
        distilled,
        CandidateScorerConfig(
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            feature_names=feature_names,
        ),
    )
    candidate_scorer_path.write_text(
        json.dumps(model.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sparse_summary, sparse_rows = run_cached_sparse_eval(
        scene=str(args.scene),
        split_name=split,
        candidate_artifact=distilled_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        cfg=CachedSparseEvalConfig(
            image_width=args.image_width,
            image_height=args.image_height,
            missing_principal_point=str(args.missing_principal_point),
            max_queries=None,
            max_keypoints=int(args.max_keypoints),
            candidate_scorer=candidate_scorer_path,
            solver_weight=1.0,
            native_weight=1.0,
            reprojection_error_px=float(args.reprojection_error_px),
            pnp_iterations=int(args.pnp_iterations),
            pnp_method=str(args.pnp_method),
            second_pnp_enabled=bool(args.second_pnp_enabled),
            second_pnp_method=str(args.second_pnp_method),
            refine_with_inliers=bool(args.refine_with_inliers),
        ),
    )
    (args.output_dir / "sparse_eval_results.json").write_text(
        json.dumps(sparse_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "internal_mainline_smoke_summary_v1",
        "scene": str(args.scene),
        "split_name": split,
        "query_features": query_summary,
        "candidate_shard": candidate_summary,
        "teacher_observation_source": teacher_observation_source,
        "geometry_teacher_observations": geometry_observation_summary,
        "distillation": distillation_summary,
        "candidate_scorer": scorer_summary,
        "sparse_eval": sparse_summary,
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        **feature_policy,
    }
    manifest = {
        "schema_version": "internal_mainline_smoke_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "internal_mainline_smoke",
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        **feature_policy,
        "completion_shard": str(args.completion_shard),
        "feature_map_cache": str(args.feature_map_cache),
        "base_candidate_artifact": str(args.base_candidate_artifact),
        "teacher_observations": str(teacher_observations),
        "point_cloud": str(args.point_cloud),
        "cameras_json": str(args.cameras_json),
        "outputs": {
            "query_feature_cache": str(query_cache),
            "candidate_shard": str(candidate_shard),
            "distilled_candidates": str(distilled_artifact),
            "candidate_scorer": str(candidate_scorer_path),
            "sparse_eval_results": str(args.output_dir / "sparse_eval_results.json"),
        },
        "hyperparameters": {
            "topk": int(args.topk),
            "max_keypoints": int(args.max_keypoints),
            "max_landmarks": None if args.max_landmarks is None else int(args.max_landmarks),
            "score_threshold": None if args.score_threshold is None else float(args.score_threshold),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "feature_names": list(feature_names),
            "image_width": args.image_width,
            "image_height": args.image_height,
            "missing_principal_point": str(args.missing_principal_point),
            "pnp_method": str(args.pnp_method),
            "pnp_iterations": int(args.pnp_iterations),
            "reprojection_error_px": float(args.reprojection_error_px),
            "dense_consistency_reprojection_px": float(args.dense_consistency_reprojection_px),
            "sparse_inlier_reprojection_px": float(args.sparse_inlier_reprojection_px),
            "hard_negative_reprojection_px": float(args.hard_negative_reprojection_px),
            "second_pnp_enabled": bool(args.second_pnp_enabled),
            "second_pnp_method": str(args.second_pnp_method),
            "refine_with_inliers": bool(args.refine_with_inliers),
        },
    }
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
