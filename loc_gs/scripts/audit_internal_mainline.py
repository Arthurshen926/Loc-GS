#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.audit import assert_internal_mainline_sources


INTERNAL_MAINLINE_RELATIVE_PATHS: tuple[str, ...] = (
    "core/camera.py",
    "core/geometry.py",
    "core/metrics.py",
    "core/pnp.py",
    "data/superpoint_query_extractor.py",
    "sparse/audit.py",
    "sparse/artifact_adapter.py",
    "sparse/candidate_artifact_merge.py",
    "sparse/cached_eval.py",
    "sparse/candidate_completion_plan.py",
    "sparse/candidate_completion_runner.py",
    "sparse/candidate_coverage.py",
    "sparse/candidate_shard_builder.py",
    "sparse/correspondences.py",
    "sparse/gate.py",
    "sparse/landmarks.py",
    "sparse/pipeline.py",
    "sparse/pose_map_frame_audit.py",
    "sparse/query_feature_cache.py",
    "sparse/real_inputs.py",
    "sparse/rerank.py",
    "sparse/results_metrics.py",
    "sparse/sparse_lgcv.py",
    "students/descriptor_fusion.py",
    "students/detector_student.py",
    "students/landmark_selector.py",
    "simulation/render_manifest.py",
    "simulation/render_runner.py",
    "simulation/query_sampler.py",
    "teacher/distillation_artifact.py",
    "teacher/geometric_observations.py",
    "teacher/labels.py",
    "teacher/online_episode.py",
    "teacher/render_teacher_jobs.py",
    "teacher/solver_feedback.py",
    "training/sparse_candidate_scorer.py",
    "scripts/build_internal_distillation_artifact.py",
    "scripts/build_internal_candidate_completion_plan.py",
    "scripts/build_internal_candidate_shard_artifact.py",
    "scripts/build_internal_candidate_shards_from_query_cache.py",
    "scripts/build_internal_candidate_shard_from_feature_maps.py",
    "scripts/build_internal_query_feature_cache.py",
    "scripts/build_internal_query_feature_cache_from_images.py",
    "scripts/build_internal_query_feature_cache_from_maps.py",
    "scripts/build_internal_query_feature_cache_from_superpoint_dir.py",
    "scripts/build_internal_render_manifest.py",
    "scripts/build_internal_render_teacher_jobs.py",
    "scripts/build_internal_solver_feedback_labels.py",
    "scripts/build_internal_simulation_plan.py",
    "scripts/build_internal_sparse_metrics_from_results.py",
    "scripts/build_internal_teacher_observations_from_geometry.py",
    "scripts/audit_internal_candidate_coverage.py",
    "scripts/audit_internal_pose_map_frame.py",
    "scripts/merge_internal_candidate_artifacts.py",
    "scripts/eval_internal_sparse_cached.py",
    "scripts/eval_sparse_distilled_cambridge.py",
    "scripts/extract_internal_superpoint_features_for_queries.py",
    "scripts/render_internal_3dgs_assets.py",
    "scripts/run_internal_mainline_smoke.py",
    "scripts/run_internal_sparse_gate.py",
    "scripts/run_internal_sparse_smoke.py",
    "scripts/train_internal_sparse_candidate_scorer.py",
    "scripts/train_internal_sparse_students.py",
)


def internal_mainline_source_paths(root: str | Path) -> list[Path]:
    base = Path(root)
    return [base / rel for rel in INTERNAL_MAINLINE_RELATIVE_PATHS]


def run_internal_mainline_audit(paths: Sequence[str | Path]) -> dict[str, int]:
    assert_internal_mainline_sources(paths)
    return {
        "checked_file_count": int(len(paths)),
        "forbidden_hit_count": 0,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit internal sparse-distilled mainline runtime dependencies.")
    parser.add_argument("--root", type=Path, default=Path("loc_gs"))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    result = run_internal_mainline_audit(internal_mainline_source_paths(args.root))
    if bool(args.json):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"checked {result['checked_file_count']} internal mainline files; forbidden hits: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
