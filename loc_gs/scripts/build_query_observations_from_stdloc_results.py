#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _read_split_image_names(path: str | Path) -> list[str]:
    split_path = Path(path)
    if not split_path.exists():
        return []
    names: list[str] = []
    for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first = stripped.split()[0].strip(",")
        if Path(first).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            names.append(first)
    return names


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sparse_inliers(row: Mapping[str, Any]) -> float:
    sparse = row.get("sparse")
    if isinstance(sparse, Mapping):
        return _float(sparse.get("inliers"), 0.0)
    return _float(row.get("sparse_inliers"), 0.0)


def build_observations(
    rows: list[dict[str, Any]],
    *,
    split_image_names: list[str],
    confidence_scale: float = 64.0,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        query_id = str(row.get("image_name") or row.get("query_id") or "")
        if not query_id and index < len(split_image_names):
            query_id = split_image_names[index]
        if not query_id:
            query_id = f"query_{index:06d}"
        inliers = max(0.0, _sparse_inliers(row))
        confidence = min(1.0, inliers / max(float(confidence_scale), 1.0))
        observations.append(
            {
                "query_id": query_id,
                "keypoint_count": int(max(0.0, _float(row.get("keypoint_count"), inliers))),
                "detector_centroid_yx": [0.5, 0.5],
                "topk_match_entropy": float(1.0 - confidence),
                "sparse_inliers": int(inliers),
                "pnp_confidence": float(confidence),
                "global_descriptor": [],
                "feature_source": "stdloc_sparse_inlier_proxy_no_pose_error",
            }
        )
    return observations


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v8 query-observation proxies from STDLoc results without TE/GT fields.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_file", default="")
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--allow_test_split", action="store_true")
    parser.add_argument("--confidence_scale", type=float, default=64.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip() or "unknown"
    if split_name.lower() == "test" and not bool(args.allow_test_split):
        raise ValueError("test split query observations are not allowed without --allow_test_split")
    rows = _load_results(args.results)
    observations = build_observations(
        rows,
        split_image_names=_read_split_image_names(args.split_file) if args.split_file else [],
        confidence_scale=float(args.confidence_scale),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "query_observations.json"
    observations_path.write_text(json.dumps(observations, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": split_name,
        "recipe": "lsf_v8_query_observation_proxy",
        "artifact": str(observations_path),
        "query_count": int(len(observations)),
        "uses_pose_error_or_gt": False,
        "routing_uses_gt_or_eval_result": False,
        "single_path_deployment": True,
        "branch_selection": False,
        "paper_safe_for_tuning": split_name.lower() != "test",
    }
    manifest = {
        "method": "loc_gs_lsf_v8_query_observation_proxy",
        **metadata,
        "results": str(args.results),
        "split_file": str(args.split_file),
        "confidence_scale": float(args.confidence_scale),
        "forbidden_fields": ["sparse_TE", "sparse_AE", "dense_TE", "dense_AE", "gt_pose_w2c"],
    }
    metrics = {
        "query_count": int(len(observations)),
        "mean_sparse_inliers": (
            float(sum(obs["sparse_inliers"] for obs in observations) / len(observations)) if observations else 0.0
        ),
        "uses_pose_error_or_gt": False,
    }
    split_audit = artifact_split_audit(metadata, branch_selection=False)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    print(json.dumps({"artifact": str(observations_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
