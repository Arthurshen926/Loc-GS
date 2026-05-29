#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import torch

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.stdloc_native.support_banks import build_named_support_banks


def _parse_ids(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _parse_int_ids(text: str) -> list[int]:
    return [int(part) for part in _parse_ids(text)]


def _load_landmark_support(path: str | Path) -> dict[int, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("landmark support must be a JSON object")
    return {
        int(landmark_id): {str(query_id): float(value) for query_id, value in dict(per_query).items()}
        for landmark_id, per_query in payload.items()
    }


def _metric_score(metrics: dict[str, Any]) -> float:
    positive = (
        float(metrics.get("support", 0.0) or 0.0)
        + float(metrics.get("viable_tuple_mass", 0.0) or 0.0)
        + float(metrics.get("logdet_H", 0.0) or 0.0)
        + float(metrics.get("min_eigenvalue", metrics.get("min_eigen", 0.0)) or 0.0)
    )
    negative = float(metrics.get("dense_worsen_risk", 0.0) or 0.0) + float(metrics.get("ambiguity", 0.0) or 0.0)
    return max(0.0, positive - negative)


def _load_landmark_support_from_failure_profile(path: str | Path) -> dict[int, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("failure profile must be a JSON object")
    candidate_gain = payload.get("candidate_query_gain", {})
    if not isinstance(candidate_gain, dict):
        raise TypeError("failure profile candidate_query_gain must be a JSON object")
    support: dict[int, dict[str, float]] = {}
    for landmark_id, per_query in candidate_gain.items():
        query_scores: dict[str, float] = {}
        for query_id, metrics in dict(per_query).items():
            if not isinstance(metrics, dict):
                continue
            score = _metric_score(dict(metrics))
            if score > 0.0:
                query_scores[str(query_id)] = float(score)
        if query_scores:
            support[int(landmark_id)] = query_scores
    return support


def _support_query_ids(landmark_support: dict[int, dict[str, float]]) -> set[str]:
    query_ids: set[str] = set()
    for per_query in landmark_support.values():
        query_ids.update(str(query_id) for query_id in per_query)
    return query_ids


def _resolve_landmark_support(args: argparse.Namespace) -> tuple[dict[int, dict[str, float]], str]:
    if args.landmark_support:
        return _load_landmark_support(args.landmark_support), str(args.landmark_support)
    if args.failure_profile:
        return _load_landmark_support_from_failure_profile(args.failure_profile), str(args.failure_profile)
    raise ValueError("one of --landmark_support or --failure_profile is required")


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v8 query-conditioned support-bank artifact.")
    parser.add_argument("--query_observations", required=True)
    parser.add_argument("--landmark_support", default="")
    parser.add_argument("--failure_profile", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--native_safe_core_ids", default="")
    parser.add_argument("--hard_query_ids", default="")
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--router_topk", type=int, choices=(1, 2), default=1)
    parser.add_argument("--allow_empty_support_overlap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip()
    if split_name.lower() == "test":
        raise ValueError("test split query observations are not allowed for support-bank construction")
    observations = json.loads(Path(args.query_observations).read_text(encoding="utf-8"))
    if not isinstance(observations, list):
        raise TypeError("query observations must be a JSON list")
    landmark_support, support_source = _resolve_landmark_support(args)
    observation_query_ids = {str(row.get("query_id", row.get("image_id", ""))) for row in observations if isinstance(row, dict)}
    support_query_ids = _support_query_ids(landmark_support)
    support_overlap_count = len(observation_query_ids & support_query_ids)
    if (
        args.failure_profile
        and support_query_ids
        and support_overlap_count == 0
        and not bool(args.allow_empty_support_overlap)
    ):
        raise ValueError(
            "support query ids do not overlap query observations; rebuild failure profile and observations from the same split"
        )
    result = build_named_support_banks(
        query_observations=observations,
        landmark_support=landmark_support,
        native_safe_core_ids=_parse_int_ids(args.native_safe_core_ids),
        hard_query_ids=_parse_ids(args.hard_query_ids),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "support_banks.pt"
    torch.save(result, artifact_path)
    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": split_name or "unknown",
        "recipe": f"lsf_v8_support_banks_top{int(args.router_topk)}",
        "artifact": str(artifact_path),
        "bank_count": int(result["metadata"]["bank_count"]),
        "query_count": int(len(result["query_ids"])),
        "support_query_count": int(len(support_query_ids)),
        "support_query_overlap_count": int(support_overlap_count),
        "routing_uses_gt_or_eval_result": False,
        "single_path_deployment": True,
        "branch_selection": False,
    }
    manifest = {
        "method": "loc_gs_lsf_v8_query_conditioned_support_banks",
        **metadata,
        "query_observations": str(args.query_observations),
        "landmark_support": support_source,
        "failure_profile": str(args.failure_profile) if args.failure_profile else "",
        "native_safe_core_count": int(len(_parse_int_ids(args.native_safe_core_ids))),
        "hard_query_count": int(len(_parse_ids(args.hard_query_ids))),
        "feature_names": result["feature_names"],
        "bank_names": [str(bank["name"]) for bank in result["banks"]],
    }
    metrics = {
        "bank_count": int(result["metadata"]["bank_count"]),
        "query_count": int(len(result["query_ids"])),
        "router_topk": int(args.router_topk),
        "support_query_count": int(len(support_query_ids)),
        "support_query_overlap_count": int(support_overlap_count),
        "unmatched_hard_query_count": int(result["metadata"].get("unmatched_hard_query_count", 0)),
    }
    split_audit = artifact_split_audit(metadata, branch_selection=False)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    print(json.dumps({"artifact": str(artifact_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
