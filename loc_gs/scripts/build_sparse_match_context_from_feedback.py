#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from loc_gs.scripts.export_ulfloc_sparse_feedback import _git_commit, _git_status


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positive_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out <= 0.0:
        return 0.0
    return float(out)


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_profile_context_queries(profile: Mapping[str, Any]) -> set[str]:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use sparse validation profile from test split")
    query_ids: set[str] = set()
    for field_name in ("protected_per_query_support", "protected_per_query_observations"):
        raw = profile.get(field_name, {})
        if not isinstance(raw, Mapping):
            continue
        for raw_query_id in raw.keys():
            query_id = str(raw_query_id).strip()
            if query_id:
                query_ids.add(query_id)
    return query_ids


def build_sparse_match_context_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    query_ids: set[str],
    split_name: str,
    max_records_per_query: int = 256,
    inlier_weight: float = 1.0,
    non_inlier_weight: float = 0.15,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(split_name).strip().lower() == "test":
        raise ValueError("refusing to build sparse match context from test split")
    max_records = max(0, int(max_records_per_query))
    per_query_scores: dict[str, dict[int, float]] = {query_id: {} for query_id in sorted(query_ids)}
    seen_record_count = 0
    used_record_count = 0
    inlier_record_count = 0
    non_inlier_record_count = 0

    for record in records:
        seen_record_count += 1
        query_id = str(record.get("query_id", record.get("image_id", ""))).strip()
        if query_id not in query_ids:
            continue
        if record.get("pnp_success") is False:
            continue
        try:
            gid = int(record.get("matched_gaussian_id", record.get("gaussian_id")))
        except (TypeError, ValueError):
            continue
        if gid < 0:
            continue
        descriptor_score = _positive_float(record.get("descriptor_score"), default=0.0)
        detector_score = _positive_float(record.get("detector_score"), default=1.0)
        if descriptor_score <= 0.0:
            continue
        pnp_inlier = _bool_value(record.get("pnp_inlier", False))
        weight = max(0.0, float(inlier_weight if pnp_inlier else non_inlier_weight))
        if weight <= 0.0:
            continue
        score = descriptor_score * max(0.05, detector_score) * weight
        if score <= 0.0:
            continue
        current = float(per_query_scores.setdefault(query_id, {}).get(gid, 0.0))
        per_query_scores[query_id][gid] = current + float(score)
        used_record_count += 1
        if pnp_inlier:
            inlier_record_count += 1
        else:
            non_inlier_record_count += 1

    query_payload: dict[str, dict[str, Any]] = {}
    context_entry_count = 0
    for query_id in sorted(per_query_scores):
        items = sorted(
            per_query_scores[query_id].items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        if max_records > 0:
            items = items[:max_records]
        if not items:
            continue
        context_entry_count += len(items)
        query_payload[query_id] = {
            "support_match_strength": {str(gid): float(value) for gid, value in items},
            "match_competition_risk": {},
            "metadata": {
                "context_entry_count": int(len(items)),
                "context_strength_total": float(sum(value for _gid, value in items)),
            },
        }

    payload = {
        "schema_version": "locgs_sparse_match_context_from_feedback_v1",
        "split_name": str(split_name),
        "official_test_used": False,
        "queries": query_payload,
    }
    metrics = {
        "query_count": int(len(query_payload)),
        "requested_query_count": int(len(query_ids)),
        "seen_record_count": int(seen_record_count),
        "used_record_count": int(used_record_count),
        "inlier_record_count": int(inlier_record_count),
        "non_inlier_record_count": int(non_inlier_record_count),
        "context_entry_count": int(context_entry_count),
        "max_records_per_query": int(max_records),
        "inlier_weight": float(inlier_weight),
        "non_inlier_weight": float(non_inlier_weight),
    }
    return payload, metrics


def _score_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for raw_gid, raw_value in raw.items():
        try:
            gid = int(raw_gid)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if gid < 0 or value <= 0.0:
            continue
        out[str(gid)] = max(float(out.get(str(gid), 0.0)), float(value))
    return out


def merge_query_match_dominance(base: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    base_split = str(base.get("split_name", "")).strip()
    context_split = str(context.get("split_name", "")).strip()
    if base_split.lower() == "test" or context_split.lower() == "test":
        raise ValueError("refusing to merge query match dominance from test split")
    if bool(base.get("official_test_used", False)) or bool(context.get("official_test_used", False)):
        raise ValueError("refusing to merge query match dominance marked as official-test-used")
    base_queries = base.get("queries", {})
    context_queries = context.get("queries", {})
    if not isinstance(base_queries, Mapping):
        base_queries = {}
    if not isinstance(context_queries, Mapping):
        context_queries = {}
    merged_queries: dict[str, dict[str, Any]] = {}
    for query_id in sorted(set(map(str, base_queries.keys())) | set(map(str, context_queries.keys()))):
        base_query = base_queries.get(query_id, {})
        context_query = context_queries.get(query_id, {})
        if not isinstance(base_query, Mapping):
            base_query = {}
        if not isinstance(context_query, Mapping):
            context_query = {}
        strength = _score_map(base_query.get("support_match_strength", {}))
        for gid, value in _score_map(context_query.get("support_match_strength", {})).items():
            strength[gid] = max(float(strength.get(gid, 0.0)), float(value))
        risk = _score_map(base_query.get("match_competition_risk", {}))
        for gid, value in _score_map(context_query.get("match_competition_risk", {})).items():
            risk[gid] = max(float(risk.get(gid, 0.0)), float(value))
        merged_queries[query_id] = {
            "support_match_strength": dict(sorted(strength.items(), key=lambda item: (-float(item[1]), int(item[0])))),
            "match_competition_risk": dict(sorted(risk.items(), key=lambda item: (-float(item[1]), int(item[0])))),
            "metadata": {
                "base_support_count": int(len(_score_map(base_query.get("support_match_strength", {})))),
                "context_support_count": int(len(_score_map(context_query.get("support_match_strength", {})))),
                "risk_count": int(len(risk)),
            },
        }
    return {
        "schema_version": "locgs_query_match_dominance_with_context_v1",
        "split_name": context_split or base_split or "unknown",
        "official_test_used": False,
        "queries": merged_queries,
        "metadata": {
            "base_query_count": int(len(base_queries)),
            "context_query_count": int(len(context_queries)),
            "merged_query_count": int(len(merged_queries)),
        },
    }


def iter_feedback_bank_records(path: Path) -> Iterable[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            if row.get("type") != "record":
                continue
            record = row.get("record")
            if isinstance(record, Mapping):
                yield record


def feedback_bank_split_name(path: Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("type") != "manifest":
                continue
            manifest = row.get("manifest", {})
            if isinstance(manifest, Mapping):
                split_name = str(manifest.get("split_name", "")).strip()
                if split_name:
                    return split_name
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback_bank", required=True, type=Path)
    parser.add_argument("--sparse_validation_profile", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--base_query_match_dominance", default=None, type=Path)
    parser.add_argument("--split_name", default=None)
    parser.add_argument("--max_records_per_query", type=int, default=256)
    parser.add_argument("--inlier_weight", type=float, default=1.0)
    parser.add_argument("--non_inlier_weight", type=float, default=0.15)
    args = parser.parse_args(argv)

    feedback_split = feedback_bank_split_name(args.feedback_bank)
    if feedback_split.strip().lower() == "test":
        raise ValueError("refusing to use sparse feedback bank from test split")
    profile = json.loads(Path(args.sparse_validation_profile).read_text(encoding="utf-8"))
    if not isinstance(profile, Mapping):
        raise TypeError("sparse validation profile must be a JSON object")
    query_ids = load_profile_context_queries(profile)
    split_name = str(args.split_name or profile.get("split_name") or feedback_split or "unknown")
    payload, metrics = build_sparse_match_context_from_records(
        iter_feedback_bank_records(args.feedback_bank),
        query_ids=query_ids,
        split_name=split_name,
        max_records_per_query=int(args.max_records_per_query),
        inlier_weight=float(args.inlier_weight),
        non_inlier_weight=float(args.non_inlier_weight),
    )
    if args.base_query_match_dominance is not None:
        base_payload = json.loads(Path(args.base_query_match_dominance).read_text(encoding="utf-8"))
        if not isinstance(base_payload, Mapping):
            raise TypeError("base query match dominance must be a JSON object")
        payload = merge_query_match_dominance(base_payload, payload)
        metrics["base_query_match_dominance"] = str(Path(args.base_query_match_dominance))
        metrics["merged_query_count"] = int(payload.get("metadata", {}).get("merged_query_count", 0))
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "query_match_dominance.json", payload)
    _write_json(output_dir / "metrics_summary.json", metrics)
    split_audit = {
        "audit_status": "passed",
        "split_name": str(split_name),
        "official_test_used": False,
        "checks": {
            "feedback_bank_split": {"status": "passed", "split_name": str(feedback_split)},
            "sparse_validation_profile_split": {
                "status": "passed",
                "split_name": str(profile.get("split_name", "unknown")),
            },
        },
    }
    _write_json(output_dir / "split_audit.json", split_audit)
    manifest = {
        "method": "sparse_match_context_from_feedback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "command": " ".join([sys.executable, "-m", "loc_gs.scripts.build_sparse_match_context_from_feedback", *sys.argv[1:]]),
        "scene": str(profile.get("scene", "unknown")),
        "split": str(split_name),
        "checkpoint_path": None,
        "map_path": None,
        "data_roots": [],
        "hyperparameters": {
            "max_records_per_query": int(args.max_records_per_query),
            "inlier_weight": float(args.inlier_weight),
            "non_inlier_weight": float(args.non_inlier_weight),
            "base_query_match_dominance": None
            if args.base_query_match_dominance is None
            else str(Path(args.base_query_match_dominance)),
        },
        "feedback_bank": str(Path(args.feedback_bank)),
        "sparse_validation_profile": str(Path(args.sparse_validation_profile)),
        "metrics_summary": metrics,
        "split_audit": split_audit,
        "official_test_used": False,
        "diagnostic_only": True,
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
