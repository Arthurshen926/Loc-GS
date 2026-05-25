#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.feedback.io import load_feedback_bank


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _command_from_args(args: argparse.Namespace) -> str:
    parts = ["python", "-m", "loc_gs.scripts.build_query_conditioned_lsf_artifacts"]
    for key, value in sorted(vars(args).items()):
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                parts.append(option)
            continue
        if value is None:
            continue
        parts.extend([option, str(value)])
    return " ".join(shlex.quote(part) for part in parts)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_source_idx(source_map: Path) -> torch.Tensor:
    path = source_map / "detector" / "sampled_idx.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing source sampled_idx.pkl: {path}")
    return torch.as_tensor(_load_pickle(path), dtype=torch.long).reshape(-1).cpu()


def _load_source_score(source_map: Path, source_idx: torch.Tensor, num_gaussians: int) -> torch.Tensor:
    scores = torch.zeros((int(num_gaussians),), dtype=torch.float32)
    path = source_map / "detector" / "sampled_scores.pkl"
    if not path.exists():
        return scores
    payload = _load_pickle(path)
    if isinstance(payload, dict) and "score_avg" in payload:
        score_avg = torch.as_tensor(payload["score_avg"], dtype=torch.float32).reshape(-1).cpu()
        if score_avg.numel() == int(num_gaussians):
            return score_avg
    raw = (
        torch.as_tensor(payload.get("sampled_scores"), dtype=torch.float32).reshape(-1).cpu()
        if isinstance(payload, dict) and "sampled_scores" in payload
        else torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    )
    if raw.numel() == source_idx.numel():
        valid = (source_idx >= 0) & (source_idx < int(num_gaussians))
        scores[source_idx[valid]] = raw[valid]
    return scores


def _normalize(values: torch.Tensor) -> torch.Tensor:
    tensor = values.float().clone()
    finite = torch.isfinite(tensor)
    if not bool(finite.any().item()):
        return torch.zeros_like(tensor)
    lo = tensor[finite].min()
    hi = tensor[finite].max()
    if float((hi - lo).abs().item()) < 1e-8:
        out = torch.zeros_like(tensor)
        out[finite] = tensor[finite].clamp(0.0, 1.0)
        return out.clamp(0.0, 1.0)
    out = torch.zeros_like(tensor)
    out[finite] = (tensor[finite] - lo) / (hi - lo)
    return out.clamp(0.0, 1.0)


_QUERY_SUFFIX = re.compile(r"(\d+)$")


def _query_int(query_id: str, fallback: int) -> int:
    match = _QUERY_SUFFIX.search(str(query_id))
    if match:
        return int(match.group(1))
    return int(fallback)


def _query_group_key(record: dict[str, Any]) -> str:
    image_id = str(record.get("image_id", "")).strip()
    if image_id:
        return image_id
    query_id = str(record.get("query_id", "")).strip()
    return query_id.split("::", 1)[0] if "::" in query_id else ""


def _gid(record: dict[str, Any]) -> int | None:
    for key in ("matched_gaussian_id", "matched_landmark_id", "gaussian_id", "landmark_id"):
        raw = record.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key)
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _is_positive(record: dict[str, Any], threshold_px: float) -> bool:
    if not bool(record.get("pnp_inlier", False)):
        return False
    err = record.get("reprojection_error_px")
    if err is None:
        return True
    try:
        value = float(err)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value <= float(threshold_px)


def _is_hard_negative(
    record: dict[str, Any],
    *,
    positive: bool,
    score_threshold: float,
    reprojection_threshold_px: float,
) -> bool:
    if positive:
        return False
    score = _float(record, "descriptor_score", default=-float("inf"))
    err = record.get("reprojection_error_px")
    if err is None:
        return False
    try:
        reproj = float(err)
    except (TypeError, ValueError):
        return False
    return (
        score >= float(score_threshold)
        and math.isfinite(reproj)
        and reproj >= float(reprojection_threshold_px)
    )


def _keypoint_yx(record: dict[str, Any]) -> list[float]:
    xy = record.get("keypoint_xy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        x = _float({"v": xy[0]}, "v", 0.0)
        y = _float({"v": xy[1]}, "v", 0.0)
        return [float(y), float(x)]
    return [0.0, 0.0]


def _select_hard_queries(
    query_stats: dict[int, dict[str, float]],
    *,
    topk: int,
    mode: str,
) -> list[int]:
    if int(topk) <= 0 or int(topk) >= len(query_stats):
        return sorted(query_stats)
    scored: list[tuple[float, int]] = []
    has_positive_queries = any(float(stats.get("positive_count", 0.0)) > 0.0 for stats in query_stats.values())
    for qid, stats in query_stats.items():
        if mode == "pose_error":
            score = float(stats.get("pose_error_t_cm", 0.0)) + 2.0 * float(stats.get("pose_error_r_deg", 0.0))
        else:
            max_pos = float(stats.get("max_positive_score", 0.0))
            max_neg = float(stats.get("max_hard_negative_score", 0.0))
            score = (
                max_neg
                - max_pos
                + 0.05 * float(stats.get("hard_negative_count", 0.0))
                - 0.02 * float(stats.get("positive_count", 0.0))
            )
            if has_positive_queries and float(stats.get("positive_count", 0.0)) <= 0.0:
                score -= 10.0
        scored.append((score, int(qid)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(qid for _, qid in scored[: int(topk)])


def _build_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    bank = load_feedback_bank(args.feedback_bank)
    manifest = dict(bank.get("manifest", {}))
    split_name = str(manifest.get("split_name", manifest.get("split", "")))
    if split_name.strip().lower() == "test":
        raise ValueError("query-conditioned LSF artifacts cannot use test split")
    if str(manifest.get("schema_version", "")).strip() == "feedback_bank_v2":
        raise ValueError(
            "build_query_conditioned_lsf_artifacts does not preserve feedback_bank_v2 string query ids; "
            "use solver_consensus_support or build_solver_tuple_bank_v2 instead"
        )

    records = [dict(record) for record in bank.get("records", [])]
    num_gaussians = int(args.num_gaussians)
    if num_gaussians <= 0:
        max_gid = max((_gid(record) or 0 for record in records), default=0)
        num_gaussians = int(max_gid) + 1
    source_map = Path(args.source_map)
    source_idx = _load_source_idx(source_map)
    if source_idx.numel() and int(source_idx.max().item()) >= num_gaussians:
        raise ValueError("num_gaussians is smaller than source sampled_idx max")
    source_set = {int(item) for item in source_idx.tolist()}
    source_score = _normalize(_load_source_score(source_map, source_idx, num_gaussians))

    positive = torch.zeros((num_gaussians,), dtype=torch.float32)
    hard_negative = torch.zeros((num_gaussians,), dtype=torch.float32)
    score_sum = torch.zeros((num_gaussians,), dtype=torch.float32)
    score_count = torch.zeros((num_gaussians,), dtype=torch.float32)
    query_stats: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    parsed_records: list[dict[str, Any]] = []
    query_fallback: dict[str, int] = {}

    for record in records:
        gid = _gid(record)
        if gid is None or gid < 0 or gid >= num_gaussians:
            continue
        raw_qid = str(record.get("query_id", ""))
        group_key = _query_group_key(record)
        if group_key:
            if group_key not in query_fallback:
                query_fallback[group_key] = len(query_fallback)
            qid = query_fallback[group_key]
        else:
            if raw_qid not in query_fallback:
                query_fallback[raw_qid] = len(query_fallback)
            qid = _query_int(raw_qid, query_fallback[raw_qid])
        score = _float(record, "descriptor_score", 0.0)
        is_pos = _is_positive(record, float(args.positive_reprojection_threshold_px))
        is_hn = _is_hard_negative(
            record,
            positive=is_pos,
            score_threshold=float(args.hard_negative_score_threshold),
            reprojection_threshold_px=float(args.hard_negative_reprojection_threshold_px),
        )
        positive[gid] = max(float(positive[gid].item()), 1.0 if is_pos else 0.0)
        hard_negative[gid] = max(float(hard_negative[gid].item()), 1.0 if is_hn else 0.0)
        score_sum[gid] += float(score)
        score_count[gid] += 1.0
        stats = query_stats[qid]
        stats["record_count"] += 1.0
        stats["positive_count"] += 1.0 if is_pos else 0.0
        stats["hard_negative_count"] += 1.0 if is_hn else 0.0
        stats["max_positive_score"] = max(float(stats.get("max_positive_score", 0.0)), float(score) if is_pos else 0.0)
        stats["max_hard_negative_score"] = max(
            float(stats.get("max_hard_negative_score", 0.0)),
            float(score) if is_hn else 0.0,
        )
        stats["pose_error_t_cm"] = max(float(stats.get("pose_error_t_cm", 0.0)), _float(record, "pose_error_t_cm", 0.0))
        stats["pose_error_r_deg"] = max(float(stats.get("pose_error_r_deg", 0.0)), _float(record, "pose_error_r_deg", 0.0))
        parsed_records.append(
            {
                "query_id": qid,
                "gid": gid,
                "score": score,
                "reprojection_error": _float(record, "reprojection_error_px", float("inf")),
                "pnp_inlier": bool(record.get("pnp_inlier", False)),
                "positive": is_pos,
                "keypoint_yx": _keypoint_yx(record),
            }
        )

    observed_avg = torch.zeros_like(score_sum)
    observed = score_count > 0
    observed_avg[observed] = score_sum[observed] / score_count[observed].clamp_min(1.0)
    selector = torch.maximum(source_score, _normalize(observed_avg))

    hard_query_ids = _select_hard_queries(
        query_stats,
        topk=int(args.hard_query_topk),
        mode=str(args.hard_query_mode),
    )
    hard_query_set = set(hard_query_ids)
    filtered = [item for item in parsed_records if int(item["query_id"]) in hard_query_set]
    if filtered:
        query_id = torch.tensor([int(item["query_id"]) for item in filtered], dtype=torch.long)
        landmark_ids = torch.tensor([[int(item["gid"])] for item in filtered], dtype=torch.long)
        cosine = torch.tensor([[float(item["score"])] for item in filtered], dtype=torch.float32)
        reproj = torch.tensor([[float(item["reprojection_error"])] for item in filtered], dtype=torch.float32)
        pnp = torch.tensor([[bool(item["pnp_inlier"])] for item in filtered], dtype=torch.bool)
        visible = torch.ones_like(pnp, dtype=torch.bool)
        keypoint_yx = torch.tensor([item["keypoint_yx"] for item in filtered], dtype=torch.float32)
        pair_label = torch.tensor([[bool(item["positive"])] for item in filtered], dtype=torch.bool)
    else:
        query_id = torch.empty((0,), dtype=torch.long)
        landmark_ids = torch.empty((0, 1), dtype=torch.long)
        cosine = torch.empty((0, 1), dtype=torch.float32)
        reproj = torch.empty((0, 1), dtype=torch.float32)
        pnp = torch.empty((0, 1), dtype=torch.bool)
        visible = torch.empty((0, 1), dtype=torch.bool)
        keypoint_yx = torch.empty((0, 2), dtype=torch.float32)
        pair_label = torch.empty((0, 1), dtype=torch.bool)
    episode_cache = {
        "query_id": query_id,
        "keypoint_yx": keypoint_yx,
        "candidate_landmark_ids": landmark_ids,
        "landmark_id": landmark_ids,
        "candidate_cosine": cosine,
        "candidate_reprojection_error": reproj,
        "candidate_pnp_inlier": pnp,
        "candidate_visible": visible,
        "pair_label": pair_label,
        "metadata": {
            "format": "query_conditioned_lsf_episode_v1",
            "scene": str(manifest.get("scene", "")),
            "split_name": split_name,
            "feedback_bank": str(args.feedback_bank),
            "hard_query_mode": str(args.hard_query_mode),
            "hard_query_topk": int(args.hard_query_topk),
            "positive_reprojection_threshold_px": float(args.positive_reprojection_threshold_px),
            "hard_negative_score_threshold": float(args.hard_negative_score_threshold),
            "hard_negative_reprojection_threshold_px": float(args.hard_negative_reprojection_threshold_px),
        },
    }

    safe_core = sorted(int(item) for item in source_set if bool(positive[item].item()))
    accepted = (positive > 0.0) & (hard_negative <= 0.0)
    candidate_ids = [int(idx) for idx in torch.where(accepted)[0].tolist() if int(idx) not in source_set]
    candidate_ids.sort(key=lambda idx: (-float(selector[idx].item()), idx))
    candidate_ids = candidate_ids[: int(args.candidate_pool_size)]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / f"episode_cache_top{int(args.hard_query_topk)}.pt"
    candidate_pool_path = output / f"candidate_pool_positive_lowrisk_top{int(args.candidate_pool_size)}.pt"
    torch.save(episode_cache, episode_path)
    torch.save(selector.float(), output / "selector_score_avg.pt")
    torch.save(positive.float(), output / "positive_support_binary.pt")
    torch.save(hard_negative.float(), output / "hard_negative_risk_binary.pt")
    torch.save(torch.tensor(safe_core, dtype=torch.long), output / "safe_core_source_positive.pt")
    torch.save(torch.tensor(candidate_ids, dtype=torch.long), candidate_pool_path)
    (output / f"hard_query_ids_top{int(args.hard_query_topk)}.txt").write_text(
        "\n".join(str(item) for item in hard_query_ids) + ("\n" if hard_query_ids else ""),
        encoding="utf-8",
    )

    summary = {
        "scene": str(manifest.get("scene", "")),
        "split_name": split_name or "unknown",
        "test_split_used": split_name.strip().lower() == "test",
        "num_gaussians": int(num_gaussians),
        "source_count": int(source_idx.numel()),
        "record_count": int(len(records)),
        "query_count": int(len(query_stats)),
        "hard_query_count": int(len(hard_query_ids)),
        "filtered_record_count": int(len(filtered)),
        "positive_landmark_count": int((positive > 0.0).sum().item()),
        "hard_negative_landmark_count": int((hard_negative > 0.0).sum().item()),
        "safe_core_count": int(len(safe_core)),
        "candidate_pool_count": int(len(candidate_ids)),
        "episode_cache": str(episode_path),
        "candidate_pool": str(candidate_pool_path),
    }
    run_manifest = {
        "method": "query_conditioned_lsf_artifact_builder",
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv) if sys.argv else _command_from_args(args),
        "feedback_bank": str(args.feedback_bank),
        "source_map": str(args.source_map),
        "output_dir": str(output),
        "split_name": split_name or "unknown",
        "test_split_used": summary["test_split_used"],
        "single_path_deployment": True,
        "branch_selection": False,
        "hyperparameters": {
            "hard_query_topk": int(args.hard_query_topk),
            "hard_query_mode": str(args.hard_query_mode),
            "candidate_pool_size": int(args.candidate_pool_size),
            "positive_reprojection_threshold_px": float(args.positive_reprojection_threshold_px),
            "hard_negative_score_threshold": float(args.hard_negative_score_threshold),
            "hard_negative_reprojection_threshold_px": float(args.hard_negative_reprojection_threshold_px),
        },
        "feedback_bank_manifest": manifest,
        "summary": summary,
    }
    (output / "episode_cache_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build query-conditioned LSF artifacts from a feedback bank.")
    parser.add_argument("--feedback_bank", required=True)
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gaussians", type=int, default=0)
    parser.add_argument("--hard_query_topk", type=int, default=256)
    parser.add_argument("--hard_query_mode", choices=("hard_negative_pressure", "pose_error"), default="hard_negative_pressure")
    parser.add_argument("--candidate_pool_size", type=int, default=4096)
    parser.add_argument("--positive_reprojection_threshold_px", type=float, default=4.0)
    parser.add_argument("--hard_negative_score_threshold", type=float, default=0.65)
    parser.add_argument("--hard_negative_reprojection_threshold_px", type=float, default=8.0)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = build_argparser().parse_args() if args is None else args
    summary = _build_artifacts(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
