#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.stdloc_native.solver_feedback_feature_fusion_v2 import (
    fuse_landmark_descriptors,
    fuse_landmark_descriptors_contrastive,
)


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must contain a dict: {path}")
    return dict(payload)


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"payload must contain a dict: {path}")
    return dict(payload)


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        return float(default)
    return max(0.0, float(number))


def _split_name(payload: Mapping[str, Any]) -> str:
    for source in (payload, payload.get("metadata", {}), payload.get("split_audit", {})):
        if not isinstance(source, Mapping):
            continue
        for key in ("split_name", "split", "source_split_name", "feedback_bank_split_name"):
            value = str(source.get(key, "")).strip()
            if value:
                return value
    return "unknown"


def _is_test_split(split_name: str) -> bool:
    split = str(split_name).strip().lower()
    return split == "test" or split.endswith("_test") or split == "official_test"


def _uses_official_test(payload: Mapping[str, Any]) -> bool:
    for source in (payload, payload.get("metadata", {}), payload.get("split_audit", {})):
        if not isinstance(source, Mapping):
            continue
        if bool(source.get("test_split_used", False)) or bool(source.get("official_test_used", False)):
            return True
    return False


def _reject_test_payload(payload: Mapping[str, Any], role: str) -> None:
    if _is_test_split(_split_name(payload)) or _uses_official_test(payload):
        raise ValueError(f"refusing to use test split {role}")


def _sampled_ids_from_cache(payload: Mapping[str, Any]) -> torch.Tensor | None:
    for key in ("sampled_idx", "keypoints_sampled_idx", "gaussian_ids", "base_gaussian_id", "base_gaussian_ids"):
        if key in payload:
            return torch.as_tensor(payload[key], dtype=torch.long).reshape(-1).detach().cpu()
    metadata = payload.get("metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("sampled_idx", "keypoints_sampled_idx", "gaussian_ids", "base_gaussian_id", "base_gaussian_ids"):
            if key in metadata:
                return torch.as_tensor(metadata[key], dtype=torch.long).reshape(-1).detach().cpu()
    return None


def _validate_cache_sampled_ids(cache: Mapping[str, Any], sampled_idx: torch.Tensor) -> None:
    cache_ids = _sampled_ids_from_cache(cache)
    if cache_ids is None:
        raise ValueError("observation cache must contain sampled ids for audit")
    if cache_ids.shape != sampled_idx.shape or not torch.equal(cache_ids, sampled_idx):
        raise ValueError("observation cache sampled ids must exactly match keypoints_sampled_idx.pkl")


def _feature_table(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    if tensor.dim() != 2:
        raise ValueError(f"{name} must have shape [landmarks, descriptor_dim]")
    return tensor


def _entry_weight(entry: Any, preferred_key: str) -> float:
    if isinstance(entry, Mapping):
        for key in (preferred_key, "weight", "support", "risk", "value"):
            if key in entry:
                return _finite_nonnegative(entry.get(key))
        return 0.0
    return _finite_nonnegative(entry)


def _lookup_landmark_weight(table: Any, gid: int, preferred_key: str) -> float:
    if not isinstance(table, Mapping):
        return 0.0
    for key in (gid, str(gid)):
        if key in table:
            return _entry_weight(table[key], preferred_key)
    return 0.0


def _lookup_view_weight(table: Any, gid: int, view_id: str, preferred_key: str) -> float:
    if not isinstance(table, Mapping):
        return 0.0
    view = str(view_id)
    candidate_keys: list[Any] = [
        (str(gid), view),
        (gid, view),
        f"{gid}:{view}",
        f"{gid}|{view}",
        f"{gid}/{view}",
        str((str(gid), view)),
        str((gid, view)),
    ]
    for key in candidate_keys:
        if key in table:
            return _entry_weight(table[key], preferred_key)
    for gid_key in (gid, str(gid)):
        nested = table.get(gid_key)
        if isinstance(nested, Mapping):
            for view_key in (view_id, view):
                if view_key in nested:
                    return _entry_weight(nested[view_key], preferred_key)
    return 0.0


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and len(value) > 0


def _impact_weights(impact: Mapping[str, Any], *, gid: int, view_id: str) -> tuple[float, float]:
    view_positive = impact.get("view_positive", {})
    view_negative = impact.get("view_negative", {})
    positive = _lookup_view_weight(view_positive, gid, view_id, "weight")
    negative = _lookup_view_weight(view_negative, gid, view_id, "weight")
    # View-level solver traces are more specific than landmark-level priors.
    # Only fall back to unary landmark impact when the impact artifact does not
    # provide any view-level supervision of that sign.
    if positive <= 0.0 and not _nonempty_mapping(view_positive):
        positive = _lookup_landmark_weight(impact.get("landmark_positive", {}), gid, "support")
    if negative <= 0.0 and not _nonempty_mapping(view_negative):
        negative = _lookup_landmark_weight(impact.get("landmark_negative", {}), gid, "risk")
    return float(positive), float(negative)


def _impact_has_positive_supervision(impact: Mapping[str, Any]) -> bool:
    return _nonempty_mapping(impact.get("view_positive", {})) or _nonempty_mapping(impact.get("landmark_positive", {}))


def _impact_has_negative_supervision(impact: Mapping[str, Any]) -> bool:
    return _nonempty_mapping(impact.get("view_negative", {})) or _nonempty_mapping(impact.get("landmark_negative", {}))


def _raw_id_to_row(
    raw_id: Any,
    *,
    sampled_to_row: Mapping[int, int],
    landmark_count: int,
    id_space: str,
) -> int | None:
    text = str(raw_id).strip()
    if text.startswith("sampled:"):
        text = text.split(":", 1)[1]
        id_space = "row_index"
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    space = id_space.strip().lower()
    if space in {"row", "row_index", "sampled_row", "landmark_row", "landmark_index"}:
        return value if 0 <= value < landmark_count else None
    if space in {"gaussian", "gaussian_id", "sampled_gaussian_id"}:
        return sampled_to_row.get(value)
    if value in sampled_to_row:
        return sampled_to_row[value]
    if 0 <= value < landmark_count:
        return value
    return None


def _row_view_id(row: Mapping[str, Any]) -> str:
    for key in ("view_id", "source_view_id", "row_source_view_id", "image_id", "query_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _append_row_observation(
    out: dict[int, list[dict[str, Any]]],
    *,
    row_idx: int,
    gid: int,
    row: Mapping[str, Any],
    impact: Mapping[str, Any],
    observation_positive_weight_scale: float = 0.0,
) -> None:
    view_id = _row_view_id(row)
    impact_pos, impact_neg = _impact_weights(impact, gid=gid, view_id=view_id)
    direct_positive = _finite_nonnegative(row.get("positive_weight", row.get("weight", 0.0)))
    direct_negative = _finite_nonnegative(row.get("negative_weight", row.get("risk", 0.0)))
    # Observation caches carry geometry/visibility weights. Those are useful
    # fallback signals, but once split-safe solver impact is available they must
    # not turn every visible observation into a solver-positive view.
    weak_positive = direct_positive * max(0.0, float(observation_positive_weight_scale))
    positive = max(impact_pos, weak_positive) if _impact_has_positive_supervision(impact) else direct_positive
    negative = max(impact_neg, direct_negative) if _impact_has_negative_supervision(impact) else direct_negative
    out.setdefault(int(row_idx), []).append(
        {
            "descriptor": row["descriptor"],
            "positive_weight": float(positive),
            "negative_weight": float(negative),
            "source_view_id": str(view_id),
            "gaussian_id": int(gid),
        }
    )


def _observations_from_explicit_cache(
    cache: Mapping[str, Any],
    *,
    sampled_idx: torch.Tensor,
    impact: Mapping[str, Any],
    observation_positive_weight_scale: float = 0.0,
) -> dict[int, list[dict[str, Any]]]:
    raw = cache.get("observations")
    if not isinstance(raw, (Mapping, list, tuple)):
        raise ValueError("synthetic observation cache must contain observations")
    sampled_to_row = {int(gid): int(row) for row, gid in enumerate(sampled_idx.tolist())}
    id_space = str(cache.get("observation_id_space", cache.get("id_space", "")))
    out: dict[int, list[dict[str, Any]]] = {}
    if isinstance(raw, Mapping):
        for raw_key, rows in raw.items():
            row_idx = _raw_id_to_row(raw_key, sampled_to_row=sampled_to_row, landmark_count=int(sampled_idx.numel()), id_space=id_space)
            if row_idx is None:
                continue
            if not isinstance(rows, (list, tuple)):
                raise ValueError("observation cache mapping values must be lists")
            gid = int(sampled_idx[int(row_idx)].item())
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                _append_row_observation(
                    out,
                    row_idx=int(row_idx),
                    gid=gid,
                    row=row,
                    impact=impact,
                    observation_positive_weight_scale=float(observation_positive_weight_scale),
                )
        return out

    for row in raw:
        if not isinstance(row, Mapping):
            continue
        if "row_index" in row:
            row_idx = _raw_id_to_row(row["row_index"], sampled_to_row=sampled_to_row, landmark_count=int(sampled_idx.numel()), id_space="row_index")
        elif "landmark_index" in row:
            row_idx = _raw_id_to_row(row["landmark_index"], sampled_to_row=sampled_to_row, landmark_count=int(sampled_idx.numel()), id_space="row_index")
        elif "gaussian_id" in row:
            row_idx = _raw_id_to_row(row["gaussian_id"], sampled_to_row=sampled_to_row, landmark_count=int(sampled_idx.numel()), id_space="gaussian_id")
        elif "landmark_id" in row:
            row_idx = _raw_id_to_row(row["landmark_id"], sampled_to_row=sampled_to_row, landmark_count=int(sampled_idx.numel()), id_space=id_space or "row_index")
        else:
            raise ValueError("observation row must contain row_index, landmark_index, gaussian_id, or landmark_id")
        if row_idx is None:
            continue
        gid = int(sampled_idx[int(row_idx)].item())
        _append_row_observation(
            out,
            row_idx=int(row_idx),
            gid=gid,
            row=row,
            impact=impact,
            observation_positive_weight_scale=float(observation_positive_weight_scale),
        )
    return out


def _candidate_tensor(payload: Mapping[str, Any], names: tuple[str, ...], reference_shape: torch.Size) -> torch.Tensor | None:
    rows, topk = int(reference_shape[0]), int(reference_shape[1])
    for name in names:
        if name not in payload:
            continue
        tensor = torch.as_tensor(payload[name], dtype=torch.float32).detach().cpu()
        if tuple(tensor.shape) == (rows, topk):
            return tensor
        if tuple(tensor.shape) == (rows,):
            return tensor.reshape(rows, 1).expand(rows, topk)
        if tuple(tensor.shape) == (rows, 1):
            return tensor.expand(rows, topk)
        raise ValueError(f"{name} must match landmark_id shape")
    return None


def _observations_from_pair_cache(
    cache: Mapping[str, Any],
    *,
    sampled_idx: torch.Tensor,
    impact: Mapping[str, Any],
    observation_positive_weight_scale: float = 0.0,
) -> dict[int, list[dict[str, Any]]]:
    if "query_desc" not in cache or "landmark_id" not in cache:
        raise ValueError("observation cache must contain observations or pair-cache query_desc/landmark_id")
    query_desc = torch.as_tensor(cache["query_desc"], dtype=torch.float32).detach().cpu()
    landmark_ids = torch.as_tensor(cache["landmark_id"], dtype=torch.long).detach().cpu()
    if query_desc.dim() != 2 or landmark_ids.dim() != 2:
        raise ValueError("pair cache query_desc must be [rows,D] and landmark_id must be [rows,topk]")
    if int(query_desc.shape[0]) != int(landmark_ids.shape[0]):
        raise ValueError("pair cache query_desc rows must match landmark_id rows")
    rows, topk = int(landmark_ids.shape[0]), int(landmark_ids.shape[1])
    mask = torch.ones(rows, topk, dtype=torch.bool)
    if "candidate_mask" in cache:
        mask = torch.as_tensor(cache["candidate_mask"], dtype=torch.bool).detach().cpu()
        if mask.shape != landmark_ids.shape:
            raise ValueError("pair cache candidate_mask must match landmark_id")
    direct_positive = _candidate_tensor(
        cache,
        ("positive_weight", "positive_weights", "solver_positive_weight", "solver_positive_weights"),
        landmark_ids.shape,
    )
    direct_negative = _candidate_tensor(
        cache,
        ("negative_weight", "negative_weights", "solver_negative_weight", "solver_negative_weights"),
        landmark_ids.shape,
    )
    row_source_view_id = [str(item) for item in cache.get("row_source_view_id", [""] * rows)]
    if len(row_source_view_id) != rows:
        raise ValueError("pair cache row_source_view_id length must match query_desc rows")

    out: dict[int, list[dict[str, Any]]] = {}
    for row in range(rows):
        for col in range(topk):
            if not bool(mask[row, col]):
                continue
            row_idx = int(landmark_ids[row, col].item())
            if row_idx < 0 or row_idx >= int(sampled_idx.numel()):
                continue
            gid = int(sampled_idx[row_idx].item())
            impact_pos, impact_neg = _impact_weights(impact, gid=gid, view_id=row_source_view_id[row])
            direct_pos = _finite_nonnegative(direct_positive[row, col].item()) if direct_positive is not None else 0.0
            direct_neg = _finite_nonnegative(direct_negative[row, col].item()) if direct_negative is not None else 0.0
            positive = (
                max(float(impact_pos), direct_pos * max(0.0, float(observation_positive_weight_scale)))
                if _impact_has_positive_supervision(impact)
                else direct_pos
            )
            negative = (
                max(float(impact_neg), direct_neg)
                if _impact_has_negative_supervision(impact)
                else direct_neg
            )
            out.setdefault(row_idx, []).append(
                {
                    "descriptor": query_desc[row],
                    "positive_weight": float(positive),
                    "negative_weight": float(negative),
                    "source_view_id": str(row_source_view_id[row]),
                    "gaussian_id": int(gid),
                }
            )
    return out


def build_observations_from_cache(
    cache: Mapping[str, Any],
    *,
    sampled_idx: torch.Tensor,
    impact: Mapping[str, Any],
    observation_positive_weight_scale: float = 0.0,
) -> dict[int, list[dict[str, Any]]]:
    _validate_cache_sampled_ids(cache, sampled_idx)
    if "observations" in cache:
        return _observations_from_explicit_cache(
            cache,
            sampled_idx=sampled_idx,
            impact=impact,
            observation_positive_weight_scale=float(observation_positive_weight_scale),
        )
    return _observations_from_pair_cache(
        cache,
        sampled_idx=sampled_idx,
        impact=impact,
        observation_positive_weight_scale=float(observation_positive_weight_scale),
    )


def _active_plan_lookup(plan: Mapping[str, Any]) -> dict[int, set[str]]:
    raw = plan.get("landmark_fusion_plan", plan.get("descriptor_fusion", {}).get("landmark_fusion_plan", {}))
    if not isinstance(raw, Mapping):
        raise ValueError("active_fusion_plan must contain landmark_fusion_plan")
    lookup: dict[int, set[str]] = {}
    for raw_gid, raw_row in raw.items():
        try:
            gid = int(raw_gid)
        except (TypeError, ValueError):
            continue
        if gid < 0 or not isinstance(raw_row, Mapping):
            continue
        views = raw_row.get("selected_view_ids", raw_row.get("view_ids", []))
        if isinstance(views, str):
            selected = {views} if views.strip() else set()
        elif isinstance(views, (list, tuple, set)):
            selected = {str(view) for view in views if str(view).strip()}
        else:
            selected = set()
        if selected:
            lookup[int(gid)] = selected
    return lookup


def filter_observations_with_active_plan(
    observations: Mapping[int, list[Mapping[str, Any]]],
    *,
    sampled_idx: torch.Tensor,
    active_fusion_plan: Mapping[str, Any],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    lookup = _active_plan_lookup(active_fusion_plan)
    kept = 0
    filtered = 0
    out: dict[int, list[dict[str, Any]]] = {}
    for raw_idx, rows in observations.items():
        try:
            row_idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if row_idx < 0 or row_idx >= int(sampled_idx.numel()):
            continue
        gid = int(sampled_idx[row_idx].item())
        selected_views = lookup.get(gid, set())
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            view_id = str(row.get("source_view_id", row.get("view_id", "")))
            if selected_views and view_id in selected_views:
                out.setdefault(row_idx, []).append(dict(row))
                kept += 1
            else:
                filtered += 1
    metadata = {
        "active_fusion_plan_enabled": True,
        "active_fusion_plan_landmark_count": int(len(lookup)),
        "active_fusion_plan_kept_observation_count": int(kept),
        "active_fusion_plan_filtered_observation_count": int(filtered),
        "active_fusion_plan_split_name": str(active_fusion_plan.get("split_name", active_fusion_plan.get("split", ""))),
    }
    return out, metadata


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a ULF-Loc sparse log dir with solver-feedback fused descriptors.")
    parser.add_argument("--input_log_dir", "--source_log_dir", dest="input_log_dir", required=True, type=Path)
    parser.add_argument("--observation_cache", required=True, type=Path)
    parser.add_argument("--impact_attribution", required=True, type=Path)
    parser.add_argument("--active_fusion_plan", default=None, type=Path)
    parser.add_argument("--output_log_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--fusion_mode", choices=("mean", "contrastive"), default="mean")
    parser.add_argument("--trust_alpha", type=float, default=0.1)
    parser.add_argument("--min_native_cosine", type=float, default=0.95)
    parser.add_argument("--min_positive_weight", type=float, default=1e-6)
    parser.add_argument("--min_negative_weight", type=float, default=1e-6)
    parser.add_argument("--observation_positive_weight_scale", type=float, default=0.0)
    parser.add_argument("--min_positive_views", type=int, default=1)
    parser.add_argument("--max_positive_views", type=int, default=8)
    parser.add_argument("--max_negative_views", type=int, default=8)
    parser.add_argument("--contrastive_steps", type=int, default=20)
    parser.add_argument("--contrastive_lr", type=float, default=0.1)
    parser.add_argument("--anchor_weight", "--contrastive_anchor_weight", dest="anchor_weight", type=float, default=1.0)
    parser.add_argument(
        "--negative_weight_scale",
        "--contrastive_negative_weight",
        dest="negative_weight_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--negative_margin", "--contrastive_negative_margin", dest="negative_margin", type=float, default=0.2)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    source = Path(args.input_log_dir)
    output = Path(args.output_log_dir)
    if not source.exists():
        raise FileNotFoundError(f"input_log_dir not found: {source}")
    feature_path = source / "keypoints_features.pkl"
    sampled_path = source / "keypoints_sampled_idx.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"input log has no keypoints_features.pkl: {source}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"input log has no keypoints_sampled_idx.pkl: {source}")

    source_features_raw = _load_pickle(feature_path)
    source_features = _feature_table(source_features_raw, name="keypoints_features.pkl")
    sampled_idx = torch.as_tensor(_load_pickle(sampled_path), dtype=torch.long).reshape(-1).detach().cpu()
    if int(sampled_idx.numel()) != int(source_features.shape[0]):
        raise ValueError("keypoints_sampled_idx.pkl must match keypoints_features.pkl rows")

    impact = _load_payload(Path(args.impact_attribution))
    _reject_test_payload(impact, "impact")
    cache = _load_payload(Path(args.observation_cache))
    _reject_test_payload(cache, "observation cache")
    observation_descriptor_source = str(cache.get("observation_descriptor_source", "unknown"))
    observation_format = str(cache.get("observation_format", "unknown"))
    observation_positive_weight_scale = max(0.0, float(args.observation_positive_weight_scale))
    observations = build_observations_from_cache(
        cache,
        sampled_idx=sampled_idx,
        impact=impact,
        observation_positive_weight_scale=float(observation_positive_weight_scale),
    )
    active_plan_metadata: dict[str, Any] = {
        "active_fusion_plan_enabled": False,
        "active_fusion_plan_landmark_count": 0,
        "active_fusion_plan_kept_observation_count": 0,
        "active_fusion_plan_filtered_observation_count": 0,
    }
    active_plan_path = Path(args.active_fusion_plan) if args.active_fusion_plan is not None else None
    if active_plan_path is not None:
        active_plan = _load_json(active_plan_path)
        _reject_test_payload(active_plan, "active fusion plan")
        observations, active_plan_metadata = filter_observations_with_active_plan(
            observations,
            sampled_idx=sampled_idx,
            active_fusion_plan=active_plan,
        )

    if str(args.fusion_mode) == "contrastive":
        fused_norm, fusion_metadata = fuse_landmark_descriptors_contrastive(
            source_features.float(),
            observations,
            min_native_cosine=float(args.min_native_cosine),
            min_positive_weight=float(args.min_positive_weight),
            min_negative_weight=float(args.min_negative_weight),
            min_positive_views=int(args.min_positive_views),
            max_positive_views=int(args.max_positive_views),
            max_negative_views=int(args.max_negative_views),
            contrastive_steps=int(args.contrastive_steps),
            contrastive_lr=float(args.contrastive_lr),
            anchor_weight=float(args.anchor_weight),
            negative_weight_scale=float(args.negative_weight_scale),
            negative_margin=float(args.negative_margin),
        )
    else:
        fused_norm, fusion_metadata = fuse_landmark_descriptors(
            source_features.float(),
            observations,
            trust_alpha=float(args.trust_alpha),
            min_native_cosine=float(args.min_native_cosine),
            min_positive_weight=float(args.min_positive_weight),
            min_positive_views=int(args.min_positive_views),
        )
    fusion_metadata = {**fusion_metadata, "fusion_mode": str(args.fusion_mode), **active_plan_metadata}
    fusion_metadata["observation_positive_weight_scale"] = float(observation_positive_weight_scale)
    source_norm = torch.linalg.norm(source_features.float(), dim=-1, keepdim=True).clamp_min(1e-12)
    restored = fused_norm * source_norm
    if source_features.dtype.is_floating_point:
        restored = restored.to(dtype=source_features.dtype)

    if output.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"output_log_dir already exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)
    _dump_pickle(output / "keypoints_features.pkl", restored)

    impact_split = _split_name(impact)
    cache_split = _split_name(cache)
    test_split_used = _is_test_split(impact_split) or _is_test_split(cache_split) or _uses_official_test(impact) or _uses_official_test(cache)
    split_audit = {
        "schema_version": "ulfloc_solver_feedback_feature_log_split_audit_v1",
        "impact_split_name": impact_split,
        "observation_cache_split_name": cache_split,
        "test_split_used": bool(test_split_used),
        "official_test_used": bool(test_split_used),
        "same_sampled_idx": True,
        "role": "offline_solver_feedback_descriptor_fusion_export",
    }
    metrics = {
        **fusion_metadata,
        "scene": str(args.scene),
        "source_log_dir": str(source),
        "output_log_dir": str(output),
        "observation_cache": str(Path(args.observation_cache)),
        "observation_descriptor_source": observation_descriptor_source,
        "observation_format": observation_format,
        "impact_attribution": str(Path(args.impact_attribution)),
        "active_fusion_plan": None if active_plan_path is None else str(active_plan_path),
        "sampled_count": int(sampled_idx.numel()),
        "descriptor_shape": list(restored.shape),
        "split_audit": split_audit,
    }
    manifest = {
        "schema_version": "ulfloc_solver_feedback_feature_log_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv if argv is None else [sys.executable, "-m", "loc_gs.scripts.build_ulfloc_solver_feedback_feature_log", *argv],
        "scene": str(args.scene),
        "source_log_dir": str(source),
        "output_log_dir": str(output),
        "observation_cache": str(Path(args.observation_cache)),
        "observation_descriptor_source": observation_descriptor_source,
        "observation_format": observation_format,
        "impact_attribution": str(Path(args.impact_attribution)),
        "active_fusion_plan": None if active_plan_path is None else str(active_plan_path),
        "fusion_metadata": fusion_metadata,
        "sampled_count": int(sampled_idx.numel()),
        "descriptor_dim": int(restored.shape[1]),
        "same_sparse_geometry": True,
        "same_sampled_idx": True,
        "observation_positive_weight_scale": float(observation_positive_weight_scale),
        "sparse_only_descriptor_override": True,
        "branch_selection": False,
        "split_audit": split_audit,
        "official_test_used": bool(test_split_used),
    }
    _write_json(output / "solver_feedback_feature_log_manifest.json", manifest)
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "metrics_summary.json", metrics)
    _write_json(output / "split_audit.json", split_audit)
    (output / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"output_log_dir": str(output), **fusion_metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
