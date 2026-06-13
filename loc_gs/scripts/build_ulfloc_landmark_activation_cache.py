#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from loc_gs.training.query_landmark_activation_v2 import (
    DEFAULT_LANDMARK_TOKEN_COMPONENTS,
    build_landmark_activation_v2_landmark_tokens,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"impact_attribution must contain a dict: {path}")
    return payload


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_log_features(source_log_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    feature_path = source_log_dir / "keypoints_features.pkl"
    sampled_path = source_log_dir / "keypoints_sampled_idx.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"source log has no keypoints_features.pkl: {source_log_dir}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"source log has no keypoints_sampled_idx.pkl: {source_log_dir}")
    features = torch.as_tensor(_load_pickle(feature_path), dtype=torch.float32).detach().cpu()
    if features.dim() == 3 and features.shape[0] == 1:
        features = features[0]
    if features.dim() != 2:
        raise ValueError("keypoints_features.pkl must have shape (landmark_count, landmark_dim)")
    sampled_idx = torch.as_tensor(_load_pickle(sampled_path), dtype=torch.long).reshape(-1).detach().cpu()
    if sampled_idx.numel() != features.shape[0]:
        raise ValueError("keypoints_sampled_idx.pkl length must match keypoints_features.pkl rows")
    return features.contiguous(), sampled_idx.contiguous()


def _lower_split(payload: Mapping[str, Any]) -> str:
    return str(payload.get("split_name", payload.get("split", ""))).strip().lower()


def _is_test_split_name(split_name: str) -> bool:
    split = str(split_name).strip().lower()
    return split == "test" or split == "official_test" or split.endswith("_test")


def _split_name(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        raise ValueError("impact_attribution split_name is required")
    return split


def _reject_test_split(payload: Mapping[str, Any]) -> None:
    split_audit = payload.get("split_audit", {})
    audit = split_audit if isinstance(split_audit, Mapping) else {}
    audit_split = str(audit.get("split_name", audit.get("split", ""))).strip()
    if (
        _is_test_split_name(_lower_split(payload))
        or _is_test_split_name(audit_split)
        or bool(audit.get("test_split_used"))
        or bool(audit.get("official_test_used"))
    ):
        raise ValueError("refusing to build landmark activation cache from test split")


def _as_query_feature(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).detach().cpu()
    if tensor.numel() == 0:
        raise ValueError("query_global_feature must be non-empty")
    return tensor.contiguous()


def _normalize_descriptor(desc: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(desc, dtype=torch.float32).reshape(-1).detach().cpu()
    if vector.numel() == 0:
        raise ValueError("query_descriptor must be non-empty")
    return torch.nn.functional.normalize(vector, p=2, dim=0)


def _keypoint_xy(value: Any) -> tuple[float, float]:
    tensor = torch.as_tensor(value if value is not None else [], dtype=torch.float32).reshape(-1)
    if tensor.numel() < 2:
        return 0.0, 0.0
    return float(tensor[0].item()), float(tensor[1].item())


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if number == number else float(default)


def _query_tokens_from_correspondences(
    payload: Mapping[str, Any],
    *,
    xy_scale: float,
) -> dict[str, torch.Tensor]:
    rows = payload.get("correspondences", [])
    if not isinstance(rows, list):
        return {}
    scale = max(float(xy_scale), 1.0)
    grouped: dict[str, dict[int, tuple[float, torch.Tensor]]] = {}
    descriptor_dim = 0
    for row in rows:
        if not isinstance(row, Mapping) or "query_descriptor" not in row:
            continue
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            continue
        try:
            keypoint_index = int(row.get("query_keypoint_index", row.get("match_row_index", len(grouped.get(query_id, {})))))
        except (TypeError, ValueError):
            keypoint_index = len(grouped.get(query_id, {}))
        desc = _normalize_descriptor(torch.as_tensor(row["query_descriptor"], dtype=torch.float32))
        descriptor_dim = max(descriptor_dim, int(desc.numel()))
        x, y = _keypoint_xy(row.get("keypoint_xy", row.get("query_xy")))
        score = _float_value(row.get("detector_score"), 0.0)
        token = torch.cat(
            [
                desc,
                torch.tensor([x / scale, y / scale, score], dtype=torch.float32),
            ],
            dim=0,
        )
        query_rows = grouped.setdefault(query_id, {})
        previous = query_rows.get(keypoint_index)
        if previous is None or score > previous[0]:
            query_rows[keypoint_index] = (score, token)
    if descriptor_dim <= 0:
        return {}
    out: dict[str, torch.Tensor] = {}
    expected_dim = descriptor_dim + 3
    for query_id, by_index in grouped.items():
        tokens: list[torch.Tensor] = []
        for _idx, (_score, token) in sorted(by_index.items()):
            if int(token.numel()) == expected_dim:
                tokens.append(token)
        if tokens:
            out[query_id] = torch.stack(tokens, dim=0).contiguous()
    return out


def _query_tokens_from_payload(payload: Mapping[str, Any], *, xy_scale: float) -> dict[str, torch.Tensor]:
    raw = payload.get("query_tokens")
    if isinstance(raw, Mapping):
        tokens: dict[str, torch.Tensor] = {}
        for query_id, value in raw.items():
            tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
            if tensor.dim() == 1:
                tensor = tensor.reshape(1, -1)
            if tensor.dim() != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
                raise ValueError("query_tokens entries must have shape [tokens, dim]")
            tokens[str(query_id)] = tensor.contiguous()
        if tokens:
            return tokens
    return _query_tokens_from_correspondences(payload, xy_scale=float(xy_scale))


def _query_features_from_payload(payload: Mapping[str, Any], *, allow_records: bool) -> dict[str, torch.Tensor]:
    for key in ("query_features", "query_global_features", "global_query_features"):
        raw = payload.get(key)
        if isinstance(raw, Mapping):
            return {str(query_id): _as_query_feature(feature) for query_id, feature in raw.items()}
    if not allow_records:
        return {}
    features: dict[str, torch.Tensor] = {}
    records = payload.get("records", [])
    if isinstance(records, list):
        for row in records:
            if not isinstance(row, Mapping) or "query_global_feature" not in row:
                continue
            features[str(row.get("query_id", ""))] = _as_query_feature(row["query_global_feature"])
    return features


def _query_features(payload: Mapping[str, Any], external_payload: Mapping[str, Any] | None = None) -> dict[str, torch.Tensor]:
    if external_payload is not None:
        features = _query_features_from_payload(external_payload, allow_records=False)
        if features:
            return features
    features = _query_features_from_payload(payload, allow_records=True)
    if not features:
        raise ValueError(
            "impact_attribution must provide query_features/records with query_global_feature, "
            "or pass --query_features"
        )
    return features


def _mapping_get(mapping: Mapping[Any, Any], key: Any) -> Any:
    if key in mapping:
        return mapping[key]
    text = str(key)
    if text in mapping:
        return mapping[text]
    try:
        integer = int(key)
    except (TypeError, ValueError):
        return None
    return mapping.get(integer)


def _landmark_id_from_entry(entry: Any) -> int | None:
    if isinstance(entry, Mapping):
        for key in ("gaussian_id", "landmark_id", "id"):
            if key in entry:
                try:
                    return int(entry[key])
                except (TypeError, ValueError):
                    return None
        return None
    try:
        return int(entry)
    except (TypeError, ValueError):
        return None


def _ids_from_query_entries(payload: Mapping[str, Any], mapping_key: str, query_id: str) -> list[int]:
    raw_mapping = payload.get(mapping_key, {})
    if not isinstance(raw_mapping, Mapping):
        return []
    raw = _mapping_get(raw_mapping, query_id)
    if raw is None:
        return []
    ids: list[int] = []
    if isinstance(raw, Mapping):
        if "gaussian_ids" in raw or "landmark_ids" in raw:
            key = "gaussian_ids" if "gaussian_ids" in raw else "landmark_ids"
            iterable = raw[key]
        else:
            iterable = raw.keys()
    else:
        iterable = raw
    for entry in iterable:
        landmark_id = _landmark_id_from_entry(entry)
        if landmark_id is not None:
            ids.append(landmark_id)
    seen: set[int] = set()
    unique: list[int] = []
    for landmark_id in ids:
        if landmark_id not in seen:
            unique.append(landmark_id)
            seen.add(landmark_id)
    return unique


def _score_from_landmark_mapping(payload: Mapping[str, Any], mapping_key: str, landmark_id: int, field: str) -> float:
    raw_mapping = payload.get(mapping_key, {})
    if not isinstance(raw_mapping, Mapping):
        return 0.0
    entry = _mapping_get(raw_mapping, landmark_id)
    if entry is None:
        return 0.0
    if isinstance(entry, Mapping):
        value = entry.get(field, entry.get("weight", entry.get("count", 0.0)))
    else:
        value = entry
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _row_indices(landmark_ids: list[int], id_to_row: Mapping[int, int]) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows: list[int] = []
    kept_ids: list[int] = []
    dropped = 0
    for landmark_id in landmark_ids:
        row = id_to_row.get(int(landmark_id))
        if row is None:
            dropped += 1
            continue
        rows.append(int(row))
        kept_ids.append(int(landmark_id))
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(kept_ids, dtype=torch.long),
        int(dropped),
    )


def _binary_label(count: int, indices: torch.Tensor) -> torch.Tensor:
    label = torch.zeros((int(count),), dtype=torch.float32)
    if indices.numel():
        valid = indices[(indices >= 0) & (indices < int(count))].long()
        if valid.numel():
            label[valid] = 1.0
    return label


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build query-conditioned ULF-Loc landmark activation cache.")
    parser.add_argument("--impact_attribution", required=True, type=Path)
    parser.add_argument("--query_features", default=None, type=Path)
    parser.add_argument("--query_token_source", default=None, type=Path)
    parser.add_argument("--query_token_xy_scale", default=640.0, type=float)
    parser.add_argument("--allow_global_query_token_diagnostic", action="store_true")
    parser.add_argument("--source_log_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--top_n", default=2048, type=int)
    parser.add_argument("--safe_core_min_support", default=1.0, type=float)
    parser.add_argument("--safe_core_max_risk", default=0.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_payload(Path(args.impact_attribution))
    _reject_test_split(payload)
    query_feature_payload = None
    if args.query_features is not None:
        query_feature_payload = _load_payload(Path(args.query_features))
        _reject_test_split(query_feature_payload)
    query_token_payload = None
    if args.query_token_source is not None:
        query_token_payload = _load_payload(Path(args.query_token_source))
        _reject_test_split(query_token_payload)
    split_name = _split_name(payload)
    landmark_features, landmark_ids = _load_log_features(Path(args.source_log_dir))
    landmark_tokens = build_landmark_activation_v2_landmark_tokens(landmark_features)
    id_to_row = {int(landmark_id): int(row) for row, landmark_id in enumerate(landmark_ids.tolist())}
    query_tokens_by_id = (
        _query_tokens_from_payload(query_token_payload, xy_scale=float(args.query_token_xy_scale))
        if query_token_payload is not None
        else {}
    )
    diagnostic_only = False
    query_token_source_role = "per_keypoint_runtime_schema"
    if not query_tokens_by_id:
        if not bool(args.allow_global_query_token_diagnostic):
            raise ValueError(
                "--query_token_source with per-keypoint query descriptors is required for activation v2; "
                "use --allow_global_query_token_diagnostic only for diagnostic fallback"
            )
        q_features = _query_features(payload, query_feature_payload)
        query_tokens_by_id = {query_id: feature.reshape(1, -1).contiguous() for query_id, feature in q_features.items()}
        diagnostic_only = True
        query_token_source_role = "global_feature_fallback"
    else:
        q_features = {
            query_id: torch.nn.functional.normalize(tokens[:, :-3].mean(dim=0), p=2, dim=0)
            if tokens.shape[1] > 3
            else torch.nn.functional.normalize(tokens.mean(dim=0), p=2, dim=0)
            for query_id, tokens in query_tokens_by_id.items()
        }

    safe_core_rows: list[int] = []
    safe_core_ids: list[int] = []
    for row, landmark_id in enumerate(landmark_ids.tolist()):
        support = _score_from_landmark_mapping(payload, "landmark_positive", int(landmark_id), "support")
        risk = _score_from_landmark_mapping(payload, "landmark_negative", int(landmark_id), "risk")
        if support >= float(args.safe_core_min_support) and risk <= float(args.safe_core_max_risk):
            safe_core_rows.append(int(row))
            safe_core_ids.append(int(landmark_id))

    examples: list[dict[str, Any]] = []
    dropped_positive_count = 0
    dropped_negative_count = 0
    for query_id in sorted(query_tokens_by_id):
        positive_landmark_ids = _ids_from_query_entries(payload, "detector_positive", query_id)
        negative_landmark_ids = _ids_from_query_entries(payload, "detector_negative", query_id)
        positive_rows, kept_positive_ids, dropped_pos = _row_indices(positive_landmark_ids, id_to_row)
        negative_rows, kept_negative_ids, dropped_neg = _row_indices(negative_landmark_ids, id_to_row)
        dropped_positive_count += dropped_pos
        dropped_negative_count += dropped_neg
        solver_positive = _binary_label(int(landmark_ids.numel()), positive_rows)
        harmful_negative = _binary_label(int(landmark_ids.numel()), negative_rows)
        protected_support = solver_positive.clone()
        geometry_visible = torch.clamp(solver_positive + _binary_label(int(landmark_ids.numel()), torch.tensor(safe_core_rows)), 0.0, 1.0)
        examples.append(
            {
                "query_id": str(query_id),
                "query_global_feature": q_features[query_id],
                "query_tokens": query_tokens_by_id[query_id].contiguous(),
                "geometry_visible": geometry_visible,
                "solver_positive": solver_positive,
                "harmful_negative": harmful_negative,
                "protected_support": protected_support,
                "positive_ids": positive_rows,
                "negative_ids": negative_rows,
                "positive_landmark_ids": kept_positive_ids,
                "negative_landmark_ids": kept_negative_ids,
            }
        )

    query_dim = int(next(iter(q_features.values())).numel())
    query_token_dim = int(next(iter(query_tokens_by_id.values())).shape[1])
    cache = {
        "schema_version": "ulfloc_landmark_activation_cache_v1",
        "scene": str(args.scene),
        "split_name": split_name,
        "query_dim": query_dim,
        "landmark_dim": int(landmark_features.shape[1]),
        "query_token_dim": query_token_dim,
        "landmark_token_dim": int(landmark_tokens.shape[1]),
        "query_token_source_role": query_token_source_role,
        "diagnostic_only": bool(diagnostic_only),
        "landmark_token_components": list(DEFAULT_LANDMARK_TOKEN_COMPONENTS),
        "top_n": int(args.top_n),
        "landmark_ids": landmark_ids,
        "landmark_features": landmark_features,
        "landmark_tokens": landmark_tokens,
        "safe_core_ids": torch.tensor(safe_core_ids, dtype=torch.long),
        "safe_core_indices": torch.tensor(safe_core_rows, dtype=torch.long),
        "examples": examples,
    }
    split_audit = {
        "schema_version": "ulfloc_landmark_activation_cache_split_audit_v1",
        "audit_status": "passed",
        "split_name": split_name,
        "test_split_used": False,
        "official_test_used": False,
        "role": "train_selfmap_query_conditioned_landmark_activation",
        "query_count": int(len(examples)),
        "landmark_count": int(landmark_ids.numel()),
    }
    source_audit = payload.get("split_audit", {})
    if isinstance(source_audit, Mapping):
        split_audit["source_split_audit"] = dict(source_audit)
    cache["split_audit"] = split_audit
    cache_path = output_dir / "cache.pt"
    torch.save(cache, cache_path)

    manifest = {
        "schema_version": "ulfloc_landmark_activation_cache_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "scene": str(args.scene),
        "split": split_name,
        "split_name": split_name,
        "impact_attribution": str(args.impact_attribution),
        "query_features": None if args.query_features is None else str(args.query_features),
        "query_token_source": None if args.query_token_source is None else str(args.query_token_source),
        "query_token_source_role": query_token_source_role,
        "diagnostic_only": bool(diagnostic_only),
        "source_log_dir": str(args.source_log_dir),
        "output_dir": str(output_dir),
        "cache_path": str(cache_path),
        "top_n": int(args.top_n),
        "safe_core_min_support": float(args.safe_core_min_support),
        "safe_core_max_risk": float(args.safe_core_max_risk),
        "query_count": int(len(examples)),
        "landmark_count": int(landmark_ids.numel()),
        "safe_core_count": int(len(safe_core_ids)),
        "dropped_positive_count": int(dropped_positive_count),
        "dropped_negative_count": int(dropped_negative_count),
        "query_token_dim": int(query_token_dim),
        "feedback_enabled": True,
        "branch_selection": False,
        "split_audit": split_audit,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"cache_path": str(cache_path), "query_count": len(examples), "safe_core_count": len(safe_core_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
