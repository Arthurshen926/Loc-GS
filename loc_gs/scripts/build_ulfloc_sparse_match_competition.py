#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml

from loc_gs.stdloc_native.query_match_competition import compute_query_match_competition


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(path), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return payload


def query_support_from_sparse_validation_profile(profile: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip().lower()
    if split_name == "test":
        raise ValueError("refusing to use sparse validation profile from test split")
    out: dict[str, dict[int, float]] = {}
    for field_name in ("protected_per_query_support", "validated_per_query_support"):
        raw_field = profile.get(field_name, {})
        if not isinstance(raw_field, Mapping):
            continue
        for raw_query_id, raw_support in raw_field.items():
            if not isinstance(raw_support, Mapping):
                continue
            query_id = str(raw_query_id)
            for raw_gid, raw_value in raw_support.items():
                try:
                    gid = int(raw_gid)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if gid < 0 or value <= 0.0:
                    continue
                out.setdefault(query_id, {})
                out[query_id][gid] = float(out[query_id].get(gid, 0.0) + value)
    return out


def _load_pickle_tensor(path: Path, *, dtype: torch.dtype) -> torch.Tensor:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return torch.as_tensor(payload, dtype=dtype).reshape((-1,) if dtype == torch.long else payload.shape).cpu()


def _load_sampled_and_features(input_log_dir: Path, config: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    sample_cfg = config.get("sample", {})
    if not isinstance(sample_cfg, Mapping):
        sample_cfg = {}
    landmark_file = str(sample_cfg.get("landmark_file_name", "keypoints_sampled_idx.pkl"))
    feature_file = str(sample_cfg.get("kpts_feature_file_name", "keypoints_features.pkl"))
    with (input_log_dir / landmark_file).open("rb") as handle:
        sampled_idx = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1).cpu()
    with (input_log_dir / feature_file).open("rb") as handle:
        features = torch.as_tensor(pickle.load(handle), dtype=torch.float32).cpu()
    if features.ndim != 2:
        features = features.reshape(int(sampled_idx.numel()), -1)
    if int(features.shape[0]) != int(sampled_idx.numel()):
        raise ValueError(f"feature rows {features.shape[0]} do not match sampled_idx length {sampled_idx.numel()}")
    return sampled_idx, F.normalize(features, dim=1)


def _dataset_namespace(args: argparse.Namespace, config: Mapping[str, Any]) -> Namespace:
    gaussian_type = str(config.get("gaussian_type", "3dgs"))
    return Namespace(
        sh_degree=3,
        source_path=str(Path(args.source_path).resolve()),
        feature_type=str(args.feature_type),
        gaussian_type=gaussian_type,
        model_path=str(Path(args.model_path).resolve()),
        images=str(args.images),
        resolution=-1,
        white_background=True,
        longest_edge=int(args.longest_edge),
        data_device=str(args.data_device),
        eval=True,
        speedup=False,
        norm_before_render=bool(config.get("dense", {}).get("norm_before_render", True))
        if isinstance(config.get("dense", {}), Mapping)
        else True,
        render_items=["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
    )


def _load_masks_like_ulfloc(dataset: Namespace) -> Any:
    # Match ULF-Loc's current lookup semantics: it checks dataset.images relative
    # to the process cwd, not source_path/images.
    path = Path(str(dataset.images)) / "masks.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _apply_ulfloc_query_masks(query_image: torch.Tensor, masks: Any, image_name: str) -> torch.Tensor:
    if masks is None:
        return query_image
    with torch.no_grad():
        obj_mask = masks[image_name][0].to(query_image.device)[None]
        sky_mask = masks[image_name][1].to(query_image.device)[None]
        distort_mask = masks[image_name][2].to(query_image.device)[None]
        mask = obj_mask & distort_mask
        query_image = query_image * mask
        query_image[sky_mask.repeat(3, 1, 1) == False] = 0
    return query_image


def _top_score_items(scores: torch.Tensor, *, limit: int = 32) -> list[dict[str, float | int]]:
    positive = torch.where(scores > 0.0)[0]
    if int(positive.numel()) == 0:
        return []
    values = scores[positive]
    order = torch.argsort(values, descending=True)[: int(limit)]
    return [
        {"gid": int(positive[pos].item()), "score": float(values[pos].item())}
        for pos in order
    ]


def _positive_score_map(scores: torch.Tensor, *, limit: int = 0) -> dict[str, float]:
    tensor = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
    positive = torch.where(tensor > 0.0)[0]
    if int(positive.numel()) == 0:
        return {}
    values = tensor[positive]
    order = torch.argsort(values, descending=True)
    if int(limit) > 0:
        order = order[: int(limit)]
    return {
        str(int(positive[pos].item())): float(values[pos].item())
        for pos in order.tolist()
    }


def _write_query_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "query_id",
        "query_feature_count",
        "support_present_count",
        "support_top1_count",
        "support_topk_count",
        "feature_weak_query_feature_count",
        "stealer_count",
        "steal_event_count",
        "support_match_strength_nonzero_count",
        "support_match_strength_total",
        "top_stealer_gid",
        "top_stealer_score",
        "top_support_gid",
        "top_support_strength",
    ]
    lines = [",".join(columns)]
    for row in rows:
        metadata = row.get("metadata", {})
        top = (row.get("top_stealers") or [{}])[0]
        top_support = (row.get("top_support_match_strength") or [{}])[0]
        values = [
            str(row.get("query_id", "")),
            str(metadata.get("query_feature_count", 0)),
            str(metadata.get("support_present_count", 0)),
            str(metadata.get("support_top1_count", 0)),
            str(metadata.get("support_topk_count", 0)),
            str(metadata.get("feature_weak_query_feature_count", 0)),
            str(metadata.get("stealer_count", 0)),
            str(metadata.get("steal_event_count", 0)),
            str(metadata.get("support_match_strength_nonzero_count", 0)),
            str(metadata.get("support_match_strength_total", 0.0)),
            str(top.get("gid", "")),
            str(top.get("score", "")),
            str(top_support.get("gid", "")),
            str(top_support.get("score", "")),
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay ULF-Loc sparse matching and score query-level competition against solver-validated support."
    )
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_log_dir", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--sparse_validation_profile", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--score_margin", default=0.05, type=float)
    parser.add_argument("--query_id", action="append", default=None)
    parser.add_argument("--max_queries", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    split_name = str(args.split_name).strip()
    if split_name.lower() == "test":
        raise ValueError("test split sparse match competition mining is not allowed")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not bool(args.force):
        raise FileExistsError(f"output_dir already exists; pass --force to replace files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_json(Path(args.sparse_validation_profile))
    query_support = query_support_from_sparse_validation_profile(profile)
    requested_queries = set(str(q) for q in (args.query_id or []))
    if requested_queries:
        query_support = {query: support for query, support in query_support.items() if query in requested_queries}
    if int(args.max_queries) > 0:
        query_support = dict(list(sorted(query_support.items()))[: int(args.max_queries)])
    if not query_support:
        raise ValueError("no query support remains after filtering sparse validation profile")

    config = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"cfg must contain a YAML object: {args.cfg}")
    sampled_idx, landmark_features = _load_sampled_and_features(Path(args.input_log_dir), config)
    num_gaussians = int(sampled_idx.max().item()) + 1
    stealer_scores = torch.zeros((num_gaussians,), dtype=torch.float32)
    support_match_strength = torch.zeros((num_gaussians,), dtype=torch.float32)

    ulf_root = Path(args.ulf_root).resolve()
    if str(ulf_root) not in sys.path:
        sys.path.insert(0, str(ulf_root))
    from encoders.feature_extractor import FeatureExtractor  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore

    dataset = _dataset_namespace(args, config)
    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError(f"unsupported Gaussian type: {dataset.gaussian_type}")
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        preload_cameras=True,
    )
    try:
        full_gaussian_count = int(gaussians.get_xyz.shape[0])
    except Exception:
        full_gaussian_count = int(num_gaussians)
    if int(full_gaussian_count) > int(stealer_scores.numel()):
        expanded_stealers = torch.zeros((int(full_gaussian_count),), dtype=torch.float32)
        expanded_stealers[: int(stealer_scores.numel())] = stealer_scores
        stealer_scores = expanded_stealers
        expanded_strength = torch.zeros((int(full_gaussian_count),), dtype=torch.float32)
        expanded_strength[: int(support_match_strength.numel())] = support_match_strength
        support_match_strength = expanded_strength

    device = torch.device(str(args.device))
    extractor = FeatureExtractor(str(args.feature_type)).to(device).eval()
    masks = _load_masks_like_ulfloc(dataset)
    rows: list[dict[str, Any]] = []
    query_dominance: dict[str, dict[str, Any]] = {}
    processed = 0
    for camera in scene.getTestCameras():
        query_id = str(camera.image_name)
        support = query_support.get(query_id)
        if support is None:
            continue
        query_image = camera.original_image.to(device)
        query_image = _apply_ulfloc_query_masks(query_image, masks, query_id)
        with torch.no_grad():
            payload = extractor.detectAndCompute(
                query_image[None],
                top_k=int(config.get("sparse", {}).get("kpts_num", 2048))
                if isinstance(config.get("sparse", {}), Mapping)
                else 2048,
            )[0]
            values = list(payload.values())
            query_features = F.normalize(values[2], dim=-1).detach().cpu()
        result = compute_query_match_competition(
            query_features=query_features,
            sampled_idx=sampled_idx,
            landmark_features=landmark_features,
            query_support=support,
            top_k=int(args.top_k),
            score_margin=float(args.score_margin),
        )
        if int(result.stealer_scores.numel()) > int(stealer_scores.numel()):
            expanded = torch.zeros((int(result.stealer_scores.numel()),), dtype=torch.float32)
            expanded[: int(stealer_scores.numel())] = stealer_scores
            stealer_scores = expanded
        if int(result.support_match_strength.numel()) > int(support_match_strength.numel()):
            expanded = torch.zeros((int(result.support_match_strength.numel()),), dtype=torch.float32)
            expanded[: int(support_match_strength.numel())] = support_match_strength
            support_match_strength = expanded
        stealer_scores[: int(result.stealer_scores.numel())] += result.stealer_scores
        support_match_strength[: int(result.support_match_strength.numel())] += result.support_match_strength
        rows.append(
            {
                "query_id": query_id,
                "support_landmark_count": int(len(support)),
                "metadata": result.metadata,
                "top_stealers": _top_score_items(result.stealer_scores, limit=32),
                "top_support_match_strength": _top_score_items(result.support_match_strength, limit=32),
            }
        )
        query_dominance[query_id] = {
            "support_match_strength": _positive_score_map(result.support_match_strength),
            "match_competition_risk": _positive_score_map(result.stealer_scores),
            "metadata": result.metadata,
        }
        processed += 1

    if processed == 0:
        raise ValueError("none of the sparse validation profile queries were present in loaded test cameras")

    score_path = output_dir / "stealer_scores.pkl"
    with score_path.open("wb") as handle:
        pickle.dump(stealer_scores, handle)
    support_strength_path = output_dir / "support_match_strength.pkl"
    with support_strength_path.open("wb") as handle:
        pickle.dump(support_match_strength, handle)
    _write_json(output_dir / "query_competition.json", {"queries": rows})
    _write_json(
        output_dir / "query_match_dominance.json",
        {
            "schema_version": "ulfloc_query_match_dominance_v1",
            "split_name": split_name,
            "official_test_used": False,
            "queries": query_dominance,
        },
    )
    _write_query_csv(output_dir / "query_competition.csv", rows)

    total_metadata = {
        "query_count": int(processed),
        "nonzero_stealer_count": int((stealer_scores > 0.0).sum().item()),
        "total_steal_score": float(stealer_scores.sum().item()),
        "max_steal_score": float(stealer_scores.max().item()) if stealer_scores.numel() else 0.0,
        "top_stealers": _top_score_items(stealer_scores, limit=64),
        "score_path": str(score_path),
        "support_match_strength_nonzero_count": int((support_match_strength > 0.0).sum().item()),
        "support_match_strength_total": float(support_match_strength.sum().item()),
        "support_match_strength_max": float(support_match_strength.max().item())
        if support_match_strength.numel()
        else 0.0,
        "top_support_match_strength": _top_score_items(support_match_strength, limit=64),
        "support_match_strength_path": str(support_strength_path),
        "query_match_dominance_path": str(output_dir / "query_match_dominance.json"),
        "score_vector_length": int(stealer_scores.numel()),
        "full_gaussian_count": int(full_gaussian_count),
    }
    split_audit = {
        "audit_status": "passed",
        "split_name": split_name,
        "official_test_used": False,
        "paper_safe_for_tuning": True,
        "checks": {
            "query_split": {
                "status": "passed",
                "split_name": split_name,
                "official_test_used": False,
            },
            "sparse_validation_profile_split": {
                "status": "passed",
                "split_name": str(profile.get("split_name", "unknown")),
            },
        },
    }
    manifest = {
        "method": "ulfloc_sparse_match_competition_replay",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "command": " ".join(sys.argv),
        "source_path": str(Path(args.source_path)),
        "model_path": str(Path(args.model_path)),
        "input_log_dir": str(Path(args.input_log_dir)),
        "cfg": str(Path(args.cfg)),
        "sparse_validation_profile": str(Path(args.sparse_validation_profile)),
        "split_name": split_name,
        "hyperparameters": {
            "top_k": int(args.top_k),
            "score_margin": float(args.score_margin),
            "query_id": list(args.query_id or []),
            "max_queries": int(args.max_queries),
            "feature_type": str(args.feature_type),
            "images": str(args.images),
            "longest_edge": int(args.longest_edge),
        },
        "branch_selection": False,
        "single_path_deployment": False,
        "diagnostic_only": True,
        "metrics_summary": total_metadata,
        "split_audit": split_audit,
    }
    _write_json(output_dir / "metrics_summary.json", total_metadata)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    (output_dir / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")
    print(json.dumps(total_metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
