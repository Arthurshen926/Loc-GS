#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from loc_gs.localization.query_landmark_activation import QueryLandmarkActivationNet


def activation_loss(
    scores: torch.Tensor,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    score = scores.reshape(-1)
    if score.numel() == 0:
        return score.sum() * 0.0
    positive = torch.as_tensor(positive_ids, dtype=torch.long, device=score.device).reshape(-1)
    negative = torch.as_tensor(negative_ids, dtype=torch.long, device=score.device).reshape(-1)
    positive = positive[(positive >= 0) & (positive < score.numel())]
    negative = negative[(negative >= 0) & (negative < score.numel())]
    target = torch.zeros_like(score)
    if positive.numel() > 0:
        target[positive] = 1.0
    bce = F.binary_cross_entropy_with_logits(score, target)
    if positive.numel() == 0 or negative.numel() == 0:
        return bce
    pos = score[positive]
    neg = score[negative]
    pair = torch.relu(score.new_tensor(float(margin)) - pos.reshape(-1, 1) + neg.reshape(1, -1)).mean()
    return bce + pair


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


def _seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _reject_test_split(cache: Mapping[str, Any]) -> None:
    split_name = str(cache.get("split_name", cache.get("split", ""))).strip().lower()
    split_audit = cache.get("split_audit", {})
    audit = split_audit if isinstance(split_audit, Mapping) else {}
    if split_name == "test" or bool(audit.get("test_split_used")) or bool(audit.get("official_test_used")):
        raise ValueError("refusing to train landmark activation from test split")


def _load_cache(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"activation cache must contain a dict: {path}")
    _reject_test_split(payload)
    for key in ("landmark_features", "examples"):
        if key not in payload:
            raise KeyError(f"activation cache is missing {key}")
    return payload


def _index_tensor(value: Any, *, landmark_count: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
    tensor = tensor[(tensor >= 0) & (tensor < int(landmark_count))]
    return torch.unique(tensor, sorted=True)


def _example_indices(
    example: Mapping[str, Any],
    *,
    index_key: str,
    landmark_key: str,
    landmark_id_to_row: Mapping[int, int],
    landmark_count: int,
    device: torch.device,
) -> torch.Tensor:
    if index_key in example:
        return _index_tensor(example[index_key], landmark_count=landmark_count, device=device)
    ids = []
    for landmark_id in torch.as_tensor(example.get(landmark_key, []), dtype=torch.long).reshape(-1).tolist():
        row = landmark_id_to_row.get(int(landmark_id))
        if row is not None:
            ids.append(int(row))
    return _index_tensor(ids, landmark_count=landmark_count, device=device)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train query-conditioned ULF-Loc landmark activation network.")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--top_n", default=None, type=int)
    parser.add_argument("--margin", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    _seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(Path(args.cache))
    split_name = str(cache.get("split_name", "unknown"))
    split_audit = cache.get("split_audit", {})
    if not isinstance(split_audit, Mapping):
        split_audit = {
            "schema_version": "ulfloc_landmark_activation_training_split_audit_v1",
            "audit_status": "unknown",
            "split_name": split_name,
            "test_split_used": False,
            "official_test_used": False,
        }

    device = torch.device(str(args.device))
    landmark_features = torch.as_tensor(cache["landmark_features"], dtype=torch.float32, device=device)
    if landmark_features.dim() != 2:
        raise ValueError("activation cache landmark_features must have shape (landmark_count, landmark_dim)")
    examples = cache.get("examples", [])
    if not isinstance(examples, list):
        raise TypeError("activation cache examples must be a list")
    landmark_ids = torch.as_tensor(
        cache.get("landmark_ids", torch.arange(landmark_features.shape[0])),
        dtype=torch.long,
    ).reshape(-1)
    landmark_id_to_row = {int(landmark_id): int(row) for row, landmark_id in enumerate(landmark_ids.tolist())}
    query_dim = int(cache.get("query_dim", 0))
    if query_dim <= 0 and examples:
        first = examples[0]
        if isinstance(first, Mapping):
            query_dim = int(torch.as_tensor(first["query_global_feature"]).reshape(-1).numel())
    if query_dim <= 0:
        raise ValueError("activation cache query_dim is required")
    landmark_dim = int(cache.get("landmark_dim", landmark_features.shape[1]))
    top_n = int(cache.get("top_n", 0) if args.top_n is None else args.top_n)
    model = QueryLandmarkActivationNet(query_dim=query_dim, landmark_dim=landmark_dim, hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: list[dict[str, float | int]] = []
    used_example_count = 0
    if int(args.epochs) > 0:
        if not examples:
            raise ValueError("activation cache has no training examples")
        for epoch in range(int(args.epochs)):
            model.train()
            total_loss = 0.0
            used = 0
            for example in examples:
                if not isinstance(example, Mapping):
                    continue
                query = torch.as_tensor(example["query_global_feature"], dtype=torch.float32, device=device).reshape(-1)
                positive = _example_indices(
                    example,
                    index_key="positive_ids",
                    landmark_key="positive_landmark_ids",
                    landmark_id_to_row=landmark_id_to_row,
                    landmark_count=int(landmark_features.shape[0]),
                    device=device,
                )
                negative = _example_indices(
                    example,
                    index_key="negative_ids",
                    landmark_key="negative_landmark_ids",
                    landmark_id_to_row=landmark_id_to_row,
                    landmark_count=int(landmark_features.shape[0]),
                    device=device,
                )
                scores = model(query, landmark_features)
                loss = activation_loss(scores, positive, negative, margin=float(args.margin))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                used += 1
            used_example_count = int(used)
            history.append({"epoch": int(epoch), "mean_loss": float(total_loss / max(used, 1)), "example_count": int(used)})

    metrics = {
        "schema_version": "ulfloc_landmark_activation_training_metrics_v1",
        "split_name": split_name,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "top_n": int(top_n),
        "margin": float(args.margin),
        "available_example_count": int(len(examples)),
        "trained_example_count": int(used_example_count if int(args.epochs) > 0 else 0),
        "landmark_count": int(landmark_features.shape[0]),
        "safe_core_count": int(torch.as_tensor(cache.get("safe_core_ids", [])).numel()),
        "final_loss": float(history[-1]["mean_loss"]) if history else None,
        "history": history,
    }
    checkpoint_path = output_dir / "landmark_activation.pth"
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "query_dim": int(query_dim),
        "landmark_dim": int(landmark_dim),
        "hidden_dim": int(args.hidden_dim),
        "top_n": int(top_n),
        "safe_core_ids": torch.as_tensor(cache.get("safe_core_ids", []), dtype=torch.long).reshape(-1).cpu(),
        "safe_core_indices": torch.as_tensor(cache.get("safe_core_indices", []), dtype=torch.long).reshape(-1).cpu(),
        "source_cache_path": str(args.cache),
        "split_name": split_name,
        "landmark_ids": landmark_ids.cpu(),
        "metrics": metrics,
        "split_audit": dict(split_audit),
        "metadata": {
            "schema_version": "ulfloc_landmark_activation_checkpoint_v1",
            "source_cache_path": str(args.cache),
            "scene": str(cache.get("scene", "unknown")),
            "split_name": split_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "schema_version": "ulfloc_landmark_activation_training_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "scene": str(cache.get("scene", "unknown")),
        "split": split_name,
        "split_name": split_name,
        "source_cache_path": str(args.cache),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "query_dim": int(query_dim),
        "landmark_dim": int(landmark_dim),
        "top_n": int(top_n),
        "safe_core_count": int(metrics["safe_core_count"]),
        "paper_safe_role": "train_selfmap_query_conditioned_landmark_activation",
        "metrics": metrics,
        "split_audit": dict(split_audit),
    }
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "split_audit.json", dict(split_audit))
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
