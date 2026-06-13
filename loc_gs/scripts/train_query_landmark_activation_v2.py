#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.training.query_landmark_activation_v2 import (
    DEFAULT_LANDMARK_TOKEN_COMPONENTS,
    QueryLandmarkActivationV2,
    query_landmark_activation_v2_loss,
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
    audit_split = str(audit.get("split_name", "")).strip().lower()
    if (
        split_name == "test"
        or audit_split == "test"
        or bool(cache.get("test_split_used", False))
        or bool(cache.get("official_test_used", False))
        or bool(audit.get("test_split_used", False))
        or bool(audit.get("official_test_used", False))
    ):
        raise ValueError("refusing to train activation v2 from test split")


def _load_cache(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"activation v2 cache must contain a dict: {path}")
    _reject_test_split(payload)
    if "examples" not in payload or not isinstance(payload["examples"], list):
        raise KeyError("activation v2 cache must contain an examples list")
    return payload


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _example_labels(example: Mapping[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "geometry_visible": _tensor(example.get("geometry_visible", []), device=device),
        "solver_positive": _tensor(example.get("solver_positive", []), device=device),
        "harmful_negative": _tensor(example.get("harmful_negative", []), device=device),
        "protected_support": _tensor(example.get("protected_support", []), device=device),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train token-based query-conditioned ULF-Loc landmark activation v2.")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--attention_top_k", default=64, type=int)
    parser.add_argument("--top_n", default=None, type=int)
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
            "schema_version": "query_landmark_activation_v2_split_audit_v1",
            "split_name": split_name,
            "test_split_used": False,
            "official_test_used": False,
            "audit_status": "unknown",
        }
    query_token_dim = int(cache.get("query_token_dim", 0))
    landmark_token_dim = int(cache.get("landmark_token_dim", 0))
    examples = cache["examples"]
    if query_token_dim <= 0 or landmark_token_dim <= 0:
        for example in examples:
            if not isinstance(example, Mapping):
                continue
            if "query_tokens" in example and "landmark_tokens" in example:
                query_token_dim = int(torch.as_tensor(example["query_tokens"]).shape[-1])
                landmark_token_dim = int(torch.as_tensor(example["landmark_tokens"]).shape[-1])
                break
    if query_token_dim <= 0 or landmark_token_dim <= 0:
        raise ValueError("activation v2 cache requires query_token_dim and landmark_token_dim")

    device = torch.device(str(args.device))
    global_landmark_tokens = None
    if "landmark_tokens" in cache:
        global_landmark_tokens = torch.as_tensor(cache["landmark_tokens"], dtype=torch.float32)
        if global_landmark_tokens.dim() != 2:
            raise ValueError("cache landmark_tokens must have shape [num_landmarks, landmark_token_dim]")
    model = QueryLandmarkActivationV2(
        query_token_dim=query_token_dim,
        landmark_token_dim=landmark_token_dim,
        hidden_dim=int(args.hidden_dim),
        attention_top_k=int(args.attention_top_k),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, float | int]] = []
    used_example_count = 0
    if int(args.epochs) > 0:
        if not examples:
            raise ValueError("activation v2 cache has no training examples")
        for epoch in range(int(args.epochs)):
            total_loss = 0.0
            used = 0
            for example in examples:
                if not isinstance(example, Mapping):
                    continue
                if "query_tokens" not in example:
                    raise KeyError("activation v2 examples must contain query_tokens")
                if "landmark_tokens" not in example and global_landmark_tokens is None:
                    raise KeyError("activation v2 cache must contain landmark_tokens or examples must contain landmark_tokens")
                query_tokens = _tensor(example["query_tokens"], device=device)
                landmark_source = example.get("landmark_tokens", global_landmark_tokens)
                landmark_tokens = _tensor(landmark_source, device=device)
                logits = model(query_tokens=query_tokens, landmark_tokens=landmark_tokens)
                terms = query_landmark_activation_v2_loss(logits, _example_labels(example, device=device))
                loss = terms["total_loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                used += 1
            used_example_count = int(used)
            history.append({"epoch": int(epoch), "mean_loss": float(total_loss / max(used, 1)), "example_count": int(used)})

    top_n = int(cache.get("top_n", 0) if args.top_n is None else args.top_n)
    landmark_ids = torch.as_tensor(cache.get("landmark_ids", []), dtype=torch.long).reshape(-1).cpu()
    checkpoint = {
        "activation_model_type": "v2_tokens",
        "query_feature_mode": "token_cross_attention",
        "landmark_token_components": list(cache.get("landmark_token_components", DEFAULT_LANDMARK_TOKEN_COMPONENTS)),
        "state_dict": model.cpu().state_dict(),
        "query_token_dim": int(query_token_dim),
        "landmark_token_dim": int(landmark_token_dim),
        "hidden_dim": int(args.hidden_dim),
        "attention_top_k": int(args.attention_top_k),
        "top_n": int(top_n),
        "landmark_ids": landmark_ids,
        "safe_core_indices": torch.as_tensor(cache.get("safe_core_indices", []), dtype=torch.long).reshape(-1).cpu(),
        "source_cache_path": str(args.cache),
        "split_name": split_name,
        "split_audit": dict(split_audit),
    }
    checkpoint_path = output_dir / "landmark_activation_v2.pth"
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        "schema_version": "query_landmark_activation_v2_training_metrics_v1",
        "activation_model_type": "v2_tokens",
        "query_feature_mode": "token_cross_attention",
        "uses_mean_query_descriptor": False,
        "landmark_token_components": list(cache.get("landmark_token_components", DEFAULT_LANDMARK_TOKEN_COMPONENTS)),
        "split_name": split_name,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "attention_top_k": int(args.attention_top_k),
        "top_n": int(top_n),
        "available_example_count": int(len(examples)),
        "trained_example_count": int(used_example_count if int(args.epochs) > 0 else 0),
        "landmark_count": int(landmark_ids.numel()) if landmark_ids.numel() else None,
        "final_loss": float(history[-1]["mean_loss"]) if history else None,
        "history": history,
    }
    manifest = {
        "schema_version": "query_landmark_activation_v2_training_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "source_cache_path": str(args.cache),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
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
