#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from loc_gs.localization.scene_matcher import (
    DEFAULT_LISTWISE_SCALAR_DIM,
    DEFAULT_LISTWISE_EXTRA_FEATURES,
    DEFAULT_SCALAR_DIM,
    LISTWISE_EXTRA_FEATURE_DIMS,
    SceneMatchListwiseNet,
    SceneMatchNet,
    listwise_extra_feature_dim,
)


def _parse_pair_files(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        paths.extend(Path(part) for part in item.split(",") if part)
    if not paths:
        raise ValueError("--pair_files must contain at least one path")
    return paths


def load_pair_tensors(paths: list[Path]) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "query_desc": [],
        "landmark_desc": [],
        "cosine": [],
        "margin": [],
        "query_score": [],
        "landmark_prior": [],
        "label": [],
    }
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        for key in chunks:
            if key == "query_score" and key not in payload:
                if "label" not in payload:
                    raise KeyError(f"{path} is missing label")
                chunks[key].append(torch.ones_like(torch.as_tensor(payload["label"]).float()).reshape(-1))
                continue
            if key not in payload:
                raise KeyError(f"{path} is missing {key}")
            tensor = torch.as_tensor(payload[key])
            if key not in {"query_desc", "landmark_desc"}:
                tensor = tensor.reshape(-1)
            chunks[key].append(tensor)
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def load_listwise_tensors(paths: list[Path]) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "query_desc": [],
        "landmark_desc": [],
        "cosine": [],
        "margin": [],
        "query_score": [],
        "landmark_prior": [],
        "candidate_mask": [],
        "reprojection_error": [],
        "label": [],
    }
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        for key in chunks:
            if key == "query_score" and key not in payload:
                if "label" not in payload:
                    raise KeyError(f"{path} is missing label")
                chunks[key].append(torch.ones_like(torch.as_tensor(payload["label"]).float()).reshape(-1))
                continue
            if key == "candidate_mask" and key not in payload:
                if "landmark_desc" not in payload:
                    raise KeyError(f"{path} is missing landmark_desc")
                landmark_desc = torch.as_tensor(payload["landmark_desc"])
                chunks[key].append(torch.ones(landmark_desc.shape[:2], dtype=torch.bool))
                continue
            if key == "reprojection_error" and key not in payload:
                if "landmark_desc" not in payload:
                    raise KeyError(f"{path} is missing landmark_desc")
                landmark_desc = torch.as_tensor(payload["landmark_desc"])
                chunks[key].append(torch.full(landmark_desc.shape[:2], float("inf"), dtype=torch.float32))
                continue
            if key not in payload:
                raise KeyError(f"{path} is missing {key}")
            tensor = torch.as_tensor(payload[key])
            if key == "label":
                tensor = tensor.reshape(-1).long()
            elif key == "candidate_mask":
                tensor = tensor.bool()
            elif key == "reprojection_error":
                tensor = tensor.float()
            chunks[key].append(tensor)
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def listwise_reprojection_verifier_loss(
    logits: torch.Tensor,
    reprojection_error: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    sigma_px: float,
) -> torch.Tensor:
    """Auxiliary verifier loss from self-localization reprojection quality.

    The selector still receives no teacher-only feature at test time.  This loss
    only teaches candidate logits to behave like a soft geometric verifier:
    small reprojection errors should beat dustbin, large errors should not.
    """

    if logits.dim() != 2 or logits.shape[1] < 2:
        raise ValueError("listwise verifier logits must have shape [N, K + 1]")
    topk = int(logits.shape[1] - 1)
    errors = reprojection_error.to(device=logits.device, dtype=torch.float32)
    mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
    if errors.shape != mask.shape or errors.shape[-1] != topk:
        raise ValueError("reprojection_error and candidate_mask must have shape [N, K]")
    valid = mask & torch.isfinite(errors)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    sigma = max(float(sigma_px), 1e-6)
    target = torch.exp(-errors.clamp_min(0.0) / sigma).clamp(0.0, 1.0)
    candidate_vs_dustbin = logits[:, :topk] - logits[:, topk].unsqueeze(1)
    return F.binary_cross_entropy_with_logits(candidate_vs_dustbin[valid], target[valid], reduction="mean")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SceneMatchNet from self-localization pair labels.")
    parser.add_argument("--pair_files", nargs="+", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--include_raw_descriptors", action="store_true")
    parser.add_argument("--listwise", action="store_true")
    parser.add_argument(
        "--listwise_extra_features",
        choices=sorted(LISTWISE_EXTRA_FEATURE_DIMS.keys()),
        default=DEFAULT_LISTWISE_EXTRA_FEATURES,
    )
    parser.add_argument("--listwise_loss_balance", choices=["binary", "none"], default="binary")
    parser.add_argument("--listwise_verifier_loss_weight", type=float, default=0.0)
    parser.add_argument("--listwise_verifier_sigma_px", type=float, default=4.0)
    parser.add_argument("--pos_weight", type=float, default=0.0)
    parser.add_argument("--balanced_batches", action="store_true")
    parser.add_argument("--balanced_positive_fraction", type=float, default=0.5)
    parser.add_argument("--samples_per_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if bool(getattr(args, "listwise", False)):
        _train_listwise(args, device)
        return
    tensors = load_pair_tensors(_parse_pair_files(args.pair_files))
    if tensors["query_desc"].numel() == 0:
        raise ValueError("pair dataset is empty")
    labels = tensors["label"].float().reshape(-1)
    descriptor_dim = int(tensors["query_desc"].shape[-1])
    dataset = TensorDataset(
        tensors["query_desc"].float(),
        tensors["landmark_desc"].float(),
        tensors["cosine"].float().reshape(-1),
        tensors["margin"].float().reshape(-1),
        tensors["query_score"].float().reshape(-1),
        tensors["landmark_prior"].float().reshape(-1),
        labels,
    )
    sampler = None
    shuffle = True
    if bool(args.balanced_batches):
        pos_mask = labels > 0.5
        pos_count = int(pos_mask.sum().item())
        neg_count = int(labels.numel() - pos_count)
        if pos_count > 0 and neg_count > 0:
            positive_fraction = min(max(float(args.balanced_positive_fraction), 1e-4), 1.0 - 1e-4)
            weights = torch.where(
                pos_mask,
                torch.full_like(labels, positive_fraction / float(pos_count)),
                torch.full_like(labels, (1.0 - positive_fraction) / float(neg_count)),
            )
            samples_per_epoch = int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else int(labels.numel())
            sampler = WeightedRandomSampler(weights.double(), num_samples=samples_per_epoch, replacement=True)
            shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = SceneMatchNet(
        descriptor_dim=descriptor_dim,
        scalar_dim=DEFAULT_SCALAR_DIM + 1,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        include_raw_descriptors=bool(args.include_raw_descriptors),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if float(args.pos_weight) > 0.0:
        pos_weight = torch.tensor(float(args.pos_weight), device=device)
    elif bool(args.balanced_batches) and sampler is not None:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos = labels.sum().clamp_min(1.0)
        neg = (labels.numel() - labels.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for q, d, cosine, margin, query_score, prior, label in tqdm(
            loader,
            desc=f"SceneMatch epoch {epoch + 1}",
            dynamic_ncols=True,
        ):
            q = q.to(device)
            d = d.to(device)
            cosine = cosine.to(device)
            margin = margin.to(device)
            query_score = query_score.to(device)
            prior = prior.to(device)
            label = label.to(device)
            logits = model(
                q,
                d,
                cosine=cosine,
                margin=margin,
                landmark_prior=prior,
                extra_scalar_features=query_score,
            )
            loss = loss_fn(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(label.numel())
            total += int(label.numel())
            correct += int(((logits.detach() >= 0.0) == (label > 0.5)).sum().item())
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / max(1, total),
                "accuracy": correct / max(1, total),
            }
        )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config,
            "state_dict": model.state_dict(),
            "metadata": {
                "pair_files": [str(path) for path in _parse_pair_files(args.pair_files)],
                "samples": int(labels.numel()),
                "positive_ratio": float(labels.mean().item()),
                "extra_scalar_features": ["query_score"],
                "pos_weight": float(pos_weight.detach().cpu().item()),
                "balanced_batches": bool(args.balanced_batches),
                "balanced_positive_fraction": float(args.balanced_positive_fraction),
                "samples_per_epoch": int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else int(labels.numel()),
                "history": history,
            },
        },
        output_path,
    )
    print(f"[scene_matcher] wrote {output_path}")


def _train_listwise(args: argparse.Namespace, device: torch.device) -> None:
    tensors = load_listwise_tensors(_parse_pair_files(args.pair_files))
    if tensors["query_desc"].numel() == 0:
        raise ValueError("listwise pair dataset is empty")
    labels = tensors["label"].long().reshape(-1)
    descriptor_dim = int(tensors["query_desc"].shape[-1])
    topk = int(tensors["landmark_desc"].shape[1])
    dataset = TensorDataset(
        tensors["query_desc"].float(),
        tensors["landmark_desc"].float(),
        tensors["cosine"].float(),
        tensors["margin"].float().reshape(-1),
        tensors["query_score"].float().reshape(-1),
        tensors["landmark_prior"].float(),
        tensors["candidate_mask"].bool(),
        tensors["reprojection_error"].float(),
        labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = SceneMatchListwiseNet(
        descriptor_dim=descriptor_dim,
        scalar_dim=DEFAULT_LISTWISE_SCALAR_DIM
        if str(getattr(args, "listwise_extra_features", DEFAULT_LISTWISE_EXTRA_FEATURES))
        == DEFAULT_LISTWISE_EXTRA_FEATURES
        else 4 + listwise_extra_feature_dim(str(args.listwise_extra_features)),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        include_raw_descriptors=bool(args.include_raw_descriptors),
        listwise_extra_features=str(args.listwise_extra_features),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_balance = str(getattr(args, "listwise_loss_balance", "binary"))
    class_weight = None
    if loss_balance == "binary":
        pos = (labels < topk).sum().float().clamp_min(1.0)
        neg = (labels >= topk).sum().float().clamp_min(1.0)
        total = labels.numel()
        class_weight = torch.ones(topk + 1, dtype=torch.float32)
        class_weight[:topk] = float(total) / (2.0 * float(pos))
        class_weight[topk] = float(total) / (2.0 * float(neg))
    elif loss_balance != "none":
        raise ValueError(f"unsupported listwise_loss_balance: {loss_balance}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=None if class_weight is None else class_weight.to(device))
    history: list[dict[str, float]] = []
    verifier_loss_weight = max(0.0, float(getattr(args, "listwise_verifier_loss_weight", 0.0)))
    verifier_sigma_px = float(getattr(args, "listwise_verifier_sigma_px", 4.0))
    verifier_valid = torch.isfinite(tensors["reprojection_error"]) & tensors["candidate_mask"].bool()
    verifier_finite_ratio = float(verifier_valid.float().mean().item())
    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_verifier_loss = 0.0
        total = 0
        correct = 0
        for q, d, cosine, margin, query_score, prior, candidate_mask, reprojection_error, label in tqdm(
            loader,
            desc=f"SceneMatchListwise epoch {epoch + 1}",
            dynamic_ncols=True,
        ):
            q = q.to(device)
            d = d.to(device)
            cosine = cosine.to(device)
            margin = margin.to(device)
            query_score = query_score.to(device)
            prior = prior.to(device)
            candidate_mask = candidate_mask.to(device)
            reprojection_error = reprojection_error.to(device)
            label = label.to(device)
            logits = model(
                q,
                d,
                cosine=cosine,
                margin=margin,
                landmark_prior=prior,
                query_score=query_score,
                candidate_mask=candidate_mask,
            )
            ce_loss = loss_fn(logits, label.clamp(0, topk))
            verifier_loss = listwise_reprojection_verifier_loss(
                logits,
                reprojection_error,
                candidate_mask,
                sigma_px=verifier_sigma_px,
            )
            loss = ce_loss + verifier_loss_weight * verifier_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(label.numel())
            total_ce_loss += float(ce_loss.detach().cpu()) * int(label.numel())
            total_verifier_loss += float(verifier_loss.detach().cpu()) * int(label.numel())
            total += int(label.numel())
            correct += int((logits.detach().argmax(dim=-1) == label.clamp(0, topk)).sum().item())
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / max(1, total),
                "ce_loss": total_ce_loss / max(1, total),
                "verifier_loss": total_verifier_loss / max(1, total),
                "accuracy": correct / max(1, total),
            }
        )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config,
            "state_dict": model.state_dict(),
            "metadata": {
                "pair_files": [str(path) for path in _parse_pair_files(args.pair_files)],
                "samples": int(labels.numel()),
                "topk": int(topk),
                "dustbin_ratio": float((labels == topk).float().mean().item()),
                "extra_scalar_features": ["query_score"]
                if str(args.listwise_extra_features) == "query_score"
                else ["query_score", "candidate_rank", "cosine_gap_to_best"],
                "listwise_extra_features": str(args.listwise_extra_features),
                "listwise_loss_balance": loss_balance,
                "listwise_verifier_loss_weight": float(verifier_loss_weight),
                "listwise_verifier_sigma_px": float(verifier_sigma_px),
                "verifier_finite_ratio": float(verifier_finite_ratio),
                "class_weight": [] if class_weight is None else [float(v) for v in class_weight.tolist()],
                "history": history,
            },
        },
        output_path,
    )
    print(f"[scene_matcher] wrote {output_path}")


if __name__ == "__main__":
    main()
