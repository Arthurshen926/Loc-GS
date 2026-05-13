#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from loc_gs.localization.scene_matcher import DEFAULT_SCALAR_DIM, SceneMatchNet


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


if __name__ == "__main__":
    main()
