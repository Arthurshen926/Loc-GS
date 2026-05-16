#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from loc_gs.localization.selfmap_episode_cache import build_gaussian_advantage_labels
from loc_gs.models.unified_lff import UnifiedLFFDescriptor


def _parse_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        paths.extend(Path(part) for part in str(item).split(",") if part)
    if not paths:
        raise ValueError("at least one episode cache is required")
    return paths


def _load_descriptor_bank(path: str | Path) -> torch.Tensor:
    descriptor_path = Path(path)
    if descriptor_path.suffix.lower() == ".ply":
        from loc_gs.stdloc_native.lff_export import _load_ply_loc_features

        return F.normalize(_load_ply_loc_features(descriptor_path).float(), p=2, dim=-1)
    payload: Any = torch.load(descriptor_path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        desc = payload
    elif isinstance(payload, dict):
        desc = None
        for key in ("descriptors", "base_descriptors", "landmark_desc", "features", "export_descriptors"):
            if key in payload:
                desc = torch.as_tensor(payload[key])
                break
        if desc is None:
            raise KeyError(f"{path} does not contain a descriptor bank")
    else:
        raise ValueError("descriptor bank must be a tensor or a dict containing descriptors")
    desc = torch.as_tensor(desc).float()
    if desc.dim() != 2:
        raise ValueError("descriptor bank must have shape [num_landmarks, descriptor_dim]")
    return F.normalize(desc, p=2, dim=-1)


def _first_positive_or_dustbin(pair_label: torch.Tensor) -> torch.Tensor:
    if pair_label.dim() != 2:
        raise ValueError("pair_label must have shape [N,K]")
    n, topk = int(pair_label.shape[0]), int(pair_label.shape[1])
    out = torch.full((n,), topk, dtype=torch.long)
    any_pos = pair_label.bool().any(dim=1)
    first = pair_label.float().argmax(dim=1).long()
    out[any_pos] = first[any_pos]
    return out


def _pair_label_from_listwise(labels: torch.Tensor, topk: int) -> torch.Tensor:
    label = labels.long().reshape(-1)
    out = torch.zeros(label.shape[0], int(topk), dtype=torch.bool)
    rows = torch.where(label < int(topk))[0]
    if rows.numel() > 0:
        out[rows, label[rows]] = True
    return out


def load_unified_lff_training_tensors(
    base_descriptor_path: str | Path,
    episode_cache_paths: list[str | Path],
    *,
    false_positive_score_threshold: float | None = 0.0,
) -> dict[str, torch.Tensor]:
    loaded_payloads = [(Path(path), torch.load(Path(path), map_location="cpu")) for path in episode_cache_paths]
    base_gaussian_id = torch.empty(0, dtype=torch.long)
    if str(base_descriptor_path):
        base = _load_descriptor_bank(base_descriptor_path)
    else:
        base = None
        for path, payload in loaded_payloads:
            if "base_landmark_desc" not in payload:
                continue
            candidate_base = F.normalize(torch.as_tensor(payload["base_landmark_desc"]).float(), p=2, dim=-1)
            if candidate_base.dim() != 2:
                raise ValueError(f"{path} base_landmark_desc must have shape [N,D]")
            if base is None:
                base = candidate_base
                if "base_gaussian_id" in payload:
                    base_gaussian_id = torch.as_tensor(payload["base_gaussian_id"]).long().reshape(-1)
            elif candidate_base.shape != base.shape:
                raise ValueError("all embedded base_landmark_desc tensors must have the same shape")
        if base is None:
            raise ValueError("base_descriptor_path is empty and no episode cache contains base_landmark_desc")
    if base_gaussian_id.numel() and base_gaussian_id.shape[0] != int(base.shape[0]):
        raise ValueError("base_gaussian_id must have one value per base descriptor")
    query_chunks: list[torch.Tensor] = []
    id_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    pair_label_chunks: list[torch.Tensor] = []
    cosine_chunks: list[torch.Tensor] = []
    gate_targets: list[torch.Tensor] = []
    for path, payload in loaded_payloads:
        if "query_desc" not in payload:
            raise KeyError(f"{path} is missing query_desc")
        if "candidate_landmark_ids" in payload:
            ids = torch.as_tensor(payload["candidate_landmark_ids"]).long()
        elif "landmark_id" in payload:
            ids = torch.as_tensor(payload["landmark_id"]).long()
        else:
            raise KeyError(f"{path} is missing candidate_landmark_ids or landmark_id")
        if ids.dim() != 2:
            raise ValueError("candidate_landmark_ids must have shape [N,K]")
        n, topk = int(ids.shape[0]), int(ids.shape[1])
        query = F.normalize(torch.as_tensor(payload["query_desc"]).float(), p=2, dim=-1)
        if query.shape[0] != n:
            raise ValueError("query_desc and candidate_landmark_ids disagree on N")
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base.shape[0])):
            raise IndexError("candidate_landmark_ids reference descriptors outside the base bank")
        mask = torch.as_tensor(payload.get("candidate_mask", torch.ones(n, topk, dtype=torch.bool))).bool()
        if mask.shape != ids.shape:
            raise ValueError("candidate_mask must match candidate_landmark_ids")
        if "listwise_label" in payload:
            label = torch.as_tensor(payload["listwise_label"]).long().reshape(-1)
        elif "label" in payload and torch.as_tensor(payload["label"]).dim() == 1:
            label = torch.as_tensor(payload["label"]).long().reshape(-1)
        elif "pair_label" in payload:
            label = _first_positive_or_dustbin(torch.as_tensor(payload["pair_label"]).bool())
        else:
            raise KeyError(f"{path} is missing listwise_label, label, or pair_label")
        if label.shape[0] != n:
            raise ValueError("listwise_label must have one label per query")
        label = label.clamp(0, topk)
        if "pair_label" in payload:
            pair_label = torch.as_tensor(payload["pair_label"]).bool()
        else:
            pair_label = _pair_label_from_listwise(label, topk)
        if pair_label.shape != ids.shape:
            raise ValueError("pair_label must match candidate_landmark_ids")
        if "candidate_cosine" in payload:
            cosine = torch.as_tensor(payload["candidate_cosine"]).float()
        elif "cosine" in payload:
            cosine = torch.as_tensor(payload["cosine"]).float()
        else:
            cosine = torch.zeros(n, topk, dtype=torch.float32)
        if cosine.shape != ids.shape:
            raise ValueError("candidate_cosine must match candidate_landmark_ids")
        if "gaussian_advantage_target" in payload:
            target = torch.as_tensor(payload["gaussian_advantage_target"]).float().reshape(-1)
            if target.shape[0] != int(base.shape[0]):
                raise ValueError("gaussian_advantage_target must have one value per base descriptor")
            gate_targets.append(target)
        else:
            valid = mask.reshape(-1)
            stats = build_gaussian_advantage_labels(
                ids.reshape(-1)[valid],
                pair_label.reshape(-1)[valid],
                num_landmarks=int(base.shape[0]),
                pair_scores=cosine.reshape(-1)[valid],
                false_positive_score_threshold=false_positive_score_threshold,
                false_positive_weight=1.0,
            )
            gate_targets.append(stats["target"])
        query_chunks.append(query)
        id_chunks.append(ids)
        mask_chunks.append(mask)
        label_chunks.append(label)
        pair_label_chunks.append(pair_label)
        cosine_chunks.append(cosine)
    if not query_chunks:
        raise ValueError("episode caches are empty")
    if gate_targets:
        gate_target = torch.stack(gate_targets, dim=0).mean(dim=0).clamp(0.0, 1.0)
    else:
        gate_target = torch.full((int(base.shape[0]),), 0.5, dtype=torch.float32)
    return {
        "base_descriptors": base,
        "base_gaussian_id": base_gaussian_id,
        "query_desc": torch.cat(query_chunks, dim=0),
        "candidate_landmark_ids": torch.cat(id_chunks, dim=0),
        "candidate_mask": torch.cat(mask_chunks, dim=0),
        "listwise_label": torch.cat(label_chunks, dim=0),
        "pair_label": torch.cat(pair_label_chunks, dim=0),
        "candidate_cosine": torch.cat(cosine_chunks, dim=0),
        "gaussian_advantage_target": gate_target,
    }


def _ranking_loss(candidate_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, margin: float) -> torch.Tensor:
    topk = int(candidate_logits.shape[1])
    positive = labels < topk
    if not bool(positive.any()):
        return candidate_logits.sum() * 0.0
    rows = torch.where(positive)[0]
    cols = labels[rows]
    pos_score = candidate_logits[rows, cols]
    neg_mask = mask[rows].clone()
    neg_mask[torch.arange(rows.numel(), device=rows.device), cols] = False
    hard_neg = candidate_logits[rows].masked_fill(~neg_mask, -1e4).amax(dim=1)
    valid = hard_neg > -9999.0
    if not bool(valid.any()):
        return candidate_logits.sum() * 0.0
    return F.relu(float(margin) - pos_score[valid] + hard_neg[valid]).mean()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train export-aligned Unified LFF-v2 descriptor gates.")
    parser.add_argument("--base_descriptor_path", required=True)
    parser.add_argument("--episode_cache", nargs="+", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha_max", type=float, default=0.05)
    parser.add_argument("--init_gate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_trust", type=float, default=1.0)
    parser.add_argument("--lambda_gate", type=float, default=1.0)
    parser.add_argument("--lambda_pair", type=float, default=0.25)
    parser.add_argument("--lambda_rank", type=float, default=0.25)
    parser.add_argument("--rank_margin", type=float, default=0.1)
    parser.add_argument("--trust_l1_weight", type=float, default=1.0)
    parser.add_argument(
        "--false_positive_score_threshold",
        type=float,
        default=0.0,
        help="Only negative candidates at or above this cosine contribute to Gaussian gate suppression.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_paths = _parse_paths(list(args.episode_cache))
    tensors = load_unified_lff_training_tensors(
        args.base_descriptor_path,
        cache_paths,
        false_positive_score_threshold=float(args.false_positive_score_threshold),
    )
    dataset = TensorDataset(
        tensors["query_desc"].float(),
        tensors["candidate_landmark_ids"].long(),
        tensors["candidate_mask"].bool(),
        tensors["listwise_label"].long(),
        tensors["pair_label"].bool(),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = UnifiedLFFDescriptor(
        tensors["base_descriptors"],
        alpha_max=float(args.alpha_max),
        init_gate=float(args.init_gate),
        init_selector=0.5,
    ).to(device)
    dustbin_logit = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [dustbin_logit],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    gate_target = tensors["gaussian_advantage_target"].to(device)
    topk = int(tensors["candidate_landmark_ids"].shape[1])
    temp = max(float(args.temperature), 1e-6)
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        total = 0
        sums = {"loss": 0.0, "ce_loss": 0.0, "pair_loss": 0.0, "rank_loss": 0.0, "trust_loss": 0.0, "gate_loss": 0.0}
        correct = 0
        for query_desc, candidate_ids, candidate_mask, listwise_label, pair_label in tqdm(
            loader,
            desc=f"UnifiedLFF epoch {epoch + 1}",
            dynamic_ncols=True,
        ):
            query_desc = query_desc.to(device)
            candidate_ids = candidate_ids.to(device)
            candidate_mask = candidate_mask.to(device)
            listwise_label = listwise_label.to(device).clamp(0, topk)
            pair_label = pair_label.to(device)
            batch = int(query_desc.shape[0])
            candidate_desc = model(candidate_ids.reshape(-1)).view(batch, topk, -1)
            q = F.normalize(query_desc.float(), p=2, dim=-1)
            candidate_logits_raw = (candidate_desc * q[:, None, :]).sum(dim=-1) / temp
            candidate_logits = candidate_logits_raw.masked_fill(~candidate_mask, -1e4)
            dustbin = dustbin_logit.expand(batch, 1)
            logits = torch.cat([candidate_logits, dustbin], dim=1)
            ce_loss = F.cross_entropy(logits, listwise_label)
            pair_valid = candidate_mask
            if bool(pair_valid.any()):
                pair_logits = candidate_logits_raw[pair_valid] - dustbin_logit
                pair_loss = F.binary_cross_entropy_with_logits(pair_logits, pair_label[pair_valid].float())
            else:
                pair_loss = logits.sum() * 0.0
            rank_loss = _ranking_loss(candidate_logits_raw, listwise_label, candidate_mask, float(args.rank_margin))
            trust = model.trust_region_loss(l1_weight=float(args.trust_l1_weight))["loss"]
            gate_loss = F.binary_cross_entropy_with_logits(model.selector_logit, gate_target)
            loss = (
                ce_loss
                + float(args.lambda_pair) * pair_loss
                + float(args.lambda_rank) * rank_loss
                + float(args.lambda_trust) * trust
                + float(args.lambda_gate) * gate_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += batch
            correct += int((logits.detach().argmax(dim=-1) == listwise_label).sum().item())
            for key, value in (
                ("loss", loss),
                ("ce_loss", ce_loss),
                ("pair_loss", pair_loss),
                ("rank_loss", rank_loss),
                ("trust_loss", trust),
                ("gate_loss", gate_loss),
            ):
                sums[key] += float(value.detach().cpu()) * batch
        row = {key: value / max(1, total) for key, value in sums.items()}
        row["epoch"] = float(epoch + 1)
        row["accuracy"] = correct / max(1, total)
        history.append(row)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "model_type": "unified_lff_descriptor",
                "num_landmarks": model.num_landmarks,
                "descriptor_dim": model.descriptor_dim,
                "alpha_max": float(args.alpha_max),
                "temperature": float(args.temperature),
            },
            "state_dict": model.state_dict(),
            "readout_state_dict": {"dustbin_logit": dustbin_logit.detach().cpu()},
            "export_descriptors": model().detach().cpu(),
            "gate": model.selector().detach().cpu(),
            "residual_gate": model.gate().detach().cpu(),
            "base_gaussian_id": tensors["base_gaussian_id"].detach().cpu(),
            "metadata": {
                "episode_cache": [str(path) for path in cache_paths],
                "base_descriptor_path": str(args.base_descriptor_path),
                "samples": int(tensors["query_desc"].shape[0]),
                "topk": topk,
                "single_path_deployment": True,
                "branch_selection": False,
                "selector_gate_decoupled": True,
                "loss_weights": {
                    "trust": float(args.lambda_trust),
                    "gate": float(args.lambda_gate),
                    "pair": float(args.lambda_pair),
                    "rank": float(args.lambda_rank),
                },
                "false_positive_score_threshold": float(args.false_positive_score_threshold),
                "history": history,
            },
        },
        output_path,
    )
    print(f"[unified_lff] wrote {output_path}")


if __name__ == "__main__":
    main()
