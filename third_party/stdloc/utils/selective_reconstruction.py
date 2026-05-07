import math

import torch


def summarize_locability_selection(
    scores,
    top_ratios=(0.05, 0.1, 0.2, 0.3),
    min_weight=0.05,
    gamma=2.0,
):
    """Summarize how a top-ratio locability budget concentrates reconstruction."""
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    scores = scores[torch.isfinite(scores)].clamp(0.0, 1.0)
    if scores.numel() == 0:
        return []

    total_score = scores.sum().clamp_min(1e-12)
    rows = []
    for ratio in top_ratios:
        ratio = float(ratio)
        keep = max(1, min(scores.numel(), math.ceil(scores.numel() * ratio)))
        selected_scores, selected_ids = torch.topk(scores, k=keep)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected[selected_ids] = True
        background_scores = scores[~selected]

        selected_weight = float(min_weight) + selected_scores.pow(float(gamma))
        background_weight = scores.new_full(
            background_scores.shape,
            float(min_weight),
        )
        selected_fraction = float(keep / scores.numel())
        score_mass = float(selected_scores.sum() / total_score)
        rows.append(
            {
                "top_ratio": ratio,
                "selected_points": int(keep),
                "selected_fraction": selected_fraction,
                "score_mass": score_mass,
                "score_mass_gain": score_mass / max(selected_fraction, 1e-12),
                "selected_score_mean": float(selected_scores.mean()),
                "background_score_mean": float(
                    background_scores.mean() if background_scores.numel() else 0.0
                ),
                "selected_to_background_score_ratio": float(
                    selected_scores.mean()
                    / background_scores.mean().clamp_min(1e-12)
                    if background_scores.numel()
                    else 0.0
                ),
                "selected_to_background_weight_ratio": float(
                    selected_weight.mean()
                    / background_weight.mean().clamp_min(1e-12)
                    if background_weight.numel()
                    else 0.0
                ),
            }
        )
    return rows


def summarize_locability_error(
    error,
    locability,
    top_ratios=(0.05, 0.1, 0.2, 0.3),
    mask=None,
):
    """Summarize reconstruction error inside top-locability budgets."""
    error = torch.as_tensor(error, dtype=torch.float32).reshape(-1)
    locability = torch.as_tensor(locability, dtype=torch.float32).reshape(-1)
    if error.shape[0] != locability.shape[0]:
        raise ValueError("error and locability must have the same number of elements")

    valid = torch.isfinite(error) & torch.isfinite(locability)
    if mask is not None:
        mask = torch.as_tensor(mask).reshape(-1).bool()
        if mask.shape[0] != error.shape[0]:
            raise ValueError("mask must have the same number of elements as error")
        valid = valid & mask

    error = error[valid]
    locability = locability[valid].clamp(0.0, 1.0)
    if error.numel() == 0:
        return []

    total_score = locability.sum().clamp_min(1e-12)
    all_error = error.mean()
    rows = []
    for ratio in top_ratios:
        ratio = float(ratio)
        keep = max(1, min(error.numel(), math.ceil(error.numel() * ratio)))
        selected_scores, selected_ids = torch.topk(locability, k=keep)
        selected = torch.zeros_like(locability, dtype=torch.bool)
        selected[selected_ids] = True
        background = ~selected

        selected_error = error[selected].mean()
        background_error = (
            error[background].mean() if background.any() else error.new_tensor(0.0)
        )
        selected_fraction = float(keep / error.numel())
        score_mass = float(selected_scores.sum() / total_score)
        rows.append(
            {
                "top_ratio": ratio,
                "selected_points": int(keep),
                "selected_fraction": selected_fraction,
                "score_mass": score_mass,
                "score_mass_gain": score_mass / max(selected_fraction, 1e-12),
                "all_error_mean": float(all_error),
                "selected_error_mean": float(selected_error),
                "background_error_mean": float(background_error),
                "selected_to_background_error_ratio": float(
                    selected_error / background_error.clamp_min(1e-12)
                    if background.any()
                    else 0.0
                ),
                "selected_score_mean": float(selected_scores.mean()),
                "background_score_mean": float(
                    locability[background].mean() if background.any() else 0.0
                ),
            }
        )
    return rows
