import torch
import torch.nn.functional as F


def _nearest_distances(source, target, chunk_size=65536):
    if source.numel() == 0 or target.numel() == 0:
        return source.new_zeros((0,))

    distances = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start : start + chunk_size]
        dists = torch.cdist(chunk, target)
        distances.append(dists.min(dim=1).values)
    return torch.cat(distances, dim=0)


def chamfer_distance_stats(source, target, chunk_size=65536):
    """Compute conservative Gaussian-center vs reference-point geometry stats."""
    source = source.float()
    target = target.float()
    source_to_target = _nearest_distances(source, target, chunk_size=chunk_size)
    target_to_source = _nearest_distances(target, source, chunk_size=chunk_size)

    def summarize(values, prefix):
        if values.numel() == 0:
            zero = 0.0
            return {
                f"{prefix}_mean": zero,
                f"{prefix}_median": zero,
                f"{prefix}_p90": zero,
            }
        return {
            f"{prefix}_mean": values.mean().item(),
            f"{prefix}_median": torch.quantile(values, 0.5).item(),
            f"{prefix}_p90": torch.quantile(values, 0.9).item(),
        }

    stats = {
        "source_count": int(source.shape[0]),
        "target_count": int(target.shape[0]),
    }
    stats.update(summarize(source_to_target, "source_to_target"))
    stats.update(summarize(target_to_source, "target_to_source"))
    stats["symmetric_chamfer_mean"] = 0.5 * (
        stats["source_to_target_mean"] + stats["target_to_source_mean"]
    )
    return stats


def sample_depth_bilinear(depth_map, points_xy):
    if points_xy.numel() == 0:
        return depth_map.new_zeros((0,))

    height, width = depth_map.shape[-2:]
    x = points_xy[:, 0].clamp(0, width - 1)
    y = points_xy[:, 1].clamp(0, height - 1)
    x_norm = 2.0 * x / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        depth_map.view(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(-1)


def projected_depth_consistency(depth_map, points_xy, point_depth):
    """Compare rendered depth with projected sparse-reference point depth."""
    depth_map = depth_map.float().squeeze()
    points_xy = points_xy.float()
    point_depth = point_depth.float().reshape(-1)
    if points_xy.numel() == 0 or point_depth.numel() == 0:
        return {
            "count": 0,
            "abs_error_mean": 0.0,
            "abs_error_median": 0.0,
            "abs_error_p90": 0.0,
            "rel_error_median": 0.0,
        }

    sampled = sample_depth_bilinear(depth_map, points_xy)
    valid = torch.isfinite(sampled) & torch.isfinite(point_depth) & (sampled > 0) & (point_depth > 0)
    if not valid.any():
        return {
            "count": 0,
            "abs_error_mean": 0.0,
            "abs_error_median": 0.0,
            "abs_error_p90": 0.0,
            "rel_error_median": 0.0,
        }

    abs_error = (sampled[valid] - point_depth[valid]).abs()
    rel_error = abs_error / point_depth[valid].clamp_min(1e-6)
    return {
        "count": int(abs_error.numel()),
        "abs_error_mean": abs_error.mean().item(),
        "abs_error_median": torch.quantile(abs_error, 0.5).item(),
        "abs_error_p90": torch.quantile(abs_error, 0.9).item(),
        "rel_error_median": torch.quantile(rel_error, 0.5).item(),
    }
