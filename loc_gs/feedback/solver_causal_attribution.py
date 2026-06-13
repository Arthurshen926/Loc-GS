from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import torch


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _split_is_test(value: str) -> bool:
    split = str(value).strip().lower()
    return split == "test" or split == "official_test" or split.endswith("_test")


def _camera_xyz(row: Mapping[str, Any]) -> torch.Tensor | None:
    if "camera_xyz" not in row:
        return None
    xyz = torch.as_tensor(row["camera_xyz"], dtype=torch.float64).reshape(-1)
    if xyz.numel() < 3:
        return None
    xyz = xyz[:3]
    if not bool(torch.isfinite(xyz).all()):
        return None
    if float(xyz[2].item()) <= 1e-9:
        return None
    return xyz


def _jacobian_rows(camera_xyz: torch.Tensor) -> torch.Tensor:
    """Approximate normalized-projection Jacobian wrt se(3).

    The exact sign convention is irrelevant for conditioning; J^T J is used.
    Coordinates are camera-frame 3D points.
    """

    x = camera_xyz[:, 0]
    y = camera_xyz[:, 1]
    z = camera_xyz[:, 2].clamp_min(1e-9)
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    jx = torch.stack(
        [
            -(x * y) * inv_z2,
            1.0 + (x * x) * inv_z2,
            -y * inv_z,
            inv_z,
            torch.zeros_like(inv_z),
            -x * inv_z2,
        ],
        dim=1,
    )
    jy = torch.stack(
        [
            -(1.0 + (y * y) * inv_z2),
            (x * y) * inv_z2,
            x * inv_z,
            torch.zeros_like(inv_z),
            inv_z,
            -y * inv_z2,
        ],
        dim=1,
    )
    return torch.stack((jx, jy), dim=1).reshape(-1, 6)


def _conditioning_metrics_from_hessian(hessian: torch.Tensor, *, support: int) -> dict[str, float | int]:
    if int(support) < 4:
        return {
            "pnp_conditioning_support_count": int(support),
            "pnp_logdet_jtj": 0.0,
            "pnp_min_eig_jtj": 0.0,
            "pnp_condition_number": 0.0,
        }
    hessian = hessian.to(dtype=torch.float64)
    eig = torch.linalg.eigvalsh(hessian).real.clamp_min(0.0)
    eps = torch.tensor(1e-9, dtype=torch.float64)
    logdet = torch.log(eig + eps).sum()
    min_eig = eig.min()
    max_eig = eig.max()
    condition = max_eig / (min_eig + eps)
    return {
        "pnp_conditioning_support_count": int(support),
        "pnp_logdet_jtj": float(logdet.item()) if math.isfinite(float(logdet.item())) else 0.0,
        "pnp_min_eig_jtj": float(min_eig.item()) if math.isfinite(float(min_eig.item())) else 0.0,
        "pnp_condition_number": float(condition.item()) if math.isfinite(float(condition.item())) else 0.0,
    }


def pnp_conditioning_metrics(rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> dict[str, float | int]:
    points = []
    for row in rows:
        xyz = _camera_xyz(row)
        if xyz is not None:
            points.append(xyz)
    support = int(len(points))
    if support < 4:
        return _conditioning_metrics_from_hessian(torch.zeros((6, 6), dtype=torch.float64), support=support)
    xyz_tensor = torch.stack(points, dim=0).to(dtype=torch.float64)
    jacobian = _jacobian_rows(xyz_tensor)
    hessian = jacobian.T @ jacobian
    return _conditioning_metrics_from_hessian(hessian, support=support)


def _group_conditioning_and_leave_one_out(
    inliers: list[tuple[int, Mapping[str, Any]]],
) -> tuple[dict[str, float | int], dict[int, dict[str, float | int]]]:
    point_rows: list[tuple[int, torch.Tensor]] = []
    for idx, row in inliers:
        xyz = _camera_xyz(row)
        if xyz is not None:
            point_rows.append((idx, xyz))
    support = int(len(point_rows))
    if support < 4:
        full = _conditioning_metrics_from_hessian(torch.zeros((6, 6), dtype=torch.float64), support=support)
        return full, {}

    xyz_tensor = torch.stack([xyz for _idx, xyz in point_rows], dim=0).to(dtype=torch.float64)
    jacobian = _jacobian_rows(xyz_tensor).reshape(support, 2, 6)
    per_point_hessian = torch.matmul(jacobian.transpose(1, 2), jacobian)
    hessian = per_point_hessian.sum(dim=0)
    full = _conditioning_metrics_from_hessian(hessian, support=support)

    if support - 1 < 4:
        logdet_drop = max(0.0, float(full["pnp_logdet_jtj"]))
        min_eig_drop = max(0.0, float(full["pnp_min_eig_jtj"]))
        return (
            full,
            {
                idx: {
                    "minimal_set_counterfactual_support_delta": -1,
                    "minimal_set_logdet_drop": logdet_drop,
                    "minimal_set_min_eig_drop": min_eig_drop,
                }
                for idx, _xyz in point_rows
            },
        )

    eps = torch.tensor(1e-9, dtype=torch.float64)
    identity6 = torch.eye(6, dtype=torch.float64)
    regularized = hessian + eps * identity6
    inv_regularized = torch.linalg.inv(regularized)

    # Exact regularized logdet drop for removing one correspondence's 2x6
    # projection Jacobian: det(H - J_i^T J_i + eps I) =
    # det(H + eps I) * det(I - J_i (H + eps I)^-1 J_i^T).
    small = torch.eye(2, dtype=torch.float64).unsqueeze(0) - jacobian @ inv_regularized @ jacobian.transpose(1, 2)
    det_small = torch.linalg.det(small).real.clamp_min(float(eps.item()))
    logdet_drop = (-torch.log(det_small)).clamp_min(0.0)

    eigvals, eigvecs = torch.linalg.eigh(hessian)
    min_eig = eigvals.real.clamp_min(0.0).min()
    min_vec = eigvecs[:, 0].real
    min_eig_drop = torch.einsum("i,nij,j->n", min_vec, per_point_hessian, min_vec).real.clamp_min(0.0)
    min_eig_drop = torch.minimum(min_eig_drop, min_eig.expand_as(min_eig_drop))

    leave_one_out: dict[int, dict[str, float | int]] = {}
    for row_number, (idx, _xyz) in enumerate(point_rows):
        leave_one_out[idx] = {
            "minimal_set_counterfactual_support_delta": -1,
            "minimal_set_logdet_drop": float(logdet_drop[row_number].item()),
            "minimal_set_min_eig_drop": float(min_eig_drop[row_number].item()),
        }
    return full, leave_one_out


def _inlier_rows(indexed_rows: list[tuple[int, Mapping[str, Any]]]) -> list[tuple[int, Mapping[str, Any]]]:
    return [(idx, row) for idx, row in indexed_rows if _bool(row.get("pnp_inlier", False))]


def _group_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("query_id", "")), str(row.get("source_role", "baseline_trace"))


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id", ""))


def _validate_not_test(row: Mapping[str, Any]) -> None:
    split = str(row.get("split_name", "")).strip()
    if split and _split_is_test(split):
        raise ValueError("test split is not allowed for solver-causal sparse feedback attribution")


def _metrics_delta(candidate: Mapping[str, float | int], baseline: Mapping[str, float | int]) -> dict[str, float | int]:
    return {
        "candidate_vs_baseline_support_delta": int(candidate["pnp_conditioning_support_count"])
        - int(baseline["pnp_conditioning_support_count"]),
        "candidate_vs_baseline_logdet_delta": float(candidate["pnp_logdet_jtj"]) - float(baseline["pnp_logdet_jtj"]),
        "candidate_vs_baseline_min_eig_delta": float(candidate["pnp_min_eig_jtj"]) - float(baseline["pnp_min_eig_jtj"]),
    }


def annotate_sparse_correspondences_with_solver_causal_fields(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add set-level PnP conditioning attribution to sparse correspondence rows.

    The annotations are causal proxies: a correspondence receives a minimal-set
    drop if removing it from the PnP inlier set worsens conditioning; candidate
    traces receive paired deltas against the baseline trace for the same query.
    """

    annotations = [dict() for _ in rows]
    grouped: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    query_row_indices: dict[str, list[int]] = defaultdict(list)
    query_source_metrics: dict[tuple[str, str], dict[str, float | int]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        _validate_not_test(row)
        grouped[_group_key(row)].append((idx, row))
        query_id = _query_id(row)
        if query_id:
            query_row_indices[query_id].append(idx)

    for key, indexed_rows in grouped.items():
        inliers = _inlier_rows(indexed_rows)
        full, leave_one_out = _group_conditioning_and_leave_one_out(inliers)
        query_source_metrics[key] = full
        for idx, row in indexed_rows:
            annotations[idx].update(
                {
                    "pnp_conditioning_support_count": int(full["pnp_conditioning_support_count"]),
                    "pnp_logdet_jtj": float(full["pnp_logdet_jtj"]),
                    "pnp_min_eig_jtj": float(full["pnp_min_eig_jtj"]),
                    "pnp_condition_number": float(full["pnp_condition_number"]),
                    "minimal_set_counterfactual_support_delta": 0,
                    "minimal_set_logdet_drop": 0.0,
                    "minimal_set_min_eig_drop": 0.0,
                }
            )
            if not _bool(row.get("pnp_inlier", False)):
                continue
            annotations[idx].update(
                leave_one_out.get(
                    idx,
                    {
                        "minimal_set_counterfactual_support_delta": 0,
                        "minimal_set_logdet_drop": 0.0,
                        "minimal_set_min_eig_drop": 0.0,
                    },
                )
            )

    paired_query_count = 0
    for query_id, row_indices in query_row_indices.items():
        baseline = query_source_metrics.get((query_id, "baseline_trace"))
        candidate = query_source_metrics.get((query_id, "candidate_trace"))
        if baseline is None or candidate is None:
            continue
        paired_query_count += 1
        delta = _metrics_delta(candidate, baseline)
        for idx in row_indices:
            annotations[idx].update(delta)

    for annotation in annotations:
        annotation.setdefault("candidate_vs_baseline_support_delta", 0)
        annotation.setdefault("candidate_vs_baseline_logdet_delta", 0.0)
        annotation.setdefault("candidate_vs_baseline_min_eig_delta", 0.0)

    return annotations, {
        "schema_version": "sparse_solver_causal_attribution_metrics_v1",
        "solver_causal_attribution_enabled": True,
        "query_source_group_count": int(len(query_source_metrics)),
        "paired_query_count": int(paired_query_count),
        "annotated_correspondence_count": int(len(annotations)),
    }
