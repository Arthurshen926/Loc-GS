from __future__ import annotations

import torch
from torch import nn


class QueryLandmarkActivationNet(nn.Module):
    """Score fixed 3D landmarks from one query-level feature vector."""

    def __init__(self, query_dim: int, landmark_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.landmark_dim = int(landmark_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_proj = nn.Linear(self.query_dim, self.hidden_dim)
        self.landmark_proj = nn.Linear(self.landmark_dim, self.hidden_dim)
        self.score = nn.Linear(self.hidden_dim, 1)

    def forward(self, query_feature: torch.Tensor, landmark_features: torch.Tensor) -> torch.Tensor:
        query = torch.as_tensor(query_feature, dtype=self.query_proj.weight.dtype, device=self.query_proj.weight.device)
        landmarks = torch.as_tensor(
            landmark_features,
            dtype=self.landmark_proj.weight.dtype,
            device=self.landmark_proj.weight.device,
        )
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if query.dim() != 2 or query.shape[0] != 1:
            raise ValueError("query_feature must have shape (query_dim,) or (1, query_dim)")
        if landmarks.dim() != 2:
            raise ValueError("landmark_features must have shape (landmark_count, landmark_dim)")
        q = self.query_proj(query)
        l = self.landmark_proj(landmarks)
        hidden = torch.relu(l + q)
        return self.score(hidden).reshape(-1)


def select_active_landmarks(
    scores: torch.Tensor,
    *,
    top_n: int,
    safe_core: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return sorted landmark row indices from the safe core plus top-N scores."""

    if isinstance(scores, torch.Tensor):
        score = scores.detach().to(dtype=torch.float32).reshape(-1)
    else:
        score = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    count = int(score.numel())
    k = min(max(0, int(top_n)), count)
    if k > 0:
        selected = [torch.topk(score, k=k).indices.to(dtype=torch.long)]
    else:
        selected = [torch.empty(0, dtype=torch.long, device=score.device)]
    if safe_core is not None:
        safe = torch.as_tensor(safe_core, dtype=torch.long, device=score.device).reshape(-1)
        safe = safe[(safe >= 0) & (safe < count)]
        selected.append(safe)
    merged = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=score.device)
    return torch.unique(merged, sorted=True)
