"""Lorentz geometry utilities and cross-curvature bilinear distance.

The code uses the Lorentz convention
    <x, y>_L = -x_0 y_0 + sum_i x_i y_i
for a hyperboloid with curvature c < 0 and radius R = 1 / sqrt(-c).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _as_column(curvature: Tensor) -> Tensor:
    if curvature.dim() == 0:
        return curvature.reshape(1, 1)
    if curvature.dim() == 1:
        return curvature.unsqueeze(-1)
    return curvature


def curvature_to_radius(curvature: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert negative curvature c to radius R = 1 / sqrt(-c)."""
    curvature = _as_column(curvature)
    positive_k = (-curvature).clamp_min(eps)
    return torch.rsqrt(positive_k)


def stable_acosh(x: Tensor, eps: float = 1e-6, max_value: float = 1e6) -> Tensor:
    """Numerically guarded arcosh with a valid domain."""
    x = torch.clamp(x, min=1.0 + eps, max=max_value)
    return torch.acosh(x)


def minkowski_dot(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    """Standard Lorentzian inner product with signature (-, +, ..., +)."""
    time = -x[..., :1] * y[..., :1]
    space = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    out = time + space
    return out if keepdim else out.squeeze(-1)


def lorentz_expmap0(
    tangent: Tensor,
    curvature: Tensor,
    eps: float = 1e-8,
    max_norm: float = 15.0,
) -> Tensor:
    """Map Euclidean tangent vectors at the origin to a Lorentz hyperboloid.

    Args:
        tangent: Tensor of shape ``[..., d]``.
        curvature: Negative curvature tensor broadcastable to ``[..., 1]``.
        eps: Numerical floor for divisions.
        max_norm: Upper bound on sqrt(-c) * ||u|| before cosh/sinh.

    Returns:
        Tensor of shape ``[..., d + 1]`` whose first coordinate is time-like.
    """
    curvature = _as_column(curvature).to(device=tangent.device, dtype=tangent.dtype)
    sqrt_k = torch.sqrt((-curvature).clamp_min(eps))
    tangent_norm = tangent.norm(dim=-1, keepdim=True).clamp_min(eps)

    theta = (sqrt_k * tangent_norm).clamp(max=max_norm)
    radius = torch.rsqrt((-curvature).clamp_min(eps))

    time = radius * torch.cosh(theta)
    spatial_scale = torch.sinh(theta) / (sqrt_k * tangent_norm)
    spatial = spatial_scale * tangent
    return torch.cat([time, spatial], dim=-1)


def project_to_hyperboloid(spatial: Tensor, curvature: Tensor, eps: float = 1e-8) -> Tensor:
    """Project spatial coordinates to the Lorentz sheet for curvature c < 0."""
    curvature = _as_column(curvature).to(device=spatial.device, dtype=spatial.dtype)
    radius = curvature_to_radius(curvature, eps=eps)
    time = torch.sqrt(spatial.pow(2).sum(dim=-1, keepdim=True) + radius.pow(2))
    return torch.cat([time, spatial], dim=-1)


class LorentzBilinearDistance(nn.Module):
    """Cross-curvature Lorentz distance with a learnable symmetric metric tensor.

    For points x and y with radii R_x and R_y, this module computes

        arg = -x^T W y / (R_x R_y)
        d(x, y) = sqrt(R_x R_y) * arcosh(clamp(arg))

    ``W`` is initialized as the standard Minkowski metric and learned as a
    symmetric perturbation around that stable starting point.
    """

    def __init__(
        self,
        coordinate_dim: int,
        eps: float = 1e-6,
        max_arg: float = 1e6,
        metric_delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if coordinate_dim < 2:
            raise ValueError("coordinate_dim must include time plus at least one spatial dim")

        base_metric = torch.eye(coordinate_dim)
        base_metric[0, 0] = -1.0
        self.register_buffer("base_metric", base_metric)
        self.metric_delta = nn.Parameter(torch.zeros(coordinate_dim, coordinate_dim))
        self.eps = eps
        self.max_arg = max_arg
        self.metric_delta_scale = metric_delta_scale

    @property
    def metric_tensor(self) -> Tensor:
        symmetric_delta = 0.5 * (self.metric_delta + self.metric_delta.transpose(0, 1))
        return self.base_metric + self.metric_delta_scale * symmetric_delta

    def bilinear_form(self, x: Tensor, y: Tensor) -> Tensor:
        """Element-wise x^T W y for matching leading dimensions."""
        weighted_x = x @ self.metric_tensor
        return (weighted_x * y).sum(dim=-1, keepdim=True)

    def forward(self, x: Tensor, y: Tensor, c_x: Tensor, c_y: Tensor) -> Tensor:
        """Element-wise cross-curvature distance for paired points."""
        scale = self.bilinear_form(x, y)
        radius_x = curvature_to_radius(c_x, eps=self.eps).to(dtype=x.dtype, device=x.device)
        radius_y = curvature_to_radius(c_y, eps=self.eps).to(dtype=x.dtype, device=x.device)
        denom = (radius_x * radius_y).clamp_min(self.eps)

        arg = -scale / denom
        effective_radius = torch.sqrt(denom)
        return (effective_radius * stable_acosh(arg, eps=self.eps, max_value=self.max_arg)).squeeze(-1)

    def pairwise(self, x: Tensor, y: Tensor, c_x: Tensor, c_y: Tensor) -> Tensor:
        """Pairwise distances between ``x`` with shape [B, D] and ``y`` [N, D]."""
        metric = self.metric_tensor
        scale = torch.einsum("bd,de,ne->bn", x, metric, y)

        radius_x = curvature_to_radius(c_x, eps=self.eps).to(dtype=x.dtype, device=x.device)
        radius_y = curvature_to_radius(c_y, eps=self.eps).to(dtype=x.dtype, device=x.device)
        denom = (radius_x @ radius_y.transpose(0, 1)).clamp_min(self.eps)

        arg = -scale / denom
        effective_radius = torch.sqrt(denom)
        return effective_radius * stable_acosh(arg, eps=self.eps, max_value=self.max_arg)

    def regularization(self) -> Tensor:
        """Small penalty that keeps the learned metric near the Minkowski prior."""
        symmetric_delta = 0.5 * (self.metric_delta + self.metric_delta.transpose(0, 1))
        return symmetric_delta.pow(2).mean()
