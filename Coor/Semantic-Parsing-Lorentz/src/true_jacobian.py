"""True autograd Jacobians for the Lorentz projection head.

These utilities compute derivatives with respect to the CLS/text embedding
only. They do not differentiate through token IDs or a full BERT input
sequence, which keeps the method exact for the Lorentz head while avoiding a
full encoder-input Jacobian.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor


def compute_true_lorentz_jacobian_wrt_cls(
    model: torch.nn.Module,
    cls_embedding: Tensor,
    create_graph: bool = True,
    vectorize: bool = True,
    max_samples: Optional[int] = None,
) -> Tensor:
    """Compute ``J = d query_lorentz_point / d cls_embedding``.

    Args:
        model: A model exposing ``lorentz_point_from_cls(cls_embedding)``.
        cls_embedding: Tensor of shape ``[B, H]``.
        create_graph: Keep the Jacobian graph for higher-order training
            gradients. This is expensive but needed when the local metric is
            part of the training objective.
        vectorize: Accepted for API compatibility. The implementation uses a
            batched autograd loop to avoid materializing ``[B, D, B, H]``.
        max_samples: Optional chunk size for memory control. All samples are
            still processed and concatenated.

    Returns:
        Tensor of shape ``[B, D, H]``, where ``D = tangent_dim + 1``.
    """
    del vectorize
    return _compute_true_jacobian_wrt_cls(
        lambda cls: _first_tensor(model.lorentz_point_from_cls(cls)),
        cls_embedding,
        create_graph=create_graph,
        max_samples=max_samples,
    )


def compute_true_tangent_jacobian_wrt_cls(
    model: torch.nn.Module,
    cls_embedding: Tensor,
    create_graph: bool = True,
    vectorize: bool = True,
    max_samples: Optional[int] = None,
) -> Tensor:
    """Compute ``J = d tangent / d cls_embedding``.

    Args:
        model: A model exposing ``tangent_from_cls(cls_embedding)``.
        cls_embedding: Tensor of shape ``[B, H]``.
        create_graph: Keep the graph for higher-order derivatives.
        vectorize: Accepted for API compatibility.
        max_samples: Optional chunk size for memory control.

    Returns:
        Tensor of shape ``[B, tangent_dim, H]``.
    """
    del vectorize
    return _compute_true_jacobian_wrt_cls(
        model.tangent_from_cls,
        cls_embedding,
        create_graph=create_graph,
        max_samples=max_samples,
    )


def jacobian_output_metric(
    J: Tensor,
    eps: float = 1e-6,
    normalize: bool = True,
    identity_mix: float = 0.1,
) -> Tensor:
    """Convert a true Jacobian into an output-space local metric.

    Args:
        J: Jacobian tensor of shape ``[B, D, H]``.
        eps: Numerical floor.
        normalize: If true, divide by average trace ``trace(C) / D``.
        identity_mix: Mix coefficient for ``I`` to stabilize the metric:
            ``C = (1 - identity_mix) * J J^T + identity_mix * I``.

    Returns:
        Tensor ``C`` of shape ``[B, D, D]``.
    """
    if J.dim() != 3:
        raise ValueError("J must have shape [B, D, H]")
    if not 0.0 <= identity_mix <= 1.0:
        raise ValueError("identity_mix must be in [0, 1]")

    batch_size, output_dim, _ = J.shape
    C = torch.bmm(J, J.transpose(1, 2))
    if normalize:
        trace = C.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        C = C / (trace / float(output_dim) + eps)

    identity = torch.eye(output_dim, dtype=J.dtype, device=J.device).expand(batch_size, -1, -1)
    return (1.0 - identity_mix) * C + identity_mix * identity


def jacobian_frobenius_norm(J: Tensor) -> Tensor:
    """Return per-sample Frobenius norm for ``J`` with shape ``[B, D, H]``."""
    if J.dim() != 3:
        raise ValueError("J must have shape [B, D, H]")
    return J.flatten(start_dim=1).norm(dim=-1)


def jacobian_effective_rank(J: Tensor, eps: float = 1e-8) -> Tensor:
    """Estimate per-sample effective rank from singular values.

    Returns a tensor of shape ``[B]``. If SVD fails for numerical reasons, the
    corresponding values are returned as NaN instead of interrupting training.
    """
    if J.dim() != 3:
        raise ValueError("J must have shape [B, D, H]")
    try:
        with torch.no_grad():
            singular_values = torch.linalg.svdvals(J.detach().float())
            probs = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(eps)
            entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
            return entropy.exp().to(device=J.device, dtype=J.dtype)
    except RuntimeError:
        return torch.full((J.size(0),), float("nan"), dtype=J.dtype, device=J.device)


def _compute_true_jacobian_wrt_cls(
    output_fn: Callable[[Tensor], Tensor],
    cls_embedding: Tensor,
    create_graph: bool,
    max_samples: Optional[int],
) -> Tensor:
    if cls_embedding.dim() != 2:
        raise ValueError("cls_embedding must have shape [B, H]")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    if max_samples is not None and cls_embedding.size(0) > max_samples:
        chunks = [
            _compute_true_jacobian_wrt_cls(output_fn, chunk, create_graph=create_graph, max_samples=None)
            for chunk in cls_embedding.split(max_samples, dim=0)
        ]
        return torch.cat(chunks, dim=0)

    with torch.enable_grad():
        cls_var = cls_embedding if cls_embedding.requires_grad else cls_embedding.detach().requires_grad_(True)
        output = output_fn(cls_var)
        if output.dim() != 2:
            raise ValueError("Jacobian output function must return shape [B, D]")

        grads = []
        for output_index in range(output.size(1)):
            grad = torch.autograd.grad(
                output[:, output_index].sum(),
                cls_var,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=False,
            )[0]
            grads.append(grad)
        return torch.stack(grads, dim=1)


def _first_tensor(value: Union[Tensor, Tuple[Tensor, ...]]) -> Tensor:
    return value[0] if isinstance(value, tuple) else value
