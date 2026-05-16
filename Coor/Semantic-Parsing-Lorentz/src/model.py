"""Dynamic-curvature Lorentz semantic parser."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distance import LorentzBilinearDistance, lorentz_expmap0
from .true_jacobian import (
    compute_true_lorentz_jacobian_wrt_cls,
    compute_true_tangent_jacobian_wrt_cls,
    jacobian_effective_rank,
    jacobian_frobenius_norm,
    jacobian_output_metric,
)


class TinyTextEncoder(nn.Module):
    """Offline Transformer encoder used for smoke tests without HF downloads."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_position_embeddings: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError("input sequence is longer than max_position_embeddings")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        key_padding_mask = None if attention_mask is None else attention_mask.eq(0)
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.layer_norm(encoded[:, 0])


class DynamicCurvaturePredictor(nn.Module):
    """Predicts one negative curvature value per input instance."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        min_abs_curvature: float = 0.05,
        max_abs_curvature: float = 5.0,
        parameterization: str = "sigmoid_bounded",
        init_abs_curvature: Optional[float] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min_abs_curvature <= 0 or max_abs_curvature <= min_abs_curvature:
            raise ValueError("curvature bounds must satisfy 0 < min_abs < max_abs")
        if parameterization not in {"sigmoid_bounded", "softplus_floor"}:
            raise ValueError("parameterization must be one of: sigmoid_bounded, softplus_floor")

        self.min_abs_curvature = min_abs_curvature
        self.max_abs_curvature = max_abs_curvature
        self.parameterization = parameterization
        final = nn.Linear(hidden_dim, 1)
        if init_abs_curvature is not None:
            init_abs_curvature = min(max(init_abs_curvature, min_abs_curvature + 1e-6), max_abs_curvature)
            with torch.no_grad():
                if parameterization == "softplus_floor":
                    offset = torch.tensor(init_abs_curvature - min_abs_curvature)
                    final.bias.fill_(float(torch.log(torch.expm1(offset.clamp_min(1e-6)))))
                else:
                    span = max_abs_curvature - min_abs_curvature
                    normalized = torch.tensor((init_abs_curvature - min_abs_curvature) / span)
                    normalized = normalized.clamp(1e-6, 1.0 - 1e-6)
                    final.bias.fill_(float(torch.logit(normalized)))
                final.weight.mul_(0.1)

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            final,
        )

    def raw(self, cls_embedding: Tensor) -> Tensor:
        return self.net(cls_embedding)

    def forward_from_raw(self, raw: Tensor) -> Tensor:
        if self.parameterization == "softplus_floor":
            abs_curvature = self.min_abs_curvature + F.softplus(raw)
        else:
            span = self.max_abs_curvature - self.min_abs_curvature
            abs_curvature = self.min_abs_curvature + span * torch.sigmoid(raw)
        return -abs_curvature

    def forward(self, cls_embedding: Tensor) -> Tensor:
        return self.forward_from_raw(self.raw(cls_embedding))


class SemanticLorentzParser(nn.Module):
    """Semantic parsing classifier using dynamic Lorentz curvature."""

    def __init__(
        self,
        num_labels: int,
        encoder_name: str = "bert-base-uncased",
        use_dummy_encoder: bool = True,
        dummy_vocab_size: int = 30522,
        hidden_dim: int = 128,
        tangent_dim: int = 64,
        max_position_embeddings: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        min_abs_curvature: float = 0.05,
        max_abs_curvature: float = 5.0,
        curvature_parameterization: str = "sigmoid_bounded",
        curvature_init_abs_c: Optional[float] = None,
        max_expmap_norm: float = 15.0,
        max_tangent_norm: float = 5.0,
        metric_delta_scale: float = 0.05,
        hf_local_files_only: bool = False,
        use_true_jacobian_metric: bool = False,
        true_jacobian_target: str = "lorentz",
        true_jacobian_create_graph: bool = True,
        true_jacobian_normalize: bool = True,
        true_jacobian_identity_mix: float = 0.1,
        true_jacobian_max_batch: Optional[int] = None,
        true_jacobian_reg_weight: float = 0.0,
        true_jacobian_complexity_weight: float = 0.0,
        detach_true_jacobian_metric: bool = False,
    ) -> None:
        super().__init__()
        if true_jacobian_target not in {"lorentz", "tangent"}:
            raise ValueError("true_jacobian_target must be 'lorentz' or 'tangent'")
        if not 0.0 <= true_jacobian_identity_mix <= 1.0:
            raise ValueError("true_jacobian_identity_mix must be in [0, 1]")
        self.num_labels = num_labels
        self.tangent_dim = tangent_dim
        self.max_expmap_norm = max_expmap_norm
        self.max_tangent_norm = max_tangent_norm
        self.use_true_jacobian_metric = use_true_jacobian_metric
        self.true_jacobian_target = true_jacobian_target
        self.true_jacobian_create_graph = true_jacobian_create_graph
        self.true_jacobian_normalize = true_jacobian_normalize
        self.true_jacobian_identity_mix = true_jacobian_identity_mix
        self.true_jacobian_max_batch = true_jacobian_max_batch
        self.true_jacobian_reg_weight = true_jacobian_reg_weight
        self.true_jacobian_complexity_weight = true_jacobian_complexity_weight
        self.detach_true_jacobian_metric = detach_true_jacobian_metric

        if use_dummy_encoder:
            self.encoder = TinyTextEncoder(
                vocab_size=dummy_vocab_size,
                hidden_dim=hidden_dim,
                max_position_embeddings=max_position_embeddings,
                dropout=dropout,
            )
            encoder_hidden_dim = hidden_dim
        else:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError("Install transformers or set use_dummy_encoder=true") from exc

            self.encoder = AutoModel.from_pretrained(
                encoder_name,
                local_files_only=hf_local_files_only,
            )
            encoder_hidden_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.encoder_hidden_dim = encoder_hidden_dim
        self.tangent_projector = nn.Sequential(
            nn.LayerNorm(encoder_hidden_dim),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, tangent_dim),
        )
        self.curvature_head = DynamicCurvaturePredictor(
            input_dim=encoder_hidden_dim,
            hidden_dim=max(encoder_hidden_dim // 2, 32),
            min_abs_curvature=min_abs_curvature,
            max_abs_curvature=max_abs_curvature,
            parameterization=curvature_parameterization,
            init_abs_curvature=curvature_init_abs_c,
            dropout=dropout,
        )

        self.label_tangent = nn.Embedding(num_labels, tangent_dim)
        nn.init.normal_(self.label_tangent.weight, mean=0.0, std=0.02)

        self.label_curvature_raw = nn.Parameter(torch.zeros(num_labels, 1))
        self.min_abs_curvature = min_abs_curvature
        self.max_abs_curvature = max_abs_curvature

        self.distance = LorentzBilinearDistance(
            coordinate_dim=tangent_dim + 1,
            metric_delta_scale=metric_delta_scale,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))

    def encode_cls(self, input_ids: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        """Encode text and return CLS/text embeddings with shape ``[B, H]``."""
        if isinstance(self.encoder, TinyTextEncoder):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def _encode_cls(self, input_ids: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        return self.encode_cls(input_ids=input_ids, attention_mask=attention_mask)

    def _bounded_tangent(self, tangent: Tensor) -> Tensor:
        return self.max_tangent_norm * torch.tanh(tangent / self.max_tangent_norm)

    def tangent_from_cls(self, cls_embedding: Tensor) -> Tensor:
        """Project CLS embeddings to bounded tangent vectors ``[B, tangent_dim]``."""
        return self._bounded_tangent(self.tangent_projector(cls_embedding))

    def curvature_from_cls(self, cls_embedding: Tensor) -> Tensor:
        """Predict negative curvatures from CLS embeddings with shape ``[B, 1]``."""
        return self.curvature_head(cls_embedding)

    def curvature_with_raw_from_cls(self, cls_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict negative curvatures and expose the raw head output."""
        raw = self.curvature_head.raw(cls_embedding)
        return self.curvature_head.forward_from_raw(raw), raw

    def lorentz_point_from_cls(
        self,
        cls_embedding: Tensor,
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Map CLS embeddings to Lorentz points.

        Args:
            cls_embedding: Tensor of shape ``[B, H]``.
            return_components: If true, return ``(point, tangent, curvature)``.

        Returns:
            Lorentz point with shape ``[B, tangent_dim + 1]`` or a tuple with
            the point, tangent vector, and curvature.
        """
        tangent = self.tangent_from_cls(cls_embedding)
        curvature = self.curvature_from_cls(cls_embedding)
        point = lorentz_expmap0(tangent, curvature, max_norm=self.max_expmap_norm)
        if return_components:
            return point, tangent, curvature
        return point

    def label_curvatures(self) -> Tensor:
        span = self.max_abs_curvature - self.min_abs_curvature
        abs_curvature = self.min_abs_curvature + span * torch.sigmoid(self.label_curvature_raw)
        return -abs_curvature

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_embeddings: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, Tensor]:
        cls_embedding = self.encode_cls(input_ids=input_ids, attention_mask=attention_mask)

        rng_state = self._capture_rng_state(cls_embedding.device) if self.use_true_jacobian_metric else None
        tangent = self.tangent_from_cls(cls_embedding)
        curvature, curvature_raw = self.curvature_with_raw_from_cls(cls_embedding)
        query_points = lorentz_expmap0(tangent, curvature, max_norm=self.max_expmap_norm)
        query_local_metric: Optional[Tensor] = None
        true_jacobian: Optional[Tensor] = None
        jacobian_frobenius: Optional[Tensor] = None
        jacobian_rank: Optional[Tensor] = None

        if self.use_true_jacobian_metric:
            if rng_state is not None:
                self._restore_rng_state(rng_state)
            true_jacobian, query_local_metric = self._compute_query_local_metric(cls_embedding)
            jacobian_frobenius = jacobian_frobenius_norm(true_jacobian)
            if self.detach_true_jacobian_metric:
                query_local_metric = query_local_metric.detach()
            if return_diagnostics:
                jacobian_rank = jacobian_effective_rank(true_jacobian)

        label_tangent = self._bounded_tangent(self.label_tangent.weight)
        label_curvature = self.label_curvatures()
        label_points = lorentz_expmap0(label_tangent, label_curvature, max_norm=self.max_expmap_norm)

        distances = self.distance.pairwise(
            query_points,
            label_points,
            curvature,
            label_curvature,
            query_local_metric=query_local_metric,
        )
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = -logit_scale * distances + self.label_bias

        output: Dict[str, Tensor] = {
            "logits": logits,
            "distances": distances,
            "curvature": curvature,
            "curvature_raw": curvature_raw,
            "label_curvature": label_curvature,
        }
        if jacobian_frobenius is not None:
            output["jacobian_frobenius"] = jacobian_frobenius
        if jacobian_rank is not None:
            output["jacobian_effective_rank"] = jacobian_rank
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        if return_embeddings:
            output["query_points"] = query_points
            output["label_points"] = label_points
            output["tangent"] = tangent
            output["cls_embedding"] = cls_embedding
        if return_diagnostics and true_jacobian is not None and query_local_metric is not None:
            output["true_jacobian"] = true_jacobian
            output["query_local_metric"] = query_local_metric
        return output

    def metric_regularization(self) -> Tensor:
        return self.distance.regularization()

    def _compute_query_local_metric(self, cls_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        if self.true_jacobian_target == "lorentz":
            jacobian = compute_true_lorentz_jacobian_wrt_cls(
                self,
                cls_embedding,
                create_graph=self.true_jacobian_create_graph,
                max_samples=self.true_jacobian_max_batch,
            )
            local_metric = jacobian_output_metric(
                jacobian,
                normalize=self.true_jacobian_normalize,
                identity_mix=self.true_jacobian_identity_mix,
            )
            return jacobian, local_metric

        jacobian = compute_true_tangent_jacobian_wrt_cls(
            self,
            cls_embedding,
            create_graph=self.true_jacobian_create_graph,
            max_samples=self.true_jacobian_max_batch,
        )
        tangent_metric = jacobian_output_metric(
            jacobian,
            normalize=self.true_jacobian_normalize,
            identity_mix=self.true_jacobian_identity_mix,
        )
        batch_size = cls_embedding.size(0)
        coordinate_dim = self.tangent_dim + 1
        local_metric = torch.eye(
            coordinate_dim,
            dtype=cls_embedding.dtype,
            device=cls_embedding.device,
        ).expand(batch_size, -1, -1).clone()
        local_metric[:, 1:, 1:] = tangent_metric
        return jacobian, local_metric

    @staticmethod
    def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
        state: Dict[str, Any] = {"cpu": torch.get_rng_state()}
        if device.type == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Dict[str, Any]) -> None:
        torch.set_rng_state(state["cpu"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
