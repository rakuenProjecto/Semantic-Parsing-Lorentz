"""Dynamic-curvature Lorentz semantic parser."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distance import LorentzBilinearDistance, lorentz_expmap0


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
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min_abs_curvature <= 0 or max_abs_curvature <= min_abs_curvature:
            raise ValueError("curvature bounds must satisfy 0 < min_abs < max_abs")

        self.min_abs_curvature = min_abs_curvature
        self.max_abs_curvature = max_abs_curvature
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cls_embedding: Tensor) -> Tensor:
        raw = self.net(cls_embedding)
        span = self.max_abs_curvature - self.min_abs_curvature
        abs_curvature = self.min_abs_curvature + span * torch.sigmoid(raw)
        return -abs_curvature


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
        max_expmap_norm: float = 15.0,
        max_tangent_norm: float = 5.0,
        metric_delta_scale: float = 0.05,
        hf_local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.tangent_dim = tangent_dim
        self.max_expmap_norm = max_expmap_norm
        self.max_tangent_norm = max_tangent_norm

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

    def _encode_cls(self, input_ids: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        if isinstance(self.encoder, TinyTextEncoder):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def _bounded_tangent(self, tangent: Tensor) -> Tensor:
        return self.max_tangent_norm * torch.tanh(tangent / self.max_tangent_norm)

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
    ) -> Dict[str, Tensor]:
        cls_embedding = self._encode_cls(input_ids=input_ids, attention_mask=attention_mask)
        tangent = self._bounded_tangent(self.tangent_projector(cls_embedding))
        curvature = self.curvature_head(cls_embedding)
        query_points = lorentz_expmap0(tangent, curvature, max_norm=self.max_expmap_norm)

        label_tangent = self._bounded_tangent(self.label_tangent.weight)
        label_curvature = self.label_curvatures()
        label_points = lorentz_expmap0(label_tangent, label_curvature, max_norm=self.max_expmap_norm)

        distances = self.distance.pairwise(query_points, label_points, curvature, label_curvature)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = -logit_scale * distances + self.label_bias

        output: Dict[str, Tensor] = {
            "logits": logits,
            "distances": distances,
            "curvature": curvature,
            "label_curvature": label_curvature,
        }
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        if return_embeddings:
            output["query_points"] = query_points
            output["label_points"] = label_points
            output["tangent"] = tangent
        return output

    def metric_regularization(self) -> Tensor:
        return self.distance.regularization()
