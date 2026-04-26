"""
Causal Next-Step Transformer
----------------------------
GPT-style per-position causal next-step density forecaster for panel data.
Temporal self-attention with a causal mask encodes the input history
``(B, I, T, D)``; at every timestep a cross-asset attention mixes across
the identifier axis, and per-position mu / sigma heads project each of
the ``T`` contextual embeddings to a one-step-ahead Gaussian prediction.
Supervision aligns each position ``k`` with target ``input_{k+1}`` (see
``VectorizedPanelDataset(target_mode="next_step")``); under the causal
mask, position ``k`` only attends to positions ``0..k``, so predicting
input ``k+1`` from position ``k``'s embedding is leak-free.

Output shape: ``(B, I, T, 2 * target_dim)`` with ``mu | sigma`` concatenated
on the last axis — the same convention as
:class:`CausalCrossAssetTransformer` but with the forecast axis equal to
the input length ``T`` rather than ``out_steps``.

At inference time, the ``t+1`` forecast for each rolling window sits at
the last position (``k = T - 1``); the predictor extracts
``prediction[:, :, -1, :]`` when the manifest's ``target_mode`` is
``"next_step"``.

Classes
-------
CausalNextStepTransformer
    Registered panel model under the name ``causal_next_step_transformer``.
    Shares :class:`CausalCrossAssetTransformerConfig` (the ``out_steps``
    field is unused by this architecture).
"""

from __future__ import annotations

import torch
from torch import nn

from density_model.models.cross_asset import CausalCrossAssetTransformerConfig
from density_model.models.panel_base import BasePanelModel
from density_model.shared.data.panel.schema import PanelBatch
from density_model.shared.models import register

__all__ = ["CausalNextStepTransformer"]


@register("causal_next_step_transformer")
class CausalNextStepTransformer(BasePanelModel):
    """
    Causal per-position next-step density forecaster with cross-asset mixing.

    Architecture
    ------------
    - Optional continuous / dynamic-categorical / static-categorical stream
      encoders and fusion — identical to
      :class:`CausalCrossAssetTransformer`.
    - Causal ``TransformerEncoder`` over the time axis (per-asset flattened
      into the batch dim so the encoder sees ``(B*I, T, d_model)``).
    - **Cross-asset attention at every timestep**: the temporal output
      ``(B, I, T, d_model)`` is reshaped to ``(B*T, I, d_model)`` so the
      attention operates across the asset axis independently at each of
      the ``T`` positions; result reshaped back to ``(B, I, T, d_model)``.
    - Per-position ``mu_head`` and ``sigma_head`` (``nn.Linear(d_model,
      target_dim)``) applied at every position.

    Output shape: ``(B, I, T, 2 * target_dim)`` with ``mu | sigma``
    concatenated on the last axis.
    """

    config_class = CausalCrossAssetTransformerConfig

    def __init__(self, config: CausalCrossAssetTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.continuous_encoder = (
            nn.Linear(config.continuous_dim, config.d_model)
            if config.continuous_dim is not None
            else None
        )
        self.dynamic_embeddings = self._build_embedding_list(
            config.dynamic_categorical_cardinalities
        )
        self.static_embeddings = self._build_embedding_list(
            config.static_categorical_cardinalities
        )

        self.dynamic_projection = (
            nn.Linear(len(self.dynamic_embeddings) * config.d_model, config.d_model)
            if self.dynamic_embeddings is not None
            else None
        )
        self.static_projection = (
            nn.Linear(len(self.static_embeddings) * config.d_model, config.d_model)
            if self.static_embeddings is not None
            else None
        )

        max_streams = 3
        fused_dim = config.d_model if config.fusion_mode == "sum" else max_streams * config.d_model
        self.fusion_projection = nn.Linear(fused_dim, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.cross_asset_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Per-position heads: d_model -> target_dim (NOT out_steps * target_dim).
        # Each of the T positions produces its own one-step-ahead Gaussian.
        self.mu_head = nn.Linear(config.d_model, config.target_dim)
        self.sigma_head = nn.Linear(config.d_model, config.target_dim)
        self.softplus = nn.Softplus()

    @classmethod
    def from_config(
        cls, config: CausalCrossAssetTransformerConfig
    ) -> CausalNextStepTransformer:
        return cls(config)

    def forward(self, batch: PanelBatch) -> torch.Tensor:  # type: ignore[override]
        """Compute per-position one-step-ahead Gaussian forecasts."""

        encoded_streams: list[torch.Tensor] = []
        observed_streams: list[torch.Tensor] = []

        if batch.X_continuous is not None:
            encoded_streams.append(
                self._encode_continuous(batch.X_continuous, batch.X_continuous_mask)
            )
            observed_streams.append(batch.X_continuous_mask.any(dim=-1))
        if batch.X_cat_continuous is not None:
            encoded_streams.append(
                self._encode_categorical(
                    batch.X_cat_continuous,
                    batch.X_cat_continuous_mask,
                    self.dynamic_embeddings,
                    self.dynamic_projection,
                    name="X_cat_continuous",
                )
            )
            observed_streams.append(batch.X_cat_continuous_mask.any(dim=-1))
        if batch.X_cat_static is not None:
            encoded_streams.append(
                self._encode_categorical(
                    batch.X_cat_static,
                    batch.X_cat_static_mask,
                    self.static_embeddings,
                    self.static_projection,
                    name="X_cat_static",
                )
            )
            observed_streams.append(batch.X_cat_static_mask.any(dim=-1))

        if not encoded_streams:
            raise ValueError(
                "CausalNextStepTransformer requires at least one active input stream."
            )

        fused = self._fuse_streams(encoded_streams)
        time_observed = torch.stack(observed_streams, dim=0).any(dim=0)

        batch_size, num_assets, sequence_length, _ = fused.shape

        # Temporal causal self-attention across T, per (asset, sample).
        temporal_input = fused.reshape(
            batch_size * num_assets, sequence_length, self.config.d_model
        )
        temporal_valid = time_observed.reshape(batch_size * num_assets, sequence_length)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=fused.device, dtype=torch.bool),
            diagonal=1,
        )
        temporal_output = self.temporal_encoder(
            temporal_input,
            mask=causal_mask,
            src_key_padding_mask=~temporal_valid,
        )
        temporal_output = temporal_output.reshape(
            batch_size, num_assets, sequence_length, self.config.d_model
        )

        # Cross-asset attention at EVERY timestep. Reshape (B, I, T, D) ->
        # (B*T, I, D) so each of T time slices gets its own attention over
        # the asset axis. Padding-mask an asset only if ALL its positions
        # up to t are unobserved; using the time-wise observed flag gives
        # a tighter mask but the current schema uses batch-level asset
        # presence, so we broadcast the all-position mask per timestep.
        temporal_output_t_major = temporal_output.permute(0, 2, 1, 3).contiguous()
        per_pos = temporal_output_t_major.reshape(
            batch_size * sequence_length, num_assets, self.config.d_model
        )
        asset_padding_mask = (~time_observed.any(dim=-1))  # (B, I) True where asset never observed
        per_pos_padding = (
            asset_padding_mask.unsqueeze(1)
            .expand(batch_size, sequence_length, num_assets)
            .reshape(batch_size * sequence_length, num_assets)
        )
        per_pos_cross, _ = self.cross_asset_attention(
            per_pos, per_pos, per_pos,
            key_padding_mask=per_pos_padding,
            need_weights=False,
        )
        per_pos_cross = per_pos_cross.reshape(
            batch_size, sequence_length, num_assets, self.config.d_model
        ).permute(0, 2, 1, 3).contiguous()  # back to (B, I, T, D)

        # Per-position heads.
        mu = self.mu_head(per_pos_cross)                     # (B, I, T, target_dim)
        sigma = self.softplus(self.sigma_head(per_pos_cross))
        sigma = sigma + self.config.sigma_min
        return torch.cat([mu, sigma], dim=-1)                # (B, I, T, 2 * target_dim)

    def _build_embedding_list(
        self, cardinalities: tuple[int, ...] | None
    ) -> nn.ModuleList | None:
        if cardinalities is None:
            return None
        return nn.ModuleList(
            nn.Embedding(cardinality, self.config.d_model) for cardinality in cardinalities
        )

    def _encode_continuous(
        self, values: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.continuous_encoder is None:
            raise ValueError(
                "Continuous inputs were provided but the model has no continuous encoder."
            )
        masked_values = torch.nan_to_num(values, nan=0.0)
        if mask is not None:
            masked_values = masked_values * mask.to(dtype=masked_values.dtype)
        return self.continuous_encoder(masked_values)

    def _encode_categorical(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None,
        embeddings: nn.ModuleList | None,
        projection: nn.Linear | None,
        *,
        name: str,
    ) -> torch.Tensor:
        if embeddings is None or projection is None:
            raise ValueError(
                f"{name} was provided but the model has no matching categorical encoder."
            )
        if values.shape[-1] != len(embeddings):
            raise ValueError(
                f"{name} feature dimension does not match configured categorical cardinalities."
            )
        embedded_columns: list[torch.Tensor] = []
        for column_index, embedding in enumerate(embeddings):
            column_values = values[..., column_index]
            column_embedding = embedding(column_values)
            if mask is not None:
                column_embedding = column_embedding * mask[..., column_index].unsqueeze(-1).to(
                    column_embedding.dtype
                )
            embedded_columns.append(column_embedding)
        return projection(torch.cat(embedded_columns, dim=-1))

    def _fuse_streams(self, encoded_streams: list[torch.Tensor]) -> torch.Tensor:
        if self.config.fusion_mode == "sum":
            fused = torch.stack(encoded_streams, dim=0).sum(dim=0)
            return self.fusion_projection(fused)
        concatenated = torch.cat(encoded_streams, dim=-1)
        if concatenated.shape[-1] != self.fusion_projection.in_features:
            padded = concatenated.new_zeros(
                *concatenated.shape[:-1], self.fusion_projection.in_features
            )
            padded[..., : concatenated.shape[-1]] = concatenated
            concatenated = padded
        return self.fusion_projection(concatenated)
