"""
Temporal-Only Transformer
-------------------------
Causal temporal transformer with optional stream encoders and no cross-asset
attention block. Emits probabilistic Gaussian forecasts shaped
``(B, ID, H, 2 * D_y)``.

Classes
-------
TemporalOnlyTransformer
    Registered panel model under the name ``temporal_only_transformer``; shares
    the ``CausalCrossAssetTransformerConfig`` architecture schema.
"""

from __future__ import annotations

import torch
from torch import nn

from density_model.models.cross_asset import CausalCrossAssetTransformerConfig
from density_model.models.panel_base import BasePanelModel
from density_model.shared.data.panel.schema import PanelBatch
from density_model.shared.models import register

__all__ = ["TemporalOnlyTransformer"]


@register("temporal_only_transformer")
class TemporalOnlyTransformer(BasePanelModel):
    """Causal temporal forecasting model without cross-asset attention."""

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
        self.mu_head = nn.Linear(config.d_model, config.out_steps * config.target_dim)
        self.sigma_head = nn.Linear(config.d_model, config.out_steps * config.target_dim)
        self.softplus = nn.Softplus()

    @classmethod
    def from_config(
        cls, config: CausalCrossAssetTransformerConfig
    ) -> TemporalOnlyTransformer:
        return cls(config)

    def forward(self, batch: PanelBatch) -> torch.Tensor:  # type: ignore[override]
        """Compute forecasts from a validated panel batch."""

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
                "TemporalOnlyTransformer requires at least one active input stream."
            )

        fused = self._fuse_streams(encoded_streams)
        time_observed = torch.stack(observed_streams, dim=0).any(dim=0)

        batch_size, num_assets, sequence_length, _ = fused.shape
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

        final_state = temporal_output[:, :, -1, :]
        mu = self.mu_head(final_state).reshape(
            batch_size,
            num_assets,
            self.config.out_steps,
            self.config.target_dim,
        )
        sigma = self.softplus(self.sigma_head(final_state)).reshape(
            batch_size,
            num_assets,
            self.config.out_steps,
            self.config.target_dim,
        )
        sigma = sigma + self.config.sigma_min
        return torch.cat([mu, sigma], dim=-1)

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
