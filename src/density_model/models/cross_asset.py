"""
Causal Cross-Asset Transformer
------------------------------
Causal temporal transformer with optional stream encoders, configurable fusion,
and last-time-step cross-asset attention.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from density_model.models.base import BasePanelModel
from density_model.models.config import CausalCrossAssetTransformerConfig
from density_model.models.schema import PanelBatch

__all__ = ["CausalCrossAssetTransformer"]


class CausalCrossAssetTransformer(BasePanelModel):
    """
    Causal panel forecasting model with temporal and cross-asset attention.

    Parameters
    ----------
    config : CausalCrossAssetTransformerConfig
        Validated model configuration.
    continuous_dim : int or None, default=None
        Number of continuous input channels.
    dynamic_categorical_dim : int or None, default=None
        Number of dynamic categorical feature columns.
    static_categorical_dim : int or None, default=None
        Number of static categorical feature columns.
    sigma_min : float, default=1e-6
        Minimum positive offset added to the softplus sigma head.
    """

    def __init__(
        self,
        config: CausalCrossAssetTransformerConfig,
        *,
        continuous_dim: int | None = None,
        dynamic_categorical_dim: int | None = None,
        static_categorical_dim: int | None = None,
        sigma_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.config = config
        self.continuous_dim = continuous_dim
        self.dynamic_categorical_dim = dynamic_categorical_dim
        self.static_categorical_dim = static_categorical_dim
        self.sigma_min = sigma_min

        self.continuous_encoder = (
            nn.Linear(continuous_dim, config.d_model) if continuous_dim is not None else None
        )
        self.dynamic_embeddings = self._build_embedding_list(config.dynamic_categorical_cardinalities)
        self.static_embeddings = self._build_embedding_list(config.static_categorical_cardinalities)

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
        self.mu_head = nn.Linear(config.d_model, config.out_steps * config.target_dim)
        self.sigma_head = nn.Linear(config.d_model, config.out_steps * config.target_dim)
        self.softplus = nn.Softplus()

    def forward(self, batch: PanelBatch) -> torch.Tensor:  # type: ignore[override]
        """
        Compute forecasts from a validated panel batch.

        Parameters
        ----------
        batch : PanelBatch
            Validated model-facing batch schema.

        Returns
        -------
        torch.Tensor
            Predicted probabilistic tensor shaped ``(B, ID, H, 2 * D_y)`` where
            the final axis concatenates ``mu`` and ``sigma``.
        """

        encoded_streams: list[torch.Tensor] = []
        observed_streams: list[torch.Tensor] = []

        if batch.X_continuous is not None:
            encoded_streams.append(self._encode_continuous(batch.X_continuous, batch.X_continuous_mask))
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
            raise ValueError("CausalCrossAssetTransformer requires at least one active input stream.")

        fused = self._fuse_streams(encoded_streams)
        time_observed = torch.stack(observed_streams, dim=0).any(dim=0)

        batch_size, num_assets, sequence_length, _ = fused.shape
        temporal_input = fused.reshape(batch_size * num_assets, sequence_length, self.config.d_model)
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
        temporal_output = temporal_output.reshape(batch_size, num_assets, sequence_length, self.config.d_model)

        final_state = temporal_output[:, :, -1, :]
        asset_padding_mask = ~time_observed.any(dim=-1)
        cross_asset_output, _ = self.cross_asset_attention(
            final_state,
            final_state,
            final_state,
            key_padding_mask=asset_padding_mask,
            need_weights=False,
        )
        mu = self.mu_head(cross_asset_output).reshape(
            batch_size,
            num_assets,
            self.config.out_steps,
            self.config.target_dim,
        )
        sigma = self.softplus(self.sigma_head(cross_asset_output)).reshape(
            batch_size,
            num_assets,
            self.config.out_steps,
            self.config.target_dim,
        )
        sigma = sigma + self.sigma_min
        return torch.cat([mu, sigma], dim=-1)

    def _build_embedding_list(self, cardinalities: tuple[int, ...] | None) -> nn.ModuleList | None:
        """
        Build embedding modules for a categorical stream.

        Parameters
        ----------
        cardinalities : tuple of int or None
            Per-column categorical vocabulary sizes.

        Returns
        -------
        torch.nn.ModuleList or None
            Embedding modules for the stream, or ``None`` when absent.
        """

        if cardinalities is None:
            return None
        return nn.ModuleList(nn.Embedding(cardinality, self.config.d_model) for cardinality in cardinalities)

    def _encode_continuous(self, values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Encode continuous inputs to the shared hidden width.

        Parameters
        ----------
        values : torch.Tensor
            Continuous input tensor.
        mask : torch.Tensor or None
            Continuous feature mask.

        Returns
        -------
        torch.Tensor
            Encoded continuous representation.
        """

        if self.continuous_encoder is None:
            raise ValueError("Continuous inputs were provided but the model has no continuous encoder.")
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
        """
        Encode a categorical input stream to the shared hidden width.

        Parameters
        ----------
        values : torch.Tensor
            Tokenized categorical input tensor.
        mask : torch.Tensor or None
            Categorical feature mask.
        embeddings : torch.nn.ModuleList or None
            Per-column embedding modules.
        projection : torch.nn.Linear or None
            Post-embedding projection to the shared hidden width.
        name : str
            Human-readable stream name for error messages.

        Returns
        -------
        torch.Tensor
            Encoded categorical representation.
        """

        if embeddings is None or projection is None:
            raise ValueError(f"{name} was provided but the model has no matching categorical encoder.")
        if values.shape[-1] != len(embeddings):
            raise ValueError(f"{name} feature dimension does not match configured categorical cardinalities.")
        embedded_columns: list[torch.Tensor] = []
        for column_index, embedding in enumerate(embeddings):
            column_values = values[..., column_index]
            column_embedding = embedding(column_values)
            if mask is not None:
                column_embedding = column_embedding * mask[..., column_index].unsqueeze(-1).to(column_embedding.dtype)
            embedded_columns.append(column_embedding)
        return projection(torch.cat(embedded_columns, dim=-1))

    def _fuse_streams(self, encoded_streams: list[torch.Tensor]) -> torch.Tensor:
        """
        Fuse encoded input streams into a shared hidden representation.

        Parameters
        ----------
        encoded_streams : list of torch.Tensor
            Encoded stream representations.

        Returns
        -------
        torch.Tensor
            Fused hidden representation shaped ``(B, ID, T, d_model)``.
        """

        if self.config.fusion_mode == "sum":
            fused = torch.stack(encoded_streams, dim=0).sum(dim=0)
            return self.fusion_projection(fused)
        concatenated = torch.cat(encoded_streams, dim=-1)
        if concatenated.shape[-1] != self.fusion_projection.in_features:
            padded = concatenated.new_zeros(*concatenated.shape[:-1], self.fusion_projection.in_features)
            padded[..., : concatenated.shape[-1]] = concatenated
            concatenated = padded
        return self.fusion_projection(concatenated)
