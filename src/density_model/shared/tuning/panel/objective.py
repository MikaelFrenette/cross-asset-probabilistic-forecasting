"""
Panel Tuning Objective
----------------------
Per-trial objective evaluated by the Optuna study. For each fold produced by
the configured panel time-series splitter, re-fits the preprocessing bundle
on the fold's training dates (leak-free), transforms both slices, vectorizes
them into ``(B, ID, T, K)`` tensors, trains the panel model for
``epochs_per_fold`` steps, records the fold's validation loss, and reports
the intermediate value to Optuna so the pruner can stop bad trials early.

Classes
-------
PanelObjective
    Callable used as the Optuna ``study.optimize`` objective. Encapsulates
    the raw canonicalized panel, pipeline config, and splitter.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from optuna import Trial
from torch.optim import Adam
from torch.utils.data import DataLoader

import density_model.models  # noqa: F401 — triggers @register side-effects

from density_model.shared.config.panel_schema import PanelPipelineConfig
from density_model.shared.data.panel.dataset import VectorizedPanelTorchDataset
from density_model.shared.data.panel.splitter import BasePanelSplitter
from density_model.shared.data.panel.vectorizer import VectorizedPanelDataset
from density_model.shared.models import ModelFactory
from density_model.shared.preprocessing.panel import PreprocessingBundle
from density_model.shared.training.panel_losses import MaskedGaussianLogLikelihood
from density_model.shared.training.panel_trainer import PanelTrainer
from density_model.shared.tuning.panel.display import render_fold_header
from density_model.shared.tuning.panel.pruning_callback import PruningCallback
from density_model.shared.tuning.search_space import apply_suggestion, suggest_values

__all__ = ["PanelObjective"]

logger = logging.getLogger(__name__)


class PanelObjective:
    """
    Per-trial callable for panel hyperparameter tuning.

    Parameters
    ----------
    base_cfg : PanelPipelineConfig
        Validated base pipeline configuration. ``base_cfg.tuning`` must be set.
    raw_panel : pandas.DataFrame
        Pre-loaded canonicalized long-format panel spanning the entire tuning
        history (train + val dates).
    splitter : BasePanelSplitter
        Time-series splitter producing ``(train_dates, val_dates)`` folds.
    """

    def __init__(
        self,
        *,
        base_cfg: PanelPipelineConfig,
        raw_panel: pd.DataFrame,
        splitter: BasePanelSplitter,
    ) -> None:
        if base_cfg.tuning is None:
            raise ValueError("PanelObjective requires base_cfg.tuning to be set.")
        self.base_cfg = base_cfg
        self.raw_panel = raw_panel
        self.splitter = splitter
        self.total_trials = _resolve_total_trials(base_cfg.tuning)

    def __call__(self, trial: Trial) -> float:
        """Evaluate one hyperparameter trial."""

        assert self.base_cfg.tuning is not None  # guarded in __init__

        suggestions = suggest_values(trial, self.base_cfg.tuning.search_space)
        trial_cfg = self._build_trial_config(suggestions)

        from optuna.trial import TrialState as _TrialState

        completed = sum(1 for t in trial.study.trials if t.state == _TrialState.COMPLETE)
        pruned = sum(1 for t in trial.study.trials if t.state == _TrialState.PRUNED)

        folds = list(
            self.splitter.generate_folds(
                self.raw_panel, date_column=self.base_cfg.data.date_column
            )
        )
        n_folds = len(folds)
        if n_folds == 0:
            raise ValueError("Panel splitter produced zero folds.")

        fold_losses: list[float] = []
        for fold_index, (train_dates, val_dates) in enumerate(folds):
            render_fold_header(
                study_name=self.base_cfg.tuning.study_name,
                trial_number=trial.number,
                n_trials=self.total_trials,
                fold_index=fold_index,
                n_folds=n_folds,
                metric=self.base_cfg.tuning.metric,
                direction=self.base_cfg.tuning.direction,
                study=trial.study,
                trial_params=dict(suggestions),
                completed=completed,
                pruned=pruned,
                fold_losses_so_far=list(fold_losses),
            )

            train_slice = self._slice_by_dates(train_dates)
            val_slice = self._slice_by_dates(val_dates)
            fold_loss = self._evaluate_fold(
                trial=trial,
                fold_index=fold_index,
                trial_cfg=trial_cfg,
                train_panel=train_slice,
                val_panel=val_slice,
            )
            fold_losses.append(fold_loss)

        return float(np.mean(fold_losses))

    def _slice_by_dates(self, dates: np.ndarray) -> pd.DataFrame:
        """Return the subset of ``self.raw_panel`` where ``date_column`` ∈ ``dates``."""

        date_column = self.base_cfg.data.date_column
        mask = self.raw_panel[date_column].isin(dates)
        if not mask.any():
            raise ValueError(
                "Panel slice produced an empty frame; splitter generated dates "
                "not present in the panel."
            )
        return self.raw_panel.loc[mask].reset_index(drop=True)

    def _evaluate_fold(
        self,
        *,
        trial: Trial,
        fold_index: int,
        trial_cfg: PanelPipelineConfig,
        train_panel: pd.DataFrame,
        val_panel: pd.DataFrame,
    ) -> float:
        """Fit preprocessing + train on one fold; return the final val_loss."""

        assert trial_cfg.tuning is not None

        torch.manual_seed(trial_cfg.training.random_seed + fold_index)

        preprocessing = PreprocessingBundle(id_column=trial_cfg.data.id_column).fit(
            train_panel,
            continuous_columns=trial_cfg.features.continuous_columns,
            categorical_columns=(
                trial_cfg.features.dynamic_categorical_columns
                + trial_cfg.features.static_categorical_columns
            ),
        )

        train_vectorized = _vectorize(preprocessing.transform(train_panel), trial_cfg)
        val_vectorized = _vectorize(preprocessing.transform(val_panel), trial_cfg)
        if train_vectorized["y"] is None or val_vectorized["y"] is None:
            raise ValueError(
                f"Fold {fold_index} produced no valid forecast windows; "
                "increase fold date counts or reduce sequence_length/horizon."
            )

        train_dataset = VectorizedPanelTorchDataset(train_vectorized)
        val_dataset = VectorizedPanelTorchDataset(val_vectorized)
        train_loader = DataLoader(
            train_dataset,
            batch_size=trial_cfg.training.batch_size,
            shuffle=trial_cfg.training.shuffle,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=trial_cfg.training.batch_size,
            shuffle=False,
        )

        resolved_params = _resolve_model_params(trial_cfg, preprocessing)
        model = ModelFactory.build(trial_cfg.model.name, resolved_params)
        optimizer = Adam(
            model.parameters(),
            lr=trial_cfg.training.learning_rate,
            weight_decay=trial_cfg.training.weight_decay,
        )

        callbacks = [
            PruningCallback(
                trial=trial,
                monitor=trial_cfg.tuning.metric,
                fold_step=fold_index,
            ),
        ]
        device = torch.device(trial_cfg.training.device)
        trainer = PanelTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=MaskedGaussianLogLikelihood(),
            metrics={},
            callbacks=callbacks,
            verbose=trial_cfg.training.verbose,
            strict=False,
            device=device,
            grad_clip=trial_cfg.training.grad_clip,
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=trial_cfg.tuning.epochs_per_fold,
        )

        final_logs = trainer.logger.last_log()
        metric_name = trial_cfg.tuning.metric
        if metric_name not in final_logs:
            raise KeyError(
                f"Metric {metric_name!r} was not produced by the fold trainer "
                f"(available: {sorted(final_logs)})."
            )
        return float(final_logs[metric_name])

    def _build_trial_config(self, suggestions: dict[str, Any]) -> PanelPipelineConfig:
        """Deep-copy the base config dict, apply suggestions, re-validate."""

        cfg_dict = self.base_cfg.model_dump()
        for path, value in suggestions.items():
            apply_suggestion(cfg_dict, path, value)
        return PanelPipelineConfig.model_validate(cfg_dict)


def _resolve_total_trials(tuning: Any) -> int | None:
    """
    Infer the total trial count for display.

    Uses ``cfg.tuning.n_trials`` when set; otherwise, for a grid sampler,
    falls back to the cardinality of the search-space cartesian product.
    Returns ``None`` for random samplers with no explicit budget (unbounded).
    """

    if tuning.n_trials is not None:
        return int(tuning.n_trials)
    if tuning.sampler == "grid" and tuning.search_space:
        return int(math.prod(len(choices) for choices in tuning.search_space.values()))
    return None


def _vectorize(panel: pd.DataFrame, cfg: PanelPipelineConfig) -> dict[str, Any]:
    return VectorizedPanelDataset(
        data=panel,
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        targets=cfg.features.target_column,
        in_steps=cfg.features.sequence_length,
        horizon=cfg.features.forecast_horizon,
        out_steps=1,
        dynamic_features=cfg.features.continuous_columns,
        dynamic_categorical_features=cfg.features.dynamic_categorical_columns or None,
        static_categorical_features=cfg.features.static_categorical_columns or None,
    ).generate_sequences()


def _resolve_model_params(
    cfg: PanelPipelineConfig, preprocessing: PreprocessingBundle
) -> dict[str, Any]:
    params: dict[str, Any] = copy.deepcopy(dict(cfg.model.params))
    params.setdefault(
        "continuous_dim",
        len(cfg.features.continuous_columns) if cfg.features.continuous_columns else None,
    )
    tokenizer = preprocessing.categorical_tokenizer
    dynamic_cols = cfg.features.dynamic_categorical_columns
    static_cols = cfg.features.static_categorical_columns
    if (dynamic_cols or static_cols) and tokenizer is None:
        raise ValueError("Categorical streams require a fitted tokenizer.")
    if dynamic_cols:
        params["dynamic_categorical_cardinalities"] = list(
            tokenizer.get_vocabulary_sizes(dynamic_cols)
        )
    if static_cols:
        params["static_categorical_cardinalities"] = list(
            tokenizer.get_vocabulary_sizes(static_cols)
        )
    return params
