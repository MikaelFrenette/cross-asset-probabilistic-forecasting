"""
Run Training
------------
Shared entrypoint that wires a PipelineConfig + RunLayout into a fully
instantiated Trainer (or DistributedTrainer) and runs a training session.

Used by scripts/train.py and the tuning objective so the same wiring logic
is not reimplemented in every caller.

Functions
---------
build_dataloaders
    Build train + val DataLoaders from a layout, with optional DDP sampler.
run_training
    Seed, build components, run the training loop, save history + summary,
    and return the trainer for downstream metric extraction.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from dlecosys.shared.config.schema import PipelineConfig
from dlecosys.shared.models import ModelFactory
from dlecosys.shared.reproducibility import seed_everything
from dlecosys.shared.run import RunLayout
from dlecosys.shared.training.callbacks import Callback
from dlecosys.shared.training.distributed import DistributedTrainer
from dlecosys.shared.training.supervised import Trainer

__all__ = ["build_dataloaders", "run_training"]

logger = logging.getLogger(__name__)


def _load_split(path) -> TensorDataset:
    data = torch.load(path, weights_only=True)
    return TensorDataset(data["X"], data["y"])


_NO_VAL = object()


def build_dataloaders(
    cfg: PipelineConfig,
    layout: Optional[RunLayout] = None,
    *,
    distributed: bool = False,
    train_dataset: Optional[TensorDataset] = None,
    val_dataset=_NO_VAL,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DistributedSampler]]:
    """Build train + val DataLoaders. Returns (train, val_or_None, train_sampler_or_none).

    Datasets are loaded from ``layout`` paths by default. Callers (e.g. the
    tuning fold loop or the ensemble runner) can inject preconstructed
    datasets to bypass disk load. Pass ``val_dataset=None`` explicitly to
    train without a validation set.
    """
    if train_dataset is None:
        if layout is None:
            raise ValueError("build_dataloaders requires layout when train_dataset is not provided")
        train_dataset = _load_split(layout.data_path("train"))

    if val_dataset is _NO_VAL:
        if layout is None:
            raise ValueError("build_dataloaders requires layout when val_dataset is not provided")
        val_dataset = _load_split(layout.data_path("val"))

    if distributed:
        sampler: Optional[DistributedSampler] = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.data.batch_size, sampler=sampler
        )
    else:
        sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.data.batch_size, shuffle=True
        )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size)

    return train_loader, val_loader, sampler


def _write_artifacts(
    trainer: Trainer,
    layout: RunLayout,
    summary_extras: Optional[Dict] = None,
) -> None:
    history_df = trainer.logger.history
    if history_df is not None and not history_df.empty:
        history_df.to_csv(layout.history_path, index=False)
    summary = trainer.logger.summary()
    if summary_extras:
        summary.update(summary_extras)
    with open(layout.summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def _render_run_summary(trainer: Trainer, layout: RunLayout, monitor: str) -> None:
    """Print a compact run summary at the end of training (rank-0 / single-GPU only)."""
    history = trainer.logger.history
    final_logs = trainer.logger.last_log()

    if history is None or history.empty:
        return

    n_epochs = len(history)
    final_val = final_logs.get(monitor)

    best_val = None
    best_epoch = None
    if monitor in history.columns:
        col = history[monitor].dropna()
        if not col.empty:
            best_idx = col.idxmin()
            best_val = float(col.loc[best_idx])
            if "epoch" in history.columns:
                best_epoch = int(history.loc[best_idx, "epoch"])

    print("=" * 72)
    print(f"Training complete: {layout.root.name}")
    print("-" * 72)
    print(f"Epochs ran   : {n_epochs}")
    if final_val is not None:
        print(f"Final {monitor}: {float(final_val):.5f}")
    if best_val is not None:
        epoch_str = f" (epoch {best_epoch})" if best_epoch is not None else ""
        print(f"Best  {monitor}: {best_val:.5f}{epoch_str}")
    print(f"Checkpoint   : {layout.checkpoint_path}")
    print(f"History      : {layout.history_path}")
    print(f"Summary      : {layout.summary_path}")
    print("=" * 72)


def run_training(
    cfg: PipelineConfig,
    layout: RunLayout,
    *,
    extra_callbacks: Optional[List[Callback]] = None,
    summary_extras: Optional[Dict] = None,
    train_dataset: Optional[TensorDataset] = None,
    val_dataset=_NO_VAL,
    render_summary: bool = True,
) -> Trainer:
    """
    Run a training session defined by ``cfg`` into the given ``layout``.

    Branches on ``cfg.distributed.enabled`` between single-device Trainer
    and DDP DistributedTrainer. In DDP mode the process group is set up
    before training and torn down in a ``finally``; history and summary
    are written by rank 0 only.

    Parameters
    ----------
    cfg : PipelineConfig
    layout : RunLayout
        Destination for checkpoints, history, and summary.
    extra_callbacks : list of Callback, optional
        Additional callbacks to append after the config-derived ones
        (used by tuning to attach a PruningCallback).
    summary_extras : dict, optional
        Extra fields to merge into the written summary.json
        (e.g. ``{"run_id": ...}`` or ``{"trial_number": ...}``).

    Returns
    -------
    Trainer
        The trainer instance after training, so callers can read
        ``trainer.logger.last_log()`` etc.
    """
    # Lazy-import builders to avoid a circular import chain:
    # config/__init__.py -> builders -> training.callbacks -> training/__init__.py -> run.
    from dlecosys.shared.config.builders import (
        build_callbacks,
        build_loss,
        build_metrics,
        build_optimizer,
        build_scheduler,
    )
    from dlecosys.shared.training.callbacks import LRSchedulerCallback

    seed_everything(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)
    tcfg = cfg.training
    distributed = cfg.distributed.enabled

    train_loader, val_loader, sampler = build_dataloaders(
        cfg,
        layout,
        distributed=distributed,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    model = ModelFactory.build(cfg.model.name, cfg.model.params)
    optimizer = build_optimizer(model, cfg.training.optimizer)
    loss_fn = build_loss(cfg.training.loss)
    metrics = build_metrics(cfg.training.metrics)
    callbacks = build_callbacks(
        cfg.training.callbacks,
        checkpoint_path=str(layout.checkpoint_path),
        tensorboard_dir=str(layout.logs_dir / "tensorboard"),
    )
    if cfg.training.scheduler is not None:
        scheduler = build_scheduler(cfg.training.scheduler, optimizer)
        monitor = None
        if cfg.training.scheduler.type == "reduce_on_plateau":
            monitor = cfg.training.callbacks.early_stopping.monitor
        callbacks.append(LRSchedulerCallback(scheduler, monitor=monitor))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer_kwargs = dict(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        callbacks=callbacks,
        amp=tcfg.amp,
        grad_accumulation_steps=tcfg.grad_accumulation_steps,
        grad_clip=tcfg.grad_clip,
    )

    if distributed:
        from dlecosys.shared.training.process_group import (
            is_main_process,
            setup,
            teardown,
        )

        setup(cfg.distributed.backend)
        try:
            trainer = DistributedTrainer(
                **trainer_kwargs,
                verbose=tcfg.verbose if is_main_process() else 0,
                train_sampler=sampler,
            )
            trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=tcfg.epochs,
            )
            if is_main_process():
                _write_artifacts(trainer, layout, summary_extras=summary_extras)
                if render_summary:
                    _render_run_summary(
                        trainer, layout, cfg.training.callbacks.early_stopping.monitor
                    )
        finally:
            teardown()
    else:
        trainer = Trainer(**trainer_kwargs, verbose=tcfg.verbose)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=tcfg.epochs,
        )
        _write_artifacts(trainer, layout, summary_extras=summary_extras)
        if render_summary:
            _render_run_summary(
                trainer, layout, cfg.training.callbacks.early_stopping.monitor
            )

    return trainer
