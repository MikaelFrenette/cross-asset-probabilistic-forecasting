"""
Panel Tuning Display
--------------------
Console UI helpers for the panel Optuna tuning runner: TF-Tuner-style
per-trial header (clears console, shows best-so-far hyperparameters and
value, progress counters, this trial's suggestions) and a final leaderboard
ranking completed trials by the tuned metric.

Functions
---------
clear_console
    Clear the terminal when stdout is a TTY.

render_trial_header
    Pre-trial banner showing the best hyperparameters so far.

after_trial_callback
    Optuna ``study.optimize`` callback that emits a one-line per-trial summary.

render_leaderboard
    Final table ranking trials by metric with pruning status.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import optuna

__all__ = [
    "after_trial_callback",
    "clear_console",
    "make_trial_params_banner",
    "render_fold_header",
    "render_leaderboard",
    "render_trial_header",
]


def clear_console() -> None:
    """Clear the terminal when stdout is a TTY."""

    if not sys.stdout.isatty():
        return
    os.system("cls" if os.name == "nt" else "clear")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_params(params: dict[str, Any]) -> list[str]:
    if not params:
        return ["(no params)"]
    width = max(len(path) for path in params)
    return [f"{path:<{width}} : {_format_value(value)}" for path, value in params.items()]


def make_trial_params_banner(params: dict[str, Any]) -> str:
    """Return a multi-line params banner used inside :func:`render_trial_header`."""

    lines = _format_params(params)
    return "\n".join(f"    {line}" for line in lines)


def render_trial_header(
    *,
    study_name: str,
    trial_number: int,
    n_trials: int | None,
    metric: str,
    direction: str,
    study: optuna.Study,
    trial_params: dict[str, Any],
    completed: int,
    pruned: int,
) -> None:
    """Print a pre-trial banner after clearing the console."""

    clear_console()
    total_str = f"{n_trials}" if n_trials else "?"
    print("=" * 72)
    print(f"Study: {study_name}  |  Trial {trial_number + 1}/{total_str}")
    print("-" * 72)
    print(f"Metric     : {metric} ({direction})")
    if completed > 0:
        try:
            best = study.best_trial
            best_params_text = ", ".join(
                f"{k}={_format_value(v)}" for k, v in best.params.items()
            )
            print(f"Best so far: {best.value:.5f} (trial {best.number})")
            print(f"    {best_params_text}")
        except ValueError:
            print("Best so far: —")
    else:
        print("Best so far: — (no completed trials yet)")
    print(f"Completed  : {completed}  |  Pruned: {pruned}")
    print("-" * 72)
    print("This trial:")
    print(make_trial_params_banner(trial_params))
    print("=" * 72)
    sys.stdout.flush()


def render_fold_header(
    *,
    study_name: str,
    trial_number: int,
    n_trials: int | None,
    fold_index: int,
    n_folds: int,
    metric: str,
    direction: str,
    study: optuna.Study,
    trial_params: dict[str, Any],
    completed: int,
    pruned: int,
    fold_losses_so_far: list[float] | None = None,
) -> None:
    """Clear the console and print a per-fold banner inside a trial."""

    clear_console()
    total_str = f"{n_trials}" if n_trials else "?"
    print("=" * 72)
    print(
        f"Study: {study_name}  |  Trial {trial_number + 1}/{total_str}  |  "
        f"Fold {fold_index + 1}/{n_folds}"
    )
    print("-" * 72)
    print(f"Metric     : {metric} ({direction})")
    if completed > 0:
        try:
            best = study.best_trial
            best_params_text = ", ".join(
                f"{k}={_format_value(v)}" for k, v in best.params.items()
            )
            print(f"Best so far: {best.value:.5f} (trial {best.number})")
            print(f"    {best_params_text}")
        except ValueError:
            print("Best so far: —")
    else:
        print("Best so far: — (no completed trials yet)")
    print(f"Completed  : {completed}  |  Pruned: {pruned}")
    if fold_losses_so_far:
        running = " / ".join(f"{v:.5f}" for v in fold_losses_so_far)
        mean_so_far = sum(fold_losses_so_far) / len(fold_losses_so_far)
        print(f"This trial : folds {fold_index}/{n_folds} done — [{running}] (mean {mean_so_far:.5f})")
    print("-" * 72)
    print("This trial:")
    print(make_trial_params_banner(trial_params))
    print("=" * 72)
    sys.stdout.flush()


def after_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Emit a one-line post-trial summary (used as an Optuna callback)."""

    if trial.state == optuna.trial.TrialState.PRUNED:
        line = f"Trial {trial.number} pruned"
    elif trial.state == optuna.trial.TrialState.FAIL:
        line = f"Trial {trial.number} FAILED"
    else:
        value_str = f"{trial.value:.5f}" if trial.value is not None else "—"
        line = f"Trial {trial.number} done — val: {value_str}"
    try:
        best = study.best_trial
        line += f" | best: {best.value:.5f} (trial {best.number})"
    except ValueError:
        pass
    print(line)
    sys.stdout.flush()


def render_leaderboard(
    *,
    study: optuna.Study,
    study_name: str,
    metric: str,
    direction: str,
    top_k: int = 10,
) -> None:
    """Print a final leaderboard with top-K trials and pruning status."""

    clear_console()
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    reverse = direction == "maximize"
    ranked = sorted(
        (t for t in completed if t.value is not None),
        key=lambda t: t.value,
        reverse=reverse,
    )

    param_columns: list[str] = []
    for trial in ranked:
        for param in trial.params:
            if param not in param_columns:
                param_columns.append(param)

    print("=" * 72)
    print(f"Study complete: {study_name}")
    print("-" * 72)
    print(f"Metric       : {metric} ({direction})")
    completion_line = f"Completed    : {len(completed)}/{len(trials)}"
    if pruned:
        completion_line += f"   ({len(pruned)} pruned)"
    if failed:
        completion_line += f"   ({len(failed)} failed)"
    print(completion_line)
    try:
        best = study.best_trial
        print(f"Best         : {best.value:.5f} (trial {best.number})")
        for path in param_columns:
            if path in best.params:
                print(f"    {path} = {_format_value(best.params[path])}")
    except ValueError:
        print("Best         : —")
    if len(ranked) > 1:
        values = [t.value for t in ranked]
        mean_v = sum(values) / len(values)
        var = sum((v - mean_v) ** 2 for v in values) / len(values)
        std_v = var ** 0.5
        print(f"Value summary: mean {mean_v:.5f} ± {std_v:.5f}")
    print("-" * 72)
    print(f"Top {min(top_k, len(ranked))} trials (by {metric}):")
    param_widths = {path: max(len(path), 8) + 2 for path in param_columns}
    header_parts = [f"  {'rank':<5}{'id':<5}{metric:<14}"]
    for path in param_columns:
        header_parts.append(f"{path:<{param_widths[path]}}")
    print("".join(header_parts))
    for rank, trial in enumerate(ranked[:top_k], start=1):
        row_parts = [f"  {rank:<5}{trial.number:<5}{trial.value:<14.5f}"]
        for path in param_columns:
            value = trial.params.get(path, "—")
            row_parts.append(f"{_format_value(value):<{param_widths[path]}}")
        print("".join(row_parts))
    print("=" * 72)
    sys.stdout.flush()
