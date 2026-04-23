"""
Panel Ensemble Display
----------------------
Console UI helpers for the panel bagging ensemble runner: TF-Tuner-style
per-estimator header (clears console, shows best-so-far, progress counters)
and a final leaderboard with top-N ranked survivors.

Functions
---------
clear_console
    Clear the terminal when stdout is a TTY.

render_header
    Pre-estimator banner showing best-so-far and this estimator's sizes.

print_estimator_result
    One-line summary emitted after an estimator finishes.

render_leaderboard
    Final table ranking estimators by OOB metric with retention marks.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable

__all__ = [
    "clear_console",
    "print_estimator_result",
    "render_header",
    "render_leaderboard",
]


def clear_console() -> None:
    """Clear the terminal when stdout is a TTY."""

    if not sys.stdout.isatty():
        return
    os.system("cls" if os.name == "nt" else "clear")


def render_header(
    *,
    run_name: str,
    estimator_id: int,
    n_estimators: int,
    monitor: str,
    in_bag_dates: int,
    oob_dates: int,
    assets: int,
    total_assets: int,
    seed: int,
    best: dict[str, Any] | None,
    completed: int,
) -> None:
    """Print a pre-estimator banner after clearing the console."""

    clear_console()
    print("=" * 72)
    print(f"Ensemble: {run_name}  |  Estimator {estimator_id + 1}/{n_estimators}")
    print("-" * 72)
    print(f"Metric     : {monitor} (minimize)")
    if best is not None:
        print(f"Best so far: {best['value']:.5f} (estimator {best['id']})")
    else:
        print("Best so far: — (no estimators completed yet)")
    print(f"Completed  : {completed}/{n_estimators}")
    print("-" * 72)
    print("This estimator:")
    print(f"    in-bag dates : {in_bag_dates}")
    print(f"    OOB dates    : {oob_dates}")
    print(f"    assets       : {assets}/{total_assets}")
    print(f"    seed         : {seed}")
    print("=" * 72)
    sys.stdout.flush()


def print_estimator_result(
    *, estimator_id: int, oob_value: float | None, best: dict[str, Any] | None
) -> None:
    """Emit a single-line summary after one estimator finishes."""

    val_str = f"{oob_value:.5f}" if oob_value is not None else "—"
    line = f"Estimator {estimator_id} done — OOB val: {val_str}"
    if best is not None:
        line += f" | best: {best['value']:.5f} (estimator {best['id']})"
    print(line)
    sys.stdout.flush()


def render_leaderboard(
    *,
    run_name: str,
    monitor: str,
    aggregation: str,
    n_estimators: int,
    results: Iterable[dict[str, Any]],
    best: dict[str, Any] | None,
    pruning_enabled: bool,
    pruning_strategy: str,
    retained_ids: Iterable[int],
    top_k: int = 10,
) -> None:
    """Print a final leaderboard with top-N estimators and retention marks."""

    clear_console()
    results_list = list(results)
    completed = [r for r in results_list if r["oob_value"] is not None]
    ranked = sorted(completed, key=lambda r: r["oob_value"])
    retained_set = set(retained_ids)

    print("=" * 72)
    print(f"Ensemble complete: {run_name}")
    print("-" * 72)
    print(f"Metric       : {monitor} (minimize)")
    print(f"Aggregation  : {aggregation}")
    print(f"Completed    : {len(completed)}/{n_estimators}")
    print(
        f"Pruning      : {pruning_strategy if pruning_enabled else 'none'}  "
        f"→ {len(retained_set)} of {len(results_list)} estimators retained"
    )
    if best is not None:
        print(f"Best         : {best['value']:.5f} (estimator {best['id']})")
    if len(completed) > 1:
        values = [r["oob_value"] for r in completed]
        mean_v = sum(values) / len(values)
        var = sum((v - mean_v) ** 2 for v in values) / len(values)
        std_v = var ** 0.5
        print(f"OOB summary  : mean {mean_v:.5f} ± {std_v:.5f}")
    print("-" * 72)
    print(f"Top {min(top_k, len(ranked))} estimators (by OOB {monitor}):")
    print(f"  {'':<3}{'rank':<6}{'id':<6}{'OOB':<14}{'in-bag':<10}{'oob':<8}{'assets':<8}")
    for rank, r in enumerate(ranked[:top_k], start=1):
        mark = "✓" if r["estimator_id"] in retained_set else " "
        print(
            f"  {mark:<3}{rank:<6}{r['estimator_id']:<6}"
            f"{r['oob_value']:<14.5f}{r['in_bag_dates']:<10}{r['oob_dates']:<8}"
            f"{r['assets']:<8}"
        )
    if pruning_enabled:
        print("(✓ = retained by pruning; used at inference)")
    print("=" * 72)
    sys.stdout.flush()
