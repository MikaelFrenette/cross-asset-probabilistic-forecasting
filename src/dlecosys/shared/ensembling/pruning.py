"""
Ensemble Pruning
----------------
Post-training selection of a subset of estimators from the trained pool.

Ranks each estimator by its final OOB metric and returns the top-N. The runner
writes the result to ``ensemble/selected.json``; ``predict.py`` reads that
file and only aggregates predictions from surviving estimators.

Classes
-------
PruningResult
    Strategy name, metric, direction, and the surviving estimator ids.

Functions
---------
select_estimators
    Apply the configured pruning strategy to a list of per-estimator results.

Extending
---------
To add a new strategy (e.g. Caruana-style greedy forward selection)::

    class _GreedyStrategy:
        def select(self, results, direction, keep): ...

Then add the branch to ``select_estimators`` and extend
``EnsemblePruningSection.strategy`` Literal in ``schema.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from dlecosys.shared.config.schema import EnsemblePruningSection

__all__ = ["PruningResult", "select_estimators"]


@dataclass
class PruningResult:
    strategy: str
    direction: str
    selected: List[int]   # estimator ids, ranked best first
    n_total: int


def select_estimators(
    cfg: EnsemblePruningSection,
    results: List[Dict],
) -> PruningResult:
    """
    Select the surviving estimators.

    When ``cfg.enabled`` is False, every estimator with a finite ``oob_value``
    survives in natural id order — predict.py aggregates over all of them.

    When enabled with ``strategy=top_n``:
        - estimators with ``oob_value is None`` are excluded
        - the rest are ranked by ``oob_value`` (ascending for minimize,
          descending for maximize)
        - the top ``min(keep, n_valid)`` are selected
    """
    valid = [r for r in results if r["oob_value"] is not None]

    if not cfg.enabled:
        return PruningResult(
            strategy="none",
            direction=cfg.direction,
            selected=[r["estimator_id"] for r in valid],
            n_total=len(results),
        )

    if cfg.strategy == "top_n":
        reverse = cfg.direction == "maximize"
        ranked = sorted(valid, key=lambda r: r["oob_value"], reverse=reverse)
        k = min(cfg.keep, len(ranked))
        return PruningResult(
            strategy="top_n",
            direction=cfg.direction,
            selected=[r["estimator_id"] for r in ranked[:k]],
            n_total=len(results),
        )

    raise ValueError(f"Unknown pruning strategy: {cfg.strategy!r}")
