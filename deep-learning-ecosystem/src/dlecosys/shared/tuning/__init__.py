"""
Hyperparameter Tuning
---------------------
Optuna-based grid and random search over pipeline configs.

Import directly from submodules rather than this package. Optuna is only
required when importing from ``objective``, ``pruning``, or ``study``;
``search_space`` has no optuna dependency.

Submodules
----------
search_space
    apply_suggestion, suggest_values, to_hashable, from_hashable
pruning
    PruningCallback
study
    build_sampler, build_pruner, build_study
objective
    build_objective
"""

__all__: list[str] = []
