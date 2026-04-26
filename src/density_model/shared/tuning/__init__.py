"""
Hyperparameter Tuning
---------------------
Optuna-driven panel-pipeline tuning. Implementation lives under
:mod:`density_model.shared.tuning.panel`. The :mod:`search_space`
submodule (``apply_suggestion``, ``suggest_values``, ``to_hashable``,
``from_hashable``) has no Optuna dependency and is reused by the panel
objective.
"""

__all__: list[str] = []
